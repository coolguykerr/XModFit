import importlib

from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLabel, QLineEdit, QVBoxLayout, QMessageBox, QCheckBox, \
    QComboBox, QListWidget, QDialog, QFileDialog, QAbstractItemView, QSplitter, QSizePolicy, QAbstractScrollArea, QHBoxLayout, QTextEdit, QShortcut,\
    QProgressDialog, QDesktopWidget, QSlider, QTabWidget, QMenuBar, QAction, QTableWidgetSelectionRange, QProgressBar, QMenu, QTableWidgetItem, QTreeWidgetItem,\
    QListWidgetItem, QTableWidget, QHeaderView, QDoubleSpinBox, QToolButton, QDialogButtonBox
from PyQt5.QtGui import QKeySequence, QFont, QDoubleValidator, QIntValidator, QTextCursor
from PyQt5.QtCore import Qt, QProcess, QObject, QEvent
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import webbrowser, shutil
from docx import Document
import os
import glob
import sys
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
from PlotWidget import PlotWidget
import copy
import numpy as np
from Data_Dialog import Data_Dialog
# from readData import read1DSAXS
from importlib import import_module, reload
from Fit_Routines import Fit
from tabulate import tabulate
import corner
import numbers
import time
import shutil
from FunctionEditor import FunctionEditor
from MultiInputDialog import MultiInputDialog
import traceback
import pandas as pd
from functools import partial
import pylab as pl
from scipy.stats import chi2
from scipy.interpolate import interp1d
import math
from mplWidget import MplWidget
import statsmodels.api as sm
import Chemical_Formula
import Structure_Factors
import utils
import xraydb
from ConsoleWidget import ConsoleWidget
from collections import OrderedDict
try:
    from collections.abc import MutableMapping
except ImportError:            # Python 2 fallback, harmless here
    from collections import MutableMapping
import datetime
import json
import hashlib
from emcee import backends
from lmfit.customhdf import CustomHDFBackend
from lmfit.minimizer import PDB_FUNCNAMES
from joint_fit_panel import JointFitPanel, NameDelegate
from Image2DWidget import Image2DWidget
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Keys of an output_params entry that are METADATA, not plottable variables. The
# generated-parameter list shows every OTHER key, so this is what keeps an image
# entry reading as "map2d_data : ['x', 'y', 'z']".
#
# 'meta' is DELIBERATELY absent: it is currently shown in that list and
# saveGenParameters branches on the literal strings "['x', 'y', 'meta']" and
# "['x', 'y', 'z', 'meta']" (see saveGenParameters), so filtering it here would
# silently break saving of every existing generated output.
OP_META_KEYS = ('names', 'plotType', 'role', 'panel', 'cmap', 'zscale', 'levels')


class _JointCompileRefused(Exception):
    """Internal: XModFit._compile_joint_guarded refused or the user declined;
    the message box has already been shown."""


class _LimitProxy(MutableMapping):
    """Backwards-compatible view of ``limits[data_key]['x']`` as xmin / xmax.

    Fit limits generalize from "x has a range" to "any named data column has an
    optional (lo, hi)", stored as ``self.limits[data_key] = {col: (lo, hi)}``.
    ``'x'`` is a reserved column name meaning "whichever DataFrame column is
    plotted as x".

    This view exists so that every pre-existing ``self.xmin[key]`` /
    ``self.xmax[key]`` read, write and delete keeps working verbatim -- which is
    what makes the generalization a refactor of one store rather than a rewrite of
    twenty call sites.

    ``index`` is 0 for the low bound and 1 for the high bound.
    """

    def __init__(self, store, index):
        self._store = store
        self._i = int(index)

    def __getitem__(self, key):
        return self._store[key]['x'][self._i]

    def __setitem__(self, key, value):
        d = self._store.setdefault(key, {})
        pair = list(d.get('x', (None, None)))
        pair[self._i] = value
        d['x'] = tuple(pair)

    def __delitem__(self, key):
        # Tolerant of a repeated delete: callers do `del self.xmin[k]` followed by
        # `del self.xmax[k]`, and both refer to the same underlying 'x' entry.
        # A missing DATASET still raises, as a mapping should.
        d = self._store[key]
        d.pop('x', None)

    def __iter__(self):
        return iter([k for k, v in self._store.items() if 'x' in v])

    def __len__(self):
        return sum(1 for v in self._store.values() if 'x' in v)

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, dict(self))



class ColumnLimitsDialog(QDialog):
    """Per-column fit limits for the selected dataset(s).

    Deliberately the same idiom as joint_fit_panel's per-dataset rangeTable:
    blank means unbounded, a non-numeric entry reverts, and the name column is
    read-only. Row 0 is the reserved 'x' column and is two-way bound to the main
    window's Xmin:Xmax line edit, so there is exactly one x window, not two.

    Non-modal on purpose: you want to watch the plot while narrowing a window.
    """

    def __init__(self, parent, names):
        QDialog.__init__(self, parent)
        self.host = parent
        self.names = list(names)
        self.setWindowTitle('Fit limits by data column')
        self._loading = False
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('Blank = unbounded. Row "x" mirrors Xmin:Xmax.\n'
                             'Applies to every selected dataset.'))
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Column', 'min', 'max'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.table)
        self.statusLabel = QLabel('')
        lay.addWidget(self.statusLabel)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        clear = bb.addButton('Clear all', QDialogButtonBox.ResetRole)
        clear.clicked.connect(self._clear_all)
        lay.addWidget(bb)
        self.resize(430, 320)
        self.refresh()

    # ------------------------------------------------------------------ ui --
    def refresh(self):
        """Rebuild from the host's current selection."""
        self._loading = True
        try:
            keys = self.host.sfnames or []
            cols = self.host.available_columns(keys[-1]) if keys else [('x', 'x')]
            self.table.setRowCount(len(cols))
            self._cols = [c for c, _ in cols]
            for r, (col, label) in enumerate(cols):
                it = QTableWidgetItem(label)
                it.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(r, 0, it)
                lo, hi = self._shared_value(keys, col)
                for c, v in ((1, lo), (2, hi)):
                    self.table.setItem(r, c, QTableWidgetItem(v))
            self._update_status(keys)
        finally:
            self._loading = False

    def _shared_value(self, keys, col):
        """(lo_text, hi_text): the shared value, '' if unset, '<varies>' if the
        selected datasets disagree -- mirroring how the x line edit aggregates."""
        vals = []
        for k in keys:
            pair = (self.host.limits.get(k, {}) or {}).get(col, (None, None))
            vals.append(tuple(pair))
        if not vals:
            return '', ''
        out = []
        for i in (0, 1):
            s = set(v[i] for v in vals)
            if len(s) > 1:
                out.append('<varies>')
            else:
                v = vals[0][i]
                out.append('' if v is None else ('%g' % v))
        return out[0], out[1]

    def _update_status(self, keys):
        n_lim = 0
        n_act = None
        for k in keys:
            lim = self.host.limits.get(k, {}) or {}
            n_lim = max(n_lim, len([c for c in lim if c != 'x']))
            m = self.host._column_mask(k)
            if m is not None:
                n_act = int(np.count_nonzero(m)) if n_act is None else n_act
        msg = '%d extra column limit(s)' % n_lim
        if n_act is not None:
            msg += '   |   %d active point(s)' % n_act
            if n_act == 0:
                msg += '  <-- nothing left to fit'
        self.statusLabel.setText(msg)
        self.statusLabel.setStyleSheet('color: red' if n_act == 0 else '')
        self.host._refresh_limits_badge()

    # --------------------------------------------------------------- edits --
    def _on_item_changed(self, item):
        if self._loading or item.column() not in (1, 2):
            return
        col = self._cols[item.row()]
        txt = item.text().strip()
        if txt in ('', '<varies>'):
            val = None
        else:
            try:
                val = float(txt)
            except ValueError:
                self._loading = True                 # revert, like rangeTable does
                item.setText('')
                self._loading = False
                val = None
        self.host.set_column_limit(col, item.column() - 1, val)
        self.refresh()

    def _clear_all(self):
        for k in list(self.host.sfnames or []):
            keep = (self.host.limits.get(k, {}) or {}).get('x')
            self.host.limits[k] = ({'x': keep} if keep is not None else {})
        self.host.update_plot()
        self.refresh()


class minMaxDialog(QDialog):
    def __init__(self, value, vary=0, minimum=None, maximum=None, expr=None, brute_step=None, parent=None, title=None):
        QDialog.__init__(self, parent)
        self.value = value
        self.vary = vary
        if minimum is None:
            self.minimum = -np.inf
        else:
            self.minimum = minimum
        if maximum is None:
            self.maximum = np.inf
        else:
            self.maximum = maximum
        self.expr = expr
        self.brute_step = brute_step
        self.createUI()
        if title is not None:
            self.setWindowTitle(title)
        
    def createUI(self):
        self.vblayout = QVBoxLayout(self)
        self.layoutWidget = pg.LayoutWidget()
        self.vblayout.addWidget(self.layoutWidget)
        
        valueLabel = QLabel('Value')
        self.layoutWidget.addWidget(valueLabel)
        self.layoutWidget.nextColumn()
        self.valueLineEdit = QLineEdit(str(self.value))
        self.layoutWidget.addWidget(self.valueLineEdit)

        self.layoutWidget.nextRow()
        varyLabel = QLabel('Fit')
        self.layoutWidget.addWidget(varyLabel)
        self.layoutWidget.nextColumn()
        self.varyCheckBox = QCheckBox()
        self.layoutWidget.addWidget(self.varyCheckBox)
        if self.vary>0:
            self.varyCheckBox.setCheckState(Qt.Checked)
        else:
            self.varyCheckBox.setCheckState(Qt.Unchecked)

        self.layoutWidget.nextRow()
        minLabel = QLabel('Minimum')
        self.layoutWidget.addWidget(minLabel)
        self.layoutWidget.nextColumn()
        self.minimumLineEdit = QLineEdit(str(self.minimum))
        self.layoutWidget.addWidget(self.minimumLineEdit)
        
        self.layoutWidget.nextRow()
        maxLabel = QLabel('Maximum')
        self.layoutWidget.addWidget(maxLabel)
        self.layoutWidget.nextColumn()
        self.maximumLineEdit = QLineEdit(str(self.maximum))
        self.layoutWidget.addWidget(self.maximumLineEdit)
        
        self.layoutWidget.nextRow()
        exprLabel = QLabel('Expr')
        self.layoutWidget.addWidget(exprLabel)
        self.layoutWidget.nextColumn()
        self.exprLineEdit = QLineEdit(str(self.expr))
        self.layoutWidget.addWidget(self.exprLineEdit)
        
        self.layoutWidget.nextRow()
        bruteStepLabel = QLabel('Brute step')
        self.layoutWidget.addWidget(bruteStepLabel)
        self.layoutWidget.nextColumn()
        self.bruteStepLineEdit = QLineEdit(str(self.brute_step))
        self.layoutWidget.addWidget(self.bruteStepLineEdit)
        
        self.layoutWidget.nextRow()
        self.cancelButton = QPushButton('Cancel')
        self.cancelButton.clicked.connect(self.cancelandClose)
        self.layoutWidget.addWidget(self.cancelButton)
        self.layoutWidget.nextColumn()
        self.okButton = QPushButton('OK')
        self.okButton.clicked.connect(self.okandClose)
        self.layoutWidget.addWidget(self.okButton)
        self.okButton.setDefault(True)
        
    def okandClose(self):
        # try:
        if type(eval(self.valueLineEdit.text())*1.0)==float:
            self.value = float(self.valueLineEdit.text())
        else:
            QMessageBox.warning(self, 'Value Error',
                                'Please enter floating point number for Value', QMessageBox.Ok)
            self.minimumLineEdit.setText(str(self.minimum))
            return
        if self.varyCheckBox.checkState() == Qt.Checked:
            self.vary = 1
        else:
            self.vary = 0

        minimum=self.minimumLineEdit.text()
        if '-inf' in minimum:
            self.minimum=-np.inf
        elif type(eval(self.minimumLineEdit.text())*1.0)==float:
            self.minimum=float(self.minimumLineEdit.text())
        else:
            QMessageBox.warning(self,'Value Error',
                                'Please enter floating point number for Minimum value',QMessageBox.Ok)
            self.minimumLineEdit.setText(str(self.minimum))
            return

        maximum = self.maximumLineEdit.text()
        if 'inf' in maximum:
            self.maximum=np.inf
        elif type(eval(self.maximumLineEdit.text())*1.0)==float:
            self.maximum = float(self.maximumLineEdit.text())
        else:
            QMessageBox.warning(self, 'Value Error',
                                'Please enter floating point number for Maximum value', QMessageBox.Ok)
            self.maximumLineEdit.setText(str(self.maximum))
            return

        self.expr=self.exprLineEdit.text()
        if self.expr != 'None':
            self.vary=0
        if self.bruteStepLineEdit.text() != 'None':
            self.brute_step = float(self.bruteStepLineEdit.text())
        else:
            self.brute_step = None
        self.accept()
        # except:
        #     QMessageBox.warning(self,'Value Error','Value, Min, Max should be floating point numbers\n\n'+traceback.format_exc(),QMessageBox.Ok)

    def cancelandClose(self):
        self.reject()

class FitResultDialog(QDialog):
    def __init__(self,fit_report,fit_info,parent=None):
        QDialog.__init__(self,parent)
        self.setWindowTitle('Fit Results')
        self.fit_report=fit_report
        self.fit_info=fit_info
        self.createUI()
        self.resize(600,400)
        
    def createUI(self):
        self.vblayout=QVBoxLayout(self)
        self.layoutWidget=pg.LayoutWidget()
        self.vblayout.addWidget(self.layoutWidget)
        
        fitReportLabel=QLabel('Fit Report')
        self.layoutWidget.addWidget(fitReportLabel,colspan=2)
        self.layoutWidget.nextRow()
        self.fitReportTextEdit=QTextEdit()
        self.fitReportTextEdit.setText(self.fit_report)
        self.layoutWidget.addWidget(self.fitReportTextEdit,colspan=2)
        
        self.layoutWidget.nextRow()
        fitInfoLabel=QLabel('Fit Info')
        self.layoutWidget.addWidget(fitInfoLabel,colspan=2)
        self.layoutWidget.nextRow()
        self.fitInfoTextEdit=QTextEdit()
        self.fitInfoTextEdit.setText(self.fit_info)
        self.layoutWidget.addWidget(self.fitInfoTextEdit,colspan=2)
        
        self.layoutWidget.nextRow()
        self.cancelButton=QPushButton('Reject')
        self.cancelButton.clicked.connect(self.cancelandClose)
        self.layoutWidget.addWidget(self.cancelButton,col=0)
        self.okButton=QPushButton('Accept')
        self.okButton.clicked.connect(self.okandClose)
        self.layoutWidget.addWidget(self.okButton,col=1)
        self.okButton.setDefault(True)
        
    def okandClose(self):
        self.accept()
        
    def cancelandClose(self):
        self.reject()


class DataListWidget(QListWidget):
    """Data-file list with a custom plain-click rule for multi-selection.

    Qt's ExtendedSelection collapses a multi-selection on any plain click. We want:
      * plain click on a dataset ALREADY in a multi-selection -> keep the whole
        selection (all curves/EDPs stay co-plotted) and just switch which dataset's
        parameters fill the tables (the "detail");
      * plain click on a NEW dataset -> collapse to it (Qt default);
      * Ctrl+click -> toggle add/remove (Qt default).
    The detail switch is delegated to the host's ``_set_detail_dataset`` (which also
    honours the "don't overwrite the tables for an unfit dataset" rule)."""
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._app = app

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and e.modifiers() == Qt.NoModifier:
            item = self.itemAt(e.pos())
            if item is not None and item.isSelected() and len(self.selectedItems()) > 1:
                # within-selection plain click: switch detail, DON'T collapse
                try:
                    self._app._set_detail_dataset(item.text())
                except Exception:
                    import traceback; traceback.print_exc()
                return
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        # keep double-click -> open-data-dialog working regardless of the press rule
        item = self.itemAt(e.pos())
        if item is not None:
            try:
                self._app.openDataDialog(item)
            except Exception:
                import traceback; traceback.print_exc()
            return
        super().mouseDoubleClickEvent(e)


class XModFit(QWidget):
    """
    This widget class is developed to provide an end-user a *Graphical User Interface* by which either they can \
    develop their own fitting functions in python or use the existing fitting functions under different categories\
     to analyze different kinds of one-dimensional data sets. `LMFIT <https://lmfit.github.io/lmfit-py/>`_ is extensively\
      used within this widget.
    
    **Features**
    
    1. Read and fit multiple data files
    2. Already available functions are categorized as per the function types and techniques
    3. Easy to add more categories and user-defined functions
    4. Once the function is defined properly all the free and fitting parameters will be available within the GUI as tables.
    5. An in-built Function editor with a function template is provided.
    6. The function editor is enabled with python syntax highlighting.
    
    **Usage**
    
    :class:`XModFit` can be used as stand-alone python fitting package by running it in terminal as::
    
        $python xmodfit.py
        
    .. figure:: Figures/Fit_widget.png
       :figwidth: 100%
       
       **Fit Widget** in action.
    
    Also it can be used as a widget with any other python application.
    """
    
    def __init__(self,parent=None):
        QWidget.__init__(self,parent)
        self.vblayout=QVBoxLayout(self)
        self.menuBar = QMenuBar(self)
        self.menuBar.setNativeMenuBar(False)
        self.create_menus()
        self.vblayout.addWidget(self.menuBar,0)
        self.mainDock=DockArea(self,parent)
        self.vblayout.addWidget(self.mainDock,5)

        self.emcee_walkers = 100
        self.emcee_steps = 100000
        self.emcee_burn = 0
        self.emcee_thin = 1
        self.emcee_cores = 1
        self.emcee_frac = self.emcee_burn/self.emcee_steps
        self.reuse_sampler = False
        self.autoCalculate = True
        self.funcDock=Dock('Functions',size=(1,6),closable=False,hideTitle=False)
        self.fitDock=Dock('Fit options',size=(1,2),closable=False,hideTitle=False)
        self.dataDock=Dock('Data',size=(1,8),closable=False,hideTitle=False)
        self.paramDock=Dock('Parameters',size=(2,8),closable=False,hideTitle=False)
        self.plotDock=Dock('Data and Fit',size=(5,8),closable=False,hideTitle=False)
        self.fitResultDock=Dock('Fit Results',size=(5,8),closable=False,hideTitle=False)
        self.metaDataDock=Dock('Meta Data',size=(5,8),closable=False,hideTitle=False)
        self.consoleDock=Dock('Console',size=(5,8),closable=False,hideTitle=False)
        self.mainDock.addDock(self.dataDock)
        self.mainDock.addDock(self.fitDock,'bottom')
        self.mainDock.addDock(self.paramDock,'right')
        self.mainDock.addDock(self.metaDataDock, 'right')
        self.mainDock.addDock(self.consoleDock, 'right')
        self.mainDock.addDock(self.metaDataDock,'above',self.consoleDock)
        self.mainDock.addDock(self.fitResultDock, 'above',self.metaDataDock)
        self.compareDock=Dock('Model Comparison',size=(5,8),closable=False,hideTitle=False)
        self.mainDock.addDock(self.compareDock,'above',self.fitResultDock)
        self.mainDock.addDock(self.plotDock,'above',self.compareDock)

        self.mainDock.addDock(self.funcDock,'above',self.dataDock)
        # The joint-fit controls are no longer a separate tab; they live under the Data
        # list (overlay), collapsed until 'Enable joint fit' -- see create_jointFitDock.
        self.special_keys=['x','params','choices','output_params','__mpar__','filepaths']
        self.curr_funcClass={}
        
        
        self.data={}
        self.meta={}
        # Fit limits, generalized: limits[data_key] = {col_name: (lo, hi)}, with
        # 'x' reserved for the plotted x column. xmin/xmax are MutableMapping
        # views onto the 'x' entry, so every existing call site is untouched.
        self.limits={}
        self.xmin=_LimitProxy(self.limits, 0)
        self.xmax=_LimitProxy(self.limits, 1)
        self.dlg_data={}
        self.plotColIndex={}
        self.plotColors={}
        # Per-dataset fit memory: composite dataset name "<num><>filepath" -> Fit.
        # Each dataset owns its own model instance + params + result, so switching
        # datasets restores that dataset's model/params/curve (see _fit_for_dataset).
        self.dataset_fits={}
        # Per-dataset covariance (spec 17 6.2): data key -> covariance_core.DatasetCovariance,
        # its (path, raw path), its mode ('model'|'data') and one-shot report notes.
        self.covariances={}
        self.covariance_paths={}
        self.covariance_modes={}
        self._cov_pending_notes={}
        self._template_fit=None        # model shown with no dataset selected; seeds new datasets
        self._switching_dataset=False  # re-entrancy guard for the select->tables->plot path
        self.jointFit=None             # persistent compiled JointFit (retains joint results
                                       # across individual-dataset views); see run_joint_fit
        self._joint_overlay_names=[]   # datasets currently overlaid in the joint plot (drives
                                       # multi-dataset EDP co-plotting in plot_extra_param)
        self._mcmc_is_joint=False      # True while the dedicated joint-fit MCMC window is open
                                       # (routes Start->run_joint_emcee, enables joint-only viz)
        self.seed_temperature=1.0      # XLayersPDB MCMC walker-seeding temperature (GUI spinbox);
                                       # 1=posterior-weighted, higher=more mode spread
        self.seed_mode='joint'         # both-shared joint seed source: 'joint' landscape (default)
                                       # vs 'individual' per-dataset landscapes
        # Deterministic, well-separated plot palette: a dataset's data (markers) and its
        # fit (line) share one color; distinct curves get distinct colors (see
        # _curve_color_for). Tableau-10.
        self._curve_palette=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                             '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        self._curve_color={}
        self._curve_color_ct=0
        self.curDir=os.getcwd()
        self.fileNumber=0
        self.fileNames={}
        self.fchanged=True
        self.chisqr='None'
        self.format='%.6e'
        self.backend=None
        self.gen_param_items=[]
        self._image_autoselected=False
        self.doubleValidator=QDoubleValidator()
        self.intValidator=QIntValidator()
        self.tApp_Clients={}
        self.tModules={}
        self.fitMethods={'Levenberg-Marquardt':'leastsq',
                         'Scipy-Least-Squares':'least_squares',
                         'Differential-Evolution': 'differential_evolution',
                         # 'Brute-Force-Method':'brute',
                         'Nelder-Mead':'nelder'}
                         # 'L-BFGS-B':'lbfgsb',
                         # 'Powell':'powell',
                         # 'Congugate-Gradient':'cg',
                         # 'Newton-CG-Trust-Region':'trust-ncg',
                         # 'COBLYA':'cobyla',
                         # 'Truncate-Newton':'tnc',
                         # 'Exact-Trust-Region':'trust-exact',
                         # 'Dogleg':'dogleg',
                         # 'Sequential-Linear-Square':'slsqp',
                         # 'Adaptive-Memory-Programming':'ampgo',
                         # 'Maximum-Likelihood-MC-Markov-Chain':'emcee'}
                         #
        
        self.create_funcDock()
        self.create_fitDock()
        self.create_dataDock()
        self.create_jointFitDock()
        self.create_plotDock()
        self.create_fitResultDock()
        self.create_compareDock()
        self.create_metaDataDock()
        self.create_consoleDock()
        self.update_catagories()
        self.create_paramDock()
        # self.xminmaxChanged()
        self.sfnames=None
        self.expressions={}

    def create_menus(self):
        self.fileMenu = self.menuBar.addMenu('&File')
        self.settingsMenu = self.menuBar.addMenu('&Settings')
        self.consoleMenu = self.menuBar.addMenu('&Console')
        self.toolMenu = self.menuBar.addMenu('&Tools')
        self.helpMenu = self.menuBar.addMenu('&Help')

        saveJoint=QAction('Save &Joint Fit (.jpar)',self)
        saveJoint.triggered.connect(lambda x: self.saveJointFit(fname=None))
        self.fileMenu.addAction(saveJoint)
        loadJoint=QAction('Load Join&t Fit (.jpar)',self)
        loadJoint.triggered.connect(lambda x: self.loadJointFit(fname=None))
        self.fileMenu.addAction(loadJoint)
        parIntoJoint=QAction('Load .par &into Joint Fit',self)
        parIntoJoint.triggered.connect(lambda x: self.loadParIntoJoint(fname=None))
        self.fileMenu.addAction(parIntoJoint)
        self.fileMenu.addSeparator()

        quit=QAction('Quit',self)
        quit.triggered.connect(self.close)
        self.fileMenu.addAction(quit)

        parFormat=QAction('&Parameter format',self)
        parFormat.triggered.connect(self.changeParFormat)
        self.settingsMenu.addAction(parFormat)

        loadConsoleScript=QAction('&Load Script',self)
        loadConsoleScript.triggered.connect(self.loadConsoleScript)
        self.consoleMenu.addAction(loadConsoleScript)

        updateDataMethod = QAction('&Update Data', self)
        updateDataMethod.triggered.connect(self.updateData)
        self.consoleMenu.addAction(updateDataMethod)

        about=QAction('&About',self)
        about.triggered.connect(self.aboutDialog)
        self.helpMenu.addAction(about)


        toolItems=os.listdir(os.path.join(os.curdir,'Tools'))
        self.toolDirs=[]
        self.toolApps={}
        for item in toolItems:
            if '__' not in item:
                self.toolDirs.append(self.toolMenu.addMenu('&%s'%item))
                tApps=glob.glob(os.path.join(os.curdir,'Tools',item,'*.py'))
                for app in tApps:
                    tname='&'+os.path.basename(os.path.splitext(app)[0])
                    self.toolApps[tname]=app
                    tApp=QAction(tname,self)
                    tApp.triggered.connect(self.launch_tApp)
                    self.toolDirs[-1].addAction(tApp)

    def changeParFormat(self):
        dlg=MultiInputDialog(inputs={'Format':self.format},title='Parameter format')
        if dlg.exec_():
            self.format=dlg.inputs['Format']
            try:
                self.update_sfit_parameters()
                self.update_mfit_parameters_new()
            except:
                pass

    def launch_tApp(self):
        tname=self.sender().text()
        module_name=".".join(os.path.splitext(self.toolApps[tname])[0].split(os.sep)[1:])
        if module_name not in sys.modules:
            self.tModules[module_name]=importlib.import_module(module_name)
        tmodule=self.tModules[module_name]
        if tmodule in self.tApp_Clients:
            self.tApp_Clients[tmodule].show()
        else:
            tclass = getattr(tmodule, tname[1:])
            self.tApp_Clients[tmodule]=tclass(self)
            self.tApp_Clients[tmodule].setWindowTitle(tname[1:])
            self.tApp_Clients[tmodule].show()

    def loadConsoleScript(self):
        fname=QFileDialog.getOpenFileName(self,'Open Console Scripts',directory=self.curDir,filter="Script Files (*.py)")[0]
        if fname!='':
            # fh=open(fname,'r')
            # commands="".join(fh.readlines())
            # print(commands)
            self.consoleWidget.print_text("%load "+fname)
            # self.consoleWidget._execute(commands,False)
            self.consoleDock.raiseDock()

    def create_consoleDock(self):
        self.consoleLayout = pg.LayoutWidget(self)
        row = 0
        col = 0
        self.consoleWidget=ConsoleWidget(parent=self)
        dataDlg=Data_Dialog()
        self.consoleWidget.push_vars({'readData':dataDlg.readData, 'data': {},
                                      'plotColIndex': {}, 'plotColors': {},
                                      'expressions': {}})
        self.consoleLayout.addWidget(self.consoleWidget, row=row, col=col)
        self.consoleDock.addWidget(self.consoleLayout)

    def aboutDialog(self):
        QMessageBox.information(self,'About','Copyright (c) %d NSF\'s ChemMAtCARS, University of Chicago.\n\n'
                                             'Developers:\n'
                                             'Mrinal K. Bera (mrinalkb@cars.uchicago.edu\n'
                                             'Wei Bu (bu@cars.uchicago.edu)'%datetime.date.today().year
                                             ,QMessageBox.Ok)
        
    def create_funcDock(self):
        self.funcLayoutWidget=pg.LayoutWidget(self)
        row=0
        col=0
        funcCategoryLabel=QLabel('Function Categories:')
        self.funcLayoutWidget.addWidget(funcCategoryLabel,row=row,col=col,colspan=2)
        
        row+=1
        col=0
        self.addCategoryButton=QPushButton('Create')
        self.addCategoryButton.clicked.connect(self.addCategory)
        self.funcLayoutWidget.addWidget(self.addCategoryButton,row=row,col=col)
        col+=1
        self.removeCategoryButton=QPushButton('Remove')
        self.removeCategoryButton.clicked.connect(self.removeCategory)
        self.funcLayoutWidget.addWidget(self.removeCategoryButton,row=row,col=col)
        
        row+=1
        col=0
        self.categoryListWidget=QListWidget()
        self.categoryListWidget.currentItemChanged.connect(self.update_functions)
        self.funcLayoutWidget.addWidget(self.categoryListWidget,row=row,col=col,colspan=2)
        
        row+=1
        col=0
        funcLabel=QLabel('Functions:')
        self.funcLayoutWidget.addWidget(funcLabel,row=row,col=col,colspan=2)
        
        row+=1
        col=0
        self.addFuncButton=QPushButton('Create')
        self.addFuncButton.clicked.connect(self.addFunction)
        self.funcLayoutWidget.addWidget(self.addFuncButton,row=row,col=col)
        col+=1
        self.removeFuncButton=QPushButton('Remove')
        self.removeFuncButton.clicked.connect(self.removeFunction)
        self.funcLayoutWidget.addWidget(self.removeFuncButton,row=row,col=col)
        
        row+=1
        col=0
        self.funcListWidget=QListWidget()
        self.funcListWidget.setSelectionMode(4)
        self.funcListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.funcListWidget.customContextMenuRequested.connect(self.funcListRightClicked)
        self.funcListWidget.itemSelectionChanged.connect(self.functionChanged)
        self.funcListWidget.itemDoubleClicked.connect(self.openFunction)
        self.funcLayoutWidget.addWidget(self.funcListWidget,row=row,col=col,colspan=2)
        
        self.funcDock.addWidget(self.funcLayoutWidget)

    def funcListRightClicked(self,pos):
        popMenu = QMenu()
        showDet = QAction("Show Details", self)
        addDet = QAction("Upload Details", self)
        modDet = QAction("Create/Modify Details", self)
        popMenu.addAction(showDet)
        popMenu.addAction(addDet)
        popMenu.addAction(modDet)
        showDet.triggered.connect(self.showDetails)
        addDet.triggered.connect(self.addDetails)
        modDet.triggered.connect(self.modifyDetails)
        popMenu.exec_(self.funcListWidget.mapToGlobal(pos))

    def showDetails(self):
        url = os.path.join(os.path.abspath(os.path.curdir), 'Function_Details', self.categoryListWidget.currentItem().text(),
                            self.funcListWidget.currentItem().text(),'help.pdf')
        if os.path.exists(url):
            webbrowser.open_new_tab('File://'+url)
        else:
            QMessageBox.warning(self,'File Error','The help files regarding the function details do not exist.',QMessageBox.Ok)
        # os.system('C:/Users/mrinalkb/Desktop/ESH738.pdf')

    def addDetails(self):
        path=os.path.join(os.path.abspath(os.path.curdir),'Function_Details',self.categoryListWidget.currentItem().text(),self.funcListWidget.currentItem().text())
        if os.path.exists(path):
            fname = QFileDialog.getOpenFileName(self,caption='Select help file',directory=self.curDir,filter="Help files (*.docx *.pdf)")[0]
            tfname=os.path.join(path,'help'+os.path.splitext(fname)[1])
            shutil.copy(fname,tfname)
        else:
            os.makedirs(path)

    def modifyDetails(self):
        category=self.categoryListWidget.currentItem().text()
        function=self.funcListWidget.currentItem().text()
        path = os.path.join(os.path.abspath(os.path.curdir), 'Function_Details', category,
                            function,'help.docx')
        if os.path.exists(path):
            webbrowser.open_new_tab('File://'+path)
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            doc=Document()
            doc.add_heading('Details of %s/%s'%(category,function),0)
            module = 'Functions.%s.%s' % (category,function)
            text=getattr(self.curr_funcClass[module], function).__init__.__doc__
            doc.add_paragraph(text)
            doc.save(path)
            webbrowser.open_new_tab('File://'+path)

        
    def addCategory(self):
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)
        tdir=QFileDialog.getExistingDirectory(self,'Select a folder','./Functions/',QFileDialog.ShowDirsOnly)
        if tdir!='': 
            cdir=os.path.basename(os.path.normpath(tdir))
            fh=open(os.path.join(tdir,'__init__.py'),'w')
            fh.write('__all__=[]')
            fh.close()
            if cdir not in self.categories:
                self.categories.append(cdir)
                self.categoryListWidget.addItem(cdir)
            else:
                QMessageBox.warning(self,'Category error','Category already exist!',QMessageBox.Ok)

        
    def removeCategory(self):
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)
        self.funcListWidget.clear()
        if len(self.categoryListWidget.selectedItems())==1:
            ans=QMessageBox.question(self,'Delete warning','Are you sure you would like to delete the category?',
                                     QMessageBox.No,QMessageBox.Yes)
            if ans==QMessageBox.Yes:
                category=os.path.abspath('./Functions/%s'%self.categoryListWidget.currentItem().text())
                #os.rename(category,)
                shutil.rmtree(category)
                self.categories.remove(self.categoryListWidget.currentItem().text())
                self.categoryListWidget.takeItem(self.categoryListWidget.currentRow())
        elif len(self.categoryListWidget.selectedItems())>1:
            QMessageBox.warning(self,'Warning','Please select only one category at a time to remove',QMessageBox.Ok)
        else:
            QMessageBox.warning(self,'Warning','Please select one category atleast to remove',QMessageBox.Ok)
            
            
    def openFunction(self):
        dirName=os.path.abspath('./Functions/%s'%self.categoryListWidget.currentItem().text())
        funcName=self.funcListWidget.currentItem().text()
        try:
            if not self.funcEditor.open: 
                self.funcEditor=FunctionEditor(funcName=funcName,dirName=dirName)
                self.funcEditor.setWindowTitle('Function editor')
                self.funcEditor.show()
                self.funcOpen=self.funcEditor.open
                self.funcEditor.closeEditorButton.clicked.connect(self.postAddFunction)
            else:
                QMessageBox.warning(self,'Warning','You cannot edit two functions together',QMessageBox.Ok)
        except:
            self.funcEditor=FunctionEditor(funcName=funcName,dirName=dirName)
            self.funcEditor.setWindowTitle('Function editor')
            self.funcEditor.show()
            self.funcEditor.closeEditorButton.clicked.connect(self.postAddFunction)
                
    def addFunction(self):
        if len(self.categoryListWidget.selectedItems())==1:
            dirName=os.path.abspath('./Functions/%s'%self.categoryListWidget.currentItem().text())
            self.funcEditor=FunctionEditor(dirName=dirName)
            self.funcEditor.setWindowTitle('Function editor')
            self.funcEditor.show()
            self.funcEditor.closeEditorButton.clicked.connect(self.postAddFunction)
        else:
            QMessageBox.warning(self,'Category Error','Please select a Category first',QMessageBox.Ok)
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)

        
        
    def postAddFunction(self):
        if self.funcEditor.funcNameLineEdit.text()!='tmpxyz':
            dirName=os.path.abspath('./Functions/%s'%self.categoryListWidget.currentItem().text())
            fh=open(os.path.join(dirName,'__init__.py'),'r')
            line=fh.readlines()
            fh.close()
            funcList=eval(line[0].split('=')[1])
            funcName=self.funcEditor.funcNameLineEdit.text()
            if funcName not in funcList:
                funcList.append(funcName)
                funcList=sorted(list(set(funcList)),key=str.lower)
                os.remove(os.path.join(dirName,'__init__.py'))
                fh=open(os.path.join(dirName,'__init__.py'),'w')
                fh.write('__all__='+str(funcList))
                fh.close()
            self.update_functions()
        
        
    
    def removeFunction(self):
        if len(self.funcListWidget.selectedItems())==1:
            ans=QMessageBox.question(self,'Warning','Are you sure you would like to remove the function',
                                     QMessageBox.No,QMessageBox.Yes)
            if ans==QMessageBox.Yes:
                dirName=os.path.abspath('./Functions/%s'%self.categoryListWidget.currentItem().text())
                fname=self.funcListWidget.currentItem().text()
                fh=open(os.path.join(dirName,'__init__.py'),'r')
                line=fh.readlines()
                fh.close()
                funcList=eval(line[0].split('=')[1])
                try:
                    os.remove(os.path.join(dirName,fname+'.py'))
                    os.remove(os.path.join(dirName,'__init__.py'))
                    fh=open(os.path.join(dirName,'__init__.py'),'w')
                    fh.write('__all__='+str(funcList))
                    fh.close()
                    self.update_functions()
                except:
                    QMessageBox.warning(self,'Remove error','Cannot remove the function because the function file\
                     might be open elsewhere.\n\n'+traceback.format_exc(),QMessageBox.Ok)
        elif len(self.funcListWidget.selectedItems())>1:
            QMessageBox.warning(self,'Warning','Please select only one function at a time to remove',QMessageBox.Ok)
        else:
            QMessageBox.warning(self,'Warning','Please select one function atleast to remove',QMessageBox.Ok)
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)
        
    def create_dataDock(self):
        self.dataLayoutWidget=pg.LayoutWidget(self)
        
        datafileLabel=QLabel('Data files')
        self.dataLayoutWidget.addWidget(datafileLabel,colspan=2)
        
        self.dataLayoutWidget.nextRow()
        self.addDataButton=QPushButton('Add files')
        self.dataLayoutWidget.addWidget(self.addDataButton)
        self.addDataButton.clicked.connect(lambda x: self.addData())
        self.removeDataButton=QPushButton('Remove Files')
        self.dataLayoutWidget.addWidget(self.removeDataButton,col=1)
        self.removeDataButton.clicked.connect(self.removeData)
        self.removeDataShortCut = QShortcut(QKeySequence.Delete, self)
        self.removeDataShortCut.activated.connect(self.removeData)
        self.dataLayoutWidget.nextRow()
        self.loadCovButton=QPushButton('Load covariance...')
        self.loadCovButton.setToolTip('Attach a covariance file to the selected dataset (single fits)')
        self.dataLayoutWidget.addWidget(self.loadCovButton)
        self.loadCovButton.clicked.connect(lambda x: self._load_covariance_dialog())
        self.removeCovButton=QPushButton('Remove covariance')
        self.dataLayoutWidget.addWidget(self.removeCovButton,col=1)
        self.removeCovButton.clicked.connect(lambda x: self._remove_covariance_selected())
        self.dataLayoutWidget.nextRow()
        self.covStatusLabel=QLabel('')     # last covariance message (attach/remove/reload)
        self.covStatusLabel.setWordWrap(True)
        self.covStatusLabel.setVisible(False)
        self.dataLayoutWidget.addWidget(self.covStatusLabel,colspan=2)
        
        
        self.dataLayoutWidget.nextRow()
        self.dataListWidget=DataListWidget(self)
        self.dataListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.dataListWidget.setItemDelegate(NameDelegate(self.dataListWidget))  # show basenames
        self.dataListWidget.itemSelectionChanged.connect(self.dataFileSelectionChanged)
        # (double-click -> openDataDialog is handled in DataListWidget.mouseDoubleClickEvent)
        self.dataLayoutWidget.addWidget(self.dataListWidget,colspan=2)
        
        self.dataDock.addWidget(self.dataLayoutWidget)

    def create_fitDock(self):
        self.fitLayoutWidget=pg.LayoutWidget(self)

        xminmaxLabel = QLabel('Xmin:Xmax')
        self.fitLayoutWidget.addWidget(xminmaxLabel)
        self.xminmaxLineEdit = QLineEdit('0:1')
        self.xminmaxLineEdit.returnPressed.connect(self.xminmaxChanged)
        self.fitLayoutWidget.addWidget(self.xminmaxLineEdit, col=1)
        # The x window stays a one-line affordance -- it is what ordinary 1-D data
        # needs and every existing read/write site depends on it. Limits on OTHER
        # data columns (e.g. q_xy and q_z of a 2D map, whose x is a row index) live
        # behind this button, which stays out of the way until used.
        self.limitsButton = QToolButton()
        self.limitsButton.setText('Limits')
        self.limitsButton.setToolTip('Fit limits on any other data column '
                                     '(blank = unbounded)')
        self.limitsButton.clicked.connect(self.openColumnLimitsDialog)
        self.fitLayoutWidget.addWidget(self.limitsButton, col=2)

        self.fitLayoutWidget.nextRow()
        fitMethodLabel = QLabel('Fit Method')
        self.fitLayoutWidget.addWidget(fitMethodLabel)
        self.fitMethodComboBox = QComboBox()
        self.fitMethodComboBox.addItems(list(self.fitMethods.keys()))
        self.fitLayoutWidget.addWidget(self.fitMethodComboBox, col=1)

        self.fitLayoutWidget.nextRow()
        fitScaleLabel = QLabel('Fit Scale')
        self.fitLayoutWidget.addWidget(fitScaleLabel)
        self.fitScaleComboBox = QComboBox()
        self.fitScaleComboBox.addItems(['Linear', 'Linear w/o error', 'Log', 'Log w/o error'])
        self.fitLayoutWidget.addWidget(self.fitScaleComboBox, col=1)

        self.fitLayoutWidget.nextRow()
        covModeLabel = QLabel('Covariance Mode')
        self.fitLayoutWidget.addWidget(covModeLabel)
        self.covModeComboBox = QComboBox()
        self.covModeComboBox.addItems(['model', 'data'])
        self.covModeComboBox.setToolTip("Selected dataset's covariance: 'model' builds the intensity-"
                                        "dependent terms from the model, 'data' from the data")
        self.covModeComboBox.setEnabled(False)       # enabled only for a dataset with a covariance
        self.covModeComboBox.currentTextChanged.connect(self._cov_mode_changed)
        self.fitLayoutWidget.addWidget(self.covModeComboBox, col=1)

        self.fitLayoutWidget.nextRow()
        fitIterationLabel = QLabel('Fit Iterations')
        self.fitLayoutWidget.addWidget(fitIterationLabel)
        self.fitIterationLineEdit = QLineEdit('1000')
        self.fitLayoutWidget.addWidget(self.fitIterationLineEdit, col=1)

        self.fitLayoutWidget.nextRow()
        self.fitButton = QPushButton('Fit')
        self.fitButton.clicked.connect(lambda x: self.doFit())
        self.fitButton.setEnabled(False)
        self.unfitButton = QPushButton('Undo fit')
        self.unfitButton.clicked.connect(self.undoFit)
        self.fitLayoutWidget.addWidget(self.unfitButton)
        self.fitLayoutWidget.addWidget(self.fitButton, col=1)

        self.fitLayoutWidget.nextRow()
        confIntervalMethodLabel=QLabel('Confidence Interval Method')
        self.confIntervalMethodComboBox=QComboBox()
        self.confIntervalMethodComboBox.addItems(['ChiSqrDist', 'MCMC'])
        self.fitLayoutWidget.addWidget(confIntervalMethodLabel)
        self.fitLayoutWidget.addWidget(self.confIntervalMethodComboBox,col=1)

        self.fitLayoutWidget.nextRow()
        self.showConfIntervalButton = QPushButton('Show Param Error')
        self.showConfIntervalButton.setDisabled(True)
        self.showConfIntervalButton.clicked.connect(lambda: self._show_param_error())
        self.calcConfInterButton = QPushButton('Calculate Param Error')
        self.calcConfInterButton.clicked.connect(self.calcConfInterval)
        self.calcConfInterButton.setDisabled(True)
        self.fitLayoutWidget.addWidget(self.showConfIntervalButton)
        self.fitLayoutWidget.addWidget(self.calcConfInterButton, col=1)

        self.fitDock.addWidget(self.fitLayoutWidget)


    def dataFileSelectionChanged(self):
        # A normal selection change repaints for the new detail, so clear any stale
        # "unfit dataset" indicator (it only applies to a detail switch WITHIN a
        # preserved multi-selection; see _set_detail_dataset).
        self._clear_nofit_indicator()
        self.sfnames=[]
        self.pfnames=[]
        for item in self.dataListWidget.selectedItems():
            self.sfnames.append(item.text())
            txt=item.text()
            fnum=txt.split('<>')[0]
            self.pfnames=self.pfnames+[fnum+':'+key for key in self.data[txt].keys()]
            self.meta[txt]['File Number'] = int(fnum)
        # Multi-select where the just-added/selected (last) dataset has NO fit: keep the
        # param tables on a REAL fit by making the detail the last FITTED selected
        # dataset (reorder so it is sfnames[-1], which drives the detail everywhere), and
        # flag the unfit one for the no-fit banner (shown after activation). Its EDP is
        # skipped in plot_extra_param. Single-select still shows the (possibly unfit)
        # dataset so the user can visualise initial values while setting up a fit.
        self._pending_nofit = None
        if len(self.sfnames) > 1 and not self._dataset_is_fitted(self.sfnames[-1]):
            _fitted = [n for n in self.sfnames if self._dataset_is_fitted(n)]
            if _fitted:
                self._pending_nofit = self.sfnames[-1]
                self.sfnames = [n for n in self.sfnames if n != _fitted[-1]] + [_fitted[-1]]
        if len(self.sfnames)>0:
            self.metaDataInfoUpdated(self.sfnames[-1])
            self.curDir = os.path.dirname(self.sfnames[-1].split('<>')[1])
            xmin=np.min([self.xmin[key] for key in self.sfnames])
            xmax=np.max([self.xmax[key] for key in self.sfnames])
            self.xminmaxLineEdit.setText('%0.6f:%0.6f'%(xmin,xmax))
            self._refresh_limits_badge()
            if getattr(self, 'columnLimitsDialog', None) is not None                     and self.columnLimitsDialog.isVisible():
                self.columnLimitsDialog.refresh()
            # self.xminmaxChanged()
            # if len(self.data[self.sfnames[-1]].keys())>1:
            #     text='{'
            #     for key in self.data[self.sfnames[-1]].keys():
            #         text+='"'+key+'":np.linspace(%.3f,%.3f,%d),'%(xmin,xmax,100)
            #     text=text[:-1]+'}'
            # else:
            #     text='np.linspace(%.3f,%.3f,100)'%(xmin,xmax)
            # self.xLineEdit.setText(text)
            self.fitButton.setEnabled(True)
        else:
            self.fitButton.setDisabled(True)
        # Per-dataset fit memory: restore the last-selected dataset's own model,
        # parameters and (via re-evaluation) curve. Order is load-bearing:
        #   (a) point self.fit at the stored Fit  ->  (b) repopulate the param tables
        #   from it  ->  (c) update_plot (tables -> fit -> curve).
        if len(self.sfnames)>0:
            detail=self.sfnames[-1]
            fit=self._fit_for_dataset(detail)
            if fit is not None:
                self._switching_dataset=True
                try:
                    # restore this dataset's Xrange expression + re-eval the model grid
                    xexpr=getattr(fit,'_xmod_xexpr',None)
                    if xexpr:
                        self.xLineEdit.setText(xexpr)
                        try:
                            self.x=eval(self.xLineEdit.text())
                            fit.func.x=self.x
                        except Exception:
                            pass
                    # auto-follow: highlight this dataset's function (guarded, best-effort)
                    self._sync_funclist_to_fit(fit)
                    # point self.fit at the stored fit + repopulate the tables from it
                    # BEFORE plotting (update_plot reads tables -> fit -> curve).
                    self._activate_dataset_fit(fit)
                finally:
                    self._switching_dataset=False
        # Banner on FIRST addition of an unfit dataset in a multi-selection: the tables
        # were kept on the fitted detail (reorder above), so signal that the just-added
        # dataset has no fit and its params/EDP aren't shown.
        if self._pending_nofit is not None:
            self._show_nofit_indicator(self._pending_nofit, self.sfnames[-1])
        # try:
        self.update_plot()
        # except:
        #     pass
        # self.update_plot()
        # self.xChanged()
        self.errorAvailable = False
        self.reuse_sampler = False
        # CI ("Calculate Param Error") reflects THIS dataset's fit status and persists
        # across switches (the dataset's Fit keeps its result/_is_fitted).
        self._update_ci_button_state()
        self._cov_sync_mode_combo()
        # Item B2 selection model: a PLAIN click is mutually exclusive with the joint
        # list (classic one-view behaviour -- selecting here releases the joint
        # selection). Hold Ctrl to KEEP the joint selection and overlay the individual
        # (dashed) + joint (solid) fitted curves.
        if len(self.sfnames) > 0 and not getattr(self, '_syncing_selection', False):
            if self._overlay_modifier_active():
                self._maybe_combined_overlay()
            else:
                jp = getattr(self, 'jointFitPanel', None)
                if jp is not None and jp.datasetList.selectedItems():
                    self._syncing_selection = True
                    jp.datasetList.blockSignals(True)
                    jp.datasetList.clearSelection()
                    jp.datasetList.blockSignals(False)
                    self._syncing_selection = False
        self._refresh_data_labels()

    def clear_individual_selection(self):
        """Clear the Data-list selection without firing dataFileSelectionChanged.
        Called by the joint panel when a joint dataset is selected (mutual exclusion)."""
        if getattr(self, '_syncing_selection', False):
            return
        self._syncing_selection = True
        try:
            self.dataListWidget.blockSignals(True)
            self.dataListWidget.clearSelection()
            self.dataListWidget.blockSignals(False)
        finally:
            self._syncing_selection = False
        # Switching to a pure joint view releases the individual selection, so the
        # individual no-fit banner/tint no longer applies -- clear it. (blockSignals
        # above suppresses dataFileSelectionChanged, which would otherwise clear it.)
        self._clear_nofit_indicator()

    def _overlay_modifier_active(self):
        """True when the overlay modifier (Ctrl) is held during a selection change.

        Plain clicks keep the two dataset lists mutually exclusive (one view at a
        time); Ctrl+click KEEPS the other list's selection so the individual + joint
        curves overlay (item B2). Isolated in one method so it can be overridden in
        headless tests (QApplication.keyboardModifiers() reads the live keyboard)."""
        try:
            return bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        except Exception:
            return False

    def clear_joint_view(self):
        """Blank-click / deselect-all in the joint-fit list drops the joint overlay.
        Falls back to the individual view if the Data list still has a selection, else
        clears the plot -- mirroring a blank-click in the general Data list."""
        self._joint_overlay_names = []
        if [it for it in self.dataListWidget.selectedItems()]:
            # repaint the individual view (points self.fit at an individual Fit)
            self.dataFileSelectionChanged()
        else:
            try:
                self.plotWidget.Plot([])
            except Exception:
                pass

    def _show_nofit_indicator(self, name, showing=None):
        """Signal that the param tables were NOT updated because ``name`` has no fit.
        Shows an amber banner AND tints the parameter tables (both, per user choice)."""
        try:
            base = os.path.basename(name.split('<>')[-1])
            msg = "⚠ '%s' has no fit yet — parameters not updated" % base
            if showing:
                msg += " (still showing '%s')" % os.path.basename(showing.split('<>')[-1])
            if getattr(self, 'noFitBanner', None) is not None:
                self.noFitBanner.setText(msg)
                self.noFitBanner.setVisible(True)
            tint = 'QTableWidget{background:#fff8e1;}'
            for t in (getattr(self, 'fixedParamTableWidget', None),
                      getattr(self, 'sfitParamTableWidget', None)):
                if t is not None:
                    t.setStyleSheet(tint)
            for mkey, t in getattr(self, 'mfitParamTableWidget', {}).items():
                try:
                    t.setStyleSheet(tint)
                except Exception:
                    pass
        except Exception:
            import traceback; traceback.print_exc()

    def _clear_nofit_indicator(self):
        """Hide the no-fit banner and remove the parameter-table tint."""
        if getattr(self, 'noFitBanner', None) is not None:
            self.noFitBanner.setVisible(False)
        for t in (getattr(self, 'fixedParamTableWidget', None),
                  getattr(self, 'sfitParamTableWidget', None)):
            if t is not None:
                t.setStyleSheet('')
        for mkey, t in getattr(self, 'mfitParamTableWidget', {}).items():
            try:
                t.setStyleSheet('')
            except Exception:
                pass

    def _dataset_is_fitted(self, name):
        """True if dataset ``name`` has a stored INDIVIDUAL fit result."""
        f = self.dataset_fits.get(name)
        return (f is not None and not hasattr(f, 'objectives')
                and (getattr(f, '_is_fitted', False) or getattr(f, 'result', None) is not None))

    def _curve_meta(self, name):
        """Parse a plotted-curve key -> (dataset_index or None, type_rank, dataset_key).
        type_rank: data=0, individual fit=1, joint fit=2, other=3. Recognizes the naming
        conventions used across the plot paths: ``N:col`` (data), ``.. [indiv:N]``,
        ``.. [joint:N]``, ``model [indiv:N]`` and a bottom-panel ``.. [N]`` suffix."""
        import re
        idx, rank = None, 3
        m = re.search(r'\[indiv:(\d+)\]', name)
        if m: idx, rank = int(m.group(1)), 1
        else:
            m = re.search(r'\[joint:(\d+)\]', name)
            if m: idx, rank = int(m.group(1)), 2
            else:
                m = re.match(r'(\d+):', name)                 # data "N:col"
                if m: idx, rank = int(m.group(1)), 0
                else:
                    m = re.search(r'\[(\d+)\]\s*$', name)      # bottom-panel "param [N]"
                    if m: idx, rank = int(m.group(1)), 0
        dkey = self._dataset_key_for_index(idx) if idx is not None else None
        return idx, rank, dkey

    def _dataset_key_for_index(self, idx):
        """The full data key ('<fnum><>path') whose file-number == idx, or None."""
        try:
            for k in self.data.keys():
                if int(k.split('<>')[0]) == idx:
                    return k
        except Exception:
            pass
        return None

    def _legend_label(self, name):
        """Friendly main-plot legend label: the dataset's short name (``num: basename``,
        the same text shown in the Data boxes), tagged by curve type. Data -> just the
        name; individual fit -> 'name (fit)'; joint fit -> 'name (joint)'. Unrecognized
        keys keep their raw text. (Uses the existing _short_name defined below.)"""
        idx, rank, dkey = self._curve_meta(name)
        if dkey is None:
            return name
        short = self._short_name(dkey)              # 'num: basename'
        if len(short) > 24:
            short = short[:23] + '…'
        if rank == 1:
            return '%s (fit)' % short
        if rank == 2:
            return '%s (joint)' % short
        return short

    def _indiv_fit_name(self, fnum=None, funcname=None):
        """Key for an INDIVIDUAL-view fit curve, carrying the dataset index so it sorts
        and labels correctly (``<model> [indiv:<fnum>]``). Falls back to the bare model
        name when no dataset index is available (e.g. a pure simulation)."""
        if funcname is None:
            it = self.funcListWidget.currentItem()
            funcname = it.text() if it is not None else 'fit'
        if fnum is None and getattr(self, 'sfnames', None):
            try:
                fnum = self.sfnames[-1].split('<>')[0]
            except Exception:
                fnum = None
        return '%s [indiv:%s]' % (funcname, fnum) if fnum is not None else funcname

    # ------------------------------------------------- per-column fit limits --
    # The extra columns are ALREADY retained: Data_Dialog.readData stores the whole
    # DataFrame at self.dlg_data[data_key]['data'] and the plotted column indices at
    # self.plotColIndex[data_key][sub_key]. So limits on any column need no new data
    # plumbing -- only a way to name the columns and build a mask from them.
    #
    # ALIGNMENT INVARIANT: externalData[key]['x'] is a straight copy of a DataFrame
    # column, so every column is row-aligned 1:1 with every sub_key's x. The mask is
    # therefore per-DATASET, not per-sub_key, which is why limits are keyed
    # (data_key, column).
    def available_columns(self, data_key):
        """[(col_name, label)] for the limits table. Row 0 is always 'x'."""
        try:
            cols = list(self.dlg_data[data_key]['data'].columns)
        except Exception:
            return [('x', 'x')]
        xname = None
        try:
            sub = list(self.plotColIndex[data_key].keys())[0]
            xname = cols[self.plotColIndex[data_key][sub][0]]
        except Exception:
            pass
        out = [('x', 'x  [%s]' % xname if xname else 'x')]
        out += [(c, str(c)) for c in cols if c != xname]
        return out

    def _column_values(self, data_key, col):
        """1-D float array for a named column, or None."""
        if col == 'x':
            try:
                sub = list(self.data[data_key].keys())[0]
                return np.asarray(self.data[data_key][sub]['x'], float)
            except Exception:
                return None
        try:
            return np.asarray(self.dlg_data[data_key]['data'][col].values, float)
        except Exception:
            return None

    def _column_mask(self, data_key, limits=None, slice_=None):
        """Boolean eligibility mask for a dataset, or None if nothing is limited.

        Only NON-'x' columns contribute: the 'x' window is already applied as the
        contiguous imin:imax slice, and doubling it here would just be slower.
        ``slice_=(i0, i1)`` returns the mask pre-sliced to match arrays that the
        caller has already sliced (the update_plot path does that).
        """
        lim = self.limits.get(data_key, {}) if limits is None else (limits or {})
        cols = [(c, v) for (c, v) in lim.items()
                if c != 'x' and v is not None and (v[0] is not None or v[1] is not None)]
        if not cols:
            return None
        ref = self._column_values(data_key, 'x')
        n = None if ref is None else len(ref)
        m = None
        for c, (lo, hi) in cols:
            v = self._column_values(data_key, c)
            if v is None:
                continue
            if n is not None and len(v) != n:
                # Violated alignment invariant: refuse rather than mask the wrong
                # rows. Loudly, because a silent misalignment is unrecoverable.
                print('XModFit: column %r of %s has %d rows but x has %d -- '
                      'column limits ignored for this dataset' % (c, data_key, len(v), n))
                return None
            k = np.ones(len(v), bool)
            if lo is not None:
                k &= v >= lo
            if hi is not None:
                k &= v <= hi
            m = k if m is None else (m & k)
        if m is None:
            return None
        if slice_ is not None:
            i0, i1 = slice_
            m = m[i0:i1]
        return m

    # ---- persistence of column limits: "qxy:0.9:1.7;qz::0.8" ----------------
    # A separate optional line rather than an extension of '#Fit Range=', so an
    # older XModFit reading a newer file just ignores it (the loader is an
    # if/elif chain of substring tests) and a newer XModFit reading an older file
    # simply finds no extra limits. Backward compatible in both directions.
    def _format_column_limits(self, data_key=None):
        key = data_key or (self.sfnames[-1] if self.sfnames else None)
        lim = (self.limits.get(key, {}) or {}) if key else {}
        recs = []
        for c in sorted(k for k in lim if k != 'x'):
            lo, hi = lim[c]
            recs.append('%s:%s:%s' % (c, '' if lo is None else repr(float(lo)),
                                      '' if hi is None else repr(float(hi))))
        return ';'.join(recs)

    @staticmethod
    def _parse_column_limits(text):
        out = {}
        for rec in str(text).strip().split(';'):
            if not rec.strip():
                continue
            parts = rec.split(':')
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            def _f(t):
                t = t.strip()
                if t == '':
                    return None
                try:
                    return float(t)
                except ValueError:
                    return None
            lo, hi = _f(parts[1]), _f(parts[2])
            if name and name != 'x' and not (lo is None and hi is None):
                out[name] = (lo, hi)
        return out

    def _apply_column_limits(self, text, data_key=None):
        d = self._parse_column_limits(text)
        keys = [data_key] if data_key else list(self.sfnames or [])
        for k in keys:
            if k is None:
                continue
            cur = self.limits.setdefault(k, {})
            for c in [c for c in cur if c != 'x']:
                cur.pop(c)
            cur.update(d)
        self._refresh_limits_badge()
        return d

    def set_column_limit(self, col, which, value):
        """Write one bound of one column for every SELECTED dataset.

        ``which`` is 0 for the low bound, 1 for the high bound. The 'x' column
        routes through xmin/xmax so the line edit stays authoritative for it.
        """
        for key in (self.sfnames or []):
            if col == 'x':
                (self.xmin if which == 0 else self.xmax)[key] = value
            else:
                d = self.limits.setdefault(key, {})
                pair = list(d.get(col, (None, None)))
                pair[which] = value
                if pair[0] is None and pair[1] is None:
                    d.pop(col, None)
                else:
                    d[col] = tuple(pair)
        if col == 'x':
            try:
                lo = np.min([self.xmin[k] for k in self.sfnames])
                hi = np.max([self.xmax[k] for k in self.sfnames])
                self.xminmaxLineEdit.setText('%0.6f:%0.6f' % (lo, hi))
            except Exception:
                pass
        self._refresh_limits_badge()
        self.update_plot()

    def _refresh_limits_badge(self):
        """Show the number of extra column limits on the button, red when a limit
        has emptied the dataset -- so '0 active' is visible BEFORE hitting Fit."""
        if not hasattr(self, 'limitsButton'):
            return
        n, empty = 0, False
        for key in (self.sfnames or []):
            lim = self.limits.get(key, {}) or {}
            n = max(n, len([c for c in lim if c != 'x']))
            m = self._column_mask(key)
            if m is not None and not np.any(m):
                empty = True
        self.limitsButton.setText('Limits' if n == 0 else 'Limits (%d)' % n)
        self.limitsButton.setStyleSheet('color: red' if empty else '')
        self.limitsButton.setToolTip(
            'Fit limits on any other data column (blank = unbounded)'
            + ('\nWARNING: current limits leave 0 points to fit' if empty else ''))

    def openColumnLimitsDialog(self):
        if not (self.sfnames or []):
            QMessageBox.warning(self, 'No data',
                                'Select a dataset first: column limits are '
                                'per-dataset.', QMessageBox.Ok)
            return
        if getattr(self, 'columnLimitsDialog', None) is None:
            self.columnLimitsDialog = ColumnLimitsDialog(self, [])
        self.columnLimitsDialog.refresh()
        self.columnLimitsDialog.show()
        self.columnLimitsDialog.raise_()

    def _joint_masks(self, spec):
        """{dataset_name: bool mask or None} for spec.compile().

        The spec cannot build these itself -- it only receives data[name][sub] and
        has no access to dlg_data -- so they are built here, where the DataFrames
        live. Per-dataset limits set in the joint panel win over the main-window
        ones for that dataset.
        """
        out = {}
        try:
            for e in spec.datasets:
                lim = e.limits if getattr(e, 'limits', None) else self.limits.get(e.name)
                out[e.name] = self._column_mask(e.name, limits=lim)
        except Exception:
            return {}
        return out

    def _autoselect_image_params(self):
        """Select the 2D-map entries when nothing else is selected.

        The 2D Map tab is driven by the genParamListWidget SELECTION, so without
        this a model that emits image output shows no tab at all until the user
        thinks to hunt for 'map2d_data' in the generated-parameter list -- which
        nothing on screen tells them to do. A map is the natural default view for
        a model that produces one, so select them once; the user can deselect.
        Never overrides an existing selection.
        """
        try:
            if getattr(self, '_image_autoselected', False):
                return          # already offered once; a deselect must stick
            if self.genParamListWidget.selectedItems():
                return
            op = self.fit.params.get('output_params', {}) or {}
            rows = []
            for i in range(self.genParamListWidget.count()):
                it = self.genParamListWidget.item(i)
                key = it.text().split(':')[0].strip()
                if self._op_is_image(op.get(key)):
                    rows.append(it)
            if not rows:
                return
            for it in rows:
                it.setSelected(True)
            self.gen_param_items = [it.text() for it in rows]
            self._image_autoselected = True
        except Exception:
            pass

    @staticmethod
    def _op_is_image(entry):
        """True for a 2-D map entry of output_params.

        Discriminate on plotType, NEVER on the displayed list text: an image entry
        renders as "['x', 'y', 'z']", textually identical to the legacy meshgrid
        form that saveGenParameters branches on.
        """
        return isinstance(entry, dict) and entry.get('plotType') == 'image'

    @staticmethod
    def _op_var_list(entry):
        """Plottable variable names of an output_params entry (metadata removed)."""
        try:
            return [k for k in entry.keys() if k not in OP_META_KEYS]
        except Exception:
            return []

    def _legend_order(self, names):
        """Order plotted-curve names for a tidy legend: by dataset INDEX (the number
        shown in the Data boxes), and within a dataset  data < individual fit < joint
        fit. Names that don't match the conventions keep a stable order at the end."""
        def key(name):
            idx, rank, _ = self._curve_meta(name)
            return (idx if idx is not None else 10**9, rank, name)
        return sorted(names, key=key)

    def _set_detail_dataset(self, name):
        """Switch the DETAIL dataset (whose fit fills the parameter tables) WITHIN a
        multi-selection, without collapsing the selection (so all curves/EDPs stay
        co-plotted). If ``name`` has no individual fit, leave the tables unchanged and
        show the no-fit indicator instead (request: don't update params for unfit data).
        Called from DataListWidget on a plain click inside a multi-selection."""
        fit = self.dataset_fits.get(name)
        fitted = (fit is not None and not hasattr(fit, 'objectives')
                  and (getattr(fit, '_is_fitted', False) or getattr(fit, 'result', None) is not None))
        if not fitted:
            showing = self.sfnames[-1] if getattr(self, 'sfnames', None) else None
            self._show_nofit_indicator(name, showing)
            return
        self._clear_nofit_indicator()
        try:
            sel = [it.text() for it in self.dataListWidget.selectedItems()]
        except Exception:
            sel = [name]
        # make the clicked dataset the detail (sfnames[-1] drives the detail elsewhere)
        self.sfnames = [n for n in sel if n != name] + [name]
        self._switching_dataset = True
        try:
            self._sync_funclist_to_fit(fit)
            self._activate_dataset_fit(fit)          # self.fit + param tables -> this dataset
        finally:
            self._switching_dataset = False
        self._update_ci_button_state()
        self._refresh_data_labels()

    def _curve_color_for(self, data_key, curve_key):
        """Deterministic, well-separated color for a (dataset, curve). A dataset's data
        and its fit share this color; distinct curves get distinct palette colors,
        assigned in first-seen order."""
        k = (data_key, curve_key)
        if k not in self._curve_color:
            self._curve_color[k] = self._curve_palette[self._curve_color_ct % len(self._curve_palette)]
            self._curve_color_ct += 1
        return self._curve_color[k]

    def _single_fit_color(self, key=None):
        """Color for the single-fit fit curve = the color of its data curve (so fit
        matches data). Returns None when no dataset is selected (palette fallback)."""
        sf = getattr(self, 'sfnames', None)
        if not sf:
            return None
        if key is None:
            try:
                key = list(self.data[sf[-1]].keys())[0]
            except Exception:
                return None
        return self._curve_color_for(sf[-1], key)

    def _refresh_data_labels(self):
        """Mark Data-list items that carry a fitted individual result (bold + green +
        tooltip) and, in the tooltip, note joint-fit membership. Never changes
        item.text() -- that string is the dataset key parsed throughout the app."""
        try:
            from PyQt5.QtGui import QColor, QBrush
        except Exception:
            return
        try:
            joint_names = set(self.jointFitPanel.spec.dataset_names)
        except Exception:
            joint_names = set()
        for i in range(self.dataListWidget.count()):
            it = self.dataListWidget.item(i)
            txt = it.text()
            stored = self.dataset_fits.get(txt)
            has_fit = stored is not None and (getattr(stored, '_is_fitted', False)
                                              or getattr(stored, 'result', None) is not None)
            _cov = getattr(self, 'covariances', {}).get(txt)
            f = it.font()
            f.setBold(bool(has_fit) or _cov is not None)   # covariance marker: bold + tooltip
            it.setFont(f)
            it.setForeground(QBrush(QColor('#1a7f37')) if has_fit else QBrush())
            tips = []
            if has_fit:
                tips.append('has individual fit')
            if txt in joint_names:
                tips.append('in joint fit')
            if _cov is not None:
                tips.append('covariance: %s' % os.path.basename(_cov.source or '<in memory>'))
            it.setToolTip('; '.join(tips))

    def openDataDialog(self,item):
        try:
            self.dataListWidget.itemSelectionChanged.disconnect()
        except:
            pass
        if len(self.dataListWidget.selectedItems())>1:
            for titem in self.dataListWidget.selectedItems():
                if titem.text()!=item.text():
                    titem.setSelected(False)
        tfname = item.text()
        fnum, fname = tfname.split('<>')
        data_dlg=Data_Dialog(data=copy.copy(self.dlg_data[tfname]),parent=self,expressions=copy.copy(self.expressions[tfname]),
                             plotIndex=copy.copy(self.plotColIndex[tfname]),colors=copy.copy(self.plotColors[tfname]))
        data_dlg.setModal(True)
        data_dlg.closePushButton.setText('Cancel')
        data_dlg.tabWidget.setCurrentIndex(1)
        data_dlg.dataFileLineEdit.setText(fname)
        if data_dlg.exec_():
            self.plotWidget.remove_data(datanames=self.pfnames)
            newFname=data_dlg.dataFileLineEdit.text()
            if fname==newFname:
                self.plotColIndex[tfname]=data_dlg.plotColIndex
                self.plotColors[tfname]=data_dlg.plotColors
                self.dlg_data[tfname]=copy.copy(data_dlg.data)
                self.data[tfname]=copy.copy(data_dlg.externalData)
                self.expressions[tfname]=copy.copy(data_dlg.expressions)
                for key in self.data[tfname].keys():
                    self.plotWidget.add_data(self.data[tfname][key]['x'],self.data[tfname][key]['y'],
                                             yerr=self.data[tfname][key]['yerr'],name='%s:%s'%(fnum,key),color=self.plotColors[tfname][key])
                self.xmin[tfname] = np.min(
                    [np.min(self.data[tfname][k1]['x']) for k1 in self.data[tfname].keys()])
                self.xmax[tfname] = np.max(
                    [np.max(self.data[tfname][k1]['x']) for k1 in self.data[tfname].keys()])
            else:
                text = '%s<>%s' % (fnum, newFname)
                self.data[text] = self.data.pop(tfname)
                self.dlg_data[text] = self.dlg_data.pop(tfname)
                item.setText(text)
                self.dlg_data[text]=copy.copy(data_dlg.data)
                self.data[text]=copy.copy(data_dlg.externalData)
                self.plotColIndex[text]=copy.copy(data_dlg.plotColIndex)
                self.plotColors[text]=copy.copy(data_dlg.plotColors)
                self.expressions[text]=copy.copy(data_dlg.expressions)
                for key in self.data[text].keys():
                    self.plotWidget.add_data(self.data[text][key]['x'], self.data[text][key]['y'], yerr=self.data[text][key]['yerr'],
                                             name='%s:%s'%(fnum,key),color=self.plotColors[text][key])
                self.xmin[text] = np.min(
                    [np.min(self.data[text][k1]['x']) for k1 in self.data[text].keys()])
                self.xmax[text] = np.max(
                    [np.max(self.data[text][k1]['x']) for k1 in self.data[text].keys()])
        self.dataFileSelectionChanged()
        self.dataListWidget.itemSelectionChanged.connect(self.dataFileSelectionChanged)
        self.consoleWidget.push_vars({'data': self.dlg_data, 'plotColIndex':self.plotColIndex, 'plotColors':self.plotColors, 'expressions':self.expressions})

    def updateData(self):
        fnames=[self.dataListWidget.item(i).text() for i in range(self.dataListWidget.count())]
        self.plotWidget.remove_data(datanames=self.pfnames)
        progressDlg=QProgressDialog('Updating all the data. Please wait...','Cancel',-1,len(fnames)-1)
        progressDlg.setAutoClose(True)
        progressDlg.forceShow()
        for num, fname in enumerate(fnames):
            if progressDlg.wasCanceled():
                progressDlg.setValue(len(fnames)-1)
                progressDlg.close()
                break
            fnum,_= fname.split('<>')
            self.dlg_data[fname]['data'].reset_index(inplace = True, drop = True)
            data_dlg = Data_Dialog(data=copy.copy(self.dlg_data[fname]), parent=self,
                                   expressions=copy.copy(self.expressions[fname]),
                                   plotIndex=copy.copy(self.plotColIndex[fname]),
                                   colors=copy.copy(self.plotColors[fname]))
            data_dlg.closePushButton.setText('Cancel')
            data_dlg.setModal(True)
            data_dlg.accept()
            self.plotColIndex[fname] = data_dlg.plotColIndex
            self.plotColors[fname] = data_dlg.plotColors
            self.dlg_data[fname] = copy.copy(data_dlg.data)
            self.data[fname] = copy.copy(data_dlg.externalData)
            self.expressions[fname] = copy.copy(data_dlg.expressions)
            for key in self.data[fname].keys():
                self.plotWidget.add_data(self.data[fname][key]['x'], self.data[fname][key]['y'], \
                                         yerr=self.data[fname][key]['yerr'], name='%s:%s' % (fnum, key),
                                         color=self.data[fname][key]['color'])
            progressDlg.setValue(num)

        self.dataFileSelectionChanged()

    def xminmaxChanged(self):
        try:
            xmin,xmax=self.xminmaxLineEdit.text().split(':')
        except:
            QMessageBox.warning(self, "Value Error", "Please supply the Xrange in this format:\n xmin:xmax",
                                QMessageBox.Ok)
            return
        for item in self.dataListWidget.selectedItems():
            fname=item.text()
            self.xmin[fname], self.xmax[fname]=float(xmin), float(xmax)
        self.update_plot()

    def doFit(self, fit_method=None, emcee_walkers=100, emcee_steps=100,
                       emcee_cores=1, reuse_sampler=False, emcee_burn=0, emcee_thin=1, backend=None):
        from joint_fit_orchestration import build_jointfit_from_selection
        self.fchanged=False
        self.tchisqr=1e30
        # self.xminmaxChanged()
        # The dedicated joint MCMC window drives the joint objective explicitly and must
        # bypass the individual-fit guards below (there is no main-window dataset/function
        # selection while it is open). Route it first; run_joint_emcee has its own guards.
        _method = fit_method if fit_method is not None else self.fitMethods[self.fitMethodComboBox.currentText()]
        if _method == 'emcee' and getattr(self, '_mcmc_is_joint', False):
            self.fit_method = 'emcee'
            self.fit_scale = self.fitScaleComboBox.currentText()
            self.run_joint_emcee(emcee_walkers=emcee_walkers, emcee_steps=emcee_steps,
                                 emcee_cores=emcee_cores, reuse_sampler=reuse_sampler,
                                 emcee_burn=emcee_burn, emcee_thin=emcee_thin, backend=backend)
            return
        if self.sfnames is None or self.sfnames==[]:
            QMessageBox.warning(self,'Data Error','Please select a dataset first before fitting',QMessageBox.Ok)
            return
        try:
            if len(self.fit.fit_params)>0:
                pass
            else:
                QMessageBox.warning(self, 'Fit Warning', 'Please select atleast a single parameter to fit', QMessageBox.Ok)
                return
        except:
            QMessageBox.warning(self, 'Fit Function Warning', 'Please select a function to fit', QMessageBox.Ok)
            return
        if len(self.funcListWidget.selectedItems())==0:
            QMessageBox.warning(self, 'Function Error',
                                'Please select a function first to fit.\n' + traceback.format_exc(), QMessageBox.Ok)
            return
        # try:
        #     self.fixedParamTableWidget.cellChanged.disconnect(self.fixedParamChanged)
        #     self.sfitParamTableWidget.cellChanged.disconnect(self.sfitParamChanged)
        #     self.mfitParamTableWidget.cellChanged.disconnect(self.mfitParamChanged)
        # except:
        #     QMessageBox.warning(self,'Function Error','Please select a function first to fit.\n'+traceback.format_exc(),QMessageBox.Ok)
        #     return
        if fit_method is None:
            self.fit_method=self.fitMethods[self.fitMethodComboBox.currentText()]
        else:
            self.fit_method=fit_method
        if self.fit_method not in ['leastsq','brute','differential_evolution','least_squares','emcee','nelder']:
            QMessageBox.warning(self,'Fit Method Warning','This method is under development and will be available '
                                                          'later.', QMessageBox.Ok)
            return
        self.fit_scale=self.fitScaleComboBox.currentText()
        # Dispatch to the joint fit only when the JOINT view is active (a dataset is
        # selected in the joint panel's list). With mutual-exclusion selection, choosing
        # an individual dataset clears the joint selection, so the Fit button then does an
        # individual fit even while joint mode is enabled.
        _jp = getattr(self, 'jointFitPanel', None)
        joint_active = (getattr(self, 'jointFitEnabled', False) and _jp is not None
                        and len(_jp.datasetList.selectedItems()) > 0
                        and len(_jp.spec.datasets) >= 2)
        # (The dedicated joint MCMC window is already routed at the top of doFit,
        # before the individual-fit guards.)
        if joint_active:
            if self.fit_method == 'emcee':
                # MCMC confidence intervals on the joint objective: reuse the already
                # fitted JointFit (do NOT recompile/re-fit), sample the global params.
                self.run_joint_emcee(emcee_walkers=emcee_walkers, emcee_steps=emcee_steps,
                                     emcee_cores=emcee_cores, reuse_sampler=reuse_sampler,
                                     emcee_burn=emcee_burn, emcee_thin=emcee_thin, backend=backend)
            else:
                self.run_joint_fit()
            return
        self.disconnect_fit_function()

        selItems=self.dataListWidget.selectedItems()
        try:
            self.dataListWidget.itemSelectionChanged.disconnect()
        except:
            pass

        """if getattr(self, 'jointFitEnabled', False) and len(selItems) > 1:
            fnames = [it.text() for it in selItems]
            self.fit = build_jointfit_from_selection(
                self.data, fnames, self.funcClass,
                cov_paths=getattr(self, '_cov_path', {}),  # {fname: sidecar .npz or *_c_factors.txt}
                raw_paths=getattr(self, '_raw_path', {}),  # {fname: original 4-col file} for *_c_factors.txt
                shared=getattr(self, 'jointSharedParams', None),  # [] Axis-2, ['theta','phi'] Axis-1
            )
            self.fit.functionCalled.connect(self.fitCallback)
            self.fit_report, self.fit_message = self.fit.perform_fit(
                self.xmin, self.xmax, fit_scale=self.fit_scale, fit_method=self.fit_method)
            # then repopulate the per-dataset tables from self.fit.master_params
            # (see joint_fit_orchestration.push_result_to_tables)
            return"""

        for item in selItems:
            item.setSelected(False)
        self.dataListWidget.itemSelectionChanged.connect(self.dataFileSelectionChanged)

        if len(selItems)>1:
            if self.fit_method=='emcee':
                QMessageBox.warning(self, 'MultiFit Warning', 'Cannot perform Errorbar estimation on more than one data. '
                                                             'Please select only one dataset at a time.',QMessageBox.Ok)
                return
            else:
                self.fitProgressDlg=QProgressDialog('Fitting multiple data...','Abort Fit',-1,len(selItems)-1,parent=self)
                self.fitProgressDlg.setModal(True)
                self.fitProgressDlg.forceShow()

        for itemNum, item in enumerate(selItems):
            if len(selItems)>1:
                if self.fitProgressDlg.wasCanceled():
                    self.fitProgressDlg.setValue(len(selItems)-1)
                    break
            self.disconnect_fit_function()
            item.setSelected(True)
            fname=item.text()
            # fit THIS dataset's own stored model/params (per-dataset memory);
            # lazily seeded from the current view if never selected/fit before.
            # _activate_dataset_fit resyncs the tables/mpar to it so the post-fit
            # readback + update_plot use matching keys (models may differ per dataset).
            _dsfit=self._fit_for_dataset(fname)
            if _dsfit is not None:
                self._activate_dataset_fit(_dsfit)
            if len(self.data[fname].keys())>1:
                x={}
                y={}
                yerr={}
                for key in self.data[fname].keys():
                    x[key]=self.data[fname][key]['x']
                    y[key]=self.data[fname][key]['y']
                    yerr[key]=self.data[fname][key]['yerr']
            else:
                key=list(self.data[fname].keys())[0]
                x=self.data[fname][key]['x']
                y=self.data[fname][key]['y']
                yerr=self.data[fname][key]['yerr']
                # if len(np.where(self.data[fname][key]['yerr']<1e-30)[0])>0:
                #     QMessageBox.warning(self,'Zero Errorbars','Some or all the errorbars of the selected data are zeros.\
                #      Please select None for the Errorbar column in the Plot options of the Data_Dialog',QMessageBox.Ok)
                #     break
            # if self.fitScaleComboBox.currentText()=='Log' and len(np.where(self.data[fname]['y']<1e-30)[0])>0:
            #     posval=np.argwhere(self.fit.y>0)
            #     self.fit.y=self.data[fname]['y'][posval].T[0]
            #     self.fit.x=self.data[fname]['x'][posval].T[0]
            #     self.fit.yerr=self.data[fname]['yerr'][posval].T[0]

            # full-length column mask: perform_fit slices it to the x window and
            # freezes it, so len(residual) is constant across iterations
            self._cov_set_x(self.fit,fname,x,y=y,yerr=yerr,mask=self._column_mask(fname),consume_notes=True)
            #self.update_plot()
            self.oldParams=copy.copy(self.fit.params)
            self.fit_stopped=False
            if self.fit.params['__mpar__']!={}:
                self.oldmpar=copy.deepcopy(self.mfitParamData)
            try:
                if self.fit_method != 'emcee':
                    self.fit.functionCalled.connect(self.fitCallback)
                else:
                    self.fit.functionCalled.connect(self.fitErrorCallback)
                self.fitData=[]
                self.showFitInfoDlg(emcee_walkers=emcee_walkers,emcee_steps=emcee_steps, emcee_burn = emcee_burn)
                self.runFit(emcee_walkers=emcee_walkers, emcee_steps=emcee_steps, emcee_burn=emcee_burn,
                            emcee_cores=emcee_cores, reuse_sampler=reuse_sampler, emcee_thin=emcee_thin,
                            backend=backend)
                if self.fit_stopped and self.fit_method != 'emcee' and getattr(self, 'temp_params', None) is not None:
                    # Restore the best params captured by fitCallback when a
                    # least-squares fit is aborted. Skip for emcee: fitErrorCallback
                    # never sets temp_params, so this would either crash (never set)
                    # or restore stale params from a prior least-squares fit.
                    self.fit.result.params = copy.copy(self.temp_params)
                #self.fit_report,self.fit_message=self.fit.perform_fit(self.xmin,self.xmax,fit_scale=self.fit_scale,\
                # fit_method=self.fit_method,callback=self.fitCallback)
                if self.fit_method != 'emcee':
                    self.final_fit_params=copy.copy(self.fit.result.params)
                self.fit._is_fitted=True   # mark this dataset "already fit"
                self.fit_info='Fit Message: %s\n'%self.fit_message

                self.closeFitInfoDlg()
                if self.fit_method != 'emcee':
                    self.errorAvailable=False
                    self.emcee_burn=0
                    self.emcee_steps=100000
                    self.emcee_frac=self.emcee_burn/self.emcee_steps
                    self.showConfIntervalButton.setDisabled(True)
                    self.disconnect_fit_function()
                    try:
                        self.sfitParamTableWidget.cellChanged.disconnect()
                        for i in range(self.mfitParamTabWidget.count()):
                            mkey = self.mfitParamTabWidget.tabText(i)
                            self.mfitParamTableWidget[mkey].cellChanged.disconnect()
                    except:
                        pass
                    for row in range(self.sfitParamTableWidget.rowCount()):
                        key=self.sfitParamTableWidget.item(row,0).text()
                        self.sfitParamTableWidget.item(row,1).setText(self.format%(self.fit.result.params[key].value))
                        try:
                            if self.fit.result.params[key].stderr is None:
                                self.fit.result.params[key].stderr = 0.0
                            self.sfitParamTableWidget.item(row, 1).setToolTip(
                                (key + ' = ' + self.format + ' \u00B1 ' + self.format) % \
                                (self.fit.result.params[key].value,
                                 self.fit.result.params[key].stderr))
                            self.meta[fname][key]=self.fit.result.params[key].value
                            self.meta[fname][key+'_stderr']=self.fit.result.params[key].stderr
                        except:
                            pass
                    self.sfitParamTableWidget.resizeRowsToContents()
                    self.sfitParamTableWidget.resizeColumnsToContents()
                    for i in range(self.mfitParamTabWidget.count()):
                        mkey=self.mfitParamTabWidget.tabText(i)
                        for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                            for col in range(1,self.mfitParamTableWidget[mkey].columnCount()):
                                parkey=self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                                key='__%s_%s_%03d'%(mkey,parkey,row)
                                self.mfitParamTableWidget[mkey].item(row,col).setText(self.format%(self.fit.result.params[key].value))
                                if self.fit.result.params[key].stderr is None:
                                    self.fit.result.params[key].stderr = 0.0
                                self.mfitParamTableWidget[mkey].item(row, col).setToolTip(
                                    (key + ' = ' + self.format + ' \u00B1 ' + self.format) % \
                                    (self.fit.result.params[key].value,
                                     self.fit.result.params[key].stderr))
                                self.meta[fname][key]=self.fit.result.params[key].value
                                self.meta[fname][key+'_stderr']=self.fit.result.params[key].stderr
                        self.mfitParamTableWidget[mkey].resizeRowsToContents()
                        self.mfitParamTableWidget[mkey].resizeColumnsToContents()
                    self.update_plot()
                    if len(selItems)==1:
                        fitResultDlg=FitResultDialog(fit_report=self.fit_report,fit_info=self.fit_info)
                        #ans=QMessageBox.question(self,'Accept fit results?',self.fit_report,QMessageBox.Yes, QMessageBox.No)
                        if fitResultDlg.exec_():
                            for i in range(self.mfitParamTabWidget.count()):
                                mkey=self.mfitParamTabWidget.tabText(i)
                                for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                                    for col in range(1, self.mfitParamTableWidget[mkey].columnCount()):
                                        parkey = self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                                        key = '__%s_%s_%03d' % (mkey, parkey, row)
                                        self.mfitParamData[mkey][parkey][row] = self.fit.result.params[key].value
                            ofname=os.path.splitext(fname.split('<>')[1])[0]
                            header='File saved on %s\n' % time.asctime()
                            header += 'Category: %s\n' % self.categoryListWidget.currentItem().text()
                            header += 'Function: %s\n' % self.funcListWidget.currentItem().text()
                            header += 'Xrange=%s\n' % self.xLineEdit.text()
                            header += 'Fit Range=%s\n' % self.xminmaxLineEdit.text()
                            _cl = self._format_column_limits()
                            if _cl:
                                header += 'Column Limits=%s\n' % _cl
                            for _cvl in self._covariance_header_lines(fname):
                                header += _cvl + '\n'
                            header += 'Fit Method=%s\n' % self.fitMethodComboBox.currentText()
                            header += 'Fit Scale=%s\n' % self.fitScaleComboBox.currentText()
                            header += 'Fit Iterations=%s\n' % self.fitIterationLineEdit.text()
                            header += 'Fixed Parameters:\n'
                            header += 'param\tvalue\n'
                            for row in range(self.fixedParamTableWidget.rowCount()):
                                txt = self.fixedParamTableWidget.item(row, 0).text()
                                if txt in self.fit.params['choices'].keys():
                                    header += txt + '\t' + self.fixedParamTableWidget.cellWidget(row, 1).currentText() + '\n'
                                elif txt in self.fit.params['filepaths'].keys():
                                    header += txt + '\t' + self.fixedParamTableWidget.cellWidget(row, 1).text().replace(
                                        '[Click] ', '') + '\n'
                                else:
                                    header += txt + '\t' + self.fixedParamTableWidget.item(row, 1).text() + '\n'
                            header += 'Single fitting parameters:\n'
                            header += 'param\tvalue\tfit\tmin\tmax\texpr\tbrute_step\n'
                            for row in range(self.sfitParamTableWidget.rowCount()):
                                parname = self.sfitParamTableWidget.item(row, 0).text()
                                par = self.sfitParamTableWidget.item(row, 1)
                                parval = par.text()
                                if par.checkState() == Qt.Checked:
                                    parfit = '1'
                                else:
                                    parfit = '0'
                                parmin = self.sfitParamTableWidget.item(row, 2).text()
                                parmax = self.sfitParamTableWidget.item(row, 3).text()
                                parexpr = self.sfitParamTableWidget.item(row, 4).text()
                                parbrute = self.sfitParamTableWidget.item(row, 5).text()
                                header += '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
                                parname, parval, parfit, parmin, parmax, parexpr, parbrute)
                            if self.fit.params['__mpar__'] != {}:
                                header += 'Multiple fitting parameters:\n'
                                header += 'param\tvalue\tfit\tmin\tmax\texpr\tbrute_step\n'
                                for i in range(self.mfitParamTabWidget.count()):
                                    mkey = self.mfitParamTabWidget.tabText(i)
                                    for col in range(self.mfitParamTableWidget[mkey].columnCount()):
                                        pkey = self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                                        if col != 0:
                                            for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                                                parname = '__%s_%s_%03d' % (mkey, pkey, row)
                                                par = self.mfitParamTableWidget[mkey].item(row, col)
                                                parval = par.text()
                                                if par.checkState() == Qt.Checked:
                                                    parfit = '1'
                                                else:
                                                    parfit = '0'
                                                parmin = str(self.fit.fit_params[parname].min)
                                                parmax = str(self.fit.fit_params[parname].max)
                                                parexpr = str(self.fit.fit_params[parname].expr)
                                                parbrute = str(self.fit.fit_params[parname].brute_step)
                                                header += '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
                                                parname, parval, parfit, parmin, parmax, parexpr, parbrute)
                                        else:
                                            for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                                                parname = '__%s_%s_%03d' % (mkey, pkey, row)
                                                par = self.mfitParamTableWidget[mkey].item(row, col)
                                                parval = par.text()
                                                header += '%s\t%s\n' % (parname, parval)
                            header+=self.fit_report+'\n'
                            if type(self.fit.x)==dict:
                                h1= 'col_names=['
                                h2= ''
                                for key in self.fit.x.keys():
                                    h1 += "'x_%s','y_%s','yerr_%s','yfit_%s'," % (key, key, key, key)
                                    h2 += 'x_%s y_%s yerr_%s yfit_%s ' % (key, key, key, key)
                                    try:
                                        fitdata=np.vstack((fitdata,self.fit.x[key][self.fit.imin[key]:self.fit.imax[key]+1],
                                                       self.fit.y[key][self.fit.imin[key]:self.fit.imax[key]+1],
                                                       self.fit.yerr[key][self.fit.imin[key]:self.fit.imax[key]+1],self.fit.yfit[key]))
                                    except:
                                        fitdata=np.vstack((self.fit.x[key][self.fit.imin[key]:self.fit.imax[key]+1],
                                                       self.fit.y[key][self.fit.imin[key]:self.fit.imax[key]+1],
                                                       self.fit.yerr[key][self.fit.imin[key]:self.fit.imax[key]+1],self.fit.yfit[key]))

                                header += h1[:-1]+']\n'
                                header += h2[:-1]
                                np.savetxt(ofname+'.xfit',fitdata.T,header=header,comments='#')
                            else:
                                header += "col_names=['x_data', 'y_data', 'yerr_data', 'yfit_data']\n"
                                header += 'x_data y_data yerr_data yfit_data'
                                fitdata = np.vstack((self.fit.x[self.fit.imin:self.fit.imax + 1],
                                                     self.fit.y[self.fit.imin:self.fit.imax + 1],
                                                     self.fit.yerr[self.fit.imin:self.fit.imax + 1],
                                                     self.fit.yfit)).T
                                np.savetxt(ofname + '.xfit', fitdata, header=header, comments='#')
                            self.saveParameters(fname=ofname+'.xpar')
                            self.calcConfInterButton.setEnabled(True)
                            self.update_plot()
                            if self.autoSaveGenParamCheckBox.isChecked():
                                self.saveGenParameters(
                                    bfname=os.path.join(os.path.dirname(ofname), 'genParam_' + os.path.basename(ofname)))
                            # self.xChanged()
                        else:
                            self.undoFit()
                            self.calcConfInterButton.setDisabled(True)
                        self.reuse_sampler=False
                    else:
                        for i in range(self.mfitParamTabWidget.count()):
                            mkey = self.mfitParamTabWidget.tabText(i)
                            for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                                for col in range(1, self.mfitParamTableWidget[mkey].columnCount()):
                                    parkey = self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                                    key = '__%s_%s_%03d' % (mkey, parkey, row)
                                    self.mfitParamData[mkey][parkey][row] = self.fit.result.params[key].value
                        ofname = os.path.splitext(fname.split('<>')[1])[0]
                        header = 'Data fitted with model: %s on %s\n' % (
                        self.funcListWidget.currentItem().text(), time.asctime())
                        header += 'Fixed Parameters\n'
                        header += '----------------\n'
                        for key in self.fit.params.keys():
                            if key not in self.fit.fit_params.keys() and key not in self.special_keys and key[
                                                                                                          :2] != '__':
                                header += key + '=' + str(self.fit.params[key]) + '\n'
                        header += self.fit_report + '\n'
                        header += "col_names=['x_data','y_data','yerr_data','yfit_data']\n"
                        header += 'x_data y_data yerr_data yfit_data\n'
                        if type(self.fit.x) == dict:
                            for key in self.fit.x.keys():
                                fitdata = np.vstack((self.fit.x[key][self.fit.imin[key]:self.fit.imax[key] + 1],
                                                     self.fit.y[key][self.fit.imin[key]:self.fit.imax[key] + 1],
                                                     self.fit.yerr[key][self.fit.imin[key]:self.fit.imax[key] + 1],
                                                     self.fit.yfit[key])).T
                                np.savetxt(ofname + '_' + key + '_fit.txt', fitdata, header=header, comments='#')
                        else:
                            fitdata = np.vstack((self.fit.x[self.fit.imin:self.fit.imax + 1],
                                                 self.fit.y[self.fit.imin:self.fit.imax + 1],
                                                 self.fit.yerr[self.fit.imin:self.fit.imax + 1],
                                                 self.fit.yfit)).T
                            np.savetxt(ofname + '.xfit', fitdata, header=header, comments='#')
                        self.saveParameters(fname=ofname + '.xpar')
                        self.calcConfInterButton.setEnabled(True)
                        self.update_plot()
                        if self.autoSaveGenParamCheckBox.isChecked():
                            self.saveGenParameters(
                                bfname=os.path.join(os.path.dirname(ofname), 'genParam_' + os.path.basename(ofname)))
                else:
                    self.errorAvailable = True
                    #self.reuse_sampler = True
                    self.emceeConfIntervalWidget.reuseSamplerCheckBox.setEnabled(True)
                    self.emceeConfIntervalWidget.reuseSamplerCheckBox.setCheckState(Qt.Checked)
                    self.disconnect_fit_function()
                    self.perform_post_sampling_tasks()
                    self.showConfIntervalButton.setEnabled(True)
            except:
                try:
                    self.closeFitInfoDlg()
                except:
                    pass
                QMessageBox.warning(self,'Minimization failed','Some of the parameters have got unreasonable values.\n'+
                                             traceback.format_exc(),QMessageBox.Ok)
                self.update_plot()
                break
            if len(selItems)>1:
                self.fitProgressDlg.setValue(itemNum)
                if itemNum<len(selItems)-1:
                    try:
                        self.dataListWidget.itemSelectionChanged.disconnect()
                    except:
                        pass
                    item.setSelected(False)
                    self.dataListWidget.itemSelectionChanged.connect(self.dataFileSelectionChanged)
        try:
            self.metaDataInfoUpdated(fname)
        except:
            pass
        self.sfitParamTableWidget.cellChanged.connect(self.sfitParamChanged)
        for i in range(self.mfitParamTabWidget.count()):
            mkey=self.mfitParamTabWidget.tabText(i)
            self.mfitParamTableWidget[mkey].cellChanged.connect(self.mfitParamChanged_new)
        self.disconnect_fit_function()
        # Restore the original multi-selection. Sequential fitting deselects each dataset
        # as it advances (so each fits/plots on its own), which otherwise leaves only the
        # LAST dataset selected. The user expects the multi-selection to persist after
        # fitting so all the freshly-fitted curves/EDPs stay co-plotted. Re-select them
        # all with signals suppressed, then refresh once.
        if len(selItems) > 1:
            try:
                self.dataListWidget.itemSelectionChanged.disconnect()
            except Exception:
                pass
            for it in selItems:
                it.setSelected(True)
            self.dataListWidget.itemSelectionChanged.connect(self.dataFileSelectionChanged)
            self.dataFileSelectionChanged()
        # Green "fitted" marker(s) should appear as soon as the fit finishes, not only
        # after the user reselects the dataset.
        self._refresh_data_labels()


    def _cov_set_x(self, fit, data_key, x, slice_=None, consume_notes=False, **kw):
        """``fit.set_x(x, **kw)`` with ``data_key``'s covariance (spec 17 6.2). The
        covariance belongs to the dataset: it is passed on every call (set_x without
        it clears it), only for single-curve x, restricted to ``slice_`` when the
        arrays are pre-sliced (update_plot), and the dataset's mode is set on the Fit
        first so residual() outside perform_fit uses it too. ``consume_notes`` hands
        the dataset's one-shot report notes to this fit. No covariance -> exactly
        ``fit.set_x(x, **kw)``."""
        cov = getattr(self, 'covariances', {}).get(data_key)
        if cov is None or type(x) == dict:
            fit.set_x(x, **kw)
            if consume_notes and data_key in getattr(self, '_cov_pending_notes', {}):
                # e.g. a covariance named in a reloaded .xpar that could not be reattached
                fit.pending_warnings = list(self._cov_pending_notes.pop(data_key))
            return
        import covariance_core as cc
        fit.covariance_mode = self.covariance_modes.get(data_key, 'model')
        try:
            if slice_ is not None:
                cov = cov.subset(np.arange(slice_[0], cov.n if slice_[1] is None else slice_[1]))
            fit.set_x(x, cov=cov, **kw)
        except (cc.CovarianceError, IndexError):
            # the data changed since the covariance was attached (fit-time row of
            # spec 6.5): keep it on the Fit unchecked, so the fit runs without it
            # and its report says why
            fit.set_x(x, **kw)
            fit.cov = cov
        if consume_notes:
            fit.pending_warnings = list(self._cov_pending_notes.pop(data_key, []))

    def disconnect_fit_function(self):
        try:
            self.fit.functionCalled.disconnect()
        except:
            pass

    def _update_ci_button_state(self):
        """Enable 'Calculate Param Error' when a fit is available. For an individual fit
        that means the CURRENT dataset is fitted; for a JointFit (joint fitting enabled)
        it means the joint fit has run -- joint-fit MCMC/CIs are now supported and do NOT
        require an individual dataset to be selected in the Data list."""
        fit = getattr(self, 'fit', None)
        fitted = (fit is not None and (getattr(fit, '_is_fitted', False)
                                       or getattr(fit, 'result', None) is not None))
        try:
            self.calcConfInterButton.setEnabled(bool(fitted))
        except Exception:
            pass

    def _ensure_ci_baseline(self):
        """Confidence intervals need an lmfit result (redchi/nfree/stderr). A .par-loaded
        fit has none, so run ONE leastsq at the loaded params to characterize the local
        minimum. Returns (ok, message); ok=False with a message when CIs are impossible."""
        fit = getattr(self, 'fit', None)
        if fit is None or hasattr(fit, 'objectives'):
            return False, 'No individual fit is active.'
        r = getattr(fit, 'result', None)
        if r is not None and getattr(r, 'redchi', None) is not None:
            return True, ''
        try:
            nvary = sum(1 for p in fit.fit_params.values() if p.vary)
        except Exception:
            nvary = 0
        if nvary == 0:
            return False, ('No fittable parameters: free at least one parameter (check its box '
                           'in the fit table) before calculating confidence intervals.')
        try:
            key = self.sfnames[-1]
            d = self.data[key]
            if len(d.keys()) > 1:
                x = {k: d[k]['x'] for k in d}; y = {k: d[k]['y'] for k in d}
                yerr = {k: d[k]['yerr'] for k in d}
            else:
                k0 = list(d.keys())[0]; x = d[k0]['x']; y = d[k0]['y']; yerr = d[k0]['yerr']
            self._cov_set_x(self.fit, key, x, y=y, yerr=yerr, mask=self._column_mask(key))
            self.fit_scale = self.fitScaleComboBox.currentText()
            self.fit.perform_fit(self.xmin[key], self.xmax[key], fit_scale=self.fit_scale,
                                 fit_method='leastsq', maxiter=1000)
        except Exception as e:
            return False, ('Could not run a baseline fit to characterize the minimum '
                           '(needed for confidence intervals):\n%s' % e)
        r = getattr(fit, 'result', None)
        if r is None or getattr(r, 'redchi', None) is None:
            return False, ('The fit could not characterize the minimum (singular covariance). '
                           'Adjust a bound or free a different parameter, then retry.')
        fit._is_fitted = True
        return True, ''

    def calcConfInterval(self):
        # This is the INDIVIDUAL-fit error button. Joint fits have their own dedicated
        # MCMC button in the Joint Fit panel, so when joint fit is enabled this button
        # refuses and points the user to the right control (keeps the two flows separate).
        if getattr(self, 'jointFitEnabled', False):
            QMessageBox.information(
                self, 'Joint fit enabled',
                'Joint fit is currently enabled, so this individual-fit error button '
                'is disabled.\n\n'
                '• To calculate errors for a single dataset, uncheck '
                '"Enable joint fit" in the Joint Fit panel first.\n\n'
                '• For joint-fit data, run "Run joint fit", then use the '
                '"Joint MCMC / Param Error…" button in the Joint Fit panel.',
                QMessageBox.Ok)
            return
        if len(self.sfnames) > 1:
            QMessageBox.warning(self,'MultiFit Warning','Confidence interval can be calculated for one data set only.'
                                               ' Please select and fit one data set before caclulating confidence interval.',
                                QMessageBox.Ok)
            return
        # A .par-loaded (or freshly restored) fit may have no lmfit result yet; establish a
        # baseline leastsq so the profiler has redchi/nfree/stderr (flag if impossible).
        if getattr(self.fit, 'result', None) is None:
            ok, msg = self._ensure_ci_baseline()
            if not ok:
                QMessageBox.warning(self, 'Confidence intervals', msg, QMessageBox.Ok)
                return
        if self.confIntervalMethodComboBox.currentText()=='ChiSqrDist':
            self.confInterval_ChiSqrDist()
        else:
            self.confInterval_emcee()


    def confInterval_ChiSqrDist(self):
        self.fit_method = self.fitMethods[self.fitMethodComboBox.currentText()]
        self.confIntervalWidget=QWidget()
        self.confIntervalWidget.setWindowModality(Qt.ApplicationModal)
        uic.loadUi('./UI_Forms/ConfInterval_ChiSqrDist.ui',self.confIntervalWidget)
        self.confIntervalWidget.setWindowTitle("ChiSqrDist Confidence Interval Calculator")
        self.chidata={}
        fitTableWidget = self.confIntervalWidget.fitParamTableWidget
        self.calcErrPushButtons={}
        self.errProgressBars={}
        self.plotErrPushButtons={}
        self.stopCalc=False
        for fpar in self.fit.result.params.keys():
            if self.fit.fit_params[fpar].vary:
                row = fitTableWidget.rowCount()
                fitTableWidget.insertRow(row)
                fitTableWidget.setCellWidget(row,0,QLabel(fpar))
                fitTableWidget.setItem(row,1,QTableWidgetItem(self.format%self.fit.result.params[fpar].value))
                if self.fit.result.params[fpar].stderr is not None and self.fit.result.params[fpar].stderr!=0.0:
                    errper=5*self.fit.result.params[fpar].stderr*100/self.fit.result.params[fpar].value
                    fitTableWidget.setItem(row,2,QTableWidgetItem('%.3f' % (errper)))
                    fitTableWidget.setItem(row,3,QTableWidgetItem('%.3f' % (errper)))
                else:
                    fitTableWidget.setItem(row, 2, QTableWidgetItem('%.3f' % 10))
                    fitTableWidget.setItem(row, 3, QTableWidgetItem('%.3f' % 10))
                fitTableWidget.setItem(row,4, QTableWidgetItem('20'))
                self.calcErrPushButtons[fpar]=QPushButton('Calculate')
                fitTableWidget.setCellWidget(row, 5, self.calcErrPushButtons[fpar])
                self.calcErrPushButtons[fpar].clicked.connect(partial(self.calcErrPushButtonClicked,row,fpar))
                self.errProgressBars[fpar]=QProgressBar()
                fitTableWidget.setCellWidget(row, 6, self.errProgressBars[fpar])
                self.confIntervalWidget.fitParamTableWidget.setItem(row, 7, QTableWidgetItem(''))
                self.confIntervalWidget.fitParamTableWidget.setItem(row, 8, QTableWidgetItem(''))
                self.plotErrPushButtons[fpar]=QPushButton('Plot')
                fitTableWidget.setCellWidget(row,9, self.plotErrPushButtons[fpar])
                self.plotErrPushButtons[fpar].clicked.connect(partial(self.plotErrPushButtonClicked,row,fpar))
        fitTableWidget.resizeColumnsToContents()
        self.confIntervalWidget.plotAllPushButton.clicked.connect(self.plotAllErrPushButtonClicked)
        self.confIntervalWidget.stopPushButton.clicked.connect(self.stopErrCalc)
        self.confIntervalWidget.calcAllPushButton.clicked.connect(self.calcAllErr)
        self.confIntervalWidget.saveAllPushButton.clicked.connect(self.saveAllErr)
        self.confIntervalWidget.confIntervalSpinBox.valueChanged.connect(self.setTargetChiSqr)
        self.confIntervalWidget.saveErrPushButton.clicked.connect(self.saveParIntervalErr)
        self.minimafitparameters = copy.copy(self.fit.result.params)
        self.confIntervalWidget.showMaximized()
        self.left_limit={}
        self.right_limit={}
        self.min_value={}
        self.calcAll=False

    def stopErrCalc(self):
        self.stopCalc=True

    def _ci_stat(self):
        """('chisqr'|'redchi', value) the CI profile uses for the last fit: chisqr when a
        covariance was active in it (absolute errors, spec 17 6.2), else today's redchi."""
        stat = getattr(self.fit, 'chi2_stat_for_ci', None)
        return stat() if stat is not None else ('redchi', self.fit.result.redchi)

    def setTargetChiSqr(self):
        self.confInterval = self.confIntervalWidget.confIntervalSpinBox.value()
        if self._ci_stat()[0] == 'chisqr':   # covariance: absolute chi2_min + dchi2(CL, 1 dof)
            self.minchisqr = self.fit.result.chisqr
            self.confIntervalWidget.minChiSqrLineEdit.setText(self.format % self.minchisqr)
            self.targetchisqr = self.minchisqr + chi2.ppf(self.confInterval * 0.01, 1)
            self.confIntervalWidget.targetChiSqrLineEdit.setText(self.format % self.targetchisqr)
            return
        self.minchisqr = self.fit.result.redchi
        self.confIntervalWidget.minChiSqrLineEdit.setText(self.format % self.minchisqr)
        self.targetchisqr = self.fit.result.redchi * chi2.isf((1.0 - self.confInterval * 0.01),
                                                              self.fit.result.nfree) / (self.fit.result.nfree)
        self.confIntervalWidget.targetChiSqrLineEdit.setText(self.format % self.targetchisqr)


    def calcAllErr(self):
        self.calcAll=True
        self.stopCalc=False
        for row in range(self.confIntervalWidget.fitParamTableWidget.rowCount()):
            if not self.stopCalc:
                fpar=self.confIntervalWidget.fitParamTableWidget.cellWidget(row,0).text()
                self.calcErrPushButtonClicked(row,fpar)
            else:
                return
        self.plotAllErrPushButtonClicked()
        self.errInfoTable = []
        for key in self.chidata.keys():
            if self.left_limit[key] is not None and self.right_limit[key] is not None:
                self.errInfoTable.append([key, self.min_value[key], self.left_limit[key] - self.min_value[key],
                         self.right_limit[key] - self.min_value[key]])
            elif self.left_limit[key] is None and self.right_limit[key] is not None:
                self.errInfoTable.append([key, self.min_value[key], None,
                                          self.right_limit[key] - self.min_value[key]])
            elif self.left_limit[key] is not None and self.right_limit is None:
                self.errInfoTable.append([key, self.min_value[key], self.left_limit[key] - self.min_value[key],
                                          None])
            else:
                self.errInfoTable.append([key, self.min_value[key], None, None])


        self.confIntervalWidget.errInfoTextEdit.clear()
        self.confIntervalWidget.errInfoTextEdit.setFont(QFont("Courier", 10))
        self.confIntervalWidget.errInfoTextEdit.append(tabulate(self.errInfoTable,
                                                                headers=["Parameter","Parameter-Value","Left-Error","Right-Error"],
                                                                stralign='left',numalign='left',tablefmt='simple'))
        self.calcAll=False


    def checkMinMaxErrLimits(self,fpar,vmin,vmax):
        self.fit.fit_params[fpar].vary=False
        xmin,xmax=self.xmin[self.sfnames[-1]],self.xmax[self.sfnames[-1]]
        for key in self.minimafitparameters:  # Putting back the minima parameters
            self.fit.fit_params[key].value = self.minimafitparameters[key].value
        self.fit.fit_params[fpar].value = vmin
        fit_report, mesg = self.fit.perform_fit(xmin, xmax, fit_scale=self.fit_scale,
                                                fit_method=self.fit_method,
                                                maxiter=int(self.fitIterationLineEdit.text()))
        if self._ci_stat()[1]>self.targetchisqr or self.fit.fit_params[fpar].min>vmin:
            left_limit_ok=True
        else:
            left_limit_ok=False
        for key in self.minimafitparameters:  # Putting back the minima parameters
            self.fit.fit_params[key].value = self.minimafitparameters[key].value
        self.fit.fit_params[fpar].value = vmax
        fit_report, mesg = self.fit.perform_fit(xmin, xmax, fit_scale=self.fit_scale,
                                                fit_method=self.fit_method,
                                                maxiter=int(self.fitIterationLineEdit.text()))
        if self._ci_stat()[1]>self.targetchisqr or self.fit.fit_params[fpar].max<vmax:
            right_limit_ok=True
        else:
            right_limit_ok=False
        self.fit.fit_params[fpar].vary=True
        return left_limit_ok, right_limit_ok


    def calcErrPushButtonClicked(self,row,fpar):
        xmin, xmax = self.xmin[self.sfnames[-1]], self.xmax[self.sfnames[-1]]
        self.stopCalc=False
        for key in self.minimafitparameters:
            self.fit.fit_params[key].value = self.minimafitparameters[key].value

        self.fit.fit_params[fpar].vary=False
        redchi_r=[]
        self.errProgressBars[fpar].setMinimum(0)
        Nval = int(self.confIntervalWidget.fitParamTableWidget.item(row, 4).text())
        self.errProgressBars[fpar].setMaximum(Nval)
        #Getting the chi-sqr value at the minima position keeping the value of fpar fixed at the minima position
        fit_report, mesg =self.fit.perform_fit(xmin, xmax, fit_scale=self.fit_scale, fit_method=self.fit_method,
                             maxiter=int(self.fitIterationLineEdit.text()))
        self.setTargetChiSqr()
        redchi_r.append([self.fit.fit_params[fpar].value, self._ci_stat()[1]])
        self.errProgressBars[fpar].setValue(1)
        value=self.fit.result.params[fpar].value
        vmax = value*(1.0+float(self.confIntervalWidget.fitParamTableWidget.item(row, 3).text())/100.0)
        vmin = value*(1.0-float(self.confIntervalWidget.fitParamTableWidget.item(row, 2).text())/100.0)

        left_limit_ok,right_limit_ok=self.checkMinMaxErrLimits(fpar,vmin,vmax)
        self.fit.fit_params[fpar].vary = False
        if left_limit_ok and right_limit_ok:
            # Fitting the right hand side of the minima starting from the first point after minima
            self.min_value[fpar]=value
            pvalues=np.linspace(value+(vmax-value)*2/Nval, vmax, int(Nval/2))
            i=1
            for parvalue in pvalues:
                if self.stopCalc:
                    for key in self.minimafitparameters:
                        self.fit.fit_params[key].value = self.minimafitparameters[key].value
                    return
                for key in self.minimafitparameters: # Putting back the minima parameters
                    self.fit.fit_params[key].value = self.minimafitparameters[key].value
                self.fit.fit_params[fpar].value=parvalue
                fit_report, mesg = self.fit.perform_fit(xmin, xmax, fit_scale=self.fit_scale,
                                                        fit_method=self.fit_method,
                                                        maxiter=int(self.fitIterationLineEdit.text()))
                if self.fit.result.success:
                    redchi_r.append([parvalue,self._ci_stat()[1]])
                i+=1
                self.errProgressBars[fpar].setValue(i)
                QApplication.processEvents()


            step=(value-vmin)*2/Nval
            redchi_l=[redchi_r[0]]

            #Fitting the left hand of the minima starting from the minima point
            pvalues=np.linspace(value-step, vmin, int(Nval / 2))
            for parvalue in pvalues:
                if self.stopCalc:
                    for key in self.minimafitparameters:
                        self.fit.fit_params[key].value = self.minimafitparameters[key].value
                    return
                for key in self.minimafitparameters: # Putting back the minima parameters
                    self.fit.fit_params[key].value = self.minimafitparameters[key].value
                self.fit.fit_params[fpar].value = parvalue
                fit_report, mesg = self.fit.perform_fit(xmin, xmax, fit_scale=self.fit_scale,
                                                        fit_method=self.fit_method,
                                                        maxiter=int(self.fitIterationLineEdit.text()))
                if self.fit.result.success:
                    redchi_l.append([parvalue, self._ci_stat()[1]])
                i+=1
                self.errProgressBars[fpar].setValue(i)
                QApplication.processEvents()

            chidata=np.array(redchi_r+redchi_l[1:])
            self.chidata[fpar]=chidata[chidata[:,0].argsort()]


            # Calculating the right-limit by interpolation
            rvalues = np.array(redchi_r)
            if self.targetchisqr < np.max(rvalues[:, 1]):
                fn=interp1d(rvalues[:, 1], rvalues[:, 0],kind='linear')
                self.right_limit[fpar] = fn(self.targetchisqr)
                self.confIntervalWidget.fitParamTableWidget.item(row, 8).setText(self.format % (self.right_limit[fpar]))
            else:
                self.right_limit[fpar] = None
                self.confIntervalWidget.fitParamTableWidget.item(row, 8).setText('None')

            # Calculating the left-limit by interpolation
            lvalues = np.array(redchi_l)
            if self.targetchisqr < np.max(lvalues[:, 1]):
                fn=interp1d(lvalues[:, 1], lvalues[:, 0],kind='linear')
                self.left_limit[fpar] = fn(self.targetchisqr)
                self.confIntervalWidget.fitParamTableWidget.item(row, 7).setText(self.format % (self.left_limit[fpar]))
            else:
                self.left_limit[fpar] = None
                self.confIntervalWidget.fitParamTableWidget.item(row, 7).setText('None')
            self.confIntervalWidget.fitParamTableWidget.resizeColumnsToContents()

            # Plotting the data
            if not self.calcAll:
                self.plotErrPushButtonClicked(row, fpar)

            #Showing the Errorbars
            self.errInfoTable = []
            key=fpar
            if self.left_limit[key] is not None and self.right_limit[key] is not None:
                self.errInfoTable.append([key, self.min_value[key], self.left_limit[key] - self.min_value[key],
                                          self.right_limit[key] - self.min_value[key]])
            elif self.left_limit[key] is None and self.right_limit[key] is not None:
                self.errInfoTable.append([key, self.min_value[key], None,
                                          self.right_limit[key] - self.min_value[key]])
            elif self.left_limit[key] is not None and self.right_limit is None:
                self.errInfoTable.append([key, self.min_value[key], self.left_limit[key] - self.min_value[key],
                                          None])
            else:
                self.errInfoTable.append([key, self.min_value[key], None, None])
            self.confIntervalWidget.errInfoTextEdit.clear()
            self.confIntervalWidget.errInfoTextEdit.setFont(QFont("Courier", 10))
            self.confIntervalWidget.errInfoTextEdit.append(tabulate(self.errInfoTable,
                                                                    headers=["Parameter", "Parameter-Value",
                                                                             "Left-Error", "Right-Error"],
                                                                    stralign='left', numalign='left',
                                                                    tablefmt='simple'))

        elif left_limit_ok:
            QMessageBox.warning(self,'Limit Warning','Max limit is not enough to reach the target chi-square for %s. Increase the Max limit'%fpar,QMessageBox.Ok)
            self.errProgressBars[fpar].setValue(0)
            QApplication.processEvents()
        else:
            QMessageBox.warning(self, 'Limit Warning', 'Min limit is not enough to reach the target chi-square for %s. Increase the Min limit'%fpar, QMessageBox.Ok)
            self.errProgressBars[fpar].setValue(0)
            QApplication.processEvents()

        # Going back to the minimum chi-sqr condition
        for key in self.minimafitparameters:
            self.fit.fit_params[key].value = self.minimafitparameters[key].value
        self.fit.fit_params[fpar].vary = True
        fit_report, mesg = self.fit.perform_fit(xmin, xmax, fit_scale=self.fit_scale,
                                                fit_method=self.fit_method,
                                                maxiter=int(self.fitIterationLineEdit.text()))

    def plotErrPushButtonClicked(self,row,fpar):
        if fpar in self.chidata.keys():
            mw=MplWidget()
            mw.setWindowModality(Qt.ApplicationModal)
            subplot=mw.getFigure().add_subplot(111)
            subplot.plot(self.chidata[fpar][:, 0], self.chidata[fpar][:, 1], 'r.')
            subplot.axhline(self.minchisqr,color='k',lw=1,ls='--')
            subplot.axhline(self.targetchisqr,color='k',lw=1,ls='-')
            subplot.axvline(self.min_value[fpar],color='b',lw=2,ls='-')
            # pl.text(self.min_value[fpar],1.01*self.minchisqr,self.format%self.min_value[fpar],rotation='vertical')
            if self.right_limit[fpar] is not None:
                subplot.axvline(self.right_limit[fpar],color='b',lw=1,ls='--')
                # pl.text(self.right_limit[fpar], 1.01*self.targetchisqr, self.format%self.right_limit[fpar],rotation='vertical')
                right_error = self.right_limit[fpar]-self.min_value[fpar]
            else:
                right_error='None'
            if self.left_limit[fpar] is not None:
                subplot.axvline(self.left_limit[fpar],color='b',lw=1,ls='--')
                # pl.text(self.left_limit[fpar], 1.01*self.targetchisqr, self.format% self.left_limit[fpar],rotation='vertical')
                left_error = self.left_limit[fpar]-self.min_value[fpar]
            else:
                left_error='None'
            subplot.set_title('%.3e$^{%.3e}_{%.3e}$'%(self.min_value[fpar], right_error, left_error))
            subplot.set_xlabel(fpar)
            subplot.set_ylabel('\u03c7$^2$')
            mw.getFigure().tight_layout()
            mw.draw()
            mw.show()
        else:
            QMessageBox.warning(self, 'Data error', 'No data available for plotting. Calculate first', QMessageBox.Ok)


    def plotAllErrPushButtonClicked(self):
        pkey=list(self.chidata.keys())
        Nplots=len(pkey)
        if Nplots>0:
            mw=MplWidget()
            mw.setWindowModality(Qt.ApplicationModal)
            rows=math.ceil(np.sqrt(Nplots))
            i=1
            for row in range(rows):
                for col in range(rows):
                    if i<=Nplots:
                        ax=mw.getFigure().add_subplot(rows,rows,i)
                        ax.plot(self.chidata[pkey[i-1]][:,0],self.chidata[pkey[i-1]][:,1],'r.')
                        ax.axhline(self.minchisqr, color='k', lw=1, ls='--')
                        ax.axhline(self.targetchisqr, color='k', lw=1, ls='-')
                        ax.axvline(self.min_value[pkey[i-1]], color='b', lw=2, ls='-')
                        # ax[row,col].text(self.min_value[pkey[i-1]], 1.01 * self.minchisqr, self.format % self.min_value[pkey[i-1]],rotation='vertical')
                        if self.right_limit[pkey[i-1]] is not None:
                            ax.axvline(self.right_limit[pkey[i-1]], color='b', lw=1, ls='--')
                            right_error=self.right_limit[pkey[i-1]]-self.min_value[pkey[i-1]]
                            # ax[row,col].text(self.right_limit[pkey[i-1]], 1.01*self.targetchisqr, self.format % self.right_limit[pkey[i-1]],rotation='vertical')
                        else:
                            right_error='None'
                        if self.left_limit[pkey[i-1]] is not None:
                            ax.axvline(self.left_limit[pkey[i-1]], color='b', lw=1, ls='--')
                            left_error=self.left_limit[pkey[i-1]]-self.min_value[pkey[i-1]]
                            # ax[row, col].text(self.left_limit[pkey[i-1]], 1.01*self.targetchisqr, self.format % self.left_limit[pkey[i-1]],rotation='vertical')
                        else:
                            left_error='None'
                        ax.set_title('%.3e$^{%.3e}_{%.3e}$'%(self.min_value[pkey[i-1]], right_error,left_error))
                        ax.set_xlabel(pkey[i-1])
                        ax.set_ylabel('\u03c7$^2$')
                    i+=1
            mw.getFigure().tight_layout()
            mw.draw()
            mw.show()

    def saveAllErr(self):
        fname=QFileDialog.getSaveFileName(self,'Provide prefix of the filename',directory=self.curDir,filter='Chi-Sqr files (*.chisqr)')[0]
        if fname!='':
            for key in self.chidata.keys():
                filename=os.path.splitext(fname)[0]+'_'+key+'.chisqr'
                header='Saved on %s\n'%(time.asctime())
                header="col_names=['%s','chi-sqr']\n"%key
                header+='%s\tchi-sqr'%key
                pl.savetxt(filename,self.chidata[key],header=header)

    def saveParIntervalErr(self):
        fname = QFileDialog.getSaveFileName(caption='Save Parameter Errors as', filter='Parameter Error files (*.perr)',
                                            directory=self.curDir)[0]
        if fname!='':
            fh=open(fname,'w')
            fh.write('# File saved on %s\n'%time.asctime())
            fh.write('# Error calculated using Chi-Sqr-Distribution Method\n')
            tlines=tabulate(self.errInfoTable, headers=["Parameter","Parameter-Value","Left-Error","Right-Error"],
                                   stralign='left',numalign='left',tablefmt='simple')
            lines=tlines.split('\n')
            for i,line in enumerate(lines):
                if i<2:
                    fh.write('#'+line+'\n')
                else:
                    fh.write(' '+line+'\n')
            fh.close()




        
    def _lockMainSelectionForMCMC(self, lock):
        """
        While the MCMC window is open, lock the main-window controls that would
        swap out self.fit / self.sfnames (the object the sampler is bound to), so
        the user can't lose track of which dataset/function is being analyzed.
        The prior enabled-state of each control is snapshotted on lock and restored
        on unlock (so controls that were legitimately disabled stay disabled).
        """
        # Only the controls that would rebind self.fit / self.sfnames. The CI buttons
        # are deliberately left out (doFit toggles showConfIntervalButton itself, and a
        # re-open is handled by the guard at the top of confInterval_emcee).
        widgets = [self.dataListWidget, self.funcListWidget, self.categoryListWidget,
                   self.addDataButton, self.removeDataButton, self.fitButton]
        if lock:
            self._mcmcLockedStates = {id(w): w.isEnabled() for w in widgets}
            for w in widgets:
                w.setEnabled(False)
        else:
            states = getattr(self, '_mcmcLockedStates', None)
            for w in widgets:
                w.setEnabled(states.get(id(w), True) if states is not None else True)
            self._mcmcLockedStates = None

    def eventFilter(self, obj, event):
        # Release the main-window selection lock when the MCMC window is closed.
        if obj is getattr(self, 'emceeConfIntervalWidget', None) and event.type() == QEvent.Close:
            self._lockMainSelectionForMCMC(False)
            # clear the joint flag so a later individual emcee never misroutes via doFit
            self._mcmc_is_joint = False
        return super(XModFit, self).eventFilter(obj, event)

    _PDB_MCMC_FUNCS = PDB_FUNCNAMES

    def _mcmc_funcname(self):
        """Function name driving the MCMC backend type + angle handling.

        Joint-aware and None-safe: while a joint fit is active the main
        ``funcListWidget`` has no current item, so the old
        ``funcListWidget.currentItem().text()`` calls crashed (leaving
        ``self.backend`` unset -> "create a new MCMC file"). For a joint fit use
        the shared model's class name (all datasets use one constrained model)."""
        if getattr(self, '_mcmc_is_joint', False):
            jf = getattr(self, 'fit', None)
            objs = getattr(jf, 'objectives', None)
            if objs:
                try:
                    return type(objs[0]['func']).__name__
                except Exception:
                    pass
        item = self.funcListWidget.currentItem()
        return item.text() if item is not None else ''

    def _mcmc_is_pdb(self):
        """True if the active MCMC function is an XLayersPDB* model (uses the
        angle-aware CustomHDFBackend + angleindex autocorr)."""
        return self._mcmc_funcname() in self._PDB_MCMC_FUNCS

    def _show_param_error(self):
        """'Show Param Error' router: open the joint MCMC window when the active fit
        is a joint fit (so joint results get their own visualizations), otherwise the
        individual window. A no-arg wrapper so QPushButton's `checked` bool never
        lands in confInterval_emcee's `joint` argument."""
        fit = getattr(self, 'fit', None)
        if hasattr(fit, 'objectives'):        # a JointFit
            self.confInterval_joint_emcee()
        else:
            self.confInterval_emcee()

    def confInterval_joint_emcee(self):
        """Open the dedicated joint-fit MCMC window (its own button in the joint
        panel). Requires a completed joint least-squares fit; makes that JointFit
        the active fit and opens the joint variant of the MCMC window."""
        jf = getattr(self, 'jointFit', None)
        if jf is None or not getattr(jf, '_is_fitted', False) or getattr(jf, 'result', None) is None:
            QMessageBox.warning(self, 'Joint MCMC',
                                'No completed joint fit yet.\n\n'
                                'Add at least 2 datasets in the Joint Fit panel and click '
                                '"Run joint fit" (or fit them with the main Fit button while '
                                'a joint dataset is selected). Then open this window to run '
                                'MCMC / parameter errors on the joint objective.', QMessageBox.Ok)
            return
        self.fit = jf                    # MCMC samples the joint objective
        # a joint MCMC is a fresh sampler unless this JointFit already carries a chain.
        # NB: lmfit MinimizerResult defines `flatchain` as a property that returns None
        # for a least-squares result, so `hasattr` is always True -- test the VALUE.
        self.errorAvailable = getattr(jf.result, 'flatchain', None) is not None
        self.confInterval_emcee(joint=True)

    def confInterval_emcee(self, joint=False):
        """Open the MCMC confidence-interval window.

        ``joint`` selects the dedicated joint-fit window: same base UI (so the
        shared sampling/post-processing plumbing is unchanged) plus extra
        joint-only visualization tabs appended by ``_augment_joint_mcmc_tabs``.
        The individual window (``joint=False``) is unchanged and gets no extra
        tabs, so joint visualizations never clutter individual analysis."""
        # If an MCMC window of the SAME kind is already open, just bring it to the
        # front. If a window of the OTHER kind is open (e.g. the individual window is
        # up and the user clicks the joint button), close it and build the requested
        # one instead of silently re-raising the wrong window.
        _existing = getattr(self, 'emceeConfIntervalWidget', None)
        if _existing is not None and _existing.isVisible():
            if bool(joint) == getattr(self, '_mcmc_is_joint', False):
                _existing.raise_()
                _existing.activateWindow()
                return
            try:
                _existing.close()          # different kind requested -> replace it
            except Exception:
                pass
        # _mcmc_is_joint is the authoritative signal for Start-routing (doFit) and the
        # joint-only visualization branches; set it now that we will build the window.
        self._mcmc_is_joint = bool(joint)
        self.fit_method = self.fitMethods[self.fitMethodComboBox.currentText()]
        if not self.errorAvailable:
             self.emcee_walkers=(self.fit.result.nvarys+1)*2
        else:
        #     # try:
             tnum=len(self.fit.result.flatchain[self.fit.result.var_names[0]])/self.emcee_walkers
             self.emcee_frac=self.emcee_burn/(tnum/(1.0-self.emcee_frac))
             emcee_burn=tnum*self.emcee_frac/(1.0-self.emcee_frac)
             self.emcee_burn=int(emcee_burn+self.emcee_steps*self.emcee_frac)
        self.emceeConfIntervalWidget = QWidget()
        # self.emceeConfIntervalWidget.setWindowModality(Qt.ApplicationModal)
        uic.loadUi('./UI_Forms/EMCEE_ConfInterval_Widget.ui', self.emceeConfIntervalWidget)
        # pop-out / zoom button on the corner tab (item 5, both windows)
        self._add_corner_popout_button()
        # Title carries the dataset + function being analyzed, so it stays unambiguous
        # even though the main window remains visible during sampling.
        try:
            _fn = self.funcListWidget.currentItem().text()
        except Exception:
            _fn = ''
        if self._mcmc_is_joint:
            try:
                _nd = len(self.jointFitPanel.spec.datasets)
            except Exception:
                _nd = len(getattr(getattr(self, 'fit', None), 'objectives', []) or [])
            self.emceeConfIntervalWidget.setWindowTitle(
                'Joint MCMC Confidence Interval — %d datasets [%s]' % (_nd, _fn))
            # append the joint-only visualization tabs (subset / summary); the
            # individual window never calls this, so its tabs stay uncluttered.
            self._augment_joint_mcmc_tabs()
        else:
            try:
                _ds = os.path.basename(self.sfnames[-1].split('<>')[1])
            except Exception:
                _ds = 'current dataset'
            self.emceeConfIntervalWidget.setWindowTitle(
                'MCMC Confidence Interval Calculator — %s [%s]' % (_ds, _fn))
        self.MCMCBackendFile='./mcmc_chain.h5'
        self.emceeConfIntervalWidget.MCMCFileLineEdit.setText(self.MCMCBackendFile)
        self.emceeConfIntervalWidget.loadMCMCFilePushButton.clicked.connect(self.openMCMCBackendFile)
        self.emceeConfIntervalWidget.startMCMCFilePushButton.clicked.connect(self.startMCMCBackendFile)
        self.emceeConfIntervalWidget.MCMCWalkerLineEdit.setText(str(self.emcee_walkers))
        self.emceeConfIntervalWidget.MCMCStepsLineEdit.setText(str(self.emcee_steps))
        self.emceeConfIntervalWidget.MCMCBurnLineEdit.setText(str(self.emcee_burn))
        self.emceeConfIntervalWidget.MCMCThinLineEdit.setText(str(self.emcee_thin))
        self.emceeConfIntervalWidget.ParallelCoresLineEdit.setText(str(self.emcee_cores))
        if not self.errorAvailable:
            self.emceeConfIntervalWidget.reuseSamplerCheckBox.setChecked(False)
            self.emceeConfIntervalWidget.reuseSamplerCheckBox.setDisabled(True)
            self.reuse_sampler=False
        else:
            self.emceeConfIntervalWidget.reuseSamplerCheckBox.setChecked(True)
            self.emceeConfIntervalWidget.reuseSamplerCheckBox.setDisabled(True)
            self.reuse_sampler=True

        if self.reuse_sampler:
            self.emceeConfIntervalWidget.reuseSamplerCheckBox.setEnabled(True)
            self.emceeConfIntervalWidget.reuseSamplerCheckBox.setCheckState(Qt.Checked)
        else:
            self.emceeConfIntervalWidget.reuseSamplerCheckBox.setCheckState(Qt.Unchecked)

        if not hasattr(self, 'MCMC_starting_step'):
            self.MCMC_starting_step = 0
            self.autoCorrTime = np.array([[0, 1]])

        self.emceeConfIntervalWidget.startSamplingPushButton.clicked.connect(self.start_emcee_sampling)
        # Connect Stop once per widget (like Start), not per sampling run. Connecting it
        # inside showFitInfoDlg stacked a new connection on every Start, so a later Stop
        # fired stopFit N times and the redundant disconnect() raised.
        self.emceeConfIntervalWidget.stopSamplingPushButton.clicked.connect(self.stopFit)
        self.emceeConfIntervalWidget.MCMCWalkerLineEdit.returnPressed.connect(self.MCMCWalker_changed)
        self.emceeConfIntervalWidget.saveConfIntervalPushButton.clicked.connect(self.saveParameterError)
        self.emceeConfIntervalWidget.addUserDefinedParamPushButton.clicked.connect(lambda x:
                                                                                   self.addMCMCUserDefinedParam(parname=None, expression=None))
        self.emceeConfIntervalWidget.removeUserDefinedParamPushButton.clicked.connect(self.removeMCMCUserDefinedParam)
        self.emceeConfIntervalWidget.saveUserDefinedParamPushButton.clicked.connect(self.saveMCMCUserDefinedParam)
        self.emceeConfIntervalWidget.loadUserDefinedParamPushButton.clicked.connect(self.loadMCMCUserDefinedParam)
        self.emceeConfIntervalWidget.userDefinedParamTreeWidget.itemDoubleClicked.connect(self.openMCMCUserDefinedParam)
        # Deferred autocorrelation: compute the ACF-vs-lag plot only when the user opens
        # the Autocorrelation tab (connected once per window here).
        self.emceeConfIntervalWidget.tabWidget.currentChanged.connect(self._on_mcmc_tab_changed)
        # XLayersPDB orientation-seeding controls, added programmatically into the free
        # grid rows just below MCMC Walkers(row2)/Steps(row3) so they sit with the other
        # MCMC settings. Only the orientation-seeding path reads them, so they are
        # disabled for non-PDB fits (and the seed-source combo only for a both-shared
        # joint fit).
        try:
            _grid = self.emceeConfIntervalWidget.gridLayout
            _is_pdb = self._mcmc_is_pdb()
            _joint = bool(getattr(self, '_mcmc_is_joint', False))
            _both_shared = False
            if _joint and _is_pdb:
                try:
                    _lay = self.fit.orientation_layout()
                    _both_shared = bool(_lay and _lay['both_shared'])
                except Exception:
                    _both_shared = False

            # --- Seed temperature (row 4, right under Walkers/Steps) ---
            _tlbl = QLabel('Seed temperature (T)')
            self.emceeConfIntervalWidget.seedTempSpinBox = QDoubleSpinBox()
            _sb = self.emceeConfIntervalWidget.seedTempSpinBox
            _sb.setDecimals(2); _sb.setMinimum(0.1); _sb.setMaximum(20.0)
            _sb.setSingleStep(0.5); _sb.setValue(float(getattr(self, 'seed_temperature', 1.0)))
            _ttip = ("Walker-seeding 'temperature' for XLayersPDB orientation seeding. It only "
                     "sets WHERE walkers START -- it does not change the final posterior.\n"
                     "  T = 1      default: seed near the best-fitting orientations (posterior-weighted).\n"
                     "  T = 2 to 5 spread walkers more widely across competing orientation modes\n"
                     "             (use if a run collapses into one mode or you suspect missed modes).\n"
                     "  T < 1      concentrate on the single best mode.\n"
                     "Every detected basin gets guaranteed walkers at any T, so secondary modes are\n"
                     "always seeded; T mainly controls how much of the walker cloud explores them.")
            if not _is_pdb:
                _ttip += "\n(Only applies to XLayersPDB fits -- disabled for this function.)"
            _sb.setEnabled(_is_pdb); _tlbl.setEnabled(_is_pdb)
            _tlbl.setToolTip(_ttip); _sb.setToolTip(_ttip)
            _grid.addWidget(_tlbl, 4, 0); _grid.addWidget(_sb, 4, 1)

            # --- Seed from: joint vs individual landscape (row 5; both-shared joint only) ---
            _slbl = QLabel('Seed from')
            self.emceeConfIntervalWidget.seedModeCombo = QComboBox()
            _cb = self.emceeConfIntervalWidget.seedModeCombo
            _cb.addItems(['Joint landscape', 'Individual landscapes'])
            _stip = ("How to build the shared-orientation seed map for a joint fit with theta & phi "
                     "shared:\n"
                     "  Joint landscape:      scan the shared (theta,phi) against ALL datasets at once,\n"
                     "                        re-profiling the shared parameters (e.g. insertion) jointly\n"
                     "                        at each orientation -- only orientations that survive the\n"
                     "                        shared constraint are seeded. Most faithful to the joint fit.\n"
                     "  Individual landscapes: scan each dataset's own model separately (shared params\n"
                     "                        pinned at the joint best fit) and sum the maps -- shows each\n"
                     "                        dataset's preferred orientations; pins (doesn't re-profile)\n"
                     "                        the shared params.")
            if not _both_shared:
                _stip += ("\n(Only applies to a joint fit with BOTH theta & phi shared; disabled here. "
                          "Per-dataset orientations are always seeded individually.)")
            _cb.setEnabled(_both_shared); _slbl.setEnabled(_both_shared)
            _cb.setToolTip(_stip); _slbl.setToolTip(_stip)
            _grid.addWidget(_slbl, 5, 0); _grid.addWidget(_cb, 5, 1)
        except Exception:
            import traceback; traceback.print_exc()
        self.emceeConfIntervalWidget.progressBar.setValue(0)
        self.EnableUserDefinedParameterButtons(enable=False)
        # Lock main-window dataset/function selection while this window is open, and
        # release the lock when it closes (via eventFilter -> Close).
        self.emceeConfIntervalWidget.installEventFilter(self)
        self._lockMainSelectionForMCMC(True)
        self.emceeConfIntervalWidget.showMaximized()
        if self.errorAvailable:
            self.update_emcee_parameters()
            self.perform_post_sampling_tasks()
            #self.cornerPlot()
            #self.emceeConfIntervalWidget.tabWidget.setCurrentIndex(4)

    # ---- MCMC backend fingerprint (dataset/model identity) + tau history ---- #
    def _mcmc_fingerprint(self):
        """Identity of the current MCMC context, used to tag/verify a backend .h5:
        function, fit type, varying-parameter names, dataset name(s) and a hash of
        the data actually being fit (so a same-shape but different dataset is caught)."""
        fp = {'funcname': self._mcmc_funcname(),
              'joint': bool(getattr(self, '_mcmc_is_joint', False))}
        try:
            fp['varnames'] = list(self.fit.result.var_names)
        except Exception:
            fp['varnames'] = []
        data = np.array([])
        try:
            # Hash the ACTIVE points, not the whole [imin,imax] window. A column
            # limit changes which points are fitted without changing imin/imax, so
            # hashing the window would let a backend .h5 from an unmasked run be
            # silently accepted and continued for a masked one -- the one failure
            # mode here that corrupts results rather than merely confusing a user.
            if fp['joint'] and hasattr(self.fit, 'objectives'):
                fp['dataset'] = sorted(str(o['name']) for o in self.fit.objectives)
                ys = []
                for o in self.fit.objectives:
                    seg = np.asarray(o['y'])[o['imin']:o['imax'] + 1]
                    im = o.get('imask')
                    ys.append(seg if im is None else seg[im[:len(seg)]])
                data = np.concatenate(ys) if ys else np.array([])
            else:
                fp['dataset'] = [os.path.basename(self.sfnames[-1])] if self.sfnames else []
                try:
                    data = np.asarray(self.fit.y)[self.fit.active_index()]
                except Exception:
                    data = np.asarray(self.fit.y)[self.fit.imin:self.fit.imax + 1]
        except Exception:
            fp['dataset'] = []
        try:
            fp['ndata'] = int(len(data))
            fp['datahash'] = (hashlib.sha1(np.ascontiguousarray(data, dtype='float64').tobytes())
                              .hexdigest()[:16]) if len(data) else ''
        except Exception:
            fp['ndata'] = 0
            fp['datahash'] = ''
        return fp

    def _write_mcmc_fingerprint(self):
        """Tag the backend .h5 with the current fingerprint AND the autocorrelation-
        time trajectory, so a later load can verify a match and continue the tau
        history. Best-effort; never breaks sampling."""
        if getattr(self, 'backend', None) is None or not os.path.exists(self.MCMCBackendFile):
            return
        try:
            fp = self._mcmc_fingerprint()
            with self.backend.open('a') as f:
                g = f[self.backend.name]
                g.attrs['xmf_fingerprint'] = json.dumps(fp)
                act = np.asarray(getattr(self, 'autoCorrTime', np.array([[0, 1]])), dtype=float)
                if 'xmf_autocorr_time' in g:
                    del g['xmf_autocorr_time']
                g.create_dataset('xmf_autocorr_time', data=act)
        except Exception:
            import traceback; traceback.print_exc()

    def _read_mcmc_fingerprint(self):
        """Return (fingerprint_dict_or_None, autocorr_time_array_or_None) from the
        currently-set backend .h5."""
        try:
            with self.backend.open('r') as f:
                g = f[self.backend.name]
                fp = json.loads(g.attrs['xmf_fingerprint']) if 'xmf_fingerprint' in g.attrs else None
                act = np.asarray(g['xmf_autocorr_time']) if 'xmf_autocorr_time' in g else None
            return fp, act
        except Exception:
            return None, None

    def _mcmc_fingerprint_mismatch(self, saved):
        """Human-readable description of how a saved fingerprint differs from the
        current context, or '' if they match."""
        cur = self._mcmc_fingerprint()
        msgs = []
        if saved.get('funcname') != cur['funcname']:
            msgs.append('function: file=%s vs current=%s' % (saved.get('funcname'), cur['funcname']))
        if bool(saved.get('joint')) != bool(cur['joint']):
            msgs.append('fit type: file=%s vs current=%s'
                        % ('joint' if saved.get('joint') else 'individual',
                           'joint' if cur['joint'] else 'individual'))
        if list(saved.get('varnames') or []) != list(cur['varnames']):
            msgs.append('varying parameters differ')
        if sorted(saved.get('dataset') or []) != sorted(cur['dataset']):
            msgs.append('dataset: file=%s vs current=%s'
                        % (saved.get('dataset'), cur['dataset']))
        elif saved.get('datahash') and cur['datahash'] and saved['datahash'] != cur['datahash']:
            msgs.append('same dataset name, but the data / fit-range differ')
        return '\n'.join('  - ' + m for m in msgs)

    def _autocorr_trajectory(self, chain, angleindex=None, npts=40):
        """Reconstruct the whole tau-vs-step curve from a loaded chain by computing
        the integrated autocorrelation time over increasing prefixes chain[:n].

        Works for ANY backend (old files with no saved trajectory too). Angle-aware
        (customautocorr) for XLayersPDB*, standard emcee autocorr otherwise. Returns
        an (npts, 2) array [[n_steps, mean_tau], ...] or None if the chain is too short."""
        chain = np.asarray(chain)
        nsteps = chain.shape[0]
        if nsteps < 10:
            return None
        if angleindex is not None:
            from lmfit.customautocorr import integrated_time as _it
            tau_of = lambda c: _it(c, angleindex, tol=0, quiet=True)
        else:
            from emcee.autocorr import integrated_time as _it
            tau_of = lambda c: _it(c, tol=0, quiet=True)
        lo = max(10, nsteps // 50)
        ns = np.unique(np.linspace(lo, nsteps, npts).astype(int))
        traj = []
        for n in ns:
            try:
                tau = float(np.mean(tau_of(chain[:int(n)])))
                if np.isfinite(tau):
                    traj.append([int(n), tau])
            except Exception:
                pass
        return np.array(traj, dtype=float) if traj else None

    def openMCMCBackendFile(self):
        """
        Open an existing backend file
        :return:
        """
        file=QFileDialog.getOpenFileName(self.emceeConfIntervalWidget, 'Select a MCMC file', directory=self.curDir,filter='MCMC Files (*.h5)')[0]
        # Keep the MCMC window in front; a dialog parented to the main window would
        # otherwise reactivate the main window when it closes.
        self.emceeConfIntervalWidget.raise_()
        self.emceeConfIntervalWidget.activateWindow()
        # self.iterations = 0
        if file != '':
            self.MCMCBackendFile=file
            self.emceeConfIntervalWidget.MCMCFileLineEdit.setText(self.MCMCBackendFile)
            if self._mcmc_is_pdb():
                self.backend = CustomHDFBackend(self.MCMCBackendFile)
            else:
                self.backend = backends.HDFBackend(self.MCMCBackendFile)
            # Verify this .h5 belongs to the dataset/model currently selected for
            # analysis before allowing a continue-sampling that would mix runs.
            saved_fp, saved_act = self._read_mcmc_fingerprint()
            if saved_fp is not None:
                mism = self._mcmc_fingerprint_mismatch(saved_fp)
                if mism:
                    ans = QMessageBox.warning(
                        self.emceeConfIntervalWidget, 'MCMC file mismatch',
                        'This MCMC file does not match the current dataset/model:\n\n%s\n\n'
                        'Continuing sampling into it would mix incompatible runs.\n'
                        'Load it anyway?' % mism,
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    self.emceeConfIntervalWidget.raise_()
                    self.emceeConfIntervalWidget.activateWindow()
                    if ans != QMessageBox.Yes:
                        self.backend = None
                        return
            else:
                QMessageBox.information(
                    self.emceeConfIntervalWidget, 'MCMC file',
                    'This MCMC file has no dataset fingerprint (older file); its match '
                    'to the current dataset cannot be verified. Proceeding.', QMessageBox.Ok)
                self.emceeConfIntervalWidget.raise_()
                self.emceeConfIntervalWidget.activateWindow()
            self.fit.result.chain=self.backend.get_chain()[..., :, :]
            if self._mcmc_is_pdb():
                angleindex = np.zeros(len(self.fit.result.var_names))
                ind = 0
                for var_name in self.fit.result.var_names:
                    if var_name == 'theta':
                        angleindex[ind] = 1
                    elif var_name == 'phi':
                        angleindex[ind] = 2
                    else:
                        angleindex[ind] = 0
                    ind += 1
            else:
                angleindex = None

            # tau-vs-step curve: PREFER the trajectory persisted in the .h5 (written at
            # the end of each sampling session) so load does NOT recompute it. Only fall
            # back to reconstructing from the chain for OLD files that predate the saved
            # curve, then to a single recomputed point. (This is the load-time speedup:
            # a saved curve is an O(1) read instead of ~40 integrated_time evaluations.)
            if saved_act is not None and np.ndim(saved_act) == 2 and saved_act.shape[1] == 2 and len(saved_act):
                self.autoCorrTime = saved_act.astype(float)
            else:
                traj = self._autocorr_trajectory(self.fit.result.chain, angleindex=angleindex)
                if traj is not None and len(traj):
                    self.autoCorrTime = traj
                else:
                    if angleindex is not None:
                        _last_tau = np.mean(self.backend.get_autocorr_time(angleindex=angleindex, tol=0))
                    else:
                        _last_tau = np.mean(self.backend.get_autocorr_time(tol=0))
                    self.autoCorrTime = np.array([[self.fit.result.chain.shape[0], _last_tau]])
            self.MCMC_starting_step=self.autoCorrTime[-1,0]
            self.emceeConfIntervalWidget.autoCorrTimeTextEdit.clear()
            self.emceeConfIntervalWidget.autoCorrTimeTextEdit.append('#MCMC_steps\tCorr_Time')
            for _r in self.autoCorrTime:
                self.emceeConfIntervalWidget.autoCorrTimeTextEdit.append('%d\t%.3f' % (_r[0], _r[1]))
            self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.clear()
            self.autoCorrPlot_ax1 = self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.fig.add_subplot(1, 1, 1)
            self.autoCorrPlot_sp, =self.autoCorrPlot_ax1.plot(self.autoCorrTime[:,0],self.autoCorrTime[:,1],'.-')
            self.autoCorrPlot_ax1.set_xlabel('Iterations')
            self.autoCorrPlot_ax1.set_ylabel('correlation_time')
            self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.fig.canvas.draw()
            self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.fig.canvas.flush_events()
            self.emceeConfIntervalWidget.tabWidget.setCurrentIndex(3)
            if self.fit.result.chain.shape[-1]==len(self.fit.result.var_names):
                # if '__lnsigma' not in self.fit.result.var_names:
                #     self.fit.result.var_names.append('__lnsigma')
                #     self.fit.result.nvarys+=1
                #self.fit.result.flatchain=self.fit.result.flatchain()
                self.perform_post_sampling_tasks()
                self.emceeConfIntervalWidget.reuseSamplerCheckBox.setEnabled(True)
                self.emceeConfIntervalWidget.reuseSamplerCheckBox.setChecked(True)
                self.update_emcee_parameters()
            else:
                QMessageBox.warning(self,'Backend Error','The number of parameters in the imported backend file '
                                                         'is not same as the current fitting parameters',
                                    QMessageBox.Ok)
                self.backend=None


    def startMCMCBackendFile(self):
        """
        Create a new backend file to save the MCMC
        :return:
        """
        file=QFileDialog.getSaveFileName(self.emceeConfIntervalWidget, 'Start MCMC file as',directory=self.curDir,filter='MCMC Files (*.h5)')[0]
        # Keep the MCMC window in front; a dialog parented to the main window would
        # otherwise reactivate the main window when it closes.
        self.emceeConfIntervalWidget.raise_()
        self.emceeConfIntervalWidget.activateWindow()
        # self.iterations=0
        if file != '':
            if os.path.exists(file):
                os.remove(file)
            self.MCMCBackendFile=file
            self.emceeConfIntervalWidget.MCMCFileLineEdit.setText(self.MCMCBackendFile)
            if self._mcmc_is_pdb():
                self.backend = CustomHDFBackend(self.MCMCBackendFile)
            else:
                self.backend = backends.HDFBackend(self.MCMCBackendFile)
            self.autoCorrTime=np.array([[0,1.0]])
            self.MCMC_starting_step = self.autoCorrTime[-1, 0]
            self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.clear()
            self.autoCorrPlot_ax1 = self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.fig.add_subplot(1, 1, 1)
            self.autoCorrPlot_sp, = self.autoCorrPlot_ax1.plot(self.autoCorrTime[:,0],self.autoCorrTime[:,1],'.')
            self.autoCorrPlot_ax1.set_xlabel('Iterations')
            self.autoCorrPlot_ax1.set_ylabel('correlation_time')
            self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.fig.canvas.draw()
            self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.fig.canvas.flush_events()
            self.emceeConfIntervalWidget.tabWidget.setCurrentIndex(3)
            self.emceeConfIntervalWidget.reuseSamplerCheckBox.setChecked(False)
            self.emceeConfIntervalWidget.autoCorrTimeTextEdit.clear()
            self.emceeConfIntervalWidget.autoCorrTimeTextEdit.append('#MCMC_steps\tCorr_Time')
            self.emceeConfIntervalWidget.autoCorrTimeTextEdit.append('%d\t%.3f' % (self.autoCorrTime[-1, 0], self.autoCorrTime[-1, 1]))
        else:
            self.backend=None

    def EnableUserDefinedParameterButtons(self,enable=False):
        self.emceeConfIntervalWidget.addUserDefinedParamPushButton.setEnabled(enable)
        self.emceeConfIntervalWidget.removeUserDefinedParamPushButton.setEnabled(enable)
        self.emceeConfIntervalWidget.saveUserDefinedParamPushButton.setEnabled(enable)
        self.emceeConfIntervalWidget.loadUserDefinedParamPushButton.setEnabled(enable)



    def openMCMCUserDefinedParam(self,item,column):
        txt=item.text(0)
        if ':chain:' not in txt:
            parname,expression=txt.split('=')
            self.addMCMCUserDefinedParam(parname=parname, expression=expression)


    def addMCMCUserDefinedParam(self, parname=None, expression=None):
        self.MCMCUserDefinedParamWidget = QWidget()
        self.MCMCUserDefinedParamWidget.setWindowModality(Qt.ApplicationModal)
        uic.loadUi('./UI_Forms/User_Defined_Param_Widget.ui', self.MCMCUserDefinedParamWidget)
        if parname is not None:
            self.MCMCUserDefinedParamWidget.parnameLineEdit.setText(parname)
            self.MCMCUserDefinedParamWidget.parnameLineEdit.setEnabled(False)
            new=False
        else:
            new=True
        if expression is not None:
            self.MCMCUserDefinedParamWidget.expressionTextEdit.setText(expression)
        availParams = self.fit.result.var_names
        self.MCMCUserDefinedParamWidget.availParamListWidget.addItems(availParams)

        self.MCMCUserDefinedParamWidget.addPushButton.clicked.connect(lambda x: self.acceptUserDefinedParams(new=new))
        self.MCMCUserDefinedParamWidget.cancelPushButton.clicked.connect(self.MCMCUserDefinedParamWidget.close)
        self.MCMCUserDefinedParamWidget.availParamListWidget.itemDoubleClicked.connect(self.appendUserDefinedExpression)
        self.MCMCUserDefinedParamWidget.show()

    def appendUserDefinedExpression(self,item):
        txt=item.text()
        self.MCMCUserDefinedParamWidget.expressionTextEdit.insertPlainText(txt)
        self.MCMCUserDefinedParamWidget.expressionTextEdit.moveCursor(QTextCursor.End)

    def acceptUserDefinedParams(self,new=True,parname=None,expression=None):
        if parname is None:
            parname=self.MCMCUserDefinedParamWidget.parnameLineEdit.text()
        if parname=='':
            QMessageBox.warning(self, 'Name Error', 'Please provide a parameter name.',QMessageBox.Ok)
            return
        elif parname in self.param_chain.keys() and new:
            QMessageBox.warning(self, 'Name Error', 'Please provide a parameter name which is not already used.', QMessageBox.Ok)
            return
        else:
            if expression is None:
                txt=self.MCMCUserDefinedParamWidget.expressionTextEdit.toPlainText()
            else:
                txt=expression
            if txt != 'None':
                try:
                    self.emceeConfIntervalWidget.userDefinedParamTreeWidget.itemSelectionChanged.disconnect(
                        self.userDefinedParameterTreeSelectionChanged)
                except:
                    pass
                if new:
                    l1 = QTreeWidgetItem([parname+'='+txt])
                else:
                    self.emceeConfIntervalWidget.userDefinedParamTreeWidget.currentItem().setText(0,parname+'='+txt)
                self.param_chain[parname] = {}
                ttxt1=copy.copy(txt[0:])
                for name in self.fit.result.var_names:
                    ttxt1 = ttxt1.replace(name, "self.final_fit_params['%s'].value" % name)
                self.final_fit_params.add(parname, value=eval(ttxt1))
                for i in range(self.chain_shape[1]):
                    ttxt = copy.copy(txt[0:])
                    for name in self.fit.result.var_names:
                        ttxt=ttxt.replace(name,"self.param_chain['%s'][%d]"%(name,i))
                    try:
                        self.param_chain[parname][i]=eval(ttxt)
                        if new:
                            l1_child = QTreeWidgetItem(['%s:chain:%d' % (parname, i)])
                            l1.addChild(l1_child)
                            self.emceeConfIntervalWidget.userDefinedParamTreeWidget.addTopLevelItem(l1)
                        if expression is None:
                            self.MCMCUserDefinedParamWidget.close()

                    except:
                        QMessageBox.warning(self, 'Expression Error',
                                            'Some problems in the expression\n' + traceback.format_exc(),
                                            QMessageBox.Ok)
                        return
                self.emceeConfIntervalWidget.userDefinedParamTreeWidget.itemSelectionChanged.connect(
                    self.userDefinedParameterTreeSelectionChanged)
            else:
                self.MCMCUserDefinedParamWidget.close()

    def removeMCMCUserDefinedParam(self):
        try:
            self.emceeConfIntervalWidget.userDefinedParamTreeWidget.itemSelectionChanged.disconnect()
        except:
            pass
        for item in self.emceeConfIntervalWidget.userDefinedParamTreeWidget.selectedItems():
            parname=item.text(0).split('=')[0]
            del self.param_chain[parname]
            self.final_fit_params.pop(parname)
            index=self.emceeConfIntervalWidget.userDefinedParamTreeWidget.indexOfTopLevelItem(item)
            self.emceeConfIntervalWidget.userDefinedParamTreeWidget.takeTopLevelItem(index)
        self.emceeConfIntervalWidget.userDefinedParamTreeWidget.itemSelectionChanged.connect(self.userDefinedParameterTreeSelectionChanged)


    def saveMCMCUserDefinedParam(self):
        fname=QFileDialog.getSaveFileName(caption='Save User-Defined parameter expression as',filter='Expression files (*.expr)'
                                          ,directory=self.curDir)[0]
        if fname!='':
            if os.path.splitext(fname)[0]=='':
                fname=fname+'.expr'
        else:
            return
        fh=open(fname,'w')
        txt ='#Parameter Expressions saved on %s\n'%(time.asctime())
        txt += '#Category:%s\n' % self.curr_category
        txt += '#Function:%s\n' % self._mcmc_funcname()   # joint-aware / None-safe
        root = self.emceeConfIntervalWidget.userDefinedParamTreeWidget.invisibleRootItem()
        child_count=root.childCount()
        for i in range(child_count):
            txt += '%s\n'%root.child(i).text(0)
        fh.write(txt)
        fh.close()


    def loadMCMCUserDefinedParam(self):
        fname=QFileDialog.getOpenFileName(self,'Open Expression file',filter='Expression files (*.expr)'
                                          ,directory=self.curDir)[0]
        if fname!='':
            try:
                self.emceeConfIntervalWidget.userDefinedParamTreeWidget.itemSelectionChanged.disconnect()
            except:
                pass
            #Removing the existing User-Defined parameters, if present
            root = self.emceeConfIntervalWidget.userDefinedParamTreeWidget.invisibleRootItem()
            child_count = root.childCount()
            for i in range(child_count):
                parname, expression = root.child(i).text(0).split('=')
                del self.param_chain[parname]
            self.emceeConfIntervalWidget.userDefinedParamTreeWidget.clear()

            fh=open(fname,'r')
            lines=fh.readlines()
            for line in lines:
                if line[0]=='#':
                    if 'Category' in line:
                        _,category=line[1:].strip().split(':')
                        if category!=self.curr_category:
                            QMessageBox.warning(self,'File Error','The expression file is not generated from the same category of functions as used here.'
                                                ,QMessageBox.Ok)
                            return
                    elif 'Function' in line:
                        _,func=line[1:].strip().split(':')
                        if func!=self.curr_module:
                            QMessageBox.warning(self, 'File Error',
                                                'The expression file is not generated from the same function as used here.'
                                                , QMessageBox.Ok)
                            return
                else:
                    parname,expression=line.strip().split('=')
                    self.acceptUserDefinedParams(new=True,parname=parname,expression=expression)
        self.emceeConfIntervalWidget.userDefinedParamTreeWidget.itemSelectionChanged.connect(self.userDefinedParameterTreeSelectionChanged)




    def MCMCWalker_changed(self):
        self.emceeConfIntervalWidget.reuseSamplerCheckBox.setCheckState(Qt.Unchecked)
        self.update_emcee_parameters()


    def update_emcee_parameters(self):
        self.emcee_walkers=int(self.emceeConfIntervalWidget.MCMCWalkerLineEdit.text())
        self.emcee_steps=int(self.emceeConfIntervalWidget.MCMCStepsLineEdit.text())
        self.emcee_burn=int(self.emceeConfIntervalWidget.MCMCBurnLineEdit.text())
        self.emcee_thin = int(self.emceeConfIntervalWidget.MCMCThinLineEdit.text())
        if self.emceeConfIntervalWidget.reuseSamplerCheckBox.isChecked():
            self.reuse_sampler=True
        else:
            self.reuse_sampler=False
        self.emcee_cores = int(self.emceeConfIntervalWidget.ParallelCoresLineEdit.text())
        try:
            self.seed_temperature = float(self.emceeConfIntervalWidget.seedTempSpinBox.value())
        except Exception:
            self.seed_temperature = getattr(self, 'seed_temperature', 1.0)
        try:
            self.seed_mode = ('individual'
                              if self.emceeConfIntervalWidget.seedModeCombo.currentIndex() == 1
                              else 'joint')
        except Exception:
            self.seed_mode = getattr(self, 'seed_mode', 'joint')

    def start_emcee_sampling(self):
        # Require an MCMC backend file before sampling; without it there is nothing to
        # write to and the autocorrelation callback (self.backend.*) would crash.
        if getattr(self, 'backend', None) is None:
            QMessageBox.warning(self.emceeConfIntervalWidget, 'No MCMC File',
                                'Please create a new MCMC file ("Start MCMC file") or load an '
                                'existing one before sampling.', QMessageBox.Ok)
            return
        try:
            self.emceeConfIntervalWidget.parameterTreeWidget.itemSelectionChanged.disconnect()
        except:
            pass
        self.emceeConfIntervalWidget.parameterTreeWidget.clear()
        self.emceeConfIntervalWidget.chainMPLWidget.clear()
        self.emceeConfIntervalWidget.correlationMPLWidget.clear()
        self.emceeConfIntervalWidget.cornerPlotMPLWidget.clear()
        self.emceeConfIntervalWidget.confIntervalTextEdit.clear()
        self.update_emcee_parameters()
        # Step count at the start of THIS sampling session (0 for a fresh run, the
        # loaded chain length on a resume) -- used for a correct ETA in fitErrorCallback.
        self._mcmc_session_start_step = float(getattr(self, 'MCMC_starting_step', 0) or 0)
        # Last step at which the full-chain tau was recomputed (throttle anchor). Seed
        # from the loaded length so the first live tau point continues the reconstructed
        # curve after the chain has grown by one (length-scaled) interval.
        self._last_autocorr_step = int(self._mcmc_session_start_step)
        if not self.errorAvailable:
            self.emcee_frac=self.emcee_burn/self.emcee_steps
        # Immediate feedback: the sampler setup (especially XLayersPDB: ray init, PDB
        # parsing and the first form-factor evaluations) can take many seconds before
        # the first progress callback fires, during which the window looked idle.
        self.emceeConfIntervalWidget.startSamplingPushButton.setEnabled(False)
        self.emceeConfIntervalWidget.fitIterLabel.setText('Initializing sampler, please wait...')
        self.emceeConfIntervalWidget.progressBar.setMaximum(0)  # busy/indeterminate until first step
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        # Warm the numba JIT (first PDB model eval compiles the njit kernels, ~1s) while
        # the "Initializing..." label is up, so that cost isn't paid silently mid-run.
        try:
            _wf = getattr(self.fit, 'func', None)
            if _wf is not None and hasattr(_wf, 'y'):
                _wf.y()
        except Exception:
            pass
        try:
            self.doFit(fit_method='emcee', emcee_walkers=self.emcee_walkers, emcee_steps=self.emcee_steps,
                           emcee_cores=self.emcee_cores, reuse_sampler=self.reuse_sampler, emcee_burn=self.emcee_burn,
                       emcee_thin=self.emcee_thin, backend=self.backend)
        finally:
            QApplication.restoreOverrideCursor()
            self.emceeConfIntervalWidget.startSamplingPushButton.setEnabled(True)
            # If sampling never started (early return in doFit), showFitInfoDlg didn't
            # set a real range, so clear the busy/indeterminate bar back to determinate.
            if self.emceeConfIntervalWidget.progressBar.maximum() == 0:
                self.emceeConfIntervalWidget.progressBar.setMaximum(100)
                self.emceeConfIntervalWidget.progressBar.setValue(0)
                self.emceeConfIntervalWidget.fitIterLabel.setText('')


    def conf_interv_status(self,params,iterations,residual,fit_scale):
        self.confIntervalStatus.setText(self.confIntervalStatus.text().split('\n')[0]+'\n\n {:^s} = {:10d}'.format('Iteration',iterations))            
        QApplication.processEvents()
        
    def runFit(self,  emcee_walkers=100, emcee_steps=100, emcee_cores=1, reuse_sampler=False, emcee_burn=30, emcee_thin=1, backend=None):
        self.start_time=time.time()
        if self.fit_method=='emcee':
            # self.iterations = 0
            if os.path.exists(self.MCMCBackendFile):
                self.fit.result.chain = self.backend.get_chain()[..., :, :]

                if self._mcmc_is_pdb():
                    angleindex = np.zeros(len(self.fit.result.var_names))
                    ind = 0
                    for var_name in self.fit.result.var_names:
                        if var_name == 'theta':
                            angleindex[ind] = 1
                        elif var_name == 'phi':
                            angleindex[ind] = 2
                        else:
                            angleindex[ind] = 0
                        ind += 1

                    data = np.array([[self.fit.result.chain.shape[0],
                                      np.mean(self.backend.get_autocorr_time(angleindex=angleindex, tol=0))]])
                else:
                    data = np.array(
                        [[self.fit.result.chain.shape[0], np.mean(self.backend.get_autocorr_time(tol=0))]])
                """data = np.array(
                    [[self.fit.result.chain.shape[0], np.mean(self.fit.fitter.sampler.get_autocorr_time(tol=0))]])"""
                self.autoCorrTime = np.vstack([self.autoCorrTime, data])
                # self.MCMC_starting_step = copy.copy(self.fit.fitter.sampler.iteration)
                self.MCMC_starting_step=self.autoCorrTime[-1,0]
            else:
                self.MCMC_starting_step=0
                self.autoCorrTime=np.array([[0,1]])
        xmin, xmax = self.xmin[self.sfnames[-1]], self.xmax[self.sfnames[-1]]
        self.fit_report,self.fit_message=self.fit.perform_fit(xmin,xmax,fit_scale=self.fit_scale, fit_method=self.fit_method,
                                                              maxiter=int(self.fitIterationLineEdit.text()),
                                                              emcee_walkers=emcee_walkers, emcee_steps=emcee_steps,
                                                              emcee_cores=emcee_cores, reuse_sampler=reuse_sampler, emcee_burn=emcee_burn,
                                                              emcee_thin=emcee_thin, backend=backend,
                                                              seed_temperature=getattr(self, 'seed_temperature', 1.0))
        
    
    def showFitInfoDlg(self, emcee_walkers=100, emcee_steps=100, emcee_burn=30):
        if self.fit_method!='emcee':
            self.fitInfoDlg=QDialog(self)
            vblayout=QVBoxLayout(self.fitInfoDlg)
            self.fitIterLabel=QLabel('Iteration: 0,\t Chi-Sqr: Not Available',self.fitInfoDlg)
            vblayout.addWidget(self.fitIterLabel)
            self.fitPlotWidget = PlotWidget()
            vblayout.addWidget(self.fitPlotWidget)
            self.stopFitPushButton=QPushButton('Stop')
            vblayout.addWidget(self.stopFitPushButton)
            self.stopFitPushButton.clicked.connect(self.stopFit)
            if len(self.sfnames)==1 or hasattr(getattr(self,'fit',None),'objectives'):
                self.fitInfoDlg.setWindowTitle('Please wait for the fitting to be completed')
                self.fitInfoDlg.setModal(True)
                self.fitInfoDlg.show()
        else:
            #self.emceeConfIntervalWidget.fitIterLabel.setText('Time left (hh:mm:ss): %s'%('N.A.'))
            self.emceeConfIntervalWidget.progressBar.setMaximum(int(emcee_steps+self.MCMC_starting_step))
            self.emceeConfIntervalWidget.progressBar.setMinimum(0)
            self.emceeConfIntervalWidget.progressBar.setValue(int(self.MCMC_starting_step))
            # Stop is connected once in confInterval_emcee, not here (avoids stacked connections).

    def stopFit(self):
        self.fit.fit_abort=True
        self.fit_stopped=True
        if self.fit_method=='emcee':
            # Idempotent: setting fit_abort above is enough, so extra Stop clicks are
            # harmless. (Previously this disconnected the button, which threw when the
            # slot had been connected/fired more than once.)
            self.reuse_sampler = True

    def closeFitInfoDlg(self):
        # Only the non-emcee branch of showFitInfoDlg creates fitInfoDlg; for
        # emcee there is no dialog to close. Guard so the unconditional call in
        # doFit doesn't crash on an MCMC run.
        if getattr(self, 'fitInfoDlg', None) is not None:
            self.fitInfoDlg.done(0)
            self.fitInfoDlg = None

        

    def fitCallback(self,params,iterations,residual,fit_scale):
        # self.fitIterLabel.setText('Iteration=%d,\t Chi-Sqr=%.5e'%(iterations,np.sum(residual**2)))
        # if np.any(self.fit.yfit):
        chisqr=np.sum(residual**2)
        if chisqr<self.tchisqr:
            self.fitData.append([iterations,chisqr])
            self.fitIterLabel.setText('Iteration=%d,\t Chi-Sqr=%.5e' % (iterations, chisqr))
            self.fitPlotWidget.add_data(x=np.array(self.fitData)[:,0],y=np.array(self.fitData)[:,1])
            self.temp_params=copy.copy(params)
            # The live model-curve redraw uses single-Fit attrs (self.fit.x/yfit/imin/imax);
            # a JointFit has none of those (per-objective instead), so skip it there and
            # just show the iteration/chi-sqr progress (a joint fit still gets a live screen).
            if not hasattr(self.fit, 'objectives'):
                if type(self.fit.x)==dict:
                    for key in self.fit.x.keys():
                        self.plotWidget.add_data(x=self.fit.x[key][self.fit.imin[key]:self.fit.imax[key]+1],y=self.fit.yfit[key],\
                                         name=self.funcListWidget.currentItem().text()+':'+key,fit=True,color=self._single_fit_color(key))
                        self.fit.params['output_params']['Residuals_%s'%key] = {'x': self.fit.x[key][self.fit.imin[key]:self.fit.imax[key]+1],
                                                                                'y': (self.fit.y[key][self.fit.imin[key]:self.fit.imax[key]+1]-self.fit.yfit[key])
                        /self.fit.yerr[key][self.fit.imin[key]:self.fit.imax[key]+1]}
                else:
                    self.plotWidget.add_data(x=self.fit.x[self.fit.imin:self.fit.imax + 1], y=self.fit.yfit, \
                                             name=self.funcListWidget.currentItem().text(), fit=True,color=self._single_fit_color())
                    self.fit.params['output_params']['Residuals']={'x':self.fit.x[self.fit.imin:self.fit.imax + 1],
                                                                   'y': (self.fit.y[self.fit.imin:self.fit.imax + 1]-self.fit.yfit)/self.fit.yerr[self.fit.imin:self.fit.imax + 1]}
            self.tchisqr=chisqr
        QApplication.processEvents()


    def fitErrorCallback(self, params, iterations, residual, fit_scale):
        if hasattr(self.fit.fitter, 'sampler'):

            if self._mcmc_is_pdb():
                angleindex = np.zeros(len(self.fit.result.var_names))
                ind = 0
                for var_name in self.fit.result.var_names:
                    if var_name == 'theta':
                        angleindex[ind] = 1
                    elif var_name == 'phi':
                        angleindex[ind] = 2
                    else:
                        angleindex[ind] = 0
                    ind += 1

            if self.fit.fitter.sampler.iteration>self.MCMC_starting_step:
                self.MCMC_starting_step=copy.copy(self.fit.fitter.sampler.iteration)
                time_taken=time.time()-self.start_time
                # ETA from THIS session's rate over the steps remaining to the target
                # (progress-bar max = session_start + requested steps). On a RESUME the
                # loaded chain length can exceed the requested step count, so the old
                # `emcee_steps - MCMC_starting_step` went NEGATIVE and time.gmtime()
                # raised OSError [Errno 22], stalling the run. Guard finite/non-negative
                # and cap to a value time.gmtime accepts.
                step_now = float(self.MCMC_starting_step)
                target = float(self.emceeConfIntervalWidget.progressBar.maximum())
                session_start = float(getattr(self, '_mcmc_session_start_step', 0) or 0)
                session_done = max(step_now - session_start, 1.0)
                remaining = max(target - step_now, 0.0)
                time_left = time_taken / session_done * remaining
                if (not np.isfinite(time_left)) or time_left < 0:
                    time_left = 0.0
                time_left = min(time_left, 359999.0)   # gmtime-safe: <= 99:59:59

                self.emceeConfIntervalWidget.fitIterLabel.setText('MCMC Step=%d, Time left (hh:mm:ss): %s'
                                                            %(self.MCMC_starting_step, time.strftime('%H:%M:%S',time.gmtime(time_left))))
                self.emceeConfIntervalWidget.progressBar.setValue(self.MCMC_starting_step)
                #print(iterations, self.fit.fitter.sampler.iteration)
                # if (iterations-self.iterations)>self.emcee_walkers and (self.MCMC_starting_step+int(iterations/self.emcee_walkers))%100==0:
                # Recompute the FULL-chain integrated autocorrelation time (discard=0,
                # the same quantity the load-time reconstruction uses -> the live points
                # continue that curve with no discontinuity). To avoid the old lurch
                # (an O(N log N) recompute every 100 steps as N grows), throttle the
                # cadence to grow with the chain: frequent early (N small, cheap), sparser
                # later, so each recompute stays a small ~constant slice of wall-time.
                step_now = int(self.MCMC_starting_step)
                _interval = max(100, step_now // 50)
                if step_now - int(getattr(self, '_last_autocorr_step', 0)) >= _interval:
                    self._last_autocorr_step = step_now
                    if self._mcmc_is_pdb():
                        _tau = self.backend.get_autocorr_time(angleindex=angleindex, tol=0)
                    else:
                        _tau = self.backend.get_autocorr_time(tol=0)
                    data = np.array([[step_now, np.mean(_tau)]])
                    self.autoCorrTime=np.vstack([self.autoCorrTime, data])
                    self.emceeConfIntervalWidget.autoCorrTimeTextEdit.append(
                        '%d\t%.3f' % (self.autoCorrTime[-1, 0], self.autoCorrTime[-1, 1]))

                    self.autoCorrPlot_sp.set_data(self.autoCorrTime[:,0],self.autoCorrTime[:,1])
                    self.autoCorrPlot_ax1.set_xlim(0.98 * min(self.autoCorrTime[:,0]), 1.02 * max(self.autoCorrTime[:,0]))
                    self.autoCorrPlot_ax1.set_ylim(0.98 * min(self.autoCorrTime[:, 1]), 1.02 * max(self.autoCorrTime[:, 1]))
                    self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.fig.canvas.draw()
                    self.emceeConfIntervalWidget.autoCorrTimeMPLWidget.fig.canvas.flush_events()
                    # Show the autocorrelation tab once, but don't yank focus back to it
                    # on every update (annoying if the user is watching another tab).
                    if self.autoCorrTime.shape[0] <= 2:
                        self.emceeConfIntervalWidget.tabWidget.setCurrentIndex(3)
                    print(iterations, self.MCMC_starting_step, self.autoCorrTime[-1,1])
                    """if self.autoCorrTime[-1,1]*50<(self.MCMC_starting_step+int(iterations/self.emcee_walkers)):
                        self.stopFit()"""
                self.start_time=time.time()
        # self.iterations = copy.copy(iterations)
        QApplication.processEvents()

    def _on_mcmc_tab_changed(self, index):
        """Lazily compute the per-parameter ACF the first time the Autocorrelation tab
        (index 2) is opened after a (re)sampling/load -- it's the costliest plot, so it
        is deferred out of perform_post_sampling_tasks to keep load/finish fast."""
        if index == 2 and getattr(self, '_acf_dirty', False):
            self._acf_dirty = False
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                QApplication.processEvents()
                self._compute_parameter_acf()
            finally:
                QApplication.restoreOverrideCursor()

    def _compute_parameter_acf(self):
        """Per-parameter autocorrelation-vs-lag over the whole chain (Autocorrelation
        tab) + the correlation-time table. Deferred: run on first view of that tab."""
        if getattr(self, 'param_chain', None) is None:
            return
        keys = getattr(self, '_sampled_keys', None) or list(self.param_chain.keys())
        angleindex = None
        if self._mcmc_is_pdb():
            angleindex = np.zeros(len(self.fit.result.var_names))
            for ind, var_name in enumerate(self.fit.result.var_names):
                angleindex[ind] = 1 if var_name == 'theta' else (2 if var_name == 'phi' else 0)
        acor = OrderedDict()
        self.emceeConfIntervalWidget.correlationMPLWidget.clear()
        ax1 = self.emceeConfIntervalWidget.correlationMPLWidget.fig.add_subplot(1, 1, 1)
        corr_time = []
        try:
            acor_mcmc = self.fit.fitter.sampler.get_autocorr_time(quiet=True)
        except Exception:
            if angleindex is not None:
                acor_mcmc = self.backend.get_autocorr_time(angleindex=angleindex, quiet=True)
            else:
                acor_mcmc = self.backend.get_autocorr_time(quiet=True)
        # Full-range ACF, incremental per-walker mean (O(N) memory) + thinned plot.
        _plot_pts = 3000
        for i, key in enumerate(keys):
            if key not in self.param_chain:
                continue
            acc = None; nw = 0
            for ikey in self.param_chain[key].keys():
                tdata = self.param_chain[key][ikey]
                res = sm.tsa.acf(tdata, nlags=len(tdata) - 1, fft=True)
                acc = res if acc is None else acc + res
                nw += 1
            acor[key] = acc / max(nw, 1)
            lags = np.arange(len(acor[key]))
            stride = max(1, len(acor[key]) // _plot_pts)
            ax1.plot(lags[::stride], acor[key][::stride], '-', label='para=%s' % key)
            if i < len(acor_mcmc):
                corr_time.append([key, acor_mcmc[i]])
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Autocorrelation')
        l = ax1.legend(loc='best')
        l.set_draggable(True)
        self.emceeConfIntervalWidget.correlationMPLWidget.draw()
        self.emceeConfIntervalWidget.corrTimeTextEdit.clear()
        self.emceeConfIntervalWidget.corrTimeTextEdit.setFont(QFont("Courier", 10))
        corr_text = tabulate(corr_time, headers=['Parameter', 'Correlation-time (Steps)'],
                             stralign='left', numalign='left', tablefmt='simple')
        self.emceeConfIntervalWidget.corrTimeTextEdit.append(corr_text)

    def perform_post_sampling_tasks(self):
        #self.emceeConfIntervalWidget.progressBar.setValue(self.emcee_steps)
        # final_fit_params is normally set by a preceding least-squares fit and is
        # used as the "truth" reference by cornerPlot, the user-defined MCMC params,
        # and the errorbar table. When MCMC is run without a prior least-squares fit
        # it is never set, so seed it from the current fit result here.
        if getattr(self, 'final_fit_params', None) is None:
            self.final_fit_params = copy.copy(self.fit.result.params)
        if self._mcmc_is_pdb():
            angleindex = np.zeros(len(self.fit.result.var_names))
            ind = 0
            for var_name in self.fit.result.var_names:
                if var_name == 'theta':
                    angleindex[ind] = 1
                elif var_name == 'phi':
                    angleindex[ind] = 2
                else:
                    angleindex[ind] = 0
                ind += 1
            data = np.array(
                [[self.MCMC_starting_step, np.mean(self.backend.get_autocorr_time(angleindex=angleindex, tol=0))]])
        else:
            data = np.array(
                [[self.MCMC_starting_step, np.mean(self.backend.get_autocorr_time(tol=0))]])
        """data = np.array(
            [[self.MCMC_starting_step, np.mean(self.fit.fitter.sampler.get_autocorr_time(tol=0))]])"""

        self.autoCorrTime = np.vstack([self.autoCorrTime, data])
        self.emceeConfIntervalWidget.fitIterLabel.setText('MCMC Step=%d, Time left (hh:mm:ss): 00:00:00'%self.MCMC_starting_step)
        self.chain=self.fit.result.chain
        self.chain_shape=self.chain.shape
        self.param_chain=OrderedDict()
        for i,key in enumerate(self.fit.result.flatchain.keys()):
            l1=QTreeWidgetItem([key])
            self.param_chain[key]=OrderedDict()
            for j in range(self.chain_shape[1]):
                self.param_chain[key][j]=self.chain[:,j,i]
                l1_child=QTreeWidgetItem(['%s:chain:%d'%(key,j)])
                l1.addChild(l1_child)
            self.emceeConfIntervalWidget.parameterTreeWidget.addTopLevelItem(l1)
        self.emceeConfIntervalWidget.parameterTreeWidget.itemSelectionChanged.connect(self.parameterTreeSelectionChanged)
        # sampled parameter keys (excludes user-defined params added below) -- used by
        # the deferred ACF computation so it never indexes acor_mcmc out of range.
        self._sampled_keys = list(self.fit.result.flatchain.keys())

        # DEFER the per-parameter ACF-vs-lag (Autocorrelation tab): it is an FFT per
        # walker per parameter over the whole chain -- the costliest post-sampling plot.
        # Mark it dirty and compute lazily the first time the user opens that tab
        # (_on_mcmc_tab_changed -> _compute_parameter_acf), so load/finish stays fast.
        self._acf_dirty = True
        self.emceeConfIntervalWidget.correlationMPLWidget.clear()
        _axacf = self.emceeConfIntervalWidget.correlationMPLWidget.fig.add_subplot(1, 1, 1)
        _axacf.text(0.5, 0.5, 'Open this tab to compute the autocorrelation…',
                    ha='center', va='center', transform=_axacf.transAxes, fontsize=9, color='#888')
        _axacf.set_xticks([]); _axacf.set_yticks([])
        self.emceeConfIntervalWidget.correlationMPLWidget.draw()
        self.emceeConfIntervalWidget.corrTimeTextEdit.clear()

        #Plotting Acceptance Ratio
        self.emceeConfIntervalWidget.acceptFracMPLWidget.clear()
        ax2=self.emceeConfIntervalWidget.acceptFracMPLWidget.fig.add_subplot(1,1,1)
        try:
            ax2.plot(self.fit.result.acceptance_fraction,'-')
        except:
            ax2.plot(self.backend.accepted,'-')
        ax2.set_xlabel('Walkers')
        ax2.set_ylabel('Acceptance Ratio')
        self.emceeConfIntervalWidget.acceptFracMPLWidget.draw()

        # Reconnect fresh each run: perform_post_sampling_tasks can fire multiple
        # times, and QPushButton.clicked passes a `checked` bool that would land in
        # cornerPlot's new `subset` arg -- wrap in a no-arg lambda. Joint plots the
        # checked subset (many per-dataset params); individual plots all params.
        try:
            self.emceeConfIntervalWidget.calcConfIntervPushButton.clicked.disconnect()
        except Exception:
            pass
        if getattr(self, '_mcmc_is_joint', False):
            self.emceeConfIntervalWidget.calcConfIntervPushButton.clicked.connect(
                lambda: self._plot_joint_subset())
        else:
            self.emceeConfIntervalWidget.calcConfIntervPushButton.clicked.connect(
                lambda: self.cornerPlot())
        # Land on the (now cheap) autocorrelation-TIME tab, NOT the correlation tab
        # (index 2): switching to index 2 would immediately trigger the deferred ACF.
        self.emceeConfIntervalWidget.tabWidget.setCurrentIndex(3)

        #Calculating User-Defined parameters
        self.EnableUserDefinedParameterButtons(enable=True)
        root = self.emceeConfIntervalWidget.userDefinedParamTreeWidget.invisibleRootItem()
        child_count = root.childCount()
        if child_count>0:
            for i in range(child_count):
                parname, expression = root.child(i).text(0).split('=')
                self.param_chain[parname]={}
                ttxt = expression[0:]
                ttxt1 = copy.copy(ttxt)
                for name in self.fit.result.var_names:
                    ttxt1 = ttxt1.replace(name, "self.final_fit_params['%s'].value" % name)
                self.final_fit_params.add(parname, value=eval(ttxt1))
                for j in range(self.chain_shape[1]):
                    ttxt2 = copy.copy(ttxt)
                    for name in self.fit.result.var_names:
                        ttxt2 = ttxt2.replace(name, "self.param_chain['%s'][%d]" % (name, j))
                    self.param_chain[parname][j] = eval(ttxt2)

        # joint window: fill the parameter-subset checklist (now that user-defined
        # params, if any, are also in param_chain)
        if getattr(self, '_mcmc_is_joint', False):
            self._populate_joint_subset_list()

        # tag the backend .h5 with the dataset/model fingerprint + tau trajectory so a
        # later load can verify the match and continue the autocorrelation history.
        self._write_mcmc_fingerprint()

        self.reuse_sampler = True

    def _corner_names(self, subset=None):
        """(names, col_indices) to plot: every sampled param except __lnsigma,
        optionally restricted to ``subset`` (an iterable of names). ``col_indices``
        index into ``self.pardata`` (whose columns follow param_chain key order)."""
        subset = None if subset is None else set(subset)
        names, cols = [], []
        for i, key in enumerate(self.param_chain.keys()):
            if key == '__lnsigma':
                continue
            if subset is not None and key not in subset:
                continue
            names.append(key)
            cols.append(i)
        return names, cols

    def _rebuild_pardata(self, first):
        """Flatten the (steps-first, walkers) chains into ``self.pardata``
        (rows = samples, cols = param_chain keys). Shared by corner/summary."""
        nkeep = self.chain_shape[0] - first
        self.pardata = np.empty((nkeep * self.chain_shape[1], len(self.param_chain.keys())))
        for i, key in enumerate(self.param_chain.keys()):
            for j in self.param_chain[key].keys():
                self.pardata[j * nkeep:(j + 1) * nkeep, i] = self.param_chain[key][j][first:]

    def _corner_truths(self, names, cols):
        """Best-fit 'truth' per plotted param; median fallback for params (e.g.
        user-defined) that are absent from final_fit_params."""
        vals = []
        for name, col in zip(names, cols):
            if name in self.final_fit_params.keys():
                vals.append(self.final_fit_params[name].value)
            else:
                vals.append(float(np.median(self.pardata[:, col])))
        return vals

    # cap the number of samples corner renders: the N x N panel histograms are the
    # dominant cost, and beyond ~20k samples they are visually identical. Quantiles /
    # error bars (calcMCMCerrorbars) still use the FULL self.pardata, so only the
    # plotted density is thinned, not the reported statistics.
    _CORNER_MAX_SAMPLES = 20000

    def _corner_render_data(self, cols):
        """self.pardata[:, cols], thinned by a uniform stride to at most
        _CORNER_MAX_SAMPLES rows (rendering only)."""
        data = self.pardata[:, cols]
        n = data.shape[0]
        if n > self._CORNER_MAX_SAMPLES:
            step = int(np.ceil(n / self._CORNER_MAX_SAMPLES))
            data = data[::step]
        return data

    def _draw_corner(self, fig, names, cols, values, bins, percentile):
        """Render a corner plot of ``self.pardata[:, cols]`` into ``fig``.
        Reused by the embedded widget, the subset re-plot, and the pop-out."""
        ndim = max(len(names), 1)
        quantiles = [(100 - percentile) / 100, 0.5, percentile / 100]
        corner.corner(self._corner_render_data(cols), labels=names, bins=bins,
                      levels=(percentile / 100,), truths=values, quantiles=quantiles,
                      show_titles=True, title_fmt='.3e', use_math_text=True,
                      title_kwargs={'fontsize': 3 * 12 / ndim},
                      label_kwargs={'fontsize': 3 * 12 / ndim}, fig=fig)
        for ax3 in fig.get_axes():
            ax3.set_xlabel('')
            ax3.set_ylabel('')
            ax3.tick_params(axis='y', labelsize=3 * 12 / ndim, rotation=0)
            ax3.tick_params(axis='x', labelsize=3 * 12 / ndim, rotation=90)
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
        return len(names)

    def cornerPlot(self, subset=None):
        """Draw the corner plot for the sampled parameters (or a ``subset`` of
        names). Caches the plotted names/cols so calcMCMCerrorbars and the pop-out
        stay consistent with what was actually drawn."""
        percentile = self.emceeConfIntervalWidget.percentileDoubleSpinBox.value()
        first = int(self.emceeConfIntervalWidget.MCMCBurnLineEdit.text())
        bins = self.emceeConfIntervalWidget.MCMCHistogramBinSpinBox.value()
        self._rebuild_pardata(first)
        names, cols = self._corner_names(subset)
        if not names:
            QMessageBox.warning(self.emceeConfIntervalWidget, 'Corner plot',
                                'No parameters selected to plot.', QMessageBox.Ok)
            return
        values = self._corner_truths(names, cols)
        # remember for calcMCMCerrorbars / pop-out
        self._corner_names_cache = names
        self._corner_cols_cache = cols
        self._corner_bins_cache = bins
        self._corner_percentile_cache = percentile
        w = self.emceeConfIntervalWidget.cornerPlotMPLWidget
        w.clear()
        self._draw_corner(w.fig, names, cols, values, bins, percentile)
        self.emceeConfIntervalWidget.tabWidget.setCurrentWidget(
            self.emceeConfIntervalWidget.cornerPlotTab)
        w.draw()
        self.calcMCMCerrorbars(burn=first, percentile=percentile)


    def calcMCMCerrorbars(self,burn=0,percentile=85):
        # use the names/cols actually plotted (subset-aware); fall back to all params
        names = getattr(self, '_corner_names_cache', None)
        cols = getattr(self, '_corner_cols_cache', None)
        if names is None or cols is None:
            names, cols = self._corner_names()
        ndim = len(names)
        mesg = [['Parameters', 'Fit-Value', 'Value(50%)', 'Left-error(%.3f%s)' % (100 - percentile,'%'), 'Right-error(%.3f%s)' % (percentile,'%')]]
        ci_rows = []
        fig = self.emceeConfIntervalWidget.cornerPlotMPLWidget.fig
        try:
            axes = np.array(fig.axes).reshape((ndim, ndim)) if ndim > 0 else None
        except ValueError:
            axes = None                              # axis count != ndim^2 (e.g. colorbar); skip titles
        for i, (key, col) in enumerate(zip(names, cols)):
            errors=corner.quantile(self.pardata[:,col],[(100-percentile)/100, 50/100, percentile/100])
            fitval = self.final_fit_params[key].value if key in self.final_fit_params.keys() else errors[1]
            mesg.append([key, fitval, errors[1], errors[1]-errors[0], errors[2]-errors[1]])
            ci_rows.append((key, fitval, errors[1], errors[1]-errors[0], errors[2]-errors[1]))
            val='%.3e'%errors[1]
            neger='%.3e'%(errors[1]-errors[0])
            poser='%.3e'%(errors[2]-errors[1])
            if axes is not None and i < ndim:
                ax = axes[i, i]
                ax.set_title(r'%s' '\n' r'$%s_{%s}^{%s}$'%(key.center(24),val,neger,poser),
                             fontsize=3 * 12 / max(ndim, 1))
        self.emceeConfIntervalWidget.cornerPlotMPLWidget.draw()
        self.emceeConfIntervalWidget.confIntervalTextEdit.clear()
        self.emceeConfIntervalWidget.confIntervalTextEdit.setFont(QFont("Courier", 10))
        txt = tabulate(mesg, headers='firstrow', stralign='left', numalign='left', tablefmt='simple')
        self.emceeConfIntervalWidget.confIntervalTextEdit.append(txt)
        # joint window: also feed the Summary tab (CI table + correlation heatmap)
        if getattr(self, '_mcmc_is_joint', False):
            self._update_joint_summary(names, cols, ci_rows, percentile)

    # ---- corner-plot zoom / pop-out (item 5, both windows) --------------- #
    def _add_corner_popout_button(self):
        """Add a 'Pop out' button to the corner-plot tab (both windows). The
        embedded N x N grid is too dense for the inline toolbar; a large resizable
        window makes matplotlib's box-zoom actually usable."""
        w = self.emceeConfIntervalWidget
        lay = w.cornerPlotTab.layout()
        if lay is None:
            return
        col = QWidget()
        vb = QVBoxLayout(col)
        vb.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton('Pop out ⤢')
        btn.setToolTip('Open the current corner plot in a large, zoomable window.')
        btn.setMaximumWidth(110)
        btn.clicked.connect(self.popOutCornerPlot)
        vb.addWidget(btn)
        vb.addStretch(1)
        lay.insertWidget(0, col)

    def popOutCornerPlot(self):
        """Render the LAST-drawn corner plot (same subset) into a standalone,
        freely-resizable window with matplotlib's navigation toolbar."""
        names = getattr(self, '_corner_names_cache', None)
        cols = getattr(self, '_corner_cols_cache', None)
        if not names or getattr(self, 'pardata', None) is None:
            QMessageBox.warning(self.emceeConfIntervalWidget, 'Corner plot',
                                'Generate a corner plot first '
                                '(Calculate Confidence Interval / Plot subset).', QMessageBox.Ok)
            return
        bins = getattr(self, '_corner_bins_cache', 30)
        percentile = getattr(self, '_corner_percentile_cache', 85)
        values = self._corner_truths(names, cols)
        dlg = QDialog(self.emceeConfIntervalWidget)
        dlg.setWindowTitle('Corner plot (zoomable) — %d parameters' % len(names))
        lay = QVBoxLayout(dlg)
        mw = MplWidget()
        lay.addWidget(mw)
        self._draw_corner(mw.fig, names, cols, values, bins, percentile)
        mw.draw()
        dlg.resize(1000, 900)
        dlg.setModal(False)
        dlg.show()
        # keep a reference so the dialog isn't garbage-collected while open
        self._cornerPopoutDlgs = getattr(self, '_cornerPopoutDlgs', [])
        self._cornerPopoutDlgs.append(dlg)

    # ---- joint-only MCMC visualization tabs (items 1 & 2) --------------- #
    _STRUCTURAL_PARAMS = ('theta', 'phi', 'protins', 'cov')

    def _is_structural_param(self, key):
        """Default-check heuristic for the joint subset list: a param is
        'structural/shared' if it is a bare (un-namespaced -> shared-across-all)
        master key, or its (de-namespaced) base name is a known structural param."""
        import re
        base = key
        m = re.match(r'^_{0,2}d\d+_(.*)$', key)         # strip d<idx>_ / __d<idx>_
        if m:
            base = m.group(1)
        if any(s in base for s in self._STRUCTURAL_PARAMS):
            return True
        return re.match(r'^_{0,2}d\d+_', key) is None   # not per-dataset -> shared

    def _augment_joint_mcmc_tabs(self):
        """Append the joint-only tabs (Param Subset + Summary) to the MCMC window.
        Called only for the joint window, so the individual window stays clean."""
        w = self.emceeConfIntervalWidget
        tabs = w.tabWidget

        # --- Param Subset tab (item 1) ---
        subTab = QWidget()
        sv = QVBoxLayout(subTab)
        sv.addWidget(QLabel(
            'Choose parameters for the corner plot / summary. Joint fits have many\n'
            'per-dataset params; plotting a subset keeps the corner plot legible.'))
        w.jointSubsetList = QListWidget()
        w.jointSubsetList.setSelectionMode(QAbstractItemView.NoSelection)
        sv.addWidget(w.jointSubsetList)
        row = QHBoxLayout()
        allBtn = QPushButton('Select all')
        allBtn.clicked.connect(lambda: self._set_joint_subset_checks('all'))
        structBtn = QPushButton('Structural / shared only')
        structBtn.clicked.connect(lambda: self._set_joint_subset_checks('structural'))
        noneBtn = QPushButton('Clear')
        noneBtn.clicked.connect(lambda: self._set_joint_subset_checks('none'))
        plotBtn = QPushButton('Plot subset ▸')
        plotBtn.clicked.connect(self._plot_joint_subset)
        for b in (allBtn, structBtn, noneBtn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(plotBtn)
        sv.addLayout(row)
        tabs.addTab(subTab, 'Param Subset')

        # --- Summary tab (item 2): correlation heatmap + median +/- CI table ---
        sumTab = QWidget()
        sm2 = QVBoxLayout(sumTab)
        sm2.addWidget(QLabel('Parameter correlation matrix (from the sampled subset):'))
        w.jointCorrMPLWidget = MplWidget()
        sm2.addWidget(w.jointCorrMPLWidget, 2)
        sm2.addWidget(QLabel('Median ± credible interval:'))
        w.jointCITable = QTableWidget(0, 5)
        w.jointCITable.setHorizontalHeaderLabels(
            ['Parameter', 'Fit-value', 'Median', '-err', '+err'])
        w.jointCITable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        w.jointCITable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        sm2.addWidget(w.jointCITable, 1)
        tabs.addTab(sumTab, 'Summary')

    def _populate_joint_subset_list(self):
        """Fill the joint subset list from param_chain (default-check structural/
        shared params so the first corner plot is legible)."""
        w = self.emceeConfIntervalWidget
        if not hasattr(w, 'jointSubsetList'):
            return
        w.jointSubsetList.clear()
        for key in self.param_chain.keys():
            if key == '__lnsigma':
                continue
            it = QListWidgetItem(key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if self._is_structural_param(key) else Qt.Unchecked)
            w.jointSubsetList.addItem(it)
        # never leave an empty default (would plot nothing): fall back to all
        n = w.jointSubsetList.count()
        if n and not any(w.jointSubsetList.item(i).checkState() == Qt.Checked for i in range(n)):
            self._set_joint_subset_checks('all')

    def _set_joint_subset_checks(self, mode):
        w = self.emceeConfIntervalWidget
        for i in range(w.jointSubsetList.count()):
            it = w.jointSubsetList.item(i)
            if mode == 'all':
                it.setCheckState(Qt.Checked)
            elif mode == 'none':
                it.setCheckState(Qt.Unchecked)
            elif mode == 'structural':
                it.setCheckState(Qt.Checked if self._is_structural_param(it.text()) else Qt.Unchecked)

    def _plot_joint_subset(self):
        w = self.emceeConfIntervalWidget
        subset = [w.jointSubsetList.item(i).text() for i in range(w.jointSubsetList.count())
                  if w.jointSubsetList.item(i).checkState() == Qt.Checked]
        if not subset:
            QMessageBox.warning(w, 'Corner plot', 'Check at least one parameter to plot.',
                                QMessageBox.Ok)
            return
        self.cornerPlot(subset=subset)

    def _update_joint_summary(self, names, cols, ci_rows, percentile):
        """Fill the joint Summary tab: median +/- CI table and the correlation
        heatmap over the currently plotted subset."""
        w = self.emceeConfIntervalWidget
        if hasattr(w, 'jointCITable'):
            t = w.jointCITable
            t.setRowCount(0)
            for (key, fitval, med, neg, pos) in ci_rows:
                r = t.rowCount()
                t.insertRow(r)
                t.setItem(r, 0, QTableWidgetItem(str(key)))
                t.setItem(r, 1, QTableWidgetItem('%.4g' % fitval))
                t.setItem(r, 2, QTableWidgetItem('%.4g' % med))
                t.setItem(r, 3, QTableWidgetItem('%.3g' % neg))
                t.setItem(r, 4, QTableWidgetItem('%.3g' % pos))
        if hasattr(w, 'jointCorrMPLWidget') and len(cols) >= 2:
            C = np.corrcoef(self.pardata[:, cols], rowvar=False)
            mw = w.jointCorrMPLWidget
            mw.clear()
            ax = mw.fig.add_subplot(1, 1, 1)
            im = ax.imshow(C, vmin=-1, vmax=1, cmap='coolwarm')
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_yticklabels(names, fontsize=6)
            mw.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            mw.fig.subplots_adjust(left=0.22, bottom=0.22, right=0.98, top=0.98)
            mw.draw()


    def parameterTreeSelectionChanged(self):
        self.emceeConfIntervalWidget.chainMPLWidget.clear()
        chaindata={}
        for item in self.emceeConfIntervalWidget.parameterTreeWidget.selectedItems():
            if ':chain:' in item.text(0):
                key,i=item.text(0).split(':chain:')
                try:
                    chaindata[key].append(int(i))
                except:
                    chaindata[key]=[int(i)]
        NRows = len(chaindata.keys())
        if NRows>0:
            ax={}
            firstkey=list(chaindata.keys())[0]
            for j,key in enumerate(chaindata.keys()):
                try:
                    ax[key]=self.emceeConfIntervalWidget.chainMPLWidget.fig.add_subplot(NRows, 1, j+1, sharex=ax[firstkey])
                except:
                    ax[key] = self.emceeConfIntervalWidget.chainMPLWidget.fig.add_subplot(NRows, 1, j+1)
                for i in chaindata[key]:
                    ax[key].plot(self.param_chain[key][i],'-')
                ax[key].set_xlabel('MC steps')
                ax[key].set_ylabel(key)
            self.emceeConfIntervalWidget.chainMPLWidget.draw()
            self.emceeConfIntervalWidget.tabWidget.setCurrentIndex(0)

    def userDefinedParameterTreeSelectionChanged(self):
        self.emceeConfIntervalWidget.userDefinedChainMPLWidget.clear()
        chaindata={}
        for item in self.emceeConfIntervalWidget.userDefinedParamTreeWidget.selectedItems():
            if ':chain:' in item.text(0):
                key,i=item.text(0).split(':chain:')
                try:
                    chaindata[key].append(int(i))
                except:
                    chaindata[key]=[int(i)]
        NRows = len(chaindata.keys())
        if NRows>0:
            ax={}
            firstkey=list(chaindata.keys())[0]
            for j,key in enumerate(chaindata.keys()):
                try:
                    ax[key]=self.emceeConfIntervalWidget.userDefinedChainMPLWidget.fig.add_subplot(NRows, 1, j+1, sharex=ax[firstkey])
                except:
                    ax[key] = self.emceeConfIntervalWidget.userDefinedChainMPLWidget.fig.add_subplot(NRows, 1, j+1)
                for i in chaindata[key]:
                    ax[key].plot(self.param_chain[key][i],'-')
                ax[key].set_xlabel('MC steps')
                ax[key].set_ylabel(key)
            self.emceeConfIntervalWidget.userDefinedChainMPLWidget.draw()
            self.emceeConfIntervalWidget.tabWidget.setCurrentIndex(1)

    def saveParameterError(self):
        fname=QFileDialog.getSaveFileName(caption='Save Parameter Errors as',filter='Parameter Error files (*.perr)',directory=self.curDir)[0]
        if os.path.splitext(fname)[1]=='':
            fname=fname+'.perr'
        text=self.emceeConfIntervalWidget.confIntervalTextEdit.toPlainText()
        fh=open(fname,'w')
        fh.write('# File save on %s\n'%time.asctime())
        fh.write('# Error calculated using MCMC Method\n')
        fh.write(text)
        fh.close()


    def undoFit(self):
        try:
            self.sfitParamTableWidget.cellChanged.disconnect()
            for i in range(self.mfitParamTabWidget.count()):
                mkey=self.mfitParamTabWidget.tabText(i)
                self.mfitParamTableWidget[mkey].cellChanged.disconnect()
        except:
            pass
        for row in range(self.sfitParamTableWidget.rowCount()):
            key=self.sfitParamTableWidget.item(row,0).text()
            self.sfitParamTableWidget.item(row,1).setText(self.format%(self.oldParams[key]))
            self.sfitParamTableWidget.item(row,1).setToolTip((key+' = '+self.format+' \u00B1 '+self.format)% (self.oldParams[key], 0.0))
        if self.fit.params['__mpar__']!={}:
            for i in range(self.mfitParamTabWidget.count()):
                mkey=self.mfitParamTabWidget.tabText(i)
                for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                    for col in range(1,self.mfitParamTableWidget[mkey].columnCount()):
                        parkey=self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                        key='__%s_%s_%03d'%(mkey,parkey,row)
                        self.mfitParamTableWidget[mkey].item(row,col).setText(self.format%(self.oldmpar[mkey][parkey][row]))
                        self.mfitParamTableWidget[mkey].item(row, col).setToolTip((key+' = '+self.format+' \u00B1 '+self.format) % \
                                                                        (self.oldmpar[mkey][parkey][row], 0.0))
            #self.mfitParamData=copy.copy(self.oldmpar)
        self.sfitParamTableWidget.cellChanged.connect(self.sfitParamChanged)
        for i in range(self.mfitParamTabWidget.count()):
            mkey = self.mfitParamTabWidget.tabText(i)
            self.mfitParamTableWidget[mkey].cellChanged.connect(self.mfitParamChanged_new)
        self.update_plot()


        
        
    def addData(self,fnames=None):
        """
        fnames        :List of filenames
        """
        if self.dataListWidget.count()==0:
            self.fileNumber=0
        try:
            self.dataListWidget.itemSelectionChanged.disconnect()
        except:
            pass
        #try:
        if fnames is None:
            tfnames,_=QFileDialog.getOpenFileNames(self,caption='Open data/macro files',directory=self.curDir,\
                                                  filter='Data/Macro files (*.txt *.dat *.chi *.rrf *.macro)')
        else:
            tfnames=copy.copy(fnames)
        fnames=[]
        for fname in tfnames:
            if os.path.splitext(fname)[1]=='.macro':
                fh=open(fname,'r')
                lines=fh.readlines()
                for line in lines:
                    tname=line.strip()
                    if os.path.exists(tname):
                        fnames.append(tname)
                    else:
                        QMessageBox.warning(self,'Path Error','The path' +tname+ 'doesnot exist. Exiting the import now.', QMessageBox.Ok)
                        return
            else:
                fnames.append(fname)
        if len(fnames)!=0:
            self.curDir=os.path.dirname(fnames[0])
            importProgressDlg = QProgressDialog('Importing multiple data...', 'Abort Import', -1, len(fnames) - 1,
                                                parent=self)
            if len(fnames)>1:
                importProgressDlg.setModal(True)
                importProgressDlg.forceShow()
            for fileNum, fname in enumerate(fnames):
                if importProgressDlg.wasCanceled():
                    importProgressDlg.setValue(len(fnames)-1)
                    break
                else:
                    importProgressDlg.setValue(fileNum)
                # else:
                #     importProgressDlg.setValue(0)
                #     importProgressDlg.close()

                data_key=str(self.fileNumber)+'<>'+fname
                data_dlg=Data_Dialog(fname=fname,parent=self)
                data_dlg.setModal(True)
                data_dlg.closePushButton.setText('Cancel')
                if len(fnames)>1:
                    data_dlg.accept()
                else:
                    #data_dlg.show()
                    data_dlg.exec_()
                if data_dlg.acceptData:
                    self.dlg_data[data_key]=data_dlg.data
                    self.plotColIndex[data_key]=data_dlg.plotColIndex
                    self.plotColors[data_key]=data_dlg.plotColors
                    self.data[data_key]=data_dlg.externalData
                    self.meta[data_key]=data_dlg.externalMeta
                    self.expressions[data_key]=data_dlg.expressions
                    for key in self.data[data_key].keys():
                        # deterministic palette color; data & its fit share it (see
                        # _curve_color_for). Stored back so the fit paths reuse it.
                        _col=self._curve_color_for(data_key,key)
                        self.data[data_key][key]['color']=_col
                        self.plotWidget.add_data(self.data[data_key][key]['x'],self.data[data_key][key]['y'],\
                                                 yerr=self.data[data_key][key]['yerr'],name='%d:%s'%(self.fileNumber,key),color=_col)
                    self.dataListWidget.addItem(data_key)
                    self.fileNames[self.fileNumber]=fname
                    self.xmin[data_key] = np.min([np.min(self.data[data_key][k1]['x']) for k1 in self.data[data_key].keys()])
                    self.xmax[data_key] = np.max([np.max(self.data[data_key][k1]['x']) for k1 in self.data[data_key].keys()])

                self.fileNumber+=1
                QApplication.processEvents()
            importProgressDlg.close()
            #     else:
            #         QMessageBox.warning(self,'Import Error','Data file has been imported before.\
            #          Please remove the data file before importing again')
            # #except:
            # #    QMessageBox.warning(self,'File error','The file(s) do(es) not look like a data file. Please format it in x,y[,yerr] column format',QMessageBox.Ok)
            self.metaDataInfoUpdated(data_key)
        self.dataListWidget.clearSelection()
        self.dataListWidget.itemSelectionChanged.connect(self.dataFileSelectionChanged)
        self.dataListWidget.setCurrentRow(self.fileNumber-1)
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)
        self.consoleWidget.push_vars({'data': self.dlg_data, 'plotColIndex':self.plotColIndex, 'plotColors':self.plotColors, 'expressions':self.expressions})

    def metaDataInfoUpdated(self,data_key):
        try:
            self.metaXAxisComboBox.currentIndexChanged.disconnect()
            self.metaYAxisComboBox.currentIndexChanged.disconnect()
            self.metaYErrAxisComboBox.currentIndexChanged.disconnect()
        except:
            pass
        self.metaXAxisComboBox.clear()
        self.metaYAxisComboBox.clear()
        self.metaYErrAxisComboBox.clear()
        self.metaXAxisComboBox.addItems(['None'] + list(self.meta[data_key].keys()))
        self.metaYAxisComboBox.addItems(['None'] + list(self.meta[data_key].keys()))
        self.metaYErrAxisComboBox.addItems(['None'] + list(self.meta[data_key].keys()))
        self.metaXAxisComboBox.currentIndexChanged.connect(self.metaDataChanged)
        self.metaYAxisComboBox.currentIndexChanged.connect(self.metaDataChanged)
        self.metaYErrAxisComboBox.currentIndexChanged.connect(self.metaDataChanged)

    def removeData(self):
        """
        """
        try:
            self.dataListWidget.itemSelectionChanged.disconnect()
        except:
            pass
        for item in self.dataListWidget.selectedItems():
            txt=item.text()
            fnum,fname=txt.split('<>')
            self.dataListWidget.takeItem(self.dataListWidget.row(item))
            for key in self.data[txt].keys():
                self.plotWidget.remove_data(['%s:%s'%(fnum,key)])
            del self.data[txt]
            del self.meta[txt]
            self.dataset_fits.pop(txt, None)   # drop this dataset's remembered fit
            self._cov_forget(txt)              # and its covariance
            if hasattr(self, 'jointFitPanel'):
                self.jointFitPanel.sync_available_datasets(list(self.data.keys()))
            del self.xmin[txt]
            del self.xmax[txt]
            self.limits.pop(txt, None)     # and any non-x column limits
            del self.expressions[txt]
            del self.plotColIndex[txt]
            del self.plotColors[txt]
            del self.dlg_data[txt]

        if self.dataListWidget.count()>0:
            self.dataFileSelectionChanged()
        else:
            self.pfnames=[]
        self.dataListWidget.itemSelectionChanged.connect(self.dataFileSelectionChanged)
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)
        self.consoleWidget.push_vars({'data': self.dlg_data, 'plotColIndex':self.plotColIndex, 'plotColors':self.plotColors, 'expressions':self.expressions})

            
        
        
    # ------------------------------------------------ covariance (single fits) ---
    def _cov_status(self, msg):
        """The data dock's covariance status line (this widget has no status bar)."""
        lab = getattr(self, 'covStatusLabel', None)
        if lab is not None:
            lab.setText(msg)
            lab.setVisible(bool(msg))

    def attach_covariance_to(self, data_key, path, raw_path=None):
        """Load the covariance at ``path`` (``raw_path``: a legacy *_c_factors.txt's raw
        data file) and attach it to dataset ``data_key``. Refuses, with a message, a
        multi-curve dataset, an unreadable file and any x/y mismatch; warns when the
        file was made for a differently named data file. Returns True if attached."""
        import covariance_core as cc
        name = os.path.basename(data_key.split('<>', 1)[-1])
        if data_key not in self.data:
            QMessageBox.warning(self, 'Covariance', 'No dataset %s is loaded.' % data_key, QMessageBox.Ok)
            return False
        if len(self.data[data_key].keys()) > 1:
            QMessageBox.warning(self, 'Covariance',
                                'Covariance not attached: %s is a multi-curve dataset (%s); a '
                                'covariance can only be attached to a single-curve dataset.'
                                % (name, ', '.join(self.data[data_key].keys())), QMessageBox.Ok)
            return False
        path = os.path.abspath(path)
        raw_path = os.path.abspath(raw_path) if raw_path else None
        try:
            cov = cc.load_covariance(path, raw_path=raw_path)
        except (cc.CovarianceError, OSError) as e:     # e.g. a legacy loader's missing raw file
            self._cov_refusal = str(e)
            QMessageBox.warning(self, 'Covariance', 'Covariance not attached to %s:\n%s' % (name, e),
                                QMessageBox.Ok)
            return False
        d = self.data[data_key][list(self.data[data_key].keys())[0]]
        probs = cov.match_problems(np.asarray(d['x'], float), np.asarray(d['y'], float))
        if cov.n != len(d['x']) and not probs:
            probs = ['covariance has %d points but the dataset has %d' % (cov.n, len(d['x']))]
        if probs:
            self._cov_refusal = 'it does not match the data: %s' % '; '.join(probs)
            QMessageBox.warning(self, 'Covariance',
                                'Covariance %s does not match dataset %s; not attached:\n%s'
                                % (os.path.basename(path), name, '\n'.join(probs)), QMessageBox.Ok)
            return False
        from Fit_Routines import _CovDowngrade
        _left = [w for w in self._cov_pending_notes.get(data_key, [])
                 if not isinstance(w, _CovDowngrade)]           # a reload failure is stale now
        if _left:
            self._cov_pending_notes[data_key] = _left
        else:
            self._cov_pending_notes.pop(data_key, None)
        self.covariances[data_key] = cov
        self.covariance_paths[data_key] = (path, raw_path)
        self.covariance_modes.setdefault(data_key, 'model')
        if cov.data_file and os.path.basename(cov.data_file) != name:
            msg = ('Covariance %s was made for data file %s but is attached to %s (x and y '
                   'match within rtol 1e-4).' % (os.path.basename(path),
                                                 os.path.basename(cov.data_file), name))
            self._cov_status(msg)
            QMessageBox.warning(self, 'Covariance', msg, QMessageBox.Ok)
        else:
            self._cov_status('Covariance %s attached to %s.' % (os.path.basename(path), name))
        self._refresh_data_labels()
        self._cov_sync_mode_combo()
        self._cov_sync_joint_panel()
        self._cov_plot_data(data_key)
        return True

    def _cov_plot_data(self, data_key):
        """Draw dataset ``data_key``'s Data curve as the data a single fit uses: a legacy
        covariance (replaces_y) is fitted on its y_corr with sigma_meas, so that is what
        is shown while it is attached; otherwise the dataset's own y (final review I1).
        Only touches a curve that a legacy covariance changed."""
        d = getattr(self, 'data', {}).get(data_key)
        if not d or len(d.keys()) != 1:
            return
        key = list(d.keys())[0]
        cov = getattr(self, 'covariances', {}).get(data_key)
        ry = getattr(cov, 'replaces_y', None)
        shown = getattr(self, '_cov_plotted_legacy', set())
        if ry is not None and len(ry) == len(d[key]['x']):
            y, yerr = np.asarray(ry, float), np.asarray(cov.sigma_meas, float)
            shown.add(data_key)
        elif data_key in shown:
            y, yerr = d[key]['y'], d[key]['yerr']
            shown.discard(data_key)
        else:
            return
        self._cov_plotted_legacy = shown
        color = d[key].get('color') or self._curve_color_for(data_key, key)
        self.plotWidget.add_data(d[key]['x'], y, yerr=yerr,
                                 name='%s:%s' % (data_key.split('<>')[0], key), color=color)

    def remove_covariance_from(self, data_key):
        """Detach dataset ``data_key``'s covariance (its next fit runs without one)."""
        had = data_key in self.covariances
        self._cov_forget(data_key)
        self._cov_plot_data(data_key)
        if had:
            self._cov_status('Covariance removed from %s.'
                             % os.path.basename(data_key.split('<>', 1)[-1]))
        self._refresh_data_labels()
        self._cov_sync_mode_combo()
        self._cov_sync_joint_panel()

    def _cov_sync_joint_panel(self):
        """Re-fill the joint panel's covTable from the attached covariances (spec 17
        6.6: a joint fit uses each dataset's attached covariance unless overridden)."""
        panel = getattr(self, 'jointFitPanel', None)
        if panel is not None and hasattr(panel, '_refresh_cov_table'):
            panel._refresh_cov_table()

    def _cov_forget(self, data_key):
        for d in (self.covariances, self.covariance_paths, self.covariance_modes,
                  self._cov_pending_notes):
            d.pop(data_key, None)

    def _load_covariance_dialog(self):
        sel = [it.text() for it in self.dataListWidget.selectedItems()]
        if len(sel) != 1:
            QMessageBox.warning(self, 'Covariance', 'Select one dataset to attach a covariance to.',
                                QMessageBox.Ok)
            return
        name = os.path.basename(sel[0].split('<>', 1)[-1])
        path = QFileDialog.getOpenFileName(self, caption='Load covariance for %s' % name,
                                           directory=self.curDir,
                                           filter='Covariance files (*.json *.npz *_c_factors.txt);;'
                                                  'All files (*)')[0]
        if not path:
            return
        raw = None
        if path.endswith('_c_factors.txt'):
            raw = QFileDialog.getOpenFileName(self, caption='Raw 4-column data file for %s'
                                              % os.path.basename(path),
                                              directory=os.path.dirname(path))[0] or None
        if self.attach_covariance_to(sel[0], path, raw_path=raw):
            self.update_plot()

    def _remove_covariance_selected(self):
        sel = [it.text() for it in self.dataListWidget.selectedItems()]
        for key in sel:
            self.remove_covariance_from(key)
        if sel:
            self.update_plot()

    def _cov_sync_mode_combo(self):
        """Show the selected dataset's covariance mode; disabled without a covariance."""
        cb = getattr(self, 'covModeComboBox', None)
        if cb is None:
            return
        key = self.sfnames[-1] if getattr(self, 'sfnames', None) else None
        has = key in self.covariances
        cb.blockSignals(True)
        cb.setCurrentText(self.covariance_modes.get(key, 'model') if has else 'model')
        cb.blockSignals(False)
        cb.setEnabled(has)

    def _cov_mode_changed(self, mode):
        key = self.sfnames[-1] if getattr(self, 'sfnames', None) else None
        if key not in self.covariances:
            return
        self.covariance_modes[key] = mode
        if getattr(self, 'fit', None) is not None and not hasattr(self.fit, 'objectives'):
            self.fit.covariance_mode = mode

    def create_paramDock(self):
        self.parSplitter=QSplitter(Qt.Vertical)

        self.fixedparamLayoutWidget=pg.LayoutWidget(self)

        # No-fit indicator banner (hidden until an unfit dataset is inspected within a
        # multi-selection -- see _set_detail_dataset). Sits above the parameter tables.
        self.noFitBanner=QLabel('')
        self.noFitBanner.setWordWrap(True)
        self.noFitBanner.setStyleSheet('color:#8a6d00;background:#fff3cd;'
                                       'border:1px solid #ffe08a;padding:3px;border-radius:3px;')
        self.noFitBanner.setVisible(False)
        self.fixedparamLayoutWidget.addWidget(self.noFitBanner,colspan=3)
        self.fixedparamLayoutWidget.nextRow()

        xlabel=QLabel('x')
        self.fixedparamLayoutWidget.addWidget(xlabel)
        self.xLineEdit=QLineEdit('np.linspace(0.001,1,100)')
        self.fixedparamLayoutWidget.addWidget(self.xLineEdit,col=1)
        self.saveSimulatedButton=QPushButton("Save Simulated Curve")
        self.saveSimulatedButton.setEnabled(False)
        self.saveSimulatedButton.clicked.connect(self.saveSimulatedCurve)
        self.fixedparamLayoutWidget.addWidget(self.saveSimulatedButton,col=2)

        self.fixedparamLayoutWidget.nextRow()
        self.saveParamButton = QPushButton('Save Parameters')
        self.saveParamButton.clicked.connect(lambda x: self.saveParameters(fname=None))
        self.fixedparamLayoutWidget.addWidget(self.saveParamButton,col=1)
        self.loadParamButton = QPushButton('Load Parameters')
        self.loadParamButton.clicked.connect(lambda x: self.loadParameters(fname=None))
        self.fixedparamLayoutWidget.addWidget(self.loadParamButton, col=2)
        
        self.fixedparamLayoutWidget.nextRow()
        fixedParamLabel=QLabel('Fixed Parameters')
        self.fixedparamLayoutWidget.addWidget(fixedParamLabel, colspan=2)
        self.autoCalcCheckBox=QCheckBox('Auto Calculate')
        self.fixedparamLayoutWidget.addWidget(self.autoCalcCheckBox)
        self.autoCalcCheckBox.setChecked(True)
        self.autoCalcCheckBox.stateChanged.connect(self.changeAutoCalc)

        self.fixedparamLayoutWidget.nextRow()
        self.fixedParamTableWidget=pg.TableWidget()
        self.fixedParamTableWidget.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.fixedParamTableWidget.setEditable(editable=True)
        self.fixedParamTableWidget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.fixedparamLayoutWidget.addWidget(self.fixedParamTableWidget,colspan=3)
        
        self.parSplitter.addWidget(self.fixedparamLayoutWidget)
        
        self.sfitparamLayoutWidget=pg.LayoutWidget()
        sfitParamLabel=QLabel('Single fitting parameters')
        self.sfitparamLayoutWidget.addWidget(sfitParamLabel)
        
        self.sfitparamLayoutWidget.nextRow()
        self.sfitParamTableWidget=pg.TableWidget()
        self.sfitParamTableWidget.setEditable(editable=True)
        self.sfitParamTableWidget.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.sfitParamTableWidget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        #self.sfitParamTableWidget.cellDoubleClicked.connect(self.editFitParam)
        self.sfitparamLayoutWidget.addWidget(self.sfitParamTableWidget,colspan=3)
        self.sfitparamLayoutWidget.nextRow()
        self.sfitLabel=QLabel('')
        self.sfitSlider=QSlider(Qt.Horizontal)
        self.sfitSlider.setMinimum(1)
        self.sfitSlider.setMaximum(1000)
        self.sfitSlider.setSingleStep(10)
        self.sfitSlider.setTickInterval(10)
        self.sfitSlider.setValue(500)
        self.sfitparamLayoutWidget.addWidget(self.sfitLabel,col=0,colspan=1)
        self.sfitparamLayoutWidget.addWidget(self.sfitSlider,col=1,colspan=2)
        self.sfitParamTableWidget.cellClicked.connect(self.update_sfitSlider)

        self.parSplitter.addWidget(self.sfitparamLayoutWidget)
        
        self.mfitparamLayoutWidget=pg.LayoutWidget()
        mfitParamLabel=QLabel('Mutiple fitting parameters')
        self.mfitparamLayoutWidget.addWidget(mfitParamLabel,col=0, colspan=3)

        self.mfitparamLayoutWidget.nextRow()
        self.mfitParamCoupledCheckBox=QCheckBox('Coupled')
        self.mfitParamCoupledCheckBox.setEnabled(False)
        self.mfitParamCoupledCheckBox.stateChanged.connect(self.mfitParamCoupledCheckBoxChanged)
        self.mfitparamLayoutWidget.addWidget(self.mfitParamCoupledCheckBox,col=0)
        self.add_mpar_button=QPushButton('Add')
        self.add_mpar_button.clicked.connect(self.add_mpar)
        self.add_mpar_button.setDisabled(True)
        self.mfitparamLayoutWidget.addWidget(self.add_mpar_button,col=1)
        self.remove_mpar_button=QPushButton('Remove')
        self.mfitparamLayoutWidget.addWidget(self.remove_mpar_button,col=2)      
        self.remove_mpar_button.clicked.connect(self.remove_mpar)
        self.remove_mpar_button.setDisabled(True)
        
        self.mfitparamLayoutWidget.nextRow()
        self.mfitParamTabWidget=QTabWidget()
        self.mfitParamTabWidget.currentChanged.connect(self.mfitParamTabChanged)
        # self.mfitParamTableWidget=pg.TableWidget(sortable=False)
        # self.mfitParamTableWidget.cellDoubleClicked.connect(self.mparDoubleClicked)
        # self.mfitParamTableWidget.setEditable(editable=True)
        # self.mfitParamTableWidget.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        # self.mfitParamTableWidget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        # #self.sfitParamTableWidget.cellDoubleClicked.connect(self.editFitParam)
        # self.mfitparamLayoutWidget.addWidget(self.mfitParamTableWidget,colspan=3)
        self.mfitparamLayoutWidget.addWidget(self.mfitParamTabWidget,colspan=3)
        self.mfitparamLayoutWidget.nextRow()
        self.mfitLabel=QLabel('')
        self.mfitSlider=QSlider(Qt.Horizontal)
        self.mfitSlider.setMinimum(1)
        self.mfitSlider.setSingleStep(10)
        self.mfitSlider.setTickInterval(10)
        self.mfitSlider.setMaximum(1000)
        self.mfitSlider.setValue(500)
        self.mfitparamLayoutWidget.addWidget(self.mfitLabel,col=0,colspan=1)
        self.mfitparamLayoutWidget.addWidget(self.mfitSlider,col=1,colspan=2)
        # self.mfitParamTableWidget.cellClicked.connect(self.update_mfitSlider)
        
        # self.mfitparamLayoutWidget.nextRow()
        # self.saveParamButton=QPushButton('Save Parameters')
        # self.saveParamButton.clicked.connect(self.saveParameters)
        # self.mfitparamLayoutWidget.addWidget(self.saveParamButton,col=1)
        # self.loadParamButton=QPushButton('Load Parameters')
        # self.loadParamButton.clicked.connect(lambda x: self.loadParameters(fname=None))
        # self.mfitparamLayoutWidget.addWidget(self.loadParamButton,col=2)
        self.parSplitter.addWidget(self.mfitparamLayoutWidget)

        self.genparamLayoutWidget=pg.LayoutWidget()
        genParameters=QLabel('Generated Parameters')
        self.autoSaveGenParamCheckBox=QCheckBox('Auto Save Selected')
        self.genparamLayoutWidget.addWidget(genParameters,col=0,colspan=2)
        self.genparamLayoutWidget.addWidget(self.autoSaveGenParamCheckBox,col=1)
        self.genparamLayoutWidget.nextRow()
        self.genParamListWidget=QListWidget()
        self.genParamListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.genParamListWidget.itemSelectionChanged.connect(self.plot_extra_param)
        self.genParamListWidget.itemDoubleClicked.connect(self.extra_param_doubleClicked)
        #self.genParamListWidget.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.genparamLayoutWidget.addWidget(self.genParamListWidget,colspan=2)
        self.genparamLayoutWidget.nextRow()
        self.saveGenParamButton=QPushButton('Save Generated Parameters')
        self.saveGenParamButton.clicked.connect(lambda x:self.saveGenParameters(bfname=None))
        self.genparamLayoutWidget.addWidget(self.saveGenParamButton,colspan=2)

        self.parSplitter.addWidget(self.genparamLayoutWidget)
        
        self.paramDock.addWidget(self.parSplitter)


    def changeAutoCalc(self):
        if self.autoCalcCheckBox.isChecked():
            self.autoCalculate=True
            self.update_plot()
        else:
            self.autoCalculate=False

    def mfitParamTabChanged(self,index):
        self.mkey=self.mfitParamTabWidget.tabText(index)
        if self.mkey!='':
            if self.mfitParamTableWidget[self.mkey].rowCount()==self.mpar_N[self.mkey]:
                self.remove_mpar_button.setDisabled(True)
            else:
                self.remove_mpar_button.setEnabled(True)


    def update_sfitSlider(self,row,col):
        if col==1:
            try:
                self.sfitSlider.valueChanged.disconnect()
                self.sfitSlider.sliderReleased.disconnect()
            except:
                pass
            key=self.sfitParamTableWidget.item(row,0).text()
            self.sfitLabel.setText(key)
            self.current_sfit_row=row
            value=self.fit.fit_params[key].value
            self.sfitSlider.setValue(500)
            self.sfitSlider.valueChanged.connect(self.sfitSliderChanged)
            self.sfitSlider.sliderReleased.connect(self.sfitSliderReleased)

    def sfitSliderChanged(self,value):
        if not self.sfitSlider.isSliderDown():
            self.sfitSlider.setDisabled(True)
            key=self.sfitParamTableWidget.item(self.current_sfit_row,0).text()
            pvalue=self.fit.fit_params[key].value+self.fit.fit_params[key].brute_step*(value-500)/500
            self.sfitParamTableWidget.item(self.current_sfit_row,1).setText(self.format%pvalue)
            QApplication.processEvents()
            self.sfitSlider.setEnabled(True)

    def sfitSliderReleased(self):
        key=self.sfitParamTableWidget.item(self.current_sfit_row,0).text()
        pvalue=self.fit.fit_params[key].value*(1+0.2*(self.sfitSlider.value()-500)/500)
        self.sfitParamTableWidget.item(self.current_sfit_row,1).setText(self.format%pvalue)
        QApplication.processEvents()

    def update_mfitSlider(self,row,col):
        if col!=0:
            try:
                self.mfitSlider.valueChanged.disconnect()
                self.mfitSlider.sliderReleased.disconnect()
            except:
                pass
            pkey = self.mfitParamTableWidget[self.mkey].horizontalHeaderItem(col).text()
            txt = self.mfitParamTableWidget[self.mkey].item(row, col).text()
            key = '__%s_%s_%03d' % (self.mkey, pkey, row)
            self.mfitLabel.setText(key)
            self.current_mfit_row=row
            self.current_mfit_col=col
            value=self.fit.fit_params[key].value
            self.mfitSlider.setValue(500)
            self.mfitSlider.valueChanged.connect(self.mfitSliderChanged)
            self.mfitSlider.sliderReleased.connect(self.mfitSliderReleased)

    def mfitSliderChanged(self,value):
        if not self.mfitSlider.isSliderDown():
            self.mfitSlider.setDisabled(True)
            pkey = self.mfitParamTableWidget[self.mkey].horizontalHeaderItem(self.current_mfit_col).text()
            txt = self.mfitParamTableWidget[self.mkey].item(self.current_mfit_row, self.current_mfit_col).text()
            key = '__%s_%s_%03d' % (self.mkey, pkey, self.current_mfit_row)
            pvalue=self.fit.fit_params[key].value+self.fit.fit_params[key].brute_step*(value-500)/500
            self.mfitParamTableWidget[self.mkey].item(self.current_mfit_row,self.current_mfit_col).setText(self.format%pvalue)
            QApplication.processEvents()
            self.mfitSlider.setEnabled(True)

    def mfitSliderReleased(self):
        pkey = self.mfitParamTableWidget[self.mkey].horizontalHeaderItem(self.current_mfit_col).text()
        txt = self.mfitParamTableWidget[self.mkey].item(self.current_mfit_row, self.current_mfit_col).text()
        key = '__%s_%s_%03d' % (self.mkey, pkey, self.current_mfit_row)
        pvalue = self.fit.fit_params[key].value * (1 + 0.2 * (self.mfitSlider.value() - 500) / 500)
        self.mfitParamTableWidget[self.mkey].item(self.current_mfit_row, self.current_mfit_col).setText(self.format % pvalue)
        QApplication.processEvents()


    def saveSimulatedCurve(self):
        """
        Saves the simulated curve in a user-supplied ascii file
        :return:
        """
        fname=QFileDialog.getSaveFileName(caption='Save As',filter='Text files (*.dat *.txt)',directory=self.curDir)[0]
        if fname!='':
            header='Simulated curve generated on %s\n'%time.asctime()
            header+='Category:%s\n'%self.curr_category
            header+='Function:%s\n'%self.funcListWidget.currentItem().text()
            for i in range(self.fixedParamTableWidget.rowCount()):
                header += '%s=%s\n' % (
                self.fixedParamTableWidget.item(i, 0).text(), self.fixedParamTableWidget.item(i, 1).text())
            for i in range(self.sfitParamTableWidget.rowCount()):
                header += '%s=%s\n' % (
                self.sfitParamTableWidget.item(i, 0).text(), self.sfitParamTableWidget.item(i, 1).text())
            for i in range(self.mfitParamTabWidget.count()):
                mkey = self.mfitParamTabWidget.tabText(i)
                for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                    vartxt = mkey+'_'+self.mfitParamTableWidget[mkey].item(row, 0).text()
                    for col in range(1, self.mfitParamTableWidget[mkey].columnCount()):
                        header += '%s_%s=%s\n' % (vartxt, self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text(),
                                              self.mfitParamTableWidget[mkey].item(row, col).text())
            if type(self.fit.x)==dict:
                text='col_names=[\'q\','
                keys=list(self.fit.x.keys())
                data=self.fit.x[keys[0]]
                for key in keys:
                    text+='\''+key+'\','
                    data=np.vstack((data,self.fit.yfit[key]))
                header+=text[:-1]+']\n'
                np.savetxt(fname,data.T,header=header,comments='#')
            else:
                header+='col_names=[\'q\',\'I\']'
                np.savetxt(fname,np.vstack((self.fit.x,self.fit.yfit)).T,header=header,comments='#')
        else:
            pass

        
    def mparDoubleClicked(self,row,col):
        mkey=self.mfitParamTabWidget.tabText(self.mfitParamTabWidget.currentIndex())
        if col!=0:
            try:
                self.mfitParamTableWidget[mkey].cellChanged.disconnect()
            except:
                pass
            pkey=self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
            key='__%s_%s_%03d'%(mkey,pkey,row)
            ovalue=self.fit.fit_params[key].value
            ovary=self.fit.fit_params[key].vary
            ominimum=self.fit.fit_params[key].min
            omaximum=self.fit.fit_params[key].max
            oexpr=self.fit.fit_params[key].expr
            obrute_step=self.fit.fit_params[key].brute_step
            dlg=minMaxDialog(ovalue,vary=ovary,minimum=ominimum,maximum=omaximum,expr=oexpr,brute_step=obrute_step,title=key)
            if dlg.exec_():
                value,vary,maximum,minimum,expr,brute_step=(dlg.value,dlg.vary,dlg.maximum,dlg.minimum,dlg.expr,dlg.brute_step)
            else:
                value,vary,maximum,minimum,expr,brute_step=copy.copy(ovalue),copy.copy(ovary),copy.copy(omaximum),copy.copy(ominimum),copy.copy(oexpr),copy.copy(obrute_step)
            self.mfitParamTableWidget[mkey].item(row,col).setText(self.format%value)
            if vary:
                self.mfitParamTableWidget[mkey].item(row, col).setCheckState(Qt.Checked)
            else:
                self.mfitParamTableWidget[mkey].item(row, col).setCheckState(Qt.Unchecked)

            try:
                self.mfitParamData[mkey][pkey][row] = value
                # self.fit.fit_params[key].set(value=value)
                if expr == 'None':
                    expr = ''
                self.fit.fit_params[key].set(value=value, vary=vary, min=minimum, max=maximum, expr=expr,
                                             brute_step=brute_step)
                self.update_plot()
            except:
                self.mfitParamTableWidget[mkey].item(row, col).setText(self.format % ovalue)
                self.mfitParamData[mkey][pkey][row] = ovalue
                self.fit.fit_params[key].set(value=ovalue, vary=ovary, min=ominimum, max=omaximum, expr=oexpr,
                                             brute_step=brute_step)
                self.update_plot()
                QMessageBox.warning(self,'Parameter Error','Some parameter value you just entered are not correct. Please enter the values carefully',QMessageBox.Ok)

            self.mfitParamTableWidget[mkey].cellChanged.connect(self.mfitParamChanged_new)
            self.errorAvailable = False
            self.reuse_sampler = False
            self.calcConfInterButton.setDisabled(True)


    def mfitParamCoupledCheckBoxChanged(self):
        if self.mfitParamCoupledCheckBox.isChecked() and self.mfitParamTabWidget.count()>1:
            mparRowCounts=[self.mfitParamTableWidget[self.mfitParamTabWidget.tabText(i)].rowCount() for i in range(self.mfitParamTabWidget.count())]
            if not all(x == mparRowCounts[0] for x in mparRowCounts):
                cur_index=self.mfitParamTabWidget.currentIndex()
                cur_key=self.mfitParamTabWidget.tabText(cur_index)
                for i in range(self.mfitParamTabWidget.count()):
                    if i != cur_index:
                        mkey=self.mfitParamTabWidget.tabText(i)
                        try:
                            self.mfitParamTableWidget[mkey].cellChanged.disconnect()
                        except:
                            pass
                        rowCount=self.mfitParamTableWidget[mkey].rowCount()
                        self.mfitParamTabWidget.setCurrentIndex(i)
                        if rowCount>mparRowCounts[cur_index]:
                            self.mfitParamTableWidget[mkey].clearSelection()
                            self.mfitParamTableWidget[mkey].setRangeSelected(
                                QTableWidgetSelectionRange(mparRowCounts[cur_index],0,rowCount-1,0),True)
                            self.remove_uncoupled_mpar()
                        elif rowCount<mparRowCounts[cur_index]:
                            for j in range(rowCount,mparRowCounts[cur_index]):
                                self.mfitParamTableWidget[mkey].clearSelection()
                                self.mfitParamTableWidget[mkey].setCurrentCell(j-1,0)
                                self.add_uncoupled_mpar()
                        self.mfitParamTableWidget[mkey].setSelectionBehavior(QAbstractItemView.SelectItems)
                self.mfitParamTabWidget.setCurrentIndex(cur_index)
                self.errorAvailable = False
                self.reuse_sampler = False
                self.calcConfInterButton.setDisabled(True)

    def add_mpar(self):
        if self.mfitParamCoupledCheckBox.isChecked() and self.mfitParamTabWidget.count()>1:
            self.add_coupled_mpar()
        else:
            self.add_uncoupled_mpar()
        self.update_plot()
        self.remove_mpar_button.setEnabled(True)
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)

    def remove_mpar(self):
        if self.mfitParamCoupledCheckBox.isChecked() and self.mfitParamTabWidget.count()>1:
            self.remove_coupled_mpar()
        else:
            self.remove_uncoupled_mpar()
        self.update_plot()
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)

    def add_coupled_mpar(self):
        cur_index=self.mfitParamTabWidget.currentIndex()
        mkey = self.mfitParamTabWidget.tabText(cur_index)
        if len(self.mfitParamTableWidget[mkey].selectedItems())!=0:
            curRow=self.mfitParamTableWidget[mkey].currentRow()
            for i in range(self.mfitParamTabWidget.count()):
                self.mfitParamTabWidget.setCurrentIndex(i)
                tkey=self.mfitParamTabWidget.tabText(i)
                self.mfitParamTableWidget[tkey].clearSelection()
                self.mfitParamTableWidget[tkey].setCurrentCell(curRow,0)
                self.add_uncoupled_mpar()
        self.mfitParamTabWidget.setCurrentIndex(cur_index)
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)

    def remove_coupled_mpar(self):
        cur_index=self.mfitParamTabWidget.currentIndex()
        mkey = self.mfitParamTabWidget.tabText(cur_index)
        selRows = list(set([item.row() for item in self.mfitParamTableWidget[mkey].selectedItems()]))
        if len(selRows) != 0:
            for i in range(self.mfitParamTabWidget.count()):
                self.mfitParamTabWidget.setCurrentIndex(i)
                tkey=self.mfitParamTabWidget.tabText(i)
                self.mfitParamTableWidget[tkey].clearSelection()
                self.mfitParamTableWidget[tkey].setRangeSelected(
                    QTableWidgetSelectionRange(selRows[0], 0, selRows[-1], 0), True)
                self.remove_uncoupled_mpar()
        self.mfitParamTabWidget.setCurrentIndex(cur_index)
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)
        
    def add_uncoupled_mpar(self):
        cur_index = self.mfitParamTabWidget.currentIndex()
        mkey=self.mfitParamTabWidget.tabText(self.mfitParamTabWidget.currentIndex())
        try:
            self.mfitParamTableWidget[mkey].cellChanged.disconnect()
        except:
            pass
        NCols=self.mfitParamTableWidget[mkey].columnCount()
        if len(self.mfitParamTableWidget[mkey].selectedItems())!=0:
            curRow=self.mfitParamTableWidget[mkey].currentRow()
            #if curRow!=0:
            self.mfitParamTableWidget[mkey].insertRow(curRow)
            self.mfitParamTableWidget[mkey].setRow(curRow,self.mfitParamData[mkey][curRow])
            self.mfitParamData[mkey]=np.insert(self.mfitParamData[mkey],curRow,self.mfitParamData[mkey][curRow],0)
            NRows = self.mfitParamTableWidget[mkey].rowCount()
            for col in range(NCols):
                pkey=self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                if col!=0:
                    for row in range(NRows-1, curRow,-1):
                        key='__%s_%s_%03d'%(mkey, pkey,row)
                        nkey = '__%s_%s_%03d' % (mkey,pkey,row-1)
                        if key in self.fit.fit_params.keys():
                            val,vary,min,max,expr,bs = self.mfitParamData[mkey][row][col],self.fit.fit_params[nkey].vary, \
                                                      self.fit.fit_params[nkey].min,self.fit.fit_params[nkey].max, \
                                                      self.fit.fit_params[nkey].expr,self.fit.fit_params[nkey].brute_step
                            self.fit.fit_params[key].set(value=val,vary=vary,min=min,max=max,expr=expr,brute_step=bs)
                        else:
                            val,vary,min,max,expr,bs=self.mfitParamData[mkey][row][col],self.fit.fit_params[nkey].vary,self.fit.fit_params[nkey].min, \
                                                 self.fit.fit_params[nkey].max,self.fit.fit_params[nkey].expr, \
                                                 self.fit.fit_params[nkey].brute_step
                            self.fit.fit_params.add(key,value=val,vary=vary,min=min,max=max,expr=expr,brute_step=bs)
                        item=self.mfitParamTableWidget[mkey].item(row,col)
                        item.setText(self.format%val)
                        item.setFlags(
                            Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                        if self.fit.fit_params[key].vary > 0:
                            item.setCheckState(Qt.Checked)
                        else:
                            item.setCheckState(Qt.Unchecked)
                        item.setToolTip((key+' = '+self.format+' \u00B1 '+self.format) % \
                                                                (self.fit.fit_params[key].value, 0.0))
                    # This is to make the newly inserted row checkable
                    item = self.mfitParamTableWidget[mkey].item(curRow, col)
                    item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                    key = '__%s_%s_%03d' % (mkey, pkey, curRow)
                    item.setText(self.format%self.fit.fit_params[key].value)
                    item.setToolTip((key + ' = ' + self.format + ' \u00B1 ' + self.format) % \
                                    (self.fit.fit_params[key].value, 0.0))
                    if self.fit.fit_params[key].vary>0:
                        item.setCheckState(Qt.Checked)
                    else:
                        item.setCheckState(Qt.Unchecked)
                else:
                    item = self.mfitParamTableWidget[mkey].item(curRow, col)
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                self.fit.params['__mpar__'][mkey][pkey].insert(curRow, self.mfitParamData[mkey][curRow][col])
            self.update_mfit_parameters_new()
            #self.update_plot()
            self.errorAvailable = False
            self.reuse_sampler = False
            self.calcConfInterButton.setDisabled(True)
            # self.remove_mpar_button.setEnabled(True)
            self.mfitParamTabWidget.setCurrentIndex(cur_index)
        else:
            QMessageBox.warning(self,'Warning','Please select a row at which you would like to add a set of parameters',QMessageBox.Ok)
        self.mfitParamTableWidget[mkey].cellChanged.connect(self.mfitParamChanged_new)
            
    def remove_uncoupled_mpar(self):
        mkey = self.mfitParamTabWidget.tabText(self.mfitParamTabWidget.currentIndex())
        selrows=list(set([item.row() for item in self.mfitParamTableWidget[mkey].selectedItems()]))
        num=self.mfitParamTableWidget[mkey].rowCount()-len(selrows)
        if num<self.mpar_N[mkey]:
            QMessageBox.warning(self,'Selection error','The minimum number of rows required for this function to work is %d.\
             You can only remove %d rows'%(self.mpar_N[mkey],num),QMessageBox.Ok)
            return
        # if self.mfitParamTableWidget[mkey].rowCount()-1 in selrows:
        #     QMessageBox.warning(self, 'Selection error',
        #                         'Cannot remove the last row. Please select the rows other than the last row', QMessageBox.Ok)
        #     return
        try:
            self.mfitParamTableWidget[mkey].cellChanged.disconnect()
        except:
            pass
        if selrows!=[]:
            selrows.sort(reverse=True)
            for row in selrows:
                maxrow=self.mfitParamTableWidget[mkey].rowCount()
                for trow in range(row,maxrow):
                    for col in range(self.mfitParamTableWidget[mkey].columnCount()):
                        pkey=self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                        if trow<maxrow-1:
                            key1='__%s_%s_%03d'%(mkey,pkey,trow)
                            key2='__%s_%s_%03d'%(mkey,pkey,trow+1)
                            self.fit.params['__mpar__'][mkey][pkey][trow] = copy.copy(self.fit.params['__mpar__'][mkey][pkey][trow + 1])
                            if col!=0:
                                self.fit.fit_params[key1]=copy.copy(self.fit.fit_params[key2])
                                del self.fit.fit_params[key2]
                        else:
                            key1='__%s_%s_%03d'%(mkey,pkey,trow)
                            # if col!=0:
                            del self.fit.params['__mpar__'][mkey][pkey][trow]
                                # del self.fit.fit_params[key1]
                self.mfitParamTableWidget[mkey].removeRow(row)
                self.mfitParamData[mkey]=np.delete(self.mfitParamData[mkey],row,axis=0)
            #updating the tooltips after removal of rows
            for col in range(1,self.mfitParamTableWidget[mkey].columnCount()):
                pkey = self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                    item=self.mfitParamTableWidget[mkey].item(row, col)
                    key = '__%s_%s_%03d' % (mkey, pkey, row)
                    item.setToolTip((key + ' = ' + self.format + ' \u00B1 ' + self.format) % \
                        (self.fit.fit_params[key].value, 0.0))
        else:
            QMessageBox.warning(self,'Nothing selected','No item is selected for removal',QMessageBox.Ok)
        self.mfitParamTableWidget[mkey].cellChanged.connect(self.mfitParamChanged_new)
        self.fit.func.output_params={'scaler_parameters': {}}
        self.update_plot()
        if self.mfitParamTableWidget[mkey].rowCount()==self.mpar_N[mkey]:
            self.remove_mpar_button.setDisabled(True)
        self.errorAvailable = False
        self.reuse_sampler = False
        self.calcConfInterButton.setDisabled(True)
            
        
    def saveGenParameters(self,bfname=None):
        # if len(self.genParamListWidget.selectedItems())==1:
        if bfname is None:
            bfname = QFileDialog.getSaveFileName(self, 'Provide the prefix of the generated files',self.curDir)[0]
        if bfname!='':
            bfname=os.path.splitext(bfname)[0]
        else:
            return
        selParams=self.genParamListWidget.selectedItems()
        for params in selParams:
            text=params.text()
            parname,var=text.split(' : ')
            fname=bfname+'_'+parname+'.txt'
            # if fname!='':
            #     if fname[-4:]!='.txt':
            #         fname=fname+'.txt'
            header='Generated output file on %s\n'%time.asctime()
            header += 'Category=%s\n' % self.curr_category
            header += 'Function=%s\n' % self.funcListWidget.currentItem().text()
            added_par=[]
            for i in range(self.fixedParamTableWidget.rowCount()):
                par, val = self.fixedParamTableWidget.item(i, 0).text(), self.fixedParamTableWidget.item(i, 1).text()
                if 'meta' in self.fit.params['output_params'][parname]:
                    if par in self.fit.params['output_params'][parname]['meta'].keys():
                        header += '%s=%s\n' % (par, str(self.fit.params['output_params'][parname]['meta'][par]))
                        added_par.append(par)
                    else:
                        header += '%s=%s\n' % (par, val)
                else:
                    header+='%s=%s\n'%(par,val)
            if 'meta' in self.fit.params['output_params'][parname]:
                for metakey in self.fit.params['output_params'][parname]['meta'].keys():
                    if metakey not in added_par:
                        header+='%s=%s\n'%(metakey,str(self.fit.params['output_params'][parname]['meta'][metakey]))
            # for fi in range(self.fixedParamTableWidget.rowCount()):
            #     par,val=self.fixedParamTableWidget.item(fi,0).text(),self.fixedParamTableWidget.item(fi,1).text()
            #     header+='%s=%s\n'%(par,val)
            for i in range(self.sfitParamTableWidget.rowCount()):
                par,val=self.sfitParamTableWidget.item(i,0).text(),self.sfitParamTableWidget.item(i,1).text()
                header+='%s=%s\n'%(par,val)
            for k in range(self.mfitParamTabWidget.count()):
                mkey=self.mfitParamTabWidget.tabText(k)
                for i in range(self.mfitParamTableWidget[mkey].rowCount()):
                    vartxt=self.mfitParamTableWidget[mkey].item(i,0).text()
                    for j in range(1,self.mfitParamTableWidget[mkey].columnCount()):
                        header+='%s_%s=%s\n'%(vartxt,self.mfitParamTableWidget[mkey].horizontalHeaderItem(j).text(),
                                              self.mfitParamTableWidget[mkey].item(i,j).text())

            if 'names' in self.fit.params['output_params'][parname]:
                header += "col_names=%s\n" % str(self.fit.params['output_params'][parname]['names'])
            else:
                lvar=eval(var)
                if 'meta' in lvar:
                    lvar.remove('meta')
                header += "col_names=%s\n" % str(lvar)

            header=header.encode("ascii","ignore")
            header=header.decode()
            if var=="['x', 'y', 'meta']" or var == "['x', 'y']":
                header+='x\ty\n'
                res=np.vstack((self.fit.params['output_params'][parname]['x'], self.fit.params['output_params'][parname]['y'])).T
                np.savetxt(fname,res,header=header,comments='#')
            elif var=="['x', 'y', 'yerr', 'meta']" or var=="['x', 'y', 'yerr']":
                header+='x\ty\tyerr\n'
                res=np.vstack((self.fit.params['output_params'][parname]['x'], self.fit.params['output_params'][parname]['y'],self.fit.params['output_params'][parname]['yerr'])).T
                np.savetxt(fname,res,header=header,comments='#')
            elif self._op_is_image(self.fit.params['output_params'][parname]):
                # A 2-D map entry: x and y are the two 1-D AXES and z is (nx, ny).
                # The legacy branch below assumes x is itself 2-D (it does
                # x.shape[1]), so without this an image entry raises IndexError out
                # of this slot, which has no try/except around it.
                e=self.fit.params['output_params'][parname]
                X,Y=np.meshgrid(np.asarray(e['x']),np.asarray(e['y']),indexing='ij')
                header+='x\ty\tz\n'
                np.savetxt(fname,np.c_[X.ravel(),Y.ravel(),np.asarray(e['z']).ravel()],
                           header=header,comments='#')
            elif var=="['x', 'y', 'z', 'meta']" or var=="['x', 'y', 'z']":
                res=[]
                header+='x\ty\tz\n'
                for i in range(self.fit.params['output_params'][parname]['x'].shape[1]):
                    for j in range(self.fit.params['output_params'][parname]['x'].shape[0]):
                        res.append([self.fit.params['output_params'][parname][t][i,j] for t in ['x','y','z']])
                res=np.array(res)
                np.savetxt(fname,res,header=header,comments='#')
            else:
                QMessageBox.warning(self,'Format error','The data is in some different format and couldnot be saved.',QMessageBox.Ok)
        # else:
        #     QMessageBox.warning(self,'Selection Error','Please select a single generated data to be saved.',QMessageBox.Ok)
        
        
    def saveParameters(self, fname=None):
        """
        Saves all the fixed and fitted parameteres in a file
        """
        if fname is None:
            fname=QFileDialog.getSaveFileName(self,caption='Save parameters as',directory=self.curDir,filter='Parameter files (*.par)')[0]
        if fname!='':
            if fname[-5:]!='.xpar':
                fname=fname+'.xpar'
            fh=open(fname,'w')
            fh.write('#File saved on %s\n'%time.asctime())
            fh.write('#Category: %s\n'%self.categoryListWidget.currentItem().text())
            fh.write('#Function: %s\n'%self.funcListWidget.currentItem().text())
            fh.write('#Xrange=%s\n'%self.xLineEdit.text())
            fh.write('#Fit Range=%s\n'%self.xminmaxLineEdit.text())
            _cl=self._format_column_limits()
            if _cl:                    # only when a non-x limit exists, so
                fh.write('#Column Limits=%s\n'%_cl)   # old files stay identical
            for _cvl in self._covariance_header_lines():   # likewise only with a covariance
                fh.write('#%s\n'%_cvl)
            fh.write('#Fit Method=%s\n'%self.fitMethodComboBox.currentText())
            fh.write('#Fit Scale=%s\n'%self.fitScaleComboBox.currentText())
            fh.write('#Fit Iterations=%s\n'%self.fitIterationLineEdit.text())
            fh.write('#Fixed Parameters:\n')
            fh.write('#param\tvalue\n')
            for row in range(self.fixedParamTableWidget.rowCount()):
                txt=self.fixedParamTableWidget.item(row,0).text()
                if txt in self.fit.params['choices'].keys():
                    fh.write(txt+'\t'+self.fixedParamTableWidget.cellWidget(row, 1).currentText()+'\n')
                elif txt in self.fit.params['filepaths'].keys():
                    fh.write(txt+'\t'+str(getattr(self,'filebutton_paths',{}).get(txt) or self.fit.params.get(txt) or '')+'\n')
                else:
                    fh.write(txt+'\t'+self.fixedParamTableWidget.item(row,1).text()+'\n')
            fh.write('#Single fitting parameters:\n')
            fh.write('#param\tvalue\tfit\tmin\tmax\texpr\tbrute_step\n')
            for row in range(self.sfitParamTableWidget.rowCount()):
                parname=self.sfitParamTableWidget.item(row,0).text()
                par=self.sfitParamTableWidget.item(row,1)
                parval=par.text()
                if par.checkState()==Qt.Checked:
                    parfit='1'
                else:
                    parfit='0'
                parmin=self.sfitParamTableWidget.item(row,2).text()
                parmax=self.sfitParamTableWidget.item(row,3).text()
                parexpr=self.sfitParamTableWidget.item(row,4).text()
                parbrute=self.sfitParamTableWidget.item(row,5).text()
                fh.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(parname,parval,parfit,parmin,parmax,parexpr,parbrute))
            if self.fit.params['__mpar__']!={}:
                fh.write('#Multiple fitting parameters:\n')
                fh.write('#param\tvalue\tfit\tmin\tmax\texpr\tbrute_step\n')
                for i in range(self.mfitParamTabWidget.count()):
                    mkey=self.mfitParamTabWidget.tabText(i)
                    for col in range(self.mfitParamTableWidget[mkey].columnCount()):
                        pkey = self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                        if col!=0:
                            for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                                parname='__%s_%s_%03d'%(mkey,pkey,row)
                                par=self.mfitParamTableWidget[mkey].item(row,col)
                                parval=par.text()
                                if par.checkState()==Qt.Checked:
                                    parfit='1'
                                else:
                                    parfit='0'
                                parmin=str(self.fit.fit_params[parname].min)
                                parmax=str(self.fit.fit_params[parname].max)
                                parexpr=str(self.fit.fit_params[parname].expr)
                                parbrute=str(self.fit.fit_params[parname].brute_step)
                                fh.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n'%(parname,parval,parfit,parmin,parmax,parexpr,parbrute))
                        else:
                            for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                                parname = '__%s_%s_%03d' % (mkey, pkey, row)
                                par = self.mfitParamTableWidget[mkey].item(row, col)
                                parval = par.text()
                                fh.write('%s\t%s\n' % (parname, parval))
            fh.close()
        
        
    def _covariance_header_lines(self, data_key=None):
        """The .xpar / fit-result header lines (without '#') for ``data_key``'s
        covariance (default: the selected dataset); [] without one, so files written
        without a covariance stay identical."""
        if data_key is None:
            data_key = self.sfnames[-1] if getattr(self, 'sfnames', None) else None
        cov = getattr(self, 'covariances', {}).get(data_key)
        if cov is None:
            return []
        path, raw = self.covariance_paths.get(data_key, (cov.source, None))
        out = ['Covariance=%s' % path]
        if raw:
            out.append('Covariance Raw=%s' % raw)
        out.append('Covariance Mode=%s' % self.covariance_modes.get(data_key, 'model'))
        return out

    def _apply_loaded_covariance(self, info):
        """Reattach the covariance named in a loaded .xpar to the selected dataset.
        No mode line -> the file predates covariance modes: data mode, and the next
        fit's report says so. A missing or non-matching file warns and the dataset
        continues without a covariance (spec 17 6.5, 6.6)."""
        path = info.get('path')
        if not path:
            return
        key = self.sfnames[-1] if getattr(self, 'sfnames', None) else None
        if key is None:
            msg = ('The parameter file names covariance %s, but no dataset is selected; '
                   'loaded without it.' % path)
            self._cov_status(msg)
            QMessageBox.warning(self, 'Covariance', msg, QMessageBox.Ok)
            return
        from Fit_Routines import _CovDowngrade

        def _note(why):      # the next fit's report says so too (spec 17 3, 6.6; review I2)
            self._cov_pending_notes[key] = [_CovDowngrade(
                'covariance %s named in the parameter file could not be reattached (%s); '
                'fitted without it' % (path, why))]
        if not os.path.isfile(path):
            msg = ('Covariance file not found: %s\n%s continues without a covariance.'
                   % (path, os.path.basename(key.split('<>', 1)[-1])))
            self._cov_status(msg)
            QMessageBox.warning(self, 'Covariance', msg, QMessageBox.Ok)
            _note('file not found')
            return
        raw = info.get('raw') or None
        if raw is not None and not os.path.isfile(raw):
            msg = ('Raw data file of covariance %s not found: %s%s%s continues without a '
                   'covariance.' % (os.path.basename(path), raw, os.linesep,
                                    os.path.basename(key.split('<>', 1)[-1])))
            self._cov_status(msg)
            QMessageBox.warning(self, 'Covariance', msg, QMessageBox.Ok)
            _note('raw data file %s not found' % raw)
            return
        self._cov_refusal = None
        if not self.attach_covariance_to(key, path, raw_path=raw):
            _note(getattr(self, '_cov_refusal', None) or 'refused')
            return
        mode = info.get('mode')
        if mode in ('model', 'data'):
            self.covariance_modes[key] = mode
        else:
            self.covariance_modes[key] = 'data'
            self._cov_pending_notes[key] = [
                'saved before covariance modes existed; reloaded in data mode' if mode is None
                else 'unknown covariance mode %r in the parameter file; reloaded in data mode' % mode]
        self._cov_sync_mode_combo()

    def loadParameters(self,fname=None):
        """
        loads parameters from a parameter file
        """
        # if self.funcListWidget.currentItem() is not None:
        if fname is None:
            fname=QFileDialog.getOpenFileName(self,caption='Open parameter file',directory=self.curDir,filter='Parameter files (*.par *.xpar)')[0]
        else:
            fname=fname
        if fname!='':
            self.curDir = os.path.dirname(fname)
            self._loading_params=True   # suppress functionChanged's per-dataset store
            try:
                self.funcListWidget.itemSelectionChanged.disconnect()
            except:
                pass
            try:
                fh=open(fname,'r')
                lines=fh.readlines()
                category=lines[1].split(': ')[1].strip()
                cat_item=self.categoryListWidget.findItems(category,Qt.MatchExactly)
                self.categoryListWidget.setCurrentItem(cat_item[0])
                self.funcListWidget.clearSelection()
                func=lines[2].split(': ')[1].strip()
                func_item=self.funcListWidget.findItems(func,Qt.MatchExactly)
                self.funcListWidget.itemSelectionChanged.connect(self.functionChanged)
                self.funcListWidget.setCurrentItem(func_item[0])
                #self.fit.func.init_params()
                if func==self.funcListWidget.currentItem().text():
                    lnum=3
                    sfline=None
                    mfline=None
                    _cov_lines={}
                    for line in lines[3:]:
                        if '#Xrange=' in line:
                            self.xLineEdit.setText(line.strip().split('=')[1])
                        elif '#Fit Range=' in line:
                            self.xminmaxLineEdit.setText(line.strip().split('=')[1])
                            fline=lnum+1
                        elif '#Column Limits=' in line:
                            # split('=',1): the value can contain '=' in future
                            self._apply_column_limits(line.strip().split('=',1)[1])
                            fline=lnum+1
                        elif '#Covariance=' in line:
                            _cov_lines['path']=line.strip().split('=',1)[1]
                            fline=lnum+1
                        elif '#Covariance Raw=' in line:
                            _cov_lines['raw']=line.strip().split('=',1)[1]
                            fline=lnum+1
                        elif '#Covariance Mode=' in line:
                            _cov_lines['mode']=line.strip().split('=',1)[1]
                            fline=lnum+1
                        elif '#Fit Method=' in line:
                            self.fitMethodComboBox.setCurrentText(line.strip().split('=')[1])
                            fline=lnum+1
                        elif '#Fit Scale=' in line:
                            self.fitScaleComboBox.setCurrentText(line.strip().split('=')[1])
                            fline=lnum+1
                        elif '#Fit Iterations=' in line:
                            self.fitIterationLineEdit.setText(line.strip().split('=')[1])
                            fline=lnum+1
                        elif line=='#Fixed Parameters:\n':
                            fline=lnum+2
                        elif line=='#Single fitting parameters:\n':
                            sfline=lnum+2
                        elif line=='#Multiple fitting parameters:\n':
                            mfline=lnum+2
                        lnum+=1
                    if sfline is None:
                        sendnum=lnum
                    else:
                        sendnum=sfline-2
                    if mfline is None:
                        mendnum=lnum
                    else:
                        mendnum=mfline-2
                    for line in lines[fline:sendnum]:
                        key,val=line.strip().split('\t')
                        try:
                            val=eval(val.strip())
                        except:
                            val=val.strip()
                        self.fit.params[key]=val
                    if hasattr(self.fit.func,
                               'update_param_set'):  # models whose fit-parameter set depends on fixed choices
                        self.fit.func.update_param_set()
                    if sfline is not None:
                        for line in lines[sfline:mendnum]:
                            parname,parval,parfit,parmin,parmax,parexpr,parbrute=line.strip().split('\t')
                            self.fit.params[parname]=float(parval)
                            self.fit.fit_params[parname].set(value=float(parval),vary=int(parfit),min=float(parmin),max=float(parmax))
                            try:
                                self.fit.fit_params[parname].set(expr=eval(parexpr))
                            except:
                                self.fit.fit_params[parname].set(expr=str(parexpr))
                            try:
                                self.fit.fit_params[parname].set(brute_step=eval(parbrute))
                            except:
                                self.fit.fit_params[parname].set(brute_step=str(parbrute))

                    if mfline is not None:
                        self.mfitParamCoupledCheckBox.setEnabled(True)
                        val={}
                        expr={}
                        pmin={}
                        pmax={}
                        pbrute={}
                        pfit={}
                        for line in lines[mfline:]:
                            tlist=line.strip().split('\t')
                            if len(tlist)>2:
                                parname,parval,parfit,parmin,parmax,parexpr,parbrute=tlist
                                val[parname]=float(parval)
                                pmin[parname]=float(parmin)
                                pmax[parname]=float(parmax)
                                pfit[parname]=int(parfit)
                                try:
                                    expr[parname]=eval(parexpr)
                                except:
                                    expr[parname]=str(parexpr)
                                try:
                                    pbrute[parname]=eval(parbrute)
                                except:
                                    pbrute[parname]=str(parbrute)
                                try: # Here the expr is set to None and will be taken care at the for loop just after this for loop
                                    self.fit.fit_params[parname].set(val[parname], vary=pfit[parname],
                                                                     min=pmin[parname],
                                                                     max=pmax[parname], expr=None,
                                                                     brute_step=pbrute[parname])
                                except:
                                    self.fit.fit_params.add(parname, value=val[parname], vary=pfit[parname],
                                                            min=pmin[parname],
                                                            max=pmax[parname], expr=None,
                                                            brute_step=pbrute[parname])
                                mkey, pkey, num = parname[2:].rsplit('_', 2)  # tolerate namespaced 'd0_Model' mkeys
                                num = int(num)
                                try:
                                    self.fit.params['__mpar__'][mkey][pkey][num] = float(parval)
                                except:
                                    self.fit.params['__mpar__'][mkey][pkey].insert(num, float(parval))
                            else:
                                parname,parval=tlist
                                mkey,pkey,num=parname[2:].rsplit('_', 2)        # tolerate namespaced 'd0_Model' mkeys
                                num=int(num)
                                try:
                                    self.fit.params['__mpar__'][mkey][pkey][num]=parval
                                except:
                                    self.fit.params['__mpar__'][mkey][pkey].insert(num,parval)
                        for parname in val.keys(): #Here is the expr is put into the parameters
                            try:
                                self.fit.fit_params[parname].set(value=val[parname], vary=pfit[parname], min=pmin[parname],
                                                                 max=pmax[parname], expr=expr[parname], brute_step=pbrute[parname])
                            except:
                                self.fit.fit_params.add(parname, value=val[parname], vary=pfit[parname], min=pmin[parname],
                                                        max=pmax[parname], expr=expr[parname], brute_step=pbrute[parname])
                    try:
                        self.fixedParamTableWidget.cellChanged.disconnect()
                        self.sfitParamTableWidget.cellChanged.disconnect()
                        for i in range(self.mfitParamTabWidget.count()):
                            mkey = self.mfitParamTabWidget.tabText(i)
                            self.mfitParamTableWidget[mkey].cellChanged.disconnect()
                    except:
                        pass
                    self.update_fixed_parameters()
                    self.update_fit_parameters()
                    self.fixedParamTableWidget.cellChanged.connect(self.fixedParamChanged)
                    self.sfitParamTableWidget.cellChanged.connect(self.sfitParamChanged)
                    for i in range(self.mfitParamTabWidget.count()):
                        mkey=self.mfitParamTabWidget.tabText(i)
                        self.mfitParamTableWidget[mkey].cellChanged.connect(self.mfitParamChanged_new)
                    if len(self.dataListWidget.selectedItems())>0:
                        self.xminmaxChanged()
                    else:
                        self.xChanged()
                    self.errorAvailable=False
                    self.reuse_sampler=False
                    # a loaded .par counts as fit -> CIs available (a baseline leastsq is
                    # run on demand in calcConfInterval to characterize the minimum).
                    self.fit._is_fitted=True
                    self._update_ci_button_state()
                    # persist the loaded model into the per-dataset store (detail
                    # dataset only) so switching away and back keeps it; refresh
                    # provenance (functionChanged stamped it before #Xrange was parsed).
                    self.fit._xmod_category=category
                    self.fit._xmod_module='Functions.%s.%s'%(category,func)
                    self.fit._xmod_funcname=func
                    self.fit._xmod_xexpr=self.xLineEdit.text()
                    self.fit._is_fitted=True   # a loaded .par counts as "already fit"
                    _sel=[it.text() for it in self.dataListWidget.selectedItems()]
                    if _sel:
                        self.dataset_fits[_sel[-1]]=self.fit
                    else:
                        self._template_fit=self.fit
                    # Mark the just-loaded dataset as fitted (green) immediately, rather
                    # than only after the user reselects it.
                    self._refresh_data_labels()
                    if _cov_lines:
                        self._apply_loaded_covariance(_cov_lines)
                else:
                    QMessageBox.warning(self, 'File error',
                                        'This parameter file does not belong to function: %s' % func, QMessageBox.Ok)
            except:
                QMessageBox.warning(self,'File Import Error','Some problems in the parameter file\n'+traceback.format_exc(), QMessageBox.Ok)
            self._loading_params=False
        # else:
        #     QMessageBox.warning(self,'Function error','Please select a function first before loading parameter file.', QMessageBox.Ok)


        
    def create_plotDock(self):
        self.plotSplitter=QSplitter(Qt.Vertical)
        #self.plotLayoutWidget=pg.LayoutWidget(self)
        # The top pane is a tab stack so the plot area can adapt to the data format:
        # tab 0 is the ordinary 1-D data+fit plot, tab 1 is a 2-D map panel that only
        # appears when the selected model emits 'image' output_params. For the ~40
        # existing 1-D functions the dock therefore looks exactly as it did before.
        self.plotTabWidget=QTabWidget()
        self.plotWidget=PlotWidget()
        self.plotWidget.setXLabel('x',fontsize=5)
        self.plotWidget.setYLabel('y',fontsize=5)
        # Main-plot legend shows dataset names (not raw "N:Col_1" keys), tagged by curve
        # type. Bottom (generated-param) plot keeps its raw labels -- they encode the
        # param + dataset and read correctly already.
        self.plotWidget.legend_formatter=self._legend_label
        self.plotTabWidget.addTab(self.plotWidget,'1D')
        self.imageWidget=Image2DWidget()
        self._plotTab2D_index=self.plotTabWidget.addTab(self.imageWidget,'2D Map')
        self._plotTab_user_choice=False      # True once the user picks a tab by hand
        self._programmatic_tab_change=False
        self._set_2D_tab_visible(False)
        self.plotTabWidget.currentChanged.connect(self._on_plotTab_changed)
        self.plotSplitter.addWidget(self.plotTabWidget)

        self.extra_param_1DplotWidget=PlotWidget()
        self.extra_param_1DplotWidget.setXLabel('x',fontsize=5)
        self.extra_param_1DplotWidget.setYLabel('y',fontsize=5)
        self.plotSplitter.addWidget(self.extra_param_1DplotWidget)

        self.plotDock.addWidget(self.plotSplitter)

    def _set_2D_tab_visible(self, on):
        """Show/hide the 2D Map tab without disturbing the user's current tab."""
        tw = self.plotTabWidget
        i = self._plotTab2D_index
        self._programmatic_tab_change = True
        try:
            if hasattr(tw, 'setTabVisible'):            # Qt >= 5.15
                tw.setTabVisible(i, bool(on))
            elif on and tw.indexOf(self.imageWidget) < 0:
                tw.insertTab(i, self.imageWidget, '2D Map')
            elif not on and tw.indexOf(self.imageWidget) >= 0:
                tw.removeTab(tw.indexOf(self.imageWidget))
            if not on and tw.currentIndex() == i:
                tw.setCurrentIndex(0)
        except Exception:
            pass
        finally:
            self._programmatic_tab_change = False

    def _show_2D_tab(self, on, activate=False):
        """Make the 2D tab available, and optionally raise it.

        Raise it only when the selection produced no 1-D curve at all AND the user
        has not yet picked a tab by hand -- so selecting a map entry alone lands you
        on the map, adding a rod curve leaves you where you were, and once you click
        a tab yourself XModFit stops second-guessing you.
        """
        self._set_2D_tab_visible(on)
        if on and activate and not self._plotTab_user_choice:
            self._programmatic_tab_change = True
            try:
                self.plotTabWidget.setCurrentIndex(self._plotTab2D_index)
            finally:
                self._programmatic_tab_change = False

    def _on_plotTab_changed(self, index):
        if not self._programmatic_tab_change:
            self._plotTab_user_choice = True

    def create_fitResultDock(self):
        self.fitResultsLayoutWidget = pg.LayoutWidget()
        fitResults = QLabel('Fit Results')
        self.fitResultsLayoutWidget.addWidget(fitResults, colspan=1)
        self.fitResultsLayoutWidget.nextRow()
        self.fitResultTextEdit = QTextEdit()
        self.fitResultsLayoutWidget.addWidget(self.fitResultTextEdit, colspan=1)
        self.fitResultDock.addWidget(self.fitResultsLayoutWidget)

    def create_metaDataDock(self):
        """
        Creates the metaPlotDataDock
        """
        self.metaDataPlotLayout = pg.LayoutWidget(self)
        row = 0
        col = 0
        xAxisLabel = QLabel('X-Axis')
        self.metaDataPlotLayout.addWidget(xAxisLabel, row=row, col=col)
        col += 1
        self.metaXAxisComboBox = QComboBox()
        self.metaXAxisComboBox.currentIndexChanged.connect(self.metaDataChanged)
        self.metaDataPlotLayout.addWidget(self.metaXAxisComboBox, row=row, col=col)
        col += 2
        yAxisLabel = QLabel('Y-Axis')
        self.metaDataPlotLayout.addWidget(yAxisLabel, row=row, col=col)
        col += 1
        self.metaYAxisComboBox = QComboBox()
        self.metaYAxisComboBox.currentIndexChanged.connect(self.metaDataChanged)
        self.metaDataPlotLayout.addWidget(self.metaYAxisComboBox, row=row, col=col)
        col += 2
        yErrLabel = QLabel('Y-Axis Error')
        self.metaDataPlotLayout.addWidget(yErrLabel, row=row, col=col)
        col += 1
        self.metaYErrAxisComboBox = QComboBox()
        self.metaYErrAxisComboBox.currentIndexChanged.connect(self.metaDataChanged)
        self.metaDataPlotLayout.addWidget(self.metaYErrAxisComboBox, row=row, col=col)

        row += 1
        col = 0
        self.metaDataPlotWidget = PlotWidget(self)
        self.metaDataPlotLayout.addWidget(self.metaDataPlotWidget, row=row, col=col, colspan=8)

        row += 1
        col = 4
        self.showMetaDataPushButton = QPushButton('Show Selected')
        self.metaDataPlotLayout.addWidget(self.showMetaDataPushButton, row=row, col=col, colspan=2)
        self.showMetaDataPushButton.clicked.connect(self.showMetaData)

        col=6
        self.saveSelectedMetaDataPushButton = QPushButton('Save Selected')
        self.metaDataPlotLayout.addWidget(self.saveSelectedMetaDataPushButton, row=row, col=col, colspan=2)
        self.saveSelectedMetaDataPushButton.clicked.connect(self.saveSelectedMetaData)

        self.metaDataDock.addWidget(self.metaDataPlotLayout)

    def metaDataChanged(self):
        xaxis = self.metaXAxisComboBox.currentText()
        yaxis = self.metaYAxisComboBox.currentText()
        yerraxis = self.metaYErrAxisComboBox.currentText()
        if xaxis!='None' and yaxis!='None':
            x=[]
            y=[]
            yerr=[]
            for key in self.sfnames:
                x.append(self.meta[key][xaxis])
                y.append(self.meta[key][yaxis])
                if yerraxis!='None':
                    yerr.append(self.meta[key][yerraxis])
                else:
                    yerr.append(1.0)
            x=np.array(x)
            y=np.array(y)
            if yerraxis!='None':
                yerr=np.array(yerr)
                self.metayerr = yerr[x.argsort()]
            self.metay = y[x.argsort()]
            self.metax = x[x.argsort()]
            if yerraxis!='None':
                self.metaDataPlotWidget.add_data(self.metax,self.metay,yerr=self.metayerr)
            else:
                self.metaDataPlotWidget.add_data(self.metax, self.metay)

    def showMetaData(self):
        if self.metaXAxisComboBox.currentText!=None and self.metaYAxisComboBox.currentText()!=None:
            meta={}
            if self.metaYErrAxisComboBox.currentText()!='None':
                meta['col_names'] = [self.metaXAxisComboBox.currentText(), self.metaYAxisComboBox.currentText(),
                                     self.metaYErrAxisComboBox.currentText()]
                data = {'data': pd.DataFrame(list(zip(self.metax, self.metay, self.metayerr)), columns=meta['col_names']),
                        'meta': meta}
            else:
                meta['col_names'] = [self.metaXAxisComboBox.currentText(), self.metaYAxisComboBox.currentText()]
                data = {
                    'data': pd.DataFrame(list(zip(self.metax, self.metay)), columns=meta['col_names']),
                    'meta': meta}
            data_dlg = Data_Dialog(data=data, parent=self, expressions={},
                                   plotIndex=None, colors=None)
            data_dlg.setModal(True)
            data_dlg.closePushButton.setText('Cancel')
            data_dlg.tabWidget.setCurrentIndex(0)
            data_dlg.dataFileLineEdit.setText('None')
            data_dlg.exec_()
        else:
            QMessageBox.warning(self,'Metadata Error','No data selected to plot',QMessageBox.Ok)

    def saveSelectedMetaData(self):
        fname=QFileDialog.getSaveFileName(self,'Save Meta-data as',directory=self.curDir,filter='Text Files (*.txt)')[0]
        fname=os.path.splitext(fname)[0]+'.txt'
        header='Meta-data file saved on %s\n'%time.asctime()
        if self.metaYErrAxisComboBox.currentText()=='None':
            header += 'col_names=["%s", "%s"]\n' % (self.metaXAxisComboBox.currentText(), self.metaYAxisComboBox.currentText())
            np.savetxt(fname, np.vstack((self.metax, self.metay)).T, header=header)
        else:
            header+='col_names=["%s", "%s", "%s"]\n'%(self.metaXAxisComboBox.currentText(),self.metaYAxisComboBox.currentText(),
                                            self.metaYErrAxisComboBox.currentText())
            np.savetxt(fname, np.vstack((self.metax, self.metay, self.metayerr)).T, header=header)

    # def saveAllMetaData(self):
    #     fname=QFileDialog.getSaveFileName(self,'Save Meta-data as',directory=self.curDir,filter='Text Files (*.txt)')[0]
    #     fname=os.path.splitext(fname)[0]+'.txt'
    #     header='Meta-data file saved on %s\n'%time.asctime()
    #     header+='col_names=['
    #     keys=self.meta[self.sfnames[-1]].keys()
    #     data=[]
    #     for fname in self.sfnames:
    #         tdata=[]
    #         tkeys=[]
    #         for key in keys:
    #             tdat=self.meta[fname][key]
    #             if type(tdat)== int or float:
    #                 tdata.append(tdat)
    #                 tkeys.append(key)
    #         data.append(tdata)
    #     for key in tkeys:
    #         header+='"%s", '%key
    #     header=header[:-2]+']\n'
    #     data=np.array(data)
    #     np.savetxt(fname,data,header=header)


    def update_catagories(self):
        """
        Reads all the modules in the the Functions directory and populates the funcListWidget
        """
        self.categoryListWidget.clear()
        self.categories=sorted([path for path in os.listdir('./Functions/') if path[:2]!='__' and os.path.isdir('./Functions/'+path)])
        #self.catagories=sorted([m.split('.')[0] for m in modules if m[:2]!='__'],key=str.lower)
        self.categoryListWidget.addItems(self.categories)

        
    def update_functions(self):
        """
        Depending upon the selected category this populates the funcListWidget
        """
        self.saveSimulatedButton.setEnabled(False)
        try:
            self.funcListWidget.itemSelectionChanged.disconnect()
            self.funcListWidget.itemDoubleClicked.disconnect()
        except:
            pass
        self.funcListWidget.clear()
        self.curr_category=self.categoryListWidget.currentItem().text()
        self.modules=[]
        for module in os.listdir('./Functions/'+self.curr_category):
            if module!='__init__.py' and module[-2:]=='py':
                self.modules.append(module[:-3])
        self.modules=sorted(self.modules,key=str.lower)
        self.funcListWidget.addItems(self.modules)
        for i in range(self.funcListWidget.count()):
            mname=self.funcListWidget.item(i).text()
            module='Functions.%s.%s'%(self.curr_category,mname)
            if module not in sys.modules or module not in self.curr_funcClass.keys():
                self.curr_funcClass[module]=import_module(module)
            else:
                self.curr_funcClass[module]=reload(self.curr_funcClass[module])
            self.funcListWidget.item(i).setToolTip(getattr(self.curr_funcClass[module],self.funcListWidget.item(i).text()).__init__.__doc__)
        self.funcListWidget.itemSelectionChanged.connect(self.functionChanged)
        self.funcListWidget.itemDoubleClicked.connect(self.openFunction)
        
    def functionChanged(self):
        if len(self.funcListWidget.selectedItems())<=1:
            self.sfitLabel.clear()
            self.mfitLabel.clear()
            self.sfitSlider.setValue(500)
            self.mfitSlider.setValue(500)
            self.gen_param_items=[]
            self._image_autoselected=False
            self.curr_module=self.funcListWidget.currentItem().text()
            module='Functions.%s.%s'%(self.curr_category,self.curr_module)
            self.mfitParamCoupledCheckBox.setEnabled(False)
            try:
                if module not in sys.modules:
                    self.curr_funcClass[module]=import_module(module)
                else:
                    self.curr_funcClass[module]=reload(self.curr_funcClass[module])
                mpath=os.path.join('Functions',self.curr_category,self.curr_module+'.py')
                fh=open(mpath,'r')
                lines=fh.readlines()
                for i,line in enumerate(lines):
                    if '__name__' in line:
                        lnum=i+1
                        break
                if 'x' in lines[lnum]:
                    self.xline=lines[lnum].split('=')[1].strip()
                else:
                    self.xline='np.linspace(0.0,1.0,100)'
                self.xLineEdit.setText(self.xline)
                self.fixedParamTableWidget.clear()
                self.sfitParamTableWidget.clear()
                self.mfitParamTabWidget.clear()
                # self.mfitParamTableWidget.clear()
                self.genParamListWidget.clear()
                self.fchanged = True
                self.update_parameters()
                # Picking a function means "use this model for the selected dataset(s)".
                # Store the freshly-built model per selected dataset (each an independent
                # clone so a later single-dataset fit doesn't mutate the others); the
                # last-selected (detail) dataset owns the live self.fit that
                # update_parameters just populated the tables/plot from. With nothing
                # selected, keep it as the template that seeds the next new dataset.
                # (skip while loadParameters is driving the func selection -- it
                #  persists the loaded model to the detail dataset itself, so we must
                #  not overwrite the OTHER selected datasets' fits with a blank template)
                if not getattr(self,'_loading_params',False):
                    sel=[it.text() for it in self.dataListWidget.selectedItems()]
                    if sel:
                        for nm in sel[:-1]:
                            self.dataset_fits[nm]=self._clone_fit(self.fit)
                        self.dataset_fits[sel[-1]]=self.fit
                    else:
                        self._template_fit=self.fit
                self.saveSimulatedButton.setEnabled(True)
                self.errorAvailable = False
                self.reuse_sampler = False
                self.calcConfInterButton.setDisabled(True)
            except:
                QMessageBox.warning(self,'Function Error','Some syntax error in the function still exists.\n'+traceback.format_exc(),QMessageBox.Ok)
        else:
            QMessageBox.warning(self,'Function Error', 'Please select one function at a time', QMessageBox.Ok)
        
    def update_parameters(self):
        """
        Depending upon the selection of the function this updates the reloads the parameters required for the function
        """
        try:
            self.fixedParamTableWidget.cellChanged.disconnect()
            self.sfitParamTableWidget.cellChanged.disconnect()
            for i in range(self.mfitParamTabWidget.count()):
                mkey=self.mfitParamTabWidget.tabText(i)
                self.mfitParamTableWidget[mkey].cellChanged.disconnect()
        except:
            pass
        try:
            self.x=eval(self.xLineEdit.text())
        except:
            QMessageBox.warning(self,'Parameter Error','The value you just entered is not correct.\n'+traceback.format_exc(),QMessageBox.Ok)
        self.curr_module=self.funcListWidget.currentItem().text()
        module='Functions.%s.%s'%(self.curr_category,self.curr_module)
        self.fit=Fit(getattr(self.curr_funcClass[module],self.funcListWidget.currentItem().text()),self.x)
        # provenance so the per-dataset store can auto-follow the function list + Xrange
        self.fit._xmod_category=self.curr_category
        self.fit._xmod_module=module
        self.fit._xmod_funcname=self.curr_module
        self.fit._xmod_xexpr=self.xLineEdit.text()
        if '__mpar__' in self.fit.params.keys() and self.fit.params['__mpar__'] != {}:
            self.mpar_keys = list(self.fit.params['__mpar__'].keys())
            pkey=list(self.fit.params['__mpar__'][self.mpar_keys[0]].keys())[0]
            self.mpar_N={}
            for mkey in self.mpar_keys:
                self.mpar_N[mkey] = len(self.fit.params['__mpar__'][mkey][pkey])
        self.update_fixed_parameters()
        self.update_fit_parameters()
        self.update_plot()
        self.xLineEdit.returnPressed.connect(self.xChanged)
        # self.mfitParamTableWidget.cellChanged.connect(self.mfitParamChanged)
        self.fixedParamTableWidget.cellChanged.connect(self.fixedParamChanged)
        self.sfitParamTableWidget.cellChanged.connect(self.sfitParamChanged)
        for i in range(self.mfitParamTabWidget.count()):
            mkey = self.mfitParamTabWidget.tabText(i)
            self.mfitParamTableWidget[mkey].cellChanged.connect(self.mfitParamChanged_new)

    def update_fixed_parameters(self):
        self.filebutton={}
        self.filebutton_paths={}   # key -> FULL path (button shows only the basename)
        try:
            self.fixedParamTableWidget.cellChanged.disconnect()
        except:
            pass
        fpdata=[]        
        for key in self.fit.params.keys():
            if key not in self.fit.fit_params.keys() and key not in self.special_keys and key[:2]!='__':
                fpdata.append((key,str(self.fit.params[key])))
        self.fixedParamData=np.array(fpdata,dtype=[('Params',object),('Value',object)])
        self.fixedParamTableWidget.setData(self.fixedParamData)
        for row in range(self.fixedParamTableWidget.rowCount()):
            self.fixedParamTableWidget.item(row,0).setFlags(Qt.ItemIsEnabled)
            ftxt=self.fixedParamTableWidget.item(row, 0).text()
            if ftxt in self.fit.params['choices'].keys():
                items=[str(item) for item in self.fit.params['choices'][ftxt]]
                combobox=QComboBox()
                combobox.addItems(items)
                self.fixedParamTableWidget.setCellWidget(row,1,combobox)
                index = combobox.findText(str(self.fit.params[ftxt]))
                combobox.setCurrentIndex(index)
                combobox.currentIndexChanged.connect(lambda x: self.fixedParamChanged(row,1))
            if ftxt in self.fit.params['filepaths'].keys():
                # Prefer the LIVE attribute value (e.g. func.fname) over the
                # __init__-time 'filepaths' snapshot: a per-dataset PDB choice is
                # written to the attribute (fname) and must survive dataset
                # switching / cloning. 'filepaths' is only stamped once at
                # construction, so reading it here would revert to the default PDB.
                fname=self.fit.params.get(ftxt) or self.fit.params['filepaths'][ftxt]
                self.filebutton_paths[ftxt]=fname          # keep the FULL path
                if fname is not None and os.path.exists(fname):
                    self.filebutton[ftxt]=QPushButton('[Click] '+os.path.basename(fname))
                    self.filebutton[ftxt].setToolTip(fname)   # full path on hover
                    self.fixedParamTableWidget.setCellWidget(row,1,self.filebutton[ftxt])
                    self.filebutton[ftxt].clicked.connect(lambda x, k=ftxt: self.openFileDlg(fname=self.filebutton_paths.get(k), key=k))
                else:
                    self.filebutton[ftxt] = QPushButton('[Click] ')
                    self.fixedParamTableWidget.setCellWidget(row, 1, self.filebutton[ftxt])
                    self.filebutton[ftxt].clicked.connect(lambda x, k=ftxt: self.openFileDlg(fname=self.filebutton_paths.get(k), key=k))
                    # NOTE: there used to be a bare self.openFileDlg(fname=None)
                    # here. It could never open a dialog (the fname guard inside
                    # skipped it), so all it did was re-write fit.params[ftxt]
                    # with the value it already held and then call update_plot()
                    # from the MIDDLE of this table rebuild -- at which point the
                    # comboboxes for later 'choices' rows do not exist yet, so
                    # update_plot's cellWidget(row,1).currentText() hit None.
                    # functionChanged already calls update_plot() right after
                    # this method returns, so the call was pure re-entrancy.
        self.fixedParamTableWidget.resizeRowsToContents()
        self.fixedParamTableWidget.resizeColumnsToContents()
        self.fixedParamTableWidget.cellChanged.connect(self.fixedParamChanged)

    def openFileDlg(self, fname=None, key=None, prompt=True):
        """Pick a file for a 'filepaths' parameter in the Fixed Parameters table.

        'key' must be supplied whenever this is called directly rather than from
        a button click: self.sender() is then not the cell button, so
        indexAt(pos()).row() returns -1 and item(-1, 0) is None -- which used to
        raise AttributeError and abort loading the whole function. Any model
        whose filepath default does not exist on disk hit that path.

        prompt=False syncs the table cell without opening a dialog, which is
        what update_fixed_parameters needs when there is no usable default.
        """
        row = -1
        if key is None:
            buttonClicked = self.sender()
            if buttonClicked is not None and hasattr(buttonClicked, 'pos'):
                row = self.fixedParamTableWidget.indexAt(buttonClicked.pos()).row()
            item = self.fixedParamTableWidget.item(row, 0) if row >= 0 else None
            if item is None:
                return
            key = item.text()
        else:
            for r in range(self.fixedParamTableWidget.rowCount()):
                it = self.fixedParamTableWidget.item(r, 0)
                if it is not None and it.text() == key:
                    row = r
                    break
            if row < 0:
                return
        if prompt:
            # A missing/blank default must still open a dialog, otherwise the
            # button is permanently dead and the parameter can never be set.
            has = fname is not None and os.path.exists(fname)
            tfname = QFileDialog.getOpenFileName(
                self, 'Provide file path for %s variable in the Fixed Parameters table' % key,
                directory=os.path.dirname(fname) if has else os.path.abspath(os.curdir),
                filter=('Files (*%s)' % os.path.splitext(fname)[1]) if has else 'All files (*)')[0]
            if tfname != '':
                self.filebutton_paths[key] = tfname        # store FULL path
                self.filebutton[key].setText('[Click] ' + os.path.basename(tfname))
                self.filebutton[key].setToolTip(tfname)
            elif has:
                self.filebutton_paths[key] = fname
                self.filebutton[key].setText('[Click] ' + os.path.basename(fname))
                self.filebutton[key].setToolTip(fname)
        self.fixedParamChanged(row, 1)


    def update_fit_parameters(self):
        self.update_sfit_parameters()
        # self.update_mfit_parameters()
        self.update_mfit_parameters_new()
        self.sfitParamTableWidget.resizeRowsToContents()
        self.sfitParamTableWidget.resizeColumnsToContents()
        mkeys = list(self.fit.params['__mpar__'].keys())
        if '__mpar__' in self.fit.params.keys() and self.fit.params['__mpar__'] != {}:
            for mkey in mkeys:
                self.mfitParamTableWidget[mkey].resizeRowsToContents()
                self.mfitParamTableWidget[mkey].resizeColumnsToContents()


    def update_sfit_parameters(self):
        try:
            self.sfitParamTableWidget.cellChanged.disconnect()
        except:
            pass
        tpdata=[]
        for key in self.fit.fit_params.keys():
            if key[:2]!='__':
                tpdata.append((key,self.fit.fit_params[key].value,self.fit.fit_params[key].min,
                               self.fit.fit_params[key].max,str(self.fit.fit_params[key].expr),self.fit.fit_params[key].brute_step))
        self.fitParamData=np.array(tpdata,dtype=[('Params',object),('Value',object),('Min',object),('Max',object),
                                                 ('Expr',object),('Brute step',float)])
        self.sfitParamTableWidget.setData(self.fitParamData)
        self.sfitParamTableWidget.setFormat(self.format,column=1)
        for row in range(self.sfitParamTableWidget.rowCount()):
            self.sfitParamTableWidget.item(row,0).setFlags(Qt.ItemIsEnabled)
            par=self.sfitParamTableWidget.item(row,0).text()
            item=self.sfitParamTableWidget.item(row,1)
            item.setFlags(Qt.ItemIsUserCheckable|Qt.ItemIsEnabled|Qt.ItemIsEditable|Qt.ItemIsSelectable)
            if self.fit.fit_params[par].vary==0:
                item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)
            item.setToolTip((par+' = '+self.format+' \u00B1 '+self.format) % (self.fit.fit_params[par].value, 0.0))
        self.sfitParamTableWidget.resizeRowsToContents()
        self.sfitParamTableWidget.resizeColumnsToContents()
        self.sfitParamTableWidget.cellChanged.connect(self.sfitParamChanged)

    def update_mfit_parameters_new(self):
        try:
            self.mfitParamTabWidget.currentChanged.disconnect()
        except:
            pass
        if '__mpar__' in self.fit.params.keys() and self.fit.params['__mpar__']!={}:
            if len(self.fit.params['__mpar__'])>1:
                self.mfitParamCoupledCheckBox.setEnabled(True)
            # self.mfitParamCoupledCheckBox.setCheckState(Qt.Unchecked)
            self.mfitParamTableWidget = {}
            self.mfitParamData = {}
            mkeys=list(self.fit.params['__mpar__'].keys())
            if self.mfitParamTabWidget.count()>0:
                for i in range(self.mfitParamTabWidget.count()-1,-1,-1):
                    try:
                        self.mfitParamTabWidget.removeTab(i)
                    except:
                        pass
            for mkey in mkeys:
                self.mfitParamTableWidget[mkey] = pg.TableWidget(sortable=False)
                #self.mfitParamTableWidget[mkey].setSelectionBehavior(QAbstractItemView.SelectRows)
                self.mfitParamTableWidget[mkey].cellClicked.connect(self.update_mfitSlider)
                self.mfitParamTableWidget[mkey].cellDoubleClicked.connect(self.mparDoubleClicked)
                self.mfitParamTabWidget.addTab(self.mfitParamTableWidget[mkey],mkey)
                pkeys=list(self.fit.params['__mpar__'][mkey].keys())
                mpar_N=len(self.fit.params['__mpar__'][mkey][pkeys[0]])
                tpdata=[]
                for i in range(mpar_N):
                    temp = []
                    for pkey in pkeys:
                        tkey='__%s_%s_%03d' % (mkey, pkey, i)
                        if tkey in self.fit.fit_params.keys():
                            temp.append(self.fit.fit_params[tkey].value)
                        else:
                            temp.append(self.fit.params['__mpar__'][mkey][pkey][i])
                    tpdata.append(tuple(temp))
                self.mfitParamData[mkey]=np.array(tpdata,dtype=[(pkey,object) for pkey in pkeys])
                self.mfitParamTableWidget[mkey].setData(self.mfitParamData[mkey])
                self.mfitParamTableWidget[mkey].setFormat(self.format)
                for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                    for col in range(self.mfitParamTableWidget[mkey].columnCount()):
                        item = self.mfitParamTableWidget[mkey].item(row, col)
                        if col==0:
                            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                        else:
                            item.setFlags(
                            Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
                            key = '__%s_%s_%03d' % (mkey, self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text(), row)
                            if self.fit.fit_params[key].vary == 0 or self.fit.fit_params[key].vary==False:
                                item.setCheckState(Qt.Unchecked)
                            else:
                                item.setCheckState(Qt.Checked)
                            item.setToolTip((key + ' = ' + self.format + ' \u00B1 ' + self.format) % (
                            self.fit.fit_params[key].value, 0.0))
                self.mfitParamTableWidget[mkey].resizeRowsToContents()
                self.mfitParamTableWidget[mkey].resizeColumnsToContents()
                self.mfitParamTableWidget[mkey].cellChanged.connect(self.mfitParamChanged_new)
            self.add_mpar_button.setEnabled(True)
            self.remove_mpar_button.setEnabled(True)
            self.mfitParamTabChanged(0)
        else:
            self.add_mpar_button.setDisabled(True)
            self.remove_mpar_button.setDisabled(True)
        self.mfitParamTabWidget.currentChanged.connect(self.mfitParamTabChanged)

        
    # def update_mfit_parameters(self):
    #     try:
    #         self.mfitParamTableWidget.cellChanged.disconnect()
    #     except:
    #         pass
    #     if '__mpar__' in self.fit.params.keys() and self.fit.params['__mpar__']!={}:
    #         mpar_keys=list(self.fit.params['__mpar__'].keys())
    #         mpar_N=len(self.fit.params['__mpar__'][mpar_keys[0]])
    #         tpdata=[]
    #         for i in range(mpar_N):
    #             temp=[]
    #             for key in mpar_keys:
    #                 if key in self.fit.fit_params.keys():
    #                     temp.append(self.fit.fit_params['__%s__%03d'%(key,i)].value)
    #                 else:
    #                     temp.append(self.fit.params['__mpar__'][key][i])
    #             tpdata.append(tuple(temp))
    #             #tpdata.append(tuple([self.fit.fit_params['__%s__%03d'%(key,i)].value for key in mpar_keys]))
    #         self.mfitParamData=np.array(tpdata,dtype=[(key,object) for key in mpar_keys])
    #         self.mfitParamTableWidget.setData(self.mfitParamData)
    #         self.mfitParamTableWidget.setFormat(self.format)
    #         self.add_mpar_button.setEnabled(True)
    #         self.remove_mpar_button.setEnabled(True)
    #         for row in range(self.mfitParamTableWidget.rowCount()):
    #             for col in range(1,self.mfitParamTableWidget.columnCount()):
    #                 item=self.mfitParamTableWidget.item(row,col)
    #                 item.setFlags(Qt.ItemIsUserCheckable|Qt.ItemIsEnabled|Qt.ItemIsEditable|Qt.ItemIsSelectable)
    #                 key='__%s__%03d'%(self.mfitParamTableWidget.horizontalHeaderItem(col).text(),row)
    #                 if self.fit.fit_params[key].vary==0:
    #                     item.setCheckState(Qt.Unchecked)
    #                 else:
    #                     item.setCheckState(Qt.Checked)
    #                 item.setToolTip((key + ' = '+self.format+' \u00B1 '+self.format) % (self.fit.fit_params[key].value, 0.0))
    #         self.mfitParamTableWidget.resizeRowsToContents()
    #         self.mfitParamTableWidget.resizeColumnsToContents()
    #     else:
    #         self.add_mpar_button.setDisabled(True)
    #         self.remove_mpar_button.setDisabled(True)
    #         self.mfitParamTableWidget.setData([])
    #     self.mfitParamTableWidget.cellChanged.connect(self.mfitParamChanged)


    def fixedParamChanged(self,row,col):
        try:
            self.fixedParamTableWidget.cellChanged.disconnect()
        except:
            pass
        txt=self.fixedParamTableWidget.item(row,0).text()
        if txt in self.fit.params['choices'].keys():
            self.fixedParamTableWidget.cellWidget(row, 1).currentIndexChanged.disconnect()
            try: # if the parameter is a number
                self.fit.params[txt]=eval(self.fixedParamTableWidget.cellWidget(row,1).currentText())
            except: # if the parameter is a string
                self.fit.params[txt] = str(self.fixedParamTableWidget.cellWidget(row, 1).currentText())
            self.fchanged = False
            self.update_plot()
            self.fixedParamTableWidget.cellWidget(row, 1).currentIndexChanged.connect(lambda x:self.fixedParamChanged(row,1))
        elif txt in self.fit.params['filepaths'].keys():
            # value is the FULL path from filebutton_paths (button shows only basename)
            self.fit.params[txt]=self.filebutton_paths.get(txt, self.fit.params.get(txt))
            self.fchanged=False
            self.update_plot()
        else:
            try: # if the parameter is a number
                val=eval(self.fixedParamTableWidget.item(row,col).text())
            except:  #if the parameter is a string
                val=self.fixedParamTableWidget.item(row,col).text()
            try:
                oldVal=self.fit.params[txt]
                self.fit.params[txt]=val
                self.fchanged = False
                self.update_plot()
            except:
                QMessageBox.warning(self,'Value Error','The value just entered is not seem to be right.\n'+traceback.format_exc(),QMessageBox.Ok)
                self.fixedParamTableWidget.item(row,col).setText(str(oldVal))
        # In a joint fit, ALL fixed parameters (including the PDB) are shared: fan the
        # edited value to every objective so the joint fit tests one structural hypothesis.
        self._fan_fixed_to_objectives(txt)
        self.fixedParamTableWidget.resizeRowsToContents()
        self.fixedParamTableWidget.resizeColumnsToContents()
        self.update_fit_parameters()
        self.fixedParamTableWidget.cellChanged.connect(self.fixedParamChanged)

    def _fan_fixed_to_objectives(self, key):
        """Force a fixed parameter (incl. the PDB ``fname``) to be shared across a joint
        fit: copy the CURRENT objective's value into every objective's model. No-op for a
        single Fit."""
        jf = getattr(self, 'fit', None)
        if not hasattr(jf, 'objectives'):
            return
        params = jf.params                      # current objective's __dict__
        if key not in params:
            return
        val = params[key]
        is_file = key in params.get('filepaths', {})
        for obj in jf.objectives:
            fd = obj['func'].__dict__
            fd[key] = val
            if is_file and isinstance(fd.get('filepaths'), dict):
                fd['filepaths'][key] = val

    def _sync_joint_fixed_params(self, src_func):
        """Unify ALL fixed parameters (scalars + PDB ``fname``/filepaths) across every
        objective of ``self.fit`` (a JointFit) to ``src_func``'s values, so a freshly
        built/loaded joint fit enforces one shared structural hypothesis. Fit-params
        (lmfit) are untouched -- sharing of VARYING params stays governed by the spec."""
        jf = getattr(self, 'fit', None)
        if not hasattr(jf, 'objectives') or src_func is None:
            return
        sd = src_func.__dict__
        fitp = getattr(src_func, 'params', None)
        fp_keys = set(fitp.keys()) if fitp is not None else set()
        src_files = sd.get('filepaths', {}) or {}
        # keys that are fixed params: in __dict__, not a fit-param, not special/private
        fixed_keys = [k for k in sd.keys()
                      if k not in fp_keys and k not in self.special_keys and not k.startswith('_')]
        for obj in jf.objectives:
            if obj['func'] is src_func:
                continue
            fd = obj['func'].__dict__
            for k in fixed_keys:
                fd[k] = sd[k]
            if isinstance(fd.get('filepaths'), dict):
                for k in src_files:
                    fd['filepaths'][k] = src_files[k]


        
        
    def sfitParamChanged(self,row,col):
        txt=self.sfitParamTableWidget.item(row,0).text()
        try:
            val=float(self.sfitParamTableWidget.item(row,col).text())
        except:
            val=self.sfitParamTableWidget.item(row,col).text()
        if col==1:
            oldVal=self.fit.fit_params[txt].value
        elif col==2:
            oldVal=self.fit.fit_params[txt].min
        elif col==3:
            oldVal=self.fit.fit_params[txt].vary
        elif col==4:
            oldVal=self.fit.fit_params[txt].expr
        elif col==5:
            oldVal=self.fit.fit_params[txt].brute_step
        if isinstance(val,numbers.Number):
            if col==1:
                if val!=self.fit.fit_params[txt].value:
                    self.fit.params[txt]=val
                    self.fit.fit_params[txt].set(value=val)
                    self.fchanged=False
                    try:
                        self.sfitParamTableWidget.cellChanged.disconnect()
                    except:
                        pass
                    self.sfitParamTableWidget.item(row,col).setText(self.format%val)
                    self.sfitParamTableWidget.cellChanged.connect(self.sfitParamChanged)
                    self.update_plot()
            elif col==2:
                self.fit.fit_params[txt].set(min=val)
            elif col==3:
                self.fit.fit_params[txt].set(max=val)
            elif col==5:
                self.fit.fit_params[txt].set(brute_step=val)
        elif isinstance(val,str):
            if col==4:
                pval=self.fit.fit_params[txt].value
                if val == 'None':
                    self.fit.fit_params[txt].set(value=pval,expr = '')
                else:
                    self.fit.fit_params[txt].set(value=pval,expr = val)
                try:
                    self.fit.fit_params[txt].value
                except:
                    try:
                        self.sfitParamTableWidget.cellChanged.disconnect()
                    except:
                        pass
                    QMessageBox.warning(self, 'Expression Error', 'Please enter correct expression using only parameters and constants', QMessageBox.Ok)
                    if oldVal is None:
                        self.fit.fit_params[txt].set(value=pval,expr='')
                    else:
                        self.fit.fit_params[txt].set(value=pval,expr=oldVal)
                    self.sfitParamTableWidget.item(row, col).setText(str(oldVal))
                    self.sfitParamTableWidget.cellChanged.connect(self.sfitParamChanged)
            else:
                QMessageBox.warning(self,'Value Error','Please input numbers only',QMessageBox.Ok)
                try:
                    self.sfitParamTableWidget.cellChanged.disconnect()
                except:
                    pass
                self.sfitParamTableWidget.item(row,col).setText(str(oldVal))
                self.sfitParamTableWidget.cellChanged.connect(self.sfitParamChanged)
        if self.sfitParamTableWidget.item(row,1).checkState()==Qt.Checked:
            self.fit.fit_params[txt].vary=1
        else:
            self.fit.fit_params[txt].vary=0
        if col==1:
            self.sfitParamTableWidget.item(row, 1).setToolTip((txt + ' = '+self.format+'\u00B1 '+self.format) % (self.fit.fit_params[txt].value, 0.0))
        self.update_sfitSlider(row,col)
        self.update_sfit_parameters()
        self.update_mfit_parameters_new()
        self.sfitParamTableWidget.setCurrentCell(row,col)
        self.sfitParamTableWidget.resizeRowsToContents()
        self.sfitParamTableWidget.resizeColumnsToContents()

    def mfitParamChanged_new(self,row,col):
        index=self.mfitParamTabWidget.currentIndex()
        mkey=self.mfitParamTabWidget.tabText(index)
        try:
            self.mfitParamTableWidget[mkey].cellChanged.disconnect()
        except:
            pass
        txt = self.mfitParamTableWidget[mkey].item(row, col).text()
        pkey=self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
        key='__%s_%s_%03d' % (mkey,pkey,row)
        try:
            if col!=0:
                float(txt) # This is for checking the numbers entered to be float or not
                oldval = self.fit.fit_params[key].value
                self.mfitParamTableWidget[mkey].item(row, col).setText(self.format % (float(txt)))
                pchanged=True
                # if float(txt)!=self.fit.fit_params[key].value:
                #     pchanged=True
                #     self.mfitParamTableWidget[mkey].item(row,col).setText(self.format%(float(txt)))
                # else:
                #     self.mfitParamTableWidget[mkey].item(row, col).setText(self.format % (float(txt)))
                #     pchanged=False
                self.fit.fit_params[key].set(value=float(txt))
                if self.mfitParamTableWidget[mkey].item(row,col).checkState()==Qt.Checked:
                    self.fit.fit_params[key].set(vary=1)
                else:
                    self.fit.fit_params[key].set(vary=0)
                self.mfitParamData[mkey][row][col]=float(txt)
                self.fit.fit_params[key].set(value=float(txt))
                self.mfitParamTableWidget[mkey].item(row, col).setToolTip((key + ' = '+self.format+' \u00B1 '+self.format)
                                                                          % (self.fit.fit_params[key].value, 0.0))
                self.fit.params['__mpar__'][mkey][pkey][row]=float(txt)
            else:
                oldval = self.fit.params['__mpar__'][mkey][pkey][row]
                self.fit.params['__mpar__'][mkey][pkey][row] = txt
                self.mfitParamData[mkey][row][col] = txt
                pchanged=True
            self.fchanged=False
            if pchanged:
                try:
                    self.fit.func.output_params={'scaler_parameters': {}}
                    self.update_plot()
                except:
                    QMessageBox.warning(self, 'Value Error', 'The value you entered are not valid!', QMessageBox.Ok)
                    self.mfitParamTableWidget[mkey].item(row, col).setText(oldval)
                    self.fit.params['__mpar__'][mkey][pkey][row] = oldval
                    self.mfitParamData[mkey][row][col]=oldval
            self.update_mfitSlider(row,col)
        except:
            QMessageBox.warning(self,'Value Error', 'Please input numbers only!', QMessageBox.Ok)
            self.mfitParamTableWidget[mkey].item(row,col).setText(str(self.fit.fit_params[key].value))
        self.mfitParamTableWidget[mkey].cellChanged.connect(self.mfitParamChanged_new)
        self.update_fit_parameters()
        self.mfitParamTabWidget.setCurrentIndex(index)
        self.mfitParamTableWidget[mkey].setCurrentCell(row,col)
        item=self.mfitParamTableWidget[mkey].item(row,col)
        item.setSelected(True)
        self.mfitParamTableWidget[mkey].scrollToItem(item)
        self.mfitParamTableWidget[mkey].resizeRowsToContents()
        self.mfitParamTableWidget[mkey].resizeColumnsToContents()


        
    # def mfitParamChanged(self,row,col):
    #     parkey=self.mfitParamTableWidget.horizontalHeaderItem(col).text()
    #     txt=self.mfitParamTableWidget.item(row,col).text()
    #     key = '__%s__%03d' % (parkey, row)
    #     try:
    #         if col!=0:
    #             float(txt) # This is for checking the numbers entered to be float or not
    #             oldval = self.fit.fit_params[key].value
    #             if float(txt)!=self.fit.fit_params[key].value:
    #                 pchanged=True
    #                 self.mfitParamTableWidget.item(row,col).setText(self.format%(float(txt)))
    #             else:
    #                 self.mfitParamTableWidget.item(row, col).setText(self.format % (float(txt)))
    #                 pchanged=False
    #             self.fit.fit_params[key].set(value=float(txt))
    #             if self.mfitParamTableWidget.item(row,col).checkState()==Qt.Checked:
    #                 self.fit.fit_params[key].set(vary=1)
    #             else:
    #                 self.fit.fit_params[key].set(vary=0)
    #             self.mfitParamData[row][col]=float(txt)
    #             self.fit.fit_params[key].set(value=float(txt))
    #             self.mfitParamTableWidget.item(row, col).setToolTip((key + ' = '+self.format+' \u00B1 '+self.format) % (self.fit.fit_params[key].value, 0.0))
    #         else:
    #             oldval = self.fit.params['__mpar__'][parkey][row]
    #             self.fit.params['__mpar__'][parkey][row] = txt
    #             self.mfitParamData[row][col] = txt
    #             pchanged=True
    #         self.fchanged=False
    #         if pchanged:
    #             try:
    #                 self.update_plot()
    #             except:
    #                 QMessageBox.warning(self, 'Value Error', 'The value you entered are not valid!', QMessageBox.Ok)
    #                 self.mfitParamTableWidget.item(row, col).setText(oldval)
    #                 self.fit.params['__mpar__'][parkey][row] = oldval
    #                 self.mfitParamData[row][col]=oldval
    #         self.mfitParamTableWidget.resizeRowsToContents()
    #         self.mfitParamTableWidget.resizeColumnsToContents()
    #         self.update_mfitSlider(row,col)
    #     except:
    #         QMessageBox.warning(self,'Value Error', 'Please input numbers only!', QMessageBox.Ok)
    #         self.mfitParamTableWidget.item(row,col).setText(str(self.fit.fit_params[key].value))
        
            
    def xChanged(self):
        try:
            self.xLineEdit.returnPressed.disconnect()
        except:
            pass
        try:
            x=eval(self.xLineEdit.text())
            #x=np.array(x)
            try:
                self.fit.params['x']=x
                self.fit.set_x(x)
            #self.fit.imin=0
            #self.fit.imax=len(self.fit.x)
            except:
                pass
            self.fchanged=False
            if len(self.funcListWidget.selectedItems())>0:
                try:
                    stime = time.time()
                    self.fit.func.__fit__=False
                    self.fit.evaluate()
                    exectime = time.time() - stime
                except:
                    QMessageBox.warning(self, 'Value error',
                                        'Something wrong with the value of the parameter which you just entered.\n'+traceback.format_exc(),
                                        QMessageBox.Ok)
                    return
                try:
                    self.genParamListWidget.itemSelectionChanged.disconnect()
                except:
                    pass
                self.genParamListWidget.clear()
                self.fit.params['output_params']['scaler_parameters']['Exec-time (sec)'] = exectime
                self.fit.params['output_params']['scaler_parameters']['Chi-Sqr']=self.chisqr
                self.fit.params['output_params']['scaler_parameters']['Red_Chi_Sqr'] = self.red_chisqr
                if len(self.fit.params['output_params']) > 0:
                    for key in self.fit.params['output_params'].keys():
                        if key == 'scaler_parameters':
                            for k in self.fit.params['output_params'][key].keys():
                                self.genParamListWidget.addItem(k + ' : ' + str(self.fit.params['output_params'][key][k]))
                        else:
                            var=self._op_var_list(self.fit.params['output_params'][key])
                            self.genParamListWidget.addItem(str(key) + ' : ' + str(var))
                    # Restore the prior generated-param selection so adding/switching a
                    # dataset keeps the EDP (etc.) plotted. Gated on gen_param_items being
                    # non-empty (functionChanged clears it), NOT on fchanged -- fchanged
                    # stays True until a fit runs, which wrongly wiped the selection when a
                    # dataset was added before fitting. Text-match ensures only keys that
                    # still exist (same model) are restored.
                    if self.gen_param_items:
                        for i in range(self.genParamListWidget.count()):
                            item = self.genParamListWidget.item(i)
                            if item.text() in self.gen_param_items:
                                item.setSelected(True)
                    self._autoselect_image_params()
                    self.plot_extra_param()
                    self.genParamListWidget.itemSelectionChanged.connect(self.plot_extra_param)
                try:
                    pfnames=copy.copy(self.pfnames)
                except:
                    pfnames=[]
                if type(self.fit.x)==dict:
                    for key in self.fit.x.keys():
                        self.plotWidget.add_data(x=self.fit.x[key][self.fit.imin[key]:self.fit.imax[key] + 1], y=self.fit.yfit[key],
                                                 name=self.funcListWidget.currentItem().text()+':'+key, fit=True,
                                                 color=self._single_fit_color(key))
                    pfnames = pfnames + [self.funcListWidget.currentItem().text() + ':' + key for key in
                                             self.fit.x.keys()]
                else:
                    _fitname=self._indiv_fit_name()
                    self.plotWidget.add_data(x=self.fit.x[self.fit.imin:self.fit.imax + 1], y=self.fit.yfit,
                                             name=_fitname, fit=True,
                                             color=self._single_fit_color())
                    pfnames = pfnames + [_fitname]

                self.plotWidget.Plot(self._legend_order(pfnames))
                # QApplication.processEvents()
                QApplication.processEvents()
        except:
            QMessageBox.warning(self,'Value Error','The value just entered is not seem to be right.\n'+traceback.format_exc(),QMessageBox.Ok)
            self.xLineEdit.setText(self.xline)
        self.xLineEdit.returnPressed.connect(self.xChanged)


    def update_plot(self):
        # JointFit-aware early return: a joint fit has no top-level x/y/yfit and its
        # residual takes MASTER params, so the single-Fit body below cannot run on it.
        # _update_plot_joint draws the current objective's data + curve instead.
        # (getattr guard: self.fit may not exist yet before any function is chosen.)
        if hasattr(getattr(self, 'fit', None), 'objectives'):
            self._update_plot_joint()
            return
        for row in range(self.fixedParamTableWidget.rowCount()):
            _it0 = self.fixedParamTableWidget.item(row, 0)
            if _it0 is None:            # table mid-rebuild; nothing to sync yet
                continue
            txt=_it0.text()
            if txt in self.fit.params['choices'].keys():
                _cw = self.fixedParamTableWidget.cellWidget(row, 1)
                if _cw is None:         # combobox not installed for this row yet
                    continue
                val = _cw.currentText()
            elif txt in self.fit.params['filepaths'].keys():
                val = getattr(self, 'filebutton_paths', {}).get(txt) or self.fit.params.get(txt) or ''
            else:
                val=self.fixedParamTableWidget.item(row,1).text()
            try:
                self.fit.params[txt]=eval(val)
            except:
                self.fit.params[txt]=str(val)
            # keep the 'filepaths' snapshot in lock-step with the live value so it
            # never reverts to the __init__ default (e.g. per-dataset PDB path).
            if txt in self.fit.params['filepaths'].keys():
                self.fit.params['filepaths'][txt]=self.fit.params[txt]
        for row in range(self.sfitParamTableWidget.rowCount()):
            txt=self.sfitParamTableWidget.item(row,0).text()
            self.fit.params[txt]=float(self.sfitParamTableWidget.item(row,1).text())
            vary,min,max,expr,bs=self.fit.fit_params[txt].vary,self.fit.fit_params[txt].min,\
                                 self.fit.fit_params[txt].max,self.fit.fit_params[txt].expr,\
                                 self.fit.fit_params[txt].brute_step
            self.fit.fit_params[txt].set(value=float(self.sfitParamTableWidget.item(row,1).text()),vary=vary,min=min,
                                         max=max,expr=expr,brute_step=bs)
        for i in range(self.mfitParamTabWidget.count()):
            mkey=self.mfitParamTabWidget.tabText(i)
            for row in range(self.mfitParamTableWidget[mkey].rowCount()):
                pkey = self.mfitParamTableWidget[mkey].horizontalHeaderItem(0).text()
                txt = self.mfitParamTableWidget[mkey].item(row, 0).text()
                self.fit.params['__mpar__'][mkey][pkey][row] = txt
                for col in range(1,self.mfitParamTableWidget[mkey].columnCount()):
                    pkey=self.mfitParamTableWidget[mkey].horizontalHeaderItem(col).text()
                    txt=self.mfitParamTableWidget[mkey].item(row,col).text()
                    tkey='__%s_%s_%03d'%(mkey,pkey,row)
                    vary,min,max,expr,bs=self.fit.fit_params[tkey].vary,self.fit.fit_params[tkey].min,\
                                         self.fit.fit_params[tkey].max,self.fit.fit_params[tkey].expr,\
                                         self.fit.fit_params[tkey].brute_step
                    self.fit.fit_params['__%s_%s_%03d'%(mkey,pkey,row)].set(value=float(txt),min=min,max=max,vary=vary,expr=expr,brute_step=bs)
        try:
            pfnames=copy.copy(self.pfnames)
        except:
            pfnames=[]
        self.chisqr='None'
        self.red_chisqr='None'
        _pi0, _pi1 = 0, None       # slice actually applied below, for the mask
        if len(self.dataListWidget.selectedItems()) > 0:
            if len(self.data[self.sfnames[-1]].keys()) > 1:
                x = {}
                y = {}
                yerr = {}
                for key in self.data[self.sfnames[-1]].keys():
                    x[key] = self.data[self.sfnames[-1]][key]['x']
                    y[key] = self.data[self.sfnames[-1]][key]['y']
                    xmin,xmax=self.xmin[self.sfnames[-1]],self.xmax[self.sfnames[-1]]
                    y[key] = y[key][np.argwhere(x[key] >= xmin)[0][0]:np.argwhere(x[key] <= xmax)[-1][0]+1]
                    yerr[key] = self.data[self.sfnames[-1]][key]['yerr']
                    yerr[key] = yerr[key][np.argwhere(x[key] >= xmin)[0][0]:np.argwhere(x[key] <= xmax)[-1][0]+1]
                    _pi0 = np.argwhere(x[key] >= xmin)[0][0]
                    _pi1 = np.argwhere(x[key] <= xmax)[-1][0]+1
                    x[key] = x[key][_pi0:_pi1]
            else:
                key = list(self.data[self.sfnames[-1]].keys())[0]
                xmin, xmax = self.xmin[self.sfnames[-1]], self.xmax[self.sfnames[-1]]
                x = self.data[self.sfnames[-1]][key]['x']
                y = self.data[self.sfnames[-1]][key]['y']
                y = y[np.argwhere(x >= xmin)[0][0]:np.argwhere(x <= xmax)[-1][0]+1]
                yerr = self.data[self.sfnames[-1]][key]['yerr']
                yerr = yerr[np.argwhere(x >= xmin)[0][0]:np.argwhere(x <= xmax)[-1][0]+1]
                _pi0 = np.argwhere(x >= xmin)[0][0]
                _pi1 = np.argwhere(x <= xmax)[-1][0]+1
                x = x[_pi0:_pi1]

        if len(self.funcListWidget.selectedItems())>0:
            try:
                if self.autoCalculate:
                    stime=time.perf_counter()
                    self.fit.func.__fit__=False
                    self.fit.evaluate()
                    ntime=time.perf_counter()
                    exectime=ntime-stime
                else:
                    exectime=0.0
            except:
                QMessageBox.warning(self, 'Evaluation Error', traceback.format_exc(), QMessageBox.Ok)
                self.fit.yfit = self.fit.func.x
                exectime=np.nan
            if len(self.dataListWidget.selectedItems()) > 0:
                # x/y/yerr were pre-sliced to [xmin,xmax] above, so the mask has
                # to be sliced identically (see Fit.imask_for on the two paths)
                self._cov_set_x(self.fit, self.sfnames[-1], x, y=y, yerr=yerr,
                                mask=self._column_mask(self.sfnames[-1],
                                                       slice_=(_pi0, _pi1)), slice_=(_pi0, _pi1))
                self._cov_plot_data(self.sfnames[-1])     # legacy covariance: shows y_corr
                try:
                    residual = self.fit.residual(self.fit.fit_params, self.fitScaleComboBox.currentText())

                    self.chisqr = np.sum(residual ** 2)
                    vary=[self.fit.fit_params[key].vary for key in self.fit.fit_params.keys()]
                    self.red_chisqr=self.chisqr/(len(residual)-np.sum(vary))
                except:
                    QMessageBox.warning(self, 'Evaluation Error', traceback.format_exc(), QMessageBox.Ok)
                    self.chisqr=None
                    self.red_chisqr=None

            try:
                self.genParamListWidget.itemSelectionChanged.disconnect()
            except:
                pass
            self.fitResultTextEdit.clear()
            try:
                self.fitResultTextEdit.append(self.fit_report)
            except:
                self.fitResultTextEdit.clear()
            self.genParamListWidget.clear()
            self.fit.params['output_params']['scaler_parameters']['Exec-time (sec)'] = exectime
            self.fit.params['output_params']['scaler_parameters']['Chi-Sqr'] = self.chisqr
            self.fit.params['output_params']['scaler_parameters']['Red_Chi_Sqr'] = self.red_chisqr
            self._cov_label_objective()
            if type(self.fit.x)==dict:
                for key in self.fit.x.keys():
                    self.plotWidget.add_data(x=self.fit.x[key][self.fit.imin[key]:self.fit.imax[key] + 1], y=self.fit.yfit[key],
                                             name=self.funcListWidget.currentItem().text()+':'+key, fit=True,
                                             color=self._single_fit_color(key))
                    if len(self.dataListWidget.selectedItems()) > 0:
                        self.fit.params['output_params']['Residuals_%s' % key] = {
                            'x': self.fit.x[key][self.fit.imin[key]:self.fit.imax[key] + 1],
                            'y': (self.fit.y[key][self.fit.imin[key]:self.fit.imax[key] + 1] - self.fit.yfit[key])
                                 / self.fit.yerr[key][self.fit.imin[key]:self.fit.imax[key] + 1]}
                    # else:
                    #     self.fit.params['output_params']['Residuals_%s' % key]={'x':self.fit.x[key][self.fit.imin[key]:self.fit.imax[key] + 1],
                    #                                                             'y':np.zeros_like(self.fit.x[key][self.fit.imin[key]:self.fit.imax[key] + 1])}
                pfnames = pfnames + [self.funcListWidget.currentItem().text() + ':' + key for key in
                                         self.fit.x.keys()]
            else:
                _fitname=self._indiv_fit_name()
                self.plotWidget.add_data(x=self.fit.x[self.fit.imin:self.fit.imax + 1], y=self.fit.yfit,
                                         name=_fitname, fit=True,
                                         color=self._single_fit_color())
                if len(self.dataListWidget.selectedItems()) > 0:
                    self.fit.params['output_params']['Residuals'] = {'x': self.fit.x[self.fit.imin:self.fit.imax + 1],
                                                                     'y': (self.fit.y[
                                                                           self.fit.imin:self.fit.imax + 1] - self.fit.yfit) / self.fit.yerr[
                                                                                                                               self.fit.imin:self.fit.imax + 1]}
                    if getattr(self.fit, 'cov', None) is not None:   # plot the whitened residual
                        self._cov_whitened_residuals(locals().get('residual'))
                # else:
                #     self.fit.params['output_params']['Residuals'] = {'x': self.fit.x[self.fit.imin:self.fit.imax + 1],
                #                                                      'y':np.zeros_like(self.fit.x[self.fit.imin:self.fit.imax + 1])}
                pfnames=pfnames+[_fitname]
            if len(self.fit.params['output_params'])>0:
                row=0
                for key in self.fit.params['output_params'].keys():
                    if key=='scaler_parameters':
                        for k in self.fit.params['output_params'][key].keys():
                            self.genParamListWidget.addItem(k + ' : ' + str(self.fit.params['output_params'][key][k]))
                            it=self.genParamListWidget.item(row)
                            it.setFlags(it.flags() & ~Qt.ItemIsSelectable)
                            row+=1
                    else:
                        var = self._op_var_list(self.fit.params['output_params'][key])
                        self.genParamListWidget.addItem(
                            str(key) + ' : ' + str(var))
                        row+=1
                if self.gen_param_items:   # restore prior selection (see update_plot note)
                    for i in range(self.genParamListWidget.count()):
                        item = self.genParamListWidget.item(i)
                        if item.text() in self.gen_param_items:
                            item.setSelected(True)
            self._autoselect_image_params()
            self.plot_extra_param()
            self.genParamListWidget.itemSelectionChanged.connect(self.plot_extra_param)

            self.consoleWidget.push_vars({'fit': self.fit})

        # Multi-select individual view: the block above only evaluates/plots the ACTIVE
        # dataset (sfnames[-1]). Overlay each OTHER selected dataset's OWN fitted curve so
        # all selected datasets show their individual fits together. Each is named/colored
        # per dataset so the curves don't collide (previously they shared one name and only
        # the last showed). Best-effort: a hiccup on one dataset must not break the plot.
        try:
            _sel = [it.text() for it in self.dataListWidget.selectedItems()]
        except Exception:
            _sel = []
        if len(_sel) > 1:
            for _dname in _sel:
                if _dname == self.sfnames[-1]:
                    continue  # already drawn as the active fit above
                _dfit = self.dataset_fits.get(_dname)
                if _dfit is None or hasattr(_dfit, 'objectives'):
                    continue
                if not (getattr(_dfit, '_is_fitted', False) or getattr(_dfit, 'result', None) is not None):
                    continue
                try:
                    _fnum = _dname.split('<>')[0]
                    _fname = getattr(_dfit, '_xmod_funcname', None) or self.funcListWidget.currentItem().text()
                    _keys = list(self.data[_dname].keys())
                    _xmn, _xmx = self.xmin[_dname], self.xmax[_dname]
                    for _k in _keys:
                        _xa = self.data[_dname][_k]['x']
                        _i0 = np.argwhere(_xa >= _xmn)[0][0]
                        _i1 = np.argwhere(_xa <= _xmx)[-1][0] + 1
                        _xs = _xa[_i0:_i1]
                        _ys = self.data[_dname][_k]['y'][_i0:_i1]
                        _es = self.data[_dname][_k]['yerr'][_i0:_i1]
                        # evaluate this dataset's stored fit on its own range
                        if len(_keys) > 1:
                            continue  # multi-column datasets: skip overlay (handled per active view)
                        _dm = self._column_mask(_dname, slice_=(_i0, _i1))
                        _dfit.set_x(_xs, y=_ys, yerr=_es, mask=_dm)
                        _dfit.func.__fit__ = False
                        _yf = _dfit.evaluate()
                        _cname = '%s [indiv:%s]' % (_fname, _fnum)   # index-bearing -> sorts + labels
                        self.plotWidget.add_data(x=_xs, y=_yf, name=_cname, fit=True,
                                                 color=self._curve_color_for(_dname, _k))
                        pfnames.append(_cname)
                except Exception:
                    pass
            # Now that every selected dataset's model has been evaluated (output_params
            # populated), re-render the bottom panel so its ED profiles co-plot across
            # ALL selected datasets (the earlier plot_extra_param ran before this loop).
            try:
                self.plot_extra_param()
            except Exception:
                pass
        self.plotWidget.Plot(self._legend_order(pfnames))
        # QApplication.processEvents()
        QApplication.processEvents()

    def _cov_whitened_residuals(self, residual):
        """With a covariance active, the plotted residual is the whitened one the fit
        minimised, against x of its rows, labelled as such (spec 17 6.2)."""
        try:
            idx = self.fit.cov_residual_index(self.fitScaleComboBox.currentText())
        except Exception:
            idx = None
        if idx is None or residual is None or len(idx) != len(residual):
            return
        self.fit.params['output_params']['Residuals'] = {'x': np.asarray(self.fit.x)[idx],
                                                         'y': np.asarray(residual),
                                                         'names': ['x', 'Residuals (whitened)']}

    def _cov_label_objective(self):
        """With a covariance active, a 'Chi-Sqr objective' entry says Chi-Sqr and
        Red_Chi_Sqr are of the whitened residual (spec 17 6.2); removed otherwise, so a
        fit without a covariance shows exactly the stock entries."""
        sp = self.fit.params['output_params']['scaler_parameters']
        try:
            on = self.fit.cov_residual_index(self.fitScaleComboBox.currentText()) is not None
        except Exception:
            on = False
        if on:
            sp['Chi-Sqr objective'] = 'whitened (covariance)'
        else:
            sp.pop('Chi-Sqr objective', None)

    def _eval_individual_curve(self, name):
        """(x, y, color) for dataset ``name``'s stored INDIVIDUAL fit curve, or None.

        Evaluated in ISOLATION -- it never touches ``self.fit`` or the param tables --
        over the dataset's current fit range, so the individual best-fit curve can be
        overlaid alongside the dataset's JOINT curve for comparison. Returns None when
        the dataset has no stored individual fit result."""
        f = self.dataset_fits.get(name)
        if f is None or hasattr(f, 'objectives'):
            return None
        func = getattr(f, 'func', None)
        if func is None or name not in self.data:
            return None
        if not (getattr(f, '_is_fitted', False) or getattr(f, 'result', None) is not None):
            return None
        first_key = list(self.data[name].keys())[0]
        x = np.asarray(self.data[name][first_key]['x'])
        try:
            xmin, xmax = self.xmin[name], self.xmax[name]
            m = (x >= xmin) & (x <= xmax)
            xin = x[m] if np.any(m) else x
        except Exception:
            xin = x
        try:
            func.x = xin
            func.__fit__ = False
            y = np.asarray(func.y()).ravel()
        except Exception:
            return None
        if y.shape != xin.shape:
            return None
        return xin, y, self._curve_color_for(name, first_key)

    def _plot_combined_overlay(self, main_sel, joint_sel):
        """Overlay, in the main plot, each dataset's data + its JOINT fit (solid) + its
        INDIVIDUAL fit (dashed), for the datasets highlighted in BOTH the joint-fit list
        (``joint_sel``) and the Data list (``main_sel``). Purely additive: it does not
        change ``self.fit`` or the param tables, so whichever view last populated them
        stays. Returns True if it drew a combined plot."""
        jf = getattr(self, 'jointFit', None)
        if jf is None or not getattr(jf, 'objectives', None):
            return False
        onames = list(getattr(jf, 'objective_names', []))
        pfnames = []
        shown = set()

        def _add_data_curves(name):
            fnum = name.split('<>')[0]
            first_key = None
            for key in self.data.get(name, {}):
                if first_key is None:
                    first_key = key
                if (name, key) in shown:
                    continue
                d = self.data[name][key]
                col = d.get('color') or self._curve_color_for(name, key)
                self.plotWidget.add_data(d['x'], d['y'], yerr=d['yerr'],
                                         name='%s:%s' % (fnum, key), color=col)
                pfnames.append('%s:%s' % (fnum, key))
                shown.add((name, key))
            return fnum, first_key

        # JOINT curves (solid) for datasets selected in the joint list
        for name in joint_sel:
            if name not in onames or name not in self.data:
                continue
            k = onames.index(name)
            obj = jf.objectives[k]
            xall = np.asarray(obj['x'])
            imin = obj.get('imin', 0) or 0
            imax = obj.get('imax', None)
            if imax is None or imax >= len(xall):
                imax = len(xall) - 1
            xin = xall[imin:imax + 1]
            func = obj['func']
            func.x = xin
            func.__fit__ = False
            try:
                yfit = np.asarray(func.y()).ravel()
            except Exception:
                continue
            fnum, first_key = _add_data_curves(name)
            jname = '%s [joint:%s]' % (type(func).__name__, fnum)
            self.plotWidget.add_data(x=xin, y=yfit, name=jname, fit=True,
                                     color=self._curve_color_for(name, first_key))
            pfnames.append(jname)

        # INDIVIDUAL curves (dashed) for datasets selected in the Data list
        for name in main_sel:
            if name not in self.data:
                continue
            fnum, _ = _add_data_curves(name)
            ind = self._eval_individual_curve(name)
            if ind is None:
                continue
            xin, yin, col = ind
            iname = '%s [indiv:%s]' % (type(self.dataset_fits[name].func).__name__, fnum)
            self.plotWidget.add_data(x=xin, y=yin, name=iname, fit=True,
                                     color=col, linestyle=Qt.DashLine)
            pfnames.append(iname)

        if not pfnames:
            return False
        # NB: do NOT overwrite self.pfnames (kept as data-only, like _update_plot_joint)
        # so remove_data(self.pfnames) flows never strip the overlaid fit curves.
        self.plotWidget.Plot(self._legend_order(pfnames))
        # Bottom panel: now that BOTH each dataset's joint and individual models have
        # been evaluated (above), refresh the generated-param plot so ED profiles etc.
        # co-plot the individual (dashed) + joint (solid) sources to match this overlay.
        try:
            self.plot_extra_param()
        except Exception:
            pass
        QApplication.processEvents()
        return True

    def _maybe_combined_overlay(self):
        """If a dataset is highlighted in BOTH the Data list and the joint-fit list, draw
        the individual+joint overlay (item B2). Called at the end of both selection
        handlers; a no-op unless both lists have a selection and a JointFit exists."""
        if getattr(self, '_syncing_selection', False):
            return False
        jp = getattr(self, 'jointFitPanel', None)
        jf = getattr(self, 'jointFit', None)
        if jp is None or jf is None:
            return False
        try:
            joint_sel = [it.text() for it in jp.datasetList.selectedItems()]
            main_sel = [it.text() for it in self.dataListWidget.selectedItems()]
        except Exception:
            return False
        if joint_sel and main_sel:
            return self._plot_combined_overlay(main_sel, joint_sel)
        return False

    def _update_plot_joint(self, overlay_names=None):
        """Draw joint-fit objective(s) in the main plot: each selected dataset's data
        curve(s) + its model curve. ``overlay_names`` (a list) overlays all of them; the
        DETAIL objective (JointFit._current) drives the param tables + output_params /
        genParamListWidget. Separate from update_plot because a JointFit has no single-Fit
        x/y/yfit and its residual needs master params. Never touches the Data-list
        selection (pfnames are built from the joint dataset names directly, so we don't
        re-enter dataFileSelectionChanged and flip back to the individual fit)."""
        jf = self.fit
        objectives = getattr(jf, 'objectives', None)
        if not objectives:
            return
        onames = jf.objective_names
        cur = getattr(jf, '_current', 0) or 0
        if cur >= len(objectives):
            cur = 0
        idxs = [onames.index(n) for n in (overlay_names or []) if n in onames]
        if not idxs:
            idxs = [cur]
        detail_i = cur if cur in idxs else idxs[-1]
        # remember which datasets are overlaid so the generated-param (EDP) plot can
        # co-plot the SAME profile from every shown dataset (see plot_extra_param).
        self._joint_overlay_names = [onames[k] for k in idxs]

        def _eval(k):
            obj = objectives[k]
            xall = np.asarray(obj['x'])
            imin = obj.get('imin', 0) or 0
            imax = obj.get('imax', None)
            if imax is None or imax >= len(xall):
                imax = len(xall) - 1
            xin = xall[imin:imax + 1]
            yin = np.asarray(obj['y'])[imin:imax + 1]
            yerrin = np.asarray(obj['yerr'])[imin:imax + 1]
            func = obj['func']
            func.x = xin
            func.__fit__ = False
            try:
                yfit = np.asarray(func.y())
            except Exception:
                QMessageBox.warning(self, 'Evaluation Error', traceback.format_exc(), QMessageBox.Ok)
                yfit = np.zeros_like(xin)
            return obj, func, xin, yin, yerrin, yfit

        pfnames = []
        detail = None
        for k in idxs:
            obj, func, xin, yin, yerrin, yfit = _eval(k)
            name = obj['name']
            if name not in self.data:
                continue
            fnum = name.split('<>')[0]
            first_key = None
            for key in self.data[name].keys():
                if first_key is None:
                    first_key = key
                d = self.data[name][key]
                col = d.get('color') or self._curve_color_for(name, key)
                self.plotWidget.add_data(d['x'], d['y'], yerr=d['yerr'],
                                         name='%s:%s' % (fnum, key), color=col)
                pfnames.append('%s:%s' % (fnum, key))
            fitname = '%s [joint:%s]' % (type(func).__name__, fnum)
            # fit line shares the color of this dataset's primary data curve
            self.plotWidget.add_data(x=xin, y=yfit, name=fitname, fit=True,
                                     color=self._curve_color_for(name, first_key))
            pfnames.append(fitname)
            if k == detail_i:
                detail = (func, xin, yin, yerrin, yfit)

        if detail is None:
            self.plotWidget.Plot(self._legend_order(pfnames))
            QApplication.processEvents()
            return
        func, xin, yin, yerrin, yfit = detail
        try:
            self.chisqr = float(np.sum(((yin - yfit) / yerrin) ** 2))
            self.red_chisqr = self.chisqr / max(1, len(yin))
        except Exception:
            self.chisqr = None
            self.red_chisqr = None
        self.fitResultTextEdit.clear()
        try:
            self.fitResultTextEdit.append(self.fit_report)
        except Exception:
            pass
        try:
            self.genParamListWidget.itemSelectionChanged.disconnect()
        except Exception:
            pass
        self.genParamListWidget.clear()
        op = func.__dict__.setdefault('output_params', {})
        op.setdefault('scaler_parameters', {})
        op['scaler_parameters']['Chi-Sqr'] = self.chisqr
        op['scaler_parameters']['Red_Chi_Sqr'] = self.red_chisqr
        try:
            op['Residuals'] = {'x': xin, 'y': (yin - yfit) / yerrin}
        except Exception:
            pass
        row = 0
        for key in op.keys():
            if key == 'scaler_parameters':
                for k in op[key].keys():
                    self.genParamListWidget.addItem(k + ' : ' + str(op[key][k]))
                    it = self.genParamListWidget.item(row)
                    it.setFlags(it.flags() & ~Qt.ItemIsSelectable)
                    row += 1
            else:
                var = self._op_var_list(op[key])
                self.genParamListWidget.addItem(str(key) + ' : ' + str(var))
                row += 1
        if self.gen_param_items:   # restore prior selection (see update_plot note)
            for j in range(self.genParamListWidget.count()):
                item = self.genParamListWidget.item(j)
                if item.text() in self.gen_param_items:
                    item.setSelected(True)
        self._autoselect_image_params()
        self.plot_extra_param()
        self.genParamListWidget.itemSelectionChanged.connect(self.plot_extra_param)
        try:
            self.consoleWidget.push_vars({'fit': self.fit})
        except Exception:
            pass
        self.plotWidget.Plot(self._legend_order(pfnames))
        QApplication.processEvents()

    def extra_param_doubleClicked(self,item):
        key=item.text().split(':')[0].strip()
        if key in self.fit.params['output_params'].keys():
            if 'x' in self.fit.params['output_params'][key].keys() and 'y' in self.fit.params['output_params'][key].keys():
                x=self.fit.params['output_params'][key]['x']
                y=self.fit.params['output_params'][key]['y']
                if 'meta' in self.fit.params['output_params'][key].keys():
                    meta = self.fit.params['output_params'][key]['meta']
                else:
                    meta = {}
                if 'yerr' in self.fit.params['output_params'][key].keys():
                    yerr=self.fit.params['output_params'][key]['yerr']

                    if 'names' in self.fit.params['output_params'][key].keys():
                        meta['col_names']=self.fit.params['output_params'][key]['names']
                        data = {'data': pd.DataFrame(list(zip(x, y, yerr)), columns=self.fit.params['output_params'][key]['names']),
                            'meta': meta}
                    else:
                        meta['col_names']=['x', 'y', 'yerr']
                        data = {'data': pd.DataFrame(list(zip(x, y, yerr)), columns=['x', 'y', 'yerr']),
                            'meta': meta}

                else:
                    if 'names' in self.fit.params['output_params'][key].keys():
                        meta['col_names'] = self.fit.params['output_params'][key]['names']
                        data = {'data': pd.DataFrame(list(zip(x, y)), columns=self.fit.params['output_params'][key]['names']),
                                'meta': meta}
                    else:
                        meta['col_names'] = ['x', 'y', 'yerr']
                        data = {'data': pd.DataFrame(list(zip(x, y)), columns=['x', 'y']),
                                'meta': meta}
                data_dlg = Data_Dialog(data=data, parent=self, expressions={},
                                       plotIndex=None, colors=None)
                data_dlg.setModal(True)
                data_dlg.closePushButton.setText('Cancel')
                data_dlg.tabWidget.setCurrentIndex(0)
                data_dlg.dataFileLineEdit.setText('None')
                data_dlg.exec_()


    def _model_output_params(self, name, which='joint'):
        """The ``output_params`` dict for dataset ``name``'s JOINT or INDIVIDUAL model,
        or None. Read-only (the models were freshly evaluated by the plot that populated
        the generated-param list), so profiles like the EDP are already current."""
        if which == 'individual':
            f = self.dataset_fits.get(name)
            if f is None or hasattr(f, 'objectives'):
                return None
            if not (getattr(f, '_is_fitted', False) or getattr(f, 'result', None) is not None):
                return None
            func = getattr(f, 'func', None)
        else:
            jf = getattr(self, 'jointFit', None)
            if jf is None:
                return None
            onames = list(getattr(jf, 'objective_names', []))
            if name not in onames:
                return None
            func = jf.objectives[onames.index(name)]['func']
        if func is None:
            return None
        return func.__dict__.get('output_params', {})

    def plot_extra_param(self):
        """Plot the selected generated parameters (e.g. EDP/ADP profiles) in the bottom
        panel, mirroring the main data/fit plot's overlay model:

          * combined view (a dataset Ctrl-selected in BOTH lists) -> co-plot the profile
            from that dataset's INDIVIDUAL model (dashed) AND its JOINT model (solid),
            in the dataset's color -- the bottom-panel analogue of item B2;
          * joint view with >1 dataset overlaid -> the profile from each dataset's joint
            model (solid), per-dataset color (item B3);
          * otherwise a single curve from ``self.fit``.
        Each source contributes every SELECTED profile, so ED profiles AND any other
        generated plots co-plot together."""
        fdata=[]
        jfit=getattr(self,'jointFit',None)
        main_sel=[it.text() for it in self.dataListWidget.selectedItems()]
        try:
            joint_sel=[it.text() for it in self.jointFitPanel.datasetList.selectedItems()] \
                if jfit is not None else []
        except Exception:
            joint_sel=[]
        combined=bool(jfit is not None and main_sel and joint_sel)

        # Build the co-plot sources: (output_params, color, linestyle, name_suffix).
        sources=[]
        def _color(name):
            fk=list(self.data[name].keys())[0] if name in self.data else None
            return self._curve_color_for(name,fk) if fk else None
        if combined:
            for name in joint_sel:                       # joint profiles: solid
                op=self._model_output_params(name,'joint')
                if op is not None:
                    sources.append((op,_color(name),None,' [joint:%s]'%name.split('<>')[0]))
            for name in main_sel:                        # individual profiles: dashed
                op=self._model_output_params(name,'individual')
                if op is not None:
                    sources.append((op,_color(name),Qt.DashLine,' [indiv:%s]'%name.split('<>')[0]))
        else:
            jf=getattr(self,'fit',None)
            joint_active=hasattr(jf,'objectives')
            overlay=list(getattr(self,'_joint_overlay_names',[])) if joint_active else []
            if joint_active and len(overlay)>1:
                for name in overlay:                     # joint multi-dataset (B3)
                    op=self._model_output_params(name,'joint')
                    if op is not None:
                        sources.append((op,_color(name),None,' [%s]'%name.split('<>')[0]))
            elif (not joint_active) and len(main_sel)>1:
                # INDIVIDUAL multi-select: co-plot each FITTED selected dataset's EDP
                # (solid, per-dataset color). Unfit datasets are SKIPPED -- their model
                # holds a clone of another dataset's params (no fit of their own), so
                # plotting their EDP would show the wrong profile in the wrong color.
                # (Single-select still plots via the else branch, for fit setup.)
                for name in main_sel:
                    if not self._dataset_is_fitted(name):
                        continue
                    op=self._model_output_params(name,'individual')
                    if op:
                        sources.append((op,_color(name),None,' [%s]'%name.split('<>')[0]))
            else:
                sources.append((self.fit.params['output_params'],None,None,''))

        def _plot_one(txt,axes,op,cname,color=None,linestyle=None):
            if txt not in op:
                return None
            try:
                x=np.asarray(op[txt][axes[0]]); y=np.asarray(op[txt][axes[1]])
            except Exception:
                return None
            # Shape guards. Without these a 2-D entry reaches PlotWidget.add_data and
            # pops a "dimensions of x, y or yerr are not matching" dialog, and axes[2]
            # of a legacy ['x','y','z'] mesh entry is silently plotted AS THE ERRORBAR.
            # The ndim test is not redundant with the length test: add_data only checks
            # len(), so a wrong-shaped yerr whose first dimension happens to match is
            # accepted without complaint.
            if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
                return None
            try:
                yerr=np.asarray(op[txt][axes[2]],dtype=float)
                if yerr.ndim != 1 or yerr.shape[0] != y.shape[0]:
                    yerr=None
            except Exception:
                yerr=None
            self.extra_param_1DplotWidget.add_data(x=x,y=y,yerr=yerr,name=cname,fit=True,
                                                   color=color,linestyle=linestyle)
            if op[txt].get('plotType')=='step':
                try:
                    self.extra_param_1DplotWidget.data[cname].opts['stepMode']='left'
                except Exception:
                    pass
            fdata.append(cname)
            return op[txt]

        images=[]                       # [(display_name, entry)] for the 2D Map tab
        for item in self.genParamListWidget.selectedItems():
            txt,axes=item.text().split(':')
            txt=txt.strip()
            try:
                axes=eval(axes)
            except Exception:
                continue
            if type(axes)==list and len(axes)>=2:
                # 2-D map entries go to the image panel and NEVER to a 1-D PlotWidget.
                # Reaching _plot_one with one would plot axes[2] as the errorbar.
                hit=False
                for (op,color,ls,suffix) in sources:
                    e=op.get(txt) if hasattr(op,'get') else None
                    if self._op_is_image(e):
                        hit=True
                        cname=txt if suffix=='' else (txt+suffix)
                        e=dict(e)
                        # one ROW per source, so a joint fit over N datasets gives
                        # N rows of (data|model|residual) rather than 3N panels
                        e['panel']='%s|%s'%(e.get('panel','map'),suffix)
                        images.append((cname,e))
                if hit:
                    continue
                src=None
                for (op,color,ls,suffix) in sources:
                    cname=txt if suffix=='' else (txt+suffix)
                    r=_plot_one(txt,axes,op,cname,color=color,linestyle=ls)
                    if r is not None:
                        src=r
                if src is not None and 'names' in src:
                    self.extra_param_1DplotWidget.setXLabel(src['names'][0],fontsize=5)
                    self.extra_param_1DplotWidget.setYLabel(src['names'][1],fontsize=5)
                else:
                    self.extra_param_1DplotWidget.setXLabel('x',fontsize=5)
                    self.extra_param_1DplotWidget.setYLabel('y',fontsize=5)
        self.extra_param_1DplotWidget.Plot(self._legend_order(fdata))
        try:
            if images:
                self.imageWidget.set_maps(images)
                # Surface the producer's own ERROR_/WARNING_ scalers ON the panel.
                # A blank map is nearly always a configuration mistake that the
                # model can name precisely (wrong file loaded, x not a row index),
                # and leaving that buried in the alphabetical generated-parameter
                # list is how "the 2D view is broken" happens.
                msgs=[]
                for (op,_c,_l,_s) in sources:
                    sp=(op.get('scaler_parameters') or {}) if hasattr(op,'get') else {}
                    for k in sorted(sp):
                        if k.startswith('ERROR_') or k.startswith('WARNING_'):
                            v=str(sp[k]).strip()
                            if v and v not in msgs:
                                msgs.append(v)
                self.imageWidget.set_warning('  |  '.join(msgs))
                self._show_2D_tab(True, activate=(len(fdata)==0))
            else:
                self.imageWidget.clear_maps()
                self._show_2D_tab(False)
        except Exception:
            pass
        self.gen_param_items=[item.text() for item in self.genParamListWidget.selectedItems()]
        QApplication.processEvents()

    jointFitEnabled = False

    def create_jointFitDock(self):
        # Overlay: the joint-fit panel sits UNDER the Data list (not a separate tab),
        # in a scroll area so its (collapsible) body never dominates the dock.
        self.jointFitPanel = JointFitPanel(parent_app=self)
        from PyQt5.QtWidgets import QScrollArea
        self.jointFitScroll = QScrollArea()
        self.jointFitScroll.setWidgetResizable(True)
        self.jointFitScroll.setWidget(self.jointFitPanel)
        self.dataLayoutWidget.nextRow()
        self.dataLayoutWidget.addWidget(self.jointFitScroll, colspan=2)

    # ---- contract the JointFitPanel calls (see joint_fit_panel.py docstring) ----
    def selected_data_names(self):
        return [it.text() for it in self.dataListWidget.selectedItems()]

    def joint_model_name(self):
        fit = getattr(self, 'fit', None)
        if fit is not None and getattr(fit, 'func', None) is not None \
           and not hasattr(fit, 'objectives'):
            return type(fit.func).__name__
        it = self.funcListWidget.currentItem()
        return it.text() if it is not None else None

    def joint_model_factory(self):
        """A FRESH model instance for one dataset of the joint fit.

        Clones the LIVE current model (self.fit.func) so it works whether the
        function was clicked OR loaded from a .par -- a .par load sets the model via
        a signal dance that does not reliably leave curr_module / the func-list
        selection in the state a from-scratch reconstruction needs. Cloning the
        already-built model sidesteps that and carries the loaded layer structure +
        starting values. Falls back to reconstruction from the function list only if
        no single-fit model is live yet."""
        import copy
        fit = getattr(self, 'fit', None)
        # a live single-fit model (a JointFit also has .func, so exclude it)
        if fit is not None and getattr(fit, 'func', None) is not None \
           and not hasattr(fit, 'objectives'):
            m = copy.deepcopy(fit.func)
            m.x = self.x
            if getattr(m, 'params', None) is None:
                m.init_params()
            return m
        # fallback: reconstruct from the current function-list selection
        it = self.funcListWidget.currentItem()
        if it is None:
            return None
        module = 'Functions.%s.%s' % (self.curr_category, self.curr_module)
        cls = getattr(self.curr_funcClass[module], it.text())
        m = cls()
        m.x = self.x
        m.init_params()
        return m

    def joint_model_for(self, name):
        """Seed model for one dataset entering the joint fit.

        Prefer that dataset's remembered INDIVIDUAL fit (Phase-1 per-dataset store),
        so each dataset enters the joint fit at its own individual best-fit; fall
        back to the current view (joint_model_factory) if it has no individual fit."""
        import copy
        fit = self.dataset_fits.get(name)
        if fit is not None and getattr(fit, 'func', None) is not None \
           and not hasattr(fit, 'objectives'):
            m = copy.deepcopy(fit.func)
            m.x = self.x
            if getattr(m, 'params', None) is None:
                m.init_params()
            return m
        return self.joint_model_factory()

    def show_joint_dataset(self, name):
        """Panel selector/list -> make the JointFit the active view for `name` and paint
        the main window (tables + plot) with that dataset's joint-recovered fit."""
        self.show_joint_datasets([name])

    def show_joint_datasets(self, names):
        """Show one or more joint-fit datasets. The LAST name is the detail dataset that
        drives the stock param tables; ALL selected datasets' curves are overlaid in the
        plot. Called by the joint panel's dataset list (multi-select) and combo."""
        jf = getattr(self, 'jointFit', None)
        if jf is None or not hasattr(jf, 'set_current') or not names:
            return
        try:
            self.fit = jf
            jf.set_current(names[-1])           # detail = last selected
            self.refresh_param_tables()
            self._update_plot_joint(overlay_names=list(names))
            # Item B2: if any of these datasets is ALSO highlighted in the Data list,
            # overlay their individual fits (dashed) on top of the joint curves.
            self._maybe_combined_overlay()
        except (ValueError, IndexError):
            pass

    # ---- per-dataset fit memory ------------------------------------------- #
    def _clone_fit(self, src):
        """Clone a ``Fit`` for the per-dataset store.

        Deepcopies only the model (``func``) -- never the ``QObject`` -- and rewires
        a fresh ``Fit`` around it WITHOUT re-running ``init_params()`` (which would
        reset the values we want to carry over).  Provenance attrs (set in
        ``update_parameters``) are copied so the function list can auto-follow.
        Returns ``None`` if ``src`` is missing or is a ``JointFit`` (which has no
        single ``.func`` to clone); callers then keep today's behavior."""
        if src is None or hasattr(src, 'objectives'):
            return None
        f = getattr(src, 'func', None)
        if f is None:
            return None
        newfit = Fit.__new__(Fit)
        QObject.__init__(newfit)
        newfit.func = copy.deepcopy(f)
        newfit.func.x = self.x
        newfit.params = newfit.func.__dict__
        newfit.fit_params = newfit.func.params
        newfit.fitter = None
        newfit.result = None
        newfit.Niter = 0
        newfit.set_x(self.x)
        for attr in ('_xmod_category', '_xmod_module', '_xmod_funcname', '_xmod_xexpr'):
            if hasattr(src, attr):
                setattr(newfit, attr, getattr(src, attr))
        return newfit

    def _fit_for_dataset(self, name, seed=None):
        """Return the stored ``Fit`` for ``name``; lazily create one by cloning the
        current view (``seed`` -> ``self.fit`` -> ``self._template_fit``).

        Returns ``None`` if nothing has ever been displayed (fresh app / no function
        picked yet), so callers fall back to the pre-existing single-``Fit`` path and
        the feature stays transparent until a model exists."""
        if name in self.dataset_fits:
            return self.dataset_fits[name]
        src = seed
        if src is None:
            src = getattr(self, 'fit', None)
            if src is None or hasattr(src, 'objectives'):
                src = self._template_fit
        newfit = self._clone_fit(src)
        if newfit is not None:
            self.dataset_fits[name] = newfit
        return newfit

    def _activate_dataset_fit(self, fit):
        """Make ``fit`` the live ``self.fit`` and resync the mpar bookkeeping + the
        param tables to it.

        Shared by dataset-switching and per-dataset batch fitting so the tables, the
        undo snapshot and ``update_plot`` always match the ACTIVE model -- selected
        datasets may now carry different models, so the tables cannot be assumed to
        match ``self.fit`` after a swap.  ``update_fixed_parameters`` /
        ``update_fit_parameters`` each self-manage their own ``cellChanged`` signal."""
        self.fit = fit
        if '__mpar__' in self.fit.params.keys() and self.fit.params['__mpar__'] != {}:
            self.mpar_keys = list(self.fit.params['__mpar__'].keys())
            pkey = list(self.fit.params['__mpar__'][self.mpar_keys[0]].keys())[0]
            self.mpar_N = {}
            for mkey in self.mpar_keys:
                self.mpar_N[mkey] = len(self.fit.params['__mpar__'][mkey][pkey])
        self.update_fixed_parameters()
        self.update_fit_parameters()

    def _sync_funclist_to_fit(self, fit):
        """Point the function list + Xrange at ``fit``'s model (auto-follow) without
        triggering ``functionChanged`` / ``update_functions`` recursively.

        Best-effort: any failure leaves the tables/plot correct (they read
        ``self.fit``); only the highlight would be stale."""
        if fit is None:
            return
        cat = getattr(fit, '_xmod_category', None)
        funcname = getattr(fit, '_xmod_funcname', None)
        if funcname is None:
            func = getattr(fit, 'func', None)
            funcname = type(func).__name__ if func is not None else None
        xexpr = getattr(fit, '_xmod_xexpr', None)
        try:
            self.categoryListWidget.currentItemChanged.disconnect()
        except Exception:
            pass
        try:
            self.funcListWidget.itemSelectionChanged.disconnect()
        except Exception:
            pass
        try:
            if cat is not None and getattr(self, 'curr_category', None) != cat:
                items = self.categoryListWidget.findItems(cat, Qt.MatchExactly)
                if items:
                    self.categoryListWidget.setCurrentItem(items[0])
                    self.update_functions()  # rebuilds funcListWidget, sets curr_category
                    try:                       # update_functions re-attaches functionChanged
                        self.funcListWidget.itemSelectionChanged.disconnect()
                    except Exception:
                        pass
            if funcname is not None:
                fitems = self.funcListWidget.findItems(funcname, Qt.MatchExactly)
                if fitems:
                    self.funcListWidget.clearSelection()
                    self.funcListWidget.setCurrentItem(fitems[0])
                    fitems[0].setSelected(True)
                    self.curr_module = funcname
            if xexpr:
                self.xLineEdit.setText(xexpr)
        except Exception:
            import traceback; traceback.print_exc()
        finally:
            self.categoryListWidget.currentItemChanged.connect(self.update_functions)
            self.funcListWidget.itemSelectionChanged.connect(self.functionChanged)

    def refresh_param_tables(self):
        """Re-render the stock param tables from self.fit (master/detail: shows the
        JointFit's *current* dataset via its .fit_params / .params view)."""
        try:
            self.update_fixed_parameters()
            self.update_fit_parameters()
        except Exception:
            pass

    def _compile_joint_guarded(self, spec):
        """``spec.compile`` for the GUI: never lets a ValueError reach a Qt slot
        (an unhandled exception there aborts the app; final review I3).

        A ShareMismatchError (share-group endpoints disagree on min/max/vary)
        asks Yes/No whether to apply the first dataset's bounds/vary to every
        member of each share group; Yes recompiles with ``on_mismatch='first'``
        (final review I4). Any other ValueError (e.g. a covariance sidecar
        problem) is shown in a warning box. Returns the JointFit, or None when
        the compile was refused or declined."""
        from Joint_Fit_Routines import ShareMismatchError
        masks = self._joint_masks(spec)
        try:
            return spec.compile(self.data, masks=masks)
        except ShareMismatchError as ex:
            ans = QMessageBox.question(
                self, 'Joint fit',
                str(ex) + "\n\nApply the first dataset's bounds/vary to all members "
                "of each share group and continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return None
            try:
                return spec.compile(self.data, masks=masks, on_mismatch='first')
            except ValueError as ex2:
                QMessageBox.warning(self, 'Joint fit', str(ex2))
                return None
        except ValueError as ex:
            QMessageBox.warning(self, 'Joint fit', str(ex))
            return None

    def _apply_joint_ranges(self, jf, spec):
        """Freeze ``jf``'s fit windows and column masks the way ``perform_fit``
        does (scalar Xrange box + the spec's per-dataset fit ranges), so
        ``active_index`` -- and so the Model Comparison -- honours #FitRange= and
        the column limits on a joint fit that was loaded but not re-fitted
        (final review I2). Returns ``jf``."""
        xmin, xmax = self._joint_range()
        jf._set_ranges(xmin, xmax, ranges=spec.fit_ranges())
        return jf

    def run_joint_fit(self):
        """Compile the panel's spec into a JointFit, fit it, and show the current
        dataset's fitted params in the stock tables (master/detail). q-range is a
        scalar parsed from the Xrange box; we deliberately do NOT call
        xminmaxChanged()/update_plot() -- those assume a single Fit (dict xmin/xmax
        and self.fit.params['choices']/self.fit.x). Joint-fit plotting is a
        separate, deferred step."""
        import numpy as _np
        spec = self.jointFitPanel.spec
        self._cov_sync_joint_panel()        # attached covariances, unless overridden
        _jf = self._compile_joint_guarded(spec)
        if _jf is None:                     # refused or declined; nothing changed
            return
        self.fit = _jf                      # fresh JointFit
        self.jointFit = self.fit            # persist so switching to an individual
                                            # dataset view and back retains joint results
        # enforce shared fixed params (incl. the PDB) across all datasets
        if getattr(self.fit, 'objectives', None):
            self._sync_joint_fixed_params(self.fit.objectives[0]['func'])
        self.fit_method = self.fitMethods[self.fitMethodComboBox.currentText()]
        self.fit_scale = self.fitScaleComboBox.currentText()
        # scalar q-bounds from the Xrange box (NOT the per-file self.xmin/self.xmax dicts)
        xmin, xmax = self._joint_range()
        # per-dataset fit windows from the panel (blank rows -> scalar fallback)
        ranges = spec.fit_ranges()
        # live progress screen (iteration / chi-sqr / Stop) -- same dialog as individual
        # fits; fitCallback is JointFit-aware (skips the single-Fit curve redraw).
        self.fitData = []
        self.tchisqr = 1e30
        try:
            self.fit.functionCalled.connect(self.fitCallback)
        except Exception:
            pass
        try:
            self.showFitInfoDlg()
        except Exception:
            pass
        try:
            self.fit_report, self.fit_message = self.fit.perform_fit(
                xmin, xmax, fit_scale=self.fit_scale, fit_method=self.fit_method,
                ranges=ranges, covariance_mode=spec.covariance_mode)
            spec.covariance_notes = []      # a .jpar reload note is reported once
        finally:
            try:
                self.closeFitInfoDlg()
            except Exception:
                pass
            self.disconnect_fit_function()
        self.jointFit._is_fitted = True          # a completed joint fit
        self.jointFit._jpar_incomplete = False
        # results live in each objective's model after the fit; show the current
        # dataset in the stock tables (guarded: a table hiccup must not eat the report)
        try:
            self.fit.set_current(self.jointFitPanel.datasetSelector.currentData()
                                 or self.fit.objective_names[0])
            self.refresh_param_tables()
            self.update_plot()          # joint-aware: draws the current dataset's curve
        except Exception:
            import traceback; traceback.print_exc()
        self._refresh_data_labels()
        # A completed joint fit enables 'Calculate Param Error' (joint MCMC/CIs), even
        # with no individual dataset selected in the Data list.
        self._update_ci_button_state()
        FitResultDialog(self.fit_report, self.fit_message, parent=self).show()

    def _joint_emcee_mode(self):
        """(covariance mode, note) for joint MCMC: the mode the fitted JointFit used, so
        C is frozen at the result it belongs to; a note when the panel's combo now names
        another mode (final review M4/T8). No covariance fit -> the panel's mode, no note."""
        panel = getattr(getattr(self, 'jointFitPanel', None), 'spec', None)
        pmode = getattr(panel, 'covariance_mode', None)
        fitted = (getattr(self.fit, 'cov_info', None) or {}).get('mode')
        if fitted is None:
            return pmode, None
        note = None
        if pmode is not None and pmode != fitted:
            note = ('joint MCMC uses covariance mode %r, the mode of the fit it samples from; '
                    'the panel now says %r (refit to change it)' % (fitted, pmode))
        return fitted, note

    def run_joint_emcee(self, emcee_walkers=100, emcee_steps=100, emcee_cores=1,
                        reuse_sampler=False, emcee_burn=0, emcee_thin=1, backend=None):
        """Run MCMC confidence intervals on the joint objective. Reuses the already
        fitted JointFit at self.fit (from a prior joint least-squares fit) -- samples
        the global master_params via the whitened/concatenated joint residual. The
        no-covariance branch samples in PARALLEL across emcee_cores (a picklable
        JointLogProb + spawn pool, with theta/phi wrapped in the main process); the
        covariance/GLS branch stays serial for now."""
        import numpy as _np
        if not hasattr(self.fit, 'objectives') or getattr(self.fit, 'result', None) is None:
            QMessageBox.warning(self.emceeConfIntervalWidget, 'Joint MCMC',
                                'Run a joint least-squares fit first, then calculate the '
                                'MCMC confidence interval.', QMessageBox.Ok)
            return
        if isinstance(emcee_cores, int) and emcee_cores > 1 and getattr(self.fit, '_use_cov', False):
            # Covariance/GLS joint MCMC stays serial until the whitened residual is
            # validated against a covariance dataset; the no-cov branch runs parallel.
            print('Covariance joint MCMC runs serially for now; ignoring cores=%d.' % emcee_cores)
        self.fit_method = 'emcee'
        self.fit_scale = self.fitScaleComboBox.currentText()
        # scalar q-range from the Xrange box (joint fits don't use the per-file dicts)
        try:
            _xmin, _xmax = self.xminmaxLineEdit.text().split(':')
            xmin, xmax = float(_xmin), float(_xmax)
        except Exception:
            xmin, xmax = -_np.inf, _np.inf
        # per-dataset fit windows must match the joint least-squares fit exactly
        try:
            ranges = {e.name: (e.fit_xmin, e.fit_xmax)
                      for e in self.jointFitPanel.spec.datasets}
        except Exception:
            ranges = None
        _cmode, _cnote = self._joint_emcee_mode()
        if _cnote:
            self.fit.pending_warnings = [_cnote]
        self.MCMC_starting_step = 0
        self.autoCorrTime = _np.array([[0, 1]])
        self.start_time = time.time()
        self.fit_stopped = False
        self.fit.fit_abort = False
        try:
            self.fit.functionCalled.connect(self.fitErrorCallback)
        except Exception:
            pass
        self.showFitInfoDlg(emcee_walkers=emcee_walkers, emcee_steps=emcee_steps, emcee_burn=emcee_burn)
        try:
            self.fit_report, self.fit_message = self.fit.perform_fit(
                xmin, xmax, fit_scale=self.fit_scale, fit_method='emcee',
                maxiter=int(self.fitIterationLineEdit.text()), emcee_walkers=emcee_walkers,
                emcee_steps=emcee_steps, emcee_cores=emcee_cores, reuse_sampler=reuse_sampler,
                emcee_burn=emcee_burn, emcee_thin=emcee_thin, backend=backend, ranges=ranges,
                seed_temperature=getattr(self, 'seed_temperature', 1.0),
                seed_mode=getattr(self, 'seed_mode', 'joint'),
                covariance_mode=_cmode)
        finally:
            self.disconnect_fit_function()
        self.errorAvailable = True
        self.emceeConfIntervalWidget.reuseSamplerCheckBox.setEnabled(True)
        self.emceeConfIntervalWidget.reuseSamplerCheckBox.setCheckState(Qt.Checked)
        self.perform_post_sampling_tasks()
        self.showConfIntervalButton.setEnabled(True)

    # ---- joint-fit save / load (.jpar) + .par-into-joint ------------------- #
    def _category_of_func(self, funcname):
        """Find which Functions/<category>/ folder defines <funcname>.py."""
        base = os.path.join(os.curdir, 'Functions')
        try:
            for cat in os.listdir(base):
                d = os.path.join(base, cat)
                if os.path.isdir(d) and os.path.exists(os.path.join(d, funcname + '.py')):
                    return cat
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_fitrange(text):
        """Parse a per-dataset '#FitRange=<xmin>:<xmax>' value into (xmin, xmax).

        Either side may be blank (-> None -> global-Xrange fallback). Returns
        (None, None) for a missing/blank/malformed value."""
        if not text or ':' not in text:
            return None, None
        lo, _, hi = text.partition(':')
        def _f(s):
            s = s.strip()
            if s == '':
                return None
            try:
                return float(s)
            except ValueError:
                return None
        return _f(lo), _f(hi)

    def _build_joint_model(self, block):
        """Build a fresh model INSTANCE from a parsed .jpar dataset block and apply
        its saved parameter values. Returns None if the model class can't be built."""
        import joint_fit_io as jio
        category = block.get('category') or self.curr_category
        func = block.get('func')
        module = 'Functions.%s.%s' % (category, func)
        try:
            if module not in self.curr_funcClass:
                self.curr_funcClass[module] = import_module(module)
            cls = getattr(self.curr_funcClass[module], func)
        except Exception:
            traceback.print_exc()
            return None
        # a PDB model must construct against a real file: pull fname from the block
        fname = None
        for line in block.get('fixed', []):
            k, _, v = line.partition('\t')
            if k == 'fname':
                fname = v
        try:
            m = cls(fname=fname) if fname else cls()
        except Exception:
            try:
                m = cls()
            except Exception:
                traceback.print_exc()
                return None
        try:
            m.x = eval(block.get('xrange') or 'np.linspace(0.0,1.0,100)')
        except Exception:
            pass
        if getattr(m, 'params', None) is None:
            m.init_params()
        try:
            jio.apply_block(m, block.get('fixed', []), block.get('sfit', []), block.get('mfit', []))
        except Exception:
            traceback.print_exc()
        return m

    def _loaded_key_for_path(self, path):
        """Map a stored file PATH to a currently-loaded dataset key ("<num><>path"),
        matching on the path portion (the <num> prefix is volatile). Full-path match
        first, then a unique-basename fallback; None if not loaded."""
        target = os.path.normcase(os.path.abspath(path))
        for k in self.data.keys():
            p = k.split('<>', 1)[1] if '<>' in k else k
            if os.path.normcase(os.path.abspath(p)) == target:
                return k
        base = os.path.basename(path)
        cand = [k for k in self.data.keys()
                if os.path.basename(k.split('<>', 1)[1] if '<>' in k else k) == base]
        return cand[0] if len(cand) == 1 else None

    def saveJointFit(self, fname=None):
        """Write the current joint-fit spec to a .jpar (datasets by path + per-dataset
        model/params + sharing + covariance + fit settings)."""
        import joint_fit_io as jio
        spec = getattr(self.jointFitPanel, 'spec', None)
        if spec is None or len(spec.datasets) < 1:
            QMessageBox.warning(self, 'Joint fit', 'No datasets in the joint fit to save.')
            return
        if fname is None:
            fname = QFileDialog.getSaveFileName(self, 'Save joint fit', self.curDir,
                                                'Joint fit (*.jpar)')[0]
            if not fname:
                return
        if os.path.splitext(fname)[1] == '':
            fname += '.jpar'
        settings = {'fit_range': self.xminmaxLineEdit.text(),
                    'column_limits': self._format_column_limits(),
                    'fit_method': self.fitMethodComboBox.currentText(),
                    'fit_scale': self.fitScaleComboBox.currentText(),
                    'fit_iterations': self.fitIterationLineEdit.text(),
                    'cov_mode': spec.covariance_mode}
        blocks = []
        for e in spec.datasets:
            if e.model is None:
                continue
            path = e.name.split('<>', 1)[1] if '<>' in e.name else e.name
            func = type(e.model).__name__
            category = self._category_of_func(func) or self.curr_category
            src = self.dataset_fits.get(e.name)
            xexpr = getattr(src, '_xmod_xexpr', None) or self.xLineEdit.text()
            fixed, sfit, mfit = jio.dump_model(e.model)
            fitrange = ''
            if e.fit_xmin is not None or e.fit_xmax is not None:
                fitrange = '%s:%s' % ('' if e.fit_xmin is None else ('%g' % e.fit_xmin),
                                      '' if e.fit_xmax is None else ('%g' % e.fit_xmax))
            blocks.append({'path': path, 'category': category, 'func': func,
                           'xrange': xexpr, 'fitrange': fitrange,
                           'collimits': self._format_column_limits(e.name)
                                        if not getattr(e, 'limits', None)
                                        else ';'.join(
                               '%s:%s:%s' % (c, '' if v[0] is None else repr(float(v[0])),
                                             '' if v[1] is None else repr(float(v[1])))
                               for c, v in sorted(e.limits.items()) if c != 'x'),
                           'cov': e.cov_path, 'raw': e.raw_path,
                           'fixed': fixed, 'sfit': sfit, 'mfit': mfit})
        shares = []
        for g in spec.groups:
            eps = [((nm.split('<>', 1)[1] if '<>' in nm else nm), lk)
                   for (nm, lk) in g.endpoints]
            shares.append({'bare': g.bare, 'name': g.name, 'endpoints': eps})
        try:
            with open(fname, 'w') as fh:
                fh.write(jio.format_jpar(settings, blocks, shares))
        except Exception:
            QMessageBox.warning(self, 'Joint fit', 'Could not save:\n' + traceback.format_exc())
            return
        QMessageBox.information(self, 'Joint fit', 'Saved joint fit to\n%s' % fname)

    def loadJointFit(self, fname=None):
        """Load a .jpar: associate datasets already in the Data list (warn+skip any
        that are not loaded), restore per-dataset models/params + sharing, and compile
        a viewable JointFit."""
        import joint_fit_io as jio
        if fname is None:
            fname = QFileDialog.getOpenFileName(self, 'Load joint fit', self.curDir,
                                                'Joint fit (*.jpar)')[0]
            if not fname:
                return
        try:
            with open(fname) as fh:
                parsed = jio.parse_jpar(fh.read())
        except Exception:
            QMessageBox.warning(self, 'Joint fit', 'Could not read joint fit:\n'
                                + traceback.format_exc())
            return
        spec = self.jointFitPanel.spec
        for nm in list(spec.dataset_names):      # start fresh from the file
            spec.remove_dataset(nm)
        # resolve each referenced path to a currently-loaded dataset key
        resolved = {b['path']: self._loaded_key_for_path(b['path']) for b in parsed['blocks']}
        unloaded = [p for p, k in resolved.items() if k is None]
        not_found, loaded_from_disk = [], []
        if unloaded:
            ans = QMessageBox.question(
                self, 'Joint fit',
                '%d dataset(s) referenced by this joint fit are not loaded:\n%s\n\n'
                'Load them from their stored paths?'
                % (len(unloaded), '\n'.join('  ' + os.path.basename(p) for p in unloaded)),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ans == QMessageBox.Yes:
                to_load = [p for p in unloaded if os.path.exists(p)]
                not_found = [p for p in unloaded if not os.path.exists(p)]
                if to_load:
                    try:
                        self.addData(fnames=to_load)
                    except Exception:
                        traceback.print_exc()
                    for p in to_load:
                        resolved[p] = self._loaded_key_for_path(p)
                        if resolved[p] is not None:
                            loaded_from_disk.append(p)
        path_to_name, skipped = {}, []
        for b in parsed['blocks']:
            dkey = resolved.get(b['path'])
            if dkey is None:
                if b['path'] not in not_found:
                    skipped.append(b['path'])
                continue
            model = self._build_joint_model(b)
            if model is None:
                skipped.append(b['path'] + '  (model build failed)')
                continue
            fit_xmin, fit_xmax = self._parse_fitrange(b.get('fitrange'))
            _cl = self._parse_column_limits(b.get('collimits') or '')
            spec.add_dataset(dkey, model=model,
                             cov_path=b.get('cov') or None, raw_path=b.get('raw') or None,
                             fit_xmin=fit_xmin, fit_xmax=fit_xmax,
                             limits=(_cl or None))
            if _cl:
                self.limits.setdefault(dkey, {}).update(_cl)
            path_to_name[b['path']] = dkey
        for g in parsed['shares']:
            eps = [(path_to_name[p], lk) for (p, lk) in g['endpoints'] if p in path_to_name]
            if len(eps) >= 2:
                spec.set_shared_endpoints(eps, bare=g.get('bare', False), master_name=g.get('name'))
        _mode, _note = jio.jpar_cov_mode(parsed)    # no #CovMode with a #Cov -> data + note
        spec.covariance_mode = _mode
        spec.covariance_notes = [_note] if _note else []
        s = parsed.get('settings', {})
        if s.get('fit_range'):
            self.xminmaxLineEdit.setText(s['fit_range'])
        if s.get('column_limits'):
            self._apply_column_limits(s['column_limits'])
        if s.get('fit_method'):
            self.fitMethodComboBox.setCurrentText(s['fit_method'])
        if s.get('fit_scale'):
            self.fitScaleComboBox.setCurrentText(s['fit_scale'])
        if s.get('fit_iterations'):
            self.fitIterationLineEdit.setText(s['fit_iterations'])
        incomplete = bool(skipped or not_found)
        refused = False
        try:
            if len(spec.datasets) >= 2:
                _jf = self._compile_joint_guarded(spec)
                if _jf is None:
                    raise _JointCompileRefused()
                self.jointFit = self._apply_joint_ranges(_jf, spec)
                self.fit = self.jointFit
                if getattr(self.fit, 'objectives', None):
                    self._sync_joint_fixed_params(self.fit.objectives[0]['func'])
                # a complete .jpar load counts as a fitted joint; an incomplete one is
                # flagged so the Model Comparison tells the user to re-run the joint fit.
                self.jointFit._is_fitted = not incomplete
                self.jointFit._jpar_incomplete = incomplete
        except _JointCompileRefused:
            refused = True                  # already shown by _compile_joint_guarded
        except Exception:
            traceback.print_exc()
        self.jointFitEnabled = True
        try:
            self.jointFitPanel.enableCheck.setChecked(True)
            self.jointFitPanel._resync_all()
        except Exception:
            traceback.print_exc()
        msg = 'Loaded joint fit from\n%s\n\n%d dataset(s) associated.' % (fname, len(path_to_name))
        if loaded_from_disk:
            msg += '\n\nLoaded from disk:\n' + '\n'.join('  ' + os.path.basename(p) for p in loaded_from_disk)
        if not_found:
            msg += ('\n\nCould NOT be found on disk (skipped):\n'
                    + '\n'.join('  ' + p for p in not_found))
        if skipped:
            msg += ('\n\nSkipped (not loaded in the Data list):\n'
                    + '\n'.join('  ' + os.path.basename(s.split('  ')[0]) for s in skipped))
        if not refused:
            QMessageBox.information(self, 'Joint fit', msg)

    def loadParIntoJoint(self, fname=None):
        """Apply one stock .par's values to EVERY dataset in the joint fit (datasets
        whose model function differs from the file are skipped with a warning)."""
        import joint_fit_io as jio
        spec = getattr(self.jointFitPanel, 'spec', None)
        if spec is None or len(spec.datasets) < 1:
            QMessageBox.warning(self, 'Joint fit',
                                'Add datasets to the joint fit first, then load a .par into it.')
            return
        if fname is None:
            fname = QFileDialog.getOpenFileName(self, 'Load .par into joint fit', self.curDir,
                                                'Parameter files (*.par)')[0]
            if not fname:
                return
        try:
            with open(fname) as fh:
                par = jio.parse_par(fh.read())
        except Exception:
            QMessageBox.warning(self, 'Joint fit', 'Could not read .par:\n' + traceback.format_exc())
            return
        applied, mismatch = 0, []
        for e in spec.datasets:
            if e.model is None:
                continue
            if par.get('func') and type(e.model).__name__ != par['func']:
                mismatch.append(e.name.split('<>', 1)[-1])
                continue
            try:
                jio.apply_block(e.model, par.get('fixed', []), par.get('sfit', []), par.get('mfit', []))
                applied += 1
            except Exception:
                traceback.print_exc()
        refused = False
        try:
            if len(spec.datasets) >= 2:
                _jf = self._compile_joint_guarded(spec)
                if _jf is None:
                    raise _JointCompileRefused()
                self.jointFit = self._apply_joint_ranges(_jf, spec)
                self.fit = self.jointFit
        except _JointCompileRefused:
            refused = True                  # already shown by _compile_joint_guarded
        except Exception:
            traceback.print_exc()
        try:
            self.jointFitPanel._resync_all()
        except Exception:
            pass
        msg = 'Applied %s to %d joint dataset(s).' % (os.path.basename(fname), applied)
        if mismatch:
            msg += ('\n\nSkipped (model function differs from the .par):\n'
                    + ', '.join(mismatch))
        if not refused:
            QMessageBox.information(self, 'Joint fit', msg)

    # ---- Model Comparison (joint vs individual fits) ---------------------- #
    def create_compareDock(self):
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit
        w = QWidget()
        lay = QVBoxLayout(w)
        btn = QPushButton('Compute joint-vs-individual comparison')
        btn.clicked.connect(self.computeModelComparison)
        lay.addWidget(btn)
        self.compareTextEdit = QTextEdit()
        self.compareTextEdit.setReadOnly(True)
        self.compareTextEdit.setFontFamily('Courier New')
        self.compareTextEdit.setText(
            'Run (or load) a joint fit, individually fit each of its datasets, '
            'then click Compute.\n\nCompares the joint fit against the collection of '
            'individual fits with an F-test / likelihood-ratio test (when the models are '
            'nested) and AIC/AICc/BIC (always).')
        lay.addWidget(self.compareTextEdit)
        self.compareDock.addWidget(w)

    def _joint_range(self):
        try:
            a, b = self.xminmaxLineEdit.text().split(':')
            return float(a), float(b)
        except Exception:
            return -np.inf, np.inf

    def _eval_chisqr(self, func, name, idx, imin=None, imax=None):
        """Plain chi-square Sum(((y-yfit)/yerr)^2) of `func` against dataset `name`'s
        PRIMARY curve at the points `idx` (indices into the full curve, normally the
        joint fit's active_index), the model evaluated on the joint objective's
        [imin, imax] window. Returns (chisqr, ndata) or (None, 0)."""
        d = self.data.get(name)
        if not d or len(idx) == 0:
            return None, 0
        key = list(d.keys())[0]
        x = np.asarray(d[key]['x']); y = np.asarray(d[key]['y']); yerr = np.asarray(d[key]['yerr'])
        from Joint_Fit_Routines import chisqr_on
        try:
            return chisqr_on(func, x, y, yerr, idx,
                             masked_x=bool(getattr(func, '__masked_x__', False)),
                             imin=imin, imax=imax)
        except Exception:
            return None, 0

    def computeModelComparison(self):
        import numpy as _np
        try:
            from scipy import stats as _stats
        except Exception:
            _stats = None
        jf = getattr(self, 'jointFit', None)
        if jf is None:
            self.compareTextEdit.setText('No joint fit yet — run or load a joint fit first.')
            return
        if not getattr(jf, '_is_fitted', False):
            if getattr(jf, '_jpar_incomplete', False):
                self.compareTextEdit.setText(
                    'The loaded .jpar references datasets that are not in the current joint '
                    'dataset list, so the joint fit is incomplete.\n\nRe-run the joint fit over '
                    'the current datasets, then compare.')
            else:
                self.compareTextEdit.setText('Run the joint fit (or load a complete .jpar) first.')
            return
        names = jf.objective_names
        def _fitted(n):
            f = self.dataset_fits.get(n)
            return f is not None and (getattr(f, '_is_fitted', False)
                                      or getattr(f, 'result', None) is not None)
        missing = [n for n in names if not _fitted(n)]
        if missing:
            self.compareTextEdit.setText(
                'Each joint dataset needs an INDIVIDUAL fit (or a loaded .par) first. Missing:\n'
                + '\n'.join('  ' + self._short_name(m) for m in missing))
            return

        # Compute PLAIN chi-square by evaluation for both models on the same range/points,
        # so the comparison is scale-consistent even for a covariance/GLS joint fit.
        RSS_ind = RSS_j = 0.0
        N = p_ind = 0
        for i, n in enumerate(names):
            indfit = self.dataset_fits[n]
            idx = jf.active_index(i)
            win = (jf.objectives[i]['imin'], jf.objectives[i]['imax'])
            ci, ni = self._eval_chisqr(indfit.func, n, idx, *win)
            cj, nj = self._eval_chisqr(jf.objectives[i]['func'], n, idx, *win)
            if ci is None or cj is None:
                self.compareTextEdit.setText(
                    'Could not evaluate dataset %s over the fit range.' % self._short_name(n))
                return
            RSS_ind += ci; RSS_j += cj; N += ni
            p_ind += sum(1 for pp in indfit.func.params.values() if pp.vary)
        p_j = sum(1 for pp in jf.master_params.values() if getattr(pp, 'vary', False))
        N_j = N   # same range/points by construction

        # nesting: same function AND same PDB across all individuals and the joint fit
        jf0 = jf.objectives[0]['func']
        joint_fn, joint_pdb = type(jf0).__name__, getattr(jf0, 'fname', None)
        def _norm(p):
            return os.path.normcase(os.path.abspath(p)) if p else None
        same_func = all(type(self.dataset_fits[n].func).__name__ == joint_fn for n in names)
        same_pdb = all(_norm(getattr(self.dataset_fits[n].func, 'fname', None)) == _norm(joint_pdb)
                       for n in names)
        nested = same_func and same_pdb

        def _ic(rss, n, p):
            if rss <= 0 or n <= 0:
                return float('nan'), float('nan'), float('nan')
            aic = n * _np.log(rss / n) + 2 * p
            aicc = aic + (2 * p * (p + 1) / (n - p - 1)) if (n - p - 1) > 0 else float('nan')
            bic = n * _np.log(rss / n) + p * _np.log(n)
            return aic, aicc, bic
        aic_j, aicc_j, bic_j = _ic(RSS_j, N_j, p_j)
        aic_i, aicc_i, bic_i = _ic(RSS_ind, N, p_ind)

        L = []
        L.append('MODEL COMPARISON  —  joint fit vs collection of individual fits')
        L.append('=' * 66)
        L.append('%-22s %16s %16s' % ('', 'Joint', 'Individuals'))
        L.append('%-22s %16d %16d' % ('# datasets', len(names), len(names)))
        L.append('%-22s %16d %16d' % ('data points N', N_j, N))
        L.append('%-22s %16d %16d' % ('free params p', p_j, p_ind))
        L.append('%-22s %16.6g %16.6g' % ('chi-square', RSS_j, RSS_ind))
        L.append('%-22s %16.4g %16.4g' % ('reduced chi-square',
                                          RSS_j / max(1, N_j - p_j), RSS_ind / max(1, N - p_ind)))
        L.append('%-22s %16.4g %16.4g' % ('AIC', aic_j, aic_i))
        L.append('%-22s %16.4g %16.4g' % ('AICc', aicc_j, aicc_i))
        L.append('%-22s %16.4g %16.4g' % ('BIC', bic_j, bic_i))
        L.append('')
        dAIC, dBIC = aic_j - aic_i, bic_j - bic_i
        L.append('Delta AIC (joint - indiv) = %+.4g     Delta BIC = %+.4g' % (dAIC, dBIC))
        amin = min(aic_j, aic_i)
        wj, wi = _np.exp(-0.5 * (aic_j - amin)), _np.exp(-0.5 * (aic_i - amin))
        s = wj + wi
        if s > 0:
            L.append('Akaike weights:  joint=%.3f   individuals=%.3f   (evidence ratio %.3g)'
                     % (wj / s, wi / s, max(wj, wi) / max(min(wj, wi), 1e-300)))
        L.append('AIC prefers: %s' % ('joint (shared)' if dAIC < 0 else 'individuals (independent)'))
        L.append('')
        if N != N_j:
            L.append('WARNING: point counts differ (indiv N=%d vs joint N=%d) — the individual'
                     % (N, N_j))
            L.append('         and joint fits likely used different q-ranges; refit over the same')
            L.append('         range for a valid comparison. Numbers below are indicative only.')
            L.append('')
        if nested and _stats is not None:
            df1, df2 = p_ind - p_j, N - p_ind
            if df1 > 0 and df2 > 0 and RSS_ind > 0:
                F = ((RSS_j - RSS_ind) / df1) / (RSS_ind / df2)
                pF = float(_stats.f.sf(F, df1, df2))
                dchi = RSS_j - RSS_ind
                pL = float(_stats.chi2.sf(dchi, df1))
                L.append('NESTED comparison (same model function AND same PDB) — valid:')
                L.append('  Extra-sum-of-squares F-test (ANOVA)')
                L.append('    %-16s %14s %6s %14s' % ('Source', 'SS', 'df', 'MS'))
                L.append('    %-16s %14.6g %6d' % ('Joint (restr.)', RSS_j, N_j - p_j))
                L.append('    %-16s %14.6g %6d %14.6g' % ('Indiv (full)', RSS_ind, df2, RSS_ind / df2))
                L.append('    %-16s %14.6g %6d %14.6g' % ('Difference', RSS_j - RSS_ind, df1,
                                                          (RSS_j - RSS_ind) / df1))
                L.append('    F = %.5g     p = %.5g' % (F, pF))
                L.append('  Likelihood-ratio (Delta chi-square) test')
                L.append('    Delta chi-square = %.5g   Delta dof = %d   p = %.5g' % (dchi, df1, pL))
                L.append('    small p  => the per-dataset freedom significantly improves the fit')
                L.append('                (the shared-parameter joint model is rejected).')
            else:
                L.append('NESTED, but degrees of freedom are degenerate (df1=%d, df2=%d) — '
                         'F-test not computable.' % (df1, df2))
        else:
            why = []
            if not same_func:
                why.append('different model functions')
            if not same_pdb:
                why.append('different PDB structures')
            if _stats is None:
                why.append('scipy.stats unavailable')
            L.append('NON-NESTED comparison (%s).' % ', '.join(why or ['non-nested']))
            L.append('  A PDB is a fixed structural hypothesis (0 continuous DOF), so the F-test /')
            L.append('  likelihood-ratio test do not apply. Use the AIC/AICc/BIC above — those are')
            L.append('  valid for non-nested models.')
        L.append('')
        L.append('Note: chi-squares are computed by evaluating each fitted model against the data')
        L.append('over the same range/points, so they are directly comparable.')
        if any(o.get('cov') is not None for o in jf.objectives):
            L.append('  * The joint fit OPTIMIZED a covariance/GLS (whitened) objective, so its')
            L.append('    parameters minimize that, not plain chi-square; the values above reflect')
            L.append('    the fitted parameters\' plain-chi-square quality.')
        L.append('  * If a PDB was chosen from a library by best fit, IC gaps understate the true')
        L.append('    model-selection cost.')
        self.compareTextEdit.setText('\n'.join(L))

    def _short_name(self, key):
        """Human-ish dataset label from the composite '<num><>path' key."""
        try:
            num, path = key.split('<>', 1)
            return '%s: %s' % (num, os.path.basename(path))
        except Exception:
            return key

if __name__=='__main__':
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # app = QApplication(sys.argv)
    # try:
    #     # app.setAttribute(Qt.AA_EnableHighDpiScaling)
    #     app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    # except:
    #     pass
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    w=XModFit()
    w.setWindowTitle('XModFit')
    resolution = QDesktopWidget().screenGeometry()
    w.setGeometry(0, 0, resolution.width() - 100, resolution.height() - 100)
    w.move(int(resolution.width() / 2) - int(w.frameSize().width() / 2),
              int(resolution.height() / 2) - int(w.frameSize().height() / 2))
    try:
        fname = sys.argv[1]
        w.addData(fnames=[fname])
    except:
        pass
    try:
        pname=sys.argv[2]
        w.loadParameters(fname=pname)
    except:
        pass
    w.showMaximized()
    # w.show()
    sys.exit(app.exec_())
        
        
        

