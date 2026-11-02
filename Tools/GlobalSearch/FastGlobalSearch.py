"""Fast global search for protein-pose reflectivity fits (XLayersPDB_XPR).

A Tools plugin, so it costs zero edits to ``xmodfit.py``: ``create_menus`` globs
``Tools/<category>/*.py`` and ``launch_tApp`` instantiates the class whose name
matches the file, handing it the ``XModFit`` instance as ``parent``.

Four steps, revisitable:

    1  Membrane seeds   enumerate from monolayer-only scan(s), load a seed file, or
                        fall back to the current parameters
    2  Search setup     N1, K, tolerances, and the editable search box
    3  Run              budget estimate, progress, cancel
    4  Modes found      the ranked list, the threshold curves, load into the table

**What this tool is, stated where a reader will see it.** The underlying method is a
good OPTIMISER and a poor ENUMERATOR, and both halves are measured. It finds the
deepest basin reliably. But against a 6x larger converged reference it finds a
median of 7 distinct certified minima where the reference finds 49, which collapses
the apparent pose span by 4x and turns about one case in eight from "two opposite
senses both fit" into "the pose is determined". So the mode list is a LOWER BOUND on
ambiguity, the UI says so permanently, and there is deliberately no single
"ambiguous / not ambiguous" verdict anywhere in the stack -- the answer is shown as
a function of the depth threshold instead, because the threshold it would need was
measured to vary ten-fold between strata of a single corpus.

The scan runs in a ``spawn`` child process. That is not for speed: the XPR backend is
process-wide single-slot, and on the Qt thread ``processEvents()`` could dispatch a
plot refresh whose ``xb.load`` drops the data arrays, silently changing the objective
mid-run. A child cannot be raced by the GUI.
"""
import os
import sys
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QProgressBar, QPushButton, QSlider,
                             QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                             QTabWidget, QTextEdit, QVBoxLayout, QWidget)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, os.path.join(_REPO, 'Functions')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The engine is imported in its own guarded block, and it must NEVER rebind QDialog:
# the Tools loader does `tclass(self)`, so if QDialog collapsed to `object` the
# failure would be the cryptic "object.__init__() takes exactly one argument".
_ENGINE_ERR = None
try:
    from fastscan import core as fs_core
    from fastscan import io as fs_io
    from fastscan import modes as fs_modes
    from fastscan import seeds as fs_seeds
    from fastscan import worker as fs_worker
except Exception as _ex:                       # noqa: BLE001
    fs_core = fs_io = fs_modes = fs_seeds = fs_worker = None
    _ENGINE_ERR = '%s: %s' % (type(_ex).__name__, _ex)

try:
    import pyqtgraph as pg
except Exception:                              # noqa: BLE001
    pg = None

try:
    from PlotWidget import PlotWidget
except Exception:                              # noqa: BLE001
    PlotWidget = None

_CAPABILITY_ATTRS = ('_param_box', 'parameter_columns', 'coverage_fraction',
                     'cov_scale', 'protins_top', 'spacez', 'box_ref_length',
                     '__mpar__')


class FastGlobalSearch(QDialog):
    """The four-step global-search dialog."""

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.host = parent
        self.setWindowTitle('Fast global search')
        self.resize(1180, 820)

        self._proc = None
        self._rx = None
        self._timer = None
        self._modeset = None
        self._seeds = []
        self._seed_labels = []
        self._seed_source = 'current-params'
        self._seeds_dropped = 0
        self._box_override = {}
        self._probe = {}

        self._lay = QVBoxLayout(self)
        self.guidance = QLabel('')
        self.guidance.setWordWrap(True)
        self.guidance.setAlignment(Qt.AlignTop)
        self.guidance.setStyleSheet('padding:14px; font-size:11pt;')
        self._lay.addWidget(self.guidance)

        self.tabs = QTabWidget()
        self._lay.addWidget(self.tabs)
        self._build_seed_tab()
        self._build_setup_tab()
        self._build_run_tab()
        self._build_modes_tab()
        self.tabs.hide()

    # ------------------------------------------------------------ readiness ---
    def _func(self):
        """The live model, re-read every time -- never cached at construction."""
        fit = getattr(self.host, 'fit', None)
        return getattr(fit, 'func', None) if fit is not None else None

    def _not_ready(self):
        if fs_core is None:
            return ('The search engine could not be imported.\n\n%s\n\nThe '
                    '`fastscan` package must sit at the repository root next to '
                    '`xmodfit.py`.' % _ENGINE_ERR)
        f = self._func()
        if f is None:
            return ('Select a category, a function and a dataset in the main '
                    'window first, then reopen this tool.')
        missing = [a for a in _CAPABILITY_ATTRS if not hasattr(f, a)]
        if missing:
            return ('%s does not support the fast global search.\n\nIt needs the '
                    'differentiable XPR backend -- exact gradients and an exact '
                    'Hessian over a unit-cube reparameterisation -- which this '
                    'model does not expose (missing: %s).\n\nUse XLayersPDB_XPR. '
                    'The numba XLayersPDB models cannot be searched this way at a '
                    'usable speed.' % (type(f).__name__, ', '.join(missing)))
        try:
            from Functions import xpr_backend as xb
            if not xb.julia_available():
                return ('The Julia/XPR backend is not available in this '
                        'environment, so there is nothing to drive the search '
                        'with.')
        except Exception as ex:                # noqa: BLE001
            return 'The XPR backend could not be loaded: %s' % ex
        if getattr(self.host, 'sfnames', None) in (None, []):
            return 'Select a dataset in the main window first.'
        fit = self.host.fit
        if isinstance(getattr(fit, 'x', None), dict):
            return ('This dataset has multiple columns. The fast scan needs a '
                    'single q/R/err curve: the multi-column residual applies a '
                    'per-curve normalisation, which would break the identity '
                    "between the scan's objective and the fit's chi-squared.")
        bad = fs_core.check_expr(f)
        if bad:
            return ('These varying parameters are driven by an expression: %s\n\n'
                    'The analytic derivatives have no chain rule for a constrained '
                    'parameter, so the search would be silently wrong. Clear the '
                    'expression or pin the parameter.' % ', '.join(bad))
        return None

    def showEvent(self, ev):
        QDialog.showEvent(self, ev)
        self.refresh()

    def refresh(self):
        why = self._not_ready()
        if why:
            self.guidance.setText(why)
            self.guidance.show()
            self.tabs.hide()
            return
        self.guidance.hide()
        self.tabs.show()
        self._refresh_seed_status()
        self._rebuild_box_table()
        self._update_budget()

    # ------------------------------------------------------- step 1: seeds ---
    def _build_seed_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        intro = QLabel(
            'Stage 0 supplies the membrane starting points. Seeding the membrane '
            'block from a homometric enumeration of a <b>protein-free</b> scan was '
            'measured to beat unseeded starts by <b>7-16x</b> in hit rate. The gain '
            'is dimension reduction, not seed accuracy, so candidates far from the '
            'optimum seed just as well -- what matters is that they are plausible '
            'and that bad ones are dropped rather than clipped.')
        intro.setWordWrap(True)
        v.addWidget(intro)

        g1 = QGroupBox('(a) Enumerate from monolayer-only scan(s)')
        f1 = QVBoxLayout(g1)
        r1 = QHBoxLayout()
        self.monolayerEdit = QLineEdit()
        self.monolayerEdit.setPlaceholderText(
            'one or more protein-free scans, semicolon separated')
        b1 = QPushButton('Browse...')
        b1.clicked.connect(self._pick_monolayers)
        self.enumerateBtn = QPushButton('Enumerate')
        self.enumerateBtn.clicked.connect(self._run_enumeration)
        r1.addWidget(self.monolayerEdit, 1)
        r1.addWidget(b1)
        r1.addWidget(self.enumerateBtn)
        f1.addLayout(r1)
        note = QLabel(
            'Deterministic and pure numpy: Patterson -> turnpike -> sign partners '
            '-> basin scan -> spectral family. About 75 s per film, then cached. '
            'Running it on protein-loaded data is refused -- the span stretches and '
            'chi-squared/N runs into the hundreds.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#666;')
        f1.addWidget(note)
        v.addWidget(g1)

        g2 = QGroupBox('(b) Load a seed file from a previous enumeration')
        f2 = QHBoxLayout(g2)
        self.seedFileEdit = QLineEdit()
        b2 = QPushButton('Browse...')
        b2.clicked.connect(self._pick_seed_file)
        f2.addWidget(self.seedFileEdit, 1)
        f2.addWidget(b2)
        v.addWidget(g2)

        g3 = QGroupBox('(c) Current parameters')
        f3 = QVBoxLayout(g3)
        self.useCurrentBtn = QPushButton('Use the current membrane as a single seed')
        self.useCurrentBtn.clicked.connect(self._use_current_seed)
        f3.addWidget(self.useCurrentBtn)
        warn = QLabel('This is the degraded path and costs 7-16x in hit rate. '
                      'It is always reported in the results banner.')
        warn.setWordWrap(True)
        warn.setStyleSheet('color:#a00;')
        f3.addWidget(warn)
        v.addWidget(g3)

        self.seedStatus = QLabel('No seeds loaded.')
        self.seedStatus.setWordWrap(True)
        v.addWidget(self.seedStatus)

        self.seedTable = QTableWidget(0, 0)
        self.seedTable.setAlternatingRowColors(True)
        v.addWidget(self.seedTable, 1)

        r = QHBoxLayout()
        self.saveSeedsBtn = QPushButton('Save seeds...')
        self.saveSeedsBtn.clicked.connect(self._save_seeds)
        r.addWidget(self.saveSeedsBtn)
        r.addStretch(1)
        v.addLayout(r)
        self.tabs.addTab(w, '1 - Membrane seeds')

    def _pick_monolayers(self):
        fns, _ = QFileDialog.getOpenFileNames(
            self, 'Protein-free (monolayer-only) scan(s)',
            getattr(self.host, 'curDir', os.getcwd()),
            'Reflectivity (*.rrf *.txt *.dat);;All files (*)')
        if fns:
            self.monolayerEdit.setText(';'.join(fns))

    def _pick_seed_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, 'Membrane seed table',
            getattr(self.host, 'curDir', os.getcwd()),
            'Seed tables (*.seeds *.json *.txt);;All files (*)')
        if not fn:
            return
        self.seedFileEdit.setText(fn)
        f = self._func()
        nbox = self._nbox(f)
        try:
            raw, labels, meta = fs_seeds.read(fn, num_boxes=nbox)
        except Exception as ex:                # noqa: BLE001
            QMessageBox.warning(self, 'Cannot read seeds', str(ex), QMessageBox.Ok)
            return
        self._install_seeds(raw, labels, 'file:%s' % os.path.basename(fn))

    def _use_current_seed(self):
        f = self._func()
        nbox = self._nbox(f)
        f.update_parameters()
        mkey = f.__mkeys__[0]
        d = np.asarray(f.__d__[mkey], dtype=float)
        rho = np.asarray(f.__rho__[mkey], dtype=float)
        seed = np.concatenate([[float(f.sig)], d[1:-1], rho[1:-1]])
        self._install_seeds([seed], ['current'], 'current-params')

    def _run_enumeration(self):
        paths = [p for p in self.monolayerEdit.text().split(';') if p.strip()]
        if not paths:
            QMessageBox.information(self, 'No scan selected',
                                    'Choose at least one protein-free scan.',
                                    QMessageBox.Ok)
            return
        try:
            from fastscan import homometry
        except Exception as ex:                # noqa: BLE001
            QMessageBox.information(
                self, 'Enumerator not available',
                'The homometric enumerator is not installed yet (%s).\n\nUse route '
                '(b) to load a seed table produced elsewhere, or route (c) to run '
                'with the current membrane as a single seed.' % ex, QMessageBox.Ok)
            return
        f = self._func()
        nbox = self._nbox(f)
        mkey = f.__mkeys__[0]
        rho = np.asarray(f.__rho__[mkey], dtype=float)
        groups, dropped = [], 0
        self.setEnabled(False)
        try:
            for p in paths:
                try:
                    cands = homometry.candidates(p, nbox=nbox, rho_top=float(rho[0]),
                                                 rho_bottom=float(rho[-1]))
                except Exception as ex:        # noqa: BLE001
                    QMessageBox.warning(self, 'Enumeration failed',
                                        '%s:\n%s' % (os.path.basename(p), ex),
                                        QMessageBox.Ok)
                    continue
                groups.append((os.path.basename(p), cands))
        finally:
            self.setEnabled(True)
        if not groups:
            return
        pooled, labels = fs_seeds.pool(groups)
        self._install_seeds(pooled, labels,
                            'homometry:%s' % ','.join(os.path.basename(p)
                                                      for p in paths))

    def _install_seeds(self, raw, labels, source):
        f = self._func()
        nbox = self._nbox(f)
        lo, hi, free, names = fs_core.scan_box(f, nbox, self._box_override)
        cols = fs_core.membrane_columns(f, nbox)
        kept, rejected = fs_seeds.screen(raw, lo[cols], hi[cols],
                                         rho_top=float(f.__rho__[f.__mkeys__[0]][0]),
                                         num_boxes=nbox)
        keep_ids = {id(k) for k in kept}
        self._seeds = kept
        self._seed_labels = [lb for s, lb in zip(raw, labels) if id(s) in keep_ids] \
            or [''] * len(kept)
        self._seed_source = source
        self._seeds_dropped = len(rejected)
        self._fill_seed_table(raw, labels, rejected, nbox)
        self._refresh_seed_status()
        self._update_budget()

    def _fill_seed_table(self, raw, labels, rejected, nbox):
        why = {i: w for i, _s, w in rejected}
        hdr = (['sig'] + ['d%d' % (i + 1) for i in range(nbox)]
               + ['rho%d' % (i + 1) for i in range(nbox)] + ['source', 'status'])
        self.seedTable.clear()
        self.seedTable.setColumnCount(len(hdr))
        self.seedTable.setHorizontalHeaderLabels(hdr)
        self.seedTable.setRowCount(len(raw))
        strike = QFont()
        strike.setStrikeOut(True)
        for r, s in enumerate(raw):
            s = np.asarray(s, dtype=float).ravel()
            for c in range(len(hdr) - 2):
                it = QTableWidgetItem('%.4g' % s[c] if c < len(s) else '')
                if r in why:
                    it.setFont(strike)
                self.seedTable.setItem(r, c, it)
            self.seedTable.setItem(r, len(hdr) - 2,
                                   QTableWidgetItem(str(labels[r] if r < len(labels)
                                                        else '')))
            st = QTableWidgetItem('DROPPED: %s' % why[r] if r in why else 'kept')
            if r in why:
                st.setForeground(QColor('#a00'))
                st.setFont(strike)
            self.seedTable.setItem(r, len(hdr) - 1, st)
        self.seedTable.resizeColumnsToContents()

    def _refresh_seed_status(self):
        if not self._seeds:
            self.seedStatus.setText(
                'No seeds loaded. The scan will fall back to the current membrane '
                'as a single seed (the degraded path).')
            return
        self.seedStatus.setText(
            '<b>%d seed(s) kept, %d dropped</b> &mdash; source: %s'
            % (len(self._seeds), self._seeds_dropped, self._seed_source))

    def _save_seeds(self):
        if not self._seeds:
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, 'Save membrane seeds',
            os.path.join(getattr(self.host, 'curDir', os.getcwd()), 'membrane.seeds'),
            'Seed tables (*.seeds)')
        if fn:
            fs_seeds.write(fn, self._seeds, self._seed_labels,
                           meta={'source': self._seed_source})

    # ------------------------------------------------------- step 2: setup ---
    def _build_setup_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox('Search')
        form = QFormLayout(g)
        self.n1Spin = QSpinBox()
        self.n1Spin.setRange(4, 4096)
        self.n1Spin.setValue(64)
        self.n1Spin.valueChanged.connect(self._update_budget)
        self.n1Spin.setToolTip(
            'Cheap screening starts. The published seeds-needed curve reaches 100% '
            'at 48 and 90% at 32, with the worst observed first hit at start 37 -- '
            'hence 64. It has never been varied on any corpus, so treat it as a '
            'starting point, not a tuned value. More starts is the ONE lever that '
            'was measured to work.')
        form.addRow('N1 screening starts', self.n1Spin)

        self.kSpin = QSpinBox()
        self.kSpin.setRange(1, 64)
        self.kSpin.setValue(6)
        self.kSpin.valueChanged.connect(self._update_budget)
        self.kSpin.setToolTip(
            'How many screened starts to converge. K IS THE SAFETY PARAMETER, not '
            'the stage-1 tolerance: the cheap ranking correlates with the converged '
            'ordering at only rho ~ 0.20, so the scheme works because the true '
            'basin lands somewhere in the top K. If you are missing basins, RAISE K '
            'before tightening stage 1.')
        form.addRow('K converged', self.kSpin)

        self.seedSpin = QSpinBox()
        self.seedSpin.setRange(0, 10 ** 6)
        self.seedSpin.setValue(2024)
        form.addRow('Sobol scramble seed', self.seedSpin)

        self.bandCheck = QCheckBox('Band-stratified start allocation')
        self.bandCheck.setChecked(True)
        self.bandCheck.setToolTip(
            'Force half the starts into each hemisphere of cos(theta). The set of '
            'comparably-good poses is typically dominated by the OPPOSITE band, and '
            'the bands are antipodal, so guaranteeing coverage of both targets the '
            'measured failure directly.')
        form.addRow('', self.bandCheck)

        self.iter1Spin = QSpinBox()
        self.iter1Spin.setRange(5, 2000)
        self.iter1Spin.setValue(30)
        self.iter1Spin.valueChanged.connect(self._update_budget)
        form.addRow('Stage 1 max iterations', self.iter1Spin)
        self.iter2Spin = QSpinBox()
        self.iter2Spin.setRange(10, 20000)
        self.iter2Spin.setValue(300)
        self.iter2Spin.valueChanged.connect(self._update_budget)
        form.addRow('Stage 2 max iterations', self.iter2Spin)
        self.cpu1Spin = QDoubleSpinBox()
        self.cpu1Spin.setRange(1.0, 3600.0)
        self.cpu1Spin.setValue(15.0)
        form.addRow('Stage 1 CPU limit (s)', self.cpu1Spin)
        self.cpu2Spin = QDoubleSpinBox()
        self.cpu2Spin.setRange(1.0, 7200.0)
        self.cpu2Spin.setValue(60.0)
        self.cpu2Spin.valueChanged.connect(self._update_budget)
        self.cpu2Spin.setToolTip(
            'A solve stopped by this limit sits at an arbitrary point on its descent '
            'path. The engine never accepts a point worse than its start, so the '
            'ranking stays safe, but such endpoints are flagged and counted.')
        form.addRow('Stage 2 CPU limit (s)', self.cpu2Spin)
        self.certifyCheck = QCheckBox('Certify endpoints with the exact Hessian')
        self.certifyCheck.setChecked(True)
        self.certifyCheck.setToolTip(
            'Without this, "converged" is not a claim: quasi-Newton solvers report '
            'success at saddle points, and roughly a fifth to a quarter of converged '
            'endpoints on this problem are saddles. Costs one Hessian per endpoint, '
            'plus a one-off compile of about 35-50 s per run.')
        form.addRow('', self.certifyCheck)
        v.addWidget(g)

        gb = QGroupBox('Search box (physical units)')
        vb = QVBoxLayout(gb)
        nb = QLabel(
            'The search box is <b>not</b> the fit bound set. theta and phi are '
            'deliberately unbounded in the fit -- finite bounds act as reflecting '
            'walls at the poles and the phi wrap -- so here they take the '
            'fundamental domain, which is lossless for optimisation. These values '
            'are a scan artefact and are never written back into the parameter '
            'table. A parameter you have pinned stays pinned.')
        nb.setWordWrap(True)
        nb.setStyleSheet('color:#666;')
        vb.addWidget(nb)
        self.boxTable = QTableWidget(0, 4)
        self.boxTable.setHorizontalHeaderLabels(['parameter', 'lo', 'hi', 'varying'])
        self.boxTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.boxTable.itemChanged.connect(self._box_edited)
        vb.addWidget(self.boxTable, 1)
        rb = QHBoxLayout()
        rst = QPushButton('Reset from the fit bounds')
        rst.clicked.connect(self._reset_box)
        rb.addWidget(rst)
        rb.addStretch(1)
        vb.addLayout(rb)
        v.addWidget(gb, 1)
        self.tabs.addTab(w, '2 - Search setup')

    def _nbox(self, f):
        return len(f.__mpar__[f.__mkeys__[0]]['d']) - 2

    def _rebuild_box_table(self):
        f = self._func()
        if f is None:
            return
        try:
            lo, hi, free, names = fs_core.scan_box(f, self._nbox(f),
                                                   self._box_override)
        except Exception as ex:                # noqa: BLE001
            self.guidance.setText('The search box could not be built: %s' % ex)
            self.guidance.show()
            return
        self.boxTable.blockSignals(True)
        rows = [c for c in range(len(lo)) if names[c]]
        self.boxTable.setRowCount(len(rows))
        for r, c in enumerate(rows):
            nm = QTableWidgetItem(names[c])
            nm.setFlags(nm.flags() & ~Qt.ItemIsEditable)
            self.boxTable.setItem(r, 0, nm)
            self.boxTable.setItem(r, 1, QTableWidgetItem('%.6g' % lo[c]))
            self.boxTable.setItem(r, 2, QTableWidgetItem('%.6g' % hi[c]))
            st = QTableWidgetItem('yes' if c in free else 'pinned')
            st.setFlags(st.flags() & ~Qt.ItemIsEditable)
            if c not in free:
                st.setForeground(QColor('#888'))
            self.boxTable.setItem(r, 3, st)
        self.boxTable.blockSignals(False)

    def _box_edited(self, item):
        if item.column() not in (1, 2):
            return
        name = self.boxTable.item(item.row(), 0).text()
        try:
            lo = float(self.boxTable.item(item.row(), 1).text())
            hi = float(self.boxTable.item(item.row(), 2).text())
        except (TypeError, ValueError):
            return
        self._box_override[name] = (lo, hi)
        self._update_budget()

    def _reset_box(self):
        self._box_override = {}
        self._rebuild_box_table()

    # --------------------------------------------------------- step 3: run ---
    def _build_run_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.budgetLabel = QLabel('')
        self.budgetLabel.setWordWrap(True)
        self.budgetLabel.setStyleSheet(
            'padding:10px; background:#f4f4f4; border:1px solid #ddd;')
        v.addWidget(self.budgetLabel)

        r = QHBoxLayout()
        self.probeBtn = QPushButton('Measure cost on this model')
        self.probeBtn.clicked.connect(self._probe_cost)
        self.probeBtn.setToolTip(
            'Times one model evaluation and one gradient on the ACTUALLY loaded '
            'template. Per-solve cost tracks atom count and profile span, so the '
            'published ~0.6 s/start does not transfer between systems -- measured '
            'here it has been several times that.')
        self.runBtn = QPushButton('Run scan')
        self.runBtn.clicked.connect(self._start)
        self.stopBtn = QPushButton('Stop')
        self.stopBtn.clicked.connect(self._stop)
        self.stopBtn.setEnabled(False)
        r.addWidget(self.probeBtn)
        r.addStretch(1)
        r.addWidget(self.runBtn)
        r.addWidget(self.stopBtn)
        v.addLayout(r)

        self.progress = QProgressBar()
        self.progress.setFormat('%v / %m')
        v.addWidget(self.progress)
        self.stageLabel = QLabel('idle')
        v.addWidget(self.stageLabel)
        self.runLog = QTextEdit()
        self.runLog.setReadOnly(True)
        self.runLog.setFont(QFont('Consolas', 9))
        v.addWidget(self.runLog, 1)
        self.tabs.addTab(w, '3 - Run')

    def _active(self):
        """(q, y, err, fname, fit_scale, window) from live GUI state."""
        host = self.host
        fname = host.sfnames[-1]
        xmin = host.xmin[fname] if hasattr(host, 'xmin') else None
        xmax = host.xmax[fname] if hasattr(host, 'xmax') else None
        q, y, e = fs_core.active_arrays(host.fit, xmin, xmax)
        scale = (host.fitScaleComboBox.currentText()
                 if hasattr(host, 'fitScaleComboBox') else 'Linear')
        return q, y, e, fname, scale, (xmin, xmax)

    def _probe_cost(self):
        f = self._func()
        try:
            from Functions import xpr_backend as xb
            q, y, e, _fn, _sc, _w = self._active()
            self.setEnabled(False)
            lo, hi, free, names = fs_core.load_for_scan(
                f, q, y, e, box=fs_core.scan_box(f, self._nbox(f),
                                                 self._box_override))
            mkey = f.__mkeys__[0]
            u = np.clip(_current_unit_point(xb, f, mkey, self._nbox(f)), 0, 1)
            import time
            xb.sim(u)
            t = time.time()
            for _ in range(10):
                xb.sim(u)
            t_sim = (time.time() - t) / 10
            xb.grad(u)
            t = time.time()
            for _ in range(3):
                xb.grad(u)
            t_grad = (time.time() - t) / 3
            self._probe = {'sim': t_sim, 'grad': t_grad, 'nfree': len(free)}
        except Exception as ex:                # noqa: BLE001
            QMessageBox.warning(self, 'Cost probe failed',
                                traceback.format_exc()[-1200:], QMessageBox.Ok)
        finally:
            self.setEnabled(True)
            # leave no unit-cube load behind for the GUI's own plotting
            try:
                f.y()
            except Exception:                  # noqa: BLE001
                pass
        self._update_budget()

    def _update_budget(self):
        if self._func() is None or fs_core is None:
            return
        n1, k = self.n1Spin.value(), self.kSpin.value()
        it1, it2 = self.iter1Spin.value(), self.iter2Spin.value()
        txt = ['<b>Budget</b>: %d screening starts, then %d converged.' % (n1, k)]
        if self._probe:
            s, g = self._probe['sim'], self._probe['grad']
            # one iteration ~ one gradient plus a few line-search evaluations
            per1 = it1 * (g + 3 * s)
            per2 = it2 * (g + 3 * s)
            per2 = min(per2, self.cpu2Spin.value())
            est = 20 + 45 + n1 * per1 + k * per2 + k * 0.3
            txt.append('Measured on the loaded template: sim %.1f ms, gradient '
                       '%.0f ms, %d free coordinates.'
                       % (s * 1e3, g * 1e3, self._probe['nfree']))
            txt.append('Estimated wall clock: <b>%.0f s (%.1f min)</b>, including '
                       'about 20 s of process startup and 45 s of one-off '
                       'derivative compilation.' % (est, est / 60.0))
            if est > 600:
                txt.append('<span style="color:#a00">That is over ten minutes. '
                           'Reduce N1, or lower the stage-2 iteration limit.</span>')
        else:
            txt.append('Press <i>Measure cost on this model</i> for a real estimate. '
                       'The published ~90 s budget was measured on a different '
                       'protein and does not transfer: cost tracks atom count and '
                       'profile span.')
        if not self._seeds:
            txt.append('<span style="color:#a00">No membrane seeds: running the '
                       'degraded path (7-16x worse hit rate).</span>')
        self.budgetLabel.setText('<br>'.join(txt))

    def _start(self):
        why = self._not_ready()
        if why:
            QMessageBox.information(self, 'Not ready', why, QMessageBox.Ok)
            return
        import multiprocessing as mp
        f = self._func()
        nbox = self._nbox(f)
        try:
            q, y, e, fname, scale, window = self._active()
        except Exception as ex:                # noqa: BLE001
            QMessageBox.warning(self, 'Cannot read the active data', str(ex),
                                QMessageBox.Ok)
            return
        if scale != 'Linear':
            QMessageBox.information(
                self, 'Fit scale is %r' % scale,
                'The scan minimises linear chi-squared, which is exactly the fit\'s '
                'own objective ONLY when the fit scale is "Linear".\n\nUnder %r your '
                'fit minimises a log residual that weights the high-q tail far more '
                'heavily, so the mode RANKING can differ, and the published '
                'calibration was measured against linear chi-squared.\n\nThe scan '
                'will run and report both numbers.' % scale, QMessageBox.Ok)

        seeds = self._seeds
        if not seeds:
            f.update_parameters()
            mkey = f.__mkeys__[0]
            d = np.asarray(f.__d__[mkey], dtype=float)
            rho = np.asarray(f.__rho__[mkey], dtype=float)
            seeds = [np.concatenate([[float(f.sig)], d[1:-1], rho[1:-1]])]

        job = fs_worker.build_job(
            self.host.fit, f, q, y, e,
            n1=self.n1Spin.value(), k=self.kSpin.value(),
            seed=self.seedSpin.value(),
            membrane_seeds=[list(map(float, s)) for s in seeds],
            iter1=self.iter1Spin.value(), tol1=1e-3,
            max_cpu1=self.cpu1Spin.value(),
            iter2=self.iter2Spin.value(), tol2=1e-8,
            max_cpu2=self.cpu2Spin.value(),
            band_stratified=self.bandCheck.isChecked(),
            certify=self.certifyCheck.isChecked(),
            seed_source=self._seed_source, seeds_dropped=self._seeds_dropped,
            fit_scale=scale, dataset=str(fname),
            x_window=(float(window[0]) if window[0] is not None else float('nan'),
                      float(window[1]) if window[1] is not None else float('nan')),
            electron_scale=float(f.electron_scale(
                getattr(f.vol_calc, '__name__', None) or str(f.vol_calc))),
            template_label=os.path.splitext(os.path.basename(str(f.fname)))[0],
        )

        ctx = mp.get_context('spawn')
        self._rx, tx = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(target=fs_worker.run, args=(tx, f, job),
                                 daemon=True)
        self.runLog.clear()
        self._log('starting a spawn child: Julia startup and one-off derivative '
                  'compilation take about a minute before the first solve')
        self._proc.start()
        total = self.n1Spin.value() + 2 * self.kSpin.value() + 1
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.runBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self.tabs.setCurrentIndex(2)

    def _log(self, msg):
        self.runLog.append(msg)

    def _poll(self):
        rx, proc = self._rx, self._proc
        if rx is None:
            return
        # Bind rx locally and re-check identity each pass: _handle() finishes the run
        # on a 'result' or 'error' message, which sets self._rx to None, so looping
        # on self._rx would raise AttributeError inside a QTimer slot.
        try:
            while self._rx is rx and rx.poll(0):
                self._handle(rx.recv())
        except (EOFError, OSError):
            self._finish_run('the child process closed the connection')
            return
        if self._rx is not rx:
            return                             # the run already finished cleanly
        if proc is not None and not proc.is_alive() and not rx.poll(0):
            self._finish_run('the child exited without a result (exit code %s)'
                             % proc.exitcode)

    def _handle(self, msg):
        t = msg.get('t')
        if t == 'progress':
            done = {'prepare': 0, 'stage1': 0, 'stage2': self.n1Spin.value(),
                    'certify': self.n1Spin.value() + self.kSpin.value()}
            base = done.get(msg['stage'], 0)
            self.progress.setValue(base + msg['i'])
            best = msg.get('best')
            self.stageLabel.setText(
                '%s  %d/%d%s' % (msg['stage'], msg['i'], msg['n'],
                                 ('   best -logL %.4f' % best) if best is not None
                                 else ''))
        elif t == 'log':
            self._log(msg['msg'])
        elif t == 'error':
            self._log('CHILD ERROR:\n' + msg['msg'])
            self._finish_run('the scan failed in the child process; see the log')
        elif t == 'result':
            res, mode_list, prov = msg['res'], msg['modes'], msg['prov']
            for wmsg in res.warnings:
                self._log('warning: ' + wmsg)
            self._modeset = fs_modes.ModeSet(mode_list, prov,
                                             cancelled=res.cancelled,
                                             warnings=res.warnings)
            self._log('done in %.0f s: %d endpoints -> %d distinct mode(s)'
                      % (res.timings.get('total', 0.0), len(res.stage2_fs),
                         len(self._modeset)))
            self._finish_run(None)
            self._show_modes()
            self.tabs.setCurrentIndex(3)

    def _stop(self):
        if self._proc is not None and self._proc.is_alive():
            self._log('terminating the child')
            self._proc.terminate()
            self._proc.join(5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(5)
        self._finish_run('cancelled')

    def _finish_run(self, why):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._rx is not None:
            try:
                self._rx.close()
            except Exception:                  # noqa: BLE001
                pass
            self._rx = None
        self._proc = None
        self.runBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        if why:
            self.stageLabel.setText(why)
            self._log(why)
        else:
            self.stageLabel.setText('finished')

    # ------------------------------------------------------- step 4: modes ---
    def _build_modes_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.banner = QLabel('')
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(
            'padding:10px; background:#fff6e0; border:1px solid #e0c070;')
        v.addWidget(self.banner)

        sp = QSplitter(Qt.Horizontal)
        self.modeTable = QTableWidget(0, 0)
        self.modeTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.modeTable.setSelectionMode(QAbstractItemView.SingleSelection)
        self.modeTable.setSortingEnabled(False)
        self.modeTable.itemSelectionChanged.connect(self._mode_selected)
        sp.addWidget(self.modeTable)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel('<b>Answer as a function of the depth threshold</b>'))
        if pg is not None:
            self.threshPlot = pg.PlotWidget()
            self.threshPlot.setLabel('bottom', 'delta(-logL) from the best')
            self.threshPlot.setLabel('left', 'distinct positions found')
            self.threshPlot.addLegend()
            rv.addWidget(self.threshPlot, 1)
        else:
            self.threshPlot = None
        rr = QHBoxLayout()
        rr.addWidget(QLabel('threshold'))
        self.threshSlider = QSlider(Qt.Horizontal)
        self.threshSlider.setRange(0, 500)
        self.threshSlider.setValue(140)
        self.threshSlider.valueChanged.connect(self._thresh_changed)
        rr.addWidget(self.threshSlider, 1)
        self.threshLabel = QLabel('')
        rr.addWidget(self.threshLabel)
        rv.addLayout(rr)
        self.threshSummary = QLabel('')
        self.threshSummary.setWordWrap(True)
        rv.addWidget(self.threshSummary)
        sp.addWidget(right)
        sp.setSizes([680, 480])
        v.addWidget(sp, 1)

        if PlotWidget is not None:
            self.curvePlot = PlotWidget()
            v.addWidget(self.curvePlot, 1)
        else:
            self.curvePlot = None

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setFont(QFont('Consolas', 9))
        self.detail.setMaximumHeight(130)
        v.addWidget(self.detail)

        r = QHBoxLayout()
        self.loadBtn = QPushButton('Load this mode into the parameter table')
        self.loadBtn.clicked.connect(self._load_selected)
        self.refineBtn = QPushButton('Load && refine (runs the normal fit)')
        self.refineBtn.clicked.connect(self._load_and_refine)
        self.saveBtn = QPushButton('Save mode set...')
        self.saveBtn.clicked.connect(self._save_modes)
        self.openBtn = QPushButton('Load mode set...')
        self.openBtn.clicked.connect(self._open_modes)
        for b in (self.loadBtn, self.refineBtn, self.saveBtn, self.openBtn):
            r.addWidget(b)
        r.addStretch(1)
        v.addLayout(r)
        self.tabs.addTab(w, '4 - Modes found')

    _COLS = ['#', 'delta(-logL)', 'delta chi2', 'chi2/N', 'theta', 'phi', 'band',
             'protins', 'cov (frac)', 'class', 'starts', 'weight', 'flags']

    def _show_modes(self):
        ms = self._modeset
        if ms is None:
            return
        f = self._func()
        p = ms.prov[0]
        w = p.power
        self.banner.setText(
            '<b>BASIN-FINDER, NOT AN ENUMERATOR.</b><br>'
            '%d screening starts &rarr; %d converged &rarr; <b>%d distinct '
            'position(s) found</b>. Seeds: %s (%d kept, %d dropped). Template: %s. '
            'Certification: %s.<br>'
            'Measured over %d reference cells: a %d-start screen finds a median of '
            '<b>%d</b> distinct certified minima where a %d-start reference finds '
            '<b>%d</b> (%.1fx), and calls <b>%.1f%%</b> of cases pose-<i>determined</i> '
            'that the reference calls pose-<i>ambiguous</i>. Apparent pose span '
            '%.1f&deg; against %.1f&deg;.<br>'
            '<b>So: the deepest basin here is reliable. The LENGTH of this list, and '
            'any reading of "only one pose fits", are a LOWER BOUND.</b> '
            'N1 and K have never been varied on any corpus.'
            % (p.n1, len(ms.modes), len(ms), p.seed_source, p.n_seeds,
               p.seeds_dropped, p.template_label or '?', p.hessian,
               293, 64, w.found_median, w.ref_budget, w.ref_median, w.shortfall,
               100.0 * w.false_determined_rate, w.span_found_deg, w.span_ref_deg))

        self.modeTable.clear()
        self.modeTable.setColumnCount(len(self._COLS))
        self.modeTable.setHorizontalHeaderLabels(self._COLS)
        self.modeTable.setRowCount(len(ms))
        for r, m in enumerate(ms):
            frac = m.cov
            try:
                frac = float(f.coverage_fraction(m.cov, m.theta, m.phi))
            except Exception:                  # noqa: BLE001
                pass
            vals = ['%d' % (r + 1), '%.3f' % m.delta, '%.3f' % (2 * m.delta),
                    '%.4g' % m.chi2_per_n, '%.2f' % m.theta, '%.2f' % m.phi,
                    m.band, '%.3f' % m.protins, '%.4f' % frac, m.hess_class,
                    '%d' % m.n_starts,
                    '' if m.laplace_wt is None else '%.3f' % m.laplace_wt,
                    ','.join(m.flags)]
            for c, s in enumerate(vals):
                it = QTableWidgetItem(s)
                if c == 8:
                    it.setToolTip('raw cov (n_s * A) = %.6g' % m.cov)
                if c == 7:
                    try:
                        it.setToolTip('top-referenced protins_top = %.4f'
                                      % float(f.protins_top(m.protins, m.theta,
                                                            m.phi)))
                    except Exception:          # noqa: BLE001
                        pass
                if c == 3:
                    it.setToolTip('over %d active points, spacez=%.3g, '
                                  'box_ref_length=%.3g. Absolute chi2 is NOT '
                                  'comparable across a different mesh.'
                                  % (p.n_active, p.spacez, p.box_ref_length))
                self.modeTable.setItem(r, c, it)
        self.modeTable.resizeColumnsToContents()
        self._thresh_changed()
        self._overlay_curves()
        if len(ms):
            self.modeTable.selectRow(0)

    def _thresh_changed(self):
        ms = self._modeset
        d = self.threshSlider.value() / 10.0
        self.threshLabel.setText('%.1f' % d)
        if ms is None or not len(ms):
            return
        w = ms.prov[0].power
        n = ms.n_distinct(d)
        span = ms.pose_span(d)
        bands = ms.bands(d)
        self.threshSummary.setText(
            '<b>%d distinct position(s) found within delta %.1f</b>; pose span '
            '%.1f&deg;; bands up/down %d/%d. A converged reference at %d starts '
            'would be expected to find about <b>%.0f</b>. The plausible ambiguity '
            'scale on comparable material is %.1f&ndash;%.1f (observed extremes '
            '%.2f&ndash;%.2f), which is why this is shown as a curve and not as a '
            'verdict.'
            % (n, d, span, bands.get('up', 0), bands.get('down', 0),
               w.ref_budget, ms.expected_at_reference(d),
               w.scale_band[0], w.scale_band[1],
               w.scale_extremes[0], w.scale_extremes[1]))
        # tint the rows inside the window
        for r, m in enumerate(ms):
            inside = m.delta <= d
            for c in range(self.modeTable.columnCount()):
                it = self.modeTable.item(r, c)
                if it is not None:
                    it.setBackground(QColor('#eef6ff') if inside
                                     else QColor(Qt.white))
        self._draw_threshold()

    def _draw_threshold(self):
        ms = self._modeset
        if self.threshPlot is None or ms is None or not len(ms):
            return
        self.threshPlot.clear()
        deepest = max(ms.deepest_found(), 1.0)
        xs = np.linspace(0.0, max(50.0, deepest * 1.2), 200)
        found = np.array([ms.n_distinct(x) for x in xs], dtype=float)
        w = ms.prov[0].power
        self.threshPlot.plot(xs, found, pen=pg.mkPen('#1f77b4', width=2),
                             name='distinct positions FOUND')
        self.threshPlot.plot(xs, found * w.shortfall,
                             pen=pg.mkPen('#888', width=2, style=Qt.DashLine),
                             name='expected at the %d-start reference (extrapolated)'
                             % w.ref_budget)
        band = pg.LinearRegionItem(values=w.scale_band, movable=False,
                                   brush=(255, 200, 0, 40))
        self.threshPlot.addItem(band)
        vline = pg.InfiniteLine(pos=deepest, angle=90,
                                pen=pg.mkPen('#a00', style=Qt.DotLine))
        self.threshPlot.addItem(vline)

    def _overlay_curves(self):
        ms = self._modeset
        if self.curvePlot is None or ms is None:
            return
        try:
            q, y, e, _fn, _sc, _w = self._active()
        except Exception:                      # noqa: BLE001
            return
        names = ['data']
        self.curvePlot.add_data(q, y, yerr=e, name='data')
        pal = getattr(self.host, '_curve_palette', None) or ['#1f77b4']
        for i, m in enumerate(ms):
            if m.R is None or len(m.R) != len(q):
                continue
            nm = 'mode %d (d=%.2f)' % (i + 1, m.delta)
            self.curvePlot.add_data(q, m.R, name=nm, fit=True,
                                    color=pal[i % len(pal)],
                                    linestyle=Qt.DashLine)
            names.append(nm)
        try:
            self.curvePlot.Plot(names)
            self.curvePlot.setXLabel('q')
            self.curvePlot.setYLabel('R/R_F' if getattr(self._func(), 'rrf', True)
                                     else 'R')
        except Exception:                      # noqa: BLE001
            pass

    def _selected(self):
        rows = self.modeTable.selectionModel().selectedRows() \
            if self.modeTable.selectionModel() else []
        if not rows or self._modeset is None:
            return None
        i = rows[0].row()
        return self._modeset[i] if i < len(self._modeset) else None

    def _mode_selected(self):
        m = self._selected()
        if m is None:
            return
        p = self._modeset.prov[0]
        L = ['mode -logL %.6f   delta %.6f   chi2/N %.5g' % (m.neglogL, m.delta,
                                                             m.chi2_per_n),
             'class %s (%s)   eigmin %s   cond %s   floored %d'
             % (m.hess_class, m.hess_source, m.eigmin, m.cond, m.n_floored),
             'starts %d   screening rank %d   flags %s'
             % (m.n_starts, m.stage1_rank, ','.join(m.flags) or '-')]
        if m.active_bounds:
            L.append('AT A SEARCH-BOX FACE: %s. The box is a scan artefact; a '
                     'refine with unbounded angles may leave this point.'
                     % ', '.join(m.active_bounds))
        L.append('objective %s, fit scale %s, %d active points, spacez %.3g'
                 % (p.objective, p.fit_scale, p.n_active, p.spacez))
        self.detail.setPlainText('\n'.join(L))

    def _load_selected(self):
        m = self._selected()
        if m is None:
            return
        try:
            self._install(m)
        except Exception:                      # noqa: BLE001
            QMessageBox.warning(self, 'Could not load the mode',
                                traceback.format_exc()[-1500:], QMessageBox.Ok)

    def _install(self, m):
        """Write a mode's VALUES into the live parameter table, and replot.

        Values only -- column 1. vary/min/max/expr are the user's fit setup, and the
        scan box (theta in [0,180], pinned yscale) must never be written back: that
        would silently change the next fit and would put bounds on angles that are
        deliberately unbounded.

        Follows undoFit's idiom: disconnect cellChanged, write, reconnect, then ONE
        update_plot. Disconnecting is mandatory -- each handler calls update_plot, so
        writing N cells live would trigger N model evaluations and can re-enter.
        """
        import copy
        host = self.host
        fit = host.fit
        stale = self._modeset.compatible_with(
            data_sha256=fs_modes.data_fingerprint(*self._active()[:3]))
        if stale:
            if QMessageBox.question(
                    self, 'Stale mode set',
                    '%s\n\nLoad it anyway?' % stale,
                    QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
                return

        nbox_now = self._nbox(self._func())
        if self._modeset.prov[0].num_boxes not in (0, nbox_now):
            QMessageBox.warning(
                self, 'Layer count changed',
                'This mode set was produced with %d boxes and the model now has %d. '
                'The parameter names no longer line up, and chi-squared is not '
                'comparable across a different mesh.'
                % (self._modeset.prov[0].num_boxes, nbox_now), QMessageBox.Ok)
            return

        # arm XModFit's own Undo Fit with the pre-load state
        host.oldParams = copy.copy(fit.params)
        if fit.params.get('__mpar__'):
            host.oldmpar = copy.deepcopy(host.mfitParamData)

        tables = [host.sfitParamTableWidget]
        for k in range(host.mfitParamTabWidget.count()):
            tables.append(host.mfitParamTableWidget[
                host.mfitParamTabWidget.tabText(k)])
        for t in tables:
            try:
                t.cellChanged.disconnect()
            except Exception:                  # noqa: BLE001
                pass
        try:
            for name, val in m.params.items():
                if name in fit.fit_params:
                    fit.fit_params[name].set(value=float(val))
                if name in fit.params:
                    fit.params[name] = float(val)
            f = self._func()
            try:
                f.canonical_params(fit.fit_params)
            except Exception:                  # noqa: BLE001
                pass
            for t in tables:
                for row in range(t.rowCount()):
                    it0 = t.item(row, 0)
                    if it0 is None:
                        continue
                    key = it0.text()
                    if key in fit.fit_params and t.item(row, 1) is not None:
                        t.item(row, 1).setText(host.format
                                               % fit.fit_params[key].value)
        finally:
            host.sfitParamTableWidget.cellChanged.connect(host.sfitParamChanged)
            for k in range(host.mfitParamTabWidget.count()):
                host.mfitParamTableWidget[
                    host.mfitParamTabWidget.tabText(k)
                ].cellChanged.connect(host.mfitParamChanged_new)
        # Hang the set on the Fit so other tools (the NGL viewer, a later docked
        # panel, the MCMC hand-off) can find it. Named `modes`, not `minima`: the
        # list contains saddles too, and calling it "minima" would be the same
        # category error as calling it "the minima". `ModeSet.minima()` is the
        # certified subset.
        try:
            fit.modes = self._modeset
        except Exception:                      # noqa: BLE001
            pass
        host.update_plot()

    def _load_and_refine(self):
        m = self._selected()
        if m is None:
            return
        self._install(m)
        try:
            self.host.doFit()
        except Exception:                      # noqa: BLE001
            QMessageBox.warning(self, 'The refine failed',
                                traceback.format_exc()[-1500:], QMessageBox.Ok)

    def _save_modes(self):
        if self._modeset is None:
            return
        try:
            default = fs_io.default_path(self._modeset.prov[0].dataset
                                         or 'modes.txt')
        except Exception:                      # noqa: BLE001
            default = 'modes.xmodes'
        fn, _ = QFileDialog.getSaveFileName(self, 'Save mode set', default,
                                            'Mode sets (*.xmodes)')
        if fn:
            fs_io.save(fn, self._modeset)
            self._log('saved %s' % fn)

    def _open_modes(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, 'Load mode set', getattr(self.host, 'curDir', os.getcwd()),
            'Mode sets (*.xmodes);;All files (*)')
        if not fn:
            return
        try:
            self._modeset = fs_io.load(fn)
        except Exception as ex:                # noqa: BLE001
            QMessageBox.warning(self, 'Cannot read the mode set', str(ex),
                                QMessageBox.Ok)
            return
        self._show_modes()
        self.tabs.setCurrentIndex(3)

    def closeEvent(self, ev):
        if self._proc is not None and self._proc.is_alive():
            self._stop()
        QDialog.closeEvent(self, ev)


def _current_unit_point(xb, model, mkey, nbox):
    """Current physical parameters as a unit-cube point, for the cost probe.

    Requires a unit-cube load to be active, which is why it is only called right
    after load_for_scan.
    """
    model.update_parameters()
    p = model._full_params(model.__d__[mkey], model.__rho__[mkey], nbox)
    return xb.to_normalized(p)
