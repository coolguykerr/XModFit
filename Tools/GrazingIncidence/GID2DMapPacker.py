"""
GID2DMapPacker.py -- Tools entry: turn a raw 2D GID map into a file GID2D can fit,
and optionally load it in one click.

Tools auto-discovery contract (xmodfit.py:646-658, launch_tApp :670-682): the file
must sit in a SUBFOLDER of Tools/, the class name must equal the file name, and it
is constructed as ``Cls(parent)`` with the main window as the first argument. The
dialog is cached and only ``.show()`` is called on reopen, so anything that must
reflect the current app state is refreshed in ``showEvent``.

Why this tool exists
--------------------
GID2D's map mode needs the DATASET's x column to be a row index into the map,
because XModFit slices with np.where on x (so x must be monotone) and raw q_xy
from a 2D map is a sawtooth. Loading the raw (q_xy, q_z, I, Ierr) file instead
makes Data_Dialog pick q_xy as x and q_z as y: every q value rounds into rows 0-2,
so the 2D panel draws ~2 pixels of a ~40000-cell grid AND the fit regresses the
model against q_z. The cure is a one-off repack, and requiring a terminal for it
is exactly the friction this removes.

The packed column order ['idx','I','Ierr','qxy','qz'] is chosen so Data_Dialog's
DEFAULTS land correctly with no configuration: x=idx, y=I, yerr=Ierr (its yerr
combo has 'None' at index 0, so its default index 3 selects the third column).
"""
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QFileDialog, QCheckBox,
                             QTextEdit, QMessageBox)

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./Functions'))


class GID2DMapPacker(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.host = parent
        self.setWindowTitle('Pack a 2D GID map for GID2D')
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel(
            'GID2D fits a 2D map through a ROW INDEX, so the dataset you load must be\n'
            'the indexed file -- not the raw (q_xy, q_z, I, Ierr) file. This writes it.'))

        row = QHBoxLayout()
        row.addWidget(QLabel('Raw map'))
        self.srcLineEdit = QLineEdit('')
        row.addWidget(self.srcLineEdit)
        b = QPushButton('Browse...')
        b.clicked.connect(self._browse_src)
        row.addWidget(b)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Output  '))
        self.dstLineEdit = QLineEdit('')
        row.addWidget(self.dstLineEdit)
        b2 = QPushButton('Browse...')
        b2.clicked.connect(self._browse_dst)
        row.addWidget(b2)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel('Columns in the raw file (0-based): q_xy, q_z, I, Ierr ='))
        self.colsLineEdit = QLineEdit('0,1,2,3')
        self.colsLineEdit.setMaximumWidth(90)
        row.addWidget(self.colsLineEdit)
        row.addStretch(1)
        lay.addLayout(row)

        self.dropSubHorizon = QCheckBox(
            'Drop q_z < 0 (below the horizon, where the model is identically zero)')
        self.dropSubHorizon.setChecked(True)
        lay.addWidget(self.dropSubHorizon)

        self.loadCheckBox = QCheckBox(
            'Load the result as a dataset and select GISAXS/GID2D')
        self.loadCheckBox.setChecked(True)
        lay.addWidget(self.loadCheckBox)

        row = QHBoxLayout()
        self.packButton = QPushButton('Pack')
        self.packButton.clicked.connect(self._pack)
        row.addWidget(self.packButton)
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        row.addWidget(close)
        lay.addLayout(row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log)
        self.resize(720, 420)
        self._prefill()

    # ------------------------------------------------------------------ ui --
    def showEvent(self, event):
        # the cached dialog must reflect the CURRENT selection, not the one that
        # was live when it was first constructed
        self._prefill()
        QDialog.showEvent(self, event)

    def _prefill(self):
        """Suggest the selected dataset's file if it looks like a raw map."""
        try:
            names = getattr(self.host, 'sfnames', None) or []
            if names and not self.srcLineEdit.text().strip():
                path = names[-1].split('<>')[-1]
                if os.path.exists(path) and not self._looks_indexed(path):
                    self.srcLineEdit.setText(path)
                    self._suggest_dst()
        except Exception:
            pass

    @staticmethod
    def _looks_indexed(path):
        try:
            import numpy as np
            d = np.loadtxt(path, comments='#', max_rows=3)
            return d.ndim == 2 and d.shape[1] >= 5
        except Exception:
            return False

    def _suggest_dst(self):
        src = self.srcLineEdit.text().strip()
        if not src:
            return
        base = os.path.splitext(os.path.basename(src))[0]
        root = os.path.abspath(os.curdir)
        self.dstLineEdit.setText(os.path.join(root, 'Data', base + '_indexed.txt'))

    def _browse_src(self):
        f = QFileDialog.getOpenFileName(self, 'Raw 2D GID map',
                                        directory=getattr(self.host, 'curDir', '.'))[0]
        if f:
            self.srcLineEdit.setText(f)
            self._suggest_dst()

    def _browse_dst(self):
        f = QFileDialog.getSaveFileName(self, 'Write indexed map to',
                                        directory=self.dstLineEdit.text() or '.')[0]
        if f:
            self.dstLineEdit.setText(f)

    def _say(self, txt):
        self.log.append(txt)
        self.log.ensureCursorVisible()

    # ---------------------------------------------------------------- work --
    def _pack(self):
        import numpy as np
        import gid2d_core as GC
        src = self.srcLineEdit.text().strip()
        dst = self.dstLineEdit.text().strip()
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, 'No input', 'Pick a raw map file first.',
                                QMessageBox.Ok)
            return
        if not dst:
            self._suggest_dst()
            dst = self.dstLineEdit.text().strip()
        try:
            cols = tuple(int(t) for t in self.colsLineEdit.text().split(','))
            if len(cols) != 4:
                raise ValueError('need exactly 4 column numbers')
        except Exception as ex:
            QMessageBox.warning(self, 'Columns', 'Could not read the column list: %s'
                                % ex, QMessageBox.Ok)
            return
        d = os.path.dirname(os.path.abspath(dst))
        if not os.path.isdir(d):
            os.makedirs(d)
        try:
            info = GC.pack_map(src, dst, drop_masked=True,
                               qz_min=0.0 if self.dropSubHorizon.isChecked() else None,
                               cols=cols)
            mp = GC.load_map(dst)
        except Exception:
            import traceback
            self._say(traceback.format_exc())
            QMessageBox.warning(self, 'Packing failed',
                                'See the log in this window.', QMessageBox.Ok)
            return

        cells = mp['n_qxy'] * mp['n_qz']
        self._say('wrote %s' % info['path'])
        self._say('  kept %d rows, dropped %d' % (info['kept'], info['dropped']))
        self._say('  %s' % info['sigma_note'])
        self._say('  grid %d q_xy x %d q_z = %d cells, %d filled (%.1f%%)'
                  % (mp['n_qxy'], mp['n_qz'], cells, mp['n'],
                     100.0 * mp['n'] / max(1, cells)))

        if not self.loadCheckBox.isChecked():
            self._say('  now: Add files -> this file (NOT the raw one)')
            return
        try:
            self.host.addData(fnames=[dst])
            # select the dataset just added, or update_plot has nothing to fit
            dl = self.host.dataListWidget
            for i in range(dl.count()):
                if os.path.abspath(dst) in dl.item(i).text():
                    dl.clearSelection()
                    dl.item(i).setSelected(True)
                    self.host.dataFileSelectionChanged()
                    break
            # point GID2D at the same file and select it. findItems needs a real
            # Qt.MatchFlag: PyQt5 rejects a bare int here.
            cl = self.host.categoryListWidget
            items = cl.findItems('GISAXS', Qt.MatchExactly)
            if items:
                cl.setCurrentItem(items[0])
                fl = self.host.funcListWidget
                fitems = fl.findItems('GID2D', Qt.MatchExactly)
                if fitems:
                    fl.setCurrentItem(fitems[0])
            func = getattr(getattr(self.host, 'fit', None), 'func', None)
            if func is not None and hasattr(func, 'mapfile'):
                func.mapfile = os.path.abspath(dst)
                try:
                    func.filepaths['mapfile'] = func.mapfile
                    self.host.fit.params['mapfile'] = func.mapfile
                except Exception:
                    pass
                self.host.update_fixed_parameters()
                self.host.update_plot()
            self._say('  loaded, GID2D selected, mapfile pointed at it.')
            self._say('  set Fit Scale -> Log: sigma is multiplicative on this data, '
                      'so in Linear one near-zero pixel can carry most of the weight.')
        except Exception:
            import traceback
            self._say(traceback.format_exc())


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = GID2DMapPacker(None)
    w.show()
    sys.exit(app.exec_())
