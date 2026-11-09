# An exception raised inside a Qt slot after pyqtgraph has replaced sys.excepthook (what
# importing xmodfit does). PyQt5 prints it and carries on; the process exits 0. The runner
# must say FAIL.
import os
import sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
import pyqtgraph.Qt  # noqa: F401  (installs pyqt_qabort_override)
app = QApplication.instance() or QApplication(sys.argv)


def boom():
    raise AssertionError('slot boom')


QTimer.singleShot(0, boom)
QTimer.singleShot(100, app.quit)
app.exec_()
sys.exit(0)
