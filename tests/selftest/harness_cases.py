"""Driver for tests/test_harness.py: exercises tests.harness in one named way per case."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tests.harness import check, check_close, skip, finish  # noqa: E402


def all_pass():
    check('a', True, 'first')
    check('b', 1 == 1)


def one_fail():
    check('a', True)
    check('b', False, 'deliberate')


def no_checks():
    pass


def skipped():
    skip('no data')


def raise_mode():
    check('boom', False, 'raises under XMF_CHECK_RAISE=1')


def close_fail():
    check_close('close', 1.0, 1.5, 0.1)


def _raise_in_qt_slot():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    def boom():
        raise AssertionError('slot boom')
    QTimer.singleShot(0, boom)
    QTimer.singleShot(100, app.quit)
    app.exec_()


def slot_exception():
    # pyqtgraph is imported after tests.harness, as in every harness that imports xmodfit
    import pyqtgraph.Qt  # noqa: F401
    _raise_in_qt_slot()
    check('later check passes', True)


def slot_exception_late_hook():
    # something replaces sys.excepthook after tests.harness was imported; check()
    # must re-install the counting hook in front of it
    sys.excepthook = lambda *a: sys.__excepthook__(*a)
    check('first check passes', True)
    _raise_in_qt_slot()
    check('later check passes', True)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
    sys.exit(finish())
