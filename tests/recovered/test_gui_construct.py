import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT); sys.path.append(ROOT)  # append so installed xraydb wins over local ./xraydb dir
import types
try:
    from xraydb import XrayDB as _R
except Exception:
    _xr = types.ModuleType("xraydb")
    class _X:
        def __init__(s, *a, **k): pass
        def __getattr__(s, n): return lambda *a, **k: 0
    _xr.XrayDB = _X; _xr.f0 = lambda *a, **k: 0; _xr.__path__ = []; sys.modules["xraydb"] = _xr
from tests.harness import check, finish
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)
import xmodfit
w = xmodfit.XModFit()
print("constructed:", type(w).__name__)
print("has dataset_fits:", hasattr(w, "dataset_fits"), "template:", hasattr(w, "_template_fit"))
print("helpers:", all(hasattr(w, h) for h in ("_clone_fit", "_fit_for_dataset", "_activate_dataset_fit", "_sync_funclist_to_fit")))
print("CONSTRUCT_OK")

check('XModFit constructs', type(w).__name__ == 'XModFit')
check('per-dataset fit store and template exist',
      hasattr(w, 'dataset_fits') and hasattr(w, '_template_fit'))
check('per-dataset fit helpers exist', all(hasattr(w, h) for h in (
    '_clone_fit', '_fit_for_dataset', '_activate_dataset_fit', '_sync_funclist_to_fit')))
check('function categories are listed', w.categoryListWidget.count() > 0,
      w.categoryListWidget.count())
sys.exit(finish())
