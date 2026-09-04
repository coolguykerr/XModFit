import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(ROOT); sys.path.append(ROOT)  # append so installed xraydb wins over local ./xraydb dir
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)
import xmodfit
w = xmodfit.XModFit()
print("constructed:", type(w).__name__)
print("has dataset_fits:", hasattr(w, "dataset_fits"), "template:", hasattr(w, "_template_fit"))
print("helpers:", all(hasattr(w, h) for h in ("_clone_fit", "_fit_for_dataset", "_activate_dataset_fit", "_sync_funclist_to_fit")))
print("CONSTRUCT_OK")
