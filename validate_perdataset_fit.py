"""
validate_perdataset_fit.py -- headless regression for Phase-1 "per-dataset fit
memory" + the shared PDB read cache.

Runs the real XModFit widget under QT_QPA_PLATFORM=offscreen and exercises:

  Part B (shared cache, Functions/pdb_cache.py):
    * all four XLayersPDB* variants import;
    * pdbread is keyed on fname and shared ACROSS instances and deepcopies
      (one parse per file per process).

  Part A (per-dataset fit memory, xmodfit.py):
    * selecting a dataset creates/uses its own Fit; distinct datasets get
      distinct Fit objects;
    * a value edited on one dataset survives switching away and back (memory);
    * the function list auto-follows the active dataset's model;
    * an actual multi-dataset fit runs, leaving each dataset an INDEPENDENT
      result on its own stored Fit;
    * removeData drops the dataset's remembered Fit.

Run (from repo root) in an env with the GUI deps, e.g.:
    QT_QPA_PLATFORM=offscreen python validate_perdataset_fit.py

Notes:
  * xraydb is stubbed: it is only used for X-ray calcs, not for the fit-memory
    path, and is not needed to exercise these code paths.
  * modal dialogs (QMessageBox / QProgressDialog / FitResultDialog) are patched
    to non-blocking no-ops so the fit can be driven without a display.
"""
import os, sys, types, copy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# --- stub xraydb ONLY if it's genuinely unavailable (see module docstring). In
#     an env where xraydb is installed (e.g. cmcars) the real one is used. ---
try:
    from xraydb import XrayDB as _RealXrayDB  # noqa: F401
except Exception:
    _xr = types.ModuleType("xraydb")
    class _XrayDB:
        def __init__(self, *a, **k): pass
        def __getattr__(self, name): return lambda *a, **k: 0
    _xr.XrayDB = _XrayDB
    _xr.f0 = lambda *a, **k: 0
    _xr.__path__ = []
    sys.modules["xraydb"] = _xr
    print("NOTE: xraydb not importable here -- using a stub for this run.")

sys.path.insert(0, ROOT)

import numpy as np
from importlib import import_module
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

_OK = {"ok": True}
def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    _OK["ok"] = _OK["ok"] and bool(cond)


def main():
    app = QApplication.instance() or QApplication(sys.argv)

    # ---------------- Part B: shared pdb cache ----------------
    import Functions.pdb_cache as pc
    for mod in ("XLayersPDB", "XLayersPDB_G", "XLayersPDB_H", "XLayersPDB_box"):
        m = import_module("Functions.XRR.%s" % mod)
        check("import %s" % mod, hasattr(m, mod))
    pdbs = [f for f in os.listdir("Data") if f.endswith(".pdb")] if os.path.isdir("Data") else []
    if pdbs:
        pdb = os.path.join("Data", sorted(pdbs, key=lambda f: os.path.getsize(os.path.join("Data", f)))[0])
        XLayersPDB = getattr(import_module("Functions.XRR.XLayersPDB"), "XLayersPDB")
        xq = np.linspace(0.02, 0.3, 15)
        pc.pdbread.cache_clear()
        a = XLayersPDB(x=xq, fname=pdb)
        m1 = pc.pdbread.cache_info().misses
        b = XLayersPDB(x=xq, fname=pdb)      # 2nd instance, same file
        m2 = pc.pdbread.cache_info().misses
        check("pdbread parses shared file once across instances", m1 == 1 and m2 == 1)
        c = copy.deepcopy(a.func) if hasattr(a, "func") else copy.deepcopy(a)
        _ = pc.pdbread(c.fname)
        check("deepcopy reuses shared pdbread cache", pc.pdbread.cache_info().misses == m2)
    else:
        print("SKIP - no PDB in ./Data for cache-sharing check")

    # ---------------- build the app ----------------
    import xmodfit
    # neutralize blocking dialogs so a fit can run headless
    class _MB:
        Ok = Yes = No = 0
        @staticmethod
        def warning(*a, **k): return 0
        @staticmethod
        def question(*a, **k): return 0
        @staticmethod
        def information(*a, **k): return 0
    xmodfit.QMessageBox = _MB
    class _Prog:
        def __init__(self, *a, **k): pass
        def wasCanceled(self): return False
        def __getattr__(self, name): return lambda *a, **k: None
    xmodfit.QProgressDialog = _Prog
    class _FRD:
        def __init__(self, *a, **k): pass
        def exec_(self): return 0
        def show(self): pass
    xmodfit.FitResultDialog = _FRD

    w = xmodfit.XModFit()
    check("XModFit constructs offscreen", hasattr(w, "dataset_fits"))

    # two tiny datasets
    d1 = os.path.join(os.environ.get("TEMP", ROOT), "vpf_ds1.txt")
    d2 = os.path.join(os.environ.get("TEMP", ROOT), "vpf_ds2.txt")
    q = np.linspace(0.02, 0.30, 40)
    for path, amp in ((d1, 1.0), (d2, 2.0)):
        y = amp * np.exp(-((q - 0.1) ** 2) / 0.002) + 0.01
        np.savetxt(path, np.c_[q, y, 0.05 * y], header="x y yerr")
    w.addData(fnames=[d1, d2])
    names = [w.dataListWidget.item(i).text() for i in range(w.dataListWidget.count())]
    check("two datasets loaded", len(names) == 2)

    # choose a cheap non-PDB model
    w.categoryListWidget.setCurrentItem(w.categoryListWidget.findItems("XRR", Qt.MatchExactly)[0])
    w.funcListWidget.setCurrentItem(w.funcListWidget.findItems("XLayers", Qt.MatchExactly)[0])

    # select ds0, tag it; select ds1; back to ds0 -> memory
    w.dataListWidget.setCurrentRow(0)
    check("ds0 gets a stored fit", names[0] in w.dataset_fits)
    fit0 = w.dataset_fits[names[0]]
    sk = [k for k in fit0.fit_params.keys() if k[:2] != "__"][0]
    fit0.fit_params[sk].set(value=0.4242)
    fit0.func.__dict__[sk] = 0.4242
    w.dataListWidget.setCurrentRow(1)
    check("ds1 has its own distinct fit", w.dataset_fits[names[1]] is not fit0)
    check("active fit follows selection", w.fit is w.dataset_fits[names[1]])
    check("function list auto-followed", w.funcListWidget.currentItem().text() == "XLayers")
    w.dataListWidget.setCurrentRow(0)
    check("ds0 fit persisted (same object)", w.dataset_fits[names[0]] is fit0)
    check("ds0 edited value remembered", abs(fit0.fit_params[sk].value - 0.4242) < 1e-9)

    # ---------------- an actual multi-dataset fit ----------------
    for nm in names:
        f = w.dataset_fits[nm]
        f.fit_params["yscale"].set(vary=1)   # ensure >=1 varying param
    w.fitIterationLineEdit.setText("15")     # bound runtime
    w.dataListWidget.clearSelection()
    w.dataListWidget.item(0).setSelected(True)
    w.dataListWidget.item(1).setSelected(True)
    try:
        w.doFit()
        ran = True
    except Exception as e:
        import traceback; traceback.print_exc()
        ran = False
    check("multi-dataset doFit ran without error", ran)
    r0 = getattr(w.dataset_fits[names[0]], "result", None)
    r1 = getattr(w.dataset_fits[names[1]], "result", None)
    check("both datasets have a fit result", r0 is not None and r1 is not None)
    check("per-dataset results are independent objects", r0 is not r1)

    # ---------------- per-dataset PDB file persistence (XLayersPDB) ----------------
    # each dataset must remember its OWN last PDB across switches (regression for
    # the fname/filepaths desync that reverted to the default PDB on switch).
    p1 = os.path.join("Data", "Pr4.pdb"); p2 = os.path.join("Data", "Nd9.pdb")
    if os.path.exists(p1) and os.path.exists(p2):
        def _assign_pdb(row, path):
            w.dataListWidget.clearSelection()
            w.dataListWidget.item(row).setSelected(True)          # -> that dataset active
            it = w.funcListWidget.findItems("XLayersPDB", Qt.MatchExactly)[0]
            w.funcListWidget.clearSelection(); w.funcListWidget.setCurrentItem(it); it.setSelected(True)
            w.functionChanged()                                    # assign XLayersPDB to it
            w.fit.func.fname = path                                # simulate picking a PDB
            w.update_fixed_parameters(); w.update_plot()           # button<-fname ; fname<-button
        _assign_pdb(0, p1)
        _assign_pdb(1, p2)
        w.dataListWidget.clearSelection(); w.dataListWidget.item(0).setSelected(True)
        check("ds0 remembers its PDB (Pr4) after switch", os.path.basename(w.fit.func.fname) == "Pr4.pdb")
        w.dataListWidget.clearSelection(); w.dataListWidget.item(1).setSelected(True)
        check("ds1 remembers its PDB (Nd9) after switch", os.path.basename(w.fit.func.fname) == "Nd9.pdb")
    else:
        print("SKIP - Data/Pr4.pdb & Data/Nd9.pdb not both present for PDB-persistence check")

    # ---------------- removeData drops the store entry ----------------
    w.dataListWidget.clearSelection()
    w.dataListWidget.item(1).setSelected(True)
    w.removeData()
    check("removeData drops the dataset's remembered fit", names[1] not in w.dataset_fits)
    check("other dataset's fit is retained", names[0] in w.dataset_fits)

    print("\nRESULT:", "ALL PASS" if _OK["ok"] else "FAILURES PRESENT")
    return 0 if _OK["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
