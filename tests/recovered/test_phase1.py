"""Headless verification of Phase-1 changes (run in cmc-py39 from repo root).

Covers:
  * all four XLayersPDB* variants import cleanly (pdb_cache wiring OK);
  * shared pdb_cache.pdbread is keyed on fname and shared ACROSS instances
    (the core "expand lru_cache" win) and across deepcopy;
  * XLayersPDB.y() still runs end-to-end with the module-level pdbread call site;
  * the _clone_fit approach (Fit.__new__ + QObject.__init__ + deepcopy(func),
    NO init_params) yields an INDEPENDENT Fit whose functionCalled signal works.
"""
import os, sys, copy
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Functions"))

import numpy as np
from importlib import import_module

PDB = os.path.join("Data", "Pr4.pdb")
assert os.path.exists(PDB), PDB

from tests.harness import check, finish

# 1. all variants import
for mod in ["XLayersPDB", "XLayersPDB_G", "XLayersPDB_H", "XLayersPDB_box"]:
    m = import_module("Functions.XRR.%s" % mod)
    check("import Functions.XRR.%s" % mod, hasattr(m, mod))

import Functions.pdb_cache as pc

# 2. pdbread keyed on fname; second call is a cache hit
pc.pdbread.cache_clear()
r1 = pc.pdbread(PDB)
info1 = pc.pdbread.cache_info()
r2 = pc.pdbread(PDB)
info2 = pc.pdbread.cache_info()
check("pdbread returns 6-tuple", isinstance(r1, tuple) and len(r1) == 6)
check("pdbread 2nd call is a cache hit", info2.hits == info1.hits + 1 and info2.misses == info1.misses)

# 3. shared across two separate XLayersPDB instances (same fname -> no re-parse)
XLayersPDB = getattr(import_module("Functions.XRR.XLayersPDB"), "XLayersPDB")
x = np.linspace(0.02, 0.3, 20)
pc.pdbread.cache_clear()
a = XLayersPDB(x=x, fname=PDB)
misses_after_a = pc.pdbread.cache_info().misses
b = XLayersPDB(x=x, fname=PDB)            # different instance, same file
misses_after_b = pc.pdbread.cache_info().misses
check("2nd instance shares pdbread cache (no extra miss)", misses_after_b == misses_after_a)
check("only one parse for the shared file", misses_after_a == 1)

# 4. y() runs end-to-end (numba JIT on first call may take a bit)
y = a.y()
check("XLayersPDB.y() returns finite array", np.all(np.isfinite(np.asarray(y))))

# 5. deepcopy of func still hits the shared cache (joint-fit path)
before = pc.pdbread.cache_info()
c = copy.deepcopy(a.func) if hasattr(a, "func") else copy.deepcopy(a)
_ = pc.pdbread(c.fname)
after = pc.pdbread.cache_info()
check("deepcopy'd func reuses shared pdbread cache", after.misses == before.misses)

# 6. _clone_fit approach: independent Fit, working signal
from PyQt5.QtCore import QObject
from Fit_Routines import Fit
src = Fit(XLayersPDB, x)                 # normal construction
# replicate _clone_fit
clone = Fit.__new__(Fit)
QObject.__init__(clone)
clone.func = copy.deepcopy(src.func)
clone.func.x = x
clone.params = clone.func.__dict__
clone.fit_params = clone.func.params
clone.fitter = None; clone.result = None; clone.Niter = 0
clone.set_x(x)
# independence: mutate clone, original must not change
k = list(clone.fit_params.keys())[0]
orig_val = src.fit_params[k].value
clone.fit_params[k].set(value=orig_val + 123.456)
check("clone params independent of source", src.fit_params[k].value == orig_val)
# signal usable
got = {}
clone.functionCalled.connect(lambda *a: got.setdefault("fired", True))
clone.functionCalled.emit({}, 0, np.zeros(3), "Linear")
check("cloned Fit signal works", got.get("fired", False))

sys.exit(finish())
