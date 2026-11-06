import os, sys, types as _t
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)

# stub xraydb (not pip-installed in cmc-py39; only used for X-ray calcs, not for
# constructing the module) -- see cmc-py39-env memory
_xr = _t.ModuleType('xraydb')
class _XrayDB:
    def __getattr__(self, name):
        return lambda *a, **k: None
_xr.XrayDB = _XrayDB
sys.modules['xraydb'] = _xr

import numpy as np

# ---------- 1. Engine: per-dataset ranges vs scalar ----------
from Joint_Fit_Routines import JointFit
jf = JointFit()
# build objectives manually (bypass model construction) -- _set_ranges only reads x/name
x0 = np.linspace(0.0, 1.0, 101)
x1 = np.linspace(0.0, 2.0, 201)
jf.objectives = [
    {"name": "A", "x": x0, "imin": 0, "imax": len(x0) - 1},
    {"name": "B", "x": x1, "imin": 0, "imax": len(x1) - 1},
]
# scalar (old behavior)
jf._set_ranges(0.2, 0.8)
sA = (jf.objectives[0]["imin"], jf.objectives[0]["imax"])
sB = (jf.objectives[1]["imin"], jf.objectives[1]["imax"])
# per-dataset by name
jf._set_ranges(0.2, 0.8, ranges={"A": (0.1, 0.5), "B": (0.5, 1.5)})
pA = (jf.objectives[0]["imin"], jf.objectives[0]["imax"])
pB = (jf.objectives[1]["imin"], jf.objectives[1]["imax"])
# per-dataset with None -> scalar fallback for A
jf._set_ranges(0.2, 0.8, ranges={"B": (0.5, 1.5)})
fA = (jf.objectives[0]["imin"], jf.objectives[0]["imax"])
print("scalar  A", sA, "x=", round(x0[sA[0]],3), round(x0[sA[1]],3))
print("scalar  B", sB, "x=", round(x1[sB[0]],3), round(x1[sB[1]],3))
print("perdata A", pA, "x=", round(x0[pA[0]],3), round(x0[pA[1]],3))
print("perdata B", pB, "x=", round(x1[pB[0]],3), round(x1[pB[1]],3))
print("fallbackA", fA, "== scalar A?", fA == sA)
assert pA != sA and pB != sB, "per-dataset ranges did not differ"
assert fA == sA, "None-entry did not fall back to scalar"
# A: 0.1..0.5 -> imin where x>=0.099 -> idx 0? x0[0]=0 >=0.099? no. 0.099*... clamp uses 0.99*lo=0.099 -> first x>=0.099 is 0.1(idx10)
assert abs(x0[pA[0]] - 0.1) < 1e-9 and abs(x0[pA[1]] - 0.5) < 1e-9
assert abs(x1[pB[0]] - 0.5) < 1e-9 and abs(x1[pB[1]] - 1.5) < 1e-9
print("ENGINE RANGES OK\n")

# ---------- 2. Spec / IO round-trip for #FitRange ----------
import joint_fit_io as jio
settings = {'fit_range': '0.02:0.55', 'fit_method': 'leastsq', 'fit_scale': 'Linear', 'fit_iterations': '1000'}
blocks = [{'path': '/data/a.rrf', 'category': 'XRR', 'func': 'XLayers',
           'xrange': 'np.linspace(0.0,0.6,100)', 'fitrange': '0.03:0.50',
           'cov': None, 'raw': None, 'fixed': ['fname\tx'], 'sfit': [], 'mfit': []},
          {'path': '/data/b.rrf', 'category': 'XRR', 'func': 'XLayers',
           'xrange': 'np.linspace(0.0,0.6,100)', 'fitrange': '',
           'cov': None, 'raw': None, 'fixed': [], 'sfit': [], 'mfit': []}]
text = jio.format_jpar(settings, blocks, [])
parsed = jio.parse_jpar(text)
b0, b1 = parsed['blocks']
print("block0 xrange :", repr(b0['xrange']))
print("block0 fitrange:", repr(b0['fitrange']))
print("block1 fitrange:", repr(b1['fitrange']))
assert b0['xrange'] == 'np.linspace(0.0,0.6,100)', b0['xrange']
assert b0['fitrange'] == '0.03:0.50', b0['fitrange']
assert b1['fitrange'] == '', repr(b1['fitrange'])
assert '#FitRange=' in text and '#Xrange=' in text
print("IO ROUND-TRIP OK\n")

# ---------- 2b. _parse_fitrange semantics (static logic, copied inline) ----------
def _parse_fitrange(text):
    if not text or ':' not in text:
        return None, None
    lo, _, hi = text.partition(':')
    def _f(s):
        s = s.strip()
        if s == '': return None
        try: return float(s)
        except ValueError: return None
    return _f(lo), _f(hi)
assert _parse_fitrange('0.03:0.50') == (0.03, 0.50)
assert _parse_fitrange('0.03:') == (0.03, None)
assert _parse_fitrange(':0.5') == (None, 0.5)
assert _parse_fitrange('') == (None, None)
assert _parse_fitrange('junk') == (None, None)
print("PARSE FITRANGE OK\n")

# ---------- 3. Corner draw helpers on a detached Figure (no Qt loop) ----------
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import corner
from collections import OrderedDict

# fake a tiny XModFit-like object with the pieces _draw_corner / heatmap use
class Fake:
    pass
f = Fake()
nsteps, nwalk, ndim = 200, 6, 4
keys = ['theta', 'phi', 'd0_qoff', '__lnsigma']
f.chain_shape = (nsteps, nwalk)
f.param_chain = OrderedDict()
rng = np.random.default_rng(0)
chain = rng.normal(size=(nsteps, nwalk, ndim))
for i, k in enumerate(keys):
    f.param_chain[k] = OrderedDict((j, chain[:, j, i]) for j in range(nwalk))

# bind the real methods
import types, xmodfit
for m in ('_corner_names', '_rebuild_pardata', '_corner_truths', '_draw_corner', '_corner_render_data', '_is_structural_param'):
    setattr(f, m, types.MethodType(getattr(xmodfit.XModFit, m), f))
f._STRUCTURAL_PARAMS = xmodfit.XModFit._STRUCTURAL_PARAMS
f._CORNER_MAX_SAMPLES = xmodfit.XModFit._CORNER_MAX_SAMPLES   # read by _corner_render_data
# final_fit_params substitute (dict-like with .value + .keys())
class P:
    def __init__(self, v): self.value = v
class FP(dict):
    pass
f.final_fit_params = FP({'theta': P(0.0), 'phi': P(0.1), 'd0_qoff': P(0.0)})

f._rebuild_pardata(first=50)
names, cols = f._corner_names(subset=['theta', 'phi'])
print("subset names:", names, "cols:", cols)
assert names == ['theta', 'phi'] and cols == [0, 1]
vals = f._corner_truths(names, cols)
fig = Figure()
n = f._draw_corner(fig, names, cols, vals, bins=20, percentile=85)
assert n == 2
# corner makes an n x n grid of axes
assert len(fig.get_axes()) == 4, len(fig.get_axes())
# __lnsigma excluded from "all"
alln, allc = f._corner_names()
assert '__lnsigma' not in alln and alln == ['theta', 'phi', 'd0_qoff']
# structural heuristic
assert f._is_structural_param('theta') and f._is_structural_param('phi')
assert not f._is_structural_param('d0_qoff')   # per-dataset, non-structural
assert f._is_structural_param('protins')       # bare shared
# correlation heatmap math
C = np.corrcoef(f.pardata[:, allc], rowvar=False)
assert C.shape == (3, 3)
print("CORNER + HEATMAP HELPERS OK\n")

print("ALL VERIFICATION PASSED")
