import os, sys, types as _t
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
SCRATCH = os.path.join(ROOT, 'tests', '_logs'); os.makedirs(SCRATCH, exist_ok=True)
_xr = _t.ModuleType('xraydb')
class X:
    def __getattr__(s, n): return lambda *a, **k: None
_xr.XrayDB = X; sys.modules['xraydb'] = _xr

import numpy as np
from PyQt5.QtWidgets import QApplication
import xmodfit
from lmfit.customhdf import CustomHDFBackend

app = QApplication([])
w = xmodfit.XModFit()

# ---- fake an individual fit context ----
class R: pass
class F: pass
w.fit = F()
w.fit.result = R()
w.fit.result.var_names = ['qoff', 'theta', 'phi', 'protins', 'cov']
w.fit.y = np.linspace(1.0, 0.01, 80)
w.fit.imin, w.fit.imax = 0, 79
w.sfnames = ['0<>C:/data/i2_tim3_1000.rrf']
w._mcmc_is_joint = False

tmp = os.path.join(SCRATCH, 'fp_test.h5')
if os.path.exists(tmp): os.remove(tmp)
w.MCMCBackendFile = tmp
w.backend = CustomHDFBackend(tmp)
w.backend.reset(10, len(w.fit.result.var_names))     # create the group
w.autoCorrTime = np.array([[0, 1.0], [100, 5.0], [200, 4.6]])

# ---- write + read ----
w._write_mcmc_fingerprint()
fp, act = w._read_mcmc_fingerprint()
print('fp dataset:', fp['dataset'], 'varnames:', fp['varnames'], 'ndata:', fp['ndata'])
assert fp is not None and act is not None
assert np.allclose(act, w.autoCorrTime), 'autoCorrTime not round-tripped'

# ---- exact match -> no mismatch ----
m0 = w._mcmc_fingerprint_mismatch(fp)
print('match mismatch-msg: %r' % m0)
assert m0 == ''

# ---- different dataset -> mismatch names 'dataset' ----
w.sfnames = ['0<>C:/data/SOMETHING_ELSE.rrf']
m1 = w._mcmc_fingerprint_mismatch(fp)
print('diff-dataset msg:', m1.replace(chr(10), ' | '))
assert 'dataset' in m1
w.sfnames = ['0<>C:/data/i2_tim3_1000.rrf']   # restore

# ---- different data (same name) -> data/fit-range mismatch ----
w.fit.y = np.linspace(2.0, 0.02, 80)
m2 = w._mcmc_fingerprint_mismatch(fp)
print('diff-data msg:', m2.replace(chr(10), ' | '))
assert 'data' in m2 or 'fit-range' in m2
w.fit.y = np.linspace(1.0, 0.01, 80)          # restore

# ---- different varnames -> parameters differ ----
w.fit.result.var_names = ['qoff', 'theta', 'phi']
m3 = w._mcmc_fingerprint_mismatch(fp)
print('diff-params msg:', m3.replace(chr(10), ' | '))
assert 'parameters' in m3

# ---- no-fingerprint file -> read returns (None, None) ----
tmp2 = os.path.join(SCRATCH, 'fp_none.h5')
if os.path.exists(tmp2): os.remove(tmp2)
w.backend = CustomHDFBackend(tmp2); w.backend.reset(10, 3)
fp2, act2 = w._read_mcmc_fingerprint()
print('no-fingerprint read:', fp2, act2)
assert fp2 is None and act2 is None

print('FINGERPRINT OK')
