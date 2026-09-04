"""Headless: does XModFit list XLayersPDB_XPR as an XRR function and build it?"""
import os
import sys
import types

REPO = r'C:\Users\kerr\Documents\GitHub\XModJointFit'
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'Functions'))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# xraydb is not installed in cmc-py39; stub it (only used for X-ray calculators).
if 'xraydb' not in sys.modules:
    m = types.ModuleType('xraydb')

    class XrayDB(object):
        def __getattr__(self, k):
            return lambda *a, **kw: None
    m.XrayDB = XrayDB
    sys.modules['xraydb'] = m

from PyQt5.QtWidgets import QApplication   # noqa: E402
app = QApplication.instance() or QApplication([])

import xmodfit                              # noqa: E402
w = xmodfit.XModFit()

cats = [w.categoryListWidget.item(i).text() for i in range(w.categoryListWidget.count())]
print('categories:', cats)
assert 'XRR' in cats, cats
w.categoryListWidget.setCurrentRow(cats.index('XRR'))
funcs = [w.funcListWidget.item(i).text() for i in range(w.funcListWidget.count())]
print('XRR functions:', funcs)
assert 'XLayersPDB_XPR' in funcs, 'XLayersPDB_XPR missing from the XRR function list'

# Selecting it builds the Fit object + parameter tables without touching Julia.
w.funcListWidget.setCurrentRow(funcs.index('XLayersPDB_XPR'))
f = w.fit.func
print('class:', type(f).__name__)
print('fixed params:', {k: v for k, v in f.__dict__.items()
                        if k in ('E', 'rrf', 'use_protein', 'vol_calc', 'fname')})
print('fit params:', list(f.params.keys()))
print('mpar:', f.__mpar__)
print('choices:', f.choices)
print('filepaths:', f.filepaths)

from lmfit.minimizer import _PDB_FUNCNAMES   # noqa: E402
assert 'XLayersPDB_XPR' in _PDB_FUNCNAMES
print('\nregistered in _PDB_FUNCNAMES: OK')
print('GUI LISTING OK')
