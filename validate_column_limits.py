"""
validate_column_limits.py -- headless regression for per-column fit limits.

Covers Stage 0/3 of the generalization: the limits store with its xmin/xmax
views, column discovery from the retained DataFrame, mask construction, the
badge, and backward-compatible .xpar / .jpar round-trips.

    QT_QPA_PLATFORM=offscreen python validate_column_limits.py
"""
import os, sys, types, faulthandler
faulthandler.enable()

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
try:
    from xraydb import XrayDB as _R  # noqa
except Exception:
    _xr = types.ModuleType("xraydb")
    class _XrayDB:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return lambda *a, **k: 0
    _xr.XrayDB = _XrayDB; _xr.f0 = lambda *a, **k: 0; _xr.__path__ = []
    sys.modules["xraydb"] = _xr
sys.path.insert(0, ROOT)

import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

_OK = {"ok": True}
TMP = os.environ.get('TEMP', ROOT)


def check(name, cond, detail=''):
    print(("PASS" if cond else "FAIL"), "-", name, (' | ' + str(detail)) if detail else '')
    _OK["ok"] = _OK["ok"] and bool(cond)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    import xmodfit
    import PyQt5.QtWidgets as _QtW
    class _MB:
        Ok = Yes = No = 0
        warning = question = information = staticmethod(lambda *a, **k: 0)
    xmodfit.QMessageBox = _MB
    _QtW.QMessageBox = _MB
    class _Prog:
        def __init__(self, *a, **k): pass
        def wasCanceled(self): return False
        def __getattr__(self, n): return lambda *a, **k: None
    xmodfit.QProgressDialog = _Prog

    # a 5-column dataset with named columns, like the packed 2D map
    p = os.path.join(TMP, 'collim_ds.txt')
    n = 80
    idx = np.arange(n, dtype=float)
    qxy = 1.2 + 0.004 * (idx % 20)
    qz = 0.01 + 0.05 * (idx // 20)
    I = 10.0 + qxy * 3.0 + qz
    Ie = 0.1 * np.ones(n)
    with open(p, 'w') as fh:
        fh.write("#col_names=['idx','I','Ierr','qxy','qz']\n")
        for j in range(n):
            fh.write('%d %.6e %.6e %.6e %.6e\n' % (j, I[j], Ie[j], qxy[j], qz[j]))

    w = xmodfit.XModFit()
    w.addData(fnames=[p, p])            # >=2 files avoids the modal single-file path
    keys = list(w.data.keys())
    k0 = keys[0]
    w.dataListWidget.item(0).setSelected(True)
    w.dataFileSelectionChanged()

    # ------------------------------------------------- proxies are neutral ---
    check("xmin/xmax behave as mappings",
          isinstance(w.xmin[k0], float) and isinstance(w.xmax[k0], float),
          '%r:%r' % (w.xmin[k0], w.xmax[k0]))
    check("xmin/xmax are stored under limits[key]['x']",
          w.limits[k0]['x'] == (w.xmin[k0], w.xmax[k0]), w.limits[k0]['x'])
    w.xmin[k0] = 5.0
    check("writing xmin updates only the low bound",
          w.limits[k0]['x'][0] == 5.0 and w.limits[k0]['x'][1] == w.xmax[k0])
    check("iteration/len see the key", k0 in list(w.xmin) and len(w.xmin) == len(keys))

    # ------------------------------------------------- column discovery ------
    cols = w.available_columns(k0)
    names = [c for c, _ in cols]
    check("available_columns returns x plus the other four real names",
          names[0] == 'x' and set(names[1:]) == {'I', 'Ierr', 'qxy', 'qz'}, names)
    check("the x label names the real column", 'idx' in cols[0][1], cols[0][1])
    check("_column_values reads a named column",
          np.allclose(w._column_values(k0, 'qxy'), qxy))

    # ------------------------------------------------- mask construction -----
    check("no limits -> no mask", w._column_mask(k0) is None)
    w.limits[k0]['qz'] = (0.05, 0.12)
    m = w._column_mask(k0)
    exp = (qz >= 0.05) & (qz <= 0.12)
    check("single-column mask matches the data",
          m is not None and np.array_equal(m, exp),
          '%d of %d active' % (int(m.sum()), n))
    w.limits[k0]['qxy'] = (1.21, 1.25)
    m2 = w._column_mask(k0)
    exp2 = exp & (qxy >= 1.21) & (qxy <= 1.25)
    check("two column limits AND together", np.array_equal(m2, exp2),
          '%d active' % int(m2.sum()))
    check("one-sided limit works",
          np.array_equal(w._column_mask(k0, limits={'qz': (None, 0.12)}), qz <= 0.12))
    check("slice_ pre-slices the mask",
          np.array_equal(w._column_mask(k0, slice_=(10, 40)), exp2[10:40]))
    check("'x' alone does NOT produce a mask (it is the imin:imax window)",
          w._column_mask(k0, limits={'x': (0.0, 1.0)}) is None)

    # ------------------------------------------------- badge --------------- -
    w._refresh_limits_badge()
    check("badge counts the extra limits", w.limitsButton.text() == 'Limits (2)',
          w.limitsButton.text())
    w.limits[k0]['qz'] = (99.0, 100.0)              # empties the dataset
    w._refresh_limits_badge()
    check("badge turns red when nothing is left to fit",
          'red' in w.limitsButton.styleSheet(), w.limitsButton.styleSheet())
    w.limits[k0].pop('qz')
    w.limits[k0].pop('qxy')
    w._refresh_limits_badge()
    check("badge resets", w.limitsButton.text() == 'Limits', w.limitsButton.text())

    # ------------------------------------------------- format / parse -------
    w.limits[k0]['qxy'] = (1.21, 1.25)
    w.limits[k0]['qz'] = (0.05, None)
    txt = w._format_column_limits(k0)
    check("format emits name:lo:hi;... with a blank for unbounded",
          txt.count(';') == 1 and 'qxy:' in txt and txt.endswith(':'), txt)
    back = w._parse_column_limits(txt)
    check("parse round-trips", back == {'qxy': (1.21, 1.25), 'qz': (0.05, None)}, back)
    check("parse tolerates junk", w._parse_column_limits('garbage;;a:b:c;qz:0.1:0.2')
          == {'qz': (0.1, 0.2)}, w._parse_column_limits('garbage;;a:b:c;qz:0.1:0.2'))
    check("parse ignores an 'x' record (Fit Range is authoritative)",
          w._parse_column_limits('x:0:1;qz:0.1:0.2') == {'qz': (0.1, 0.2)})

    # ------------------------------------------------- .xpar round-trip -----
    w.categoryListWidget.setCurrentItem(
        w.categoryListWidget.findItems('XRR', Qt.MatchExactly)[0])
    w.funcListWidget.setCurrentItem(
        w.funcListWidget.findItems('XLayers', Qt.MatchExactly)[0])
    par = os.path.join(TMP, 'collim.xpar')
    w.saveParameters(fname=par)
    body = open(par).read()
    check(".xpar contains a #Column Limits= line", '#Column Limits=' in body,
          [l for l in body.splitlines() if 'Column Limits' in l])
    w.limits[k0] = {'x': w.limits[k0]['x']}          # wipe, then reload
    w.loadParameters(fname=par)
    got = {c: v for c, v in (w.limits.get(k0, {}) or {}).items() if c != 'x'}
    check(".xpar reload restores the column limits",
          got == {'qxy': (1.21, 1.25), 'qz': (0.05, None)}, got)

    # an OLD .xpar with no #Column Limits= must still load, and add none
    old_par = os.path.join(TMP, 'collim_old.xpar')
    with open(old_par, 'w') as fh:
        fh.write(''.join(l + '\n' for l in body.splitlines()
                         if not l.startswith('#Column Limits=')))
    w.limits[k0] = {'x': w.limits[k0]['x']}
    try:
        w.loadParameters(fname=old_par)
        got2 = {c: v for c, v in (w.limits.get(k0, {}) or {}).items() if c != 'x'}
        check("an old .xpar without the line still loads and adds no limits",
              got2 == {}, got2)
    except Exception as ex:
        check("an old .xpar without the line still loads and adds no limits",
              False, repr(ex))

    # ------------------------------------------------- joint_fit_io ---------
    import joint_fit_io as jio
    txt2 = jio.format_jpar({'fit_range': '0:1', 'column_limits': 'qz:0.1:0.2'},
                           [{'path': 'a.txt', 'category': 'XRR', 'func': 'XLayers',
                             'xrange': '', 'fitrange': '0.1:0.5',
                             'collimits': 'qxy:1.2:1.3', 'cov': None, 'raw': None,
                             'fixed': [], 'sfit': [], 'mfit': []}], [])
    if not isinstance(txt2, str):
        txt2 = os.linesep.join(txt2)   # format_jpar returns a list of lines
    check(".jpar writes both global and per-dataset column limits",
          '#Column Limits=qz:0.1:0.2' in txt2 and '#ColumnLimits=qxy:1.2:1.3' in txt2)
    parsed = jio.parse_jpar(txt2)
    check(".jpar parses the global column limits",
          parsed['settings'].get('column_limits') == 'qz:0.1:0.2',
          parsed['settings'].get('column_limits'))
    check(".jpar parses the per-dataset column limits",
          parsed['blocks'][0].get('collimits') == 'qxy:1.2:1.3',
          parsed['blocks'][0].get('collimits'))
    old_jpar = '\n'.join(l for l in txt2.splitlines()
                         if not l.startswith('#Column Limits=')
                         and not l.startswith('#ColumnLimits='))
    parsed_old = jio.parse_jpar(old_jpar)
    check("an old .jpar without the lines still parses",
          parsed_old['blocks'][0].get('collimits', '') == ''
          and not parsed_old['settings'].get('column_limits'))

    # ------------------------------------------------- the dialog ----------
    w.limits[k0]['qz'] = (0.05, 0.12)
    w.openColumnLimitsDialog()
    dlg = w.columnLimitsDialog
    check("dialog lists every column as a row", dlg.table.rowCount() == len(cols),
          '%d rows' % dlg.table.rowCount())
    r_qz = [r for r in range(dlg.table.rowCount())
            if dlg.table.item(r, 0).text() == 'qz'][0]
    check("dialog shows the stored value",
          dlg.table.item(r_qz, 1).text() == '0.05', dlg.table.item(r_qz, 1).text())
    dlg.table.item(r_qz, 2).setText('0.30')       # edit -> host store
    check("editing the dialog writes through to limits",
          w.limits[k0]['qz'] == (0.05, 0.30), w.limits[k0].get('qz'))
    dlg.table.item(r_qz, 2).setText('')           # blank -> unbounded
    check("blanking a cell means unbounded",
          w.limits[k0]['qz'] == (0.05, None), w.limits[k0].get('qz'))
    dlg.table.item(r_qz, 1).setText('not a number')
    check("a non-numeric entry reverts rather than raising",
          w.limits[k0].get('qz') is None or w.limits[k0]['qz'][0] is None,
          w.limits[k0].get('qz'))
    dlg._clear_all()
    check("Clear all keeps the x window and drops the rest",
          'x' in w.limits[k0] and len(w.limits[k0]) == 1, w.limits[k0])

    # two datasets selected -> <varies>
    w.limits[keys[0]]['qz'] = (0.05, 0.12)
    w.limits[keys[1]]['qz'] = (0.20, 0.30)
    for i in range(w.dataListWidget.count()):
        w.dataListWidget.item(i).setSelected(True)
    w.dataFileSelectionChanged()
    dlg.refresh()
    r_qz = [r for r in range(dlg.table.rowCount())
            if dlg.table.item(r, 0).text() == 'qz'][0]
    check("<varies> shown when the selection disagrees",
          dlg.table.item(r_qz, 1).text() == '<varies>', dlg.table.item(r_qz, 1).text())
    w.set_column_limit('qz', 0, 0.07)
    check("an edit writes to EVERY selected dataset",
          all((w.limits[k].get('qz') or (None,))[0] == 0.07 for k in keys),
          [w.limits[k].get('qz') for k in keys])

    # ------------------------------------------------- removeData cleanup --
    w.dataListWidget.clearSelection()
    w.dataListWidget.item(1).setSelected(True)
    w.removeData()
    check("removeData drops the limits entry", keys[1] not in w.limits,
          list(w.limits.keys()))

    print()
    print("ALL PASS" if _OK["ok"] else "FAILURES PRESENT")
    return 0 if _OK["ok"] else 1


if __name__ == '__main__':
    sys.exit(main())
