"""
Validate the parallel Phase-1 grid scan:
  (1) parallel result == serial result (same worker code, deterministic)
  (2) the full 544-cell scan surfaces BOTH known minima
  (3) wall-clock speedup vs serial estimate

Guarded for Windows spawn (Pool re-imports this module).
"""
import os
import sys
import time
import numpy as np

REPO = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from importlib import import_module
XLayersPDB = import_module("Functions.XRR.XLayersPDB").XLayersPDB
import pdb_seed_worker as psw


def build_func():
    RRF = r"C:\Users\kerr\Documents\XLayersProtTest\mTIM3 analysis\i2_tim3_1000.rrf"
    raw = np.loadtxt(RRF)
    xall, yall, eall = raw[:, 0], raw[:, 1], raw[:, 2]
    m = (xall >= 0.026) & (xall <= 0.550)
    xd, yd, ed = xall[m], yall[m], eall[m]
    mpar = {'Model': {'Layers': ['top', 'tail', 'head', 'bottom'],
                      'd':   [0.0, 11.3182806, 8.75928561, 1.0],
                      'rho': [0.0, 0.23396671, 0.46419943, 0.334],
                      'mu':  [0.0, 0.0, 0.0, 0.0],
                      'sig': [3.4, 3.4, 3.4, 3.4]}}
    func = XLayersPDB(x=xd, E=10.0, fname='./Data/tim3_hmmm_ext_all_bestframe.pdb',
                      mpar=mpar, dz=0.5, fslice=0.5, rrf=True, fix_sig=True,
                      qoff=9.830348e-05, yscale=1.0, bkg=0.0,
                      theta=40.88, phi=74.08, protins=6.094141, cov=0.1582754)
    func.x = xd
    func.params['__Model_d_001'].value = 11.3182806
    func.params['__Model_d_002'].value = 8.75928561
    func.params['__Model_rho_001'].value = 0.23396671
    func.params['__Model_rho_002'].value = 0.46419943
    return func, xd, yd, ed


NUIS = ['qoff', 'protins', 'cov', '__Model_d_001', '__Model_rho_001',
        '__Model_d_002', '__Model_rho_002']
START = np.array([9.830348e-05, 6.094141, 0.1582754, 11.3182806, 0.23396671,
                  8.75928561, 0.46419943])


def build_cells(thetares=10):
    thetameshrange = np.arange(0, 180 + thetares, thetares)
    cells = []
    for ti, tm in enumerate(thetameshrange):
        if tm == 0 or tm == 180:
            phires = 360
        else:
            phires = int(360 / np.round((np.cos((tm - thetares/2)*np.pi/180) -
                                         np.cos((tm + thetares/2)*np.pi/180)) /
                                        (1 - np.cos((thetares/2)*np.pi/180))))
        for pi, pm in enumerate(np.arange(0, 360, phires)):
            cells.append((ti, pi, float(tm), float(pm)))
    return cells


def main():
    ncpu = os.cpu_count() or 2
    workers = max(2, ncpu - 2)
    func, xd, yd, ed = build_func()
    cells = build_cells(10)
    print("cells:", len(cells), " workers:", workers, " cpu:", ncpu)

    # (1+3) full parallel scan
    t0 = time.perf_counter()
    par = psw.map_cell_costs(cells, func, xd, yd, ed, 'Linear', NUIS, START, workers)
    t_par = time.perf_counter() - t0
    par_map = {(ti, pi): c for (ti, pi, c) in par}
    print("PARALLEL full scan: %.1f s" % t_par)

    # (1) serial on a spread-out subset, compare to parallel
    subset = cells[::13]  # ~42 cells
    t0 = time.perf_counter()
    ser = psw.map_cell_costs(subset, func, xd, yd, ed, 'Linear', NUIS, START, 1)
    t_ser = time.perf_counter() - t0
    maxdiff = 0.0
    for (ti, pi, c) in ser:
        maxdiff = max(maxdiff, abs(c - par_map[(ti, pi)]))
    print("SERIAL subset (%d cells): %.1f s  -> per-cell %.3f s" %
          (len(subset), t_ser, t_ser / len(subset)))
    print("max |parallel - serial| over subset: %.3e" % maxdiff)
    print("serial-estimate full scan: %.1f s -> speedup ~%.1fx" %
          (len(cells) * t_ser / len(subset), (len(cells) * t_ser / len(subset)) / t_par))

    # (2) two lowest minima
    allc = sorted(((c, ti, pi) for (ti, pi, c) in par), key=lambda z: z[0])
    tr = 10
    tmr = np.arange(0, 180 + tr, tr)
    print("\nlowest grid cells (cost, chi^2=2*cost, theta, phi):")
    seen = []
    for c, ti, pi in allc:
        tm = tmr[ti]
        if tm == 0 or tm == 180:
            phires = 360
        else:
            phires = int(360 / np.round((np.cos((tm - tr/2)*np.pi/180) -
                                         np.cos((tm + tr/2)*np.pi/180)) /
                                        (1 - np.cos((tr/2)*np.pi/180))))
        pm = (np.arange(0, 360, phires))[pi]
        # collect distinct basins (>20 deg apart)
        if all(abs(tm - s[0]) + min(abs(pm - s[1]), 360 - abs(pm - s[1])) > 25 for s in seen):
            seen.append((tm, pm))
            print("  cost=%.3f  chi2=%.3f  theta=%.0f  phi=%.0f" % (c, 2*c, tm, pm))
        if len(seen) >= 4:
            break
    print("\nknown minima: primary chi2=148.9 @ theta~41 phi~74 ; secondary chi2=157.2 @ theta~31 phi~149")


if __name__ == '__main__':
    main()
