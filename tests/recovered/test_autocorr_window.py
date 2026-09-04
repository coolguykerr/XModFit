"""Show the periodic autocorr cost is now bounded (windowed) and constant vs chain length."""
import os, sys, time, tempfile
REPO = r"C:\Users\kerr\Documents\GitHub\XModJointFit"
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np
from lmfit.customhdf import CustomHDFBackend

nw, nd = 20, 9
ang = np.zeros(nd)
rng = np.random.RandomState(0)


def make_backend(N):
    bp = tempfile.mktemp(suffix='.h5')
    be = CustomHDFBackend(bp)
    be.reset(nw, nd)
    be.grow(N, None)
    with be.open('a') as f:
        g = f[be.name]
        g['chain'][:] = rng.randn(N, nw, nd)
        g.attrs['iteration'] = N
    return be


def timeit(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


print("chain_len |  full-chain autocorr  |  windowed (last 5000)")
for N in [5000, 20000, 50000, 100000]:
    be = make_backend(N)
    tau_full = None
    def full():
        global tau_full
        tau_full = be.get_autocorr_time(angleindex=ang, discard=0, tol=0)
    def win():
        be.get_autocorr_time(angleindex=ang, discard=max(0, N - 5000), tol=0)
    tf = timeit(full)
    tw = timeit(win)
    print("%9d | %8.3f s (mean tau %.1f) | %8.3f s   (%.1fx cheaper)"
          % (N, tf, float(np.mean(tau_full)), tw, tf / tw))
    os.remove(be.filename)
