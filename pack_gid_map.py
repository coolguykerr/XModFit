#!/usr/bin/env python3
"""
pack_gid_map.py -- turn a raw 2D GID map into the file GID2D can fit.

    python pack_gid_map.py <raw file> [output file]

Why this step exists at all
--------------------------
XModFit needs the x column to be MONOTONE (it slices with np.where on x), all
curves of a dataset to be equal length, and one x column per dataset. Raw q_xy
from a 2D map is a sawtooth -- it repeats once per q_z row -- so it cannot be x.
A row index can, which is what this writes.

It matters which file you load as the dataset, not just as ``mapfile``: the fit
compares the model against the DATASET's y column. Load the raw file and
Data_Dialog picks q_xy as x and q_z as y, so the map draws from ~2 pixels and the
fit is regressing the model against q_z. GID2D says so on the 2D panel, but the
cure is this script.

What it drops, and why
----------------------
  * sigma <= 0 or I <= 0   -- the detector mask. It CANNOT be handled later:
    Fit_Routines.residual computes (data-fit)/err outside the model, so a
    sigma = 0 row is 0/0 whatever the model returns. In the DPPC file these 2595
    pixels are scattered over 335 distinct q_z values, so no contiguous
    xmin:xmax excludes them either.
  * sigma below 1e-6 x median(sigma) -- the same mask with floating-point
    residue instead of a clean zero. Measured on the DPPC file: 26 such pixels,
    sigma down to 1.9e-16, and because 1/sigma reaches 5e15 they carried ~100%
    of chi^2 between them. The sorted-sigma gap at the cut is a factor 1.6e11,
    so the threshold is not a judgement call.
  * q_z < 0 -- below the horizon, where the model is identically zero.

Fit it in Log scale: sigma is multiplicative here (sigma/I median ~0.47), so in
Linear scale one near-zero pixel can carry most of the weight.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Functions'))
import gid2d_core as GC          # noqa: E402


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    src = argv[1]
    if not os.path.exists(src):
        print('no such file: %s' % src)
        return 1
    if len(argv) > 2:
        dst = argv[2]
    else:
        base = os.path.splitext(os.path.basename(src))[0]
        here = os.path.dirname(os.path.abspath(__file__))
        dst = os.path.join(here, 'Data', base + '_indexed.txt')
    d = os.path.dirname(os.path.abspath(dst))
    if not os.path.isdir(d):
        os.makedirs(d)
    info = GC.pack_map(src, dst, drop_masked=True, qz_min=0.0)
    print('wrote %s' % info['path'])
    print('  kept %d rows, dropped %d' % (info['kept'], info['dropped']))
    print('  %s' % info['sigma_note'])
    mp = GC.load_map(dst)
    print('  grid %d q_xy x %d q_z = %d cells, %d filled (%.1f%%)'
          % (mp['n_qxy'], mp['n_qz'], mp['n_qxy'] * mp['n_qz'], mp['n'],
             100.0 * mp['n'] / (mp['n_qxy'] * mp['n_qz'])))
    print('')
    print('Now in XModFit:')
    print('  1. Add files -> %s        (NOT the raw file)' % os.path.basename(dst))
    print('  2. Functions -> GISAXS -> GID2D')
    print('  3. the mapfile button in Fixed Parameters -> the same file')
    print('  4. Fit Scale -> Log      (errors here are multiplicative)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
