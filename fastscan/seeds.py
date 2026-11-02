"""Membrane seed candidates for the fast global search: screening, pooling, file IO.

A "seed" here is the membrane block of the model in PHYSICAL units, ordered exactly
as the search needs it::

    [sig, d_1 .. d_n, rho_1 .. rho_n]          (1 + 2n floats, n = num_boxes)

i.e. the interior boxes only -- the bounding media are load-time constants and are
not fitted. That is the same layout the seeding study used as
``CONFIG['membrane_seeds']``, so a seed table produced by the offline harness drops
straight in.

Why seeds matter at all: seeding the membrane block from a homometric enumeration of
a protein-free scan (arm B) beat unseeded Sobol (arm A) by 7-16x in hit rate per
dataset. The measured mechanism is DIMENSION REDUCTION, not seed accuracy --
candidates 0.150 and 0.520 away from the optimum seed equally well (paired test,
p = 0.618). So there is no point being clever here; what matters is that the
candidates are plausible and that bad ones are DROPPED rather than clipped.

The screen is a reimplementation of the study's ``julia/bakeoff/seed_utils.py``
rather than a copy of it: that file is hardwired to the study's config format, and
the recommendation on record is to reimplement it here.
"""
import io
import json
import os

import numpy as np

# Physicality window for a lipid/protein film, el/Angs^3. Water is 0.334 and protein
# ~0.44, so 0.62 is generous without admitting nonsense.
#
# ONE definition, used by three callers that must agree or the UI contradicts
# itself: this screen, the scan box's rho bounds, and the per-mode physicality flag.
# Note the homometric enumerator uses 0.60 for its own internal filtering
# (`physical_members(rho_hi=0.60)`); that is its own basin-scan concern and is
# deliberately left alone -- this constant governs what the SEARCH will accept.
RHO_PHYS_MAX = 0.62
RHO_PHYS_FLOOR = 0.0      # strictly non-negative: a true zero is fine, -0.017 is not


class SeedError(ValueError):
    """A seed table could not be read or does not match the model."""


def split(seed, num_boxes=None):
    """``[sig, d..., rho...]`` -> ``(sig, d, rho)`` as floats/arrays."""
    s = np.asarray(seed, dtype=float).ravel()
    if num_boxes is None:
        if len(s) < 3 or (len(s) - 1) % 2:
            raise SeedError('a seed must be [sig, d_1..d_n, rho_1..rho_n]; '
                            'got length %d, which is not 1+2n' % len(s))
        num_boxes = (len(s) - 1) // 2
    want = 1 + 2 * num_boxes
    if len(s) != want:
        raise SeedError('seed has length %d but the model has %d boxes '
                        '(expected %d)' % (len(s), num_boxes, want))
    return float(s[0]), s[1:1 + num_boxes], s[1 + num_boxes:]


def screen(seeds, lo, hi, rho_top=0.0, rho_max=RHO_PHYS_MAX, num_boxes=None):
    """Filter seeds to the physically sensible ones. Returns ``(kept, rejected)``.

    ``lo``/``hi`` are the matching physical bounds the optimiser will actually use,
    in the same ``[sig, d..., rho...]`` layout. ``rejected`` is a list of
    ``(index, seed, reason)`` so the caller can show WHY each was dropped -- which is
    the whole point of screening as a visible step.

    **Never clips.** The failure this guards against is quiet: clipping ``rho =
    -0.02`` to exactly ``0.0`` produces a vacuum layer, and every start assigned to
    that seed is then wasted rather than rejected. Drop and report.

    If everything is rejected the unscreened list is returned (with the rejections
    still reported), because a scan seeded by an implausible membrane is still better
    than no scan -- but the caller must surface that it happened.
    """
    seeds = [np.asarray(s, dtype=float).ravel() for s in seeds]
    if not seeds:
        return [], []
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    kept, rejected = [], []
    for i, s in enumerate(seeds):
        try:
            sig, d, rho = split(s, num_boxes)
        except SeedError as ex:
            rejected.append((i, s, str(ex)))
            continue
        why = None
        if np.any(rho < rho_top + RHO_PHYS_FLOOR):
            why = ('electron density below the top medium: rho=%s'
                   % np.array2string(rho, precision=3))
        elif np.any(rho > rho_max):
            why = ('electron density above %.2f: rho=%s'
                   % (rho_max, np.array2string(rho, precision=3)))
        elif np.any(d <= 0):
            why = ('non-positive thickness: d=%s'
                   % np.array2string(d, precision=2))
        elif not np.isfinite(s).all():
            why = 'non-finite entry'
        elif len(lo) == len(s) and (np.any(s < lo - 1e-9) or np.any(s > hi + 1e-9)):
            out = [k for k in range(len(s))
                   if s[k] < lo[k] - 1e-9 or s[k] > hi[k] + 1e-9]
            why = ('outside the search box at %s'
                   % ', '.join(_label(k, num_boxes or (len(s) - 1) // 2)
                               for k in out))
        if why is None:
            kept.append(s)
        else:
            rejected.append((i, s, why))
    return (kept if kept else seeds), rejected


def _label(k, num_boxes):
    if k == 0:
        return 'sig'
    if k <= num_boxes:
        return 'd%d' % k
    return 'rho%d' % (k - num_boxes)


def pool(groups, tol=1e-6):
    """Merge seed lists from several scans, dropping exact/near duplicates.

    ``groups`` is an iterable of ``(label, [seed, ...])``. Returns
    ``(seeds, labels)`` where ``labels[i]`` names the scan seed ``i`` came from, so
    the UI can show provenance per candidate.

    Duplicates are merged on Euclidean distance in the seed vector, which is
    adequate: this is only deciding how many distinct STARTS to spend, not making a
    scientific claim about distinctness. First occurrence wins, so the order of
    ``groups`` sets precedence.
    """
    seeds, labels = [], []
    for label, group in groups:
        for s in group:
            s = np.asarray(s, dtype=float).ravel()
            if any(len(s) == len(t) and np.linalg.norm(s - t) <= tol for t in seeds):
                continue
            seeds.append(s)
            labels.append(label)
    return seeds, labels


def from_basin(cand):
    """One homometric basin candidate -> a ``[sig, d..., rho...]`` seed.

    ``cand`` is an entry of ``out['basin']`` from the enumerator, carrying
    ``sigma`` (roughness), ``zpos`` (interface positions) and ``rbx`` (box
    densities). The interior thicknesses are the successive differences of the
    interface positions.
    """
    sig = float(cand['sigma'])
    d = np.diff(np.asarray(cand['zpos'], dtype=float).ravel())
    rho = np.asarray(cand['rbx'], dtype=float).ravel()
    if len(d) != len(rho):
        raise SeedError('basin candidate has %d thicknesses but %d densities; '
                        'zpos and rbx disagree' % (len(d), len(rho)))
    return np.concatenate([[sig], d, rho])


def to_unit_cube(seeds, lo, hi):
    """Seeds -> the unit cube, for writing into the membrane columns of a start.

    Clips to [0,1] only to absorb floating-point overshoot at a bound; a seed that
    is genuinely outside the box should have been dropped by :func:`screen` first,
    which is why this is not the place that decides.
    """
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    span = np.where(hi > lo, hi - lo, 1.0)
    return [np.clip((np.asarray(s, dtype=float).ravel() - lo) / span, 0.0, 1.0)
            for s in seeds]


# ------------------------------------------------------------------- file IO ---

_MAGIC = '#XModFit Membrane Seeds v1'


def read(path, num_boxes=None):
    """Read a seed table. Returns ``(seeds, labels, meta)``.

    Three accepted shapes, so a table from any stage of this workflow just opens:

    * our own ``.seeds`` text format (magic line above);
    * a JSON object with a ``membrane_seeds`` key -- the seeding study's own job
      config, so its output is directly reusable;
    * a plain whitespace/comma table, one seed per row.
    """
    if not os.path.isfile(path):
        raise SeedError('no such seed file: %s' % path)
    with io.open(path, 'r', encoding='utf-8', errors='replace') as fh:
        text = fh.read()

    stripped = text.lstrip()
    if stripped.startswith('{'):
        try:
            obj = json.loads(text)
        except ValueError as ex:
            raise SeedError('%s looks like JSON but will not parse: %s'
                            % (os.path.basename(path), ex))
        if 'membrane_seeds' not in obj:
            raise SeedError('%s is JSON but has no "membrane_seeds" key'
                            % os.path.basename(path))
        seeds = [np.asarray(s, dtype=float).ravel() for s in obj['membrane_seeds']]
        meta = {k: obj[k] for k in ('tag', 'nbox', 'rho_top', 'rho_bottom', 'data')
                if k in obj}
        meta['source'] = os.path.abspath(path)
        return seeds, ['%s' % meta.get('tag', os.path.basename(path))] * len(seeds), meta

    seeds, labels, meta = [], [], {'source': os.path.abspath(path)}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            if '=' in line:
                k, _, v = line[1:].partition('=')
                meta[k.strip()] = v.strip()
            continue
        parts = [p for p in line.replace(',', ' ').split() if p]
        label = ''
        # trailing non-numeric token is a provenance label
        try:
            float(parts[-1])
        except ValueError:
            label = parts[-1]
            parts = parts[:-1]
        try:
            seeds.append(np.asarray([float(p) for p in parts], dtype=float))
        except ValueError:
            raise SeedError('cannot parse seed row in %s: %r'
                            % (os.path.basename(path), raw[:80]))
        labels.append(label or os.path.basename(path))
    if not seeds:
        raise SeedError('%s contained no seed rows' % os.path.basename(path))
    if num_boxes is not None:
        for s in seeds:
            split(s, num_boxes)          # raises with a useful message on mismatch
    return seeds, labels, meta


def write(path, seeds, labels=None, meta=None):
    """Write a ``.seeds`` table, so an enumeration is done once and reused."""
    labels = labels or [''] * len(seeds)
    meta = meta or {}
    n = None
    if seeds:
        n = (len(np.asarray(seeds[0]).ravel()) - 1) // 2
    L = [_MAGIC]
    for k in sorted(meta):
        L.append('#%s=%s' % (k, meta[k]))
    if n:
        L.append('#num_boxes=%d' % n)
        L.append('#columns=' + ' '.join(['sig']
                                        + ['d%d' % (i + 1) for i in range(n)]
                                        + ['rho%d' % (i + 1) for i in range(n)])
                 + ' source')
    for s, lab in zip(seeds, labels):
        row = '\t'.join('%.8g' % v for v in np.asarray(s, dtype=float).ravel())
        L.append(row + (('\t' + lab) if lab else ''))
    with io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(L) + '\n')
    return path
