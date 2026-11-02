"""Stage 0: homometric enumeration of a protein-free scan into membrane seeds.

This is the only genuinely **enumerative** stage of the fast global search:
Patterson autocorrelation -> turnpike interface positions -> sign partners -> box
model basin scan -> spectral zero-flip family. Deterministic, pure numpy, no PDB and
no Julia. About a minute per film, then cached.

Why it is worth the minute: seeding the membrane block from this beat unseeded Sobol
starts by **7-16x** in hit rate per dataset. The measured mechanism is dimension
reduction rather than seed accuracy -- candidates 0.150 and 0.520 away from the
optimum seed equally well (paired test, p = 0.618) -- so the job here is to produce
a handful of PLAUSIBLE membranes, not an accurate one.

The pipeline itself is vendored unmodified in :mod:`fastscan._xrr_vendor`; this
module is the XModFit-facing adaptation and owns three things the vendored code does
not: array input (so a scan already loaded in the GUI needs no round trip through a
file), caching per input file, and **the refusal to enumerate protein-loaded data**.

That last one is not a nicety. Running this on a scan that has protein on it stretches
the fitted span from ~22 to 33-37 Angstroms, drives chi-squared/N to the hundreds or
thousands, and rails the q-offset -- and it does all that while still returning
plausible-looking candidates. It has to be refused loudly and overridden explicitly.
"""
import os

import numpy as np

from . import seeds as S

# Physical span of a lipid/protein-free film, Angstroms. Above this the scan almost
# certainly has something else on it -- measured: protein-loaded TIM1 scans fitted
# spans of 33-37 where the lipid-only scans sat near 22.
SPAN_PROTEIN_FREE_MAX = 30.0
# chi-squared/N above which the box model plainly does not describe the data. This is
# the check that catches what the span screen cannot -- see protein_loaded_warning.
CHI2_PER_N_SUSPECT = 50.0
# Median reflectivity over the upper half of the q range, below which the curve must
# be raw R rather than R/R_F. Measured separation is six orders of magnitude, so the
# exact value does not matter.
RAW_R_HIGH_Q_MAX = 1e-3

_cache = {}


class HomometryError(RuntimeError):
    """The enumeration could not run, or refused to."""


def _vendor():
    from . import _xrr_vendor as X
    return X


# ------------------------------------------------------------------ patterson ---

def patterson(q, r, e=None):
    """Patterson analysis from ARRAYS. The array form of the vendored
    ``resolve_patterson``, which only takes a file path.

    Returns the same dict: ``q r e z G Gerr zpk hpk span res dz``.
    """
    from scipy.signal import find_peaks

    X = _vendor()
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    e = np.ones_like(q) if e is None else np.asarray(e, dtype=float)
    order = np.argsort(q)
    q, r, e = q[order], r[order], e[order]
    good = np.isfinite(q) & np.isfinite(r)
    q, r, e = q[good], r[good], e[good]
    if len(q) < 8:
        raise HomometryError('need at least 8 finite points to enumerate; got %d'
                             % len(q))

    qu, ru, eu, dq = X.precondition(q, r, e)
    z, G, dzp, taper = X.patterson(qu, ru, dq, beta=6.0)
    Gerr = X.patterson_error(qu, eu, dq, taper, len(z), dzp)
    pos = z > 5.0
    Gabs = np.abs(G)
    pk, _ = find_peaks(Gabs[pos], height=0.05 * Gabs[pos].max())
    zpk = z[pos][pk]
    hpk = G[pos][pk]
    span = float(zpk.max()) if len(zpk) else 30.0
    return dict(q=q, r=r, e=np.maximum(e, 1e-6), z=z, G=G, Gerr=Gerr, zpk=zpk,
                hpk=hpk, span=span, res=2 * np.pi / qu.max(),
                dz=np.pi / qu.max())


def read_scan(path):
    """``(q, r, e)`` from a reflectivity file, via the vendored reader."""
    return _vendor().load_rrf(path)


def convention_warning(q, r, floor=RAW_R_HIGH_Q_MAX):
    """A reason to believe this is RAW R rather than R/R_F, or None.

    Checked first, and separately from the span test, because the two have the same
    symptom and completely different cures. Comparing a raw-R curve against an
    R/R_F model zeroes the profiled scale and explodes chi-squared, and it also
    destroys the Patterson: the Fresnel q^-4 decay dominates the transform and
    manufactures peaks at hundreds of Angstroms.

    The discriminator is enormous and needs no tuning. Measured on the fixtures in
    this tree: R/R_F curves have a median reflectivity of ~0.15 over the upper half
    of their q range, a raw-R curve ~1.5e-07. Six orders of magnitude.
    """
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    good = np.isfinite(q) & np.isfinite(r)
    q, r = q[good], r[good]
    if len(q) < 4:
        return None
    hi = q >= (q.min() + 0.5 * (q.max() - q.min()))
    if not hi.any():
        return None
    med = float(np.median(r[hi]))
    if med < floor:
        return ('The median reflectivity over the upper half of the q range is '
                '%.2e, which means this is RAW R, not R/R_F. Fresnel-normalise it '
                'first.\n\nEnumerating raw R does not merely fit badly: the q^-4 '
                'decay dominates the Patterson transform and manufactures peaks at '
                'hundreds of Angstroms, so the candidate membranes would be '
                'meaningless. Recover R_F from a matched raw/normalised pair rather '
                'than assuming a critical edge -- series differ in q_c by around '
                '0.7%%, and using one series'' factor on another visibly corrupts '
                'the reference.' % med)
    return None


def protein_loaded_warning(pat, span_max=SPAN_PROTEIN_FREE_MAX):
    """A reason to believe this scan is NOT protein-free, or None.

    Cheap -- it runs off the Patterson span alone, before the expensive basin scan --
    so the user finds out in a second rather than a minute.

    **This is a screen, not a guarantee, and the limit is resolution.** The span it
    can see is capped by pi/q_max, so on a short q range a protein-loaded scan can
    present a perfectly innocent span: measured in this tree, a protein-loaded TIM3
    scan out to q = 0.55 resolves a single Patterson peak at 23.8 A and passes this
    test. The complementary check is the post-fit chi-squared/N
    (``CHI2_PER_N_SUSPECT``), which caught exactly that case at 380. Both are
    reported; neither alone is sufficient.
    """
    span = float(pat['span'])
    if span > span_max:
        return ('The outermost resolved Patterson peak is at %.1f A, beyond the '
                '%.0f A expected of a protein-free film. This scan looks like it '
                'has protein on it. Enumerating protein-loaded data returns '
                'plausible-looking candidates that are wrong: the span stretches, '
                'chi-squared/N runs into the hundreds, and the q-offset rails.'
                % (span, span_max))
    return None


# ---------------------------------------------------------------- enumeration ---

def enumerate_scan(source, nbox, rho_top=0.0, rho_bottom=0.334, contrast=None,
                   allow_protein=False, use_cache=True, verbose=False):
    """Run stage 0 on one scan. Returns the vendored ``run_pipeline``-shaped dict
    with an added ``warnings`` list.

    ``source`` is a file path, or a ``(q, r, e)`` tuple. ``contrast`` defaults to
    ``rho_bottom - rho_top``, which is what the vendored pipeline's own
    configurations use.
    """
    X = _vendor()
    if contrast is None:
        contrast = float(rho_bottom) - float(rho_top)

    if isinstance(source, (str, bytes, os.PathLike)):
        path = os.path.abspath(str(source))
        if not os.path.isfile(path):
            raise HomometryError('no such scan: %s' % path)
        key = (path, os.path.getmtime(path), int(nbox), float(rho_top),
               float(rho_bottom), float(contrast), bool(allow_protein))
        if use_cache and key in _cache:
            return _cache[key]
        q, r, e = read_scan(path)
        label = os.path.basename(path)
    else:
        q, r, e = source
        key = None
        label = 'arrays'

    warnings = []
    # Convention FIRST: it is the cheapest, the most specific, and it has the same
    # symptom as a protein-loaded scan, so diagnosing it second would report the
    # wrong cause. It is also not overridable -- there is no sense in which
    # enumerating raw R is what the user meant.
    bad_convention = convention_warning(q, r)
    if bad_convention:
        raise HomometryError(bad_convention)

    pat = patterson(q, r, e)
    why = protein_loaded_warning(pat)
    if why:
        if not allow_protein:
            raise HomometryError(why + '\n\nPass allow_protein=True to override.')
        warnings.append('OVERRIDDEN: ' + why)

    L = float(pat['span'])
    dz = float(pat['dz'])
    rt, rb = float(rho_top), float(rho_bottom)

    # The body of the vendored `run_pipeline`, but driven from an already-computed
    # Patterson dict so arrays work and the refusal above can happen first.
    basin0 = X.scan_basin(pat['q'], pat['r'], pat['e'], nbox, L, dz, rt, rb,
                          kind='kin')
    basin = []
    for _chi, th0, _ys, _qo in basin0[:8]:
        try:
            thp, ssrp, ys, qoff = X.refine_spanbounded(
                pat['q'], pat['r'], pat['e'], th0, nbox, L, rt, rb, kind='parratt',
                nfev=250, span_lo=X.SPAN_LO, span_hi=X.SPAN_HI, rho_hi=X.RHO_HI)
        except Exception:                      # noqa: BLE001 -- a failed refine is
            continue                           # just a candidate we do not get
        if not np.isfinite(ssrp):
            continue
        basin.append((ssrp / len(pat['q']), thp, ys, qoff))
    basin.sort(key=lambda t: t[0])
    # drop genuine runaways, then junk far above the best fit -- the vendored
    # pipeline's own two filters
    basin = [c for c in basin if not X.on_boundary(c[1], nbox, L, rt, rb)[0]]
    if basin:
        cmin = min(c[0] for c in basin)
        basin = [c for c in basin if c[0] < 3.0 * cmin]
    if not basin:
        raise HomometryError(
            'the basin scan produced no usable candidate for %s. Either the box '
            'count (%d) does not suit this film, or the data are not a clean '
            'protein-free reflectivity curve.' % (label, nbox))

    out = dict(name=label, N=nbox, rt=rt, rb=rb, pat=pat, L=L, dz=dz,
               q=pat['q'], r=pat['r'], e=pat['e'], basin=[], warnings=warnings)
    for chi, th, ys, qoff in basin:
        zpos, rbx, sg, span = X._unpack_sb(th, nbox)
        try:
            mem, g, sgg, m = X.homometric_family(th, nbox, dz, rt, rb)
        except ValueError:                     # zero net contrast -> family undefined
            mem, g, sgg, m = [], None, sg, 0
        phys = X.physical_members(mem, rt, dz) if len(mem) else []
        out['basin'].append(dict(chi=chi, th=th, zpos=zpos, rbx=rbx, sigma=sg,
                                 span=span, yscale=ys, qoff=qoff, family=mem,
                                 n_family=len(mem), n_physical=len(phys), g=g,
                                 m=m, dz=dz))
        if verbose:
            print('  candidate chi2/N=%.1f  z=%s rho=%s sig=%.2f  family %d (%d '
                  'physical)' % (chi, np.round(zpos[1:-1], 1), np.round(rbx, 2),
                                 sg, len(mem), len(phys)))

    best = out['basin'][0]['chi']
    if best > CHI2_PER_N_SUSPECT:
        out['warnings'].append(
            'the best box-model fit to this scan is chi-squared/N = %.0f, which is '
            'far too high for a clean protein-free curve. The candidates below may '
            'be meaningless -- check the data convention (raw R against an R/R_F '
            'model zeroes the profiled scale and explodes chi-squared).' % best)

    if key is not None and use_cache:
        _cache[key] = out
    return out


def candidates(source, nbox, rho_top=0.0, rho_bottom=0.334, contrast=None,
               allow_protein=False, include_family=False, verbose=False):
    """Membrane seeds from one scan: a list of ``[sig, d..., rho...]`` vectors.

    This is the entry point step 1 route (a) calls. Screening against the search box
    is the caller's job (``seeds.screen``), because only the caller knows the box.

    ``include_family`` additionally emits the physically admissible members of each
    homometric family. Off by default: the family members live on the Patterson grid
    rather than as box models, so converting them is lossy, and the measured gain
    from seeding is dimension reduction rather than coverage of the family.
    """
    out = enumerate_scan(source, nbox, rho_top=rho_top, rho_bottom=rho_bottom,
                         contrast=contrast, allow_protein=allow_protein,
                         verbose=verbose)
    seeds = []
    for cand in out['basin']:
        try:
            seeds.append(S.from_basin(cand))
        except S.SeedError:
            continue
    if include_family:
        seeds.extend(_family_seeds(out))
    return seeds


def _family_seeds(out):
    """Physically admissible homometric partners, as box seeds where that is exact.

    A family member is a step profile on the Patterson grid. It maps back to a box
    model exactly only when its non-zero steps are as many as the box count; anything
    else would need a fit, which belongs in stage 1 rather than here.
    """
    extra = []
    nbox = out['N']
    X = _vendor()
    for cand in out['basin']:
        mem = cand.get('family') or []
        if not len(mem):
            continue
        for g in X.physical_members(mem, out['rt'], cand['dz']):
            nz = np.flatnonzero(np.abs(g) > 1e-3 * max(np.max(np.abs(g)), 1e-12))
            if len(nz) != nbox + 1:
                continue
            z = nz * cand['dz']
            rho = out['rt'] + np.cumsum(g)[nz[:-1]]
            d = np.diff(z)
            if len(d) != nbox or len(rho) != nbox:
                continue
            extra.append(np.concatenate([[cand['sigma']], d, rho]))
    return extra


def summary(out):
    """A few lines a UI can show under the candidate table."""
    L = ['%s: span %.1f A, resolution %.1f A, Nyquist dz %.2f A, %d candidate(s)'
         % (out['name'], out['L'], out['pat']['res'], out['dz'],
            len(out['basin']))]
    L.append('Patterson peaks at z = %s'
             % np.array2string(np.round(out['pat']['zpk'], 1)))
    for i, c in enumerate(out['basin']):
        L.append('  %d: chi2/N %.1f  sig %.2f  d %s  rho %s  family %d (%d physical)'
                 % (i + 1, c['chi'], c['sigma'],
                    np.array2string(np.diff(c['zpos']), precision=2),
                    np.array2string(c['rbx'], precision=3),
                    c['n_family'], c['n_physical']))
    L.extend(out.get('warnings') or [])
    return '\n'.join(L)
