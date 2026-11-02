"""Candidate solutions of the fast global search: the ``Mode`` record and ``ModeSet``.

Qt-free and Julia-free by design, so every rule in here is unit-testable headlessly.

THE CENTRAL DESIGN CONSTRAINT, and the reason this module looks the way it does:

    The two-stage search is a good OPTIMISER and a poor ENUMERATOR, and both halves
    of that are measured. It finds the deepest basin reliably (51/53 datasets
    out-of-sample on the best template). But against a 6x-larger converged
    reference it finds a median of 7 distinct certified minima where the reference
    finds 49, which collapses the admissible pose span from 90.4 to 23.7 degrees
    and silently converts 11.9% of cases from "the data admits two opposite senses"
    to "the pose is determined".

    So the length of a mode list is a property of the BUDGET, not of the data, and
    an ambiguity verdict computed from it fails in the direction of over-confidence
    -- which is worse than no verdict, because the whole point of the feature is to
    stop users trusting a single basin.

    A single hard-coded ambiguity threshold is also indefensible: the replicate
    scale it would encode varies TEN-FOLD between strata of one corpus (2.55 to
    25.90), and 7 of 9 template comparisons are indistinguishable from sampling
    noise. The number the original design plan used (11-20) has been retired.

Therefore, structurally:

  * there is NO ``is_ambiguous`` attribute and NO threshold constant in this module.
    The API is :meth:`ModeSet.n_distinct` and :meth:`ModeSet.pose_span` -- FUNCTIONS
    OF DELTA. Any caller that wants a boolean must pick a threshold explicitly, in
    its own code, where a reader can see it. The retired number is not available to
    be hard-coded because there is nowhere to put it.
  * a :class:`ModeSet` cannot be constructed without a :class:`Provenance`, and the
    provenance carries the method's measured power, so a set reopened in a year
    still reports what it could and could not have seen.
  * the list is named "modes", never "minima": it contains saddles too.
    :meth:`ModeSet.minima` is the certified subset, and that is what may be handed
    to a sampler.
"""
import hashlib
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Classification labels. MIN/SADDLE/DEGENERATE come from the eigenvalues of the
# Hessian restricted to the FREE coordinates; the '-BOUND' suffix is ours, see
# `classify`.
MIN, SADDLE, DEGENERATE, UNCERTIFIED = 'MIN', 'SADDLE', 'DEGENERATE', 'UNCERTIFIED'


@dataclass(frozen=True)
class Power:
    """Measured power of the METHOD. Constants of the published calibration.

    These are not properties of any one run -- they are what a 64-start screen was
    measured to achieve against a 384-start converged reference over 293 paired
    (dataset, template) cells. They live in the provenance so the humility statement
    travels with the data.
    """
    ref_budget: int = 384
    found_median: int = 7
    ref_median: int = 49
    false_determined_rate: float = 0.119
    span_found_deg: float = 23.7
    span_ref_deg: float = 90.4
    # plausible interval and observed extremes of the replicate/ambiguity SCALE.
    # Present so a UI can shade a band instead of drawing one line.
    scale_band: Tuple[float, float] = (4.7, 23.2)
    scale_extremes: Tuple[float, float] = (2.55, 25.90)
    # N1 and K have never been varied on any corpus (K never at all), and no
    # archive records a per-solve CPU time. Callers must not present them as tuned.
    budget_is_calibrated: bool = False

    @property
    def shortfall(self):
        """How many distinct minima the converged reference finds per one found here."""
        return float(self.ref_median) / max(self.found_median, 1)


@dataclass(frozen=True)
class DedupRule:
    """Identity of a mode. POSITION ONLY.

    ``-logL`` is deliberately absent. It is a CONSEQUENCE of position, not a
    coordinate, and including it in the identity rule splits one basin into several
    -- measured as a ~3x over-count. Depth is for ranking and for the depth cut,
    never for identity.
    """
    gc_deg: float = 10.0        # great-circle separation in (theta, phi)
    d_protins: float = 1.0      # Angstroms
    d_cov: float = 1.0          # raw cov units


@dataclass
class Provenance:
    """Everything needed to say whether two mode sets are comparable, and to re-run.

    Identity fields (the ones ``ModeSet.merge`` refuses to union across) are the mesh
    and data fields: absolute ``-logL`` is not comparable across ``spacez``,
    ``box_ref_length``, ``vol_calc``, box count, window or mask.
    """
    algorithm: str = 'twostage-v1'
    # budget actually spent
    n1: int = 0
    n1_ok: int = 0              # starts that returned a finite objective
    k: int = 0
    sobol_seed: int = 0
    band_stratified: bool = True
    iter1: int = 30
    tol1: float = 1e-3
    iter2: int = 300
    tol2: float = 1e-8
    n_cpu_limited: int = 0      # solves truncated by max_cpu_time -> ranking suspect
    # seeds
    seed_source: str = 'current-params'
    n_seeds: int = 1
    seeds_dropped: int = 0
    # model / mesh identity
    template_path: str = ''
    template_label: str = ''
    num_boxes: int = 0
    spacez: float = 0.5
    box_ref_length: float = 30.0
    vol_calc: str = 'approx'
    rho_top: float = 0.0
    rho_bottom: float = 0.334
    electron_scale: float = 1.0
    # data identity
    objective: str = 'xpr_logL'
    fit_scale: str = 'Linear'
    dataset: str = ''
    x_window: Tuple[float, float] = (float('nan'), float('nan'))
    n_active: int = 0
    data_sha256: str = ''
    # parameterisation
    varying_names: Tuple[str, ...] = ()
    free: Tuple[int, ...] = ()
    param_lower: Optional[np.ndarray] = None
    param_upper: Optional[np.ndarray] = None
    # certification
    hessian: str = 'none'       # exact | gauss-newton | finite-difference | none
    # timings, seconds
    timings: Dict[str, float] = field(default_factory=dict)
    # environment, for reproducibility of the Sobol sequence
    scipy_version: str = ''
    xpr_fingerprint: str = ''
    power: Power = field(default_factory=Power)

    IDENTITY = ('num_boxes', 'spacez', 'box_ref_length', 'vol_calc', 'rho_top',
                'rho_bottom', 'objective', 'fit_scale', 'dataset', 'n_active',
                'data_sha256', 'varying_names')

    def identity(self):
        return tuple(getattr(self, k) for k in self.IDENTITY)

    def incompatible_with(self, other):
        """A human-readable reason these two sets must not be merged, or None."""
        for k in self.IDENTITY:
            a, b = getattr(self, k), getattr(other, k)
            if a != b:
                return ('%s differs (%r vs %r): -logL is not comparable across it'
                        % (k, a, b))
        return None


# eq=False is load-bearing, not a style choice: the generated __eq__ would compare
# every field, several of which are numpy arrays, so `mode in some_list` raises
# "truth value of an array is ambiguous". A Mode is an object with an identity, not
# a value to be compared, so identity equality is also the correct semantics.
@dataclass(eq=False)
class Mode:
    """One candidate solution.

    ``u`` (the unit-cube point) is CANONICAL and ``params`` is derived. The Hessian,
    the dedup metric and the mass matrix are all in unit-cube coordinates, because
    the physical<->normalised map lives inside the AD tape; storing physical as
    canonical would make every consumer need the box to get back. It also makes the
    Laplace normalising Jacobian a constant that cancels.
    """
    u: np.ndarray
    neglogL: float
    params: Dict[str, float] = field(default_factory=dict)   # physical, lmfit names
    # pose, physical, kept explicit because dedup and display both need them hot
    theta: float = float('nan')
    phi: float = float('nan')
    protins: float = float('nan')
    cov: float = float('nan')
    # depth
    delta: float = 0.0
    chi2: float = float('nan')          # = 2*neglogL, the XPR objective
    n_active: int = 0
    chi2_fit: Optional[float] = None    # Fit.residual at the same point, user scale
    # certification
    hess_class: str = UNCERTIFIED
    hess_source: str = 'none'
    hessian: Optional[np.ndarray] = None    # exact, free block, unit cube
    gn: Optional[np.ndarray] = None         # Gauss-Newton J^T W J, free block
    eigmin: Optional[float] = None
    cond: Optional[float] = None
    n_floored: int = 0
    null_vector: Optional[np.ndarray] = None
    marginal_err: Optional[Dict[str, float]] = None
    active_bounds: Tuple[str, ...] = ()
    # within-run bookkeeping
    n_starts: int = 1
    stage1_rank: int = -1
    start_index: int = -1
    # weights
    laplace_logw: Optional[float] = None
    laplace_wt: Optional[float] = None
    # labels
    flags: Tuple[str, ...] = ()
    template_label: str = ''
    R: Optional[np.ndarray] = None      # cached model curve, for the overlay

    @property
    def chi2_per_n(self):
        return self.chi2 / self.n_active if self.n_active else float('nan')

    @property
    def band(self):
        """Which hemisphere of ``cos(theta)`` the pose sits in.

        The up/down question is the one that actually bites, and the admissible set
        is typically dominated by the opposite band, so this is worth a column of
        its own rather than being buried in a sphere plot.
        """
        if not np.isfinite(self.theta):
            return '?'
        return 'up' if np.cos(np.deg2rad(self.theta)) > 0 else 'down'

    @property
    def is_min(self):
        return self.hess_class.startswith(MIN)


# ------------------------------------------------------------------ geometry ---

def unit_vector(theta_deg, phi_deg):
    """Orientation as a unit vector. theta is polar from +z, phi azimuthal."""
    t = np.deg2rad(np.asarray(theta_deg, dtype=float))
    p = np.deg2rad(np.asarray(phi_deg, dtype=float))
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)],
                    axis=-1)


def great_circle(theta_a, phi_a, theta_b, phi_b):
    """Angle between two orientations, degrees.

    Use this and never ``|dtheta| + |dphi|``, which is wrong in two ways that both
    occur in practice: at theta = 0 every phi is the SAME orientation, and phi wraps
    (359 and 1 are 2 degrees apart, not 358).
    """
    va = unit_vector(theta_a, phi_a)
    vb = unit_vector(theta_b, phi_b)
    return float(np.rad2deg(np.arccos(np.clip(np.sum(va * vb, axis=-1), -1.0, 1.0))))


def same_position(a, b, rule=DedupRule()):
    """Whether two modes are the same solution. Position only -- never depth."""
    return (great_circle(a.theta, a.phi, b.theta, b.phi) <= rule.gc_deg
            and abs(a.protins - b.protins) <= rule.d_protins
            and abs(a.cov - b.cov) <= rule.d_cov)


def dedup(modes, rule=DedupRule()):
    """Greedy ascending-depth merge on POSITION. Returns the kept representatives.

    The deepest member of each cluster represents it, and ``n_starts`` accumulates
    over the cluster -- which is the basin-size proxy and the fallback weight when
    no curvature is available.
    """
    order = sorted((m for m in modes if np.isfinite(m.neglogL)),
                   key=lambda m: m.neglogL)
    kept = []
    for m in order:
        for k in kept:
            if same_position(m, k, rule):
                k.n_starts += m.n_starts
                break
        else:
            kept.append(m)
    return kept


# ------------------------------------------------------------ certification ---

def classify(hessian_free, u_free=None, lo_tol=1e-6, rel_tol=1e-8):
    """``(label, eigmin, cond, null_vector)`` from the Hessian of ``-logL``.

    ``hessian_free`` must ALREADY be restricted to the free coordinates. Pinned
    coordinates carry a ``(hi-lo)`` factor of ~1e-9 through the chain rule, so their
    Hessian rows are numerically zero and including them drags every endpoint to
    DEGENERATE. Restricting is mandatory, not cosmetic.

    ``u_free`` is the endpoint in the unit cube; when given, coordinates railed at a
    face are excluded from the eigenvalue test and the label gains a ``-BOUND``
    suffix. This is a correction to the published method, which tests the full free
    block regardless: a CONSTRAINED minimum sitting on a box face has an ascent
    direction pointing out of the box, so it is reported SADDLE when it is in fact
    the best point available inside the search region.
    """
    H = np.asarray(hessian_free, dtype=float)
    if H.size == 0 or not np.isfinite(H).all():
        return UNCERTIFIED, None, None, None
    H = 0.5 * (H + H.T)
    suffix = ''
    idx = np.arange(H.shape[0])
    if u_free is not None:
        u = np.asarray(u_free, dtype=float).ravel()
        railed = (u <= lo_tol) | (u >= 1.0 - lo_tol)
        if railed.any():
            suffix = '-BOUND'
            idx = idx[~railed]
    if len(idx) == 0:
        # every free coordinate is at a bound: nothing left to certify
        return MIN + suffix, None, None, None
    Hs = H[np.ix_(idx, idx)]
    w, V = np.linalg.eigh(Hs)
    scale = max(float(np.max(np.abs(w))), 1.0)
    tol = rel_tol * scale
    if w[0] > tol:
        label = MIN
        null = None
    elif w[0] < -tol:
        label = SADDLE
        null = V[:, 0]
    else:
        label = DEGENERATE
        null = V[:, 0]
    cond = (float(np.max(np.abs(w)) / np.min(np.abs(w)))
            if np.min(np.abs(w)) > 0 else float('inf'))
    return label + suffix, float(w[0]), cond, null


def laplace_weights(modes, floor_rel=1e-6):
    """Laplace weights over the certified minima. Mutates and returns ``modes``.

    Two guards, both measured rather than defensive:

    * use the **Gauss-Newton** ``J^T W J``, not the exact Hessian. The difference is
      ``||H - J^T J|| / ||H|| = 0.002``, and GN is positive definite by
      construction, where the exact ``-H`` has a negative eigenvalue and
      ``det^(-1/2)`` is undefined. It does.
    * **floor the eigenvalues.** ``det^(-1/2)`` is dominated by the smallest one,
      and this problem has a genuinely soft direction (``sig, d1, d2, rho1, rho2``
      correlate at |r| = 1.00, with condition number ~7e5 at the optimum). Unfloored
      the weights are numerical noise.

    ``n_floored`` is recorded per mode and must be surfaced: a weight resting on
    three floored directions is a choice, not a measurement.

    Modes with no curvature fall back to the basin-size proxy ``n_starts``.
    """
    live = [m for m in modes if m.is_min]
    if not live:
        return modes
    usable = [m for m in live if m.gn is not None or m.hessian is not None]
    if not usable:
        tot = float(sum(m.n_starts for m in live)) or 1.0
        for m in live:
            m.laplace_logw = None
            m.laplace_wt = m.n_starts / tot
        return modes

    logw = []
    for m in usable:
        G = np.asarray(m.gn if m.gn is not None else m.hessian, dtype=float)
        G = 0.5 * (G + G.T)
        w = np.linalg.eigvalsh(G)
        wmax = float(np.max(np.abs(w))) if w.size else 1.0
        floor = floor_rel * max(wmax, 1e-300)
        wf = np.maximum(w, floor)
        m.n_floored = int(np.sum(w < floor))
        # logL - 0.5*logdet(G); the physical<->u Jacobian is common to all modes
        # in one set and cancels in the normalisation, which is why u is canonical.
        logw.append(-m.neglogL - 0.5 * float(np.sum(np.log(wf))))
    logw = np.asarray(logw, dtype=float)
    logw -= logw.max()
    wt = np.exp(logw)
    wt /= wt.sum()
    for m, lw, w_ in zip(usable, logw, wt):
        m.laplace_logw = float(lw)
        m.laplace_wt = float(w_)
    weighted = {id(m) for m in usable}
    for m in live:
        if id(m) not in weighted:
            # a certified minimum with no curvature: leave the weight UNSET rather
            # than inventing one, so the UI can say so instead of showing a number
            # that means something different from its neighbours.
            m.laplace_logw = None
            m.laplace_wt = None
    return modes


# ------------------------------------------------------------------ the set ---

class ModeSet(object):
    """A ranked list of modes plus the provenance that says what it can mean.

    Cannot be built without a :class:`Provenance` -- that is deliberate, see the
    module docstring.
    """

    def __init__(self, modes, prov, rule=None, cancelled=False, warnings=None):
        if prov is None:
            raise ValueError('a ModeSet needs a Provenance: without it the list has '
                             'no stated budget, and its length is meaningless')
        self.prov = list(prov) if isinstance(prov, (list, tuple)) else [prov]
        self.dedup_rule = rule or DedupRule()
        self.cancelled = bool(cancelled)
        self.warnings = list(warnings or [])
        self.stale_reason = None
        self.current = 0
        self.modes = self._rank(modes)

    # -- construction -------------------------------------------------------
    def _rank(self, modes):
        ms = sorted((m for m in modes if np.isfinite(m.neglogL)),
                    key=lambda m: m.neglogL)
        if ms:
            best = ms[0].neglogL
            for m in ms:
                m.delta = float(m.neglogL - best)
        return ms

    # -- basic access -------------------------------------------------------
    def __len__(self):
        return len(self.modes)

    def __iter__(self):
        return iter(self.modes)

    def __getitem__(self, i):
        return self.modes[i]

    def best(self):
        return self.modes[0] if self.modes else None

    def minima(self):
        """The certified subset. This, and only this, may be handed to a sampler.

        A saddle is not a mode: starting a chain there wastes it. Saddles stay in
        ``modes`` so the UI can report them.
        """
        return [m for m in self.modes if m.is_min]

    # -- the threshold API: functions of delta, never a stored verdict -------
    def within(self, delta):
        """Modes no deeper than ``delta`` below the best."""
        return [m for m in self.modes if m.delta <= delta]

    def n_distinct(self, delta):
        """How many distinct positions were FOUND within ``delta``.

        A lower bound on the number of comparably-good poses, never "the" number:
        at this budget the measured shortfall against a converged reference is
        ``prov.power.shortfall``-fold.
        """
        return len(dedup(self.within(delta), self.dedup_rule))

    def pose_span(self, delta):
        """Widest great-circle separation among modes within ``delta``, degrees."""
        ms = self.within(delta)
        if len(ms) < 2:
            return 0.0
        return max(great_circle(a.theta, a.phi, b.theta, b.phi)
                   for i, a in enumerate(ms) for b in ms[i + 1:])

    def bands(self, delta):
        """``{'up': n, 'down': n}`` within ``delta``.

        Both bands populated is the signal that the pose is NOT determined, and it
        is the case a 64-start screen most often misses.
        """
        out = {'up': 0, 'down': 0, '?': 0}
        for m in self.within(delta):
            out[m.band] = out.get(m.band, 0) + 1
        return {k: v for k, v in out.items() if v or k in ('up', 'down')}

    def expected_at_reference(self, delta):
        """Extrapolated distinct count at the converged reference budget.

        ``n_distinct(delta) * power.shortfall``. An EXTRAPOLATION from a measured
        median ratio, and must be labelled as one wherever it is shown -- but it is
        the number that puts the budget shortfall on the same axes as the answer.
        """
        return self.n_distinct(delta) * self.prov[0].power.shortfall

    def deepest_found(self):
        return self.modes[-1].delta if self.modes else 0.0

    # -- weights ------------------------------------------------------------
    def laplace_weights(self, floor_rel=1e-6):
        laplace_weights(self.modes, floor_rel=floor_rel)
        return [m.laplace_wt for m in self.modes]

    # -- housekeeping -------------------------------------------------------
    @property
    def templates(self):
        return sorted({p.template_label for p in self.prov if p.template_label})

    def compatible_with(self, data_sha256=None, n_active=None, fit_scale=None,
                        num_boxes=None, spacez=None):
        """A reason this set no longer describes the live state, or None."""
        p = self.prov[0]
        for name, live, was in (('the data or window', data_sha256, p.data_sha256),
                                ('the active point count', n_active, p.n_active),
                                ('the fit scale', fit_scale, p.fit_scale),
                                ('the box count', num_boxes, p.num_boxes),
                                ('spacez', spacez, p.spacez)):
            if live is not None and was not in (None, '', 0) and live != was:
                return ('%s changed since this scan (%r -> %r), so its -logL values '
                        'no longer describe this fit' % (name, was, live))
        return None

    def merge(self, other):
        """Union two sets. REFUSES when their provenance makes depths incomparable.

        The PDB template is the one axis allowed to differ, because it does not
        touch the mesh or the vector layout. Modes from different templates are
        never merged into one row even when they coincide in pose: cross-template
        agreement is the single most useful signal a multi-template run produces,
        and evidence merged away is evidence you cannot show.
        """
        why = self.prov[0].incompatible_with(other.prov[0])
        if why is not None:
            raise ValueError('refusing to merge these mode sets: %s' % why)
        out = ModeSet(list(self.modes) + list(other.modes),
                      list(self.prov) + list(other.prov),
                      rule=self.dedup_rule,
                      cancelled=self.cancelled or other.cancelled,
                      warnings=self.warnings + other.warnings)
        return out


def data_fingerprint(q, y, err):
    """SHA-256 of the ACTIVE (q, R, err) triple.

    The only exactly-right staleness test: it catches a changed file, a changed x
    window, a changed column limit and a changed error column, and is insensitive to
    path and to comments.
    """
    h = hashlib.sha256()
    for a in (q, y, err):
        a = np.ascontiguousarray(np.asarray(a, dtype=float).ravel())
        h.update(str(a.shape).encode('ascii'))
        h.update(a.tobytes())
    return h.hexdigest()
