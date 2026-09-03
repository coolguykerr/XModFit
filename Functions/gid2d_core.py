r"""
gid2d_core.py -- shared geometry / quadrature / IO helpers for GID2D.

This module is deliberately NOT inside a ``Functions/<category>/`` folder:
``xmodfit.update_functions`` (xmodfit.py:4889-4901) lists every ``.py`` in a
category folder as a fittable model and then does ``getattr(module, filename)``,
so a helper placed there crashes the function list. Same reason ``pdb_cache.py``
lives here.

Provenance
----------
The lattice / signed-q_z / Ewald / cell-inversion block is a VENDORED COPY of
``grazing-incidence-scattering/scripts/gic_core.py`` (skill asset). It is
vendored rather than imported because that file ships inside a session-scoped
skill directory whose path changes between sessions, and an XModFit model must
import cleanly from the repo alone. Conventions are preserved exactly so that
numbers here and numbers from the skill's ``selfcheck.py`` agree. ``selfcheck()``
at the bottom re-asserts the identities that matter, in place; run it after any
edit to this file.

Conventions (do not "simplify" these -- each one is a documented trap)
---------------------------------------------------------------------
1. q_z is SIGNED: ``q_z = -(G . t_hat) tan t``. Sum over the SIX first-order
   points and evaluate only on q_z >= 0. Never take ``abs(q_z)``: it inflates
   off-plane integrated intensity ~2x and fabricates peaks whose rod maxima are
   genuinely below the horizon. The observed (11)+(1-1):(02) = 2:1 arises
   because a rod centred at q_z = 0 is HALVED by the horizon while one centred
   well above it is nearly whole.
2. Areas need a cross product: ``A = 2 pi^2 / |b1 x b2|``, not
   ``2 pi^2 / (|b1||b2|)``. Those agree only for a rectangular cell (psi = 0 or
   90 deg); the norm form overstates the area by up to ~7% at intermediate tilt
   azimuth.
3. The Ewald cross-section radius ``sqrt(k^2 - (q_z - k sin ai)^2)`` is defined
   exactly once, here, and shared by any condition test and any drawing.
4. The Vineyard convention matches ``Functions/GISAXS/Rod_Lipid.py:58-64``:
   ``trans(qbeta, qc/2)`` with ``qbeta = q_z - k sin(alpha_i)``. ``qc`` is the
   critical TOTAL momentum transfer, so the single-side critical value is qc/2.
   Keeping this identical means GID2D and Rod_Lipid agree on the same data.
"""

import os
import numpy as np

TAU = 2.0 * np.pi
HC = 12.39842            # keV * Angstrom
R_E = 2.8179403e-5       # classical electron radius, Angstrom
RHO_E_WATER = 0.3342     # electron density of water, e/A^3

# numpy < 2.0 (the cmc-py39 env ships 1.21.6) has no np.trapezoid
trapz = getattr(np, "trapezoid", np.trapz)


# ------------------------------------------------------------------ beam ----
def k_of(energy_kev):
    """Wavevector magnitude, A^-1."""
    return TAU / (HC / energy_kev)


def k_inplane(energy_kev, alpha_i_deg):
    """In-plane projection of k_in. alpha_i enters only through cos(alpha_i),
    which is why it is NOT the Bragg-condition knob in grazing incidence."""
    return k_of(energy_kev) * np.cos(np.radians(alpha_i_deg))


def critical_angle(energy_kev, rho_e=RHO_E_WATER):
    """alpha_c in radians: delta = r_e lambda^2 rho_e / 2pi, alpha_c = sqrt(2 delta)."""
    lam = HC / energy_kev
    return float(np.sqrt(2.0 * R_E * lam * lam * rho_e / TAU))


def qc_of(energy_kev, rho_e=RHO_E_WATER):
    """Critical TOTAL momentum transfer, q_c = 2 k sin(alpha_c) ~ 4 sqrt(pi r_e rho_e).
    Independent of energy to the accuracy of the small-angle expansion."""
    return float(2.0 * k_of(energy_kev) * np.sin(critical_angle(energy_kev, rho_e)))


# --------------------------------------------------------------- lattice ----
class Lattice2D:
    """Chain lattice with tilt, plus the perpendicular-plane (packing) net.

    The surface net is the perpendicular-plane hexagonal net stretched by
    1/cos(t) along t_hat, then given an extra uniaxial strain (1+eps) along a.
    With eps = 0 the perpendicular net is EXACTLY hexagonal for any t and psi,
    so every departure from sqrt(3) there is backbone strain, not tilt.

    Centred rectangular cell, a along a nearest-neighbour row, b = sqrt(3) a0.
    Centring kills h+k odd, so the first order is (1,1), (1,-1), (0,2) -- (0,2)
    is the FIRST allowed reflection along b*, not a second order.
    """

    FAMILIES = ((1, 1), (1, -1), (0, 2))
    SIX = ((1, 1), (1, -1), (0, 2), (-1, -1), (-1, 1), (0, -2))

    def __init__(self, a0=4.84, t_deg=0.0, psi_deg=0.0, eps=0.0):
        self.a0, self.eps = float(a0), float(eps)
        self.t = np.radians(t_deg)
        self.psi = np.radians(psi_deg)
        self.t_hat = np.array([np.cos(self.psi), np.sin(self.psi)])

        f = 1.0 / np.cos(self.t) - 1.0
        stretch = lambda v: v + f * (v @ self.t_hat) * self.t_hat
        strain = lambda v: np.array([v[0] * (1.0 + self.eps), v[1]])
        self.A1 = strain(stretch(np.array([self.a0, 0.0])))
        self.A2 = strain(stretch(np.array([0.0, np.sqrt(3.0) * self.a0])))

        A = np.column_stack([self.A1, self.A2])
        B = TAU * np.linalg.inv(A).T
        self.b1, self.b2 = B[:, 0], B[:, 1]

        # chain axis and an orthonormal basis of the plane perpendicular to it
        self.n = np.array([np.sin(self.t) * self.t_hat[0],
                           np.sin(self.t) * self.t_hat[1], np.cos(self.t)])
        e2 = np.cross([0.0, 0.0, 1.0], [self.t_hat[0], self.t_hat[1], 0.0])
        nrm = np.linalg.norm(e2)
        self.e2 = e2 / nrm if nrm > 1e-12 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(self.n, self.e2)
        self.e1 = e1 / np.linalg.norm(e1)
        self.P1, self.P2 = self.perp(self.A1), self.perp(self.A2)

    def perp(self, v2):
        """Project an in-surface vector into the plane perpendicular to n."""
        v = np.array([v2[0], v2[1], 0.0])
        v = v - (v @ self.n) * self.n
        return np.array([v @ self.e1, v @ self.e2])

    def G(self, h, k):
        return h * self.b1 + k * self.b2

    def qz_signed(self, h, k):
        """SIGNED out-of-plane position of the rod maximum. See convention 1."""
        return float(-(self.G(h, k) @ self.t_hat) * np.tan(self.t))

    def reflections(self, six=True):
        """First-order reflections. Use six=True for anything that integrates
        intensity: the three-family form double-counts Friedel mates if you also
        take abs(q_z)."""
        out = []
        for (h, k) in (self.SIX if six else self.FAMILIES):
            g = self.G(h, k)
            fam = ((1, 1) if (h, k) in ((1, 1), (-1, -1)) else
                   (1, -1) if (h, k) in ((1, -1), (-1, 1)) else (0, 2))
            out.append(dict(hk=(h, k), fam=fam, g=g,
                            qxy=float(np.linalg.norm(g)),
                            qz=self.qz_signed(h, k)))
        return out

    def area_per_chain(self):
        """2 pi^2 / |b1 x b2|. See convention 2 -- the norm form is wrong."""
        cross = abs(self.b1[0] * self.b2[1] - self.b1[1] * self.b2[0])
        return float(2.0 * np.pi ** 2 / cross)

    def area_per_molecule(self, chains=2):
        return chains * self.area_per_chain()

    def sheet(self, h, k):
        """Geometry of the diffracting sheet for (hk): a lattice row swept along
        the chain axis, with Q as its normal. tan(chi) = cos(psi_G) tan t."""
        g = self.G(h, k)
        q = float(np.linalg.norm(g))
        qz = self.qz_signed(h, k)
        Q = np.array([g[0], g[1], qz])
        Qm = float(np.linalg.norm(Q))
        u = np.array([-g[1], g[0], 0.0]) / q          # in-plane row direction
        N = np.cross(u, self.n)
        N = N / np.linalg.norm(N)
        if N @ Q < 0:
            N = -N
        cpsi = float((g @ self.t_hat) / q)
        return dict(chi=float(np.arctan2(abs(qz), q)),
                    chi_formula=float(np.arctan(abs(cpsi) * np.tan(self.t))),
                    d_xy=TAU / q, d_sheet=TAU / Qm, Qm=Qm,
                    u_hat=u, N_hat=N, Q=Q, cos_psi_G=cpsi)

    def perp_row_spacing(self, h, k):
        """Row spacing of the (hk) family in the perpendicular-plane net.
        Equals d_sheet: the tilted sheets are the close-packed planes of the
        real packing and the surface only shows their projection."""
        Bp = TAU * np.linalg.inv(np.column_stack([self.P1, self.P2])).T
        return float(TAU / np.linalg.norm(h * Bp[:, 0] + k * Bp[:, 1]))


# ----------------------------------------------------------- cell from data --
def cell_from_peaks(q11, q02):
    """Centred-rectangular cell from two in-plane peak positions. Assumes
    a . b = 0; for an oblique cell use cell_from_peaks_oblique."""
    bstar = q02 / 2.0
    inner = q11 * q11 - bstar * bstar
    if inner <= 0:
        raise ValueError("q11 too small for this q02: no rectangular solution")
    astar = np.sqrt(inner)
    return dict(astar=float(astar), bstar=float(bstar),
                a=float(TAU / astar), b=float(TAU / bstar),
                area_per_chain=float(2 * np.pi ** 2 / (astar * bstar)))


def cell_from_peaks_oblique(q11, q1m1, q02):
    """General form, valid when (11) and (1-1) are no longer degenerate.
    Reduces to cell_from_peaks when q11 == q1m1. The familiar
    a* = sqrt(q11^2 - q02^2/4) fails SILENTLY on an oblique cell (7.4% in area
    for a 27 deg tilt at psi = 33 deg), which is why this needs the third peak."""
    b2 = q02 * q02 / 4.0
    a2 = (q11 * q11 + q1m1 * q1m1) / 2.0 - b2
    ab = (q11 * q11 - q1m1 * q1m1) / 4.0
    det = a2 * b2 - ab * ab
    if det <= 0:
        raise ValueError("inconsistent peak triple")
    return dict(area_per_chain=float(2 * np.pi ** 2 / np.sqrt(det)),
                astar=float(np.sqrt(a2)), bstar=float(np.sqrt(b2)),
                a_dot_b_star=float(ab))


def tilt_from_peaks_nn(qz11, q11, q02):
    """NN tilt: (02) stays at q_z = 0 and tan t = q_z(11) / a*. Returns a SIGNED
    tilt in radians -- the DPPC NN preset gives -18.85 deg. Document the sign at
    the call site rather than sprinkling abs()."""
    return float(np.arctan(qz11 / np.sqrt(q11 * q11 - q02 * q02 / 4.0)))


def tilt_from_peaks_nnn(qz02, q02):
    """NNN tilt: G(02) is parallel to t_hat, so tan t = q_z(02)/q_xy(02). The one
    case where the naive ratio is correct, because the peak sits exactly on the
    envelope q_z = q_xy tan t."""
    return float(np.arctan(qz02 / q02))


# ------------------------------------------------------------ Ewald sphere --
def ewald_cross_section(qz, energy_kev, alpha_i_deg):
    """Radius of the sphere's horizontal cross-section at height qz, no
    accessibility clamp. At qz = 0 this is k cos(ai). Note qz = 0 is NOT
    reachable for alpha_i > 0: the lowest accessible height is k sin(alpha_i),
    part of why a nominally in-plane (02) peak is reported above zero."""
    k = k_of(energy_kev)
    u = qz - k * np.sin(np.radians(alpha_i_deg))
    v = k * k - u * u
    return float(np.sqrt(v)) if v > 0 else float("nan")


def ewald_radius(qz, energy_kev, alpha_i_deg):
    """As ewald_cross_section but clamped to the lowest accessible height, so
    one definition is shared by the condition test and any drawing."""
    k = k_of(energy_kev)
    si = np.sin(np.radians(alpha_i_deg))
    u = max(k * si, qz) - k * si
    v = k * k - u * u
    return float(np.sqrt(v)) if v > 0 else float("nan")


def alpha_f(qz, energy_kev, alpha_i_deg):
    """Out-of-plane detector angle. Qz = k(sin af + sin ai), so a peak moves DOWN
    in af as ai rises even though its q_z is unchanged."""
    k = k_of(energy_kev)
    s = qz / k - np.sin(np.radians(alpha_i_deg))
    return float(np.arcsin(np.clip(s, -1.0, 1.0)))


# ------------------------------------------------------------ lineshapes ----
def sigma_size(L):
    """In-plane peak sigma from coherence length L. Scherrer FWHM = 2 pi / L."""
    return float((TAU / max(L, 1e-9)) / 2.3548)


def transmission(qz, energy_kev, alpha_i_deg, qc):
    """|T(alpha_f)|^2, the exit-side Vineyard/Yoneda factor.

    Matches Rod_Lipid.trans(qbeta, qc/2) exactly -- see convention 4. Structure,
    which qz_nodes() depends on:
        qb = qz - k sin(ai), clipped at 0
        qb <  qc/2 :  |T|^2 = 4 qb^2 / (qc/2)^2      (exact parabola, rises 0->4)
        qb >= qc/2 :  |T|^2 = 4 qb^2 / (qb + sqrt(qb^2-(qc/2)^2))^2   (4 -> 1)
    so there are exactly two non-smooth points: qb = 0 (clip kink, |T|^2 == 0
    below it) and qb = qc/2 (SQRT BRANCH POINT at the maximum, infinite
    one-sided derivative). Uniform trapezoid over these converges
    non-monotonically at ~1% for practical node counts; qz_nodes() fixes it.
    """
    k = k_of(energy_kev)
    qb = np.clip(np.asarray(qz, float) - k * np.sin(np.radians(alpha_i_deg)), 0.0, None)
    c = abs(qc) / 2.0
    if c < 1e-12:
        return np.where(qb > 0, 1.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tr = 2.0 * qb / (qb + np.sqrt(qb ** 2 - c ** 2 + 0j))
    out = np.abs(tr) ** 2
    return np.where(qb > 0, out, 0.0)


def radial_profile(qxy, q0, L_dom, lineshape="Gaussian", eta=0.5, powder=True):
    """In-plane radial lineshape centred on |G| = q0, unit area in q_xy.

    ``powder`` applies the analytic 2D-powder ring-average Lorentz factor
    sqrt(2 pi) sigma / sqrt(|q_xy| q0) from references/kinematics.md. Keeping
    the sqrt(2 pi) sigma normalisation (which the feasibility prototype dropped)
    matters: without it the peak AREA scales with L_dom, so L_dom and yscale
    become needlessly correlated.

    Gaussian is what a finite domain gives; a Lorentzian tail is what a finite
    positional correlation length gives. This data has enough q_xy wing coverage
    (1.235-1.568) to discriminate, hence the choice.
    """
    qxy = np.asarray(qxy, float)
    sig = sigma_size(L_dom)
    d = qxy - q0
    if lineshape == "Gaussian":
        prof = np.exp(-d * d / (2.0 * sig * sig)) / (np.sqrt(TAU) * sig)
    elif lineshape == "Lorentzian":
        gam = sig * 2.3548 / 2.0                       # match FWHM
        prof = (gam / np.pi) / (d * d + gam * gam)
    elif lineshape == "PseudoVoigt":
        gam = sig * 2.3548 / 2.0
        g = np.exp(-d * d / (2.0 * sig * sig)) / (np.sqrt(TAU) * sig)
        l = (gam / np.pi) / (d * d + gam * gam)
        e = float(np.clip(eta, 0.0, 1.0))
        prof = (1.0 - e) * g + e * l
    else:
        raise ValueError("unknown lineshape %r" % (lineshape,))
    if powder:
        prof = prof * np.sqrt(TAU) * sig / np.sqrt(np.maximum(np.abs(qxy) * q0, 1e-12))
    return prof


def rod_profile(qz, qz0, t_rad, L_chain):
    """|F_chain|^2 for a uniform rod of length L_chain along the chain axis,
    centred on the SIGNED qz0.

    The cos(t) factor is NOT optional here: Rod_Lipid omits it and lets H_lipid
    absorb it, which is fine standalone and wrong in GID2D because the same t
    also sets the peak positions.
    """
    return np.sinc(L_chain * np.cos(t_rad) * (np.asarray(qz, float) - qz0) / TAU) ** 2


# ------------------------------------------------------------- quadrature ---
def qz_nodes(lo, hi, energy_kev, alpha_i_deg, qc, n=401, dqz_max=None):
    """q_z nodes for integrating |T_f|^2 x (smooth) with the trapezoid rule.

    THE SECTION-9 FIX. The naive np.linspace grid converges non-monotonically
    (4.9e-3 at n=51, WORSENING to 2.7e-2 at n=101, then 5.9e-3 / 1.5e-3 / ...)
    because |T_f|^2 has a sqrt branch point at q_b = q_c/2. Verified here:
    turning |T_f|^2 off entirely restores textbook 4x-per-doubling convergence
    (1.7e-5 at n=51), so the cusp is the whole cause. The earlier control that
    appeared to exonerate it set qc -> 0 rather than removing |T_f|^2, which
    leaves both the clip kink and a 0/0, i.e. it was still testing the cusp.

    Construction, with z0 = k sin(ai) and z1 = z0 + qc/2:
      * [lo, z0)  dropped -- |T_f|^2 is identically zero there, not merely small
      * [z0, z1]  uniform: |T_f|^2 = 4 qb^2/(qc/2)^2 exactly, a parabola
      * (z1, hi]  nodes at q_b = (qc/2) cosh(s), s uniform. This grades
                  q_z - z1 ~ s^2, the matched grading for a sqrt branch point.
      * dqz_max   optional cap on the coarse end so a narrow rod (large L_ch)
                  stays resolved; use ~ (2 pi / (L_ch cos t)) / 8.

    Against adaptive scipy.integrate.quad ground truth, at L_ch = 35 A and
    q_xy = 1.489: ~158 nodes gives +9.5e-4 (uniform -4.5e-2), ~458 nodes gives
    +6.6e-5 (uniform -4.8e-3), monotone, holding for L_ch = 19-100 A.
    """
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        return np.array([lo])
    c = abs(qc) / 2.0
    z0 = max(k_of(energy_kev) * np.sin(np.radians(alpha_i_deg)), lo)
    if z0 >= hi:                                   # window entirely sub-horizon
        return np.array([hi])
    n = max(16, int(n))
    if c < 1e-12:                                  # no cusp: uniform is optimal
        z = np.linspace(z0, hi, n)
    else:
        z1 = z0 + c
        if z1 >= hi:                               # window ends inside parabola
            z = np.linspace(z0, hi, n)
        else:
            n_mid = max(8, int(0.12 * n))
            smax = np.arccosh((hi - z0) / c)
            z = np.concatenate([
                np.linspace(z0, z1, n_mid),
                z0 + c * np.cosh(np.linspace(0.0, smax, max(16, n - n_mid))),
            ])
    if dqz_max and dqz_max > 0:
        z = np.concatenate([z, np.arange(z0, hi + dqz_max, dqz_max)])
    z = np.unique(z)
    return z[(z >= lo - 1e-15) & (z <= hi + 1e-15)]


def tilt_quadrature(n):
    """Nodes/weights for a unit-variance Gaussian, weights summing to 1.

    Used to average the lattice over a spread in chain tilt: t_j = t + x_j sig_t,
    with q_z0 RECOMPUTED per node (the whole point -- a tilt spread smears
    off-plane reflections along q_z = q_xy tan t while leaving q_z = 0
    reflections sharp, because q_z(02) = 0 independently of t under NN tilt).
    n = 1 collapses to the single-tilt model exactly.
    """
    n = max(1, int(n))
    if n == 1:
        return np.array([0.0]), np.array([1.0])
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


# --------------------------------------------------------------------- IO ---
_MAP_CACHE = {}


def _stamp(path):
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


def load_map(path):
    """Load the indexed 2D-map format, cached on (path, mtime_ns, size).

    Keyed on the file STAMP, not the path alone, for the same reason it was
    needed in XLayersPDB_dev_smooth: an edited-in-place file must invalidate.

    Expected columns: ``idx I Ierr qxy qz``, ``#`` comments, idx monotone
    increasing. Returns a dict with float arrays plus grid metadata. The
    geometry is read from the qxy/qz COLUMNS, never recomputed from an
    ``idx = i_qz*n_qxy + i_qxy`` formula -- pack_map drops masked rows, so that
    arithmetic no longer holds.
    """
    key = _stamp(path)
    hit = _MAP_CACHE.get(key)
    if hit is not None:
        return hit
    d = np.loadtxt(path, comments="#")
    if d.ndim != 2 or d.shape[1] < 5:
        raise ValueError("%s: expected >=5 columns (idx I Ierr qxy qz), got %s"
                         % (path, d.shape))
    idx, I, Ierr, qxy, qz = (d[:, 0].astype(float), d[:, 1], d[:, 2],
                             d[:, 3], d[:, 4])
    if np.any(np.diff(idx) <= 0):
        raise ValueError("%s: column 0 must be strictly increasing" % path)
    uqxy, uqz = np.unique(qxy), np.unique(qz)
    out = dict(idx=idx, I=I, Ierr=Ierr, qxy=qxy, qz=qz,
               n=len(idx), uqxy=uqxy, uqz=uqz,
               n_qxy=len(uqxy), n_qz=len(uqz),
               rectangular=bool(len(uqxy) * len(uqz) == len(idx)),
               path=os.path.abspath(path))
    _MAP_CACHE[key] = out
    if len(_MAP_CACHE) > 8:                        # bounded, keep the newest
        for k in list(_MAP_CACHE)[:-8]:
            _MAP_CACHE.pop(k, None)
    return out


def row_lookup(x, mp):
    """Map a float row index to integer rows of the loaded map.

    self.x may arrive non-integer, because XModFit's x line-edit accepts e.g.
    np.linspace(0, 41796, 300) -- that is also how the no-data "Calculate" path
    works. Nearest-neighbour, clipped into range.
    """
    i = np.rint(np.asarray(x, float)).astype(np.int64)
    return np.clip(i, 0, mp["n"] - 1)


def pack_map(src, dst, drop_masked=True, sigma_floor_frac=1e-6,
             qz_min=0.0, qz_max=None, qxy_min=None, qxy_max=None,
             cols=(0, 1, 2, 3)):
    """Convert a raw ``qxy qz I Ierr`` map into the indexed format GID2D reads.

    Why the index column: XModFit needs x MONOTONE (Fit_Routines.py:174-176 and
    xmodfit.py:5637-5650 slice with np.where on x), all keys equal length
    (Data_Dialog builds them from one DataFrame), and one x column per dataset.
    Raw q_xy from a map is a sawtooth, so it cannot be x. A row index can.

    Why masked rows MUST be dropped here rather than windowed later: XModFit
    computes ``(data-fit)/err`` in Fit_Routines.residual, OUTSIDE the model, so
    a sigma = 0 row is 0/0 whatever y() returns -- there is no in-model escape.
    And they cannot be trimmed by xmin:xmax either: in the DPPC file the 2595
    masked pixels are scattered over 335 distinct q_z values, with only one
    full row, so no contiguous index range excludes them.

    Rows with Ierr <= 0 or I <= 0 are dropped together (in the DPPC file
    sigma == 0 <=> I == 0 exactly, plus 29 rows at I ~ -1e-14 from rounding).
    I <= 0 also has to go because fit_scale='Log' takes log10(data).

    sigma_floor_frac is NOT cosmetic, and dropping only the exact zeros is not
    enough. Measured on the DPPC file: 26 further pixels carry sigma down to
    1.9e-16 -- the same mask, with floating-point residue instead of a clean
    zero. Because 1/sigma reaches 5e15 there, those 26 pixels out of 39018 carry
    ~100% of chi^2 (the top 10 alone carry 97%), so a fit would be driven
    entirely by mask residue. The separation is unambiguous: sorted sigma jumps
    by a factor 1.6e11 between rank 25 (8.7e-15) and rank 26 (1.4e-3), so any
    floor from 1e-9 to 1e-5 x median drops exactly those 26. The default sits in
    the middle of that 11-decade gap. The gap actually found is written into the
    output header, so a different instrument's data can be sanity-checked.
    """
    d = np.loadtxt(src)
    iq, iz, ii, ie = cols
    qxy, qz, I, Ierr = d[:, iq], d[:, iz], d[:, ii], d[:, ie]
    keep = np.ones(len(d), bool)
    n0 = len(d)
    gap_note = 'sigma floor not applied'
    if drop_masked:
        keep &= (Ierr > 0) & (I > 0)
        if sigma_floor_frac and sigma_floor_frac > 0:
            pos = Ierr[keep]
            if len(pos):
                floor = sigma_floor_frac * float(np.median(pos))
                below = keep & (Ierr < floor)
                keep &= Ierr >= floor
                srt = np.sort(pos)
                lo = srt[srt < floor]
                hi = srt[srt >= floor]
                gap = (float(hi[0] / lo[-1]) if len(lo) and len(hi) and lo[-1] > 0
                       else float('inf'))
                gap_note = ('sigma floor %.3e (= %.0e x median) dropped %d '
                            'near-zero-sigma pixels; gap at the floor = %.3g'
                            % (floor, sigma_floor_frac, int(below.sum()), gap))
    if qz_min is not None:
        keep &= qz >= qz_min
    if qz_max is not None:
        keep &= qz <= qz_max
    if qxy_min is not None:
        keep &= qxy >= qxy_min
    if qxy_max is not None:
        keep &= qxy <= qxy_max
    qxy, qz, I, Ierr = qxy[keep], qz[keep], I[keep], Ierr[keep]
    n = len(qxy)
    if n == 0:
        raise ValueError("packing kept 0 rows")
    with open(dst, "w") as fh:
        fh.write("#indexed 2D GID map for Functions/GISAXS/GID2D.py\n")
        fh.write("#source=%s\n" % os.path.abspath(src))
        fh.write("#kept %d of %d rows (dropped %d: masked/nonpositive/out-of-window)\n"
                 % (n, n0, n0 - n))
        fh.write("#%s\n" % gap_note)
        fh.write("#qxy %.6f..%.6f (%d unique)   qz %.6f..%.6f (%d unique)\n"
                 % (qxy.min(), qxy.max(), len(np.unique(qxy)),
                    qz.min(), qz.max(), len(np.unique(qz))))
        fh.write("#col_names=['idx','I','Ierr','qxy','qz']\n")
        for j in range(n):
            fh.write("%d %.6e %.6e %.6e %.6e\n"
                     % (j, I[j], Ierr[j], qxy[j], qz[j]))
    return dict(kept=n, dropped=n0 - n, path=os.path.abspath(dst),
                sigma_note=gap_note)


# -------------------------------------------------------------- presets -----
PRESETS = {
    "hex": dict(a0=4.84, t_deg=0.0, psi_deg=0.0, eps=0.0, L=250.0,
                note="Untilted hexagonal (rotator / LS-like). One peak, qz = 0."),
    "dppc_nn": dict(a0=4.8743, t_deg=18.85, psi_deg=0.0, eps=-0.0321, L=200.0,
                    note="DPPC monolayer, NN tilt. q=1.464 (qz=0.43) and "
                         "q=1.489 (qz~0), 42.0878 A^2/molecule."),
    "dppc_nnn": dict(a0=4.9137, t_deg=26.56, psi_deg=90.0, eps=0.0, L=200.0,
                     note="NNN tilt, qz = 0.33 and 0.66 (exact 1:2). From a "
                          "DPPC MULTILAYER caption, not a monolayer."),
}


def preset(name):
    p = PRESETS[name]
    return Lattice2D(p["a0"], p["t_deg"], p["psi_deg"], p["eps"])


# ------------------------------------------------------------- selfcheck ----
def selfcheck(verbose=True):
    """Re-assert, in place, the identities the vendored block must satisfy.

    Run after any edit to this file. These are algebraic identities, not
    approximations, so tolerances are tight; a growing residual means the model
    is wrong, not the arithmetic.
    """
    msgs = []

    def ok(name, cond, detail=""):
        msgs.append(("PASS" if cond else "FAIL", name, detail))
        assert cond, "%s  %s" % (name, detail)

    # -- signed q_z sums to zero over the six, and is never folded
    for nm in PRESETS:
        L = preset(nm)
        s = sum(R["qz"] for R in L.reflections(six=True))
        ok("qz_signed sums to 0 over SIX [%s]" % nm, abs(s) < 1e-12, "%.2e" % s)

    # -- area needs the cross product. The norm form 2pi^2/(|b1||b2|) agrees
    #    EXACTLY at psi = 0 and 90 deg (rectangular cell) and departs in
    #    between: measured here as -1.5% at t=33 deg, psi=45 deg.
    for t, ps in ((18.85, 0.0), (27.0, 90.0), (33.0, 0.0)):
        L = Lattice2D(4.8743, t, ps, 0.0)
        a_norm = 2 * np.pi ** 2 / (np.linalg.norm(L.b1) * np.linalg.norm(L.b2))
        ok("norm area == cross area at psi=%g (rectangular)" % ps,
           abs(a_norm / L.area_per_chain() - 1) < 1e-12)
    L = Lattice2D(4.8743, 33.0, 45.0, 0.0)
    a_norm = 2 * np.pi ** 2 / (np.linalg.norm(L.b1) * np.linalg.norm(L.b2))
    ok("norm area departs on an oblique cell (t=33, psi=45)",
       abs(a_norm / L.area_per_chain() - 1) > 0.01,
       "norm/cross-1 = %+.5f" % (a_norm / L.area_per_chain() - 1))

    # -- the bigger trap: inverting MEASURED peaks with the rectangular formula
    #    a* = sqrt(q11^2 - q02^2/4) assumes a.b = 0 and fails SILENTLY. Measured
    #    here as +6.4% in area at t=27 deg, psi=33 deg, rising to +10.2% at t=33.
    L = Lattice2D(4.8743, 27.0, 33.0, 0.0)
    R = {r["hk"]: r for r in L.reflections()}
    ob = cell_from_peaks_oblique(R[(1, 1)]["qxy"], R[(1, -1)]["qxy"],
                                 R[(0, 2)]["qxy"])["area_per_chain"]
    rc = cell_from_peaks(R[(1, 1)]["qxy"], R[(0, 2)]["qxy"])["area_per_chain"]
    ok("rectangular inversion overstates area on an oblique cell",
       rc / ob - 1 > 0.05, "rect/oblique-1 = %+.5f" % (rc / ob - 1))

    # -- oblique inversion must round-trip EXACTLY, over the whole t,psi domain
    mx = 0.0
    for t in np.linspace(0.0, 45.0, 16):
        for ps in np.linspace(0.0, 90.0, 19):
            L = Lattice2D(4.8743, t, ps, -0.03)
            R = {r["hk"]: r for r in L.reflections()}
            ob = cell_from_peaks_oblique(R[(1, 1)]["qxy"], R[(1, -1)]["qxy"],
                                         R[(0, 2)]["qxy"])["area_per_chain"]
            mx = max(mx, abs(ob / L.area_per_chain() - 1))
    ok("oblique inversion round-trips over t,psi sweep", mx < 1e-12, "%.2e" % mx)

    # -- DPPC NN preset reproduces the published cell
    L = preset("dppc_nn")
    A = L.area_per_molecule()
    ok("dppc_nn A/molecule = 42.0878", abs(A - 42.0878) < 5e-3, "%.4f" % A)

    # -- eps = 0 leaves the perpendicular net EXACTLY hexagonal for any t, psi
    for t in (0.0, 12.0, 27.0, 41.0):
        for ps in (0.0, 17.0, 45.0, 90.0):
            L = Lattice2D(4.84, t, ps, 0.0)
            r = np.linalg.norm(L.P2) / np.linalg.norm(L.P1)
            ok("perp net hexagonal (t=%g psi=%g)" % (t, ps),
               abs(r - np.sqrt(3.0)) < 1e-9, "b/a_perp = %.12f" % r)

    # -- sheet tip angle: two independent routes must agree
    for t in (7.0, 18.85, 33.0):
        L = Lattice2D(4.8743, t, 21.0, -0.02)
        for (h, k) in L.FAMILIES:
            sh = L.sheet(h, k)
            ok("chi == atan(cos psi_G tan t) (%d%d, t=%g)" % (h, k, t),
               abs(sh["chi"] - sh["chi_formula"]) < 1e-9,
               "%.3e" % (sh["chi"] - sh["chi_formula"]))
            ok("d_sheet == perp row spacing (%d%d, t=%g)" % (h, k, t),
               abs(sh["d_sheet"] - L.perp_row_spacing(h, k)) < 1e-9)

    # -- inversion round-trips the simulated positions
    L = preset("dppc_nn")
    R = {r["hk"]: r for r in L.reflections(six=True)}
    inv = cell_from_peaks_oblique(R[(1, 1)]["qxy"], R[(1, -1)]["qxy"], R[(0, 2)]["qxy"])
    ok("oblique inversion A/molecule = 42.0878",
       abs(2 * inv["area_per_chain"] - 42.0878) < 5e-3,
       "%.4f" % (2 * inv["area_per_chain"]))
    tnn = np.degrees(tilt_from_peaks_nn(R[(1, 1)]["qz"], R[(1, 1)]["qxy"], R[(0, 2)]["qxy"]))
    ok("NN tilt inversion = -18.85 deg (SIGNED)", abs(tnn + 18.85) < 1e-3, "%.4f" % tnn)

    # -- psi is EXACTLY a no-op at t = 0 (1/cos0 - 1 = 0), so it must not be fit
    L0, L9 = Lattice2D(4.84, 0.0, 0.0, 0.0), Lattice2D(4.84, 0.0, 73.0, 0.0)
    ok("psi is a no-op at t=0", np.allclose(L0.b1, L9.b1) and np.allclose(L0.b2, L9.b2))

    # -- transmission: the two analytic branches meet at the branch point
    E, ai, qc = 15.126, 0.10, 0.0217
    z1 = k_of(E) * np.sin(np.radians(ai)) + qc / 2
    ok("|T|^2 = 4 at the branch point",
       abs(float(transmission(np.array([z1]), E, ai, qc)[0]) - 4.0) < 1e-9)
    zb = np.linspace(0.0, z1, 200)
    qb = np.clip(zb - k_of(E) * np.sin(np.radians(ai)), 0, None)
    ok("|T|^2 is the exact parabola below the branch point",
       np.max(np.abs(transmission(zb, E, ai, qc) - 4 * qb ** 2 / (qc / 2) ** 2)) < 1e-9)
    ok("|T|^2 == 0 below the horizon",
       float(transmission(np.array([0.5 * k_of(E) * np.sin(np.radians(ai))]),
                          E, ai, qc)[0]) == 0.0)

    # -- qz_nodes never produces non-finite or out-of-range nodes
    for kw in (dict(alpha_i_deg=0.0, qc=qc), dict(alpha_i_deg=ai, qc=0.0),
               dict(alpha_i_deg=0.0, qc=0.0), dict(alpha_i_deg=ai, qc=qc)):
        z = qz_nodes(0.0, 1.4, E, n=401, dqz_max=0.02, **kw)
        ok("qz_nodes finite and in range %s" % kw,
           np.all(np.isfinite(z)) and z.min() >= -1e-15 and z.max() <= 1.4 + 1e-12
           and np.all(np.diff(z) > 0))

    # -- tilt quadrature: n=1 collapses exactly, weights normalised
    x1, w1 = tilt_quadrature(1)
    ok("tilt_quadrature(1) collapses", x1[0] == 0.0 and w1[0] == 1.0)
    for n in (3, 5, 7, 9):
        x, w = tilt_quadrature(n)
        ok("tilt weights sum to 1 (n=%d)" % n, abs(w.sum() - 1.0) < 1e-12)
        ok("tilt quadrature has unit variance (n=%d)" % n,
           abs(float(w @ (x * x)) - 1.0) < 1e-10, "%.3e" % float(w @ (x * x)))

    if verbose:
        for st, nm, dt in msgs:
            if st == "FAIL" or dt:
                print("  %s  %-52s %s" % (st, nm, dt))
        print("gid2d_core.selfcheck: %d assertions, all pass" % len(msgs))
    return len(msgs)


if __name__ == "__main__":
    selfcheck()
    print()
    for nm in PRESETS:
        L = preset(nm)
        print("%-9s A/molecule = %7.3f   a = %.4f  b = %.4f"
              % (nm, L.area_per_molecule(), np.linalg.norm(L.A1), np.linalg.norm(L.A2)))
        for R in L.reflections(six=False):
            h, k = R["hk"]
            sh = L.sheet(h, k)
            print("          (%2d,%2d)  qxy=%.4f  qz=%+.4f  chi=%6.2f  d_sheet=%.4f"
                  % (h, k, R["qxy"], R["qz"], np.degrees(sh["chi"]), sh["d_sheet"]))
