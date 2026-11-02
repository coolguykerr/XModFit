"""Re-apply the XModFit divergences on top of an upstream XPR.jl.

    python julia/merge_upstream.py [path/to/XPR-main/XPR/src/XPR.jl]

Reads the upstream file, applies the patches below, and writes
``julia/XPR/src/XPR.jl``. Default source is ``<repo>/../XPR-main/XPR/src/XPR.jl``,
or ``$XPR_UPSTREAM_JL``.

Why a script and not a hand-edit: the vendored copy carries a handful of divergences
from upstream (documented in the header this script writes, and in
``handoff/05-xpr-julia.md`` 3.2). Every upstream drop has to have them re-applied,
and doing that by hand is how one of them quietly goes missing. Each substitution
here is anchored on exact upstream text and asserted to fire exactly once, so an
upstream edit that moves an anchor fails loudly instead of dropping a patch.

Trailing whitespace is stripped line by line before matching, so the anchors need
not reproduce upstream's stray spaces; since this script normalises every revision
the same way, the vendored-to-vendored diff stays clean.

BASELINE: the 2026-09-10 upstream, which took four pieces of our feedback --
``spacez`` and ``min_box_length_sum`` are now real keyword arguments, the protein
is bounds-checked with a hard ``error()``, and ``XPRsim._import_pdb`` gives the CAL
ion calcium's mass. Two of our former divergences became upstream features and were
retired here: the ``max_box_length``/``max_sig`` slab-sizing keywords (upstream now
has ``max_box_length``/``sig_max``/``spacez``/``d_protein_bounds``/
``min_box_length_sum``) and ``slab_headroom`` (obsolete -- the fixed-slab-count
design cannot overrun a buffer, so there is nothing to buy headroom for).

AFTER RUNNING: `python validate_xpr.py` is the gate. It pins the published
demo.ipynb log-likelihoods AND the equivalence of the two insertion conventions,
which is what catches a physics change dressed as a refactor.
"""
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

UP = (sys.argv[1] if len(sys.argv) > 1 else
      os.environ.get('XPR_UPSTREAM_JL') or
      os.path.join(_REPO, os.pardir, 'XPR-main', 'XPR', 'src', 'XPR.jl'))
OUT = os.path.join(_HERE, 'XPR', 'src', 'XPR.jl')
if not os.path.isfile(UP):
    sys.exit('no upstream XPR.jl at %s -- pass its path as an argument' % UP)

src = io.open(UP, encoding='utf-8', newline='').read().replace('\r\n', '\n')
# Strip trailing whitespace line by line so the anchors below need not reproduce
# upstream's stray spaces. This script IS the merge tool, so re-running it on a
# future upstream revision normalises that revision the same way and the
# vendored-to-vendored diff stays clean.
src = '\n'.join(l.rstrip() for l in src.split('\n'))


def sub(old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, 'anchor found %d times (want %d):\n%s' % (c, n, old[:400])
    src = src.replace(old, new)


# ---------------------------------------------------------------- P1: header note
sub('''Implementation is optimized for memory and speed while maintaining compatibility with autodifferentation capabilities.
"""
''',
'''Implementation is optimized for memory and speed while maintaining compatibility with autodifferentation capabilities.

--------------------------------------------------------------------------------
XModFit vendored copy. Upstream: XPR-main/XPR (Elias Boshara).

Baseline: the 2026-09-10 upstream revision. Recent upstream history that matters
when reading this file:

  2026-09-08  `d_protein` redefined from the TOP of the protein to its MASS
              CENTROID, and `smear_xy_interface!` rebuilt onto one lattice, both
              for smoothness. `load_data` gained `masses_py`; `rotate_protein!`
              returns `protein_height`, the old top-referenced insertion.
  2026-09-09  Fixed slab COUNT with parameter-dependent spacing: `box_cells[b]`
              cells per box at spacing `d_b/box_cells[b]`, rigid `spacez` above
              z=0 / below the stack / in the Gaussian tails. Removed the last
              non-smoothness (a kink when a cumulative box length landed on a
              lattice node) and, as a side effect, the segfault class -- every
              loop bound is now a load-time constant.
  2026-09-10  `spacez` and `min_box_length_sum` exposed as keyword arguments; the
              protein is bounds-checked in `XPR!` with a hard `error()`.

Divergence 1: two optional keyword arguments on `load_data`/`prealloc` that
upstream still hard-codes, each defaulting to the upstream literal so behaviour is
unchanged unless a caller opts in:

    xray_energy     10.0    keV; upstream hard-codes 10.0 inside prealloc
    fres            true    Fresnel normalisation; upstream hard-codes true

`xray_energy` is inert today: XPR allocates its absorption array and never writes
to it, so beta == 0 and the wavelength only ever appears multiplied by beta. It is
wired up so `E` behaves correctly the day absorption is added.

Divergence 2: four new exported functions. `XPR_edp(params)` returns the smeared
electron-density profile so XModFit can plot an EDP like its other XRR models;
`XPR_sim_edp(params)` returns the reflectivity, the protein height and that profile
from a single `XPR!` call (asking for them separately would evaluate the model
twice); `XPR_jac(params, active)` gives dR_i/dp_j by forward-mode AD over a chosen
subset of the parameters, so least-squares fitting need not finite-difference it;
and `XPR_maxh(theta, phi)` returns the centroid-to-top offset alone, so the
old-style insertion can be reported without running the model.

Divergence 3: `load_data(...; param_lower, param_upper)` optionally reparameterises
the model onto the unit cube [0,1]^(8+2n). The mapping happens inside `XPR!`, so it
is part of the differentiated computation and XPR_grad/XPR_hess/XPR_jac come back in
NORMALISED coordinates with no chain rule on the caller's side -- which is what a
bounded global search wants, and what keeps a ~1e13 condition number tractable. Omit
the bounds and everything behaves exactly as before, in physical units.

Divergence 4: the O(N^2) union-of-circles workspace is sized to 1 unless
`vol_calc="exact"` actually selects the code that reads it (see prealloc).

RETIRED divergences, now upstream features -- do not re-add:
  * `max_box_length` / `max_sig` slab-sizing keywords -> upstream `max_box_length`,
    `sig_max`, `spacez`, `d_protein_bounds`, `min_box_length_sum`.
  * `slab_headroom` -> obsolete. The slab count no longer depends on the
    parameters, so no proposal can overrun a preallocated array.
--------------------------------------------------------------------------------
"""
''')

# ------------------------------------------------------- P2: SimData + to_physical
sub('''    num_boxes::Int
    vol_calc::String
end
''',
'''    num_boxes::Int
    vol_calc::String
    # Optional [0,1]^n reparameterisation (XModFit patch). When `normalized`, every
    # entry point takes a unit-cube vector and maps it to physical units INSIDE the
    # AD tape, so XPR_grad/XPR_hess/XPR_jac come back in normalised coordinates with
    # no chain rule on the caller's side. Bounds are supplied at load time, so the
    # same package serves both an n-box XModFit fit and a bounded global search.
    normalized::Bool
    p_lo::Vector{Float64}
    p_scale::Vector{Float64}
end

# physical parameter vector from whatever the caller passed
@inline function to_physical(simdata, params)
    return simdata.normalized ? (simdata.p_scale .* params .+ simdata.p_lo) : params
end
''')

# ------------------------------------------------------------------ P3: load_data
sub('''    sig_max=5.0, spacez=0.5, min_box_length_sum=0.0)''',
'''    sig_max=5.0, spacez=0.5, min_box_length_sum=0.0,
    xray_energy=10.0, fres=true, param_lower=nothing, param_upper=nothing)''')

sub('''                                    vol_calc, rho_top, rho_bottom, max_box_length, d_protein_bounds,
                                    sig_max, spacez, min_box_length_sum)
''',
'''                                    vol_calc, rho_top, rho_bottom, max_box_length, d_protein_bounds,
                                    sig_max, spacez, min_box_length_sum;
                                    xray_energy=xray_energy, fres=fres,
                                    param_lower=param_lower, param_upper=param_upper)
''')

# -------------------------------------------------------------------- P4: prealloc
sub('''                    sig_max, spacez, min_box_length_sum)

    N = include_protein ? length(radii) : 1''',
'''                    sig_max, spacez, min_box_length_sum;
                    xray_energy=10.0, fres=true,
                    param_lower=nothing, param_upper=nothing)

    # normalisation bounds, if the caller supplied them
    nparam = 8 + 2 * num_boxes
    if param_lower === nothing || param_upper === nothing
        normalized = false
        p_lo = zeros(nparam)
        p_scale = ones(nparam)
    else
        p_lo = Vector{Float64}(Array(param_lower))
        p_hi = Vector{Float64}(Array(param_upper))
        length(p_lo) == nparam || error("param_lower must have 8+2*num_boxes = $nparam entries")
        length(p_hi) == nparam || error("param_upper must have 8+2*num_boxes = $nparam entries")
        any(p_hi .<= p_lo) && error("param_upper must exceed param_lower elementwise")
        normalized = true
        p_scale = p_hi .- p_lo
    end

    N = include_protein ? length(radii) : 1''')

sub('''    xray_energy  = 10.0                                                                      # energy of xrays, in inverse units of x
    λ''',
'''    # energy of xrays (keV), in inverse units of x -- XModFit patch: caller-settable
    λ''')

sub('''    # include Fresnel normalization
    fres = true

    max_sd_away = 10''',
'''    # include Fresnel normalization -- XModFit patch: caller-settable (was hard-coded true)

    max_sd_away = 10''')

sub('''        cell_size = 2.0 * max_r
        max_b = trunc(Int, (sq_max_l / cell_size) + 1)''',
'''        cell_size = 2.0 * max_r
        max_b = (vol_calc == "exact") ? trunc(Int, (sq_max_l / cell_size) + 1) : 1''')

sub('''        num_boxes,
        vol_calc
    )
''',
'''        num_boxes,
        vol_calc,
        normalized,
        p_lo,
        p_scale
    )
''')

sub('''    # *** decreasing any of the below three values runs the risk of seg fault *** #
    cell_max = simdata.N                                        # number of circles per cell in mesh grid [worst case = N]
    max_intersect = Int(simdata.N * (simdata.N - 1) / 2)        # total number of intersections [worst case = N(N-1)/2]
    max_neighbor = simdata.N                                    # total number of atoms intersecting a given atom [worst case = N]''',
'''    # *** decreasing any of the below three values runs the risk of seg fault *** #
    # XModFit patch: only the vol_calc="exact" path (greens! and its callees) ever
    # touches these. They are O(N^2): for a 1815-atom protein max_intersect is 1.6e6,
    # so TempVars.int_pts is 6.6e6 entries -- 53 MB at Float64, but TempVars is
    # PARAMETERISED BY T, so the nested duals ForwardDiff.hessian uses carry ~169
    # components each and that one array alone reaches ~9 GB. Under "approx" it is
    # allocated and never read, which made the Hessian ruinously expensive for no
    # reason. Size them to nothing unless the exact path is actually selected.
    exact_path = (vol_calc == "exact")
    cell_max = exact_path ? simdata.N : 1                       # circles per mesh cell [worst case = N]
    max_intersect = exact_path ? Int(simdata.N * (simdata.N - 1) / 2) : 1   # [worst case = N(N-1)/2]
    max_neighbor = exact_path ? simdata.N : 1                   # atoms intersecting a given atom [worst case = N]''')

sub('''    adjacency = -ones(Int, simdata.N, max_neighbor)             # tracks which circles are adjacent to each other
''',
'''    adjacency = -ones(Int, exact_path ? simdata.N : 1, max_neighbor)  # which circles are adjacent
''')

sub('''    circ_to_pt_idx = zeros(Int, simdata.N, 2*max_neighbor)      # maps circles to intersection points list
''',
'''    circ_to_pt_idx = zeros(Int, exact_path ? simdata.N : 1, 2*max_neighbor)  # circles -> points
''')

# ------------------------------------------------------------------------ P5: XPR!
sub('''    # get memory structs
    T = eltype(params)
    simdata = GLOBAL_SD[]
    areavars = GLOBAL_AV[]
    tempvars = get_tempvars(T, simdata, areavars)
''',
'''    # get memory structs
    simdata = GLOBAL_SD[]
    areavars = GLOBAL_AV[]

    # XModFit patch: map the unit cube to physical units when the caller loaded
    # bounds. Done HERE so it is part of the differentiated computation.
    params = to_physical(simdata, params)

    T = eltype(params)
    tempvars = get_tempvars(T, simdata, areavars)
''')

# ----------------------------------------- P6: XPR_edp / XPR_sim_edp / XPR_jac / maxh
sub('''####### exported functions #######

export load_data
export XPR_sim_ref
export XPR_grad
export XPR_hess

end''',
'''# XModFit patch: the smeared electron-density profile behind the reflectivity.
# XPR! leaves rho in tempvars but not the count B that says how much of it is live,
# so the two cheap trailing steps (smearing lattice + erf smear) are re-run to
# recover it. Everything expensive (protein rotation, slice areas, per-slab
# electron densities) is already in tempvars from the XPR! call and is reused.
function XPR_edp(params)
    params = Array(params)
    simdata = GLOBAL_SD[]
    areavars = GLOBAL_AV[]

    XPR!(params)                                   # fills tempvars for these params
    params = to_physical(simdata, params)          # the same mapping XPR! applied
    T = eltype(params)
    tempvars = get_tempvars(T, simdata, areavars)

    C           = params[5]
    sig         = simdata.fit_sig ? params[6] : 3.4
    nbox        = simdata.num_boxes
    box_lengths = params[9:nbox+8]
    box_densities = params[nbox+9:2*nbox+8]

    M = get_slab_boundaries!(simdata, tempvars, box_lengths)
    B = smear_xy_interface!(simdata, tempvars, box_densities, C, M)
    eval_smeared!(tempvars, B, sig)

    return collect(view(tempvars.w_mid, 1:(B-1))), collect(view(tempvars.ρ, 1:(B-1)))
end

# XModFit patch: reflectivity AND profile from a SINGLE XPR! call.
# Calling XPR_sim_ref and then XPR_edp runs the whole forward model twice -- the
# protein rotation, slice areas and per-slab electron densities included -- which
# doubles the cost of every likelihood evaluation in a fit or an MCMC chain. This
# runs XPR! once and re-runs only the two cheap trailing steps to recover B.
function XPR_sim_edp(params)
    params = Array(params)
    simdata = GLOBAL_SD[]
    areavars = GLOBAL_AV[]

    logL, R, prot_h = XPR!(params)
    Rout = copy(R)
    params = to_physical(simdata, params)          # the same mapping XPR! applied
    T = eltype(params)
    tempvars = get_tempvars(T, simdata, areavars)

    C             = params[5]
    sig           = simdata.fit_sig ? params[6] : 3.4
    nbox          = simdata.num_boxes
    box_lengths   = params[9:nbox+8]
    box_densities = params[nbox+9:2*nbox+8]

    M = get_slab_boundaries!(simdata, tempvars, box_lengths)
    B = smear_xy_interface!(simdata, tempvars, box_densities, C, M)
    eval_smeared!(tempvars, B, sig)

    return logL, Rout, prot_h, collect(view(tempvars.w_mid, 1:(B-1))),
           collect(view(tempvars.ρ, 1:(B-1)))
end

# XModFit patch: Jacobian of the REFLECTIVITY (not the log-likelihood) with respect
# to a subset of the parameters, dR_i/dp_j, for least-squares fitting.
# `active` is a 1-based index vector selecting which parameters to differentiate;
# ForwardDiff seeds only those, so the cost scales with how many are actually
# varying rather than with the full 8+2n vector.
const JAC_CHUNK = 4

@inline function _jac_chunked(f, x0, ::Val{K}) where {K}
    return ForwardDiff.jacobian(f, x0,
        ForwardDiff.JacobianConfig(f, x0, ForwardDiff.Chunk{K}()))
end

function XPR_jac(params, active)
    p0 = Array(params)
    idx = Int.(Array(active))
    function f(u)
        q = similar(u, length(p0))
        @inbounds for i in eachindex(p0)
            q[i] = p0[i]
        end
        @inbounds for (k, i) in enumerate(idx)
            q[i] = u[k]
        end
        return XPR!(q)[2]
    end
    # Fixed chunk size: ForwardDiff specialises its Dual type on the chunk, and the
    # default chunk is chosen from length(x). Letting it float means Julia
    # recompiles the whole model (~3 s) every time the SET of varying parameters
    # changes -- which in a GUI is constantly. Pinning it costs a little on very
    # wide Jacobians and buys exactly one compilation.
    x0 = p0[idx]
    # chunk cannot exceed the number of variables, so this caps at JAC_CHUNK and
    # bounds the number of distinct Dual specialisations to JAC_CHUNK, not to the
    # number of parameter subsets a user might fit.
    k = min(JAC_CHUNK, length(x0))
    return _jac_chunked(f, x0, Val(k))
end

# XModFit patch: the orientation-dependent offset between the two definitions of the
# insertion parameter, WITHOUT running the forward model.
#
# Upstream measures `d_protein` from the centre of mass; the previous convention
# measured it from the TOP of the protein, and `rotate_protein!` reports the
# conversion as `protein_height = d_protein + maxh`. `maxh` depends only on
# (theta, phi) and the structure, so exposing it on its own lets XModFit report the
# old-style insertion -- and propagate its error, and convert a whole MCMC chain --
# without one forward evaluation per sample. `mindepth` is the matching extent below
# the centre of mass. `k` is the index of the atom that attains the maximum: `maxh`
# is a `maximum`, so it is only piecewise smooth in (theta, phi), and a caller
# differentiating it needs to know which term is active.
#
# NB `prot_extent` in SimData is the rotation-INVARIANT bounding radius, which is
# what upstream's out-of-bounds check uses; `maxh`/`mindepth` here are the actual
# oriented extents and are <= it. Python mirrors the check with prot_extent so it
# agrees with Julia exactly.
function XPR_maxh(theta_deg, phi_deg)
    simdata = GLOBAL_SD[]
    simdata.include_protein || return (0.0, 0.0, 0)
    theta = theta_deg * pi / 180
    phi = phi_deg * pi / 180
    r3 = [sin(theta)*sin(phi), sin(theta)*cos(phi), cos(theta)]
    proj = vec(r3' * simdata.coordinates')
    hi = proj .+ simdata.radii
    lo = proj .- simdata.radii
    k = argmax(hi)
    return (hi[k], -minimum(lo), k)
end

####### exported functions #######

export load_data
export XPR_sim_ref
export XPR_grad
export XPR_hess
export XPR_edp
export XPR_sim_edp
export XPR_jac
export XPR_maxh

end''')

assert src.count('to_physical') == 4, src.count('to_physical')
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(src)
print('wrote %s (%d lines)' % (OUT, src.count('\n') + 1))
