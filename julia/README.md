# Julia backends for XModFit

## XPR

`XPR/` is a vendored copy of the Julia reflectivity package from
[`XPR-main`](../../XPR-main) (Elias Boshara): a smoothed, everywhere-differentiable
reimplementation of the XLayersPDB physics — an n-box film plus an optional rigid
protein read from a PDB — whose log-likelihood carries exact ForwardDiff gradients
and Hessians.

XModFit reaches it through:

| file | role |
| --- | --- |
| `Functions/xpr_backend.py` | one Julia session per process, load fingerprinting, slab-budget guard, `sim`/`grad`/`hess`/`edp` |
| `Functions/XRR/XLayersPDB_XPR.py` | the XModFit model class (lmfit parameters, `y()`, EDP output) |

### Setup

```bash
python -m pip install -r requirements_xpr.txt
```

That is the only pip step. The first `import juliacall` then downloads a private
Julia 1.11.9 (a 262 MB zip, ~1 GB installed), and the first `using XPR` resolves and
precompiles ForwardDiff + SpecialFunctions from `XPR/Manifest.toml`. Budget
10–20 minutes for that once; afterwards a session starts in ~6 s and a warm model
evaluation takes 2–4 ms.

Verified with Julia 1.11.9 and juliacall 0.9.28 under Python 3.9 in both `cmc-py39`
and `cmcars`.

### Where the 1 GB goes — one shared copy, handled for you

Left to itself, juliapkg picks the Julia install directory like this
(`juliapkg/state.py`):

| condition | project directory |
| --- | --- |
| `$PYTHON_JULIAPKG_PROJECT` set | that path |
| else in a venv, or `$CONDA_PREFIX` set | `<env>/julia_env` |
| else | `<depot>/environments/pyjuliapkg` |

The middle row is a trap: it keys on `CONDA_PREFIX`, so **every activated conda
environment would get its own ~1 GB Julia install and its own precompile** — and
running the same interpreter by absolute path (no activation) takes the third row
instead, so the two disagree about whether anything is installed at all.

`xpr_backend.get_jl()` therefore defaults `PYTHON_JULIAPKG_PROJECT` to the third
row, `~/.julia/environments/pyjuliapkg` (honouring `$JULIA_DEPOT_PATH`), so one copy
is shared by every environment. Set the variable yourself, or pass
`-X juliapkg-project`, to override.

Keep `juliacall` at the same version in every environment that shares a project —
juliapkg pins PythonCall to an exact version in the shared `Project.toml`, so a
mismatch makes them re-resolve against each other on every startup. That is what
the pins in `requirements_xpr.txt` are for.

### If juliapkg cannot reach the network

juliapkg queries `julialang-s3.julialang.org/bin/versions.json` on **every** startup
to check the Julia version, over `urllib` — i.e. the stdlib SSL default context. In
an environment where that context is broken, startup fails even though Julia is
already installed.

`get_jl()` handles this: if the first `import juliacall` fails and a Julia
executable is already present in the configured project, it sets
`PYTHON_JULIAPKG_OFFLINE=yes` and retries once. That is safe because juliacall
fails inside `juliapkg.executable()`, before `libjulia` is loaded. If the retry also
fails, the original error is reported.

Known case: `cmcars` ships OpenSSL 3.5.7, which rejects a malformed certificate in
the Windows certificate store with `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`, while
`cmc-py39` on OpenSSL 3.0.13 tolerates the same store. `SSL_CERT_FILE` does **not**
help, because `load_default_certs` enumerates the Windows store before consulting
it. Note the consequence of the offline fallback: juliapkg cannot install or upgrade
anything in that environment, so do Julia-side updates (new packages, a Julia
version bump) from an environment whose SSL works.

### Where the project is found

`Functions/xpr_backend.julia_project_dir()` searches, in order:

1. `$XPR_JULIA_PROJECT`
2. `<repo>/julia/XPR` — this directory
3. `<repo>/../XPR-main/XPR` — a sibling working copy

So set `XPR_JULIA_PROJECT` to develop against your own checkout without editing
anything.

### Divergence from upstream `XPR.jl`

Two changes, both marked in the source:

1. **Optional keywords on `load_data`/`prealloc`**, every one defaulting to the
   upstream hard-coded value: `xray_energy` (was a literal 10.0 keV),
   `fres` (was a literal `true`), and the three that size the preallocated slab
   arrays — `max_box_length` (was a literal 30 Å per box), `max_sig` (was a
   literal 5 Å) and `slab_headroom` (new, default 0).

   The sizing ones matter because overrunning those arrays is a segfault, not an
   exception, and a segfault inside the GUI takes XModFit down with it.
   `xpr_backend` sizes them from the live parameters and re-checks every call.

   `xray_energy` is currently **inert**: XPR allocates its absorption array and
   never writes to it, so `beta == 0` and the wavelength only ever appears
   multiplied by `beta` in the Parratt recursion. The keyword is wired up so that
   `E` behaves correctly the day absorption is added.

2. **`XPR_edp(params)`** and **`XPR_sim_edp(params)`**, new exported functions
   returning the smeared electron-density profile — the second alongside the
   reflectivity, from a single `XPR!` call. Asking for `XPR_sim_ref` and `XPR_edp`
   separately evaluates the whole forward model twice, which doubles the cost of
   every likelihood evaluation; both re-run only the two cheap trailing steps to
   recover the live slab count.

3. **`XPR_jac(params, active)`**, the Jacobian `dR_i/dp_j` by forward-mode AD over a
   chosen subset of the parameters, so a least-squares fit does not have to
   finite-difference it. The ForwardDiff chunk is pinned (`JAC_CHUNK = 4`) because
   the default is derived from `length(x)`, and letting it float makes Julia
   recompile the whole model (~3 s) every time the set of varying parameters
   changes.

### Analytic Jacobian

`Fit_Routines` hands it to `leastsq` automatically when the model can supply one and
at least 4 parameters vary (`fit.use_jacobian = False` opts out). Measured on the
1697-atom test protein, 5 varying parameters: **261 model calls -> 41**, same optimum
to 1.3e-5. Per iteration it replaces `nvarys+1` evaluations with one evaluation plus
one AD pass, worth 1.1-1.6x depending on how many parameters vary.

Two things it is deliberately NOT used for:

* **`least_squares`** — lmfit wraps `Dfun` in `Minimizer.__jacobian`, which applies
  the bounded-to-internal gradient rescaling; the `least_squares` path takes its own
  `jac` with a different signature and no such wrapper.
* **The emcee seed scan** — those sub-fits profile only the nuisance parameters
  (usually 2), where 3 finite-difference evaluations already beat one AD pass, and in
  the parallel path every worker would additionally pay the ~3 s specialisation.
  Measured: 30 s -> 34 s serial, 51 s -> 74 s at 16 workers. Hence the `nvarys >= 4`
  gate in `Fit.can_jac`.

### Grid-alignment kinks

XPR lays its profile on a fixed `spacez = 0.5 A` lattice, so R has a small slope
discontinuity wherever a layer boundary or the smearing extent lands exactly on a
node — `sum(d)` a multiple of 0.5, or `10*sig` likewise. The AD derivative is the
correct one-sided one; central differences straddle the kink and disagree by ~1e-2.
Round-number defaults (`d = [11.0, 9.0]`, `sig = 3.4`) sit exactly on them, which is
worth knowing before trusting a finite-difference cross-check — or before running
gradient-based MCMC in stage 3.

### Why the first MCMC step is slow

Three fixed costs, in order, none of them the sampler:

| | cost | when |
| --- | --- | --- |
| Julia session start | ~7 s | once per process |
| ForwardDiff/XPR JIT on the first `y()` | ~8 s | once per process |
| (theta, phi) orientation grid seed scan | ~29 s | once per emcee run, if theta and phi both vary |

The scan is XModFit's own walker seeding for every `XLayersPDB*` model, not
something XPR adds: 560 grid cells, each profiling the nuisance parameters with an
independent `least_squares`, so ~14 700 model evaluations before sampling starts.
Sampling itself then runs at ~2 ms per evaluation.

To skip the scan, set `theta` and `phi` to `vary=0` **before** the least-squares
fit, not after: `Fit.set_emcee_minimizer` copies its parameters from
`self.result.params`, so a `vary` flag changed after the fit has no effect on the
chain.
