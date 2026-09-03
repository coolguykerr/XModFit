<!-- Template for handoff/NN-<topic>.md. Keep the section order and headings. -->

# PDB-in-layers XRR models (XLayersPDB family + layer models) — XModJointFit change record

**Category:** Specular X-ray reflectivity from a rigid protein template folded into a layered
interface (`XLayersPDB` and its four siblings), plus two plain layer models added alongside
(`XLayers_Beamfootprint`, a 2-char tweak to `XLayers_Biphasic`), and the shared PDB read cache.
**Status:** Mixed. `XLayersPDB` = Production. `XLayersPDB_dev_smooth` = Working-but-unvalidated-live
(engine-tested, heavily self-documented, never clicked through the GUI as far as the notes say).
`XLayersPDB_G` / `_H` / `_box` = Experimental, and all three carry the same latent `NameError`
(section 6). `XLayers_Beamfootprint` = Experimental/one-off.
**Owner chat(s):** `functions-category-folder-rule.md`, `pdb-mcmc-seeding.md`,
`merge-with-upstream-develop.md`; XPR relationship from `xpr-julia-backend.md`.
**Last touched:** committed content 2025-09-08 (`d3e353e`) and 2026-08-05 (`c5f697d`, merge
`a4fd04e`); the *registration* tuples that name these models are uncommitted working-tree edits
(latest, ~2026-08/09).

## 1. Why this exists

Specular XRR from a protein at a lipid interface is not well described by a box stack alone: the
protein has a fixed internal electron-density distribution set by its atomic structure, and the
only genuinely free structural unknowns are its *pose* and *how much of the surface it covers*.
`XLayersPDB` therefore takes a PDB file as a rigid template, rotates it by two Euler angles
(`theta`, `phi`, `'zx'` convention), slides it into the stack by an insertion depth (`protins`),
and superposes its z-projected electron density onto the layer profile weighted by a coverage
fraction (`cov`). The layer stack itself keeps the usual XModFit `XLayers` meaning
(`d`/`rho`/`mu`/`sig` per layer, `mpar['Model']`), so a Langmuir monolayer is `air / tail / head /
buffer` and a supported bilayer adds substrate/oxide/water layers. The systems this was built for
are the TIM-family membrane receptors at lipid monolayers — the shipped default PDB is
`./Data/tim3_hmmm_ext_all_bestframe.pdb`, and `Data/` also carries hTIM3, hTIM4, TIM1 and IBAR
templates. The result is a 4-parameter pose/coverage description plus a small layer model, which is
what makes joint fitting across a coverage or pressure series worthwhile.

The four sibling variants all attack one weakness of the original: the *lateral area* used to
normalise the protein's electron density and its excluded-volume fraction was computed by voxelising
the rotated coordinates, which makes the model non-smooth in `theta`/`phi`/`protins` and therefore
hostile to derivative-based fitting. `XLayers_Beamfootprint` is unrelated physics — the low-angle
over-illumination (beam wider than the sample) correction, forward-modelled inside the model instead
of pre-corrected out of the data.

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `Functions/XRR/XLayersPDB.py` | 525 | modified | The production model. Our changes: shared `pdbread`, cache sizes, one CSE, merge resolution. |
| `Functions/pdb_cache.py` | 92 | new (`c5f697d`) | Module-level, input-keyed PDB parse + mendeleev element table, shared across instances. |
| `Functions/XRR/XLayersPDB_box.py` | 557 | new (`d3e353e`) | Membrane-conformal z grid + interface interpolation; covariance-ellipse lateral area. |
| `Functions/XRR/XLayersPDB_G.py` | 750 | new (`d3e353e`) | Exact union-of-discs areas by Green's theorem, per z slab. Slowest, most exact. |
| `Functions/XRR/XLayersPDB_H.py` | 850 | new (`d3e353e`) | Hybrid: Green's-theorem footprint area, analytic sphere-cap z profile. |
| `Functions/XRR/XLayersPDB_dev_smooth.py` | 958 | new (`c5f697d`) | The smoothness rewrite. Self-contained, no `pdb_cache`, extensive measured docstring. |
| `Functions/XRR/XLayers_Beamfootprint.py` | 223 | new (`c5f697d`) | `XLayers` + an erf beam-footprint factor (`fwhm`, `sub_wth`). No `rrf`. |
| `Functions/XRR/XLayers_Biphasic.py` | 280 | modified | 2-char change: unused `sumd=None` kwarg on `calcProfile`. |
| `Functions/XRR/XLayersPDB_XPR.py` | 584 | new-untracked | **Out of scope** — see `handoff/05-xpr-julia.md`. Named here only because the registration tuples list it. |

State legend: `new` = added in a local commit; `modified` = upstream file we changed;
`new-untracked` = exists only in the working tree (not committed anywhere).

Registration/special-casing sites (all **uncommitted working-tree edits** — `HEAD` still has the
older or-chains): `lmfit/minimizer.py:94`, `xmodfit.py:2634`, `Fit_Routines.py:491`. See section 5.

## 3. How it works

### 3.1 `XLayersPDB` main path

`Fit.__init__` sets `self.params = self.func.__dict__` and `self.fit_params = self.func.params`
(`Fit_Routines.py:49,51`), so the fitter writes each varying value to *both* the model attribute and
the model's own lmfit `Parameters`. `Fit.evaluate` calls `func.y()`.

`XLayersPDB.y()` (`Functions/XRR/XLayersPDB.py:463`):

1. Fold the angles into the fundamental domain (`:469-475`): `phi %= 360`, and if `theta % 360 >
   180` replace with `360-theta` and `phi += 180`. This writes to `self.params[...]`, not to the
   `self.theta`/`self.phi` attributes used two lines later — see audit hook #6.
2. `pdbread(self.fname)` (`:488`) — the module-level shared cache.
3. `protvolcalc(...)` (`:489`): voxelise the *unrotated* vdW spheres on an `fslice` cubic lattice and
   return the union volume `protvol` (`:241-277`).
4. `ED_profrot(...)` (`:490`, body `:280-339`): rotate by `R.from_euler('zx', [[phi, theta]])`;
   voxelise a 2-D `Area_grid` to get the projected footprint `minareagrid` (`:330-332`); accumulate
   per-slab electrons and segment volumes with the analytic sphere/slab cap formula
   `SphereElecFrac` (`:78-104`). Returns `protpos` (slab centres, descending), `protrho` = electrons
   per slab / (footprint volume), and `layarea` = the fraction of each slab *not* occupied by
   protein (`:336`), with the sphere-overlap double count removed globally by rescaling the segment
   volumes so they sum to `protvol`.
5. `calcProfComb(...)` (`:342-391`): merge the protein slab edges and the membrane interfaces into
   one node list (`np.unique`, `:368`), then `protmembcomb` (numba, `:37-65`) walks the merged list
   with two pointers and, inside the protein span, mixes
   `rho = cov*protrho[l] + (1 + cov*(layarea[l]-1))*rho[k]` (`:59`) — i.e. the protein contributes
   `cov` of its own density and displaces `cov*(1-layarea)` of the layer. Roughness/absorption come
   from the containing membrane layer. The sharp profile is then erf-smoothed once
   (`sldCalFun`/`sldCalFun_numba`) onto a `dz` lattice.
6. Parratt recursion via `py_parratt` -> `parratt_numba`, imported from `XRR.XLayers` (`:27`).
   Fresnel normalisation if `rrf`.

### 3.2 What we changed in `XLayersPDB` vs upstream

Three things, all in the committed bucket, all confirmed by `git diff origin/develop...HEAD`:

* **`pdbread` extracted to `Functions/pdb_cache.py`** (`c5f697d`). The method version was
  `@lru_cache(maxsize=2)` on an *instance method*, so the cache key included `self`: two model
  objects — e.g. two datasets in a joint fit, or a `deepcopy` — never shared a parse of the same
  file, and >2 live instances thrashed, re-running `Bio.PDB.PDBParser` *and* the mendeleev
  `fetch_table` on every dataset switch. The replacement is keyed on `fname` alone
  (`pdb_cache.py:55-56`) with the element table hoisted to its own `maxsize=1` cache
  (`pdb_cache.py:41-52`). Call sites: `XLayersPDB.py:210` (`__init__`) and `:488` (`y`).
* **`lru_cache` sizes 2/3 -> 16** on `protvolcalc`, `ED_profrot`, `calcProfComb`
  (`:240`, `:279`, `:341`) so a multi-dataset joint fit does not evict on every dataset.
* **One common-subexpression extraction** in `SphereElecFrac` (`:93-97`, from `d3e353e`): the sphere
  segment volume is computed once instead of twice. Algebraically identical; upstream's inline form
  is kept as a string literal at `:99-102`.

**Merge resolution (`a4fd04e`).** `XLayersPDB.py` was one of the four conflict files. Resolution:
take origin's `from XRR.XLayers import parratt_numba` (`:27`) instead of keeping a private copy, and
keep our `pdb_cache` import (`:31-34`). The now-unused `prange` import was dropped. Consequence
worth knowing: `XLayersPDB` gets upstream's optimised shared Parratt kernel, while *every* variant
below still carries its own stale inline copy.

`Functions/XRR/XLayers_Biphasic.py`: the only surviving local change is an unused
`sumd=None` keyword inserted into `calcProfile`'s signature (`:88`). Origin's simplification was
taken; this 2-char remnant is pre-existing WIP.

### 3.3 Why `pdb_cache.py` sits in `Functions/` and not `Functions/XRR/`

Hard constraint, documented in the module's own docstring (`pdb_cache.py:16-19`) and in
`functions-category-folder-rule.md`. `xmodfit.py`'s `update_functions` treats every subdirectory of
`./Functions/` as a *category* and every `.py` in the selected category folder (bar `__init__.py`)
as a fittable model, then does `getattr(module, <filename>)` expecting a class named exactly after
the file. A helper dropped into `Functions/XRR/` would be listed as a function and crash the list
with `AttributeError`. `Functions/` root is safe because categories must be directories. The same
rule is restated in `Functions/gid2d_core.py:7` and `Functions/xpr_backend.py:22`. The import in
each model is wrapped `try: from Functions.pdb_cache import pdbread / except: from pdb_cache import
pdbread` (`XLayersPDB.py:31-34`) because `sys.path` gains `./Functions` at module top.

### 3.4 The four variants

All four keep the same public parameter set (`theta`, `phi`, `protins`, `cov`, `qoff`, `yscale`,
`bkg`, plus `mpar` d/rho/mu/sig) so a `.par` file is nominally interchangeable. They differ only in
how the protein is discretised and how the lateral area is measured.

**`XLayersPDB_box`** (557 lines). Two real changes.
(a) *Membrane-conformal z grid*: `ED_profrot` gains `protins` and `d` arguments (`:320`) and anchors
the protein slab lattice to the membrane frame (`ztop_grid = -sum(d[:-1]) + protins`, grid nodes on
`fslice` multiples of that frame, `:355-364`), so protein slab edges stop sliding relative to the
interfaces. `calcProfComb` then only merges membrane interfaces that lie *outside* the protein span
(`:410-412`).
(b) *Interface-crossing interpolation*: `protmembcomb` (`:67-111`) detects a slab that straddles a
membrane interface (`flag == 1`) and thickness-weights `rho`, `mu` and `sig` (in quadrature) across
it instead of snapping to one side.
(c) The voxel footprint is replaced by the 95% confidence ellipse of the bivariate normal fitted to
the projected atom centres: `volnorm = 5.991*pi*zslice*sqrt(lam1*lam2)` (`:368-370`). vdW radii are
not used. `Area_gridcalc` (`:171`) is now dead in this file.
`__init__` no longer pre-computes the profile (`:251-254`, commented out) because `ED_profrot` now
needs `d`. Its own inline `parratt_numba` (`:44`).

**`XLayersPDB_G`** (750 lines). Replaces both areas with *exact* union-of-discs geometry:
`find_circle_intersections_grid_n` (`:114`) builds the intersection graph on a uniform bin grid,
`compute_ubp` (`:183`) finds the pairwise intersection points, `process_arcs` (`:220`) keeps the arcs
on the union boundary, and `greens_grid_n` (`:280`) integrates them by Green's theorem. Used twice
per pose: once for the footprint (`xyarea`, `:526`) and once *per z slab* for the cross-sectional
area (`SphereVolFrac`, `:359`, which slices each sphere at the slab centre and takes the union area
of the resulting discs). Consequences: `layarea = 1 - Volfrac/xyarea` is a clean dimensionless area
ratio (`:531-532`); `protvolcalc` is gone entirely (no `protvol`, no global overlap rescale); and it
is by far the most expensive variant, since `greens_grid_n` runs O(N_slabs) times per pose. `G` =
Green. Its `__init__` (`:467`) calls `pdbread` and discards the result — a dead cache-warm.

**`XLayersPDB_H`** (850 lines). The hybrid: `greens_grid_n` for the *footprint* only (`:627`),
`box`'s conformal z grid and `protins`/`d` signature (`:554`), but the *original* analytic
sphere-cap `SphereElecFrac` for the z profile — so `Volfrac` is still a sum of segment *volumes*
(`:629-630`). Keeps `protvolcalc` (`:514`) but never uses `protvol` in `ED_profrot` — the argument is
accepted and ignored. `Vol_gridcalc`/`Area_gridcalc` are retained but dead. `H` = hybrid.

**`XLayersPDB_dev_smooth`** (958 lines). A ground-up smoothness rewrite with a ~180-line docstring
(`:465-632`) recording measurements and rejected alternatives. Distinct in kind:
* *Self-contained* — inlines its own Parratt, erf builder, sphere/slab overlap, area quadrature and
  union-volume quadrature, and does **not** use `Functions/pdb_cache.py`. It has its own template
  cache `_TEMPLATE`/`_read_template` (`:369-408`) keyed on `(abspath, mtime_ns, size)` via `_stamp`
  (`:373`), so an edited-in-place PDB invalidates, plus `_SHIELD` and an `lru_cache`d `_protarea`
  (`:421`). The template is *centred* at read time (`:400`) so that `max_i(|r_i|+rad_i)` is
  rotation-invariant, which is what lets one lab-frame lattice serve every pose.
* *Smooth area* — `soft_area` (`:291`): still a pixel quadrature but on a lattice fixed in the lab
  frame, with a smooth cell-membership weight, and the O(w^2..w^4) smoothing bias removed by
  Richardson-style extrapolation over several kernel widths (`AREA_PRESETS`, `:48-52`;
  `_extrap_weights`, `:284`). `AREA_NPAD = 5.0` (`:53`) is flagged as a smoothness, not a cost, knob.
* *Conformal protein lattice* — `_lattice` (`:722`) subdivides each finite membrane layer into a
  *fixed integer* number of sub-slabs (`_nsub`, `:714`, frozen at construction), so interfaces are
  always exactly slab boundaries. This removes `np.unique`, removes `protmembcomb`'s pointer chase
  entirely, and (per the docstring) removes an off-by-one where a membrane node inside the protein
  span advanced the protein pointer. Price: `fslice` becomes a *target*, drifting if `d` is fitted.
* *Smooth anchor* — `protins` references the topmost atom *surface* through a log-sum-exp surrogate
  (`ztop_soft`, `:441`) instead of a hard `max`; `anchor` ∈ `softtop`/`top`/`centroid`,
  `softbeta` = 10 /A calibrated.
* *Depth-resolved overlap* — `overlap` ∈ `global`/`shielded`/`dipole` (`:755-776`), where `global`
  reproduces `XLayersPDB`'s single `protvol/sum(Vol)` factor and the other two weight each atom by
  its own union share (and, for `dipole`, let that share vary linearly across the atom).
  `atom_shielding` (`:330`) computes the shares at a fixed spacing `VOL_H = 0.125` (`:57`), decoupled
  from `fslice`.
* *`fix_sig` semantics changed* — index 0 (bulk above the film) is tied to `sig[1]` as well
  (`:891-899`), and the default `mpar` gives `sig[0] = 3.0`. Rationale in the docstring's SIGMA
  section; leaving `sig[0]=0` left a visible staircase for a protruding protein.
* *Registration warning* — the docstring states `protins_new = protins_old - fslice`, i.e. historical
  `XLayersPDB` insertion depths are biased deep by about one `fslice`.
* New scalar outputs: `proj_area_Angs2`, `prot_volume_Angs3`, `ztop_apex_Angs`, `nsub`,
  `n_prot_slabs` (`:932-936`).

### 3.5 `XLayers_Beamfootprint`

A fork of an *older* `Functions/XRR/XLayers.py` (file mtime 2026-04-16, i.e. pre-merge) with:
`fwhm` (beam FWHM at the sample, micrometres, default 15) and `sub_wth` (substrate width,
millimetres, default 10) added as fit parameters (`:103-104`); `rrf` removed (the Fresnel branch is
commented out, `:206-211`, and `self.rrf` is not set — only a stale `'rrf'` entry survives in
`self.choices`, `:81`, harmless because it is only consulted for names that are actual parameters);
`bkg`'s upper bound raised from 1 to inf; and the correction itself at `:214`:

```
beam_frac = erf(((1000*sub_wth/2) * lam * x) / (sqrt(2) * 4*pi * (fwhm/2.355)))
refq = beam_frac * refq
```

That is the fraction of a Gaussian beam of sigma = FWHM/2.355 intercepted by a sample of half-width
`sub_wth/2` at incidence `alpha ~ lam*q/(4*pi)` — i.e. `erf((W/2)*sin(alpha)/(sqrt2*sigma))`.
Two earlier attempts (inverse-`q` and flat-profile clipping) are left commented at `:212-213`.

**Its relation to the covariance joint fit.** It is the *alternative* to the pre-correction route,
not part of it. The joint GLS path never touches this model: `Joint_Fit_Routines.py:102-103,143-148,
485-504` reads a per-point footprint sensitivity `v` and a single shared beam-FWHM variance
`var_fwhm` out of an externally corrected bundle (`joint_covariance.load_bundle_npz`,
`joint_covariance.py:133-197`) and adds `var_fwhm * outer(v, v)` as a rank-1 term to the full
covariance (`joint_covariance.py:313-350`). So: *pre-correct the data and propagate the beam-width
error through the covariance* is the joint-fit route; `XLayers_Beamfootprint` is *forward-model the
suppression inside the model with `fwhm` as a fit parameter*. Nothing in the repo wires the two
together, and `XLayers_Beamfootprint` has no PDB/protein support, so it cannot currently be the
model in a protein covariance joint fit.

## 4. How to use it

**GUI.** Functions tab -> category `XRR` -> pick a function. All six PDB-family files plus
`XLayers_Beamfootprint` appear automatically, because every `.py` in `Functions/XRR/` is listed
(section 3.3). Set `fname` to the PDB (it is declared in `self.filepaths`, so the GUI gives it a file
browser), then `E`, `dz`, `fslice`, `rrf`, `fix_sig` as fixed parameters and
`qoff/yscale/bkg/theta/phi/protins/cov` plus the `mpar` table as fitting parameters.

**Parameter meanings.** `theta`, `phi` = degrees, `'zx'` intrinsic Euler order, folded to
`theta ∈ [0,180]`, `phi ∈ [0,360)`. `protins` = insertion depth in A; larger = deeper, referenced to
the *bottom* of the finite stack (`-sum(d[:-1]) + protins`) and, in `XLayersPDB`, to the padded
envelope of the rotated template (hence the one-`fslice` registration offset that `_dev_smooth`
documents). `cov` = surface coverage fraction, `[0,1]`. `d`/`rho`/`mu`/`sig` per layer as in
`XLayers`; row 0 is the top medium with `d=0`, the last row is the bulk below; `fix_sig=True` ties
`sig[2..]` to `sig[1]` (and in `_dev_smooth` also `sig[0]`). `XLayers_Beamfootprint` adds
`fwhm` (um) and `sub_wth` (mm).

**Worked example, in-repo.** `Data/i2_tim3_1000_test_fitparam.par` is a real `XLayersPDB` fit:
4 layers `top/tail/head/bottom`, fit range 0.026-0.550, `fslice=dz=0.5`, `rrf=True`, 9 varying
parameters, converged at `theta=31.93`, `phi=149.64`, `protins=13.996`, `cov=0.1469`,
`d_tail=11.33`, `d_head=8.72`, `rho_tail=0.2338`, `rho_head=0.4623`, `sig=3.4` fixed.
`Data/d6_tim3_1000_test_fitparam.par` is a second one. Both use
`./Data/tim3_hmmm_ext_all_bestframe.pdb`.

**Which variant should a new user pick?**
* `XLayersPDB` for anything you need to compare against past results or run in the joint fit — it is
  the only one referenced by a `.par` file in the repo and the only one in all three registration
  tuples.
* `XLayersPDB_dev_smooth` if you are running a derivative-based fit or orientation seeding and care
  about the pose landscape. Use its recommended `fslice=0.5, dz=0.25`, and remember
  `protins_new = protins_old - fslice` when porting a starting point. Note `pdb-mcmc-seeding.md`
  measured it *not* helping MCMC mixing (section 7, claim #1).
* `_G`/`_H`/`_box`: do not pick these for new work as they stand — see section 6.

**Env.** No env vars. Requires numba, biopython, mendeleev, scipy, lmfit. `_dev_smooth` additionally
declares itself drop-in with no other project files present.

## 5. Contact surface with stock XModFit

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|
| `Functions/XRR/XLayersPDB.py` | `pdbread` method (upstream `:239-267`) | Deleted; replaced by `Functions.pdb_cache.pdbread`, called at `:210` and `:488` | Rewrite of an existing method's home (behaviour intended identical) |
| `Functions/XRR/XLayersPDB.py` | `protvolcalc`/`ED_profrot`/`calcProfComb` decorators (`:240,279,341`) | `lru_cache` maxsize 2/3 -> 16 | Additive (tuning) |
| `Functions/XRR/XLayersPDB.py` | `SphereElecFrac` (`:93-97`) | CSE of the segment-volume expression | Additive (algebraically identical) |
| `Functions/XRR/XLayersPDB.py` | imports (`:20,22-25`) | `Bio.PDB.Atom`, `fetch_table`/`get_table` now unused but left in place | Dead code left behind |
| `Functions/XRR/XLayers_Biphasic.py` | `calcProfile` signature (`:88`) | Added unused `sumd=None` kwarg | Additive, and probably accidental |
| `Fit_Routines.py` | `set_emcee_minimizer` (`:490-497`) | Upstream gates `fiterr` on `funcname=='XLayersPDB'`; ours widens to `('XLayersPDB','XLayersPDB_dev_smooth','XLayersPDB_XPR')` and adds a `stderr is None` guard nested inside | Additive to upstream's gate |
| `lmfit/minimizer.py` | new `_PDB_FUNCNAMES` (`:88-95`), used at `:1515, 1825, 1929, 2102` | Upstream `origin/develop` has *no* `XLayersPDB` references at all; `HEAD` has 4-name or-chains; the working tree replaces them with a 6-name tuple | Additive to upstream, refactor of our own committed code |
| `xmodfit.py` | new `_PDB_MCMC_FUNCS` (`:2634`) + `_mcmc_funcname` (`:2636`) / `_mcmc_is_pdb` (`:2655`), used at 10 sites (`:2804, 3006, 3035, 3111, 3394, 3501, 3547, 3592, 3643, …`) | Upstream has no `XLayersPDB` references; `HEAD` has eight `currentItem().text() == 'XLayersPDB' or ... == 'XLayersPDB_box'` chains; the working tree replaces them with a joint-aware, `None`-safe helper | Additive to upstream |
| `Functions/XRR/` (new files) | `XLayersPDB_G/_H/_box/_dev_smooth.py`, `XLayers_Beamfootprint.py` | New model classes; auto-registered by the category-folder rule | Purely additive |
| `Functions/pdb_cache.py` | new | Shared helper in `Functions/` root | Purely additive |

Merge history for the reader: per `merge-with-upstream-develop.md`, `XLayersPDB.py` and
`XLayers_Biphasic.py` were 2 of the 4 conflict files against `origin/develop` (tip `81e53c8`), and
both were resolved in `a4fd04e` in favour of origin's version plus our additions. All the *new*
files applied without conflict.

**The registration tuples disagree with each other.** This is the single most merge-relevant thing in
this document, and it is confirmed by reading:

| Site | Names listed |
|---|---|
| `lmfit/minimizer.py:94-95` `_PDB_FUNCNAMES` | `XLayersPDB, _box, _G, _H, _dev_smooth, _XPR` (6) |
| `xmodfit.py:2634` `_PDB_MCMC_FUNCS` | `XLayersPDB, _box, _dev_smooth` (3) |
| `Fit_Routines.py:491` (inline tuple) | `XLayersPDB, _dev_smooth, _XPR` (3) |

So for `_G`, `_H` and `_XPR` the minimizer will run the angle-aware orientation seeding, the
picklable parallel objective and the `sin(theta)` prior branch, while `xmodfit` hands it a stock
`emcee.backends.HDFBackend` instead of `CustomHDFBackend` and passes `angleindex=None`
(`xmodfit.py:3006-3009`); and for `_box`, `_G`, `_H` `Fit_Routines` skips the `fiterr` computation.
Pick one tuple and import it everywhere before folding in.

## 6. Known limitations, gotchas and TODOs

1. **`_G`, `_H` and `_box` raise `NameError` when `__fit__` is True.** In each `y()`, the call
   `self.ED_profrot(..., self.protins, self.__d__[mkey], ...)` uses `mkey` *before* it is assigned:
   `_G` uses it at `:696` and assigns at `:699`; `_H` at `:797` / `:800`; `_box` at `:523` / `:526`.
   The name only exists because it leaks out of the `for mkey in self.__mpar__.keys():` loop inside
   the `if not self.__fit__:` block above it. During plotting (`__fit__` False) that loop runs and
   the leak masks the bug; whenever `func.__fit__` is True it does not. The single-dataset path
   happens to be safe today because upstream commented out `self.func.__fit__=True`
   (`Fit_Routines.py:237`), but `Joint_Fit_Routines.py:452` sets it explicitly, so any joint fit
   using these three variants would crash. One-line fix: move the `mkey = list(...)[0]` assignment
   above the `ED_profrot` call.
2. **`_H`'s `layarea` looks dimensionally wrong.** `Functions/XRR/XLayersPDB_H.py:632` is
   `layarea = 1 - flipud(Volfrac)/xyarea`, but `Volfrac` there comes from the sphere-cap
   `SphereElecFrac` (`:104`) and is a *volume* (A^3), while `xyarea` is an area (A^2). `_G`, whose
   `Volfrac` is a genuine cross-sectional *area*, uses the same expression correctly (`:531-532`).
   `_H` most likely should divide by `area_foot = xyarea*fslice` (already computed at `:628`) — at
   the usual `fslice=0.5` that is a factor-2 error in the excluded-volume term.
3. **`_H` accepts and ignores `protvol`.** `ED_profrot(..., protvol)` at `:554`; the argument appears
   nowhere in the body. `protvolcalc` (`:514`) is therefore paid for and thrown away.
4. **Five stale copies of `parratt_numba`.** `XLayersPDB` now imports the shared, upstream-optimised
   kernel from `XRR.XLayers` (`:27`), but `_G:45`, `_H:37`, `_box:44`, `_dev_smooth:64` and
   `XLayers_Beamfootprint:19` each carry a private pre-optimisation copy (no hoisting of `qc`/`betaf`
   out of the q loop). `XLayers_Beamfootprint`'s copy also returns `ref, r` — the *last* scalar
   complex amplitude, not the `refc` array (`:40`) — and is decorated `@njit(parallel=True)` while
   looping with plain `range`, so the `parallel` flag does nothing. `_G`/`_H`/`_box` use `prange`
   under `parallel=False`, which numba silently degrades to `range`.
5. **Dead kernels left in the variants.** `Area_gridcalc` in `_box:171`; `Vol_gridcalc` and
   `Area_gridcalc` in `_H:128,153`. `_G` has no `protvolcalc` at all but its `__init__:467` still
   calls `pdbread` and discards the result.
6. **`_dev_smooth` bypasses `Functions/pdb_cache.py`.** It has its own `_TEMPLATE` cache
   (`:369-408`) with a *centred* template and an mtime-stamped key. Two consequences: the
   `pdb_cache.py` docstring's claim that every variant shares one parse is now inaccurate; and the
   same PDB will be parsed twice (once per cache) if `XLayersPDB` and `_dev_smooth` are both live.
   The mtime key is arguably the better design — `pdb_cache.pdbread` keys on the filename only, so
   editing a PDB in place will *not* invalidate it (noted for `_dev_smooth` at
   `Functions/gid2d_core.py:411`).
7. **`lru_cache` on instance methods.** `protvolcalc`/`ED_profrot`/`calcProfComb`/`sldCalFun` are
   still `@lru_cache` on *methods*, so the cache is per-class, keyed on `self`, and keeps every model
   instance alive forever. Bumping maxsize to 16 mitigates joint-fit eviction but makes the
   retention worse. `pdb-mcmc-seeding.md` measured the O(N_atoms) tuple hash at <1% of an eval and
   explicitly dropped optimising it.
8. **`_dev_smooth`'s `fslice` is not exact.** `_nsub` (`:714`) freezes the subdivision count from the
   `d` in force at construction. A fit that moves `d` far from its start silently drifts the
   effective slab thickness. Its docstring says so; there is no runtime warning.
9. **Insertion-depth registration break.** `_dev_smooth`'s docstring reports `XLayersPDB`'s effective
   `protins` is deep by ~1 `fslice` (measured -0.996/-0.992/-0.982/-0.973 x `fslice` at
   `fslice`=2/1/0.5/0.25). Any `.par` moved between the two needs `protins - fslice`. Nothing in the
   code enforces or warns about this.
10. **Angle fold has no effect within the same evaluation** — see audit hook #6.
11. **Deliberately deferred** (from `pdb-mcmc-seeding.md`): covariance/GLS joint MCMC is still
    serial; the two spawn Pools (seeding + sampling) are not unified. Both are 02/03 territory but
    they gate how these models behave at `cores>1`.
12. **No `.par`/`.xfit` in the repo uses any variant.** Only `XLayersPDB` appears in
    `Data/*_fitparam.par`. There is therefore no regression fixture for `_G`/`_H`/`_box`/
    `_dev_smooth`/`XLayers_Beamfootprint`.

## 7. Claims to verify (audit hooks)

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|
| 1 | `_dev_smooth` did *not* improve MCMC mixing: tau ~384 vs 238 for `XLayersPDB`, acceptance 0.075 vs 0.139, ~3x slower per eval (71 vs 25 ms/step) on `i2_tim3`, 5000 steps / 20 walkers / 8 workers | `pdb-mcmc-seeding.md` | Re-run both at `_dev_smooth`'s recommended `dz=0.25` (the comparison forced `dz=0.5`) and for >50x tau; the note itself flags the chain as too short and the run as a single sample |
| 2 | Every smoothness figure in `_dev_smooth`'s docstring (`protins` 2.23e-4 -> 6.9e-9, `theta` 1.14e-4 -> 7.0e-8, `phi` 1.68e-4 -> 2.1e-8, `d` 7.45e-6 -> 2.0e-10, `sig` 7.38e-6 -> 5.3e-11) | `XLayersPDB_dev_smooth.py:609-616` | Sweep each parameter with both models and recompute the local-cubic residual relative to \|R\|; none of this is scripted anywhere in the repo |
| 3 | `_dev_smooth`'s area-preset bias/spread/cost table (`reference` -0.002%/0.091%/65 ms, `default` +0.212%/0.118%/25 ms, `fast` +0.907%/0.355%/14 ms over five poses) and `AREA_NPAD` truncation figures (5 -> 6e-9, 4 -> 2e-7, 3.5 -> 2e-6) | `XLayersPDB_dev_smooth.py:39-56` | Compare `soft_area` at each preset against `_G`'s exact `greens_grid_n` for the same rotated coordinates — that is an independent exact reference already in the tree |
| 4 | `_dev_smooth`'s discretisation error law `err% ~ 0.31*fslice^2` and `~1.15*dz^2` at `cov=0.5`, giving 0.081% at `fslice=0.5, dz=0.25`, and `dz=1.0` unusable at 1.1% | `XLayersPDB_dev_smooth.py:587-598` | Richardson study of `y()` vs `dz` and `fslice` at fixed pose |
| 5 | `protins_new = protins_old - fslice` reproduces an old `XLayersPDB` fit in `_dev_smooth` | `XLayersPDB_dev_smooth.py:563-575` | Load `Data/i2_tim3_1000_test_fitparam.par` into both models, shift `protins` by `-fslice`, compare R and the EDP |
| 6 | The angle fold in `y()` actually constrains the pose | this document could not confirm it | `XLayersPDB.py:469-475` writes `self.params['theta'/'phi'].value`, but `:492` passes the *attributes* `self.theta, self.phi` to `ED_profrot`. `Fit.residual` (`Fit_Routines.py:~250`) writes both attribute (`self.params = func.__dict__`) and `func.params` each iteration, so the fold appears to be overwritten before it is ever read. Check whether the fold has any effect outside the MCMC path, where `customsampler.py` wraps angles independently |
| 7 | `pdb_cache.pdbread` is byte-for-byte equivalent to the removed method | `pdb_cache.py:60-61` | Diff `pdb_cache.pdbread` against `git show origin/develop:Functions/XRR/XLayersPDB.py` lines 240-267 and assert array equality on a real PDB. `validate_perdataset_fit.py:68-86` checks cache *sharing* (miss counts across instances and a `deepcopy`) but not numerical equality |
| 8 | The `SphereElecFrac` CSE at `XLayersPDB.py:93-97` is bit-identical to upstream | this document (algebraic reading only) | Floating-point reassociation is possible; evaluate both forms on a real template and compare exactly |
| 9 | `_box`'s covariance-ellipse footprint is an acceptable proxy | implied by its existence | `_dev_smooth`'s docstring (`:435-441`) says the ellipse proxy inflates coverage by a factor spanning 1.024-1.215 across poses, i.e. a ~15-19% pose-dependent spread. If true, `_box`'s `cov` is not comparable to `XLayersPDB`'s. Verify against `greens_grid_n` |
| 10 | `_H`'s factor-of-`fslice` `layarea` problem is real and not compensated elsewhere | this document (dimensional argument only) | Print `layarea` from `_G` and `_H` at the same pose/`fslice`; if `_H` is low by `1/fslice` the diagnosis holds |
| 11 | XPR's `theta`/`phi` are transferable to `XLayersPDB`'s, but `cov` and `protins` are not | `xpr-julia-backend.md` (self-corrected 2026-09-02) | Out of scope here; cross-reference `handoff/05-xpr-julia.md`. Relevant because `_XPR` shares `_PDB_FUNCNAMES`' angle-seeding path |
| 12 | `_G`/`_H`/`_box` ever produced a usable fit | nothing in the repo | No `.par`, no validator, no test references them. Given hook-driven finding #1 in section 6 they cannot have been used in a joint fit. Ask the user whether any published result came from them |

## 8. Consolidation candidates

| Item | Recommendation | Risk |
|---|---|---|
| `XLayersPDB_G.py`, `XLayersPDB_H.py` | **Drop from the fold-in** (or move to an `experiments/` folder outside `Functions/`). Both crash under `__fit__=True` (section 6.1), `_H` has a probable units bug, neither is referenced by any `.par`, and `_dev_smooth` explicitly *rejected* the Green's-theorem approach they embody (`:621-622`). Their real value is `greens_grid_n` as an **exact area reference** for auditing `soft_area` — preserve that function, e.g. as `Functions/pdb_area_exact.py` (must NOT live in `Functions/XRR/`, section 3.3). | Low if the user confirms no published result used them; moderate if `_G`'s exact area is the only validation path for `_dev_smooth`'s quadrature. |
| `XLayersPDB_box.py` | **Merge its ideas into `_dev_smooth`, then drop the file.** Its two genuine contributions — the membrane-conformal z grid and the interface-crossing interpolation — are both superseded by `_dev_smooth`'s conformal lattice (which makes interfaces exactly slab boundaries, so no interpolation is needed). Its ellipse area survives in `_dev_smooth` as `area_mode='ellipse'`. | Low. But it is the only variant besides `XLayersPDB` in *both* `_PDB_FUNCNAMES` and `_PDB_MCMC_FUNCS`, so check with the user whether it was ever the working model. |
| `_PDB_FUNCNAMES` / `_PDB_MCMC_FUNCS` / the `Fit_Routines` tuple | **Merge into one.** Export `_PDB_FUNCNAMES` from `lmfit/minimizer.py` (or better, a neutral module) and import it in `xmodfit.py:2634` and `Fit_Routines.py:491`. | Low, and it fixes a real inconsistency (section 5). Do it *after* deciding which variants survive, so the list is short. |
| Five private `parratt_numba` copies | **Replace with `from XRR.XLayers import parratt_numba`**, as the merge already did for `XLayersPDB`. Fix `XLayers_Beamfootprint`'s `return ref, r`. | Low for `_Beamfootprint` and the doomed variants; **do not** do it to `_dev_smooth`, whose self-containment is a deliberate design property. |
| `XLayers_Biphasic.py`'s `sumd=None` kwarg | **Drop it** unless the user confirms it is genuine WIP. `merge-with-upstream-develop.md` already flagged it as "reapply the 2-line tweak if genuine". | Nil — it is unused. |
| `XLayers_Beamfootprint.py` | **Keep, but rebase onto the current `XLayers.py`** (it forked a pre-merge version and thus loses upstream's Parratt optimisation), and consider making `rrf` work again rather than commenting it out. Also decide whether it should be a *flag* on `XLayers` rather than a separate class, since the category-folder rule means every fork is a new GUI entry. | Low. Note it is unreachable from the covariance joint fit today (section 3.5); if the intent is footprint-in-model for protein data, a `XLayersPDB` + footprint combination does not exist yet. |
| `Functions/pdb_cache.py` | **Keep, in `Functions/` root** — the location is a hard constraint. Consider adopting `_dev_smooth`'s `(abspath, mtime_ns, size)` key so an edited PDB invalidates, and update the docstring, which currently claims `_G/_H/_box` and "any future variant" use it (`_dev_smooth` does not). | Low. Changing the key changes cache behaviour under `cache_clear()`-based tests; `validate_perdataset_fit.py:68-86` asserts miss counts and would need updating. |
| Dead imports in `XLayersPDB.py` (`Bio.PDB.Atom`, `fetch_table`/`get_table`, `:20,22-25`) | **Remove.** They are pure residue of the `pdbread` extraction. | Nil, but it touches a merge-conflict file — do it in a separate commit so the upstream diff stays readable. |
