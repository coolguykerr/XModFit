# 2D grazing-incidence diffraction (GID) map fitting — XModJointFit change record

**Category:** Fit a whole (q_xy, q_z) GID map with one in-plane lattice model, plus the
2D image plot tab and the map-packing tooling it needs.
**Status:** Working-but-unvalidated-live. The code paths are covered by two validators
and a geometry selfcheck; the SCIENCE is not done — no accepted fit of the DPPC map
exists, and the tilt interpretation is explicitly unsettled (see 6.1).
**Owner chat(s):** memory `gid2d-function.md` (primary), `choices-eval-gotcha.md`,
`column-fit-limits.md`, `functions-category-folder-rule.md`. Original spec was an
external `GID2D_XModFit_handoff.md` in `~/Documents/Claude/` (NOT in this repo).
**Last touched:** 2026-08-28 (data/`.xpar`), code 2026-08-27.

## 1. Why this exists

A Langmuir-monolayer GID measurement is a 2D map: intensity over in-plane momentum
transfer q_xy and out-of-plane q_z. The conventional workflow slices it twice —
integrate over q_z for the in-plane Bragg peaks, then over a q_xy band for each Bragg
rod — and fits the slices with unrelated models, so the chain tilt from the rods is
never forced to agree with the peak positions it also determines. GID2D fits the full
map with ONE centred-rectangular chain lattice (NN spacing `a0`, chain tilt `t`, tilt
azimuth `psi`, uniaxial strain `eps`) and emits the 1D in-plane profile and the rods as
*derived outputs* of that same model. The immediate target is the DPPC monolayer map
`Data/dppc_q2d_gid_indexed.txt` (49 q_xy x 811 q_z, 38992 measured pixels), whose two
features — a sharp peak near q_xy 1.467 and a broad one near 1.356 — are the thing to
be explained. It sits alongside `Functions/GISAXS/Rod_Lipid.py` (single-rod model) and
deliberately shares its Vineyard/transmission convention so the two agree numerically.
Fitting a 2D map needed two pieces of infrastructure XModFit lacked: an image plot panel
(PlotWidget is 1-D only) and a fit window that is a q_xy x q_z rectangle rather than a
contiguous index interval.

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `Functions/GISAXS/GID2D.py` | 792 | new-untracked | the fittable model; 3 modes, all parameters, outputs, diagnostics |
| `Functions/gid2d_core.py` | 699 | new-untracked | vendored lattice/Ewald/Vineyard geometry, cosh-graded q_z quadrature, pack/load IO, `selfcheck()` |
| `Image2DWidget.py` | 586 | new-untracked | the 2D Map tab widget: N linked image panels, fractional q ticks, red banner |
| `Tools/GrazingIncidence/GID2DMapPacker.py` | 242 | new-untracked | Tools dialog: browse raw map -> Pack -> load + select GID2D + point `mapfile` |
| `Tools/GrazingIncidence/GIDViewerDialog.py` | 359 | new-untracked | Tools dialog: QWebEngineView wrapper around the GID explorer page |
| `Tools/GrazingIncidence/gid_template.html` | 1651 | new-untracked | the explorer page, with a `__CFG__` injection point |
| `Tools/GrazingIncidence/gid_explorer.html` | 1651 | new-untracked | byte-identical to the template except `__CFG__` -> `{}` (see 8) |
| `pack_gid_map.py` | 79 | new-untracked | CLI equivalent of the packer |
| `validate_gid2d.py` | 343 | new-untracked | numerical regression: sum rule, quadrature, pack/load, outputs, tilt spread |
| `validate_gid2d_gui.py` | 365 | new-untracked | headless GUI regression for the 2D tab and the packer |
| `Data/dppc_q2d_gid_indexed.txt` | 38998 | new-untracked | the packed DPPC map (the file you LOAD) |
| `Data/dppc_q2d_gid_indexed.xpar` / `.xfit` | 60 / 39082 | new-untracked | a saved parameter set + fit output — a RUNAWAY fit, not a reference (see 6.2) |
| `xmodfit.py` | — | modified (uncommitted) | 2D tab wiring, image routing, auto-select, `saveGenParameters` image branch |
| `Fit_Routines.py` | — | modified (uncommitted) | `__masked_x__` opt-in in `_install_func_x` (owned by `08-core-app.md`) |

State legend: `new` = added in a local commit; `modified` = upstream file we changed;
`new-untracked` = exists only in the working tree. Bucket note: NOTHING here is in a
local commit — every new file is untracked, and the `xmodfit.py`/`Fit_Routines.py` hunks
are working-tree-only (`git diff origin/develop...HEAD -- xmodfit.py` has zero hits for
`Image2DWidget`/`_op_is_image`/`_autoselect_image_params`; `git diff --` has ten).

## 3. How it works

### 3.1 The model and its three modes

`GID2D.y()` (`Functions/GISAXS/GID2D.py:349`) branches on `mode`:

- `mode='inplane'` (`:388`) — `x` is q_xy; the model is `_reduce_inplane`, the integral
  of the 2D intensity over `[qzmin, qzmax]` on the graded q_z grid.
- `mode='rod'` (`:397`) — `x` is q_z; the model is `_reduce_rod` over the q_xy band
  `[qxymin, qxymax]`, defaulting to the union of the per-family bands from
  `_rod_bands()`.
- `mode='map2d'` (fall-through, `:409`) — `x` is a *float row index* into `mapfile`.
  `load_map` + `row_lookup` (`gid2d_core.py:407`, `:444`) resolve each x to a row, and
  the residual is the whole 2D map.

Intensity is assembled in `_model` (`:255`):
`I = yscale * |T_f|^2 * SUM_six W_hk(q_xy) F_rod(q_z - q_z0_hk) * exp(-(q_z sig_z)^2)`
plus `bkg0 + bkg1 q_xy + bkg2 q_z + bkg3 q_z^2` (optionally x `|T_f|^2` when
`bkg_trans`) and an optional amorphous arc; everything below the horizon is forced to
zero (`:282`). `_peak` (`:238`) sums the SIX first-order reflections with SIGNED q_z and
averages over a Gauss-Hermite tilt quadrature (`_lattices`, `:215`), so `sig_t` smears
off-plane rods while leaving the q_z = 0 (02) rod sharp — the stated discriminator
between a tilt distribution and two coexisting phases. `dwba` has only the
`'kinematic'` branch; anything else raises `NotImplementedError` (`:261`).

### 3.2 `window='model'` vs `window='engine'`

XModFit computes `(data - fit)/err` in `Fit_Routines.residual`, OUTSIDE the model, so a
model cannot mask a point by returning NaN. The original trick (`:437-450`) returns the
DATA value on rows outside the q window, making their residual identically zero. That is
`window='model'`, the default, kept because earlier `.par` files assume it; its cost is
a zero-padded residual vector, so lmfit's reduced chi^2 is optimistic. `window='engine'`
returns the raw model everywhere (`:444-448`) and relies on the per-column mask engine
to drop excluded rows, giving an honest dof.

Why a mask engine and not `xmin:xmax`: a q_xy x q_z rectangle is NOT an interval in
row-index space — q_xy cycles once per q_z row, so a rectangle selects a comb of rows.
The column-limits machinery (`_LimitProxy` `xmodfit.py:73`, `ColumnLimitsDialog` `:120`,
`_column_mask` `:1319`, `Fit.set_mask`/`imask_for` `Fit_Routines.py:114`/`:141`) is
documented in `handoff/08-core-app.md`; GID2D is its first consumer. GID2D also opts
into `__masked_x__ = True` (`GID2D.py:24`, honoured at `Fit_Routines.py:219`) so it
receives only the active rows — safe because `row_lookup` is nearest+clip and every
quadrature grid is built inside the model, never from `self.x`.

The four `qxymin/qxymax/qzmin/qzmax` parameters were deliberately NOT retired when the
mask arrived: they have a second job (q_z quadrature limits for `inplane`, q_xy band for
`rod`) the mask cannot do (`GID2D.py:80-87`). All four are `vary=0` on purpose — a
window edge enters the objective discontinuously, so LM/TRF cannot move it, and it
trades directly against `yscale`/`bkg0` (`:200-208`).

### 3.3 The vendored core, and the quadrature fix

`Functions/gid2d_core.py` sits in `Functions/` ROOT, not `Functions/GISAXS/`, because
`xmodfit.update_functions` treats every `.py` in a category folder as a model class
named after the file and would crash the function list — the same reason
`Functions/pdb_cache.py` lives there. The lattice / signed-q_z / Ewald / cell-inversion
block is a VENDORED COPY of `grazing-incidence-scattering/scripts/gic_core.py` (skill
asset), copied rather than imported because that path is session-scoped
(`gid2d_core.py:10-19`). Four conventions are asserted in place by `selfcheck()`
(`:557`, >50 assertions; `python Functions/gid2d_core.py`): q_z is signed and summed
over the six reflections; areas use `|b1 x b2|` not `|b1||b2|`; one shared Ewald
cross-section definition; `transmission()` matches
`Functions/GISAXS/Rod_Lipid.py:58-64`'s `trans(qbeta, qc/2)`.

The load-bearing numerical decision is `qz_nodes` (`:332`). `|T_f|^2` has exactly two
non-smooth points — the horizon clip at `q_b = 0` and a **sqrt branch point at
`q_b = q_c/2`, which is also its maximum** (`transmission`, `:262-274`). Uniform
trapezoid nodes over that converge NON-MONOTONICALLY (4.9e-3 at n=51, worsening to
2.7e-2 at n=101, then 5.9e-3) at ~1% for practical node counts — large enough to
masquerade as physics. The fix drops `[lo, z0)` entirely (`|T_f|^2` is identically zero
there), puts uniform nodes on the exact parabola `[z0, z1]`, and grades the rest as
`q_b = (q_c/2) cosh(s)` with `s` uniform — the matched grading for a sqrt branch point;
`dqz_max` caps the coarse end at ~1/8 of the rod lobe so a narrow rod stays resolved.
`validate_gid2d.test_quadrature` asserts monotone convergence, < 1e-4 by n=401, and
>20x better than uniform there.

This is also the first of two claims from the original handoff document that were found
to be **WRONG**: its section 9 blamed the non-monotone convergence on something other
than the Vineyard cusp. It IS the cusp. The control that appeared to exonerate it set
`qc -> 0` instead of removing `|T_f|^2`, which leaves both the clip kink and a 0/0 —
i.e. it was still testing the cusp (`qz_nodes` docstring, `:336-341`).

### 3.4 The packed-file format, and the sigma floor

`pack_map` (`gid2d_core.py:455`) turns a raw `qxy qz I Ierr` map into the indexed
format. Column order `['idx','I','Ierr','qxy','qz']` is **load-bearing**: it makes
`Data_Dialog`'s DEFAULTS land on x=idx, y=I, yerr=Ierr with no configuration (its yerr
combo has `'None'` at index 0, so its default index 3 picks the third column). Do not
reorder. An index column is needed because XModFit requires x MONOTONE (it slices with
`np.where` on x) and raw q_xy from a map is a sawtooth.

`pack_map` DROPS masked rows rather than windowing them, with no later workaround:
`(data-fit)/err` is computed outside the model, so a `sigma = 0` row is 0/0 whatever
`y()` returns, and the DPPC file's 2595 masked pixels are scattered over 335 distinct
q_z values so no contiguous `xmin:xmax` excludes them. `I <= 0` must go too, because
`fit_scale='Log'` takes `log10(data)`.

The **sigma floor** (`sigma_floor_frac`, default `1e-6 x median`) is the subtle part.
Dropping the exact zeros is not enough: the same detector mask leaves pixels with
floating-point residue down to ~1.9e-16, and since `1/sigma` reaches 5e15 there, a
couple of dozen pixels out of ~39000 carried essentially ALL of chi^2 (top 10 ~97%). The
threshold is not a judgement call — sorted sigma jumps by ~1.6e11 across the cut, so any
floor from 1e-9 to 1e-5 x median removes the same set. The gap found is written into the
output header; the shipped file reads `sigma floor 2.508e-05 (= 1e-06 x median) dropped
37 near-zero-sigma pixels; gap at the floor = 1.62e+11`. `load_map` caches on
`(abspath, mtime_ns, size)` (`:402`, `:419`), bounded to 8 entries, so an edited-in-place
map invalidates; geometry is read from the qxy/qz COLUMNS, never recomputed as
`idx = i_qz*n_qxy + i_qxy`, because dropped rows break that arithmetic
(`mp['rectangular']` is False for the DPPC file).

### 3.5 The 2D Map tab

`Image2DWidget` is a separate widget, not a PlotWidget mode (`Image2DWidget.py:1-45`):
`PlotWidget.add_data` requires equal-length 1-D arrays, and `PlotWidget.Plot` calls
`plotWidget.clear()`, destroying everything added with `addItem` — also why
`PlotWidget.addROI` is dead code. **Rule: never `addItem()` into a PlotWidget.**

The image is drawn in **measurement-pixel space**, not q space. The DPPC sampling is
deliberately non-uniform (dq_xy 0.0043..0.0088, a 2.0x refinement across the peak; dq_z
drifting 0.00073..0.00154) and `pg.ImageItem` can only draw a uniform rectangle, so one
image pixel is one measured pixel and the q values go on the axes via
`AxisItem.setTicks`, which accepts FRACTIONAL positions. `_pixel_ticks` (`:338`) inverts
the sampling with `np.interp`, so a tick labelled 1.50 sits at exactly q=1.50 (snapping
to the nearest column was measured at 98% of its own error bound). At <= 64 columns a
minor tick is drawn per measured pixel, so the sampling change is visible on the axis.

Entries follow an `output_params` "image" convention (`x`, `y` 1-D axes; `z` 2-D
(nx, ny) with NaN = no measured pixel; `plotType='image'`; `role` in
`data|model|residual`; `panel` group key; `cmap`; `zscale` in `log|linear|signed`;
`names`). NaN renders fully transparent once a LUT is set. `_rebuild` (`:227`) groups by
`panel`, orders columns data | model | residual (`ROLE_ORDER`, `:55`), gives the
non-signed panels ONE shared colorbar and the residual its own, and caps at
`MAX_ROWS = 3` groups so a 10-dataset joint fit does not make 30 panels. Shared levels
come from the DATA panel only (`_apply_levels`, `:437-457`) so data and model are
pixel-comparable; the residual bar is ALWAYS symmetric about zero (`:427-435`) —
asymmetric levels on a diverging map would put white somewhere other than zero, i.e. the
panel would lie. `set_maps` refreshes in place unless the panel signature changes, which
preserves zoom and dragged levels across a Calculate.

GID2D builds the three entries in `_map_images` (`GID2D.py:504`), scattering onto the
measurement grid with exact `searchsorted` lookups on `uqxy`/`uqz` rather than a
reshape. Two details matter: the RAW model is captured *before* the window overwrite
(`:430-435`) so the model panel shows the model OUTSIDE the fit window — the main reason
to look at a map at all; and the residual is NaN outside `active` (`:543-549`), because
windowed-out pixels satisfy `res == dat` by construction and would otherwise paint an
identically-zero "perfect fit" region. The window is thus VISIBLE in the residual panel.

### 3.6 Wiring into xmodfit.py, and the self-arming `__fit__` gate

`OP_META_KEYS` (`xmodfit.py:70`) strips image metadata keys from the
generated-parameter list text at all three population sites via `_op_var_list` (`:1524`,
used at `:6107`, `:6319`, `:6630`). The tab is created hidden at `:5209-5214`;
`_set_2D_tab_visible` (`:5224`) uses `setTabVisible` on Qt >= 5.15 and insert/remove
otherwise, guarded by `_programmatic_tab_change` so a programmatic change is not
mistaken for a user choice. `_show_2D_tab` (`:5243`) raises the tab only when the
selection produced no 1-D curve AND the user has not picked a tab by hand;
`_on_plotTab_changed` (`:5259`) sets that latch. `plot_extra_param` (`:6802-6861`)
routes any `plotType == 'image'` entry to `imageWidget.set_maps` and NEVER to the 1-D
PlotWidget (reaching `_plot_one` with one would plot `axes[2]` as the errorbar),
namespacing `panel` per source so a joint fit gives N rows not 3N panels; it then
scrapes every `ERROR_`/`WARNING_` scaler into `imageWidget.set_warning` — the red
banner. `_autoselect_image_params` (`:1483`) selects the map entries once per function
pick (`_image_autoselected`, reset at `:569`, `:5449`); without it the tab never
appears, since it is driven by the `genParamListWidget` SELECTION, empty on a fresh
pick. Discrimination is on `plotType`, never the displayed text, because an image entry
renders as `['x', 'y', 'z']` — textually identical to the legacy meshgrid form
(`_op_is_image`, `:1514`). `saveGenParameters` gained an image branch (`:4899-4908`)
because the legacy `['x','y','z']` branch does `x.shape[1]` and would raise IndexError
in a slot with no try/except. Main path: `GID2D.y()` builds the entries on its report
path -> selection -> `plot_extra_param` -> `set_maps` + `set_warning` -> `_show_2D_tab`.

That report path is gated. XModFit sets `func.__fit__ = False` on every explicit
Calculate/plot path and never sets it True, so `y()` sets it True at its own end
(`:454`) — the expensive extras run once per Calculate and are skipped inside a fit.
That alone produced an EMPTY map, because `update_plot` calls `evaluate()` BEFORE
`set_x()` (cited as `xmodfit.py:6207`): the first evaluation runs on the x line-edit
expression and only the second, via `residual()` after `set_x`, sees the real dataset.
The gate is therefore also x-change aware (`:372-374`), catching
evaluate -> set_x -> residual while a fit loop (constant x) still reports once.

## 4. How to use it

Experiment/sample: Langmuir monolayer GID at a liquid surface; the shipped case is a
DPPC monolayer at E = 15.126 keV, alpha_i = 0.10 deg, q_c = 0.0217.

Workflow from a raw map to a fit:

1. Least friction: **Tools -> GrazingIncidence -> GID2DMapPacker**. Browse to the raw
   `qxy qz I Ierr` file (column indices editable, default `0,1,2,3`), leave "Drop
   q_z < 0" and "Load the result" checked, press **Pack**. It writes
   `Data/<name>_indexed.txt`, loads it as a dataset, selects GISAXS/GID2D, points
   `mapfile` at its own output, calls `update_plot`, and logs kept/dropped counts, the
   sigma-floor note and the grid fill fraction. CLI equivalent:
   `python pack_gid_map.py <raw> [out]`.
2. Manual equivalent: `Add files -> <name>_indexed.txt` (**NOT** the raw file), then
   Functions -> GISAXS -> GID2D, then the `mapfile` button -> the same indexed file.
3. Set **Fit Scale -> Log**. Not optional; see 6.1.
4. Set the x line-edit to `np.arange(0., <n_rows>.)` if it is not already the dataset's
   own x — the stock default is a 300-point linspace, filling 300 of ~39000 cells, which
   makes the map look broken for a reason nothing on screen explains. GID2D reports
   `map_coverage_frac` / `WARNING_map_coverage` for this.
5. Choose the window: either set `qxymin/qxymax/qzmin/qzmax` with `window='model'`, or
   set `window='engine'` and define it via the **Limits** button on the `qxy`/`qz`
   columns. With `window='engine'` and NO column limits, EVERY row is fitted, including
   the sub-critical ones.

Fixed parameters (`choices`, `GID2D.py:148-156`): `mode` (`map2d`/`inplane`/`rod`),
`lineshape` (`Gaussian`/`Lorentzian`/`PseudoVoigt`), `powder`, `vineyard`, `bkg_trans`,
`arc`, `dwba`, `fisher`, `window`; plus `n_tilt`, `nq`, `mapfile`. Fitting parameters:
`a0 t psi eps | L_dom L_ch sig_t sig_z eta | E alpha_i qc | qxyoff qzoff | yscale
bkg0..3 arc_amp arc_Q0 arc_sig | qxymin qxymax qzmin qzmax`. Defaults worth knowing:
`yscale=100.0` and `bkg0=40.0` are this dataset's starting values, not 1.0/0.0 — with
1.0/0.0 the model is ~1e-9 against data of ~50 and the model panel opens black; and
`mapfile` falls back to `Data/dppc_q2d_gid_indexed.txt` when present (`:127-131`) so
selecting the function never throws at load time.

Numbers to sanity-check after any change:

- `A_per_molecule` vs `A_per_molecule_from_peaks` must agree to < 1e-6 (round-trip
  through the oblique cell inversion).
- `band_over_half_separation` < 1.0 — the rod band must be narrower than half the peak
  separation. Recorded failures: a 2-theta window wider than the separation gave
  ratio 1.00, a band wider than half gave 0.94, a fixed-column integration of a curving
  feature gave 0.65; only a proper `|q_xy|` band returned 1.98 (`_rod_bands`, `:306`).
- integrated (11)+(1-1) : (02) = **1.998** with `bkg0 = 0`, `powder=False`,
  `vineyard=False`, `alpha_i=0`. With a background present it reads 1.632 — pinning
  `bkg0=0` in `validate_gid2d.mk()` (`:39-49`) is why the test is meaningful.
- DPPC NN preset: `a0=4.8743, t=18.85, psi=0, eps=-0.0321` -> 42.0878 A^2/molecule,
  peaks at q_xy 1.464 (q_z 0.43) and 1.489 (q_z ~ 0).
- `weight_frac_top_10_px_LINEAR` vs `..._LOG`, and `median_sigma_over_I`.
- `map_coverage_frac` > 0.9 and `WARNING_map_coverage` empty.

Regression commands (deliberately NOT run for this document):
`python validate_gid2d.py`, `QT_QPA_PLATFORM=offscreen python validate_gid2d_gui.py`,
`python Functions/gid2d_core.py`. Env: cmc-py39 (numpy 1.21 — hence
`trapz = getattr(np, 'trapezoid', np.trapz)` at `gid2d_core.py:50`).

## 5. Contact surface with stock XModFit

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|
| `xmodfit.py` | `:59` import, `:70` `OP_META_KEYS` | new import + module constant | additive |
| `xmodfit.py` | `:1483` `_autoselect_image_params`, `:1514` `_op_is_image`, `:1524` `_op_var_list`, `:5224`/`:5243`/`:5259` tab helpers | new methods | additive |
| `xmodfit.py` | `:6107`, `:6319`, `:6630` genParamListWidget population | `str(list(entry.keys()))` -> `_op_var_list(entry)`; `_autoselect_image_params()` call added | small rewrite of the displayed text |
| `xmodfit.py` | `:5200-5215` `create_plotDock` | `plotTabWidget` gains a hidden '2D Map' tab + 3 latch attrs | additive (1-D tab unchanged) |
| `xmodfit.py` | `:6802-6861` `plot_extra_param` | image entries intercepted before the 1-D path; banner scrape; tab show/hide | additive branch, but short-circuits the existing `len(axes)>=2` path for images |
| `xmodfit.py` | `:4899-4908` `saveGenParameters` | new `_op_is_image` branch before the legacy `['x','y','z']` one | additive |
| `xmodfit.py` | `:569`, `:5449` init / `functionChanged` | `_image_autoselected = False` | additive |
| `Fit_Routines.py` | `:200-234` `evaluate` / `_install_func_x` | `__masked_x__` opt-in | additive; owned by `08-core-app.md` |
| `Functions/` root, `Tools/` | `gid2d_core.py`; new `GrazingIncidence/` folder | helper module in a non-category dir; new Tools submenu (auto-discovered at `xmodfit.py:645-657`) | additive |

Merge risk is concentrated in `plot_extra_param` and the three list-population sites,
which are large upstream functions that also carry unrelated joint-fit edits.

## 6. Known limitations, gotchas and TODOs

### 6.1 Two data-analysis findings that change results

**(a) Fit this data in LOG scale.** The errors are multiplicative — `sigma/I` has a
median of ~0.41 on the ROI (q_xy 1.40-1.54, q_z 0.02-0.60, 9875 pixels) and is capped
at 1.0 — so Linear weighting `1/sigma^2` diverges wherever `I -> 0` below the critical
angle. Measured: ONE pixel at (1.4018, 0.0841) held **72.2%** of the total Linear
weight and the top ten held 99.1%, and that is real data, not mask residue the sigma
floor could remove. XModFit's `Log` residual is `(ln d - ln f) * d/err` (log10 before 2026-11-10; the
relative weights below are unchanged), whose
weight is `(I/sigma)^2` — flat for a multiplicative error model; the same top pixel then
holds 0.08% and the top ten 0.66%. It changes the answer: `L_dom` 144 -> 313 A,
`L_ch` 9.4 -> 15.3 A, `bkg0` -4.3 -> +53. This is a correctness issue, not cosmetics.
`GID2D._weight_report` (`:642`) computes both concentrations from the DATA alone and
emits `RECOMMENDED_fit_scale` / `WARNING_weighting`, which the GUI surfaces as the red
banner. **This is the second claim from the original handoff document that was found
WRONG:** its section 3.1 said "treat sigma as given", which is unsafe here.

**(b) The physics interpretation is NOT established by the map.** First look: with `t`
fixed at 30.5 deg the model's (02) lands on the data's sharp peak (1.4666 vs 1.4674) but
its (11) sits at q_xy = 1.28 against the data's broad feature at ~1.356 — visibly wrong
side by side on the 2D tab. Freeing `t`/`eps` from that start runs away (`a0 = 17 A`,
`t = 60 deg`, peaks at q_xy = 0.27, i.e. it just fits the background). So the
NN-chain-tilt reading at t = 30.5 deg is a MODEL CHOICE the map does not obviously
support. Do not report a tilt from this map without the section-4 discriminator of the
original spec. `_identifiability` (`:737`) exists precisely because `t` and `eps` are
identifiable only if the broad feature is accepted as the (11)/(1-1) family.

### 6.2 The shipped `.xpar` is a runaway, not a reference

`Data/dppc_q2d_gid_indexed.xpar` records `a0 = 4.549, t = 47.92, psi = -55.75,
L_dom = 2.01e4, L_ch = 151.5` with `#Fit Scale=Linear` and `E = 10.0`. All three are
wrong for this dataset: Linear contradicts 6.1(a), `E = 10` contradicts the beamline
value 15.126 that every other file uses, and the lattice parameters are the runaway
described in 6.1(b). Treat the file as a *format* example only. It also shows that
`E` reverts to a non-DPPC value on some path (constructor default is 15.126) — worth
tracing before anyone reuses it.

### 6.3 The `choices` eval gotcha

`xmodfit.update_plot` and `fixedParamChanged` both coerce a Fixed-Parameters cell with
`eval(val)`, falling back to `str(val)` only on exception. So any `choices` value whose
text is a valid Python expression comes back as **that object, not the string**. GID2D
originally shipped `mode: ['map', 'inplane', 'rod']` and `func.mode` arrived as
`<class 'map'>` — the builtin; it appeared to work only because map mode is the
fall-through branch of `y()`. Renamed to `'map2d'`, and `y()` now normalises with
`getattr(v, '__name__', None) or str(v)` for `mode`, `lineshape`, `dwba` (`:382-386`).
Never use a Python builtin, keyword or literal as a `choices` value (`map filter list
set type id all any sum min max input next iter open dir hash bin hex oct abs pow int
str None nan e inf`, or anything numeric). `'True'`/`'False'` -> bools IS intended.

### 6.4 Loading the RAW map silently misfits

`x` in `map2d` mode is a row index, so the DATASET must be the indexed file. Load the
raw `(qxy, qz, I, Ierr)` file and `Data_Dialog` picks q_xy as x and q_z as y: all 41797
q values round into rows 0-2, so 2 of 39739 grid cells are filled (panel looks blank)
AND the fit regresses the model against q_z. `_coverage_message` (`GID2D.py:470`) names
this cause specifically — "x IS NOT A ROW INDEX", "you have almost certainly loaded the
RAW file", "the fit is ... meaningless", plus the exact `pack_map` one-liner — and
`plot_extra_param` shows it as a red banner. `validate_gid2d_gui.py:282-311`
regression-tests the chain.

### 6.5 Other traps recorded

- **Three display defaults each made the panel look broken.** (i) The tab is driven by
  the `genParamListWidget` SELECTION, empty on a fresh function pick — hence
  `_autoselect_image_params`. (ii) The `evaluate()`-before-`set_x()` ordering vs the
  self-arming `__fit__` flag (3.7). (iii) The panel opened LINEAR despite every entry
  declaring `zscale='log'`, putting this data's median at 22% of the level range where
  CET-L3 is near-black — `set_maps` now honours the declared zscale once and stops
  second-guessing after a hand toggle (`Image2DWidget.py:186-195`, `_user_log`).
- **Do not force pyqtgraph axis sizes.** Missing tick labels in an offscreen harness are
  an artifact of never showing the MAIN window (the child gets no layout pass);
  `autoExpandTextSpace` converges when shown. Forcing `tickTextWidth = 44`/`54` made it
  measurably worse (6/6 labels -> 0/6).
- `psi` is EXACTLY a no-op at `t = 0` and degenerate at small `t`;
  `scaler_parameters['psi_identifiable']` reports this rather than fitting it silently.
  `sig_t > 0` with `n_tilt < 3` emits `WARNING_n_tilt` (unconverged tilt average).
- `_last_xsig`, the private x-signature cache set at `GID2D.py:375`, leaks into the
  Fixed Parameters table and into saved `.xpar` files (visible in the shipped one),
  because `update_fixed_parameters` filters only on `key[:2] != '__'`
  (`xmodfit.py:5553`). Cosmetic but confusing; `__last_xsig__` would hide it.

### 6.6 Explicitly NOT done

- **The fit-and-interpret step.** The question the model was built to answer — does one
  NN-tilted phase with `sig_t` free fit BOTH observed features? — has not been answered.
- `Tools/GrazingIncidence/GID2DMapDialog.py` — planned, does not exist. `grep` finds no
  reference to it anywhere in the repo, so nothing is broken by its absence.
- **GID explorer coupling.** `GIDViewerDialog.cfg_from_func` reads ONLY `func.E` (plus
  layer count/thickness for reference); it does not read `a0/t/psi/eps` from a live GID2D
  instance and cannot push them back. The dialog's docstring justifies that for the
  XLayers family (no in-plane lattice exists there), but GID2D DOES have those
  quantities, so the justification no longer holds and the coupling is simply unbuilt.
- Skill updates (the `grazing-incidence-scattering` skill was not updated).
- `dwba='dwba'` — a declared choice that raises `NotImplementedError`.
- The raw source map is not in the repo: `validate_gid2d.RAW` points at
  `T:\SURFACE\NX_school\2026sep\extracted_data\dppc_q2d_gid.txt`, so test [6]
  (pack/load) SKIPS off-site. Only the packed file ships.

## 7. Claims to verify (audit hooks)

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|
| 1 | Cosh-graded nodes are "59x better than uniform at n=401" | memory `gid2d-function` | `validate_gid2d.test_quadrature` only asserts >20x; run it and record the actual ratio. The docstring's own numbers (+9.5e-4 vs -4.5e-2 at ~158 nodes) imply ~47x. |
| 2 | The sigma floor drops "26 pixels of mask residue"; `median sigma/I` is 0.41 | memory `gid2d-function`, `pack_gid_map.py:27,34`, `_weight_report` docstring | The shipped file's header says **37** pixels, and memory/`pack_gid_map` say sigma/I ~0.47 vs the docstring's 0.41 (probably whole map vs ROI). Re-run `pack_map` and recompute both; label which region each number refers to. |
| 3 | Log scale moves `L_dom` 144 -> 313 A, `L_ch` 9.4 -> 15.3 A, `bkg0` -4.3 -> +53 | `_weight_report` docstring | No `.xpar` in the repo records either fit. Re-run both and save them, or the headline claim of 6.1(a) has no artifact behind it. |
| 4 | One pixel at (1.4018, 0.0841) holds 72.2% of the Linear weight | `_weight_report` docstring | Recompute `1/sigma^2` on the stated ROI of the shipped packed file. Cheap and decisive. |
| 5 | `window='engine'` gives identical parameters and chi^2, with redchi 0.131 -> 0.271 = exactly n/n_active | memory `column-fit-limits` | Re-run both windows on the same start point and diff the parameter table. |
| 6 | In `window='engine'` mode the 1-D reductions and the reported chi^2 still describe the fit | `_group_integrate` docstring, `GID2D.py:331-337` ("a reduction can never disagree with the fit") | `y()` sets `active = err > 0` at `:448`, discarding the q window. With `__masked_x__` the engine already pre-slices `self.x`, so this is probably consistent — but if column limits are NOT set, the reductions silently cover the whole map. Construct both cases and compare `chi2_per_N_active_only_LINEAR` against lmfit's chi^2. |
| 7 | In `window='engine'` the model panel still shows the model OUTSIDE the fit window | inferred from `:430-435`'s stated purpose | With `__masked_x__`, only active rows reach `_map_images`, so out-of-window pixels should be NaN. Check whether the out-of-window view — the stated main reason to look at a map — is actually lost in engine mode. |
| 8 | The packed column order makes `Data_Dialog` defaults land on x=idx, y=I, yerr=Ierr with no configuration | memory, `GID2DMapPacker` docstring | `validate_gid2d_gui.py:343` asserts it. Re-run headless; also confirm the claim about the yerr combo's index-0 `'None'` by reading `Data_Dialog.py`. |
| 9 | Forcing `tickTextWidth` to 44/54 takes tick labels from 6/6 to 0/6 | memory | Not reproduced here; if anyone "fixes" axis sizing, re-measure before believing it. |
| 10 | `update_plot` calls `evaluate()` before `set_x()` at `xmodfit.py:6207` | `GID2D.py:365` comment | The file has since been edited; confirm the line number and that the ordering still holds, since the whole x-change-aware gate depends on it. |
| 11 | GID2D and `Rod_Lipid` agree on `|T_f|^2` to < 2e-3 | `validate_gid2d.test_rod_lipid_agreement` | Run it; the stated residual is attributed to `HC` differing in the 5th digit (12.39842 vs 12.3984). Consider making `Rod_Lipid` import `gid2d_core.HC`. |
| 12 | GID2D works under a JOINT fit (`Joint_Fit_Routines.py:445-455` honours `__masked_x__`) | inferred from the code path only | Nothing in either validator exercises GID2D jointly, and `plot_extra_param` namespaces `panel` per source specifically for that case. Build a two-map joint fit and check the panel grouping and `MAX_ROWS`. |
| 13 | `t = 30.5 deg` puts (02) at 1.4666 vs the data's 1.4674 and (11) at 1.28 vs 1.356 | memory `gid2d-function` | Reproduce from the shipped map; these four numbers are the entire basis for 6.1(b). |

## 8. Consolidation candidates

- **`gid_explorer.html` vs `gid_template.html`** — byte-identical apart from line 184
  (`XMODFIT_CFG = {}` vs `= __CFG__`); two 79 KB copies. **Drop** the explorer and
  generate it with `python GIDViewerDialog.py --browser out.html`, which exists for
  exactly this. Risk: low; no `.py` references the explorer filename.
- **`pack_gid_map.py` vs `GID2DMapPacker.py`** — thin wrappers over the same `pack_map`,
  whose rationale is restated a third time in `pack_map`'s own docstring. **Keep both**
  (the CLI is the only headless route and is quoted in `GID2D.py:495-498`); consider
  printing `pack_map.__doc__` rather than restating it.
- **`GIDViewerDialog.py` + HTML** — exploration tool with almost no coupling to the fit
  (only `func.E`); its JS duplicates lattice math that now exists in `gid2d_core.py`.
  **Keep**, but unify the math if the 6.6 coupling is built, or JS and Python will drift.
- **`gid2d_core.py` vendored copy** — provenance documented (`:10-19`) but no version
  stamp of upstream `gic_core.py`. **Keep** vendored; add the source revision.
- **`_last_xsig` in the Fixed Parameters table** (6.5) — rename to `__last_xsig__` so
  the `key[:2] != '__'` filter hides it. Risk: it is already written into
  `Data/dppc_q2d_gid_indexed.xpar`; check the loader tolerates an unknown fixed param.
- **`Data/dppc_q2d_gid_indexed.xfit` (39082 lines, 3.9 MB)** — a full evaluation of the
  runaway fit. **Drop** the pair unless a real fit replaces it; both are regenerable and
  a 3.9 MB artifact of a known-bad fit is worse than none.
