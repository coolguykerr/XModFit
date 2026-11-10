<!-- Template for handoff/NN-<topic>.md. Keep the section order and headings. -->

# Core application (main window, fit engine wrapper, plot widget) — XModJointFit change record

**Category:** Changes to stock XModFit's own three files — `xmodfit.py`, `Fit_Routines.py`,
`PlotWidget.py` — that are NOT joint-fit, MCMC, GID2D/2D-map or structure-viewer work.
**Status:** Production for per-dataset fit memory and plotting/legends (committed, GUI-exercised).
Working-but-unvalidated-live for per-column fit limits and the mask engine (headless regressions
pass; never clicked through the live GUI). Working-but-unvalidated-live for the analytic-Jacobian
plumbing (only consumer is `XLayersPDB_XPR`).
**Owner chat(s):** `column-fit-limits.md` (sub-feature a), `multimodal-seeding-plan.md`
"Plotting/UX requests — CATEGORY B" (sub-features b, c), `xpr-julia-backend.md` (sub-feature d),
`choices-eval-gotcha.md`, `merge-with-upstream-develop.md`, `gid2d-function.md`.
**Last touched:** `Fit_Routines.py` 2026-09-01, `xmodfit.py` 2026-08-27, `PlotWidget.py` 2026-08-25
(commits `c5f697d`/`babe1aa` are 2026-08-05/06; everything later is uncommitted).

## 1. Why this exists

Stock XModFit is a one-dataset-at-a-time fitter: one live `Fit`, one contiguous `xmin:xmax`
window, one plot legend keyed by raw column names. Four analysis needs broke those assumptions.
(i) A 2-D GID map is loaded as a long table whose `x` is a row index, so the region you want to fit
is a `q_xy` x `q_z` RECTANGLE, which is not an interval in index space — hence optional limits on
any named data column, delivered to the engine as a boolean per-point mask. (ii) Reflectivity
campaigns fit ten to twenty films with the same model but different parameters, so each dataset
must own its model, parameters and result, and fitting a multi-selection must use each dataset's
OWN starting values rather than the last one's. (iii) Once a joint fit exists, the scientific
question is "how much did sharing cost this dataset?", which means seeing a dataset's individual
best fit and its joint best fit on the same axes (and the same electron-density panel) at once.
(iv) A differentiable model (`XLayersPDB_XPR`) can supply an exact Jacobian and needs its unbounded
Euler angles folded back into a conventional window before the fit report is written. The work is
deliberately additive: `xmin`/`xmax` still exist as mappings, `imin`/`imax` still mean "the
contiguous x window", and models that know nothing about masks or Jacobians are unaffected. Models
exercised: `XLayers`, `XLayersPDB`, `XLayersPDB_XPR`, `GID2D`.

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `xmodfit.py` | ~7740 (+3841/-~250 vs upstream) | modified | main window: limits store + dialog, per-dataset fit memory, legends, overlays, EDP co-plot |
| `Fit_Routines.py` | 528 (+332/-~20 vs upstream) | modified | `Fit` wrapper: mask engine, window unification, analytic Jacobian, result canonicalisation |
| `PlotWidget.py` | 530 (+122/-~30 vs upstream) | modified | data-line toggle, per-curve linestyle, legend formatter + legend rebuild |
| `validate_fit_mask.py` | ~230 | new-untracked | engine-level mask + `_window_indices` regression |
| `validate_column_limits.py` | ~300 | new-untracked | GUI-level limits store, column discovery, badge, `.xpar`/`.jpar` round-trip |
| `validate_masked_x.py` | ~150 | new-untracked | `__masked_x__` opt-in and honest-dof check |
| `validate_perdataset_fit.py` | ~230 | new (committed) | per-dataset fit memory + shared PDB cache regression |
| `validate_jointfit_gui.py` | ~430 | modified | includes the plain-click/Ctrl-overlay selection-model checks |
| `Image2DWidget.py` | — | new-untracked | imported by `xmodfit.py:59`; belongs to `handoff/06-*` (2D map) |
| `joint_fit_panel.py` | — | new (committed) | `xmodfit.py:58` imports `NameDelegate` from it for the MAIN Data list |

Bucket summary (verified by grepping `HEAD:xmodfit.py` / `HEAD:Fit_Routines.py`): **committed**
(`c5f697d`/`babe1aa`) = per-dataset fit memory (`dataset_fits`, `_clone_fit`, `_fit_for_dataset`,
`_activate_dataset_fit`, `_sync_funclist_to_fit`, `_refresh_data_labels`,
`_update_ci_button_state`), the deterministic palette (`_curve_color_for`, `PlotWidget` Tableau-10
fallback), `filebutton_paths` + the `openFileDlg` signature, `_short_name`, the model-comparison
dock, the `showFitInfoDlg`/`stopFit`/`closeFitInfoDlg` guards, and (in `Fit_Routines.py`) only the
`fiterr` guard + `nan_policy` change. **Uncommitted** = everything else: the whole per-column
limits/mask engine, `DataListWidget`/detail-dataset/no-fit banner, all legend and overlay work
(`_legend_order`, `_legend_label`, `_curve_meta`, `_indiv_fit_name`, `_plot_combined_overlay`,
`_maybe_combined_overlay`, `_eval_individual_curve`, `_model_output_params`,
`_overlay_modifier_active`, `clear_joint_view`), the `PlotWidget` data-line toggle / linestyle /
`legend_formatter` / `legendItem.clear()` fix, and `Fit._canonicalize_result` / `can_jac` /
`residual_jac`.

## 3. How it works

### 3a. Per-column fit limits and the mask engine

**Store.** `self.limits[data_key] = {col_name: (lo, hi)}` (`xmodfit.py:532`), with `'x'` a reserved
column name meaning "whichever DataFrame column is plotted as x". `self.xmin`/`self.xmax` are no
longer dicts but `_LimitProxy` MutableMapping views onto the `'x'` entry (`533-534`, class at
`73-116`), index 0 = low bound, 1 = high. Every pre-existing `self.xmin[key]` read (`1054`,
`3419`, `6217`), write (`addData` `4231-4232`, `xminmaxChanged` `1718`, `openDataDialog`
`1651-1654`) and delete (`removeData` `4286-4287`) therefore works verbatim. `__delitem__`
(`102-107`) is deliberately tolerant of a repeated delete because `removeData` deletes `xmin[k]`
then `xmax[k]`, both the same underlying `'x'` entry; a missing DATASET still raises. That one
choice is what made the generalisation a refactor of one store rather than a rewrite of ~20 call
sites.

**Column discovery and mask construction.** No new data plumbing was needed: `Data_Dialog.readData`
already retains the whole DataFrame at `self.dlg_data[data_key]['data']` and the plotted column
indices at `self.plotColIndex[data_key][sub_key]`. `available_columns` (`xmodfit.py:1290-1304`)
returns `[('x','x  [<plotted col>]'), (col, col), ...]`; `_column_values` (`1306-1317`) returns a
float array for a named column, special-casing `'x'` to read `self.data[key][sub]['x']`. The stated
ALIGNMENT INVARIANT (`1286-1289`) is that `externalData[key]['x']` is a straight copy of a
DataFrame column, so every column is row-aligned 1:1 with every sub_key's x — hence limits keyed
`(data_key, column)` and not per sub_key. `_column_mask(data_key, limits=None, slice_=None)`
(`1319-1356`) then ANDs `v >= lo` and `v <= hi` over every non-`'x'` limited column; `'x'` is
deliberately excluded because the x window is already applied as the contiguous `imin:imax` slice
and doubling it would only be slower. If a column's length disagrees with x it prints a loud
message and returns `None` rather than masking the wrong rows (`1339-1344`). `slice_=(i0,i1)`
pre-slices the mask to match arrays the caller has already sliced — used by `update_plot`
(`6253-6255`, `6367`), whereas `doFit` passes FULL arrays (`1867`) and lets `perform_fit` do the
windowing.

**Engine side (`Fit_Routines.py`).** The engine takes a boolean mask and never learns about column
names; column names, DataFrames and limit dicts stay in the GUI layer.
* `Fit.__init__` / `set_x` gain `mask=` (`43`, `64`). `set_x(mask=None)` deliberately CLEARS a
  previously installed mask (`65-69`, `102`): `set_x` means "new data", and a mask sized for the
  old x is a silent misalignment. `set_mask` (`114-139`) validates length and accepts a
  dict-of-masks for dict-x data.
* `imask_for(key=None)` (`141-158`) returns the FROZEN window mask when there is one, else derives
  `mask[imin:imax+1]` on the fly. That is what lets both call paths work: `runFit`/`doFit` install
  a full-length mask and let `perform_fit` compute `imin/imax`, while `update_plot` pre-slices the
  arrays and calls `residual()` directly with `imin=0`.
* `_freeze_imask` (`160-171`) is called ONCE per `perform_fit` (`431`) and never inside
  `residual()`. **Load-bearing:** lmfit's `leastsq` requires a constant residual length across
  iterations, so a mask that could be recomputed per iteration is a latent crash.
* `_count_active` (`173-184`) sets `n_active` (`432`); a `ValueError` fires when window + limits
  leave 0 points (`433-436`), so the failure is a message rather than an empty-array exception
  deep inside lmfit. `active_index(key=None)` (`186-198`) is the single source of truth for "what
  is in the fit" — `arange(imin, imax+1)` filtered by the window mask — used by the MCMC
  fingerprint and available for honest chi2/dof reporting.

**Where the mask is applied inside `residual`.** `Fit_Routines.py:266-271` (dict branch) and
`291-296` (scalar branch), and crucially **before** `maxd = np.max(data)` at `272`. Two
independent reasons, both in the comment at `259-265`: (1) `maxd` is a data-derived normalisation,
so an excluded hot pixel would otherwise still rescale the whole residual vector — exactly the
failure the window exists to prevent; (2) the `Log` branch takes `log10(data)` and an excluded
non-positive row gives `-inf`/NaN, which `nan_policy='raise'` (`443`, `446`, `448`, `455`) turns
into an aborted fit. Hence the rule **drop, don't NaN**: excluded points are removed from the
residual vector, never zeroed or NaN'd.

**Window unification.** `_window_indices(x, lo, hi)` (`Fit_Routines.py:8-38`) replaced three
inconsistent copies. The dict branch of `perform_fit` and `JointFit._set_ranges` used
`0.99*lo` while the scalar branch used `lo` exactly; for a NEGATIVE `lo` the 0.99 factor moves the
bound the wrong way, so the two branches disagreed in opposite directions. The new function is
exact, clamps a degenerate/non-overlapping window to the full extent and PRINTS that it did
(`33-37`) — more forgiving than the old individual path (which raised `IndexError`), so it says so
rather than silently fitting everything. `Joint_Fit_Routines.py:43` imports it, so there is one
definition.

**Covariance composition order** (the riskiest edit; lives in `handoff/01-joint-fit.md`'s file but
is part of this design): slice `[imin,imax]` -> apply `imask` -> concatenate -> apply `_cov_mask`.
`_cov_mask` is DEFINED against the concatenation, so `imask` must be folded in symmetrically in
both `_build_chol` and `residual` — see `Joint_Fit_Routines.py:475-480`, `521`, `552`, with a
hard length guard at `Joint_Fit_Routines.py:532-537` because getting it wrong misaligns the
whitening silently.

**MCMC fingerprint.** `_mcmc_fingerprint` (`xmodfit.py:2871-2909`) now hashes the ACTIVE points
(`data = np.asarray(self.fit.y)[self.fit.active_index()]`, `xmodfit.py:2899`; joint equivalent at
`2890-2895`) instead of the `imin:imax` window. A column limit changes which points are fitted
WITHOUT changing `imin/imax`, so hashing the window would let a backend `.h5` from an unmasked run
be silently accepted and continued for a masked one — the one failure mode here that corrupts
results rather than merely confusing a user. Cross-reference `handoff/02-*` / `03-*`.

**Opt-in `__masked_x__`.** A model may declare `__masked_x__ = True` to receive ONLY the active
points, saving the evaluation of the excluded ones (comment at `Fit_Routines.py:202-208` claims
~8x for a 2-D map windowed from 39k rows to 5k). Honoured in `_install_func_x`
(`Fit_Routines.py:213-234`), split out of `evaluate()` so `residual_jac` can reproduce exactly the
same abscissa without also evaluating the model; `residual`/`residual_jac` then skip re-masking via
the `_yfit_masked` flag (`270-271`, `295-296`, `361-362`). **Unsafe** for any model that integrates
or differences over `self.x` (`np.trapz`, `np.diff`, FFT) or anchors on `self.x[0]`/`self.x[-1]`.
Only declarer: `Functions/GISAXS/GID2D.py:24`, whose `row_lookup` is nearest+clip so a gappy x is
fine (see `handoff/06-*`).

**Persistence.** A separate optional line, not an extension of `#Fit Range=`, so old and new files
load in either direction: the loader is an if/elif chain of substring tests, so an older XModFit
just ignores `#Column Limits=`. Format `"qxy:0.9:1.7;qz::0.8"`, written by `_format_column_limits`
(`xmodfit.py:1363-1371`), parsed by `_parse_column_limits` (`1373-1394`), installed by
`_apply_column_limits` (`1396-1407`). Wired into `saveParameters` (`4938-4940`), `loadParameters`
(`5037-5040`), the `.xfit` header (`1962-1964`) and the joint `.jpar` (`7309`, `7329`, `7405`,
`7421`).

**GUI.** `openColumnLimitsDialog` (`xmodfit.py:1454-1464`) opens a non-modal `ColumnLimitsDialog`
(`120-237`) behind a `limitsButton` next to the Xmin:Xmax line edit (`970-975`). Row 0 is the
reserved `'x'` row and is two-way bound to the line edit, so there is exactly one x window and not
two: `set_column_limit(col, which, value)` (`1409-1434`) routes `col == 'x'` through the
`xmin`/`xmax` proxies and rewrites the line edit from the min/max across the selection.
`_refresh_limits_badge` (`1436-1452`) shows the count of extra limits on the button and turns it
RED when a limit has emptied a dataset, so "0 active" is visible BEFORE hitting Fit; also re-run on
every selection change (`1057-1059`). `_joint_masks(spec)` (`1466-1481`) builds
`{dataset_name: mask or None}` for `spec.compile()`, because the spec only receives
`data[name][sub]` and has no access to `dlg_data`; a per-dataset limit set in the joint panel wins
over the main-window one.

**Workflow enabled.** Fit only the `0.9 < q_xy < 1.7`, `q_z < 0.8` rectangle of a 2-D GID map, or
exclude a beam-stop shadow / hot detector row by any column the file carries (incident angle,
monitor count, temperature) without editing the data file.

### 3b. Per-dataset fit memory and multi-dataset workflow

**Store.** `self.dataset_fits[name] -> Fit`, keyed by the composite `"<num><>filepath"` dataset key
(`xmodfit.py:541`), plus `self._template_fit` (`542`) for the model shown with no dataset selected,
which seeds newly added datasets. `_clone_fit` (`6962-6989`) deep-copies only the model (`func`),
never the `QObject`, and rewires a fresh `Fit` around it WITHOUT re-running `init_params()` (which
would reset the values being carried over); returns `None` for a `JointFit` (no single `.func`).
Provenance attrs `_xmod_category` / `_xmod_module` / `_xmod_funcname` / `_xmod_xexpr` are stamped
in `update_parameters` (`5522-5526`) and copied by the clone so the function list can auto-follow.
`_fit_for_dataset(name, seed=None)` (`6991-7008`) returns the stored `Fit` or lazily clones from
`seed` -> `self.fit` -> `self._template_fit`, returning `None` on a fresh app so callers fall back
to the single-`Fit` path and the feature stays transparent until a model exists.
`_activate_dataset_fit(fit)` (`7010-7027`) points `self.fit` at it and resyncs `mpar_keys`/`mpar_N`
plus both parameter tables — necessary because selected datasets may carry DIFFERENT models, so
the tables cannot be assumed to match `self.fit` after a swap.

**Selection path (order is load-bearing).** `dataFileSelectionChanged` (`xmodfit.py:1026-1129`):
clear stale no-fit indicator -> rebuild `sfnames`/`pfnames` -> (a) point `self.fit` at the stored
`Fit` -> (b) repopulate the tables from it -> (c) `update_plot()` (tables -> fit -> curve), all
under the `_switching_dataset` re-entrancy guard (`1080-1097`). The dataset's own Xrange
expression is restored from `_xmod_xexpr` first (`1083-1090`).

**Sequential multi-dataset fitting + selection persistence.** `doFit` (`1721-2151`) loops the
selected items (`1827`) and for each one calls `_fit_for_dataset(fname)` +
`_activate_dataset_fit` (`1839-1841`) BEFORE `set_x` and `runFit`, so each dataset is fitted from
its OWN current parameters and the post-fit table readback uses matching keys. Routing recap:
joint-panel selection -> `run_joint_fit` for the whole spec (`1780-1789`); Data-list selection ->
one individual fit per selected dataset; `emcee` blocked for >1 dataset (`1817-1821`). The loop
deselects each dataset as it advances (`2124`), which used to leave only the LAST one selected;
`doFit:2140-2148` now re-selects ALL originally selected items with signals suppressed and fires
one `dataFileSelectionChanged()`, so the multi-selection persists and every freshly fitted
curve/EDP stays co-plotted. `_refresh_data_labels()` (`2151`, defined `1593-1620`) marks fitted
items bold + green with a tooltip noting joint-fit membership, and never touches `item.text()`
(that string is the dataset key parsed throughout the app).

**Detail dataset.** `DataListWidget` (`xmodfit.py:414-450`, installed at `949`) overrides
`mousePressEvent`: a plain left click on an item ALREADY in a multi-selection routes to
`_set_detail_dataset(name)` and does NOT collapse the selection (all curves/EDPs stay); a plain
click on a new item collapses (Qt default); Ctrl+click toggles (Qt default). `mouseDoubleClickEvent`
keeps double-click -> `openDataDialog` (the `itemDoubleClicked` connection was removed, `947-951`).
`_set_detail_dataset` (`1541-1568`) reorders `sfnames` so the clicked dataset is `[-1]` (that index
drives the detail everywhere), re-syncs the function list and the tables, and refreshes the CI
button + labels.

**No-fit indicator.** If the clicked dataset has no individual fit, `_set_detail_dataset` leaves
the parameter tables UNCHANGED and shows the indicator instead (`1550-1553`).
`_show_nofit_indicator` (`1174-1196`) sets an amber banner (`noFitBanner`, created at `4314-4319`
above the fixed-parameter tables) AND an amber `#fff8e1` tint on the fixed, single-fit and every
multi-fit table; `_clear_nofit_indicator` (`1198-1210`) undoes both. Lifecycle:
`dataFileSelectionChanged` clears it on any normal selection change (`1030`);
`clear_individual_selection` (`1131-1146`) also clears it EXPLICITLY, because it uses
`blockSignals` so `dataFileSelectionChanged` never fires — without that the banner and tint leaked
from an individual multi-select into a pure joint view (only the mfit/layers tables reset, because
they are rebuilt). A Ctrl overlay does NOT call `clear_individual_selection`, so the banner
correctly persists while an unfit individual dataset is still shown in the combined view.
Separately, `dataFileSelectionChanged:1039-1050` handles the FIRST addition of an unfit dataset to
a multi-selection: it reorders `sfnames` to put the last FITTED dataset at `[-1]` (keeping the
tables/chisqr on a real fit) and records `_pending_nofit`, then shows the banner after activation
(`1101-1102`). Single-select of an unfit dataset is unchanged — that is normal fit setup.
`_dataset_is_fitted(name)` (`1212-1216`) is the predicate: a stored `Fit` that is not a `JointFit`
and has `_is_fitted` or a `result`.

**Related plumbing.** `functionChanged` (`5475-5491`) treats picking a function as "use this model
for the selected dataset(s)" and stores an independent clone per selected dataset, the detail
dataset owning the live `self.fit`; skipped while `_loading_params` is set (`5008`, `5185`),
because `loadParameters` persists the loaded model to the detail dataset itself (`5160-5178`) and
must not overwrite the other datasets' fits with a blank template. A loaded `.par` counts as fitted
(`_is_fitted=True`), so CIs become available via `_update_ci_button_state` (`2160-2171`), with a
baseline leastsq run on demand by `_ensure_ci_baseline` (`2173+`). `removeData` drops the
remembered fit and the limits (`4283`, `4286-4288`).

### 3c. Plotting and legends

**`PlotWidget.py`.**
* **Data-line toggle.** New `dataLineCheckBox` ('Data line', `PlotWidget.py:66-69`, added to the
  option row at `82`), DEFAULT OFF = markers-only for DATA so data does not read as a second fit
  curve when a fit is overlaid; FIT curves are always lines. This changes the stock default
  appearance (upstream drew data as `.-`). `dataLineChanged` (`179-197`) re-pens EVERY curve then
  re-Plots — `Plot()` alone re-adds existing items without recomputing pens, so the pen must be
  rebuilt here for the toggle to take effect.
* **`_pen_for(dname, color)`** (`163-177`) centralises pen logic: `None` for a DATA curve when the
  toggle is off or width is 0 (pyqtgraph then draws symbols with no line), else a pen honouring
  width and the per-curve `linestyle`. Called from both `add_data` branches (`229`, `251`) and
  `updatePlot` (`455`), replacing the old inline `lineWidthLineEdit.text()=='0'` special case.
* **`linestyle`** — `self.linestyle[dname]` (`36`) plus an `add_data(..., linestyle=)` kwarg
  (`202`, stored at `225`), a `Qt.PenStyle`; used to draw the individual fit dashed against the
  solid joint fit. The matplotlib branch maps to `'-'` / `'.-'` / `'.'` (`318-322`).
* **`legend_formatter`** (`39`) is an optional `callable(key) -> label`; `_legend_text` (`151-161`)
  applies it with a raw-key fallback, used in the pyqtgraph `legendItem.addItem` (`415`) and the
  two matplotlib `label=` sites (`334`, `344`).
* **The bug `legend_formatter` caused, and the fix.** `Plot()` used to clear old entries with
  `legendItem.removeItem(name)`, which matches by the DISPLAYED label. Once the label differed from
  the curve key, per-name removal silently failed and the legend accumulated EVERY curve ever
  plotted, duplicated and out of order. Fixed by rebuilding the legend from scratch each redraw
  with `legendItem.clear()` (`366-379`), falling back to removal by item reference for older
  pyqtgraph without `LegendItem.clear()`.
* Committed separately (`c5f697d`): the color-less fallback is a deterministic Tableau-10 palette
  cycled by `_palette_ct` (`244-249`) instead of `pg.intColor(np.random.choice(...))`.

**`xmodfit.py` legends.** `_curve_meta(name)` (`1218-1237`) is the shared parser: it maps a
plotted-curve key to `(dataset_index, type_rank, dataset_key)` recognising `N:col` (data, rank 0),
`... [indiv:N]` (rank 1), `... [joint:N]` (rank 2) and the bottom-panel `... [N]` suffix (rank 0),
via `_dataset_key_for_index` (`1239-1247`). `_legend_order(names)` (`1532-1539`) sorts by dataset
index then rank so a legend reads data < individual fit < joint fit per dataset; applied at every
`Plot(...)` call site (`6141`, `6384`, `6493`, `6593`, `6645`, bottom panel `6838`).
`_legend_label(name)` (`1249-1264`) is installed as the MAIN plot's formatter only
(`create_plotDock`, `xmodfit.py:5207`) — the bottom generated-param plot keeps its raw labels,
which encode param + dataset and already read correctly. It reuses the pre-existing `_short_name`
(`7696-7702`, `'num: basename'`, matching the Data boxes), truncates to 24 chars and tags
`(fit)`/`(joint)`. `_indiv_fit_name(fnum, funcname)` (`1266-1278`) builds `<model> [indiv:<fnum>]`
so an individual-view fit curve carries its dataset index and therefore sorts and labels correctly
(previously the detail fit was a bare `<model>`, unindexed, and got dumped at the end of the
legend); used in `update_plot` (`6296`) and `xChanged` (`6135`). Colors: `_curve_color_for`
(`1570-1578`) assigns a deterministic Tableau-10 color per `(dataset, curve)` in first-seen order,
`_single_fit_color` (`1580-1591`) makes a single fit take its data curve's color, and `addData`
stores the assigned color back into `self.data[...]['color']` (`4225-4228`) so every fit path
reuses it.

**Combined individual+joint overlay (item B2).** The two dataset lists (main `dataListWidget` and
the joint panel's `datasetList`) are mutually exclusive on a PLAIN click — classic one-view
behaviour — but Ctrl+click keeps the other list's selection and OVERLAYS. The gate is
`_overlay_modifier_active()` (`1148-1158`), which reads `QApplication.keyboardModifiers() &
Qt.ControlModifier` and is isolated in one method so headless tests can monkeypatch it.
`dataFileSelectionChanged:1118-1128`: with the modifier -> `_maybe_combined_overlay()`, else clear
the joint list's selection. `_maybe_combined_overlay` (`6504-6521`) is a no-op unless BOTH lists
have a selection and a `jointFit` exists. `_plot_combined_overlay(main_sel, joint_sel)`
(`6421-6502`) draws each dataset's data once (`_add_data_curves`, dedup via `shown`), each
joint-selected dataset's JOINT curve solid as `<Model> [joint:<fnum>]` (`6451-6473`), and each
Data-list-selected dataset's INDIVIDUAL curve DASHED (`linestyle=Qt.DashLine`) as
`<Model> [indiv:<fnum>]` in the same dataset color (`6476-6487`). Purely additive: it never
changes `self.fit` or the parameter tables, and it deliberately does NOT overwrite `self.pfnames`
(`6491-6492`) so `remove_data(self.pfnames)` flows never strip the overlaid fit curves.
`_eval_individual_curve(name)` (`6388-6419`) evaluates the stored `dataset_fits[name].func` in
ISOLATION over the dataset's own fit range, returning `(x, y, color)` or `None`; it never touches
`self.fit` or the tables. `clear_joint_view()` (`1160-1172`) makes a blank click / deselect-all in
the JOINT list clear the joint overlay, falling back to the individual view if the Data list still
has a selection, else `plotWidget.Plot([])` — symmetric with a blank click in the Data list. Note
the individual view's fits also use `[indiv:N]` keys and are SOLID; only the combined-overlay
individual curve is dashed (same key, `_plot_combined_overlay` draws last so its linestyle wins).

**Multi-dataset electron-density-profile co-plotting.** `plot_extra_param` (`6711-6863`) builds a
`sources` list of `(output_params, color, linestyle, name_suffix)` MIRRORING the data plot's
overlay model: (a) combined view (a dataset Ctrl-selected in BOTH lists) -> each dataset's JOINT
profile solid `[joint:N]` and its INDIVIDUAL profile dashed `[indiv:N]` in the same dataset color
(`6738-6746`); (b) joint view with >1 dataset overlaid -> one solid profile per dataset `[N]`
(`6751-6755`), driven by `self._joint_overlay_names` recorded in `_update_plot_joint`
(`xmodfit.py:546`, `6523+`); (c) individual multi-select -> one solid profile per FITTED selected
dataset `[N]` (`6756-6767`); (d) otherwise the single `self.fit` curve (`6768-6769`). In branch
(c) every contributor requires a fit (`_dataset_is_fitted`, `6763`) because an unfit dataset's
model holds a clone of ANOTHER dataset's parameters and would plot the wrong profile in the wrong
color; single-select is unchanged so initial values still plot during fit setup.
`_model_output_params(name, which)` (`6688-6709`) fetches a model's `output_params` READ-ONLY —
the models were freshly evaluated by the plot that populated the generated-param list.
Ordering matters twice: `update_plot:6380-6383` re-calls `plot_extra_param()` AFTER its
multi-select overlay loop (which evaluates each dataset via `_dfit.evaluate()`), and
`_plot_combined_overlay:6497-6500` calls it at its END after evaluating both models — otherwise
the bottom panel would render from stale `output_params`.

**EDP-selection reset fix.** The generated-parameter selection is restored by matching item text
against `self.gen_param_items`. The restore used to be gated `if not self.fchanged:`, but
`fchanged` stays True from function-pick until the FIRST fit, so ADDING a dataset before fitting
wiped the EDP selection. The gate is now `if self.gen_param_items:` at all three sites — `xChanged`
(`xmodfit.py:6115`), `update_plot` (`6323`), `_update_plot_joint` (`6633`) — because
`functionChanged` already clears `gen_param_items` (`5448`) and the restore matches by text, so a
genuine function change restores nothing while a same-model dataset add/switch always restores.

**Workflows enabled.** Overlay a dataset's individual best fit against its joint best fit, in both
the R(q) panel and the electron-density panel, to see what sharing cost; co-plot the EDPs of a
whole film series from one multi-selection; read a legend saying "3: filmA.txt (joint)" instead of
"3:Col_1".

### 3d. Fit engine miscellany

**`Fit._canonicalize_result`** (`Fit_Routines.py:466-484`), called after `minimize` on both the
least-squares and emcee paths (`458`, `463`), before `fit_report`. It calls the model's optional
`canonical_params(params)` on both `result.params` and `fit_params`. Why a reported angle could be
outside its window: `XLayersPDB_XPR`'s Euler angles are deliberately UNBOUNDED (finite bounds
would be a reflecting wall for emcee at the poles and at the phi wrap), and the fold inside `y()`
is LOCAL — lmfit sets the parameters from its own variable vector every residual call and writes
that vector into `result.params` at the end, so the fold is discarded and the reported angle is
the raw optimiser coordinate. The notes report a reproduced case of theta=379.3, phi=-8208.2 for a
fit that was evaluating 19.3 / 71.8 with identical R. The canonicalisation re-labels an identical
model, so chi2 and the residual are untouched; exceptions are swallowed (`483-484`) so a badly
behaved hook cannot lose a completed fit. Only implementer:
`Functions/XRR/XLayersPDB_XPR.py:247`.

**Analytic Jacobian.** `can_jac(fit_params, fit_scale)` (`Fit_Routines.py:315-336`) gates on:
`self.use_jacobian` (opt-out flag, `Fit_Routines.py:54`); NOT a dict-x/joint fit (`319-320`, "not
wired up"); the model exposes `dydp` (`321-322`); no parameter has an `expr` (constraints would
need a chain rule that is skipped, `323-324`); at least 4 varying parameters (`330-331` — measured
on `XLayersPDB_XPR` as a clear win from 4 up and a loss at 2, because one AD pass costs about what
the `nvarys+1` finite differences do and the model pays a one-off Julia specialisation); and
`parameter_columns(names)` not returning `None` (`332-336` — e.g. the first/last layer's `rho` is
baked into the Julia `SimData` as `Float64` and is unmappable, so the whole fit falls back to
finite differences). `residual_jac` (`338-373`) mirrors `residual`'s scaling term by term for all
four fit scales (derivations in the docstring, `340-347`) and applies the same mask. It is wired
in for **`leastsq` only** (`437-443`): lmfit wraps `Dfun` in `Minimizer.__jacobian`, which applies
the bounded->internal gradient rescaling, whereas the `least_squares` path takes its own `jac` with
a different signature and is left on finite differences. Measured effect (notes): 261 -> 41 model
calls, same optimum. Only consumer:
`Functions/XRR/XLayersPDB_XPR.py:441` (`parameter_columns`) and `:526` (`dydp`) — see
`handoff/05-xpr-julia.md`.

**`openFileDlg` signature + the `filepaths` rework.** Upstream was `openFileDlg(self, fname=None)`,
reading the row from `self.sender().pos()`. Now
`openFileDlg(self, fname=None, key=None, prompt=True)` (`xmodfit.py:5598-5643`). `key` must be
supplied when called directly rather than from a button click, because `self.sender()` is then not
the cell button, so `indexAt(pos()).row()` returns -1 and `item(-1, 0)` is `None` — which used to
raise `AttributeError` and abort loading the whole function, hit by any model whose filepath
default does not exist on disk. `prompt=False` syncs the table cell without opening a dialog,
which is what `update_fixed_parameters` needs when there is no usable default. In the same area:
`self.filebutton_paths` (`5546`) keeps the FULL path while the button shows only the basename with
the full path as a tooltip (`5575-5584`, `5636-5642`); `update_fixed_parameters` prefers the LIVE
attribute value over the `__init__`-time `filepaths` snapshot (`5568-5575`) so a per-dataset PDB
choice survives dataset switching and cloning; `fixedParamChanged` (`5810-5814`) and
`saveParameters` (`4951`) read the full path from `filebutton_paths` instead of stripping
`'[Click] '` off the button text. A dead re-entrant `self.openFileDlg(fname=None)` inside the
table rebuild was removed, reason recorded at `5585-5593`.

**Robustness fixes in existing methods** (mostly commit `babe1aa`, described in
`merge-with-upstream-develop.md`): `update_plot` skips rows whose item or combobox is not installed
yet (`6158-6167`); `closeFitInfoDlg` (`3459-3465`) guards on the dialog existing and nulls it (only
the non-emcee branch of `showFitInfoDlg` creates it, and `doFit` calls it unconditionally);
`stopFit` (`3450-3457`) is idempotent instead of disconnecting the Stop button; `showFitInfoDlg`
(`3428-3448`) shows the modal wait dialog for a joint fit too; `loadParameters` uses
`parname[2:].rsplit('_', 2)` (`5118`, `5126`) to tolerate namespaced `d0_Model` mpar keys.

**The `choices` / fixed-parameter `eval()` gotcha (stock behaviour, unchanged, recorded here
because it is a core-app trap).** `fixedParamChanged` (`xmodfit.py:5803-5806`) and `update_plot`
(`6172-6175`) both coerce a Fixed-Parameters cell with `eval(val)` and only fall back to
`str(val)` on exception. So any `self.choices` value (or string fixed parameter) whose text is a
valid Python expression comes back as THAT OBJECT, not the string. Found 2026-08-27: `GID2D`
shipped `choices={'mode': ['map', 'inplane', 'rod']}` and `func.mode` arrived as `<class 'map'>`,
the builtin; it only appeared to work because map mode was the fall-through branch of `y()`. The
value was renamed to `'map2d'`. `'inplane'`, `'rod'`, `'Gaussian'`, `'kinematic'` are all
`NameError`s and survive as strings; `'True'`/`'False'` eval to bools, which is intended. It fails
silently — no exception, no wrong-looking value in the GUI, just a parameter that never compares
equal to the string you tested against. **Rule:** never use a `choices` value that is a Python
builtin, keyword or literal (`map`, `list`, `set`, `type`, `id`, `all`, `any`, `sum`, `min`, `max`,
`open`, `int`, `str`, `None`, `nan`, `e`, `inf`, anything numeric, ...), and defensively normalise
in `y()` with `getattr(v, '__name__', None) or str(v)`.

### Unclassified changes

Changes swept out of the diffs that belong to no category in this handoff set:

* **Model-comparison dock** — `create_compareDock` (`xmodfit.py:7507-7523`), `_joint_range`
  (`7525-7530`), `_eval_chisqr` (`7532-7551`), `computeModelComparison` (`7553-7694`) with a
  nested `_ic` (`7611`); its own `Dock('Model Comparison')` (`514-518`, `593-600`). Compares a
  joint fit against the collection of individual fits with an F-test / likelihood-ratio test when
  nested, plus AIC/AICc/BIC always. Committed in `c5f697d`. Built on per-dataset fit memory but its
  subject is joint-vs-individual, so it arguably belongs in `handoff/01-joint-fit.md` — recorded
  here so it is not lost.
* **`saveGenParameters` image branch** (`xmodfit.py:4897-4907`) — a 2-D map entry has 1-D axes and
  a 2-D `z`; the legacy branch does `x.shape[1]`, so without this an image entry raised
  `IndexError` out of a slot with no try/except. -> `handoff/06-*`.
* **`OP_META_KEYS`** (`xmodfit.py:70`) and `_op_var_list` / `_op_is_image` (`1515-1531`) —
  generated-parameter metadata filtering; `'meta'` is deliberately ABSENT from `OP_META_KEYS`
  because `saveGenParameters` branches on the literal strings `"['x', 'y', 'meta']"` /
  `"['x', 'y', 'z', 'meta']"`. -> `handoff/06-*`.
* **File menu additions** (`xmodfit.py:612-623`): Save/Load Joint Fit (.jpar), Load .par into Joint
  Fit. -> `handoff/01-*`.
* **`NameDelegate` on the main Data list** (`xmodfit.py:949-950`) — paints `'<num>: <basename>'`
  while leaving `item.text()` (the parsed composite key) untouched. The class lives in
  `joint_fit_panel.py:63-68`, so the core app now imports from a joint-fit file (`xmodfit.py:58`).
* **`nan_policy` `'raise'` -> `'propagate'` for the emcee `Minimizer`s only**
  (`Fit_Routines.py:506`, `512`), the `fiterr` `stderr is None` guard, and the funcname tuple
  widened to `('XLayersPDB', 'XLayersPDB_dev_smooth', 'XLayersPDB_XPR')` (`491-498`). Committed;
  -> `handoff/02-*`/`03-*`.

## 4. How to use it

**Environment.** `C:\Users\kerr\.conda\envs\cmc-py39\python.exe`, headless with
`QT_QPA_PLATFORM=offscreen` (see the `cmc-py39-env` memory).

**Per-column fit limits (GUI).** Select one or more datasets. In the Fit dock, next to
`Xmin:Xmax`, click **Limits**. Row `x` mirrors the `Xmin:Xmax` line edit; every other row is a
column of the loaded file. Blank = unbounded; a non-numeric entry reverts; `<varies>` when the
selected datasets disagree. Edits apply to EVERY selected dataset. The status line reports
`N extra column limit(s) | M active point(s)`; **Clear all** drops the non-x limits and keeps the x
window. Limits are per-dataset and are dropped with the dataset. Persistence: `.xpar` / `.xfit`
header line `#Column Limits=qxy:0.9:1.7;qz::0.8` (empty field = unbounded on that side), written
only when a non-x limit exists so unlimited files stay byte-identical; joint `.jpar` blocks carry
`collimits`. Old and new files load in either direction.

**Model author's contract.** `__masked_x__ = True` on the model class -> receive only the active
points in `self.x` (only if `y()` is pointwise in x or looks x up as an index).
`fit.use_jacobian = False` -> force finite differences. `dydp(names) -> (npts, len(names))` plus
`parameter_columns(names) -> None`-when-unmappable -> analytic Jacobian (`leastsq` only, >= 4
varying params, no `expr`). `canonical_params(params)` -> fold periodic params before reporting.

**Single-fit covariance (added 2026-11-10, COV T7; design `17-spec-general-covariance.md`
§6.2/§6.6, plan `18-plan-general-covariance.md`).** Select ONE dataset in the Data list and
click **Load covariance...** (`*.cov.json` v1 from LSS_Reader, a legacy `*.npz`, or a
`*_c_factors.txt`, which then asks for its raw 4-column file). `attach_covariance_to(data_key,
path, raw_path)` refuses, with a message, a multi-curve dataset, an unreadable file and any x/y
mismatch (rtol 1e-4, naming the first mismatching index), and warns when the file names a
different data file. The covariance belongs to the dataset (`XModFit.covariances[data_key]`,
`covariance_paths`, `covariance_modes`), not to the one `Fit` the GUI moves between datasets: a
dataset without one fits exactly as before. Attached datasets are bold with a tooltip naming the
file; **Remove covariance** detaches; the **Covariance Mode** combo (`model` default / `data`) is
per dataset and enabled only with a covariance. With a covariance the fit uses absolute errors
(`scale_covar=False`, χ²_red reported next to them) and the χ² confidence interval targets
χ²_min + χ².ppf(CL, 1). Persistence: `.xpar` / `.xfit` header lines `#Covariance=<path>`,
`#Covariance Raw=<path>` (legacy text only) and `#Covariance Mode=<model|data>`, written only
with a covariance so other files stay byte-identical. An `.xpar` with `#Covariance=` and no mode
line reloads in `data` mode and the next fit's report says "saved before covariance modes
existed; reloaded in data mode"; a missing covariance or raw file warns and the dataset
continues without one. The joint panel inherits these attachments (doc 01 §3.4). Verified:
`validate_covariance_gui.py` G1-G5 (single fit) and J2 (joint pre-fill), headless.

**Multi-dataset individual fitting.** Ctrl+click several datasets in the Data list, then press
**Fit**: each is fitted in turn from its own stored parameters (progress dialog with Abort), and
the whole selection is restored afterwards so all fitted curves and EDPs stay co-plotted. Fitted
datasets appear bold + green. Within a multi-selection, a PLAIN click on an already-selected
dataset switches only which dataset's parameters fill the tables; if that dataset has no fit the
tables are left alone and an amber banner + table tint say so.

**Individual vs joint overlay.** With a completed joint fit, select datasets in the joint panel
list, then **Ctrl+click** the same dataset(s) in the main Data list (or vice versa). The main plot
shows data + joint fit (solid) + individual fit (dashed) in one color per dataset, and the bottom
generated-parameter panel mirrors it for the EDP or any other selected profile. A plain click in
either list returns to a single view; a blank click in the joint list clears the joint overlay.
The **Data line** checkbox in the plot option row adds connecting lines through data points
(default off = markers only; fit curves are always lines).

**Headless regressions** (all `QT_QPA_PLATFORM=offscreen python <file>` from the repo root):
`validate_fit_mask.py` (engine mask, freezing, `_window_indices`), `validate_column_limits.py`
(limits store, xmin/xmax views, column discovery, badge, `.xpar`/`.jpar` round-trip),
`validate_masked_x.py` (`__masked_x__` and the honest-dof check), `validate_perdataset_fit.py`
(per-dataset fit memory + PDB cache), `validate_jointfit_gui.py` (plain-click/Ctrl-overlay
selection model; monkeypatches `w._overlay_modifier_active`). Also relevant:
`validate_joint_mask.py`, `validate_xlayerspdb_joint.py`. Any headless test that triggers a spawn
`Pool` needs a `__main__` guard.

## 5. Contact surface with stock XModFit

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|
| `Fit_Routines.py` | module scope | new `_window_indices` (`8-38`) | additive |
| `Fit_Routines.py` | `Fit.__init__` (`43-57`) | signature `+mask=`; new `use_jacobian=True` | additive |
| `Fit_Routines.py` | `Fit.set_x` (`64-102`) | signature `+mask=`; calls `set_mask(mask)`, which CLEARS a stale mask | **behaviour change** |
| `Fit_Routines.py` | `Fit` new methods (`114-198`) | `set_mask`, `imask_for`, `_freeze_imask`, `_count_active`, `active_index` | additive |
| `Fit_Routines.py` | `Fit.evaluate` (`200-211`) | body split out into `_install_func_x` (`213-234`), which may mask x | additive (opt-in) |
| `Fit_Routines.py` | `Fit.residual` (`247-307`) | mask applied in BOTH branches BEFORE `maxd` (`266-271`, `291-296`) | **rewrite of a hot path** |
| `Fit_Routines.py` | `Fit` new methods (`315-373`) | `can_jac`, `residual_jac` | additive |
| `Fit_Routines.py` | `Fit.perform_fit` (`415-464`) | `+seed_temperature`, `+mask`; `_window_indices` replaces the two inline `np.where` forms incl. `0.99*xmin`; freeze + `n_active` + 0-point `ValueError`; `Dfun=` for `leastsq`; `_canonicalize_result()` on both paths | **rewrite** |
| `Fit_Routines.py` | new `_canonicalize_result` (`466-484`) | model hook before `fit_report` | additive |
| `Fit_Routines.py` | `set_emcee_minimizer` (`487-516`) | `+seed_temperature`; funcname tuple widened; `stderr is None` guard; `nan_policy` `raise`->`propagate` | mixed — see `handoff/02-*` |
| `PlotWidget.py` | `PlotWidget.__init__` (`33-39`) | new `self.linestyle`, `self.legend_formatter` | additive |
| `PlotWidget.py` | `createPlotWidget` (`59-82`) | new `dataLineCheckBox` in the option row | additive, but changes the DEFAULT data appearance to markers-only |
| `PlotWidget.py` | new `_legend_text`, `_pen_for`, `dataLineChanged` (`151-197`) | pen/label centralisation | additive |
| `PlotWidget.py` | `add_data` (`202-271`) | `+linestyle=` kwarg; both pen constructions go through `_pen_for`; deterministic palette fallback (`244-249`) | additive + **behaviour change** on the random color |
| `PlotWidget.py` | `Plot` matplotlib branch (`318-344`) | `plot_type` honours the toggle; `label=self._legend_text(dname)` | **behaviour change** |
| `PlotWidget.py` | `Plot` pyqtgraph branch (`366-379`, `415`) | legend rebuilt via `legendItem.clear()`; `addItem(..., _legend_text(dname))` | **rewrite of legend cleanup** |
| `PlotWidget.py` | `updatePlot` (`447-466`) | pen logic replaced by `_pen_for` | **rewrite** |
| `xmodfit.py` | imports (`3-5`, `49-59`) | `+QListWidgetItem, QTableWidget, QHeaderView, QDoubleSpinBox, QToolButton, QDialogButtonBox`, `+QObject, QEvent`, `+MutableMapping`, `+json`, `+hashlib`, `+CustomHDFBackend`, `+JointFitPanel, NameDelegate`, `+Image2DWidget` | additive; the last two are NEW-FILE dependencies |
| `xmodfit.py` | module scope (`70`, `73-237`, `414-450`) | `OP_META_KEYS`, `_LimitProxy`, `ColumnLimitsDialog`, `DataListWidget` | additive |
| `xmodfit.py` | `XModFit.__init__` (`514-600`) | `self.xmin={}`/`self.xmax={}` REPLACED by `self.limits={}` + two `_LimitProxy` views; `dataset_fits`, `_template_fit`, `_switching_dataset`, `jointFit`, `_joint_overlay_names`, `_mcmc_is_joint`, `seed_temperature`, `seed_mode`, `_curve_palette`, `_curve_color`, `_image_autoselected`; `compareDock`; `create_jointFitDock`/`create_compareDock` | **rewrite of the xmin/xmax store** (source-compatible), rest additive |
| `xmodfit.py` | `create_fitDock` (`958-975`) | new `limitsButton` at col 2 of the Xmin:Xmax row | additive |
| `xmodfit.py` | `create_dataDock` (`947-951`) | `QListWidget` -> `DataListWidget`; `NameDelegate` installed; `itemDoubleClicked` connection REMOVED (handled in the subclass) | **behaviour change** |
| `xmodfit.py` | `create_paramDock` (`4310-4320`) | `noFitBanner` inserted above the fixed-param tables | additive |
| `xmodfit.py` | `create_plotDock` (`5194-5224`) | plot area becomes a `QTabWidget` (1D + 2D Map); `plotWidget.legend_formatter=self._legend_label` | **rewrite** (2D part -> `handoff/06-*`) |
| `xmodfit.py` | `create_menus` (`612-623`) | three joint-fit file actions | additive (-> `handoff/01-*`) |
| `xmodfit.py` | `dataFileSelectionChanged` (`1026-1129`) | no-fit clear; unfit-in-multiselect reorder + `_pending_nofit`; limits badge + dialog refresh; per-dataset fit restore under `_switching_dataset`; CI button; Ctrl-overlay vs mutual-exclusion branch; `_refresh_data_labels` | **substantial rewrite** — highest conflict risk in the file |
| `xmodfit.py` | `doFit` (`1721-2151`) | joint-MCMC route hoisted above all guards; joint dispatch on `joint_active`; per-dataset activation in the loop; `set_x(..., mask=...)`; `temp_params` restore guarded for emcee; `#Column Limits=` in the `.xfit` header; selection re-selected after the loop; `_refresh_data_labels` | **substantial rewrite** |
| `xmodfit.py` | `runFit` (`3387-3425`) | `+seed_temperature=` passed to `perform_fit`; `_mcmc_is_pdb()` replaces `funcListWidget.currentItem().text()` checks | additive (-> `handoff/02-*`) |
| `xmodfit.py` | `update_plot` (`6150-6386`) | JointFit early return; None-guards on table items/comboboxes; `set_x(..., mask=_column_mask(..., slice_=(i0,i1)))`; `_indiv_fit_name`; `_single_fit_color`; `gen_param_items` gate; multi-select overlay loop; `_legend_order` | **substantial rewrite** |
| `xmodfit.py` | `xChanged` (`6066-6147`) | `_op_var_list`; `gen_param_items` gate; `_indiv_fit_name`; `_single_fit_color`; `_legend_order` | modified |
| `xmodfit.py` | `plot_extra_param` (`6711-6863`) | rewritten around a `sources` list; shape guards in `_plot_one`; image entries routed to the 2D panel; `_legend_order` | **rewrite** |
| `xmodfit.py` | `fitCallback` (`3469-3496`) | JointFit guard (`3481`); `_single_fit_color` | modified |
| `xmodfit.py` | `addData` (`4225-4232`) | deterministic color assigned and stored back into `self.data[...]['color']` | **behaviour change** (was the dialog's color) |
| `xmodfit.py` | `removeData` (`4283-4288`) | drops `dataset_fits[txt]` and `limits[txt]` | additive |
| `xmodfit.py` | `openDataDialog` (`1622-1674`) | unchanged logic, but writes `xmin`/`xmax` through the proxies | none (source-compatible) |
| `xmodfit.py` | `saveParameters` (`4936-4953`) | `#Column Limits=` line; filepaths value from `filebutton_paths` | additive + **behaviour change** on filepaths |
| `xmodfit.py` | `loadParameters` (`5008-5186`) | `_loading_params` flag; `#Column Limits=` branch; `rsplit('_', 2)` for mpar keys; loaded `.par` marked `_is_fitted` + stored per dataset | additive + **behaviour change** (CI now enabled after a load) |
| `xmodfit.py` | `saveGenParameters` (`4897-4907`) | image branch | additive (-> `handoff/06-*`) |
| `xmodfit.py` | `functionChanged` (`5447-5491`) | `_image_autoselected` reset; per-dataset model store gated on `_loading_params` | additive |
| `xmodfit.py` | `update_parameters` (`5520-5526`) | provenance attrs stamped on the new `Fit` | additive |
| `xmodfit.py` | `update_fixed_parameters` (`5544-5596`) | `filebutton_paths`; live-attribute preference; basename + tooltip; keyed `openFileDlg` lambdas; dead re-entrant call removed | **rewrite** of the filepaths rows |
| `xmodfit.py` | `openFileDlg` (`5598-5643`) | signature `(fname=None, key=None, prompt=True)`; row resolution by key; missing/blank default still opens a dialog | **rewrite** |
| `xmodfit.py` | `fixedParamChanged` (`5795-5834`) | filepaths value from `filebutton_paths`; `_fan_fixed_to_objectives(txt)` tail | additive (tail -> `handoff/01-*`) |
| `xmodfit.py` | `showFitInfoDlg` / `stopFit` / `closeFitInfoDlg` (`3428-3465`) | joint-aware modal; Stop connected once; Stop idempotent; dialog-existence guard | **behaviour change** (commit `babe1aa`) |
| `xmodfit.py` | `_mcmc_fingerprint` (`2871-2909`) | hashes `active_index()` points | **behaviour change** (-> `handoff/02-*`) |
| `Joint_Fit_Routines.py` | `_set_ranges`, `_build_chol`, `residual` | imports `_window_indices`; `imask` folded in symmetrically; length guard | -> `handoff/01-*` |

Merge notes for folding upstream. `dataFileSelectionChanged`, `doFit`, `update_plot`,
`plot_extra_param` and `create_plotDock` are the five regions where a textual conflict is close to
certain; all five are heavily commented in-place, which should help a three-way merge but will also
produce large hunks. `XModFit.__init__`'s `self.xmin={}` / `self.xmax={}` -> `_LimitProxy` swap is
a two-line change that MUST be taken from this fork or the whole limits feature silently
disappears while still importing. `Fit.residual`'s mask block sits immediately above `maxd`, which
is also where upstream's fit-scale branches live, so a conflict there needs care: the mask must
stay ABOVE `maxd`. `merge-with-upstream-develop.md` records that the earlier merge (`a4fd04e`) hit
exactly four content conflicts and that `Fit_Routines.py` was one of them, resolved by nesting the
fork's `stderr is None` guard inside upstream's `funcname=='XLayersPDB'` gate.

## 6. Known limitations, gotchas and TODOs

* **Never clicked through the live GUI:** per-column limits, the limits dialog and the badge. The
  headless regressions cover the store, the mask, the badge text and the file round-trips — not the
  dialog interaction.
* **`ColumnLimitsDialog(parent, names)` ignores `names`** (`xmodfit.py:134` is never read;
  `refresh()` reads `self.host.sfnames` and `openColumnLimitsDialog` passes `[]`). Dead parameter.
* **The dialog's column list comes from `sfnames[-1]` only** (`xmodfit.py:162`) while edits apply
  to EVERY selected dataset (`set_column_limit`, `1415`). Selecting datasets with different
  columns lets you write a limit on a column another dataset does not have; `_column_mask` then
  skips it silently (`1337-1338`).
* **`__masked_x__` and the 1-D plot paths (likely live bug).** With a mask AND `__masked_x__`,
  `self.fit.yfit` is shorter than `self.fit.x[imin:imax+1]`, but `update_plot`
  (`xmodfit.py:6297`) and `fitCallback` still pass that full slice as `x` to
  `PlotWidget.add_data`, whose length check (`PlotWidget.py:215`) fails and pops a modal "The
  dimensions of x, y or yerr are not matching" dialog (`PlotWidget.py:270`). The same slice feeds
  the `Residuals` output_params entry (`xmodfit.py:6301-6304`). Only `GID2D` declares
  `__masked_x__` and it renders in the 2D panel, which is presumably why this has not been hit —
  but it is a trap for the next `__masked_x__` model. Audit hook #6.
* **`fitCallback` still names the live curve with the bare model name** (not `_indiv_fit_name`), so
  during a fit the legend entry is unindexed, then replaced by the indexed one at the end.
* **`PlotWidget.remove_data` leaks bookkeeping**: deletes `self.data`/`self.dataErr` but not
  `self.linestyle`/`self.fit`/`self.yerr` (`PlotWidget.py:426-444`), and its
  `legendItem.removeItem(dname)` (`436`) is now a no-op for the same label-mismatch reason that
  forced `legendItem.clear()` in `Plot`. Harmless today because `Plot` rebuilds the legend.
* **Data line default off changes every plot's appearance** relative to stock XModFit. Deliberate,
  but it is the first thing an upstream user will notice.
* **`_window_indices` is forgiving where the old individual path raised.** A typo'd window that
  misses the data no longer raises `IndexError`; it fits EVERYTHING and prints to stdout. Run
  without a visible console, that message is lost.
* **Fit-button behaviour with a MIXED selection is surprising** (deferred by the user, 2026-08-25):
  `doFit` routes on `joint_active`, so if ANY joint dataset is selected — including a combined Ctrl
  overlay where individual datasets are also selected — Fit short-circuits to a fresh joint fit of
  the WHOLE joint spec and SILENTLY ignores the Data-list selection. No crash, just surprising.
  To be revisited during the planned overlay overhaul; related: the pending overlay UI cleanup
  (fold the joint overlay controls into the Fit-options panel) from
  `merge-with-upstream-develop.md`.
* **Analytic Jacobian is single-dataset only** — `can_jac` returns `False` for dict-x and joint
  fits (`Fit_Routines.py:319-320`) and is not wired to the `least_squares` method.
  `_canonicalize_result` swallows all exceptions (`483-484`), so a hook that raises silently leaves
  raw optimiser coordinates in the report.
* **Cosmetic:** `xmodfit.py:1058` is one very long line with a run of spaces where a line
  continuation was flattened. It parses, but it looks like a botched edit.
* **The "is this dataset fitted?" predicate is duplicated** in seven places in two slightly
  different forms — see section 8. The `choices` `eval()` trap (section 3d) is also still live.

## 7. Claims to verify (audit hooks)

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|
| 1 | The `0.99*xmin` -> exact-bound change shifts `imin` by ZERO points on real data at its own `#Fit Range=0.026:0.550`; it can shift by one point only when a datum falls in `[0.99*xmin, xmin)`, and only on the dict/joint paths | `column-fit-limits.md` | Load `Data/i2_tim3_1000_test.txt`, compute `_window_indices(x, 0.026, 0.55)` and the old `np.where(x>=0.99*0.026)[0][0]`; assert equal. Then construct an x containing a point in `[0.99*lo, lo)` and assert they differ by 1 |
| 2 | GID2D `window='model'` vs `window='engine'` give identical parameters and chi2, with redchi 0.131 -> 0.271 = exactly `n/n_active` | `column-fit-limits.md`, `gid2d-function.md` | `QT_QPA_PLATFORM=offscreen python validate_masked_x.py` and confirm the printed ratio equals `n/n_active` to floating point |
| 3 | `__masked_x__` gives ~8x on a 2-D map windowed from 39k rows to 5k | `Fit_Routines.py:202-208` comment | Time `Fit.evaluate()` on `GID2D` with `__masked_x__` True vs False at the same mask; report the ratio |
| 4 | Analytic Jacobian: 261 -> 41 model calls with the same optimum on `XLayersPDB_XPR`; a clear win from 4 varying parameters up and a loss at 2 | `xpr-julia-backend.md` | Run the same XPR fit with `fit.use_jacobian` True/False, count `Fit.function_calls`/`Niter`, compare `result.params` and wall time; repeat at nvarys = 2, 3, 4, 5 to confirm the threshold |
| 5 | The reported angle bug: lmfit writes its raw variable vector into `result.params`, so a local fold inside `y()` is discarded (reproduced as theta=379.3, phi=-8208.2 for a fit evaluating 19.3 / 71.8 with identical R) | `xpr-julia-backend.md` | Fit `XLayersPDB_XPR` with `canonical_params` monkeypatched to a no-op and confirm out-of-window angles in `fit_report`; then confirm `_canonicalize_result` folds them with `chisqr` unchanged to machine precision |
| 6 | This document's own suspicion: `__masked_x__` + a mask breaks `update_plot`'s fit-curve `add_data` and the `Residuals` entry (length mismatch) | this doc, from reading `xmodfit.py:6296-6304` and `PlotWidget.py:215` | Headless: build a `Fit` on a `__masked_x__` model with a non-trivial mask, call `update_plot`, and assert no "dimensions ... not matching" warning and that `output_params['Residuals']['x']` and `['y']` have equal length |
| 7 | The covariance composition order (slice -> imask -> concatenate -> `_cov_mask`) is symmetric in `_build_chol` and `residual`, and the length guard fires when it is not | `column-fit-limits.md`; guard read at `Joint_Fit_Routines.py:532-537` | `validate_joint_mask.py` covers the cov+footprint branch; additionally perturb one of the two sites and assert the guard raises rather than silently mis-whitening |
| 8 | The MCMC fingerprint change means a backend `.h5` from an unmasked run is now REJECTED for a masked run of the same dataset | `column-fit-limits.md` | Sample a few steps unmasked, write the fingerprint, apply a column limit, reopen the `.h5` and assert `_mcmc_fingerprint_mismatch` reports the `datahash`/`ndata` difference |
| 9 | `.xpar`/`.jpar` with and without `#Column Limits=` load in BOTH directions (old reader / new file and new reader / old file) | `column-fit-limits.md` | `validate_column_limits.py` claims to cover this; verify it actually reads a file WITHOUT the line and asserts `limits` has only the `'x'` entry, and that stock `origin/develop`'s `loadParameters` ignores the extra line |
| 10 | `validate_xlayerspdb_joint.py` proves the OLD workflow is intact: a real joint XLayersPDB fit with shared `qoff`, a Nov-2024 `.par` load, and an unchanged MCMC fingerprint | `column-fit-limits.md` | Run it; confirm it asserts fingerprint equality against a pre-mask reference value and not merely self-consistency |
| 11 | The banner-lifecycle and detail-switch behaviours were all validated headless (`verify_banner_lifecycle.py`, `verify_detail_switch.py`, `verify_multifit_selection.py`, `verify_legend.py`, `verify_plotting_B.py`) | `multimodal-seeding-plan.md` | Those `verify_*.py` files are NOT in the repo (they lived in per-session scratchpads). Either re-derive them or fold the assertions into `validate_jointfit_gui.py`, which is the only surviving GUI regression |
| 12 | Legend cleanup: exactly one legend entry per plotted curve, and the legend SHRINKS on a selection change | `multimodal-seeding-plan.md` (`verify_legend.py`) | Headless: plot 3 curves, `Plot(2 names)`, and assert `len(legendItem.items) == 2`; repeat after 20 redraws to confirm no accumulation |
| 13 | `pdbread` is shared across instances and deepcopies (one parse per file per process), so `_clone_fit`'s `deepcopy(func)` is cheap | `validate_perdataset_fit.py` docstring | Run `validate_perdataset_fit.py`; independently time `_clone_fit` on an `XLayersPDB` with a large PDB |
| 14 | `_LimitProxy` is fully source-compatible: every upstream `xmin`/`xmax` use still works | this doc, by inspection of the call sites listed in 3a | `grep -n "self\.xmin\|self\.xmax" xmodfit.py joint_fit_*.py` and check each use against the `MutableMapping` methods implemented (note: `.get`, `.items`, `.keys` come from the ABC and will work; `in` works; `.copy()` returns a plain dict) |
| 15 | Loading a `.par` now enables "Calculate Param Error" because the loaded fit counts as fitted, with a baseline leastsq run on demand | code at `xmodfit.py:5160-5178`, `2173+` | Load a `.par` headless and confirm the CI path produces sane `stderr`/`redchi` rather than reporting the loaded point as the minimum |

## 8. Consolidation candidates

| Item | Recommendation | Risk |
|---|---|---|
| The "is this dataset fitted?" predicate, written five times: `_dataset_is_fitted` (`xmodfit.py:1212`), inline in `_set_detail_dataset` (`1548-1549`), `_refresh_data_labels` (`1609-1610`), `update_plot` (`6350`), `_eval_individual_curve` (`6401`), `_model_output_params` (`6696`), and `computeModelComparison`'s `_fitted` (`7573-7576`) | Merge — route all of them through `_dataset_is_fitted` | Low. Note `_refresh_data_labels` and `computeModelComparison._fitted` OMIT the `not hasattr(f,'objectives')` guard, so merging tightens them slightly; verify nothing depends on a `JointFit` in `dataset_fits` |
| `ColumnLimitsDialog.__init__(parent, names)` / `self.names` | Drop the parameter | Low — only caller passes `[]` |
| `PlotWidget.remove_data` bookkeeping (`linestyle`/`fit`/`yerr` not deleted; `legendItem.removeItem(dname)` is a no-op) | Keep the no-op removal (harmless, `Plot` rebuilds) but delete the dict entries | Low |
| `_LimitProxy` vs a plain `limits` dict | Keep. The proxy is the whole reason this was additive; removing it means touching ~20 call sites | High if dropped |
| Tableau-10 palette defined twice: `PlotWidget.py:244-248` (fallback) and `xmodfit.py:557-558` (`_curve_palette`) | Merge — export one module-level constant from `PlotWidget` | Low |
| `_indiv_fit_name` unused by `fitCallback` (bare model name there) | Use it, for consistent legend keys during a fit | Low |
| Model-comparison dock (`create_compareDock`, `_eval_chisqr`, `computeModelComparison`, `_joint_range`) | Move — document and own it in `handoff/01-joint-fit.md`; it is joint-vs-individual analysis, not core app | Low (documentation move only) |
| `_joint_masks` (`xmodfit.py:1466`) lives in the core app but exists only for `spec.compile()` | Keep in `xmodfit.py` — it needs `dlg_data`, which the spec cannot see. Add a pointer from `handoff/01-*` | Low |
| `Image2DWidget` / `_op_is_image` / `_op_var_list` / `_set_2D_tab_visible` / `_show_2D_tab` / `_on_plotTab_changed` / `saveGenParameters` image branch | Move — owned by `handoff/06-*` (2D map); listed here only so the sweep is complete | Low |
| `_column_mask`'s `slice_=` parameter | Keep, but consider replacing the two call sites with a `Fit.set_mask` + `imask_for` round trip so only ONE place knows how the arrays were sliced | Medium — the pre-sliced `update_plot` path is exactly what `imask_for`'s dual-path design exists to serve |
