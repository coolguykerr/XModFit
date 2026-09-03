# Structure viewer (NGL) — XModJointFit change record

**Category:** live 3-D viewer that renders the PDB protein oriented and inserted into the fitted XRR layer stack, driven read-only by XLayersPDB fit parameters
**Status:** Working-but-unvalidated-live (Windows only; audit pass deliberately deferred, GUI not run for this doc)
**Owner chat(s):** `memory/ngl-viewer-improvements.md` (requirements catalogue, 2026-08-24); code originated from the `anthropic-skills:xlayerspdb-ngl-viewer` skill
**Last touched:** code committed 2026-08-05 (c5f697d); improvement catalogue written 2026-08-24

## 1. Why this exists

XLayersPDB fits specular reflectivity to a box model of an interface with a rigid PDB
structure superposed on it: two Euler angles (`theta`, `phi`) set the protein's
orientation, `protins` sets how deep it is pushed into the stack, and the per-layer
`d`/`rho` set the interface itself. Those numbers are physically meaningful but visually
opaque — a fit can be numerically excellent and structurally nonsense (protein sticking
out of the membrane, upside down, or inserted past the substrate). The viewer closes that
loop: it shows the actual atoms, in the actual fitted pose, sitting inside
density-coloured slabs built from the actual fitted thicknesses, and it updates as the fit
moves. It is a diagnostic and figure-making aid, not an input: every parameter it displays
is read-only, and the only way to change the pose is XModFit's own parameter tables.
It renders the box-model interface, not the smeared electron-density profile, so it is a
companion to the EDP plots rather than a replacement. Intended for the `XLayersPDB`
family only (see section 4).

## 2. Files

| File | Lines | State | Role |
|---|---|---|---|
| `Tools/StructureViewer/NGLViewerDialog.py` | 567 | new (c5f697d) | Qt dialog, model reader, `build_html()`, live push, camera cache |
| `Tools/StructureViewer/viewer_template.html` | 215 | new (c5f697d) | the whole viewer page: CSS, panel markup, geometry + event JS |
| `Tools/StructureViewer/ngl.js` | 88 (1.29 MB, minified) | new (c5f697d) | bundled NGL 2.4.0 (`NGL.Version="2.4.0"`) |
| `xmodfit.py` | 1 line added | modified | `Qt.AA_ShareOpenGLContexts` at `xmodfit.py:7716` |
| `requirements_xmodfit_viewer.txt` | 173 | new-untracked | full pinned env + `PyQtWebEngine==5.15.6` and its rationale (lines 156-173) |
| `Tools/StructureViewer/__pycache__/` | — | new-untracked | build artefact, should be gitignored |

`scripts/make_browser_viewer.py` — the no-Qt browser generator that the notes and the
template's own header comment refer to — **does not exist in this repo**. There is no
`scripts/` directory at all. It lives (if anywhere) in the originating skill package, not
here. Three comments in `viewer_template.html` (lines 14, 25, 78: `scripts/make_browser_viewer.py`,
`references/geometry.md`, `references/ngl_recipes.md`, `assets/ngl.js`) are therefore
stale paths pointing outside the repo — the real library is the sibling `ngl.js`, not
`assets/ngl.js`.

## 3. How it works

**Launch.** Stock XModFit auto-discovers tools: `xmodfit.py:646-658` lists every
subdirectory of `Tools/`, globs `Tools/<dir>/*.py`, and adds one menu action per file.
`launch_tApp` (`xmodfit.py:670-682`) imports the module and instantiates
`getattr(module, <filename>)(self)` — so the class name must equal the filename, and the
XModFit main widget is passed as the *first* positional arg. Menu path: **Tools >
StructureViewer > NGLViewerDialog**. The dialog instance is cached in `tApp_Clients`, so
reopening only calls `show()` — which is why the class deliberately loads nothing in
`__init__` and drives the first load from `showEvent` (`NGLViewerDialog.py:352-361`).

**Finding the model.** `_find_xlayers_func` (`:217-230`) reads `parent.fit.func`, falling
back to `parent.func`. `_looks_like_xlayers` (`:233-237`) accepts anything with
`fname`, `__mpar__`, `theta`, `phi`, `protins` — this is what excludes plain `XLayers`
(no `fname`) and admits every `XLayersPDB*` variant. Crucially the func is **re-resolved
on every poll** (`_resolve_func`, `:341-349`); a change of identity forces a full reload.
The comment at `:276-282` records why: the dialog is cached by XModFit, so a func captured
at construction goes stale the moment the user selects XLayersPDB or loads data
afterwards. Until a renderable fit exists, `_show_placeholder` (`:364-381`) shows a
guidance page instead of a modal, and only re-renders when the message text changes so
the 400 ms poll does not flicker.

**Reading the fit.** `_layer_payload` (`:180-198`) calls `func.update_parameters()` then
reads `func.__d__[mkey]`, `func.__rho__[mkey]` and `func.__mpar__[mkey]["Layers"]` — the
box-model arrays, not the fine profile (`Functions/XRR/XLayersPDB.py:452-461`).
`scene_cfg_from_func` (`:200-210`) adds `theta`/`phi`/`protins`/`fslice` and the whole PDB
file text, producing the CFG dict `{pdb, fit:{theta,phi,protins,fslice,names,d,rho,colors}}`.
Scalars are read from the **attributes** first, lmfit params only as fallback
(`_snapshot`, `:453-480`); the comment says this is for consistency with `ED_profrot`,
and it is also correct during a fit because `Fit_Routines.py:247-250` writes each varying
value straight into `func.__dict__`.

**Layer colours (Python side).** `_DENSITY_STOPS` / `_density_color`
(`:153-178`) map electron density to RGB through a fixed **absolute** piecewise-linear
ramp (violet -> purple -> blue at water 0.334 -> green/chartreuse -> gold -> orange -> red
-> crimson at 4.42, ~gold metal). The 25-line comment at `:127-152` records the reason:
an earlier version normalised rho to the stack's own min/max, so every slab colour jumped
whenever any single rho was edited. Stability across edits was chosen over contrast — note
this directly conflicts with catalogued improvement #3 (section 6).

**Page assembly.** `build_html` (`:212-215`) does two plain text replaces on the template:
`__NGLJS__` -> the full contents of `ngl.js`, `__CFG__` -> `json.dumps(cfg)`. Both
placeholders sit in the real markup (`viewer_template.html:80` and `:85`). The result is
written to a `tempfile.mkstemp(suffix=".html")` and loaded via `QUrl.fromLocalFile`
(`:435-439`). Assets are located by `_find_ngl_js` (`:240-255`) and `_find_template`
(`:86-103`): explicit arg -> `$XMODFIT_NGL_JS` / `$XMODFIT_NGL_TEMPLATE` -> next to the
module -> parent dir -> cwd.

**Geometry in the page** (`viewer_template.html:84-170`, the "VIEWER ENGINE" script).
`rotRows` (`:104-105`) builds the same extrinsic 'zx' rotation the model uses.
`recompute` (`:140-158`) is the heart: it rotates the raw atom list, computes
`ztop/zbot/zslice/zref` exactly as `ED_profrot` does, sets
`zshift = (-sum(d[:-1]) + protins) - zref`, then **bakes** the rotated+shifted coordinates
into `comp.structure` atom by atom (`:149-152`) and calls `refreshPosition()`. Baking
rather than applying a component matrix is the load-bearing choice: it keeps cartoon,
licorice and the molecular surface mutually consistent, and it means every atom's `z`
is already a signed depth in the interface frame. This matches
`XLayersPDB.calcProfComb` (`Functions/XRR/XLayersPDB.py:359-361`), where the top protein
slice centre is placed at `-sum(d[:-1]) + protins`.

**Slabs.** `buildSlabs` (`:117-138`) forms `mp[i] = -cumsum(d)[i]`, so interfaces run
*downward* in -z from `mp[0]` (=0 for the usual `d[0]=0` top medium). Finite layers
`1..n-2` become boxes spanning `mp[i-1]..mp[i]`; the bottom medium is drawn from
`z_deep = mp[n-2]` down to `floor = min(z_deep-5, tip-5)` (`BOT_MARGIN=5`, `:86`), so the
subphase always visibly encloses the protein tip. No air box is drawn. XY half-width is
`fixedHalf = R+2` where `R` is the max atom radius from the origin *before* rotation
(`:162-164`), i.e. rotation-invariant so the box does not breathe as `theta/phi` change.
Boxes are hand-built triangle meshes with explicit normals (`addBox`, `:106-116`) so they
are lit, and rendered as a `buffer` rep with `opacity`, `side:"double"`,
`depthWrite:false`. The 8-line comment at `:127-137` documents a real NGL bug: NGL adds
the buffer to the picking scene *before* honouring `disablePicking`, so the see-through
slab intercepted every hover and the tooltip labelled the mesh; the fix walks
`bufferList` and sets each `pickingGroup.visible = false`.

**Live update, no reload.** A 400 ms `QTimer` (`POLL_MS`, `:268`, started `:333-336`)
calls `refresh_now` (`:497-527`). It diffs `_snapshot()` against `_last`; a changed
`fname` forces `reload_scene()`, anything else is pushed as
`setFit(theta, phi, protins, layers)` into the running page. The layer payload is sent
only when names/d/rho/fslice actually changed, so a pure orientation nudge does not
rebuild the legend. `window.setFit` (`:203-207`) updates `F`/`cur`, optionally rebuilds
the layer rows and legend, then calls `recompute()` — representations, highlights, camera
and background all survive.

**Camera.** `window.getOrientation`/`window.applyOrientation`
(`viewer_template.html:95-96`) read/write the camera `Matrix4` via
`stage.viewerControls`. Python polls the matrix asynchronously into `self._orientation`
(`_cache_orientation`, `:483-495`) and injects it as `cfg["orientation"]` on the next
reload (`:404-406`); the page applies it at `:169`, else falls back to `stage.autoView()`.
That is the entire persistence mechanism — it is per-dialog-instance and in memory only,
never written to disk.

**Under a joint fit** the viewer needs no special handling: `JointFit` exposes a `func`
property (`Joint_Fit_Routines.py:1067-1073`) whose docstring explicitly names
"NGLViewerDialog duck-typing". It returns the *currently selected objective's* model, so
the viewer shows one dataset at a time; `set_current` changes the object identity, which
`_resolve_func` detects as a new func and triggers a full reload with the camera
preserved. Joint engine details: `handoff/01-*`.

## 4. How to use it

1. Install `PyQtWebEngine` into the environment that runs XModFit
   (`pip install PyQtWebEngine`, or `conda install -c conda-forge pyqtwebengine`). Without
   it the dialog still opens but shows an install message
   (`NGLViewerDialog.py:303-323`) — it does not crash.
2. `xmodfit.py` must set `Qt.AA_ShareOpenGLContexts` before constructing the
   `QApplication` (`xmodfit.py:7716`). This is already committed. Without it the tool
   fails to open even with PyQtWebEngine installed.
3. In XModFit select **Functions > XRR > XLayersPDB** (or a variant), load the dataset
   and the PDB, then **Tools > StructureViewer > NGLViewerDialog**.
4. The window is read-only for fit values. Panel controls: main representation
   (cartoon/tube/backbone/ball+stick/spacefill), colour-by (secondary structure, chain,
   rainbow, resname, B-factor, hydrophobicity, element, **Depth (vs interfaces)**), all
   side chains, molecular surface + its opacity, background colour, ions (vdW spheres),
   ligands (licorice), slab opacity, opt-in side chains for an NGL selection, whole-residue
   highlight/hide/reset, layer legend, hover readout
   (`viewer_template.html:47-75`). "Sync to current fit" forces a full reload and is the
   only path that shows a modal on failure.
5. Works with the model classes that expose `fname`+`theta`+`phi`+`protins`: confirmed
   present in `XLayersPDB`, `XLayersPDB_G`, `XLayersPDB_H`, `XLayersPDB_box`,
   `XLayersPDB_dev_smooth`, `XLayersPDB_XPR` (all also define `update_parameters` and
   `__d__`). Plain `XLayers` is correctly rejected. Models themselves: `handoff/04-pdb-xrr-models.md`.
6. Standalone, no fitting stack:
   `python Tools/StructureViewer/NGLViewerDialog.py fitparam.par structure.pdb Tools/StructureViewer/ngl.js`
   — `stub_func_from_par` (`:533-560`) parses `theta`/`phi`/`protins`/`fslice` and
   `__<mkey>_(Layers|d|rho)_NNN` rows out of a tab-separated `.par`.
7. Env overrides: `XMODFIT_NGL_JS`, `XMODFIT_NGL_TEMPLATE`.
8. `open_viewer(func, ngl_js_path=None, parent=None)` (`:529-531`) and
   `dlg.refresh_now` are the programmatic API; `refresh_now` can be connected to a
   parameter-changed signal for poll-free updates.

**What the picture means.** z is depth along the interface normal, increasing downward as
-z; `z=0` is the top interface (top medium `d[0]` is nominally 0). Each slab is one fitted
box of thickness `d[i]` coloured by its absolute `rho[i]`. The protein is the rigid PDB
rotated by (`theta`,`phi`) and translated so its topmost density slice sits at
`-sum(d[:-1]) + protins`; larger `protins` pushes it deeper. The bottom slab is the
semi-infinite subphase, drawn only far enough to enclose the protein tip. Coverage (`cov`)
weighting and roughness smearing are NOT represented — this is the sharp box model plus
one copy of the molecule.

## 5. Contact surface with stock XModFit

| Upstream file | Function / region | Change | Additive? |
|---|---|---|---|
| `xmodfit.py` | `__main__`, before `QApplication(sys.argv)` (`:7716`) | add `QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)` | yes, one line |
| `xmodfit.py` | Tools menu loader (`:646-682`) | **unchanged** — the viewer relies on the existing auto-discovery | n/a |
| `Tools/` | new `StructureViewer/` subdirectory | new files only | yes |
| `Joint_Fit_Routines.py` | `JointFit.func` property (`:1067-1073`) | exists partly to satisfy this viewer's duck-typing | new file, see handoff/01 |
| `Functions/XRR/XLayersPDB.py` | none | viewer reads it; no edits | n/a |

Merge risk is low: one upstream line plus a self-contained tool directory. The
`AA_ShareOpenGLContexts` line is the only thing that can conflict, and only if upstream
touches the same block.

## 6. Known limitations, gotchas and TODOs

**The six catalogued improvements — all NOT IMPLEMENTED.** From
`memory/ngl-viewer-improvements.md` (2026-08-24), a requirements list, not a changelog.
Each hook below was verified against the current code.

1. **Canned orthographic / side-on views (the biggest ask).** Today the page calls
   `stage.autoView()` when there is no saved orientation
   (`viewer_template.html:169`, the `else` branch), which lands looking from *above* —
   the least useful angle, since the layers stack along z. Want (i) a view-cube-style set
   of buttons snapping to the 6 straight-on orthographic views, (ii) a **side view as the
   default**, camera along +x or +y with z vertical. Hook: precomputed `Matrix4` arrays
   fed through the existing `window.applyOrientation` (`:96`); no new plumbing needed.
2. **Protein depth colours not length-scale normalised.** `DEPTHID`
   (`viewer_template.html:92`) maps `t = (depthRef.top - atom.z)/(depthRef.top - depthRef.deep)`
   to a hand-rolled RGB ramp. `depthRef.top/deep` are the shallowest and deepest
   *membrane* interfaces, set in `recompute` (`:154`) and again in the colour-scheme
   handler (`:196`); anchoring to the stack rather than the protein tip is deliberate
   (comment at `:155`) so the palette does not re-normalise every frame. Requested: define
   the ramp domain from the overall length scale of the system and pick a better palette.
3. **Slab colours too subtle / poor colormap.** Requested: colours normalised to the
   system's *contrast* (rho spread). **Caution — this conflicts with a deliberate past
   fix.** The current map is intentionally absolute (`_DENSITY_STOPS`,
   `NGLViewerDialog.py:153-178`) precisely because the older stack-relative map made every
   slab colour jump on any rho edit (comment `:127-152`). Any contrast-normalised scheme
   must re-solve that stability problem, and belongs on the Python side where `colors` is
   computed, not only in the JS (`F.colors[i]` is consumed at
   `viewer_template.html:123-124` and `:101-102`).
4. **Panel layout: fixed width, duplicated layer info.** `#panel` is a hard 300 px column
   and `#vp` is offset by 300 px (`viewer_template.html:31-32`), so nothing scales.
   Worse, two sections show the same layers: "Layers (live from fit)" (`#layerInputs`,
   `:51`, filled by `rebuildLayerDisplay`, `:98-99`, name + d) and "Layers legend"
   (`#legend`, `:71`, filled by `rebuildLegend`, `:100-102`, swatch + name + rho).
   Requested: one responsive table `swatch | name | d (A) | rho`. Both builders are
   re-called from `setFit` (`:206`), so merging them means one function instead of two.
5. **macOS compatibility unknown.** Only ever run on Windows. A `PySide6` import fallback
   exists (`NGLViewerDialog.py:61-65,80-81`) but is untested. Risks: QtWebEngine GPU/WebGL
   behaviour, retina/DPI scaling, and whether `AA_ShareOpenGLContexts`
   (`xmodfit.py:7716`) is sufficient there. Needs a Mac test pass and platform guards.
6. **Distance / protrusion measurements.** Nothing measures anything today. The geometry
   is already favourable because `recompute` bakes interface-frame coordinates
   (`viewer_template.html:149-152`): an atom's `z` *is* its signed distance to any
   interface plane `z = mp[k]`. `lastTip = minz + zshift` (`:154`) is the deepest atom and
   `z_deep = mp[n-2]` (`:120`) the deepest membrane interface, so the default readout
   "protrusion into the bulk subphase" is `max(0, z_deep - lastTip)`; it would sit next to
   the existing `#hoverdbg` panel (`:73`, handler `:183-193`). Atom<->atom pairs could use
   NGL's native `distance` representation.

**Other gotchas found by reading the code:**

- The poll never stops. There is no `hideEvent`/`closeEvent` and `self._timer` is never
  stopped, so `refresh_now` runs at 2.5 Hz for the life of the XModFit session even when
  the dialog is hidden (XModFit caches, never destroys, the instance).
- The viewer is read-only about *values* but not side-effect-free: `_layer_payload`
  (`:185`) calls `func.update_parameters()` on every poll, which re-syncs
  `__d__/__rho__/__mu__/__sig__` from `params`. Benign in principle, but it is a write
  into the live model at 2.5 Hz, including while a fit runs.
- Every full reload writes a fresh ~1.3 MB temp HTML (the whole minified `ngl.js` plus the
  whole PDB text, inlined). The previous temp file is unlinked (`:432-434`) but the last
  one leaks on close.
- `buildSlabs`/`recompute` index `mp[n-2]`, so a stack with fewer than 2 layers is
  undefined behaviour. Not reachable through the normal GUI (the default mpar has 2).
- `_viridis` (`NGLViewerDialog.py:119-125`) is dead code — no caller anywhere in the repo.
- The page `<title>` is still `Tim4 NGL (live)` (`viewer_template.html:27`).
- `requirements_xmodfit_viewer.txt:161-166` warns that `PyQtWebEngine` must resolve a
  WebEngine Qt matching `PyQt5-Qt5==5.15.2`; a newer `PyQtWebEngine-Qt5` mixes patch
  levels and "can break the windows platform plugin". Pin `PyQtWebEngine==5.15.2` if that
  happens.
- `launch_tApp` (`xmodfit.py:673-675`) guards on `module_name not in sys.modules` but then
  reads `self.tModules[module_name]`; if the module were already imported by another path
  this would `KeyError`. Upstream code, pre-existing, mentioned only because it is on this
  tool's launch path.

## 7. Claims to verify (audit hooks)

| # | Claim | Asserted by | Suggested check |
|---|---|---|---|
| 1 | The whole viewer actually renders correctly in the live GUI (slabs, surface, hover, legend) | code comments ("all from the approved browser prototype") | Run XModFit in cmc-py39 with PyQtWebEngine, load an XLayersPDB fit, open the tool, screenshot |
| 2 | The 400 ms poll keeps up with a running fit without stalling the GUI | `NGLViewerDialog.py:268` design | Start a least-squares fit with the viewer open; watch for UI lag and confirm the pose animates |
| 3 | The `disablePicking` + `pickingGroup.visible=false` workaround still works in bundled NGL 2.4.0 | comment `viewer_template.html:127-137` | Hover the protein through a slab; the tooltip and `#hoverdbg` must name a residue, not the mesh |
| 4 | Camera persistence across a PDB/dataset switch actually works (async `runJavaScript` callback may lag the reload) | `:483-495` | Orient the view, switch the loaded PDB, confirm the camera does not reset |
| 5 | The baked JS geometry matches `ED_profrot`/`calcProfComb` numerically, not just structurally | template comment + this doc's read of both | Print `zprot[0]` from Python and compare with the JS `-sum(d[:-1])+protins` placement for the same theta/phi/protins |
| 6 | Under a joint fit, switching the current objective reloads the viewer to that dataset's model | `Joint_Fit_Routines.py:1067-1073` docstring | Open the viewer during a joint fit, call `set_current` from the GUI/console, confirm reload |
| 7 | `AA_ShareOpenGLContexts` is genuinely required (the tool "fails to open" without it) | `requirements_xmodfit_viewer.txt:168-172` | Comment out `xmodfit.py:7716` and reopen the tool |
| 8 | The absolute density ramp's control points are the right physical anchors (tails 0.23, water 0.334, headgroups 0.42-0.50, Si/oxide 0.66-0.70, Au 4.42) | comment `NGLViewerDialog.py:127-163` | Spot-check against tabulated electron densities |
| 9 | The non-`XLayersPDB` variants (`_G`, `_H`, `_box`, `_dev_smooth`, `_XPR`) render sensibly, not just import-compatibly | this doc only checked attribute presence | Open each with the same PDB and compare the slab stack against its EDP plot |
| 10 | `stub_func_from_par` still parses current `.par` output | `:533-560` | Save a `.par` from a current XLayersPDB fit and run the standalone entry point |
| 11 | PySide6 fallback path works at all | `:61-65` | Import the module in a PySide6-only env |

## 8. Consolidation candidates

- **`Tools/GrazingIncidence/GIDViewerDialog.py` (untracked, out of this category) is a
  near-clone of `NGLViewerDialog.py`** — same `_HAVE_QT`/`_HAVE_WEBENGINE` guard, same
  `build_html`, same `refresh_now` + `QTimer` polling, same `open_viewer` factory, same
  docstring phrasing. Recommendation: **merge** the shared scaffolding into one small
  base class or helper module under `Tools/` before upstreaming, so the QtWebEngine
  guard, asset resolution and poll/push loop are fixed in one place. Risk: moderate — the
  Tools loader requires `class name == file name`, so a shared base must live in a
  non-`Tools/<dir>/*.py` location or it will appear as a bogus menu entry (every `.py` in
  a Tools subfolder becomes a menu item).
- **`_viridis` (`NGLViewerDialog.py:119-125`)** — dead, superseded by `_density_color`.
  Recommendation: **drop**. Risk: none.
- **`Tools/StructureViewer/__pycache__/`** — committed nothing, but present untracked and
  holding both a cpython-39 and a cpython-311 `.pyc`. Recommendation: **drop** and add to
  `.gitignore`. Risk: none.
- **Stale external references in `viewer_template.html:14,24-25,78,83`**
  (`scripts/make_browser_viewer.py`, `references/geometry.md`,
  `references/ngl_recipes.md`, `assets/ngl.js`). Recommendation: either **port**
  `make_browser_viewer.py` in (it lets the page be debugged in a real browser with
  devtools, which QWebEngineView makes painful) or **rewrite** the comments. Risk: none
  for the comment fix.
- **`ngl.js`**: 1.29 MB minified blob with a dangling `sourceMappingURL=ngl.js.map` (no
  `.map` shipped). Recommendation: **keep** — offline, no-CDN operation is the point — but
  record the version (NGL 2.4.0) visibly and confirm upstream accepts the file size.
- **`requirements_xmodfit_viewer.txt`** duplicates most of `requirements.txt` and the four
  other `requirements_*.txt` variants in the repo root, differing mainly by the trailing
  `PyQtWebEngine` block. Recommendation: **merge** — keep one base requirements file plus
  a tiny `requirements-viewer.txt` containing only the optional extra. Risk: low, but
  check which of the five files the working conda env was actually built from first.
