"""
NGLViewerDialog.py
==================
Embedded NGL viewer for XModFit's XLayersPDB fit, as a NATIVE Qt dialog
(QWebEngineView + bundled ngl.js) -- no browser, no Jupyter.

Feature set (all from the approved browser prototype):
  * baked-coordinate transform ('zx' extrinsic rotation + insertion applied to
    atoms, so cartoon/licorice/surface stay mutually consistent);
  * fixed box XY (rotation-invariant); box Z with water floor
    min(deepest interface - 5, protein tip - 5); no drawn air box;
  * density-coloured, lit, see-through slabs (normals + depthWrite:false) with an
    opacity slider; single-shell depth-writing molecular surface
    (side:"front", depthWrite:true) with its own opacity slider;
  * whole-residue highlight, opt-in per-selection side chains (attached to CA),
    all-side-chains toggle, hover tooltip;
  * READ-ONLY parameter display: theta/phi/protins and the per-layer names +
    thicknesses are shown for reassurance but are NOT editable here -- they are
    driven entirely by the XModFit fit (change them in XModFit's parameter
    tables, not in this panel);
  * user-selectable BACKGROUND COLOUR (default white);
  * LIVE auto-update: the dialog polls the model and pushes changes into the
    running page WITHOUT reloading (representations, highlights, camera and
    background are preserved). This covers orientation, per-layer thickness AND
    a change in the NUMBER of layers (add/remove a box in XModFit) or their
    densities -- the slab stack, layer rows and legend all rebuild live. Only a
    change of the PDB file itself triggers a full reload. Also exposes
    refresh_now() so you can connect an XModFit parameter-changed signal for
    instant, poll-free updates;
  * the live model is RE-RESOLVED from the parent widget on every show/poll
    (never frozen at construction), so opening the tool before selecting
    XLayersPDB/loading data, then setting them up, just works -- no XModFit
    restart needed. A guidance page is shown until a renderable fit is present.

Launch from XModFit:
    from NGLViewerDialog import open_viewer
    dlg = open_viewer(func, ngl_js_path=r"C:/path/to/ngl.js", parent=self)
    # optional, for instant updates instead of polling:
    #   self.someParamChangedSignal.connect(dlg.refresh_now)

Standalone test (stub from a .par + PDB, no fitting stack):
    python NGLViewerDialog.py fitparam.par structure.pdb ngl.js
"""
from __future__ import annotations
import os, re, json, tempfile
import numpy as np

# Qt is required only to *display* the dialog. The HTML-generation helpers
# (scene_cfg_from_func, build_html, stub_func_from_par) work without any Qt
# binding, so a browser-only workflow can import this module Qt-free.
_HAVE_QT = True
_HAVE_WEBENGINE = True
_QT_BINDING = None
try:
    from PyQt5.QtCore import QUrl, QTimer
    from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                                 QPushButton, QLabel, QMessageBox)
    _QT_BINDING = "PyQt5"
except ImportError:
    try:
        from PySide6.QtCore import QUrl, QTimer
        from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                                       QPushButton, QLabel, QMessageBox)
        _QT_BINDING = "PySide6"
    except ImportError:
        _HAVE_QT = False
        QDialog = object  # lets the HTML-generation helpers import without Qt

# QtWebEngine ships as a SEPARATE package (PyQtWebEngine for PyQt5). Import it on
# its own so a missing WebEngine does NOT drop the whole Qt block and silently
# rebind QDialog to `object`. If that happened, XModFit's Tools loader would call
# NGLViewerDialog(parent) -> super().__init__(parent) -> object.__init__(self,
# parent) and crash with the cryptic "object.__init__() takes exactly one
# argument". Keeping QtWebEngine separate means QDialog stays a real dialog and
# the constructor can show a clear, actionable message instead of crashing.
if _HAVE_QT:
    try:
        if _QT_BINDING == "PyQt5":
            from PyQt5.QtWebEngineWidgets import QWebEngineView
        else:
            from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError:
        _HAVE_WEBENGINE = False
        QWebEngineView = None

def _find_template(explicit=None):
    """Resolve viewer_template.html: explicit arg -> $XMODFIT_NGL_TEMPLATE ->
    next to this file -> cwd. Mirrors _find_ngl_js so the tool locates its two
    sibling assets (ngl.js and the HTML scaffold) the same way."""
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("XMODFIT_NGL_TEMPLATE")
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.join(here, "viewer_template.html"),
              os.path.join(here, "..", "viewer_template.html"),
              os.path.join(os.getcwd(), "viewer_template.html")]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def _template():
    """Read the viewer page scaffold from the sibling viewer_template.html.
    (Previously this string lived base64-encoded in this module; it now sits in
    a plain .html file so the CSS/markup/JS are human-readable and editable.)"""
    path = _find_template()
    if path is None:
        raise FileNotFoundError(
            "viewer_template.html was not found. It ships next to "
            "NGLViewerDialog.py; restore it there, or set the "
            "XMODFIT_NGL_TEMPLATE environment variable to its full path.")
    with open(path, encoding="utf-8") as fh:
        return fh.read()

def _viridis(t):
    try:
        import matplotlib.pyplot as plt
        return [float(c) for c in plt.get_cmap("viridis")(float(t))[:3]]
    except Exception:
        t = float(max(0.0, min(1.0, t)))
        return [0.267 + t*(0.993-0.267), 0.005 + t*(0.906-0.005), 0.329 + t*(0.144-0.329)]

# ---------------------------------------------------------------------------
# Fixed, ABSOLUTE electron-density -> colour map for the layer slabs + legend.
# The colour of a layer depends only on its own electron density (el/A^3),
# NOT on the current min/max of the stack. This keeps the palette STABLE while
# densities are edited: water stays blue, a given rho always maps to the same
# colour. (The original map normalised rho to the stack's own min/max, so every
# slab colour jumped whenever any single rho changed -- the behaviour fixed here.)
#
# The scale is PIECEWISE LINEAR over the (rho, RGB) control points below, so it
# stays legible across the wide, physically real range of densities -- from
# near-vacuum, through lipid tails (~0.23, BELOW water), water (~0.334),
# protein/headgroups (~0.42-0.50), oxide/Si substrates (~0.66-0.70), up to
# heavy/metal layers (gold metal ~4.4 el/A^3). Ordered low->high the hue runs
# violet -> purple -> blue -> green -> chartreuse -> gold -> orange -> red ->
# crimson, so:
#   * sub-water  0.00 -> 0.334 : violet -> purple -> blue  (tails read PURPLE,
#     visually distinct from water's blue -- not clamped onto it)
#   * soft matter 0.334 -> 0.60 : blue -> green -> chartreuse -> gold. Green sits
#     just above water, so moieties with contrast only a little above water
#     (e.g. lipid HEADGROUPS ~0.42-0.50) jump to a high-contrast green/chartreuse
#     instead of a washed-out pale -- fine sensitivity in this narrow band.
#   * dense       0.60  -> 4.42 : gold -> orange -> red -> dark crimson
# so BOTH low- and high-ED layers stay distinct instead of collapsing onto the
# water blue or onto a single gold. rho below the first breakpoint clamps to
# violet; above the last clamps to crimson. Retune by editing / adding /
# removing (rho, RGB) pairs -- keep them in ascending rho.
_DENSITY_STOPS = [
    (0.000, (0.29, 0.00, 0.51)),  # deep violet  -- vacuum / near-zero ED (gas, void)
    (0.230, (0.62, 0.28, 0.78)),  # purple       -- lipid tails / low-ED (below water)
    (0.334, (0.16, 0.44, 0.85)),  # blue         -- bulk water / buffer
    (0.420, (0.13, 0.72, 0.42)),  # emerald green-- just above water: headgroups pop (high contrast vs blue)
    (0.500, (0.62, 0.82, 0.18)),  # chartreuse   -- yellow-green, bridges green -> gold
    (0.600, (0.90, 0.68, 0.15)),  # gold         -- "gold hues around 0.6"
    (1.200, (0.92, 0.42, 0.06)),  # orange
    (2.400, (0.72, 0.11, 0.09)),  # red
    (4.420, (0.34, 0.03, 0.16)),  # dark crimson -- ~electron density of gold metal
]

def _density_color(rho):
    """Absolute electron-density -> RGB (0..1) for a layer slab / legend swatch.
    Piecewise-linear over _DENSITY_STOPS; stable across density edits."""
    rho = float(rho)
    stops = _DENSITY_STOPS
    if rho <= stops[0][0]:
        return list(stops[0][1])
    if rho >= stops[-1][0]:
        return list(stops[-1][1])
    for (r0, c0), (r1, c1) in zip(stops, stops[1:]):
        if rho <= r1:
            s = (rho - r0) / (r1 - r0) if r1 > r0 else 0.0
            return [c0[k] + s * (c1[k] - c0[k]) for k in range(3)]
    return list(stops[-1][1])   # rho within range but unmatched (shouldn't happen)

def _layer_payload(func):
    """Per-layer names / thicknesses / densities / density-colours from a live
    XLayersPDB (or stub). Shared by the initial scene build and the live push so
    that adding, removing, or editing a layer in XModFit's parameter table is
    reflected identically by both code paths."""
    func.update_parameters()
    mkey = func.__mkeys__[0]
    d = [float(x) for x in func.__d__[mkey]]
    rho = [float(x) for x in func.__rho__[mkey]]
    n = len(d)
    try:
        names = [str(x) for x in func.__mpar__[mkey]["Layers"]]
    except Exception:
        names = [f"layer{i}" for i in range(n)]
    if len(names) < n:                       # names shorter than d (paranoia)
        names += [f"layer{i}" for i in range(len(names), n)]
    # Absolute density -> colour (see _density_color): stable across rho edits.
    colors = {i: _density_color(rho[i]) for i in range(1, n)}
    return names, d, rho, colors

def scene_cfg_from_func(func):
    """Read a live XLayersPDB (or stub) into the CFG payload the page expects."""
    fname = getattr(func, "fname", None)
    if not fname or not os.path.exists(fname):
        raise FileNotFoundError(f"PDB file not found (func.fname={fname!r}). Load the structure in the fit first.")
    names, d, rho, colors = _layer_payload(func)
    with open(fname) as fh:
        pdb = fh.read()
    return dict(pdb=pdb, fit=dict(theta=float(func.theta), phi=float(func.phi),
                protins=float(func.protins), fslice=float(getattr(func, "fslice", 0.5)),
                names=names, d=d, rho=rho, colors=colors))

def build_html(cfg, ngl_js_path):
    with open(ngl_js_path, encoding="utf-8", errors="replace") as fh:
        ngljs = fh.read()
    return _template().replace("__NGLJS__", ngljs).replace("__CFG__", json.dumps(cfg))

def _find_xlayers_func(parent):
    """Locate the live XLayersPDB instance from a running XModFit widget.

    XModFit builds a Fit_Routines.Fit for the selected model and holds it as
    ``self.fit``; the model instance is ``self.fit.func``. A couple of fallbacks
    are tried for robustness across builds.
    """
    if parent is None:
        return None
    fit = getattr(parent, "fit", None)
    func = getattr(fit, "func", None) if fit is not None else None
    if func is None:
        func = getattr(parent, "func", None)
    return func


def _looks_like_xlayers(func):
    """True only for an XLayersPDB-style model the viewer can actually render."""
    return (func is not None
            and all(hasattr(func, a) for a in ("fname", "__mpar__", "theta",
                                               "phi", "protins")))


def _find_ngl_js(explicit=None):
    """Resolve ngl.js: explicit arg -> $XMODFIT_NGL_JS -> next to this file -> cwd."""
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("XMODFIT_NGL_JS")
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.join(here, "ngl.js"),
              os.path.join(here, "..", "ngl.js"),
              os.path.join(os.getcwd(), "ngl.js")]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


class NGLViewerDialog(QDialog):
    """Native Qt window XModFit launches; serves the approved NGL scene and
    live-updates orientation/layer parameters from the model.

    XModFit's Tools loader instantiates a tool as ``TheClass(parent)`` where
    ``parent`` is the main XModFit widget, so ``parent`` is the FIRST argument.
    When launched that way, the live model is pulled from ``parent.fit.func``
    and ``ngl.js`` is resolved next to this file. The explicit
    ``func`` / ``ngl_js_path`` keywords keep the standalone API working.
    """
    POLL_MS = 400

    def __init__(self, parent=None, func=None, ngl_js_path=None):
        if not _HAVE_QT:
            raise RuntimeError(
                "NGLViewerDialog requires a Qt binding (PyQt5 or PySide6); this "
                "module was imported without one.")
        super().__init__(parent)
        # Keep the live XModFit widget so the model can be RE-RESOLVED on every
        # show/poll. XModFit caches this dialog instance (launch_tApp reuses
        # tApp_Clients[...] and only calls show() on reopen), so a func captured
        # once at construction goes stale the moment the user selects XLayersPDB
        # or loads data afterwards -- which is exactly the "same error after I
        # set things up" bug. We therefore never trust a construction-time func.
        self._parent_widget = parent
        if func is None:
            func = _find_xlayers_func(parent)
        self.func = func
        self.ngl_js_path = _find_ngl_js(ngl_js_path)
        self.setWindowTitle("XLayersPDB - NGL viewer")
        self.resize(1180, 780)
        # scene state (all set before any showEvent/poll can fire)
        self._webengine_ok = _HAVE_WEBENGINE
        self._loaded = False        # a REAL scene is currently loaded
        self._loading = False       # a real-scene load is in flight
        self._pending_real = False  # last load initiated was a real scene (vs placeholder)
        self._placeholder = False   # the guidance page is currently shown
        self._placeholder_msg = None
        self._last = None
        self._tmp = None
        self._orientation = None     # cached NGL camera (16-float matrix) to preserve
                                     # the view across dataset/PDB reloads
        # QtWebEngine present? If not, the real QDialog still exists (unlike the
        # old code, which crashed at super().__init__). Show a clear message and
        # stop, instead of raising a cryptic error inside the Tools loader.
        if not _HAVE_WEBENGINE:
            self._webengine_ok = False
            _pkg = ("PyQtWebEngine" if _QT_BINDING == "PyQt5"
                    else "PySide6 with QtWebEngine")
            lay = QVBoxLayout(self)
            info = QLabel(
                "QtWebEngine is not available in this Python environment.\n\n"
                "The NGL viewer renders in an embedded Chromium view, which "
                "ships separately from " + (str(_QT_BINDING) or "Qt") + ".\n\n"
                "Install it into the SAME environment that runs XModFit:\n\n"
                "    pip install " + _pkg + "\n"
                "    (conda:  conda install -c conda-forge pyqtwebengine)\n\n"
                "Then restart XModFit and reopen this tool.")
            info.setWordWrap(True)
            info.setStyleSheet("padding:16px; font-size:13px;")
            lay.addWidget(info)
            QMessageBox.warning(
                self, "NGL viewer - QtWebEngine missing",
                "QtWebEngine is not installed.\n\nRun:  pip install " + _pkg +
                "\nin the environment that runs XModFit, then restart.")
            return
        self.view = QWebEngineView(self)
        self.view.loadFinished.connect(self._on_loaded)

        bar = QHBoxLayout()
        sync = QPushButton("Sync to current fit"); sync.clicked.connect(lambda: self.reload_scene(user=True))
        bar.addWidget(sync); bar.addStretch(1)
        self.status = QLabel(""); bar.addWidget(self.status)
        lay = QVBoxLayout(self); lay.addLayout(bar); lay.addWidget(self.view, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self.refresh_now)
        self._timer.start()
        # NB: do NOT reload_scene() here -- showEvent drives the first load, so
        # the very same path handles first-open and every cached reopen.

    # -- re-resolve the live model from the parent widget on demand -----------
    def _resolve_func(self):
        """Re-read parent.fit.func. If it changed identity (user reselected the
        function or reloaded data -> a fresh Fit/func), force a reload."""
        newfunc = _find_xlayers_func(self._parent_widget)
        if newfunc is not None and newfunc is not self.func:
            self.func = newfunc
            self._loaded = False
            self._last = None
        return self.func

    # -- shown (incl. every cached reopen): make sure the scene is current ----
    def showEvent(self, event):
        super().showEvent(event)
        if self._webengine_ok:
            self._ensure()

    def _ensure(self):
        """Load the real scene if the model is ready; otherwise show guidance.
        Safe to call repeatedly (idempotent while loaded or a load is in flight)."""
        if not self._loaded and not self._loading:
            self.reload_scene()

    # -- guidance page for any not-ready state (no modal spam on the poll) ----
    def _show_placeholder(self, msg):
        # Only re-render when the message actually changes, so the 400 ms poll
        # can call this every tick without flicker or wasted work.
        if self._placeholder and msg == self._placeholder_msg:
            return
        self._placeholder = True
        self._placeholder_msg = msg
        self._pending_real = False
        self._loaded = False
        self.view.setHtml(
            "<html><body style='font-family:system-ui,Arial;background:#12141a;"
            "color:#e8e8ee;display:flex;align-items:center;justify-content:center;"
            "height:100%;margin:0'><div style='max-width:520px;padding:24px;"
            "line-height:1.55;font-size:15px'><h2 style='color:#9ab;font-weight:600'>"
            "NGL viewer</h2><p>" + msg + "</p><p style='color:#889;font-size:13px'>"
            "The viewer updates automatically &mdash; you do not need to close and "
            "reopen it.</p></div></body></html>")
        self.status.setText("waiting for an XLayersPDB fit\u2026")

    def _prepare(self):
        """Resolve the model and build (cfg, html), or raise ValueError with a
        short, user-facing reason for any not-ready state."""
        self._resolve_func()
        if not _looks_like_xlayers(self.func):
            raise ValueError("Select the <b>XLayersPDB</b> function (Functions "
                "&rsaquo; XRR &rsaquo; XLayersPDB) and load its dataset and PDB.")
        if not self.ngl_js_path or not os.path.exists(self.ngl_js_path):
            raise ValueError("<b>ngl.js</b> was not found. Place it next to "
                "NGLViewerDialog.py (same Tools subfolder), or set the "
                "XMODFIT_NGL_JS environment variable to its full path.")
        if _find_template() is None:
            raise ValueError("<b>viewer_template.html</b> was not found. It ships "
                "next to NGLViewerDialog.py (same Tools subfolder); restore it "
                "there, or set the XMODFIT_NGL_TEMPLATE environment variable to "
                "its full path.")
        try:
            cfg = scene_cfg_from_func(self.func)
        except FileNotFoundError:
            raise ValueError("Load the structure (PDB) into the XLayersPDB fit; "
                "the viewer will then render automatically.")
        # preserve the camera across a reload (dataset/PDB switch) if we have one
        if self._orientation:
            cfg["orientation"] = self._orientation
        return cfg, build_html(cfg, self.ngl_js_path)

    # -- full reload (structure / density / colours / names) --
    def reload_scene(self, user=False):
        # user=True only for the explicit "Sync to current fit" button, which
        # gets a modal on failure; the auto/poll path uses the placeholder page.
        try:
            cfg, html = self._prepare()
        except ValueError as exc:
            if user:
                QMessageBox.warning(self, "NGL viewer",
                                    re.sub(r"</?b>", "", str(exc)))
            else:
                self._show_placeholder(str(exc))
            return
        except Exception as exc:
            if user:
                QMessageBox.warning(self, "NGL viewer", str(exc))
            else:
                self._show_placeholder("Could not build the scene: " + str(exc))
            return
        self._loaded = False
        self._loading = True
        self._pending_real = True
        self._placeholder = False
        if self._tmp and os.path.exists(self._tmp):
            try: os.unlink(self._tmp)
            except OSError: pass
        fd, path = tempfile.mkstemp(suffix=".html")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(html)
        self._tmp = path
        self.view.load(QUrl.fromLocalFile(path))
        m = cfg["fit"]
        self.status.setText(f"theta={m['theta']:.2f}  phi={m['phi']:.2f}  protins={m['protins']:.2f} A   (live)")

    def _on_loaded(self, ok):
        # Ignore loadFinished from the placeholder page: only a real scene load
        # establishes the live baseline.
        if not self._pending_real:
            return
        self._loading = False
        self._loaded = bool(ok)
        self._last = self._snapshot()   # baseline; push only later changes

    # -- read current orientation/layers from the model (params first) --
    def _snapshot(self):
        f = self.func
        def val(pname, attr):
            # ED_profrot uses the scalar ATTRIBUTES (self.theta/phi/protins) to
            # place the protein, so read those first for consistency with the
            # rendered geometry; fall back to the lmfit param if absent.
            try:
                return float(getattr(f, attr))
            except Exception:
                try:
                    return float(f.params[pname].value)
                except Exception:
                    return None
        th, ph, pi = val("theta", "theta"), val("phi", "phi"), val("protins", "protins")
        if None in (th, ph, pi):
            return None
        fslice = float(getattr(f, "fslice", 0.5))
        fname = getattr(f, "fname", None)
        names = d = rho = colors = None
        try:
            names, d, rho, colors = _layer_payload(f)
            d = [round(float(x), 6) for x in d]
            rho = [round(float(x), 6) for x in rho]
        except Exception:
            names = d = rho = colors = None
        return {"theta": round(th, 6), "phi": round(ph, 6), "protins": round(pi, 6),
                "fslice": round(fslice, 6), "fname": fname,
                "names": names, "d": d, "rho": rho, "colors": colors}

    # -- push changes into the running page (no reload) --
    def _cache_orientation(self):
        """Ask the page for the current camera matrix and stash it (async). Used to
        restore the view after a PDB reload so switching datasets doesn't reorient."""
        if not self._webengine_ok or not self._loaded:
            return
        try:
            def _store(val):
                if val:
                    self._orientation = list(val)
            self.view.page().runJavaScript(
                "window.getOrientation?getOrientation():null", _store)
        except Exception:
            pass

    def refresh_now(self):
        if not self._webengine_ok:
            return
        self._cache_orientation()   # keep the latest camera for the next reload
        self._resolve_func()
        if not self._loaded:
            # not ready, or a valid model just appeared / was replaced after the
            # window was already shown -> (re)load or show guidance. reload_scene()
            # itself picks the right not-ready message; no reopen ever required.
            self._ensure()
            return
        snap = self._snapshot()
        if snap is None or snap == self._last:
            return
        prev = self._last
        self._last = snap
        # A different PDB needs a full reload (re-parse structure & radii)
        if prev is None or snap["fname"] != prev.get("fname"):
            self.reload_scene(); return
        # Only ship the (heavier) layer payload when the layers actually changed,
        # so a pure orientation nudge doesn't rebuild the legend/rows or flicker.
        layer_changed = (snap["names"] != prev.get("names") or snap["d"] != prev.get("d")
                         or snap["rho"] != prev.get("rho") or snap["fslice"] != prev.get("fslice"))
        if layer_changed and snap["names"] is not None and snap["d"] is not None:
            layers = json.dumps({"names": snap["names"], "d": snap["d"], "rho": snap["rho"],
                                 "colors": snap["colors"], "fslice": snap["fslice"]})
        else:
            layers = "null"
        self.view.page().runJavaScript(
            f"if(window.setFit)setFit({snap['theta']},{snap['phi']},{snap['protins']},{layers});")
        self.status.setText(f"theta={snap['theta']:.2f}  phi={snap['phi']:.2f}  protins={snap['protins']:.2f} A   (live)")

def open_viewer(func, ngl_js_path=None, parent=None):
    dlg = NGLViewerDialog(parent=parent, func=func, ngl_js_path=ngl_js_path)
    dlg.show(); return dlg

def stub_func_from_par(par_path, pdb_path):
    single, names, d, rho, fslice = {}, {}, {}, {}, 0.5
    for raw in open(par_path):
        ln = raw.rstrip("\r\n")
        if not ln or ln.startswith("#"):
            continue
        p = ln.split("\t")
        if p[0] == "fslice":
            fslice = float(p[1])
        elif p[0] in ("theta", "phi", "protins"):
            single[p[0]] = float(p[1])
        else:
            m = re.match(r"__(\w+?)_(Layers|d|rho)_(\d+)", p[0])
            if m:
                kind, i = m.group(2), int(m.group(3))
                (names if kind == "Layers" else d if kind == "d" else rho)[i] = (
                    p[1] if kind == "Layers" else float(p[1]))
    n = max(names) + 1
    class Stub: pass
    s = Stub()
    s.fname, s.fslice = pdb_path, fslice
    s.theta, s.phi, s.protins = single["theta"], single["phi"], single["protins"]
    s.__mkeys__ = ["Model"]
    s.__d__ = {"Model": [d[i] for i in range(n)]}
    s.__rho__ = {"Model": [rho[i] for i in range(n)]}
    s.__mpar__ = {"Model": {"Layers": [names[i] for i in range(n)]}}
    s.update_parameters = lambda: None
    return s

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("usage: python NGLViewerDialog.py fitparam.par structure.pdb ngl.js"); sys.exit(1)
    app = QApplication(sys.argv)
    open_viewer(stub_func_from_par(sys.argv[1], sys.argv[2]), sys.argv[3])
    sys.exit(app.exec_())