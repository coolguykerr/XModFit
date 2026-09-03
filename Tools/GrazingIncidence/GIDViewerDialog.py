"""
GIDViewerDialog.py
==================
Embedded grazing-incidence diffraction explorer for XModFit, as a NATIVE Qt
dialog (QWebEngineView + a self-contained HTML page) -- no browser, no Jupyter.

Follows the same Tools-menu conventions as the XLayersPDB NGL viewer, so if that
one works in your install this one will too. See the "Install as an XModFit
Tool" section of the grazing-incidence-scattering skill for the five rules the
loader enforces; the two that bite hardest are that the file must live in a
*subfolder* of Tools/ and that the class name must equal the file name.

WHAT IT READS FROM THE FIT -- and what it does not
--------------------------------------------------
Only the photon energy. XModFit's XLayers family takes ``E`` in keV as a
constructor argument and stores it as ``func.E``, so syncing it makes the Ewald
sphere match the actual beamline instead of a 10 keV default. That is a real
coupling, not an invented one.

Everything else stays under the user's control, because XModFit has no in-plane
lattice model: a0, tilt, tilt azimuth and strain are GID quantities with no
counterpart in a specular reflectivity fit. The layer count and total thickness
are reported for reference only. Comparing a GID area-per-molecule against an
XRR tail-region electron density IS a worthwhile cross-check, but it needs
assumptions (chains per molecule, which boxes are "tail") that this tool
deliberately does not make for you.

Consequence, and a deliberate difference from the NGL viewer: this tool has no
"not ready" state. It opens and works standalone with no fit selected, and picks
up the energy if and when a model exposing ``E`` becomes current.

Launch from XModFit:
    from GIDViewerDialog import open_viewer
    dlg = open_viewer(parent=self)
    # optional, for instant updates instead of polling:
    #   self.someParamChangedSignal.connect(dlg.refresh_now)

Standalone test (no XModFit, no fitting stack):
    python GIDViewerDialog.py            # opens at the default 10 keV
    python GIDViewerDialog.py 15.126     # opens at a given energy
    python GIDViewerDialog.py --browser out.html    # write a page, no Qt needed
"""
from __future__ import annotations
import os
import json
import tempfile
import webbrowser

# Qt is needed only to *display* the dialog. The HTML helpers (cfg_from_func,
# build_html) work with no Qt binding at all, so a browser-only workflow can
# import this module Qt-free.
_HAVE_QT = True
_HAVE_WEBENGINE = True
_QT_BINDING = None
try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout,
                                 QHBoxLayout, QPushButton, QLabel)
    _QT_BINDING = "PyQt5"
except ImportError:
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout,
                                       QHBoxLayout, QPushButton, QLabel)
        _QT_BINDING = "PySide6"
    except ImportError:
        _HAVE_QT = False
        QDialog = object          # lets the HTML helpers import without Qt

# QtWebEngine ships as a SEPARATE package (PyQtWebEngine for PyQt5). Import it
# on its own so a missing WebEngine does NOT drop the whole Qt block and
# silently rebind QDialog to `object` -- if that happened, XModFit's Tools
# loader would call GIDViewerDialog(parent) -> super().__init__(parent) ->
# object.__init__(self, parent) and crash with the cryptic
# "object.__init__() takes exactly one argument".
if _HAVE_QT:
    try:
        if _QT_BINDING == "PyQt5":
            from PyQt5.QtWebEngineWidgets import QWebEngineView
        else:
            from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError:
        _HAVE_WEBENGINE = False
        QWebEngineView = None

TEMPLATE_NAME = "gid_template.html"


# --------------------------------------------------------------- helpers ----
def _find_template(explicit=None):
    """Resolve gid_template.html: explicit -> $XMODFIT_GID_TEMPLATE -> next to
    this file -> parent dir -> cwd. Mirrors the NGL tool's resolver so both
    behave the same way when installed side by side."""
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("XMODFIT_GID_TEMPLATE")
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.join(here, TEMPLATE_NAME),
              os.path.join(here, "..", TEMPLATE_NAME),
              os.path.join(os.getcwd(), TEMPLATE_NAME)]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def _template(explicit=None):
    path = _find_template(explicit)
    if path is None:
        raise FileNotFoundError(
            f"{TEMPLATE_NAME} was not found. It ships next to "
            "GIDViewerDialog.py; restore it there, or set the "
            "XMODFIT_GID_TEMPLATE environment variable to its full path.")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _find_func(parent):
    """Locate the live model instance from a running XModFit widget. XModFit
    builds a Fit_Routines.Fit for the selected model and holds it as
    ``self.fit``; the model is ``self.fit.func``."""
    if parent is None:
        return None
    fit = getattr(parent, "fit", None)
    func = getattr(fit, "func", None) if fit is not None else None
    if func is None:
        func = getattr(parent, "func", None)
    return func


def cfg_from_func(func):
    """Build the (small) payload the page expects.

    Returns {} when there is nothing to report, which is a perfectly normal
    state -- the page then runs standalone. Never raises: a viewer that refuses
    to open because a fit is not ready would be worse than useless here, since
    the tool is useful with no fit at all.
    """
    if func is None:
        return {}
    cfg = {"model": type(func).__name__}
    E = getattr(func, "E", None)
    try:
        if E is not None:
            cfg["E"] = float(E)
    except (TypeError, ValueError):
        pass
    # layer stack, reference only
    try:
        mkeys = getattr(func, "__mkeys__", None) or list(getattr(func, "__mpar__", {}))
        if mkeys:
            d = getattr(func, "__d__", {}).get(mkeys[0])
            if d is None:
                mp = getattr(func, "__mpar__", {}).get(mkeys[0], {})
                d = mp.get("d")
            if d:
                cfg["nlayers"] = int(len(d))
                cfg["dtot"] = float(sum(float(v) for v in d))
    except Exception:
        pass
    return cfg


def build_html(cfg=None, template_path=None):
    """Inject the config into the page. The page is self-contained -- no
    external JS, no CDN -- so it renders offline and in a QWebEngineView with
    no extra permissions."""
    return _template(template_path).replace("__CFG__", json.dumps(cfg or {}))


def write_browser_page(out_path, cfg=None, template_path=None):
    """Write a standalone page (no Qt) for quick verification."""
    html = build_html(cfg, template_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(out_path)


# ---------------------------------------------------------------- dialog ----
class GIDViewerDialog(QDialog):
    """Native Qt window XModFit launches.

    XModFit's Tools loader instantiates a tool as ``TheClass(parent)`` where
    ``parent`` is the main XModFit widget, so ``parent`` is the FIRST argument.
    The live model is re-resolved from that widget on every show and every poll
    -- never frozen at construction -- because XModFit caches tool instances and
    only calls show() on reopen, so a func captured once goes stale as soon as
    the user selects a different model or loads data.
    """
    POLL_MS = 600

    def __init__(self, parent=None, func=None, template_path=None):
        if not _HAVE_QT:
            raise RuntimeError(
                "GIDViewerDialog requires a Qt binding (PyQt5 or PySide6); "
                "this module was imported without one.")
        super().__init__(parent)
        self._parent_widget = parent
        self.func = func if func is not None else _find_func(parent)
        self.template_path = template_path
        self.setWindowTitle("Grazing-incidence diffraction explorer")
        self.resize(1320, 900)

        self._webengine_ok = _HAVE_WEBENGINE
        self._loaded = False
        self._last = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        if not self._webengine_ok:
            lay.addWidget(QLabel(
                "<b>QtWebEngine is not installed.</b><br><br>"
                "This viewer needs <code>PyQtWebEngine</code> (PyQt5) or "
                "PySide6's QtWebEngine, which is a separate package from PyQt5 "
                "core. Install it into the SAME environment that runs XModFit, "
                "using the SAME package manager that provided PyQt5 (mixing "
                "pip and conda Qt breaks the platform plugin), then restart "
                "XModFit.<br><br>For a pip PyQt5 5.15.9 environment:<br>"
                "<code>pip install PyQtWebEngine==5.15.6</code><br><br>"
                "Meanwhile you can still use the tool in a browser: "
                "<code>python GIDViewerDialog.py --browser gid.html</code>"))
            return

        self.view = QWebEngineView(self)
        lay.addWidget(self.view, 1)

        bar = QHBoxLayout()
        self.status = QLabel("standalone (no fit attached)")
        self.status.setStyleSheet("color:#51687A;font-size:11px")
        b_sync = QPushButton("Sync from fit")
        b_sync.setToolTip("Re-read the photon energy from the currently "
                          "selected XModFit model")
        b_sync.clicked.connect(self.refresh_now)
        b_reload = QPushButton("Reload page")
        b_reload.setToolTip("Rebuild the page from gid_template.html "
                            "(use after editing the template)")
        b_reload.clicked.connect(self.reload_scene)
        b_browser = QPushButton("Open in browser")
        b_browser.setToolTip("Write a standalone copy and open it in your "
                             "default browser")
        b_browser.clicked.connect(self._open_in_browser)
        bar.addWidget(self.status, 1)
        for b in (b_sync, b_reload, b_browser):
            bar.addWidget(b)
        lay.addLayout(bar)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_now)
        self._timer.start(self.POLL_MS)

    # -- model resolution ---------------------------------------------------
    def _resolve_func(self):
        newfunc = _find_func(self._parent_widget)
        if newfunc is not None and newfunc is not self.func:
            self.func = newfunc
            self._last = None      # force a push on the next refresh
        return self.func

    # -- lifecycle ----------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        if self._webengine_ok and not self._loaded:
            self.reload_scene()

    def reload_scene(self):
        if not self._webengine_ok:
            return
        self._resolve_func()
        cfg = cfg_from_func(self.func)
        try:
            html = build_html(cfg, self.template_path)
        except FileNotFoundError as exc:
            self.view.setHtml(
                "<html><body style='font-family:system-ui,Arial;padding:28px;"
                "color:#14212A'><h3>Template not found</h3><p>" + str(exc)
                + "</p></body></html>")
            self._loaded = False
            return
        self.view.setHtml(html)
        self._loaded = True
        self._last = cfg
        self._set_status(cfg)

    def _set_status(self, cfg):
        if not cfg:
            self.status.setText("standalone (no fit attached) \u2014 all "
                                "parameters are yours to set")
        elif "E" in cfg:
            self.status.setText(
                f"energy {cfg['E']:.4f} keV from {cfg.get('model','the fit')}"
                + (f"; {cfg['nlayers']} layers" if cfg.get("nlayers") else ""))
        else:
            self.status.setText(
                f"{cfg.get('model','model')} selected, but it exposes no E "
                "\u2014 set the energy in the panel")

    # -- push changes into the running page (no reload) ----------------------
    def refresh_now(self):
        if not self._webengine_ok:
            return
        self._resolve_func()
        if not self._loaded:
            self.reload_scene()
            return
        cfg = cfg_from_func(self.func)
        if cfg == self._last:
            return
        self._last = cfg
        self.view.page().runJavaScript(
            f"if(window.setFromFit)setFromFit({json.dumps(cfg)});")
        self._set_status(cfg)

    def _open_in_browser(self):
        cfg = cfg_from_func(self._resolve_func())
        path = os.path.join(tempfile.gettempdir(), "gid_explorer_xmodfit.html")
        write_browser_page(path, cfg, self.template_path)
        webbrowser.open("file://" + path)
        self.status.setText("opened a standalone copy in your browser")


def open_viewer(parent=None, func=None, template_path=None):
    dlg = GIDViewerDialog(parent=parent, func=func, template_path=template_path)
    dlg.show()
    return dlg


# ------------------------------------------------------------ standalone ----
class _StubFunc:
    """Minimal stand-in so the identical code path can be exercised with no
    XModFit and no fitting stack."""
    def __init__(self, E=10.0, d=(0.0, 15.0, 20.0, 1.0)):
        self.E = float(E)
        self.__mkeys__ = ["Model"]
        self.__d__ = {"Model": tuple(d)}
        self.__mpar__ = {"Model": {"d": list(d)}}


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "--browser":
        out = args[1] if len(args) > 1 else "gid_explorer.html"
        E = float(args[2]) if len(args) > 2 else None
        cfg = cfg_from_func(_StubFunc(E)) if E else {}
        print("wrote", write_browser_page(out, cfg))
        sys.exit(0)
    if not _HAVE_QT:
        print("No Qt binding available. Use:\n"
              "  python GIDViewerDialog.py --browser out.html [energy_keV]")
        sys.exit(1)
    E = float(args[0]) if args else None
    app = QApplication(sys.argv)
    open_viewer(func=_StubFunc(E) if E else None)
    sys.exit(app.exec_())
