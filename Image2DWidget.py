r"""
Image2DWidget.py -- linked-panel 2D map viewer for XModFit.

Why this is not a PlotWidget
----------------------------
``PlotWidget.add_data`` (PlotWidget.py:202) requires equal-length 1-D arrays, and
``PlotWidget.Plot`` calls ``plotWidget.clear()`` (PlotWidget.py:366) which destroys
every item in the PlotItem -- including anything added with ``addItem``. That is why
``PlotWidget.addROI`` (:507) is dead code, and it is why an ImageItem cannot live
inside a PlotWidget. Hence a separate widget.

    RULE: never addItem() into a PlotWidget. The next Plot() destroys it.

Why MEASUREMENT-PIXEL space and not q space
-------------------------------------------
The q sampling of a GID map is deliberately non-uniform -- for the DPPC example
dq_xy runs 0.0043..0.0088 (2.0x, refined across the peak) and dq_z drifts
0.00073..0.00154. ``pg.ImageItem`` can only draw a uniform rectangle, so a q-space
rectangle would mis-place half the columns. Instead one image pixel is one measured
pixel and the real q values are placed on the axes with ``AxisItem.setTicks``, which
accepts FRACTIONAL positions -- so ``np.interp(v, uq, arange(n))`` puts a tick at
exactly the right place. This is also what the GID explorer's area-detector panel
does (Tools/GrazingIncidence/gid_template.html:1155-1190: "The image is the raw
(2th, a_f) grid the intensity was computed on").

Consequence worth knowing: the axis is genuinely non-linear. With <= 64 columns a
minor tick is drawn at every measured pixel, so the change in sampling is VISIBLE
on the axis rather than merely documented.

Entry contract (the 'image' output_params convention)
----------------------------------------------------
    x     : 1-D, len nx, strictly increasing -- real coordinate along the image x axis
    y     : 1-D, len ny, strictly increasing -- real coordinate along the image y axis
    z     : 2-D, shape (nx, ny), float. NaN = no measured pixel.
    plotType : 'image'
    role  : 'data' | 'model' | 'residual'   -- sets column order and the readout
    panel : group key; entries sharing it get one row, linked axes and shared levels
    cmap  : pyqtgraph colormap name
    zscale: 'log' | 'linear' | 'signed'
    names : [xlabel, ylabel, zlabel]

``z`` is col-major to match pyqtgraph's default ``imageAxisOrder``. NaN renders fully
transparent once a lookup table is set (verified: NaN pixel ARGB = (0,0,0,0)), so a
gappy grid needs no alpha handling.
"""

import sys

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout, QWidget)

ROLE_ORDER = ('data', 'model', 'residual')
MAX_ROWS = 3                    # a 10-dataset joint fit must not make 30 panels


class Image2DWidget(QWidget):
    """N linked 2-D map panels in measurement-pixel space with real-q axis labels."""

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._entries = []          # [(display_name, entry_dict)] as last given
        self._panels = []           # per-panel bookkeeping, see _rebuild
        self._bars = []
        self._sig = None
        self._user_levels = False   # True once the user drags a colorbar
        self._user_log = False      # True once the user toggles log10 by hand
        self._setting_log = False
        self._n_hidden = 0
        self._createWidgets()

    # ------------------------------------------------------------ layout ----
    def _createWidgets(self):
        self.vbLayout = QVBoxLayout(self)
        self.plotLayout = pg.LayoutWidget()
        self.vbLayout.addWidget(self.plotLayout)

        row, col = 0, 0
        self.logCheckBox = QCheckBox('log10 I')
        self.logCheckBox.setTristate(False)
        self.logCheckBox.setToolTip(
            'Log10 of the INTENSITY only. The axes are measurement-pixel indices and '
            'are never log-scaled, so the image can never move relative to its q tick '
            'labels. Non-positive pixels become NaN and render transparent -- with a '
            'negative background that is informative, not a bug. Ignored for the '
            'residual panel (zscale=signed).')
        self.logCheckBox.stateChanged.connect(self._on_log_changed)

        self.bgCheckBox = QCheckBox('White BG')
        self.bgCheckBox.stateChanged.connect(self._on_bg_changed)

        loLabel = QLabel('Levels %')
        self.loPctLineEdit = QLineEdit('0.5')
        self.loPctLineEdit.setMaximumWidth(50)
        self.loPctLineEdit.returnPressed.connect(self.autolevel)
        self.hiPctLineEdit = QLineEdit('99.5')
        self.hiPctLineEdit.setMaximumWidth(50)
        self.hiPctLineEdit.returnPressed.connect(self.autolevel)

        self.autoButton = QPushButton('Auto levels')
        self.autoButton.clicked.connect(self.autolevel)

        self.cmapComboBox = QComboBox()
        self.cmapComboBox.addItems(['CET-L3', 'viridis', 'inferno', 'CET-L16', 'plasma'])
        self.cmapComboBox.currentIndexChanged.connect(self._on_cmap_changed)

        self.linkCheckBox = QCheckBox('Link axes')
        self.linkCheckBox.setChecked(True)
        self.linkCheckBox.stateChanged.connect(self._on_link_changed)

        for w in (self.logCheckBox, self.bgCheckBox, loLabel, self.loPctLineEdit,
                  self.hiPctLineEdit, self.autoButton, self.cmapComboBox,
                  self.linkCheckBox):
            self.plotLayout.addWidget(w, row=row, col=col)
            col += 1
        self._ncols_controls = col

        row += 1
        self.glw = pg.GraphicsLayoutWidget()
        self.plotLayout.addWidget(self.glw, row=row, col=0, colspan=self._ncols_controls)
        self.glw.scene().sigMouseMoved.connect(self._on_mouse_moved)

        row += 1
        # A blank/near-blank map is almost always a configuration mistake, and the
        # producer usually knows exactly which one. Surfacing it HERE rather than
        # only in the generated-parameter list is the difference between "the panel
        # is broken" and "I loaded the wrong file".
        self.warnLabel = QLabel('')
        self.warnLabel.setWordWrap(True)
        self.warnLabel.setStyleSheet('color: #ff5555; font-weight: bold;')
        self.warnLabel.setVisible(False)
        self.plotLayout.addWidget(self.warnLabel, row=row, col=0,
                                  colspan=self._ncols_controls)

        row += 1
        self.crosshairLabel = QLabel('move the cursor over a panel')
        self.plotLayout.addWidget(self.crosshairLabel, row=row, col=0,
                                  colspan=self._ncols_controls)
        self._on_bg_changed()

    def set_warning(self, text):
        """Show (or clear) a warning banner above the crosshair readout."""
        text = str(text or '').strip()
        self.warnLabel.setText(text)
        self.warnLabel.setVisible(bool(text))

    # -------------------------------------------------------- public API ----
    def has_maps(self):
        return bool(self._panels)

    def panel_count(self):
        return len(self._panels)

    def clear_maps(self):
        self.glw.clear()
        self._panels, self._bars, self._entries = [], [], []
        self._sig = None
        self._n_hidden = 0
        self.set_warning('')
        self.crosshairLabel.setText('move the cursor over a panel')

    def set_maps(self, entries, title=None):
        """Show the given entries. Returns the number of panels drawn.

        ``entries`` is [(display_name, entry_dict)]. A rebuild happens only when the
        panel SIGNATURE changes; otherwise the existing ImageItems are refreshed in
        place, which is what keeps the zoom (and dragged colour levels) alive across
        a Calculate.
        """
        entries = [(n, e) for (n, e) in (entries or [])
                   if isinstance(e, dict) and e.get('plotType') == 'image']
        if not entries:
            self.clear_maps()
            return 0
        sig = self._signature(entries)
        self._entries = entries
        if sig != self._sig:
            self._sig = sig
            self._user_levels = False
            # Honour the entries' declared zscale. Without this the panel opens
            # linear, and for data spanning ~1e-3..7e2 with a median at 22% of
            # that range the whole map renders as near-black dark red -- CET-L3's
            # lower 40% is barely distinguishable from the background. Log is what
            # the producer asked for by setting zscale='log'; the user can still
            # override it, and once they do we stop second-guessing.
            if not self._user_log:
                want = any(e.get('zscale') == 'log' for _, e in entries)
                if want != self.logCheckBox.isChecked():
                    self._setting_log = True
                    try:
                        self.logCheckBox.setChecked(want)
                    finally:
                        self._setting_log = False
            self._rebuild(entries, title=title)
            for p in self._panels:
                p['plot'].autoRange()
        else:
            self._refresh(entries)
        return len(self._panels)

    def set_log(self, on):
        self.logCheckBox.setChecked(bool(on))

    def autolevel(self):
        self._user_levels = False
        self._apply_levels()

    def tick_spec(self, axis='bottom', row=0, col=0):
        """[(pos, label)] major ticks of a panel's axis. Test hook."""
        for p in self._panels:
            if p['row'] == row and p['col_index'] == col:
                return list(p['ticks'][axis])
        return []

    # ------------------------------------------------------- construction ---
    @staticmethod
    def _signature(entries):
        out = []
        for name, e in entries:
            z = np.asarray(e.get('z'))
            out.append((name, e.get('role'), e.get('panel'),
                        z.shape if z.ndim == 2 else None))
        return tuple(out)

    def _rebuild(self, entries, title=None):
        self.glw.clear()
        self._panels, self._bars = [], []

        groups = []                                  # preserve first-seen order
        for name, e in entries:
            key = e.get('panel', 'map')
            for g in groups:
                if g[0] == key:
                    g[1].append((name, e))
                    break
            else:
                groups.append((key, [(name, e)]))
        self._n_hidden = max(0, len(groups) - MAX_ROWS)
        groups = groups[:MAX_ROWS]

        for r, (gkey, members) in enumerate(groups):
            members = sorted(
                members,
                key=lambda ne: (ROLE_ORDER.index(ne[1]['role'])
                                if ne[1].get('role') in ROLE_ORDER else len(ROLE_ORDER)))
            # Column order: the shared-scale panels, then their one colorbar, then
            # each signed (residual) panel followed by its own colorbar. Putting the
            # shared bar directly after the panels it serves is what makes it read
            # as belonging to them.
            members = ([m for m in members if m[1].get('zscale') != 'signed'] +
                       [m for m in members if m[1].get('zscale') == 'signed'])
            n_shared = len([m for m in members if m[1].get('zscale') != 'signed'])
            gcol = 0
            shared = []                              # non-residual images of this row
            for ci, (name, e) in enumerate(members):
                p = self.glw.addPlot(row=r, col=gcol, title=name)
                img = pg.ImageItem()
                p.addItem(img)
                p.setDefaultPadding(0.0)
                p.showGrid(x=False, y=False)
                nm = e.get('names') or ['x', 'y', 'z']
                if gcol == 0:
                    p.setLabel('left', nm[1] if len(nm) > 1 else 'y')
                p.setLabel('bottom', nm[0])
                rec = dict(name=name, entry=e, plot=p, img=img, row=r, col_index=ci,
                           role=e.get('role'), group=gkey,
                           ticks={'bottom': [], 'left': []},
                           vline=pg.InfiniteLine(angle=90, movable=False,
                                                 pen=pg.mkPen('#888888', width=1)),
                           hline=pg.InfiniteLine(angle=0, movable=False,
                                                 pen=pg.mkPen('#888888', width=1)))
                rec['vline'].setZValue(10)
                rec['hline'].setZValue(10)
                p.addItem(rec['vline'], ignoreBounds=True)
                p.addItem(rec['hline'], ignoreBounds=True)
                self._panels.append(rec)
                if e.get('zscale') != 'signed':
                    shared.append(rec)
                gcol += 1

                # the shared bar goes immediately after the last shared panel...
                if shared and ci == n_shared - 1:
                    self._bars.append(self._add_bar(r, gcol, [s['img'] for s in shared],
                                                    shared[0]['entry'], symmetric=False))
                    gcol += 1
                # ...and a residual panel gets its own bar immediately to its right
                if e.get('zscale') == 'signed':
                    self._bars.append(self._add_bar(r, gcol, [img], e, symmetric=True))
                    gcol += 1

        self._apply_link()
        self._refresh(entries)
        self._apply_ticks()
        self._on_bg_changed()
        if self._n_hidden and self._panels:
            p0 = self._panels[0]['plot']
            p0.setTitle('%s   (+%d more group(s) not shown)'
                        % (self._panels[0]['name'], self._n_hidden))

    def _add_bar(self, row, col, images, entry, symmetric):
        nm = entry.get('names') or ['x', 'y', 'z']
        label = nm[2] if len(nm) > 2 else 'z'
        cmap = self._colormap(entry, symmetric)
        bar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=cmap, interactive=True,
                              label=label)
        try:
            bar.setImageItem(images)
        except Exception:
            for im in images:
                bar.setImageItem(im)
        self.glw.addItem(bar, row=row, col=col)
        try:
            bar.sigLevelsChanged.connect(self._on_user_levels)
        except Exception:
            pass
        return dict(bar=bar, images=images, entry=entry, symmetric=symmetric)

    def _colormap(self, entry, symmetric):
        name = 'CET-D1A' if symmetric else (entry.get('cmap')
                                            or self.cmapComboBox.currentText())
        try:
            return pg.colormap.get(name)
        except Exception:
            return pg.colormap.get('viridis')

    # ------------------------------------------------------------ ticks -----
    @staticmethod
    def _nice_step(span, target=6):
        if not np.isfinite(span) or span <= 0:
            return 1.0
        raw = span / float(max(1, target))
        p = 10.0 ** np.floor(np.log10(raw))
        m = raw / p
        return (1.0 if m < 1.5 else 2.0 if m < 3.5 else 5.0 if m < 7.5 else 10.0) * p

    @classmethod
    def _pixel_ticks(cls, uq, target=6):
        """[major, minor] tick lists for an AxisItem over pixel indices.

        Positions are FRACTIONAL pixel coordinates obtained by inverting the
        (non-uniform) q sampling with np.interp, so a tick labelled 1.50 sits at
        exactly q=1.50 rather than at the nearest measured column. Snapping to the
        nearest column was measured at 98% of its own error bound on the DPPC q_xy
        axis, which is why it is not done.
        """
        uq = np.asarray(uq, float)
        n = len(uq)
        if n < 2:
            return [[(0.0, '%.4g' % (uq[0] if n else 0.0))], []]
        idx = np.arange(n, dtype=float)
        lo, hi = float(uq[0]), float(uq[-1])
        step = cls._nice_step(hi - lo, target=target)
        dec = max(0, int(-np.floor(np.log10(step))) + 1)
        major = []
        v = np.ceil(lo / step) * step
        while v <= hi + 1e-12:
            major.append((float(np.interp(v, uq, idx)), ('%.*f' % (dec, v))))
            v += step
        if n <= 64:
            minor = [(float(i), '') for i in range(n)]       # one per measured pixel
        else:
            minor = []
            half = step / 2.0
            v = np.ceil(lo / half) * half
            while v <= hi + 1e-12:
                minor.append((float(np.interp(v, uq, idx)), ''))
                v += half
        return [major, minor]

    def _apply_ticks(self):
        for p in self._panels:
            e = p['entry']
            bx = self._pixel_ticks(e['x'])
            by = self._pixel_ticks(e['y'])
            p['plot'].getAxis('bottom').setTicks(bx)
            p['plot'].getAxis('left').setTicks(by)
            p['ticks']['bottom'] = bx[0]
            p['ticks']['left'] = by[0]

    # ------------------------------------------------------------ images ----
    def _z_for_display(self, entry):
        z = np.array(entry['z'], dtype=float)
        if entry.get('zscale') == 'signed':
            return z
        if self.logCheckBox.isChecked():
            with np.errstate(invalid='ignore', divide='ignore'):
                z = np.where(z > 0, z, np.nan)
                z = np.log10(z)
        return z

    def _refresh(self, entries):
        by_key = {}
        for name, e in entries:
            by_key[(name, e.get('role'), e.get('panel'))] = e
        for p in self._panels:
            e = by_key.get((p['name'], p['role'], p['group']), p['entry'])
            p['entry'] = e
            z = self._z_for_display(e)
            nx, ny = z.shape
            p['img'].setImage(z, autoLevels=False)
            p['img'].setRect(QRectF(-0.5, -0.5, float(nx), float(ny)))
        self._apply_levels()

    def _pct(self):
        try:
            lo = float(self.loPctLineEdit.text())
        except Exception:
            lo = 0.5
        try:
            hi = float(self.hiPctLineEdit.text())
        except Exception:
            hi = 99.5
        lo = min(max(lo, 0.0), 100.0)
        hi = min(max(hi, 0.0), 100.0)
        if hi <= lo:
            lo, hi = 0.5, 99.5
        return lo, hi

    def _apply_levels(self):
        if self._user_levels:
            return
        lo_p, hi_p = self._pct()
        for b in self._bars:
            if b['symmetric']:
                # zscale='signed': symmetric about zero, ALWAYS. Asymmetric levels on
                # a diverging map would put white somewhere other than zero, i.e. the
                # panel would lie. The percentile boxes are deliberately ignored here.
                z = self._z_for_display(b['entry'])
                v = np.nanpercentile(np.abs(z), 99.0) if np.any(np.isfinite(z)) else 1.0
                if not np.isfinite(v) or v <= 0:
                    v = 1.0
                levels = (-float(v), float(v))
            else:
                # levels from the DATA panel only, so data and model are directly
                # pixel-comparable rather than each self-scaled
                src = None
                for p in self._panels:
                    if p['img'] in b['images'] and p['role'] == 'data':
                        src = p
                        break
                if src is None:
                    for p in self._panels:
                        if p['img'] in b['images']:
                            src = p
                            break
                if src is None:
                    continue
                z = self._z_for_display(src['entry'])
                if not np.any(np.isfinite(z)):
                    continue
                a = float(np.nanpercentile(z, lo_p))
                c = float(np.nanpercentile(z, hi_p))
                if not (np.isfinite(a) and np.isfinite(c)) or c <= a:
                    a, c = (a, a + 1.0) if np.isfinite(a) else (0.0, 1.0)
                levels = (a, c)
            try:
                b['bar'].setLevels(levels)
            except Exception:
                for im in b['images']:
                    im.setLevels(levels)

    # ------------------------------------------------------------- slots ----
    def _on_log_changed(self):
        if not self._setting_log:
            self._user_log = True        # a hand toggle wins from now on
        self._user_levels = False
        if self._entries:
            self._refresh(self._entries)

    def _on_cmap_changed(self):
        for b in self._bars:
            if not b['symmetric']:
                try:
                    b['bar'].setColorMap(self._colormap(b['entry'], False))
                except Exception:
                    pass

    def _on_user_levels(self, *args):
        self._user_levels = True

    def _apply_link(self):
        if not self._panels:
            return
        on = self.linkCheckBox.isChecked()
        anchor = self._panels[0]['plot']
        for p in self._panels[1:]:
            p['plot'].setXLink(anchor if on else None)
            p['plot'].setYLink(anchor if on else None)

    def _on_link_changed(self):
        self._apply_link()

    def _on_bg_changed(self):
        white = self.bgCheckBox.isChecked()
        self.glw.setBackground('w' if white else 'k')
        pen = 'k' if white else 'w'
        for p in self._panels:
            for ax in ('left', 'bottom'):
                p['plot'].getAxis(ax).setPen(pen)
                p['plot'].getAxis(ax).setTextPen(pen)

    def _on_mouse_moved(self, pos):
        for p in self._panels:
            vb = p['plot'].vb
            if not vb.sceneBoundingRect().contains(pos):
                continue
            pt = vb.mapSceneToView(pos)
            nx, ny = np.asarray(p['entry']['z']).shape
            ix, iz = int(round(pt.x())), int(round(pt.y()))
            if ix < 0 or ix >= nx or iz < 0 or iz >= ny:
                continue
            for q in self._panels:
                q['vline'].setPos(ix)
                q['hline'].setPos(iz)
            self.crosshairLabel.setText(self._readout(ix, iz, group=p['group']))
            return

    def _readout(self, ix, iz, group=None):
        """Text for pixel (ix, iz). Callable without a Qt event -- test hook."""
        ps = [p for p in self._panels if group is None or p['group'] == group]
        if not ps:
            return ''
        e0 = ps[0]['entry']
        nm = e0.get('names') or ['x', 'y', 'z']
        parts = ['%s=%.5f' % (nm[0].split(' ')[0], float(e0['x'][ix])),
                 '%s=%.5f' % (nm[1].split(' ')[0] if len(nm) > 1 else 'y',
                              float(e0['y'][iz])),
                 'px(%d,%d)' % (ix, iz)]
        for role, lab in (('data', 'I_data'), ('model', 'I_model'),
                          ('residual', '(d-f)/s')):
            hit = [p for p in ps if p['role'] == role]
            if not hit:
                continue
            v = np.asarray(hit[0]['entry']['z'])[ix, iz]
            parts.append('%s=%s' % (lab, 'nan' if not np.isfinite(v)
                                    else ('%+.2f' % v if role == 'residual'
                                          else '%.4e' % v)))
        return '   '.join(parts)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # synthetic 49 x 811 map with NON-UNIFORM axes and NaN holes, i.e. the awkward
    # shape the real DPPC map has
    ux = np.concatenate([np.arange(1.2351, 1.4280, 0.0088),
                         np.arange(1.4280, 1.5160, 0.0044),
                         np.arange(1.5160, 1.5690, 0.0088)])
    uz = 0.001 + np.cumsum(np.linspace(0.00073, 0.00154, 811))
    nx, ny = len(ux), len(uz)
    X, Z = np.meshgrid(ux, uz, indexing='ij')
    peak = 100.0 * np.exp(-(X - 1.4674) ** 2 / (2 * 0.0067 ** 2)) \
        * np.sinc(19.0 * 0.95 * (Z - 0.43) / (2 * np.pi)) ** 2
    peak = peak + 60.0 * np.exp(-(X - 1.4885) ** 2 / (2 * 0.0067 ** 2)) \
        * np.sinc(19.0 * 0.95 * Z / (2 * np.pi)) ** 2
    model = peak + 40.0 + 5.0 * np.sin(8 * Z)
    rng = np.random.RandomState(0)
    data = model * (1.0 + 0.05 * rng.randn(nx, ny))
    data[rng.rand(nx, ny) < 0.02] = np.nan
    resid = (data - model) / (0.4 * np.abs(model) + 1e-9)
    ax = dict(x=ux, y=uz, plotType='image', panel='map2d')
    ents = [
        ('map2d_data', dict(ax, z=data, role='data', cmap='CET-L3', zscale='log',
                            names=['q_xy (1/Angs)', 'q_z (1/Angs)', 'I data'])),
        ('map2d_model', dict(ax, z=model, role='model', cmap='CET-L3', zscale='log',
                             names=['q_xy (1/Angs)', 'q_z (1/Angs)', 'I model'])),
        ('map2d_residual', dict(ax, z=resid, role='residual', cmap='CET-D1A',
                                zscale='signed',
                                names=['q_xy (1/Angs)', 'q_z (1/Angs)', '(d-f)/sigma'])),
    ]
    w = Image2DWidget()
    n = w.set_maps(ents)
    print('panels: %d   grid %dx%d   NaN holes %d'
          % (n, nx, ny, int(np.isnan(data).sum())))
    print('q_xy major ticks:', w.tick_spec('bottom'))
    print('q_z  major ticks:', w.tick_spec('left'))
    print('readout(24,203):', w._readout(24, 203))
    # tick placement must be EXACT (fractional positions), not snapped
    err = max(abs(np.interp(p, np.arange(nx, dtype=float), ux) - float(l))
              for p, l in w.tick_spec('bottom'))
    print('worst q_xy tick label error: %.3e' % err)
    w.setWindowTitle('Image2DWidget demo')
    w.setGeometry(100, 100, 1300, 600)
    w.show()
    sys.exit(app.exec_())
