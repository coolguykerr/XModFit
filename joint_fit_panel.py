"""
joint_fit_panel.py -- the "Joint Fit" dock (the third tab, alongside
Functions / Data) for XModJointFit.

It is a thin QWidget that drives a ``JointFitSpec`` and asks the host XModFit
window (its ``parent_app``) to do the heavy lifting (data access, param-table
refresh, running the fit).  The coupling to XModFit is a SMALL, documented
contract so this panel is testable head-less with a stub parent:

    parent_app.data                     dict: fname -> {curve -> {'x','y','yerr'}}
    parent_app.selected_data_names()    -> list[str] currently selected in the
                                           main Data list (what "Add selected" adds)
    parent_app.joint_model_factory()    -> a FRESH model instance of the currently
                                           selected function (the constrained model)
    parent_app.joint_model_name()       -> str, the constrained model's display name
    parent_app.fit                      the compiled JointFit (set by run)
    parent_app.refresh_param_tables()   re-render the stock param tables from
                                           parent_app.fit.fit_params (master/detail)
    parent_app.run_joint_fit()          compile spec -> JointFit -> fit -> fan out

Design notes
------------
* MODEL IS CONSTRAINED (user decision): every dataset in a joint fit uses the one
  currently-selected function.  ``JointFitSpec`` still carries a per-dataset
  ``model_class``/``model`` field and ``compile()`` already handles heterogeneous
  models, so lifting the constraint later is: expose a per-dataset model picker
  here + set ``spec.constrain_model = False``.  Nothing else changes.
* Per-dataset PARAMETER VALUES are shown as ARRAYS in the READ-ONLY readout only
  (user decision).  Editing is master/detail: pick a dataset in the selector and
  the stock param tables show/edit THAT dataset; shared params are badged.
* Persistent model instances: adding a dataset instantiates the constrained model
  ONCE and stores the instance in the spec, so edits made in the param tables
  survive a re-``compile`` (compile uses ``model=`` instances, not fresh classes).
* Axis-1 (sharing, this panel) and Axis-2 (covariance sidecars, this panel) are
  independent, non-exclusive controls -- per plan section 5a.
"""

from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QComboBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView, QFileDialog, QGroupBox,
    QStyledItemDelegate,
)
from PyQt5.QtCore import Qt

from joint_fit_spec import JointFitSpec


def display_name(key):
    """Human-facing label for a composite dataset key '<num><>path' -> '<num>: <basename>'.
    The full key (with path) stays the item's stored text; only the display changes."""
    try:
        num, path = key.split('<>', 1)
        return '%s: %s' % (num, os.path.basename(path))
    except Exception:
        return key


class NameDelegate(QStyledItemDelegate):
    """Paints dataset items as '<num>: <basename>' while leaving item.text() (the full
    composite key parsed throughout the app) untouched."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.text = display_name(option.text)


# --------------------------------------------------------------------------- #
#  pure logic (no Qt) -- kept module-level so it is unit-testable head-less
# --------------------------------------------------------------------------- #
def enumerate_shareable_params(model):
    """Ordered list of a model's shareable local keys.

    Scalars first (``qoff``, ``yscale``, ...), then multipar entries
    (``__Model_d_000`` ...).  ``__mpar__`` and private bookkeeping keys are
    skipped.  Returns ``[(local_key, label)]`` for the sharing table."""
    if getattr(model, "params", None) is None:
        model.init_params()
    scalars, multipar = [], []
    for k in model.params.keys():
        if k == "__mpar__":
            continue
        if k.startswith("__"):
            body = k[2:]
            try:
                mkey, pkey, num = body.rsplit("_", 2)
                multipar.append((k, f"{mkey}: {pkey}[{int(num)}]"))
            except ValueError:
                continue
        else:
            scalars.append((k, k))
    return scalars + multipar


def array_summary_rows(spec, jf=None):
    """Build the read-only readout: one row per parameter with its scope and
    value(s).  Shared -> single value; per-dataset -> the array across datasets.

    Uses the compiled ``JointFit`` (``jf``) master params if given (post-fit /
    live values); otherwise reads the spec's per-dataset model instances.
    Returns ``[(label, scope, value_str)]``.
    """
    names = spec.dataset_names
    if not names:
        return []
    template = _template_model(spec)
    if template is None:
        return []
    rows = []
    for lkey, label in enumerate_shareable_params(template):
        scope = spec.scope_of(names[0], lkey)
        # a per-dataset param is 'per-dataset' for names[0]; a subset-shared one
        # may be shared for some datasets -- report the first dataset's scope,
        # which is what the selector's default (first dataset) shows.
        if scope.startswith("shared"):
            v = _param_value(spec, jf, names[0], lkey)
            vs = "—" if v is None else _fmt(v)
        else:
            vals = [_param_value(spec, jf, n, lkey) for n in names]
            vs = "[" + ", ".join("—" if v is None else _fmt(v) for v in vals) + "]"
        rows.append((label, scope, vs))
    return rows


def _template_model(spec):
    for e in spec.datasets:
        if e.model is not None:
            return e.model
    return None


def _param_value(spec, jf, dataset_name, lkey):
    """Current value of (dataset, local_key): prefer the compiled JointFit's
    per-dataset model (reflects a finished fit); fall back to the spec model."""
    for e in spec.datasets:
        if e.name == dataset_name and e.model is not None:
            p = getattr(e.model, "params", None)
            if p is not None and lkey in p:
                return p[lkey].value
    return None


def _fmt(v):
    try:
        return "%.4g" % float(v)
    except (TypeError, ValueError):
        return str(v)


class JointDatasetList(QListWidget):
    """Joint-fit dataset list with the same plain-click rule as the main Data list:
    a plain click on a dataset ALREADY in a multi-selection keeps the whole selection
    (all joint curves stay overlaid) and just switches which dataset's parameters show
    (the detail); a plain click on a new dataset collapses (Qt default); Ctrl+click
    toggles add/remove. Routes the detail switch to ``JointFitPanel._on_joint_detail_click``."""
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self._panel = panel

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and e.modifiers() == Qt.NoModifier:
            item = self.itemAt(e.pos())
            if item is not None and item.isSelected() and len(self.selectedItems()) > 1:
                try:
                    self._panel._on_joint_detail_click(item.text())
                except Exception:
                    import traceback; traceback.print_exc()
                return
        super().mousePressEvent(e)


# --------------------------------------------------------------------------- #
#  the dock widget
# --------------------------------------------------------------------------- #
class JointFitPanel(QWidget):
    def __init__(self, parent_app=None):
        super().__init__()
        self._refreshing = False        # guards itemChanged during programmatic fills
        self.parent_app = parent_app
        self.spec = JointFitSpec()
        self.spec.constrain_model = True          # single model for now (see module doc)
        self._build_ui()

    # ---- UI ------------------------------------------------------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)

        self.enableCheck = QCheckBox("Enable joint fit (fit selected datasets together)")
        self.enableCheck.stateChanged.connect(self._on_enable)
        root.addWidget(self.enableCheck)

        # Everything below the toggle lives in a collapsible body, so with joint fit OFF
        # the Data screen shows just this one checkbox (the overlay under the Data list).
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)

        self.modelLabel = QLabel("Model: (select a function first)")
        body.addWidget(self.modelLabel)

        self.fixedShareLabel = QLabel(
            "In a joint fit, all fixed parameters are shared across every dataset.")
        self.fixedShareLabel.setWordWrap(True)
        self.fixedShareLabel.setStyleSheet("color: #555; font-style: italic;")
        body.addWidget(self.fixedShareLabel)

        # --- datasets in this joint fit ---
        dsBox = QGroupBox("Datasets in this joint fit")
        dsl = QGridLayout(dsBox)
        self.addBtn = QPushButton("Add selected \u2190 Data")
        self.addBtn.clicked.connect(self.add_selected_datasets)
        self.removeBtn = QPushButton("Remove")
        self.removeBtn.clicked.connect(self.remove_selected_datasets)
        self.datasetList = JointDatasetList(self)
        self.datasetList.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.datasetList.setItemDelegate(NameDelegate(self.datasetList))   # show basenames
        self.datasetList.itemSelectionChanged.connect(self._on_dataset_list_selection)
        dsl.addWidget(self.addBtn, 0, 0)
        dsl.addWidget(self.removeBtn, 0, 1)
        dsl.addWidget(self.datasetList, 1, 0, 1, 2)
        body.addWidget(dsBox)

        # --- shared parameters ---
        shBox = QGroupBox("Shared parameters (checked = tied across all datasets)")
        shl = QVBoxLayout(shBox)
        self.sharedTable = QTableWidget(0, 2)
        self.sharedTable.setHorizontalHeaderLabels(["Parameter", "Shared"])
        self.sharedTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.sharedTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sharedTable.itemChanged.connect(self._on_shared_item_changed)
        shl.addWidget(self.sharedTable)
        body.addWidget(shBox)

        # --- covariance sidecars (Axis-2, optional) ---
        covBox = QGroupBox("Correlated-error covariance (optional, per dataset)")
        covl = QVBoxLayout(covBox)
        self.covTable = QTableWidget(0, 3)
        self.covTable.setHorizontalHeaderLabels(["Dataset", "Covariance file", "Raw file"])
        self.covTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.covTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.covTable.cellDoubleClicked.connect(self._on_cov_double_clicked)
        covl.addWidget(QLabel("Double-click a cell to attach a sidecar (.npz or *_c_factors.txt)."))
        covl.addWidget(self.covTable)
        body.addWidget(covBox)

        # --- per-dataset fit range (xmin:xmax) ---
        rngBox = QGroupBox("Per-dataset fit range (blank = use the global Xrange box)")
        rngl = QVBoxLayout(rngBox)
        self.rangeTable = QTableWidget(0, 3)
        self.rangeTable.setHorizontalHeaderLabels(["Dataset", "xmin", "xmax"])
        self.rangeTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        # only xmin/xmax are editable; the dataset column is read-only (set per-cell)
        self.rangeTable.itemChanged.connect(self._on_range_item_changed)
        rngl.addWidget(QLabel("Double-click xmin/xmax to edit; each row fits only "
                              "its own q-window."))
        rngl.addWidget(self.rangeTable)
        body.addWidget(rngBox)

        # --- master/detail dataset selector ---
        selRow = QHBoxLayout()
        selRow.addWidget(QLabel("Show / edit dataset:"))
        self.datasetSelector = QComboBox()
        # items display basenames; the full key is stored as itemData and read via
        # currentData(), so downstream code always gets the real dataset key.
        self.datasetSelector.currentIndexChanged.connect(self._on_selector_changed)
        selRow.addWidget(self.datasetSelector, 1)
        body.addLayout(selRow)

        # --- read-only array readout ---
        roBox = QGroupBox("Parameter readout (per-dataset values shown as arrays)")
        rol = QVBoxLayout(roBox)
        self.readoutTable = QTableWidget(0, 3)
        self.readoutTable.setHorizontalHeaderLabels(["Parameter", "Scope", "Value(s)"])
        self.readoutTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.readoutTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        rol.addWidget(self.readoutTable)
        body.addWidget(roBox)

        # --- save / load ---
        ioRow = QHBoxLayout()
        self.saveJointBtn = QPushButton("Save joint fit…")
        self.saveJointBtn.clicked.connect(lambda: self._p("saveJointFit"))
        self.loadJointBtn = QPushButton("Load joint fit…")
        self.loadJointBtn.clicked.connect(lambda: self._p("loadJointFit"))
        self.loadParBtn = QPushButton("Load .par into joint…")
        self.loadParBtn.clicked.connect(lambda: self._p("loadParIntoJoint"))
        ioRow.addWidget(self.saveJointBtn)
        ioRow.addWidget(self.loadJointBtn)
        ioRow.addWidget(self.loadParBtn)
        body.addLayout(ioRow)

        # --- run ---
        self.runBtn = QPushButton("Run joint fit")
        self.runBtn.clicked.connect(self.run)
        body.addWidget(self.runBtn)

        # --- joint MCMC (its own window; enabled only after a joint fit) ---
        self.jointMCMCBtn = QPushButton("Joint MCMC / Param Error…")
        self.jointMCMCBtn.setToolTip(
            "Open the dedicated joint-fit MCMC window (separate from the "
            "individual-fit MCMC). Run a joint fit first; the window explains what "
            "to do if none has been run yet.")
        # Always enabled so it is discoverable; the host guards on 'no joint fit yet'
        # with an actionable message (the button was previously left disabled after a
        # joint fit run via the main Fit button, which never re-enabled it).
        self.jointMCMCBtn.clicked.connect(lambda: self._p("confInterval_joint_emcee"))
        body.addWidget(self.jointMCMCBtn)

        root.addWidget(self.body)
        self.body.setVisible(False)       # collapsed until 'Enable joint fit' is ticked

        self.refresh_model_label()
        self._refresh_shared_table()      # show hint/preview immediately on open

    # ---- host callbacks (defensive: no-op if parent lacks the method) ---- #
    def _p(self, method, *a, **k):
        fn = getattr(self.parent_app, method, None)
        return fn(*a, **k) if callable(fn) else None

    def sync_available_datasets(self, available_names):
        """Called by the host when the master Data list changes (esp. on delete).

        Prunes any joint-fit dataset that is no longer loaded, updates the UI, and
        shows a non-modal notice naming what was dropped so a joint fit is never
        silently left referencing a deleted dataset."""
        gone = self.spec.prune_missing(available_names)
        if gone:
            self._resync_all()
            try:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self, "Joint fit updated",
                    "Removed from the joint fit (deleted from Data): %s"
                    % ", ".join(gone))
            except Exception:
                pass
        return gone

    def _on_enable(self, state):
        on = bool(state)
        if self.parent_app is not None:
            self.parent_app.jointFitEnabled = on
        if getattr(self, 'body', None) is not None:
            self.body.setVisible(on)     # collapse/expand the overlay body

    def refresh_model_label(self):
        name = self._p("joint_model_name") or "(select a function first)"
        self.modelLabel.setText("Model: %s  (single model for all datasets)" % name)

    # ---- dataset add / remove ------------------------------------------- #
    def add_selected_datasets(self):
        names = self._p("selected_data_names") or []
        if not names:
            self._warn("No datasets selected in the Data list.\n"
                       "Select one or more datasets there, then click Add.")
            return
        # get ONE model instance up front so a missing/None factory is caught
        # loudly instead of silently attaching model-less datasets (which render
        # an empty parameter table).
        try:
            probe = self._p("joint_model_factory")
        except Exception as e:
            self._warn("Could not build a model (joint_model_factory raised):\n%s\n\n"
                       "Check that a function is selected in the Functions tab and "
                       "that the integration methods in xmodfit.py are present." % e)
            return
        if probe is None:
            self._warn("No model available (joint_model_factory returned None).\n\n"
                       "Select a function in the Functions tab first. If one is "
                       "selected, the joint_model_factory / joint_model_name contract "
                       "methods may be missing from your xmodfit.py integration.")
            return
        added = 0
        for nm in names:
            if nm in self.spec.dataset_names:
                continue
            # seed each dataset from its OWN remembered individual fit if it has one
            # (joint_model_for), else the current view (joint_model_factory).
            model = self._p("joint_model_for", nm)
            if model is None:
                model = self._p("joint_model_factory")
            self.spec.add_dataset(nm, model=model)
            added += 1
        self._resync_all()

    def _warn(self, msg):
        try:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Joint fit", msg)
        except Exception:
            pass

    def remove_selected_datasets(self):
        for it in self.datasetList.selectedItems():
            self.spec.remove_dataset(it.text())
        self._resync_all()

    # ---- shared flagging ------------------------------------------------ #
    def _on_shared_item_changed(self, item):
        # fires when the user toggles the checkbox (Qt has already updated the
        # state); ignore programmatic fills via the guard.
        if self._refreshing or item.column() != 1:
            return
        lkey = self.sharedTable.item(item.row(), 0).data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self.spec.set_shared(lkey)                       # tied across all datasets
        else:
            self.spec.unset_shared(lkey)                     # back to per-dataset
        self._refresh_readout()

    # ---- covariance pickers --------------------------------------------- #
    def _on_cov_double_clicked(self, row, col):
        name = self.covTable.item(row, 0).data(Qt.UserRole) or self.covTable.item(row, 0).text()
        if col == 1:
            f, _ = QFileDialog.getOpenFileName(self, "Covariance sidecar", "",
                                               "Sidecar (*.npz *_c_factors.txt)")
            if f:
                self.spec.add_dataset(name, cov_path=f)
        elif col == 2:
            f, _ = QFileDialog.getOpenFileName(self, "Raw 4-column data file")
            if f:
                self.spec.add_dataset(name, raw_path=f)
        self._refresh_cov_table()

    # ---- dataset list drives the main-window view ----------------------- #
    def _on_dataset_list_selection(self):
        """Selecting dataset(s) in the joint list populates the main Parameters tables
        (detail = last selected) and overlays the selected datasets' joint curves.
        Guarded so programmatic list refills don't fire it."""
        if self._refreshing:
            return
        names = [it.text() for it in self.datasetList.selectedItems()]
        if not names:
            # blank-click / deselect-all in the joint list -> drop the joint view
            # (symmetric with a blank-click in the general Data list).
            self._p("clear_joint_view")
            return
        # Selection model (item B2): a PLAIN click is mutually exclusive with the
        # individual Data list (classic one-view behaviour). Hold Ctrl to KEEP the
        # individual selection so the two views overlay (host's _maybe_combined_overlay).
        if not self._p("_overlay_modifier_active"):
            self._p("clear_individual_selection")
        # mirror the detail into the combo without recursing
        self.datasetSelector.blockSignals(True)
        _idx = self.datasetSelector.findData(names[-1])
        if _idx >= 0:
            self.datasetSelector.setCurrentIndex(_idx)
        self.datasetSelector.blockSignals(False)
        if getattr(self.parent_app, "show_joint_datasets", None):
            self._p("show_joint_datasets", names)
        else:
            self._p("show_joint_dataset", names[-1])

    def _on_joint_detail_click(self, name):
        """Plain click on an already-selected joint dataset within a multi-selection:
        switch the DETAIL (which dataset's parameters show) WITHOUT collapsing the
        selection -- all joint curves stay overlaid. Does not touch the individual
        selection (so a combined overlay is preserved)."""
        if self._refreshing:
            return
        names = [it.text() for it in self.datasetList.selectedItems()]
        if not names:
            return
        ordered = [n for n in names if n != name] + [name]   # clicked = detail (last)
        self.datasetSelector.blockSignals(True)
        _idx = self.datasetSelector.findData(name)
        if _idx >= 0:
            self.datasetSelector.setCurrentIndex(_idx)
        self.datasetSelector.blockSignals(False)
        if getattr(self.parent_app, "show_joint_datasets", None):
            self._p("show_joint_datasets", ordered)
        else:
            self._p("show_joint_dataset", name)

    # ---- master/detail selector ----------------------------------------- #
    def _on_selector_changed(self, *args):
        self._on_select_dataset(self.datasetSelector.currentData())

    def _on_select_dataset(self, name):
        if not name:
            return
        # Prefer the host's joint-view painter (activates the JointFit, refreshes the
        # param tables AND plots this dataset's joint curve). Fall back to the old
        # tables-only path if the host predates show_joint_dataset.
        if getattr(self.parent_app, "show_joint_dataset", None):
            self._p("show_joint_dataset", name)
            return
        jf = getattr(self.parent_app, "fit", None)
        if jf is not None and hasattr(jf, "set_current"):
            try:
                jf.set_current(name)
                self._p("refresh_param_tables")
            except (ValueError, IndexError):
                pass

    # ---- run ------------------------------------------------------------ #
    def run(self):
        if len(self.spec.datasets) < 2:
            self._warn("A joint fit needs at least 2 datasets. Add more via "
                       "'Add selected \u2190 Data'.")
            return
        if any(e.model is None for e in self.spec.datasets):
            self._warn("Some datasets have no model attached. Remove and re-add "
                       "them with a function selected/loaded first.")
            return
        fn = getattr(self.parent_app, "run_joint_fit", None)
        if not callable(fn):
            self._warn("run_joint_fit is not wired into xmodfit.py — see the "
                       "integration doc, parent contract (section 3).")
            return
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._warn("Joint fit failed:\n%s\n\n(full traceback in the console)" % e)
            return
        self._refresh_readout()

    # ---- resync all sub-widgets from the spec --------------------------- #
    def _resync_all(self):
        self._reexpand_shared_all()
        self._refresh_dataset_list()
        self._refresh_selector()
        self._refresh_shared_table()
        self._refresh_cov_table()
        self._refresh_ranges_table()
        self._refresh_readout()
        self._p("_refresh_data_labels")   # update joint-membership marks in the Data list

    def _refresh_dataset_list(self):
        prev = {it.text() for it in self.datasetList.selectedItems()}
        self.datasetList.blockSignals(True)   # don't fire _on_dataset_list_selection
        self.datasetList.clear()
        for nm in self.spec.dataset_names:
            item = QListWidgetItem(nm)
            self.datasetList.addItem(item)
            if nm in prev:
                item.setSelected(True)
        self.datasetList.blockSignals(False)

    def _refresh_selector(self):
        cur = self.datasetSelector.currentData()
        names = self.spec.dataset_names
        self.datasetSelector.blockSignals(True)
        self.datasetSelector.clear()
        for nm in names:
            self.datasetSelector.addItem(display_name(nm), nm)   # display basename, data=key
        if cur in names:
            self.datasetSelector.setCurrentIndex(names.index(cur))
        self.datasetSelector.blockSignals(False)

    def _refresh_shared_table(self):
        self._refreshing = True
        try:
            self.sharedTable.setRowCount(0)
            template = _template_model(self.spec)
            if template is None:
                # no dataset yet: try to list the selected function's params so the
                # user can SEE what will be shareable, but keep the checkboxes
                # disabled until >=1 dataset is added (sharing needs datasets).
                try:
                    template = self._p("joint_model_factory")
                except Exception:
                    template = None      # host not ready at dock-construction time
                actionable = False
            else:
                actionable = len(self.spec.datasets) >= 1
            if template is None:
                if self.spec.datasets:
                    # datasets ARE in the joint fit but none carries a model -->
                    # joint_model_factory returned None at Add time. This is the
                    # silent-empty-table failure; name it explicitly.
                    msg = ("Datasets added but no model is attached "
                           "(joint_model_factory returned None). Remove and re-add "
                           "with a function selected, or check the xmodfit.py contract.")
                else:
                    msg = "Select a function, then Add datasets to configure sharing"
                self.sharedTable.setRowCount(1)
                hint = QTableWidgetItem(msg)
                hint.setFlags(Qt.ItemIsEnabled)
                self.sharedTable.setItem(0, 0, hint)
                self.sharedTable.setSpan(0, 0, 1, 2)
                return
            shared_now = set()
            for g in self.spec.groups:
                for (_, lk) in g.endpoints:
                    shared_now.add(lk)
            for lkey, label in enumerate_shareable_params(template):
                r = self.sharedTable.rowCount()
                self.sharedTable.insertRow(r)
                it0 = QTableWidgetItem(label)
                it0.setData(Qt.UserRole, lkey)
                self.sharedTable.setItem(r, 0, it0)
                it1 = QTableWidgetItem("")
                flags = Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
                if not actionable:
                    flags = Qt.ItemIsUserCheckable      # visible but not toggleable
                it1.setFlags(flags)
                it1.setCheckState(Qt.Checked if lkey in shared_now else Qt.Unchecked)
                self.sharedTable.setItem(r, 1, it1)
        finally:
            self._refreshing = False

    def _reexpand_shared_all(self):
        """Keep 'shared across all' (bare) groups covering the CURRENT dataset set.

        set_shared snapshots dataset names at click time; without this, a param
        checked as shared BEFORE a dataset is added would silently exclude the
        new dataset. Heterogeneous-key groups (set_shared_endpoints) are not bare
        and are left untouched."""
        names = self.spec.dataset_names
        for g in self.spec.groups:
            if g.bare and g.endpoints:
                lkey = g.endpoints[0][1]
                g.endpoints = [(n, lkey) for n in names]

    def _refresh_cov_table(self):
        self.covTable.setRowCount(0)
        for e in self.spec.datasets:
            r = self.covTable.rowCount()
            self.covTable.insertRow(r)
            nameItem = QTableWidgetItem(display_name(e.name))   # show basename
            nameItem.setData(Qt.UserRole, e.name)               # keep the real key
            self.covTable.setItem(r, 0, nameItem)
            self.covTable.setItem(r, 1, QTableWidgetItem(os.path.basename(e.cov_path) if e.cov_path else ""))
            self.covTable.setItem(r, 2, QTableWidgetItem(os.path.basename(e.raw_path) if e.raw_path else ""))

    # ---- per-dataset fit range ------------------------------------------ #
    def _refresh_ranges_table(self):
        """Fill the fit-range table from the spec (one row per dataset).

        xmin/xmax cells show the entry's stored bound, or blank when None
        (meaning: fall back to the global Xrange box at fit time)."""
        self._refreshing = True
        try:
            self.rangeTable.setRowCount(0)
            for e in self.spec.datasets:
                r = self.rangeTable.rowCount()
                self.rangeTable.insertRow(r)
                nameItem = QTableWidgetItem(display_name(e.name))
                nameItem.setData(Qt.UserRole, e.name)
                nameItem.setFlags(Qt.ItemIsEnabled)          # read-only
                self.rangeTable.setItem(r, 0, nameItem)
                self.rangeTable.setItem(
                    r, 1, QTableWidgetItem("" if e.fit_xmin is None else "%g" % e.fit_xmin))
                self.rangeTable.setItem(
                    r, 2, QTableWidgetItem("" if e.fit_xmax is None else "%g" % e.fit_xmax))
        finally:
            self._refreshing = False

    def _on_range_item_changed(self, item):
        """Write an edited xmin/xmax back to the spec entry. Blank clears the
        bound (-> global fallback); non-numeric text is reverted."""
        if self._refreshing or item.column() not in (1, 2):
            return
        name = self.rangeTable.item(item.row(), 0).data(Qt.UserRole)
        entry = next((e for e in self.spec.datasets if e.name == name), None)
        if entry is None:
            return
        txt = item.text().strip()
        val = None
        if txt != "":
            try:
                val = float(txt)
            except ValueError:
                # revert to the stored value
                self._refreshing = True
                cur = entry.fit_xmin if item.column() == 1 else entry.fit_xmax
                item.setText("" if cur is None else "%g" % cur)
                self._refreshing = False
                return
        if item.column() == 1:
            entry.fit_xmin = val
        else:
            entry.fit_xmax = val

    def _refresh_readout(self):
        jf = getattr(self.parent_app, "fit", None)
        rows = array_summary_rows(self.spec, jf=jf)
        self.readoutTable.setRowCount(0)
        for (label, scope, vs) in rows:
            r = self.readoutTable.rowCount()
            self.readoutTable.insertRow(r)
            self.readoutTable.setItem(r, 0, QTableWidgetItem(label))
            self.readoutTable.setItem(r, 1, QTableWidgetItem(scope))
            self.readoutTable.setItem(r, 2, QTableWidgetItem(vs))