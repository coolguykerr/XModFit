"""
joint_fit_spec.py -- the editable JointFit *specification* layer (Axis-1 GUI).

This sits ABOVE the ``JointFit`` engine (``Joint_Fit_Routines.py``) and the
one-shot ``build_jointfit_from_selection`` orchestrator.  Its job is the thing
the engine deliberately does NOT hold: a *persistent, editable* description of a
joint fit that the GUI binds to and that survives dataset add/remove --

    * an ORDERED list of datasets (each with an optional per-dataset model class
      / covariance sidecar / raw file), keyed by dataset NAME, not index;
    * a declarative sharing specification (which local parameter is shared, over
      which subset of datasets), stored BY NAME so it is robust to add/remove;

plus the operations the plan (section 7) and the user asked for: add a dataset,
remove a dataset (pruning any sharing that referenced it), and flag a parameter
as shared / per-dataset.

Design constraints honoured (from XModFit_joint_fitting_plan.md):
  * section 5a composability: this layer only decides the parameter MAP (Axis 1);
    it never touches the residual/covariance branch (Axis 2).  A spec may carry a
    covariance sidecar per dataset AND share parameters -- the two are orthogonal.
  * why NAME-keyed: the engine namespaces per-dataset master keys by POSITION
    (``d{i}_qoff`` / ``__d{i}_Model_d_002``).  Removing a middle dataset renumbers
    every later dataset's keys, so a spec that stored sharing by index would
    silently re-target after an edit.  Storing endpoints as (name, local_key) and
    only resolving to indices at ``compile()`` time keeps edits safe; the ORDERED
    name list is the stable contract (plan section 7: "ordered filenames").

``compile()`` produces a FRESH ``JointFit`` each call (mirroring
``build_master_params``, which already rebuilds from scratch), so this layer needs
NO change to the validated engine.  Engine-level ``remove_dataset`` is therefore
optional; edit the spec and recompile instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np

from Joint_Fit_Routines import JointFit
import joint_covariance as jcov


# --------------------------------------------------------------------------- #
#  per-dataset entry
# --------------------------------------------------------------------------- #
@dataclass
class DatasetEntry:
    """One dataset in a joint fit.

    name       : unique dataset key (the GUI list item text / file name).
    model_class: model class for this dataset (e.g. XLayers).  May differ per
                 dataset -- a wafer_A_buffer.txt run and a wafer_B_bilayer.txt
                 run legitimately carry different layer counts.  Ignored if
                 ``model`` is given.
    model      : a pre-built model INSTANCE (overrides model_class).  Use this
                 to carry a per-dataset starting structure / .par into the fit.
    cov_path   : optional covariance sidecar (*.npz self-contained, or a
                 *_c_factors.txt needing ``raw_path``).  Attaching one on ANY
                 dataset turns on Axis-2 GLS whitening for the whole fit.
    raw_path   : original 4-column data file, required only for *_c_factors.txt.
    fit_xmin   : optional per-dataset lower fit bound (q). ``None`` -> fall back
                 to the joint fit's scalar Xrange. Independent of ``model.x``
                 (the model's evaluation grid); this is the [imin,imax] window.
    fit_xmax   : optional per-dataset upper fit bound (q). ``None`` -> scalar.
    limits     : optional {column_name: (lo, hi)} fit limits on data columns OTHER
                 than x. The x window stays in fit_xmin/fit_xmax -- those ARE the
                 'x' entry, and not migrating them keeps every existing caller and
                 the .jpar round-trip untouched. A column limit is not an interval
                 in index space, so the engine applies it as a boolean mask; the
                 mask itself is built by the HOST (which owns the DataFrames) and
                 passed to compile(), because this module is deliberately free of
                 XModFit data structures.
    """
    name: str
    model_class: Optional[type] = None
    model: Optional[object] = None
    cov_path: Optional[str] = None
    raw_path: Optional[str] = None
    fit_xmin: Optional[float] = None
    fit_xmax: Optional[float] = None
    limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None


# --------------------------------------------------------------------------- #
#  share-group (declarative, name-keyed)
# --------------------------------------------------------------------------- #
@dataclass
class ShareGroup:
    """A set of (dataset_name, local_key) endpoints tied to ONE master param.

    * endpoints stored BY NAME, resolved to engine (idx, local_key) at compile.
    * ``bare`` keeps an un-namespaced master name (share-across-ALL), which
      preserves the emcee angle gate for ``theta``/``phi`` (Axis-1 MCMC).
    * heterogeneous local keys are allowed (same physical layer at a different
      row in different datasets) -- the engine's ``share_endpoints`` handles it.
    """
    endpoints: List[Tuple[str, str]]              # [(dataset_name, local_key), ...]
    bare: bool = False
    name: Optional[str] = None                    # explicit master key (optional)

    def prune_dataset(self, dataset_name: str) -> "ShareGroup":
        self.endpoints = [(d, k) for (d, k) in self.endpoints if d != dataset_name]
        return self

    @property
    def n_members(self) -> int:
        return len(self.endpoints)


# --------------------------------------------------------------------------- #
#  the spec
# --------------------------------------------------------------------------- #
class JointFitSpec:
    """Editable specification of a joint fit; compiles to a ``JointFit``."""

    def __init__(self, default_model_class: Optional[type] = None):
        self.datasets: List[DatasetEntry] = []      # ordered
        self.groups: List[ShareGroup] = []
        self.default_model_class = default_model_class
        # When True (current default), every dataset uses one model (the GUI does
        # not expose a per-dataset model picker).  The per-dataset model_class /
        # model fields on DatasetEntry and the heterogeneous-model handling in
        # compile() are already in place, so allowing per-dataset models later is
        # exactly: set this False + expose a picker in the panel.  Nothing in
        # compile() reads this flag -- it is a GUI-affordance switch only.
        self.constrain_model: bool = True

    # ---- dataset selection / add / remove --------------------------------- #
    def add_dataset(self, name, model_class=None, model=None,
                    cov_path=None, raw_path=None, fit_xmin=None, fit_xmax=None,
                    limits=None):
        """Add a dataset (idempotent on name: re-adding UPDATES the entry)."""
        for e in self.datasets:
            if e.name == name:
                if model_class is not None: e.model_class = model_class
                if model is not None:       e.model = model
                if cov_path is not None:    e.cov_path = cov_path
                if raw_path is not None:    e.raw_path = raw_path
                if fit_xmin is not None:    e.fit_xmin = fit_xmin
                if fit_xmax is not None:    e.fit_xmax = fit_xmax
                if limits is not None:      e.limits = limits
                return e
        e = DatasetEntry(name=name,
                         model_class=model_class or self.default_model_class,
                         model=model, cov_path=cov_path, raw_path=raw_path,
                         fit_xmin=fit_xmin, fit_xmax=fit_xmax, limits=limits)
        self.datasets.append(e)
        return e

    def remove_dataset(self, name):
        """Remove a dataset AND prune it from every share-group.

        A share-group that drops below 2 members is deleted (a "shared" param
        with one endpoint is just a per-dataset param, so it falls back to the
        default per-dataset treatment automatically at compile time)."""
        self.datasets = [e for e in self.datasets if e.name != name]
        for g in self.groups:
            g.prune_dataset(name)
        self.groups = [g for g in self.groups if g.n_members >= 2]

    @property
    def dataset_names(self) -> List[str]:
        return [e.name for e in self.datasets]

    def reorder(self, names: List[str]):
        """Set the dataset order explicitly (the stable per-dataset key contract)."""
        by_name = {e.name: e for e in self.datasets}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(f"reorder: unknown datasets {missing}")
        self.datasets = [by_name[n] for n in names]

    def prune_missing(self, available_names) -> List[str]:
        """Drop any spec dataset no longer present in the host's master data list.

        Called when the user deletes a dataset from XModFit's Data list: without
        this the spec keeps a stale name and ``compile()`` would ``KeyError`` on
        ``data[name]``.  Reuses ``remove_dataset`` so sharing groups referencing a
        gone dataset are pruned and under-2-member groups demote to per-dataset.
        Returns the names that were removed (for a GUI notice)."""
        available = set(available_names)
        gone = [e.name for e in self.datasets if e.name not in available]
        for nm in gone:
            self.remove_dataset(nm)
        return gone

    # ---- shared / per-dataset flagging ------------------------------------ #
    def set_shared(self, local_key, dataset_names=None, bare=None, master_name=None):
        """Flag ``local_key`` as SHARED across all datasets (default) or a subset.

        dataset_names=None  -> share across every dataset that exposes the key
                               (``bare`` defaults True: keep un-namespaced master).
        dataset_names=[...] -> share across just that subset (``bare`` False).
        Replaces any existing group that already shares this key over an
        overlapping set (so toggling in the GUI is idempotent)."""
        names = list(dataset_names) if dataset_names is not None else list(self.dataset_names)
        if bare is None:
            bare = dataset_names is None
        endpoints = [(n, local_key) for n in names]
        # drop any prior group covering this exact key (simple GUI semantics)
        self.groups = [g for g in self.groups
                       if not all(k == local_key for (_, k) in g.endpoints)]
        self.groups.append(ShareGroup(endpoints=endpoints, bare=bare, name=master_name))

    def set_shared_endpoints(self, endpoints, bare=False, master_name=None):
        """Flag a set of heterogeneous ``(dataset_name, local_key)`` endpoints as
        one shared parameter (same physical layer, different row per dataset)."""
        self.groups.append(ShareGroup(endpoints=list(endpoints), bare=bare,
                                       name=master_name))

    def unset_shared(self, local_key):
        """Make ``local_key`` per-dataset again (remove groups sharing only it)."""
        self.groups = [g for g in self.groups
                       if not all(k == local_key for (_, k) in g.endpoints)]

    def scope_of(self, dataset_name, local_key) -> str:
        """'shared-all' / 'shared-subset' / 'per-dataset' for a (dataset, param).

        Used by the GUI to badge a parameter row.  A parameter is 'shared-all'
        only if its group covers every dataset that exposes the key."""
        for g in self.groups:
            if (dataset_name, local_key) in g.endpoints:
                covered = {d for (d, _) in g.endpoints}
                have = {e.name for e in self.datasets}   # (key-existence checked at compile)
                return "shared-all" if covered >= have else "shared-subset"
        return "per-dataset"

    # ---- compile to a JointFit -------------------------------------------- #
    def compile(self, data, masks=None) -> JointFit:
        """Build a fresh ``JointFit`` from this spec + XModFit's ``self.data``.

        ``data`` is XModFit's ``data[fname][curve] -> {'x','y','yerr',...}``.
        Datasets are registered in the spec's order (the stable key contract);
        covariance sidecars are loaded and attached; share-groups are translated
        into the engine's ``share_endpoints`` and applied via
        ``build_master_params(groups=...)``.
        """
        if len(self.datasets) < 2:
            raise ValueError("a joint fit needs at least 2 datasets")

        missing = [e.name for e in self.datasets if e.name not in data]
        if missing:
            raise ValueError(
                "joint-fit datasets no longer loaded: %s. "
                "They were removed from the Data list; call prune_missing() "
                "(the GUI does this on data removal) or re-add them." % missing)

        jf = JointFit()
        name_to_idx = {}
        for idx, e in enumerate(self.datasets):
            x, y, yerr = _primary_curve(data[e.name])
            cov = None
            if e.cov_path:
                cov = jcov.load_dataset_cov(e.cov_path, x=x, raw_path=e.raw_path)
            if e.model is not None:
                jf.add_dataset(e.name, func=e.model, x=x, y=y, yerr=yerr, cov=cov,
                               mask=(masks or {}).get(e.name))
            else:
                mc = e.model_class or self.default_model_class
                if mc is None:
                    raise ValueError(f"dataset '{e.name}' has no model class")
                jf.add_dataset(e.name, funcClass=mc, x=x, y=y, yerr=yerr, cov=cov,
                               mask=(masks or {}).get(e.name))
            name_to_idx[e.name] = idx

        # translate name-keyed groups -> engine (idx, local_key) endpoint groups,
        # dropping endpoints whose dataset no longer exists or lacks the key.
        engine_groups = []
        for g in self.groups:
            members = []
            for (dname, lkey) in g.endpoints:
                if dname not in name_to_idx:
                    continue
                fidx = name_to_idx[dname]
                if lkey in jf.objectives[fidx]["func"].params:
                    members.append((fidx, lkey))
            if len(members) >= 2:
                engine_groups.append({"members": members,
                                      "bare": g.bare, "name": g.name})

        jf.build_master_params(groups=engine_groups or None)
        return jf


# --------------------------------------------------------------------------- #
#  helper (same convention as joint_fit_orchestration._primary_curve)
# --------------------------------------------------------------------------- #
def _primary_curve(dataset):
    key = list(dataset.keys())[0]
    d = dataset[key]
    return (np.asarray(d["x"], dtype=float),
            np.asarray(d["y"], dtype=float),
            np.asarray(d["yerr"], dtype=float))