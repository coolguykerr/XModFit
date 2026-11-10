"""
joint_fit_orchestration.py -- glue between the XModJointFit GUI and JointFit.

This is the "minimal-first" orchestration path from the implementation plan
(section 7 / section 9 step 3): route a multi-dataset selection to a single
``JointFit``, optionally attaching a per-dataset covariance sidecar, without
first building the full remembered-"Joint Fit"-tab.  Keeping it in its own
module means the ~10-line hook inside the 4000-line ``xmodfit.py`` is tiny and
testable in isolation.

GUI hook (inside ``xmodfit.py.doFit``), replacing the per-file loop when the
user has ticked a "Joint fit selected datasets" box::

    from joint_fit_orchestration import build_jointfit_from_selection
    if self.jointFitEnabled and len(selItems) > 1:
        fnames = [it.text() for it in selItems]
        self.fit = build_jointfit_from_selection(
            self.data, fnames, self.funcClass,
            cov_paths=self._cov_path,          # {fname: sidecar path}
            raw_paths=self._raw_path,          # {fname: original data file} (for *_c_factors.txt)
            shared=self.jointSharedParams,     # e.g. [] for Axis-2, ['theta','phi'] for Axis-1
        )
        self.fit.functionCalled.connect(self.fitCallback)
        self.fit_report, self.fit_message = self.fit.perform_fit(
            self.xmin, self.xmax, fit_scale=self.fit_scale, fit_method=self.fit_method)
        ... update tables / plot from self.fit.master_params ...
        return

``self.funcClass`` is the currently-selected model class (XModFit already
tracks the chosen function).  For Axis-2 correlated-error work it is XLayers.
"""

import warnings

import numpy as np

from Joint_Fit_Routines import JointFit
from joint_fit_spec import load_or_note


def _multi_curve_msg(name, dataset):
    """The joint_fit_spec wording (01 6.14) for a dataset with several sub-curves.
    Without a name the curve keys identify the dataset."""
    if name is None:
        name = 'with curves (%s)' % ', '.join(map(str, dataset))
    return ("dataset %s has %d curves; only the first (%s) enters the joint fit"
            % (name, len(dataset), next(iter(dataset))))


def _primary_curve(dataset, warn=True, name=None):
    """XModFit stores data as data[fname][curve_key] -> {'x','y','yerr',...}.
    Return the first curve's arrays (Axis-2 datasets are single-curve).

    The later sub-curves are dropped; with ``warn`` that is reported through
    ``warnings.warn`` (01 6.14). ``build_jointfit_from_selection`` passes
    ``warn=False`` and records it in ``JointFit.warnings`` instead."""
    if warn and len(dataset) > 1:
        warnings.warn(_multi_curve_msg(name, dataset), stacklevel=2)
    key = list(dataset.keys())[0]
    d = dataset[key]
    return (np.asarray(d["x"], dtype=float),
            np.asarray(d["y"], dtype=float),
            np.asarray(d["yerr"], dtype=float))


def build_jointfit_from_selection(data, fnames, funcClass,
                                  cov_paths=None, raw_paths=None,
                                  shared=None, funcs=None, covariance_mode=None):
    """Assemble a ``JointFit`` from a GUI multi-selection.

    Parameters
    ----------
    data      : dict         XModFit's ``self.data`` (data[fname][curve] -> arrays).
    fnames    : list[str]    selected dataset file names, in display order.
    funcClass : type         model class (e.g. XLayers). Ignored where ``funcs`` given.
    cov_paths : dict|None    {fname: sidecar path (*.cov.json, .npz or
                             *_c_factors.txt)}, loaded with
                             ``covariance_core.load_covariance``. If a fname has
                             a path, its covariance is attached (turns on
                             cross-dataset GLS whitening for the fit); a file
                             that cannot be loaded fits that dataset without it,
                             with a downgrade warning (spec 17 6.5).
    raw_paths : dict|None    {fname: original 4-column data file} -- required only
                             for *_c_factors.txt sidecars (to recover the sf column).
    shared    : list|None    bare parameter names tied across datasets (Axis 1).
                             None / [] -> share nothing (Axis 2 / independent).
    funcs     : dict|None    {fname: pre-built model instance} to override funcClass.
    covariance_mode : 'model' | 'data' | None   the JointFit's covariance mode
                             (None keeps the JointFit default, 'model').
    """
    cov_paths = cov_paths or {}
    raw_paths = raw_paths or {}
    funcs = funcs or {}

    jf = JointFit()
    if covariance_mode is not None:
        jf.covariance_mode = covariance_mode
    multi_curve = []
    for fname in fnames:
        x, y, yerr = _primary_curve(data[fname], warn=False)
        if len(data[fname]) > 1:
            multi_curve.append(_multi_curve_msg(fname, data[fname]))
        cov, note = None, None
        cp = cov_paths.get(fname)
        if cp:
            cov, note = load_or_note(fname, cp, raw_paths.get(fname))
        f = funcs.get(fname)
        if f is not None:
            obj = jf.add_dataset(fname, func=f, x=x, y=y, yerr=yerr, cov=cov)
        else:
            obj = jf.add_dataset(fname, funcClass=funcClass, x=x, y=y, yerr=yerr, cov=cov)
        if note is not None:                # re-added to every fit's warnings
            obj["cov_notes"].append(note)
    jf.build_master_params(shared=shared)
    # 01 6.14, as in JointFitSpec.compile: appended after build_master_params,
    # which resets the build-time warnings.
    jf.warnings.extend(multi_curve)
    return jf


def push_result_to_tables(jf, set_scalar, set_multipar):
    """Optional helper to fan a finished JointFit's master params back out to
    per-dataset GUI tables.  ``set_scalar(dataset_idx, name, value)`` and
    ``set_multipar(dataset_idx, mkey, pkey, num, value)`` are GUI callbacks.

    Master-key grammar (see JointFit._master_key):
        shared              -> bare name (applies to every dataset)
        d{i}_<name>         -> scalar for dataset i
        __d{i}_<mkey>_<pkey>_<num> -> multipar entry for dataset i
    """
    for mkey in jf.master_params:
        val = jf.master_params[mkey].value
        if mkey.startswith("__d"):
            body = mkey[2:]                       # d{i}_<mkey>_<pkey>_<num>
            head, pkey, num = body.rsplit("_", 2)
            i = int(head.split("_", 1)[0][1:])    # strip the leading 'd'
            base_mkey = head.split("_", 1)[1]
            set_multipar(i, base_mkey, pkey, int(num), val)
        elif mkey.startswith("d") and "_" in mkey and mkey[1:mkey.index("_")].isdigit():
            i = int(mkey[1:mkey.index("_")])
            name = mkey[mkey.index("_") + 1:]
            set_scalar(i, name, val)
        else:
            # shared bare param: apply to all datasets
            for i in range(len(jf.objectives)):
                set_scalar(i, mkey, val)