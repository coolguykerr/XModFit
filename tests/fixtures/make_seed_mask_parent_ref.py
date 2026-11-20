"""
Regenerate tests/fixtures/seed_mask_parent_ref.json -- the frozen mask=None
reference for validate_seed_mask.py check (j).

The values are the parallel MCMC log-probabilities computed by the PARENT
commit's code (pdb_logprob.py / pdb_joint_logprob.py at PARENT, i.e. before A1.18
added the column mask), at the fixed parameter vectors validate_seed_mask.py
uses, on the mask-off single-model and joint set-ups it builds. Check (j) then
proves mask=None still gives exactly those numbers without needing git history.

Run deliberately, from a checkout that still contains PARENT:

    "C:/Users/kerr/.conda/envs/cmc-py39/python.exe" tests/fixtures/make_seed_mask_parent_ref.py

This script is NOT a harness (tests/run_all.py does not discover tests/fixtures).
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PARENT = '98618c3'      # last commit before A1.18 touched pdb_logprob / pdb_joint_logprob
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'seed_mask_parent_ref.json')


def load_parent(modname):
    """Import PARENT's copy of `modname` under a private name."""
    src = subprocess.run(['git', 'show', '%s:%s.py' % (PARENT, modname)], cwd=ROOT,
                         capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory(prefix='seedmask_') as d:
        path = os.path.join(d, '_parent_%s.py' % modname)
        with open(path, 'wb') as fh:
            fh.write(src)
        spec = importlib.util.spec_from_file_location('_parent_%s' % modname, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def generate():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)    # noqa: F841
    import validate_seed_mask as V
    old = load_parent('pdb_logprob')
    oldj = load_parent('pdb_joint_logprob')
    x, y, e = V.load_data()
    out = {
        'what': 'mask=None parallel MCMC lnprob from the parent commit code '
                '(validate_seed_mask.py check (j)); regenerate with '
                'tests/fixtures/make_seed_mask_parent_ref.py',
        'parent_commit': PARENT,
        'data': 'Data/i2_tim3_1000_test.txt', 'pdb': 'Data/1ubq.pdb',
        'window': [V.XMIN, V.XMAX], 'fit_scale': V.SCALE,
        'varying': ['theta', 'phi', 'protins', 'cov'],
        'joint_setup': 'datasets y*1.0 and y*1.3; theta, phi, protins shared (bare); cov local',
    }

    # single model, mask off: the PdbLogProb data the minimizer ships
    fit = V.make_fit(x, y, e, None)
    obj0, _, iargs0 = V.capture_single_logprob(fit)
    vn = list(obj0.var_names)
    thetas = V.theta_vectors(fit.emcee_params, vn)
    model, xs, ys, es = iargs0[:4]
    ref = old.PdbLogProb(vn, obj0.bounds, obj0.fit_scale, obj0.is_weighted,
                         has_lnsigma=obj0.has_lnsigma, model=model, x=xs, y=ys, err=es)
    out['single'] = {'var_names': vn, 'thetas': [list(map(float, t)) for t in thetas],
                     'lnprob': [float(ref(t)) for t in thetas]}

    # joint no-cov, mask off: the JointLogProb data _attach_joint_parallel ships
    jf = V.make_joint(x, y, e, [None, None])
    jobj0, _, jargs0 = V.capture_joint_logprob(jf)
    vn = list(jobj0.var_names)
    thetas = V.theta_vectors(jf.emcee_params, vn)
    models, datas = jargs0
    ref = oldj.JointLogProb(vn, jobj0.bounds, jobj0.fit_scale, jobj0.is_weighted,
                            jobj0.param_map, jobj0.priors, jf.emcee_params.copy(),
                            has_lnsigma=jobj0.has_lnsigma,
                            models=models, datas=[tuple(dd[:3]) for dd in datas])
    out['joint'] = {'var_names': vn, 'thetas': [list(map(float, t)) for t in thetas],
                           'lnprob': [float(ref(t)) for t in thetas]}

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
        fh.write('\n')
    print('wrote', OUT)
    print('single', out['single']['lnprob'])
    print('joint ', out['joint']['lnprob'])


if __name__ == '__main__':
    generate()
