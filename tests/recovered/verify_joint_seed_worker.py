import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np
from lmfit import Parameters
from scipy.optimize import least_squares
PI = np.pi

def shape(theta, phi, x):
    tr = theta*PI/180; pr = phi*PI/180
    return 1.0 + 0.3*np.cos(tr)*np.cos(2*PI*x) + 0.3*np.sin(tr)*np.cos(pr)*np.sin(2*PI*x)

class SynPDB:
    def __init__(self):
        self.params = Parameters()
        self.params.add('theta', value=50.0, min=-np.inf, max=np.inf, vary=True)
        self.params.add('phi',   value=90.0, min=-np.inf, max=np.inf, vary=True)
        self.params.add('protins', value=8.0, min=-np.inf, max=np.inf, vary=True)  # SHARED nuisance
        self.params.add('scale', value=1.0, min=-np.inf, max=np.inf, vary=True)    # per-dataset
        self.x = np.linspace(0.02, 0.5, 50); self.output_params = {}
    def init_params(self): pass
    def y(self):
        p = self.params
        return p['scale'].value * (shape(p['theta'].value, p['phi'].value, self.x)
                                   + 0.02 * p['protins'].value * np.cos(6*PI*self.x))

def main():
    from Joint_Fit_Routines import JointFit
    import pdb_joint_seed_worker as PJS
    jf = JointFit(); rng = np.random.RandomState(0); x = np.linspace(0.02, 0.5, 50)
    for i, s in enumerate((1.0, 1.5)):
        m = SynPDB(); m.x = x
        y = s * (shape(55, 110, x) + 0.02*8.5*np.cos(6*PI*x)) + 0.01*rng.randn(len(x))
        jf.add_dataset('D%d' % i, func=m, x=x, y=y, yerr=0.01*np.ones_like(x))
    groups = [{'members': [(0, 'theta'), (1, 'theta')], 'bare': True, 'name': None},
              {'members': [(0, 'phi'),   (1, 'phi')],   'bare': True, 'name': None},
              {'members': [(0, 'protins'),(1, 'protins')], 'bare': True, 'name': None}]
    jf.build_master_params(groups=groups)
    jf.perform_fit(0.02, 0.5, fit_method='leastsq')
    print('shared master varying:', [k for k in jf.master_params if jf.master_params[k].vary])

    var_names = [k for k in jf.result.params if jf.result.params[k].vary]
    prof = [k for k in var_names if k not in ('theta', 'phi')]
    models = [o['func'] for o in jf.objectives]
    datas = [(o['x'][o['imin']:o['imax']+1], o['y'][o['imin']:o['imax']+1],
              o['yerr'][o['imin']:o['imax']+1]) for o in jf.objectives]

    def ref_cost(theta, phi):
        p = jf.result.params.copy()
        def resid(fvars):
            for n, v in zip(prof, fvars): p[n].value = v
            p['theta'].value = theta; p['phi'].value = phi
            p.update_constraints()
            return jf.residual(p, 'Linear')
        start = [jf.result.params[n].value for n in prof]
        return float(least_squares(resid, start).cost)

    scan = PJS.JointSeedScan(var_names, jf.param_map, 'Linear', jf.result.params.copy(), {})
    cells = [(0, 0, 40.0, 60.0), (1, 0, 70.0, 130.0), (2, 0, 110.0, 250.0), (3, 0, 20.0, 300.0)]
    serial = PJS.map_cell_costs(cells, scan, models, datas, workers=1)
    par = PJS.map_cell_costs(cells, PJS.JointSeedScan(var_names, jf.param_map, 'Linear',
                             jf.result.params.copy(), {}), models, datas, workers=2)
    print('  theta  phi     ref_cost      worker       parallel')
    for (c, (ti, pi, cw), (tj, pj, cp)) in zip(cells, serial, par):
        cr = ref_cost(c[2], c[3])
        print('  %5.0f %5.0f  %12.6e %12.6e %12.6e' % (c[2], c[3], cr, cw, cp))
        assert abs(cr - cw) < 1e-8 * max(1, abs(cr)), (cr, cw)
        assert abs(cw - cp) < 1e-9, (cw, cp)
    print('JOINT SEED WORKER == joint residual (serial==parallel) OK')

if __name__ == '__main__':
    main()
