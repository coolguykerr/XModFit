"""The ``.xmodes`` sidecar: persisting a mode set with the provenance that dates it.

A sidecar rather than an extension of ``.xpar`` for two reasons. ``loadParameters``
locates its sections by counting lines, so any inserted block breaks it; and a mode
set is keyed to (dataset x template x window x mesh), which is a different object
from "the current parameters".

Grammar follows ``joint_fit_io``: a magic first line, ``#key=value`` headers, and a
block-accumulating parse rather than a line-offset scan. Text, so it diffs and greps
like every other XModFit parameter file.

Two deliberate choices:

* ``u`` (the unit-cube point) is authoritative and the physical ``#params`` table is
  a derived convenience. The box is in the header, so physical values are
  recomputable; storing it the other way round would make every reader need the box.
* the **power** block is written into the file. The humility statement is a property
  of the run, not of whichever app version opens it later, so it travels with the
  data.

``#data_sha256`` is of the ACTIVE ``(q, R, err)`` triple, which is the only exactly
right staleness test: it catches a changed file, a changed x window, a changed column
limit and a changed error column, while being insensitive to path and comments.
"""
import io as _io
import os
import time

import numpy as np

from . import modes as M

MAGIC = '#XModFit Mode Set v1'


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return '1' if v else '0'
    if isinstance(v, (int, np.integer)):
        return '%d' % v
    if isinstance(v, float) or isinstance(v, np.floating):
        return '%.10g' % v
    return str(v)


def format_xmodes(ms):
    """Serialise a :class:`~fastscan.modes.ModeSet` to text."""
    p = ms.prov[0]
    L = [MAGIC, '#saved=%s' % time.asctime()]
    for k in ('dataset', 'objective', 'fit_scale', 'n_active', 'data_sha256',
              'num_boxes', 'spacez', 'box_ref_length', 'vol_calc', 'rho_top',
              'rho_bottom', 'electron_scale', 'algorithm', 'n1', 'n1_ok', 'k',
              'sobol_seed', 'band_stratified', 'iter1', 'tol1', 'iter2', 'tol2',
              'n_cpu_limited', 'seed_source', 'n_seeds', 'seeds_dropped',
              'hessian', 'scipy_version', 'xpr_fingerprint'):
        L.append('#%s=%s' % (k, _fmt(getattr(p, k))))
    L.append('#x_window=%s:%s' % (_fmt(p.x_window[0]), _fmt(p.x_window[1])))
    L.append('#cancelled=%s' % _fmt(ms.cancelled))
    L.append('#dedup=position-only gc_deg=%s d_protins=%s d_cov=%s'
             % (_fmt(ms.dedup_rule.gc_deg), _fmt(ms.dedup_rule.d_protins),
                _fmt(ms.dedup_rule.d_cov)))
    for prov in ms.prov:
        L.append('#template\t%s\t%s\t%s'
                 % (prov.template_label, prov.template_path,
                    _fmt(prov.electron_scale)))
    w = p.power
    L.append('#power=ref_budget=%d found=%d ref=%d false_determined=%s '
             'span_found=%s span_ref=%s'
             % (w.ref_budget, w.found_median, w.ref_median,
                _fmt(w.false_determined_rate), _fmt(w.span_found_deg),
                _fmt(w.span_ref_deg)))
    L.append('#power_scale=band=%s:%s extremes=%s:%s calibrated=%s'
             % (_fmt(w.scale_band[0]), _fmt(w.scale_band[1]),
                _fmt(w.scale_extremes[0]), _fmt(w.scale_extremes[1]),
                _fmt(w.budget_is_calibrated)))
    for k, v in sorted((p.timings or {}).items()):
        L.append('#timing_%s=%s' % (k, _fmt(v)))
    for warn in ms.warnings:
        L.append('#warning=%s' % warn.replace('\n', ' '))

    lo = np.asarray(p.param_lower, dtype=float) if p.param_lower is not None else None
    hi = np.asarray(p.param_upper, dtype=float) if p.param_upper is not None else None
    if lo is not None:
        L.append('#box:')
        L.append('#column\tlo\thi\tvarying')
        for c in range(len(lo)):
            L.append('%d\t%.10g\t%.10g\t%d' % (c, lo[c], hi[c],
                                               1 if c in (p.free or ()) else 0))
    L.append('#varying=%s' % ' '.join(p.varying_names))

    for i, m in enumerate(ms.modes):
        L.append('#mode=%d\ttemplate=%s' % (i + 1, m.template_label))
        L.append('#neglogL=%.10f\tdelta=%.10f\tchi2=%.10f' % (m.neglogL, m.delta,
                                                              m.chi2))
        L.append('#class=%s\tsource=%s\teigmin=%s\tcond=%s\tfloored=%d'
                 % (m.hess_class, m.hess_source, _fmt(m.eigmin), _fmt(m.cond),
                    m.n_floored))
        L.append('#starts=%d\tstage1_rank=%d\tstart_index=%d\tlaplace_wt=%s'
                 % (m.n_starts, m.stage1_rank, m.start_index, _fmt(m.laplace_wt)))
        L.append('#flags=%s' % ','.join(m.flags))
        L.append('#bounds_active=%s' % ','.join(m.active_bounds))
        L.append('#u=%s' % ' '.join('%.17g' % v for v in np.asarray(m.u).ravel()))
        L.append('#params:')
        for k in sorted(m.params):
            L.append('%s\t%.10g' % (k, m.params[k]))
        if m.R is not None:
            L.append('#R=%s' % ' '.join('%.10g' % v
                                        for v in np.asarray(m.R).ravel()))
    return '\n'.join(L) + '\n'


def parse_xmodes(text):
    """Parse ``.xmodes`` text into a :class:`~fastscan.modes.ModeSet`."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith(MAGIC):
        raise ValueError('not an XModFit mode set (missing the %r magic line)'
                         % MAGIC)
    hdr, templates, warns, box, mode_blocks = {}, [], [], [], []
    cur, section = None, None
    for raw in lines[1:]:
        line = raw.rstrip('\n')
        if not line.strip():
            continue
        if line.startswith('#mode='):
            cur = {'hdr': {}, 'params': {}}
            mode_blocks.append(cur)
            section = None
            for part in line[1:].split('\t'):
                if '=' in part:
                    k, _, v = part.partition('=')
                    cur['hdr'][k.strip()] = v.strip()
            continue
        if line.startswith('#params:'):
            section = 'params'
            continue
        if line.startswith('#box:'):
            section = 'box'
            continue
        if line.startswith('#column'):
            continue
        if line.startswith('#template\t'):
            templates.append(line.split('\t')[1:])
            continue
        if line.startswith('#warning='):
            warns.append(line[len('#warning='):])
            continue
        if line.startswith('#'):
            body = line[1:]
            if cur is not None:
                for part in body.split('\t'):
                    if '=' in part:
                        k, _, v = part.partition('=')
                        cur['hdr'][k.strip()] = v.strip()
            elif '=' in body:
                k, _, v = body.partition('=')
                hdr[k.strip()] = v.strip()
            continue
        # a data row
        if section == 'params' and cur is not None:
            k, _, v = line.partition('\t')
            try:
                cur['params'][k.strip()] = float(v)
            except ValueError:
                pass
        elif section == 'box':
            parts = line.split('\t')
            if len(parts) >= 3:
                box.append((float(parts[1]), float(parts[2]),
                            int(parts[3]) if len(parts) > 3 else 0))

    def g(key, cast=str, default=None):
        if key not in hdr:
            return default
        try:
            return cast(hdr[key])
        except (TypeError, ValueError):
            return default

    def gb(key, default=False):
        return bool(int(hdr.get(key, '1' if default else '0') or 0))

    xw = (hdr.get('x_window') or 'nan:nan').split(':')
    prov = M.Provenance(
        algorithm=g('algorithm', str, 'twostage-v1'),
        n1=g('n1', int, 0), n1_ok=g('n1_ok', int, 0), k=g('k', int, 0),
        sobol_seed=g('sobol_seed', int, 0), band_stratified=gb('band_stratified', True),
        iter1=g('iter1', int, 30), tol1=g('tol1', float, 1e-3),
        iter2=g('iter2', int, 300), tol2=g('tol2', float, 1e-8),
        n_cpu_limited=g('n_cpu_limited', int, 0),
        seed_source=g('seed_source', str, ''), n_seeds=g('n_seeds', int, 0),
        seeds_dropped=g('seeds_dropped', int, 0),
        template_label=(templates[0][0] if templates else ''),
        template_path=(templates[0][1] if templates and len(templates[0]) > 1 else ''),
        num_boxes=g('num_boxes', int, 0), spacez=g('spacez', float, 0.5),
        box_ref_length=g('box_ref_length', float, 30.0),
        vol_calc=g('vol_calc', str, 'approx'),
        rho_top=g('rho_top', float, 0.0), rho_bottom=g('rho_bottom', float, 0.334),
        electron_scale=g('electron_scale', float, 1.0),
        objective=g('objective', str, 'xpr_logL'),
        fit_scale=g('fit_scale', str, 'Linear'), dataset=g('dataset', str, ''),
        x_window=(float(xw[0]), float(xw[-1])),
        n_active=g('n_active', int, 0), data_sha256=g('data_sha256', str, ''),
        varying_names=tuple((hdr.get('varying') or '').split()),
        free=tuple(i for i, b in enumerate(box) if b[2]),
        param_lower=(np.array([b[0] for b in box]) if box else None),
        param_upper=(np.array([b[1] for b in box]) if box else None),
        hessian=g('hessian', str, 'none'),
        scipy_version=g('scipy_version', str, ''),
        xpr_fingerprint=g('xpr_fingerprint', str, ''),
        timings={k[len('timing_'):]: float(v) for k, v in hdr.items()
                 if k.startswith('timing_')},
    )

    built = []
    for blk in mode_blocks:
        h = blk['hdr']

        def hf(key, cast=float, default=float('nan')):
            v = h.get(key)
            if v in (None, '', 'None'):
                return default
            try:
                return cast(v)
            except (TypeError, ValueError):
                return default

        u = np.array([float(x) for x in (h.get('u') or '').split()], dtype=float)
        params = blk['params']
        Rv = h.get('R')
        m = M.Mode(
            u=u, neglogL=hf('neglogL'), params=params,
            theta=params.get('theta', float('nan')),
            phi=params.get('phi', float('nan')),
            protins=params.get('protins', float('nan')),
            cov=params.get('cov', float('nan')),
            delta=hf('delta', float, 0.0), chi2=hf('chi2'),
            n_active=prov.n_active,
            hess_class=h.get('class', M.UNCERTIFIED),
            hess_source=h.get('source', 'none'),
            eigmin=(None if h.get('eigmin') in (None, 'None', '') else hf('eigmin')),
            cond=(None if h.get('cond') in (None, 'None', '') else hf('cond')),
            n_floored=int(hf('floored', float, 0)),
            n_starts=int(hf('starts', float, 1)),
            stage1_rank=int(hf('stage1_rank', float, -1)),
            start_index=int(hf('start_index', float, -1)),
            laplace_wt=(None if h.get('laplace_wt') in (None, 'None', '')
                        else hf('laplace_wt')),
            flags=tuple(x for x in (h.get('flags') or '').split(',') if x),
            active_bounds=tuple(x for x in (h.get('bounds_active') or '').split(',')
                                if x),
            template_label=h.get('template', ''),
            R=(np.array([float(x) for x in Rv.split()]) if Rv else None),
        )
        built.append(m)

    ms = M.ModeSet(built, prov, cancelled=gb('cancelled'), warnings=warns)
    # The Hessians are deliberately NOT stored: they are nvary^2 floats that are
    # only meaningful against this exact data and mesh, and are re-derivable in a
    # fraction of a second at the stored point. So a reloaded set reports its
    # stored classification but carries no curvature, and weights are not
    # recomputable until the caller asks for curvature again.
    return ms


def save(path, ms):
    with _io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(format_xmodes(ms))
    return path


def load(path):
    if not os.path.isfile(path):
        raise ValueError('no such mode set: %s' % path)
    with _io.open(path, 'r', encoding='utf-8', errors='replace') as fh:
        return parse_xmodes(fh.read())


def default_path(dataset_path):
    """``<dataset>.xmodes`` beside the data, which is where .xpar files go too."""
    base = os.path.splitext(str(dataset_path))[0]
    return base + '.xmodes'


def export_mode_as_xpar(mode, model, path, category='XRR'):
    """Write ONE mode as a normal ``.xpar`` the stock ``loadParameters`` can read.

    This is how a mode gets promoted to "the answer" outside this tool, and it
    reuses ``joint_fit_io.dump_model`` so there is no second definition of the
    ``.par`` block grammar. The model is deep-copied, so the live one is untouched.
    """
    import copy

    from joint_fit_io import dump_model

    mc = copy.deepcopy(model)
    for k, v in mode.params.items():
        if k in mc.params:
            mc.params[k].set(value=float(v))
        if k in mc.__dict__:
            mc.__dict__[k] = float(v)
    mc.update_parameters()
    block = dump_model(mc)
    hdr = ['#File saved on %s' % time.asctime(),
           '#Category: %s' % category,
           '#Function: %s' % type(mc).__name__,
           '#Fast global search mode: neglogL=%.6f delta=%.6f class=%s'
           % (mode.neglogL, mode.delta, mode.hess_class)]
    with _io.open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(hdr) + '\n' + block.rstrip('\n') + '\n')
    return path
