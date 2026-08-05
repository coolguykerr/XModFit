"""
joint_fit_io.py -- serialization for XModJointFit joint fits (``.jpar``) and the
model-instance param dump/apply used by both ``.jpar`` and "load .par into joint".

Pure logic (no Qt, no host imports) so it is unit-testable head-less. The host
(xmodfit.py) does the file dialogs, dataset<->path matching, and model-class
construction; this module only knows the grammar and how to read/write a model
instance's parameters.

A ``.jpar`` is a superset of the stock ``.par`` grammar: global fit settings, then
one ``.par``-style block per dataset (keyed by the dataset's FILE PATH, since the
in-session ``"<num><>path"`` key's number prefix is volatile), then a shared-
parameter section.

Grammar
-------
    #XModFit Joint Fit v1
    #Fit Range=<xmin:xmax>
    #Fit Method=<method>
    #Fit Scale=<scale>
    #Fit Iterations=<n>
    #Dataset: <path>
    #Category: <category>
    #Function: <funcname>
    #Xrange=<x expression>
    #Cov: <cov_path>            (optional)
    #Raw: <raw_path>            (optional)
    #Fixed Parameters:
    #param\tvalue
    <key>\t<value>
    ...
    #Single fitting parameters:
    #param\tvalue\tfit\tmin\tmax\texpr\tbrute_step
    <parname>\t<value>\t<fit>\t<min>\t<max>\t<expr>\t<brute_step>
    ...
    #Multiple fitting parameters:
    #param\tvalue\tfit\tmin\tmax\texpr\tbrute_step
    <__mkey_pkey_NNN>\t...              (7 fields = numeric, 2 fields = label)
    ...
    #Dataset: <path2>
    ...
    #Shared:
    #group\tbare=<0|1>\tname=<master_name or ->
    <path>\t<local_key>
    <path>\t<local_key>
    #group\t...
"""

from __future__ import annotations

# keys on a model's __dict__ that are NOT user "fixed parameters"
_SPECIAL_KEYS = ('x', 'params', 'choices', 'output_params', '__mpar__', 'filepaths')

_MAGIC = '#XModFit Joint Fit v1'


# --------------------------------------------------------------------------- #
#  read a model instance -> .par section lines
# --------------------------------------------------------------------------- #
def dump_model(model):
    """Return ``(fixed_lines, sfit_lines, mfit_lines)`` for a model INSTANCE.

    Mirrors xmodfit.saveParameters but reads the model directly (``model.params``
    = lmfit fit params, ``model.__dict__`` = scalars + ``__mpar__``) instead of the
    GUI tables, so it works head-less and per dataset."""
    d = model.__dict__
    fit_params = getattr(model, 'params', None)
    fp_keys = set(fit_params.keys()) if fit_params is not None else set()

    fixed = []
    for key in d.keys():
        if key in fp_keys or key in _SPECIAL_KEYS or key.startswith('_'):
            continue
        fixed.append('%s\t%s' % (key, d[key]))

    sfit = []
    if fit_params is not None:
        for key in fit_params.keys():
            if key.startswith('__'):
                continue
            p = fit_params[key]
            sfit.append('%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
                key, p.value, int(bool(p.vary)), p.min, p.max, p.expr, p.brute_step))

    mfit = []
    mpar = d.get('__mpar__', {}) or {}
    for mkey in mpar.keys():
        for pkey in mpar[mkey].keys():
            for i in range(len(mpar[mkey][pkey])):
                parname = '__%s_%s_%03d' % (mkey, pkey, i)
                if fit_params is not None and parname in fit_params:
                    p = fit_params[parname]
                    mfit.append('%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
                        parname, p.value, int(bool(p.vary)), p.min, p.max, p.expr, p.brute_step))
                else:
                    mfit.append('%s\t%s' % (parname, mpar[mkey][pkey][i]))
    return fixed, sfit, mfit


# --------------------------------------------------------------------------- #
#  apply parsed section lines -> a model instance
# --------------------------------------------------------------------------- #
def apply_block(model, fixed_lines, sfit_lines, mfit_lines):
    """Apply parsed ``.par`` section lines into a model INSTANCE.

    Mirrors xmodfit.loadParameters (lines applied to ``self.fit``), but targets the
    given ``model`` (``model.__dict__`` for scalars/``__mpar__``, ``model.params``
    for lmfit fit params). Tolerates namespaced mkeys via ``rsplit('_', 2)`` and
    grows ``__mpar__`` arrays via ``insert`` when the saved model had more layers."""
    d = model.__dict__
    fit_params = model.params

    # fixed
    for line in fixed_lines:
        key, val = line.split('\t', 1)
        try:
            d[key] = eval(val)
        except Exception:
            d[key] = val

    # single fitting params
    for line in sfit_lines:
        parname, parval, parfit, parmin, parmax, parexpr, parbrute = line.split('\t')
        d[parname] = float(parval)
        fit_params[parname].set(value=float(parval), vary=int(parfit),
                                min=float(parmin), max=float(parmax))
        try:
            fit_params[parname].set(expr=eval(parexpr))
        except Exception:
            fit_params[parname].set(expr=None if parexpr in ('None', 'none', '') else str(parexpr))
        try:
            fit_params[parname].set(brute_step=eval(parbrute))
        except Exception:
            pass

    # multiple fitting params (numeric rows first with expr=None, then re-apply expr)
    mpar = d.get('__mpar__', {})
    exprs = {}
    for line in mfit_lines:
        tlist = line.split('\t')
        if len(tlist) > 2:
            parname, parval, parfit, parmin, parmax, parexpr, parbrute = tlist
            try:
                fit_params[parname].set(value=float(parval), vary=int(parfit),
                                        min=float(parmin), max=float(parmax),
                                        expr=None)
            except Exception:
                fit_params.add(parname, value=float(parval), vary=int(parfit),
                               min=float(parmin), max=float(parmax), expr=None)
            try:
                fit_params[parname].set(brute_step=eval(parbrute))
            except Exception:
                pass
            try:
                exprs[parname] = eval(parexpr)
            except Exception:
                exprs[parname] = None if parexpr in ('None', 'none', '') else str(parexpr)
            mkey, pkey, num = parname[2:].rsplit('_', 2)
            num = int(num)
            try:
                mpar[mkey][pkey][num] = float(parval)
            except Exception:
                mpar[mkey][pkey].insert(num, float(parval))
        else:
            parname, parval = tlist
            mkey, pkey, num = parname[2:].rsplit('_', 2)
            num = int(num)
            try:
                mpar[mkey][pkey][num] = parval
            except Exception:
                mpar[mkey][pkey].insert(num, parval)
    for parname, e in exprs.items():
        try:
            fit_params[parname].set(expr=e)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#  format a .jpar
# --------------------------------------------------------------------------- #
def format_jpar(settings, blocks, shares):
    """Assemble a ``.jpar`` string.

    settings : dict with 'fit_range','fit_method','fit_scale','fit_iterations'
    blocks   : list of dicts, each:
        {'path','category','func','xrange','cov'(opt),'raw'(opt),
         'fixed':[..],'sfit':[..],'mfit':[..]}
    shares   : list of dicts {'bare':bool,'name':str|None,
                              'endpoints':[(path, local_key), ...]}
    """
    out = [_MAGIC,
           '#Fit Range=%s' % settings.get('fit_range', ''),
           '#Fit Method=%s' % settings.get('fit_method', ''),
           '#Fit Scale=%s' % settings.get('fit_scale', ''),
           '#Fit Iterations=%s' % settings.get('fit_iterations', '')]
    for b in blocks:
        out.append('#Dataset: %s' % b['path'])
        out.append('#Category: %s' % b.get('category', ''))
        out.append('#Function: %s' % b.get('func', ''))
        out.append('#Xrange=%s' % b.get('xrange', ''))
        if b.get('cov'):
            out.append('#Cov: %s' % b['cov'])
        if b.get('raw'):
            out.append('#Raw: %s' % b['raw'])
        out.append('#Fixed Parameters:')
        out.append('#param\tvalue')
        out.extend(b.get('fixed', []))
        out.append('#Single fitting parameters:')
        out.append('#param\tvalue\tfit\tmin\tmax\texpr\tbrute_step')
        out.extend(b.get('sfit', []))
        out.append('#Multiple fitting parameters:')
        out.append('#param\tvalue\tfit\tmin\tmax\texpr\tbrute_step')
        out.extend(b.get('mfit', []))
    out.append('#Shared:')
    for g in shares:
        out.append('#group\tbare=%d\tname=%s' % (1 if g.get('bare') else 0,
                                                  g.get('name') or '-'))
        for (path, lkey) in g.get('endpoints', []):
            out.append('%s\t%s' % (path, lkey))
    return '\n'.join(out) + '\n'


# --------------------------------------------------------------------------- #
#  parse a .jpar
# --------------------------------------------------------------------------- #
def parse_jpar(text):
    """Parse a ``.jpar`` string into
    ``{'settings':{...}, 'blocks':[{...}], 'shares':[{...}]}``.

    Each block: path, category, func, xrange, cov, raw, and the three parsed
    section line-lists 'fixed'/'sfit'/'mfit' (data lines only, comments stripped).
    Each share: {'bare':bool,'name':str|None,'endpoints':[(path,local_key),...]}."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith('#XModFit Joint Fit'):
        raise ValueError('not a .jpar file (missing magic header)')

    settings = {}
    blocks = []
    shares = []
    cur = None            # current block dict
    section = None        # 'fixed'|'sfit'|'mfit' within a block
    in_shares = False
    cur_group = None

    def _finish_group():
        nonlocal cur_group
        if cur_group is not None:
            shares.append(cur_group)
            cur_group = None

    for raw in lines[1:]:
        line = raw.rstrip('\n')
        if line == '':
            continue
        if line.startswith('#Fit Range='):
            settings['fit_range'] = line.split('=', 1)[1]; continue
        if line.startswith('#Fit Method='):
            settings['fit_method'] = line.split('=', 1)[1]; continue
        if line.startswith('#Fit Scale='):
            settings['fit_scale'] = line.split('=', 1)[1]; continue
        if line.startswith('#Fit Iterations='):
            settings['fit_iterations'] = line.split('=', 1)[1]; continue
        if line.startswith('#Dataset:'):
            cur = {'path': line.split(':', 1)[1].strip(), 'category': '', 'func': '',
                   'xrange': '', 'cov': None, 'raw': None,
                   'fixed': [], 'sfit': [], 'mfit': []}
            blocks.append(cur); section = None; in_shares = False
            continue
        if line.startswith('#Category:'):
            if cur is not None: cur['category'] = line.split(':', 1)[1].strip()
            continue
        if line.startswith('#Function:'):
            if cur is not None: cur['func'] = line.split(':', 1)[1].strip()
            continue
        if line.startswith('#Xrange='):
            if cur is not None: cur['xrange'] = line.split('=', 1)[1]
            continue
        if line.startswith('#Cov:'):
            if cur is not None: cur['cov'] = line.split(':', 1)[1].strip()
            continue
        if line.startswith('#Raw:'):
            if cur is not None: cur['raw'] = line.split(':', 1)[1].strip()
            continue
        if line.startswith('#Fixed Parameters:'):
            section = 'fixed'; continue
        if line.startswith('#Single fitting parameters:'):
            section = 'sfit'; continue
        if line.startswith('#Multiple fitting parameters:'):
            section = 'mfit'; continue
        if line.startswith('#Shared:'):
            in_shares = True; section = None; continue
        if in_shares and line.startswith('#group'):
            _finish_group()
            bare = 'bare=1' in line
            name = None
            for tok in line.split('\t'):
                if tok.startswith('name='):
                    nm = tok.split('=', 1)[1]
                    name = None if nm in ('-', '', 'None') else nm
            cur_group = {'bare': bare, 'name': name, 'endpoints': []}
            continue
        if line.startswith('#'):
            continue  # a column-header comment (#param ...) or unknown comment
        # data line
        if in_shares:
            if cur_group is not None and '\t' in line:
                path, lkey = line.split('\t', 1)
                cur_group['endpoints'].append((path, lkey))
        elif cur is not None and section is not None:
            cur[section].append(line)
    _finish_group()
    return {'settings': settings, 'blocks': blocks, 'shares': shares}


def parse_par(text):
    """Parse a STOCK XModFit ``.par`` into
    ``{category, func, xrange, settings, fixed, sfit, mfit}`` (data lines only).

    Used by the host's "load .par into joint" to apply one file's values to every
    dataset already in the joint fit."""
    out = {'category': '', 'func': '', 'xrange': '', 'settings': {},
           'fixed': [], 'sfit': [], 'mfit': []}
    section = None
    for raw in text.splitlines():
        line = raw.rstrip('\n')
        if line.startswith('#Category:'):
            out['category'] = line.split(':', 1)[1].strip(); continue
        if line.startswith('#Function:'):
            out['func'] = line.split(':', 1)[1].strip(); continue
        if line.startswith('#Xrange='):
            out['xrange'] = line.split('=', 1)[1]; continue
        if line.startswith('#Fit Range='):
            out['settings']['fit_range'] = line.split('=', 1)[1]; continue
        if line.startswith('#Fit Method='):
            out['settings']['fit_method'] = line.split('=', 1)[1]; continue
        if line.startswith('#Fit Scale='):
            out['settings']['fit_scale'] = line.split('=', 1)[1]; continue
        if line.startswith('#Fit Iterations='):
            out['settings']['fit_iterations'] = line.split('=', 1)[1]; continue
        if line.startswith('#Fixed Parameters:'):
            section = 'fixed'; continue
        if line.startswith('#Single fitting parameters:'):
            section = 'sfit'; continue
        if line.startswith('#Multiple fitting parameters:'):
            section = 'mfit'; continue
        if line.startswith('#') or line == '':
            continue
        if section:
            out[section].append(line)
    return out
