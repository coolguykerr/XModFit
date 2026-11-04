"""Single subprocess runner over every XModJointFit test harness.

    python tests/run_all.py [--quick] [--only SUBSTR] [--timeout SECONDS]
                            [--dir DIR ...] [--registry PATH]

Each harness runs in its own process (own interpreter, cwd=ROOT), its full output goes
to tests/_logs/<path with / as __>.log, and a table of status / seconds / interpreter /
note is printed. A timed-out harness has its whole process tree killed.

Statuses: PASS, FAIL, SKIP (exit 77 or --quick), TIMEOUT, KNOWN-FAIL,
KNOWN-FAIL NOW PASSES. Exit code 1 if any FAIL or TIMEOUT, else 0.

Registry entry keys (all optional): interp ('cmc-py39' | 'cmcars'), external (str),
slow (bool), known (str: expected to fail), smoke (bool: assertion-free script),
timeout (int seconds).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'tests', '_logs')

INTERPS = {
    'cmc-py39': 'C:/Users/kerr/.conda/envs/cmc-py39/python.exe',
    'cmcars': 'C:/Users/kerr/AppData/Local/anaconda3/envs/cmcars/python.exe',
}

REGISTRY = {
    'validate_xpr.py': {
        'external': 'XPR-main/test_data, sibling of the repo',
        'interp': 'cmcars', 'slow': True},
    # validate_fastscan.py Part B needs the Julia/XPR backend (see its header).
    'validate_fastscan.py': {'slow': True, 'interp': 'cmcars'},
}

DEFAULT_TIMEOUT = 600


def discover(dirs):
    if dirs:
        pats = [os.path.join(d, '*.py') for d in dirs]
    else:
        pats = [os.path.join(ROOT, 'validate_*.py'),
                os.path.join(ROOT, 'tests', 'recovered', '*.py'),
                os.path.join(ROOT, 'tests', 'test_*.py')]
    found = set()
    for p in pats:
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        for f in glob.glob(p):
            found.add(os.path.relpath(f, ROOT).replace(os.sep, '/'))
    return sorted(found)


def run_one(rel, interp, timeout):
    """Run one harness. Returns (status_core, seconds) with status_core in
    PASS/FAIL/SKIP/TIMEOUT."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, rel.replace('/', '__') + '.log')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONIOENCODING='utf-8')
    t0 = time.time()
    with open(log, 'wb') as fh:
        proc = subprocess.Popen(
            [INTERPS[interp], os.path.join(ROOT, rel)], cwd=ROOT, env=env,
            stdout=fh, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
            return 'TIMEOUT', time.time() - t0
    secs = time.time() - t0
    return ('PASS' if rc == 0 else 'SKIP' if rc == 77 else 'FAIL'), secs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--quick', action='store_true',
                    help='skip slow entries and entries with an external requirement')
    ap.add_argument('--only', metavar='SUBSTR')
    ap.add_argument('--timeout', type=int, default=None,
                    help='per-harness timeout in seconds (default %d; overrides '
                         'registry timeouts)' % DEFAULT_TIMEOUT)
    ap.add_argument('--dir', action='append', metavar='DIR')
    ap.add_argument('--registry', metavar='PATH')
    args = ap.parse_args(argv)

    registry = REGISTRY
    if args.registry:
        with open(args.registry, encoding='utf-8') as f:
            registry = json.load(f)

    rows = []
    n_bad = 0
    now_pass = []
    for rel in discover(args.dir):
        if args.only and args.only not in rel:
            continue
        ent = registry.get(rel, {})
        interp = ent.get('interp', 'cmc-py39')
        notes = []
        if ent.get('smoke'):
            notes.append('smoke')
        if ent.get('external'):
            notes.append('external: ' + ent['external'])
        if ent.get('slow'):
            notes.append('slow')
        if args.quick and (ent.get('slow') or ent.get('external')):
            rows.append((rel, 'SKIP', 0.0, interp, 'quick; ' + '; '.join(notes)))
            continue
        timeout = args.timeout or ent.get('timeout') or DEFAULT_TIMEOUT
        print('running %s ...' % rel, flush=True)
        status, secs = run_one(rel, interp, timeout)
        if ent.get('known'):
            notes.append('known: ' + ent['known'])
            if status == 'PASS':
                status = 'KNOWN-FAIL NOW PASSES'
                now_pass.append(rel)
            elif status in ('FAIL', 'TIMEOUT'):
                status = 'KNOWN-FAIL'
        if status in ('FAIL', 'TIMEOUT'):
            n_bad += 1
        rows.append((rel, status, secs, interp, '; '.join(notes)))

    print()
    print('%-22s %8s  %-8s  %-50s %s' % ('status', 'seconds', 'interp', 'harness', 'note'))
    for rel, status, secs, interp, note in rows:
        print('%-22s %8.1f  %-8s  %-50s %s' % (status, secs, interp, rel, note))
    counts = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    print()
    print('summary: ' + ', '.join('%d %s' % (v, k) for k, v in sorted(counts.items())))
    for rel in now_pass:
        print('NOTE: %s is registered known-failing but now passes; remove its '
              'registry entry.' % rel)
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
