"""Self-test for tests/harness.py. Itself a harness: uses a local counter, not tests.harness."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_SCRIPT = os.path.join(ROOT, 'tests', 'selftest', 'harness_cases.py')

CASES = {            # case -> (expected exit code, substring expected in stdout+stderr)
    'all_pass':   (0,  'ALL PASS (2 checks)'),
    'one_fail':   (1,  'FAILURES PRESENT (1 of 2)'),
    'no_checks':  (1,  'NO CHECKS RAN'),
    'skipped':    (77, 'SKIP - no data'),
    'raise_mode': (1,  'AssertionError'),     # run with XMF_CHECK_RAISE=1
    'close_fail': (1,  'FAIL'),
    # audit N1: a Qt-slot exception is a failed check even though the process survives
    'slot_exception':           (1, 'FAILURES PRESENT (1 of 2)'),
    'slot_exception_late_hook': (1, 'FAILURES PRESENT (1 of 3)'),
}

n_fail = 0
for case, (want_code, want_text) in CASES.items():
    env = dict(os.environ)
    env.pop('XMF_CHECK_RAISE', None)
    if case == 'raise_mode':
        env['XMF_CHECK_RAISE'] = '1'
    p = subprocess.run([sys.executable, CASES_SCRIPT, case], cwd=ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    out = p.stdout + p.stderr
    ok = p.returncode == want_code and want_text in out
    print('%s - %s | exit %s (want %s), %r %s' % (
        'PASS' if ok else 'FAIL', case, p.returncode, want_code, want_text,
        'found' if want_text in out else 'MISSING'))
    if not ok:
        n_fail += 1
        print(out)

sys.exit(1 if n_fail else 0)
