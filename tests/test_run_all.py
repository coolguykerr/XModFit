"""Self-test of tests/run_all.py against tests/selftest/fixtures."""
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from tests.harness import check, finish  # noqa: E402

pid_file = os.path.join(tempfile.gettempdir(), 'xmf_child_pid_%d.txt' % os.getpid())
if os.path.exists(pid_file):
    os.remove(pid_file)
env = dict(os.environ, XMF_CHILD_PID_FILE=pid_file)
t0 = time.time()
p = subprocess.run(
    [sys.executable, 'tests/run_all.py', '--dir', 'tests/selftest/fixtures',
     '--timeout', '5', '--registry', 'tests/selftest/registry.json'],
    cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    universal_newlines=True, encoding='utf-8')
wall = time.time() - t0
print(p.stdout)

# Table rows: "<status> <secs> <interp> tests/selftest/fixtures/<name>.py <note>".
# Status may contain spaces, so take everything before the seconds column.
rows = {}
for line in p.stdout.splitlines():
    m = re.match(r'^(\S.*?)\s+(\d+\.\d)\s+\S+\s+tests/selftest/fixtures/(\w+)\.py\s*(.*)$',
                 line)
    if m:
        rows[m.group(3)] = (m.group(1), m.group(4))

# The table is sorted by path; the brief's order is checked by name.
want = [('ok', 'PASS'), ('bad', 'FAIL'), ('skip', 'SKIP'), ('hang', 'TIMEOUT'),
        ('hang_child', 'TIMEOUT'), ('unicode', 'PASS'),
        ('fixed', 'KNOWN-FAIL NOW PASSES'), ('smoke', 'PASS'),
        # audit N1: exit 0 with a traceback in the log is FAIL unless the registry allows it
        ('tb_exit0', 'FAIL'), ('tb_allowed', 'PASS'), ('tb_not_allowed', 'FAIL'),
        ('slot_exc', 'FAIL'),
        # review F1-F3: the allowance is the exception TYPE; indented tracebacks; skips
        ('tb_msg_mentions', 'FAIL'), ('tb_key_named', 'FAIL'), ('tb_indented', 'FAIL'),
        ('tb_qualified', 'PASS'), ('tb_skip', 'FAIL')]
for name, st in want:
    got = rows.get(name, (None, ''))[0]
    check('status of %s' % name, got == st, 'got %r want %r' % (got, st))
check('smoke note says smoke', 'smoke' in rows.get('smoke', ('', ''))[1])
for name in ('tb_exit0', 'tb_not_allowed', 'slot_exc', 'tb_msg_mentions', 'tb_key_named',
             'tb_indented', 'tb_skip'):
    check('%s note explains the traceback' % name,
          'traceback in log' in rows.get(name, ('', ''))[1], rows.get(name, ('', ''))[1])
check('slot_exc note names the slot exception',
      'AssertionError' in rows.get('slot_exc', ('', ''))[1])
check('tb_indented note names the indented ValueError, not the later line',
      'ValueError: indented' in rows.get('tb_indented', ('', ''))[1],
      rows.get('tb_indented', ('', ''))[1])
for name in ('tb_allowed', 'tb_qualified'):
    check('%s note counts the allowed traceback' % name,
          '1 allowed traceback' in rows.get(name, ('', ''))[1], rows.get(name, ('', ''))[1])
check('runner exit code is 1', p.returncode == 1, 'rc %s' % p.returncode)

alive = None
if os.path.exists(pid_file):
    with open(pid_file) as f:
        pid = f.read().strip()
    out = subprocess.run(['tasklist', '/FI', 'PID eq %s' % pid],
                         stdout=subprocess.PIPE, universal_newlines=True).stdout
    alive = re.search(r'\b%s\b' % re.escape(pid), out) is not None
    os.remove(pid_file)
check('hang_child grandchild was killed', alive is False, 'alive=%s' % alive)
check('wall time under 60 s', wall < 60, '%.1f s' % wall)

# audit N3: --only that matches nothing must not look like a green run
p2 = subprocess.run(
    [sys.executable, 'tests/run_all.py', '--dir', 'tests/selftest/fixtures',
     '--registry', 'tests/selftest/registry.json', '--only', 'no_such_harness_xyz'],
    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    universal_newlines=True, encoding='utf-8')
print(p2.stdout)
check('--only with no match exits 2', p2.returncode == 2, 'rc %s' % p2.returncode)
check('--only with no match says so', 'no harness matched' in p2.stdout)
sys.exit(finish())
