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
        ('fixed', 'KNOWN-FAIL NOW PASSES'), ('smoke', 'PASS')]
for name, st in want:
    got = rows.get(name, (None, ''))[0]
    check('status of %s' % name, got == st, 'got %r want %r' % (got, st))
check('smoke note says smoke', 'smoke' in rows.get('smoke', ('', ''))[1])
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
sys.exit(finish())
