import os
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3600)'])
with open(os.environ['XMF_CHILD_PID_FILE'], 'w') as f:
    f.write(str(child.pid))
time.sleep(3600)
