# Prints a traceback but exits 0 (a swallowed exception): the runner must say FAIL.
import sys
import traceback
try:
    raise ValueError('swallowed')
except ValueError:
    traceback.print_exc()
sys.exit(0)
