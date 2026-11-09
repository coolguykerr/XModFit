# A traceback, then exit 77: a skip must not hide it. The runner must say FAIL (review F3).
import sys
import traceback
try:
    raise ValueError('swallowed before the skip')
except ValueError:
    traceback.print_exc()
sys.exit(77)
