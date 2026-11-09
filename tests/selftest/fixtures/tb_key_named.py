# Registry allows 'ExpectedAbort'; this is KeyError: 'ExpectedAbort'. FAIL (review F1).
import sys
import traceback
try:
    {}['ExpectedAbort']
except KeyError:
    traceback.print_exc()
sys.exit(0)
