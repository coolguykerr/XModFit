# Registry allows 'ExpectedAbort', but the exception TYPE is TypeError; only its message
# mentions the allowed name. The runner must say FAIL (review F1).
import sys
import traceback
try:
    raise TypeError("cannot pickle 'ExpectedAbort' object")
except TypeError:
    traceback.print_exc()
sys.exit(0)
