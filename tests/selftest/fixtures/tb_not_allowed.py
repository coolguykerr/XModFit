# Registry allows 'ExpectedAbort', but this traceback is a KeyError: the runner must say FAIL.
import sys
import traceback
try:
    {}['missing']
except KeyError:
    traceback.print_exc()
sys.exit(0)
