# Prints an expected traceback; registry allows 'ExpectedAbort': the runner must say PASS.
import sys
import traceback


class ExpectedAbort(Exception):
    pass


try:
    raise ExpectedAbort('on purpose')
except ExpectedAbort:
    traceback.print_exc()
sys.exit(0)
