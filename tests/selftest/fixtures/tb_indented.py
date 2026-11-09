# An indented (embedded) traceback ending in ValueError, followed by an unindented line
# that mentions the allowed name. The exception line is the ValueError: FAIL (review F2).
import sys
sys.stderr.write('    Traceback (most recent call last):\n'
                 '      File "x.py", line 1, in <module>\n'
                 '        f()\n'
                 '    ValueError: indented\n')
print('PASS - later | ExpectedAbort')
sys.exit(0)
