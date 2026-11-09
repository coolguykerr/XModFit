# A module-qualified allowed exception type: PASS, with the allowed count in the note.
import sys
sys.stderr.write('Traceback (most recent call last):\n'
                 '  File "x.py", line 1, in <module>\n'
                 '    f()\n'
                 'pkg.mod.ExpectedAbort: on purpose\n')
sys.exit(0)
