"""Shared check helper for the XModJointFit test harnesses.

Usage (from a harness, after sys.path.insert(0, ROOT)):
    from tests.harness import check, check_close, skip, finish, counts
Exit-code convention: 0 pass, 1 fail, 77 skip.

Importing this module installs a counting sys.excepthook. PyQt5 hands an exception raised
inside a Qt slot to sys.excepthook and carries on, so without this the harness would exit
0. Each such exception counts as one failed check ("uncaught exception"). pyqtgraph only
replaces sys.excepthook when it is still the default, so a hook installed before xmodfit
is imported survives; check() and finish() re-install it (chaining to whatever replaced
it) in case a harness imports this module late or something else overrides the hook.
tests/run_all.py also fails any exit-0 row whose log contains a traceback.
"""
import os
import sys

_n_checks = 0
_n_fail = 0


def _counting_excepthook(etype, value, tb):
    global _n_checks, _n_fail
    if _counting_excepthook.active:      # a hook we chained to wrapped us: do not loop
        sys.__excepthook__(etype, value, tb)
        return
    # A failed check raised under XMF_CHECK_RAISE=1 was already counted by check().
    if not getattr(value, '_xmf_counted', False):
        _n_checks += 1
        _n_fail += 1
        print('FAIL - uncaught exception | %s: %s'
              % (getattr(etype, '__name__', etype), value), flush=True)
    nxt = _counting_excepthook.chained
    _counting_excepthook.active = True
    try:
        (nxt if nxt is not None else sys.__excepthook__)(etype, value, tb)
    finally:
        _counting_excepthook.active = False


_counting_excepthook.chained = None
_counting_excepthook.active = False


def _ensure_excepthook():
    """Install the counting hook unless it is already the active one."""
    cur = sys.excepthook
    if cur is _counting_excepthook:
        return
    _counting_excepthook.chained = None if cur is sys.__excepthook__ else cur
    sys.excepthook = _counting_excepthook


_ensure_excepthook()


def check(name, cond, detail=''):
    """Print PASS/FAIL, count it, return bool(cond).

    With XMF_CHECK_RAISE=1 a failing check raises AssertionError(name).
    """
    global _n_checks, _n_fail
    _ensure_excepthook()
    ok = bool(cond)
    _n_checks += 1
    print('%s - %s | %s' % ('PASS' if ok else 'FAIL', name, detail))
    if not ok:
        _n_fail += 1
        if os.environ.get('XMF_CHECK_RAISE') == '1':
            err = AssertionError(name)
            err._xmf_counted = True          # the excepthook must not count it again
            raise err
    return ok


def check_close(name, got, want, tol):
    """abs(got-want) <= tol; prints in validate_xpr.py's format, then counts via check()."""
    ok = abs(got - want) <= tol
    print('%-42s got %13.4f  want %13.4f  %s'
          % (name, got, want, 'OK' if ok else 'FAIL'))
    return check(name, ok, 'got %.4f want %.4f tol %g' % (got, want, tol))


def skip(reason):
    """Print SKIP and exit 77 -- unless a check already failed or an uncaught
    (e.g. Qt-slot) exception was counted: then the skip must not hide it, and
    this exits 1."""
    if _n_fail:
        print('SKIP - %s' % reason)
        print('FAILURES PRESENT (%d of %d) before the skip: exiting as a failure'
              % (_n_fail, _n_checks))
        sys.exit(1)
    print('SKIP - %s' % reason)
    sys.exit(77)


def counts():
    """(n_checks, n_failures)."""
    return _n_checks, _n_fail


def finish():
    """Print the summary and return the exit code (0 pass, 1 fail or no checks)."""
    _ensure_excepthook()
    if _n_checks == 0:
        print('NO CHECKS RAN')
        return 1
    if _n_fail:
        print('FAILURES PRESENT (%d of %d)' % (_n_fail, _n_checks))
        return 1
    print('ALL PASS (%d checks)' % _n_checks)
    return 0
