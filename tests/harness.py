"""Shared check helper for the XModJointFit test harnesses.

Usage (from a harness, after sys.path.insert(0, ROOT)):
    from tests.harness import check, check_close, skip, finish, counts
Exit-code convention: 0 pass, 1 fail, 77 skip.
"""
import os
import sys

_n_checks = 0
_n_fail = 0


def check(name, cond, detail=''):
    """Print PASS/FAIL, count it, return bool(cond).

    With XMF_CHECK_RAISE=1 a failing check raises AssertionError(name).
    """
    global _n_checks, _n_fail
    ok = bool(cond)
    _n_checks += 1
    print('%s - %s | %s' % ('PASS' if ok else 'FAIL', name, detail))
    if not ok:
        _n_fail += 1
        if os.environ.get('XMF_CHECK_RAISE') == '1':
            raise AssertionError(name)
    return ok


def check_close(name, got, want, tol):
    """abs(got-want) <= tol; prints in validate_xpr.py's format, then counts via check()."""
    ok = abs(got - want) <= tol
    print('%-42s got %13.4f  want %13.4f  %s'
          % (name, got, want, 'OK' if ok else 'FAIL'))
    return check(name, ok, 'got %.4f want %.4f tol %g' % (got, want, tol))


def skip(reason):
    """Print SKIP and exit 77."""
    print('SKIP - %s' % reason)
    sys.exit(77)


def counts():
    """(n_checks, n_failures)."""
    return _n_checks, _n_fail


def finish():
    """Print the summary and return the exit code (0 pass, 1 fail or no checks)."""
    if _n_checks == 0:
        print('NO CHECKS RAN')
        return 1
    if _n_fail:
        print('FAILURES PRESENT (%d of %d)' % (_n_fail, _n_checks))
        return 1
    print('ALL PASS (%d checks)' % _n_checks)
    return 0
