"""Driver for tests/test_harness.py: exercises tests.harness in one named way per case."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tests.harness import check, check_close, skip, finish  # noqa: E402


def all_pass():
    check('a', True, 'first')
    check('b', 1 == 1)


def one_fail():
    check('a', True)
    check('b', False, 'deliberate')


def no_checks():
    pass


def skipped():
    skip('no data')


def raise_mode():
    check('boom', False, 'raises under XMF_CHECK_RAISE=1')


def close_fail():
    check_close('close', 1.0, 1.5, 0.1)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
    sys.exit(finish())
