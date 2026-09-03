import numpy as np

from lmfit import customautocorr
from emcee.state import State
from emcee.backends import Backend

__all__ = ["CustomBackend"]

class CustomBackend(Backend):
    def __init__(self, dtype=None):
        super().__init__()

    def get_autocorr_time(self, angleindex=[], discard=0, thin=1, **kwargs):
        """Compute an estimate of the autocorrelation time for each parameter

        Args:
            thin (Optional[int]): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Other arguments are passed directly to
        :func:`emcee.autocorr.integrated_time`.

        Returns:
            array[ndim]: The integrated autocorrelation time estimate for the
                chain for each parameter.

        """
        # NB: the base emcee Backend.get_autocorr_time is a pure function
        # (get_chain -> integrated_time, no side effects). Calling super() here read
        # the FULL chain (discard=0) and computed a standard autocorrelation whose
        # result was thrown away -- so every call autocorrelated twice, and the
        # discarded one grew with chain length. Dropped; the real (angle-aware,
        # windowed-by-discard) estimate below is what we return.
        x = self.get_chain(discard=discard, thin=thin)
        return thin * customautocorr.integrated_time(x, angleindex, **kwargs)