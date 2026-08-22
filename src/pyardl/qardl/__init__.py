"""Quantile ARDL: a long-run relation that can differ across the distribution.

The framework of Cho, Kim & Shin (2015). Every parameter — the
adjustment speed included — becomes a function of the quantile, so the
relation may hold in the tails and not in the middle, or adjust faster
below the median than above it.
"""

from pyardl.qardl.estimate import (
    check_loss,
    quantile_regression,
    quantile_regression_lp,
)

__all__ = ["check_loss", "quantile_regression", "quantile_regression_lp"]
