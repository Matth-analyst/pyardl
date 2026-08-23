"""Quantile ARDL: a long-run relation that can differ across the distribution.

The framework of Cho, Kim & Shin (2015). Every parameter — the
adjustment speed included — becomes a function of the quantile, so the
relation may hold in the tails and not in the middle, or adjust faster
below the median than above it. A mean regression averages all of that
into one number and reports it as *the* relationship.
"""

from pyardl.qardl.estimate import (
    check_loss,
    quantile_regression,
    quantile_regression_lp,
)
from pyardl.qardl.model import QARDL, QARDLResults

__all__ = [
    "QARDL",
    "QARDLResults",
    "check_loss",
    "quantile_regression",
    "quantile_regression_lp",
]
