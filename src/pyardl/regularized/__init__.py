"""Regularized ARDL order selection (LASSO / Elastic Net, spec 41).

An alternative to :meth:`pyardl.core.ardl.ARDL.select_order` for a large
number of candidate regressors, where the exhaustive/sequential grid
search becomes combinatorially expensive — not a replacement.
"""

from pyardl.regularized.model import RegularizedOrderResults, select_order_regularized

__all__ = ["RegularizedOrderResults", "select_order_regularized"]
