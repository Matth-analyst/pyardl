r"""DF-GLS test of Elliott, Rothenberg & Stock (1996).

The augmented Dickey-Fuller test run on GLS-detrended data. Removing the
deterministic part under a local alternative rather than under the null
recovers most of the power the classical ADF loses, at no cost in size.
The gain is largest exactly where it matters: against roots close to,
but below, one.

References
----------
.. [1] Elliott, G., Rothenberg, T. J. & Stock, J. H. (1996). Efficient
       tests for an autoregressive unit root. *Econometrica*, 64(4),
       813-836.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pyardl.unitroot.gls import (
    LagMethod,
    Trend,
    adf_regression,
    gls_detrend,
    ols_detrend,
    select_lags,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike

__all__ = ["dfgls", "DFGLSResults"]


@dataclass(frozen=True)
class DFGLSResults:
    """Outcome of a DF-GLS test.

    Attributes
    ----------
    stat : float
        The t-ratio on the lagged level of the detrended series.
    lags : int
        Lag order used.
    nobs : int
        Observations in the ADF regression.
    trend : str
        ``'c'`` or ``'ct'``.
    lag_method : str
        How ``lags`` was chosen.
    critical_values : dict
        Level to critical value. The test is **left-tailed**: reject when
        ``stat`` falls below the bound.
    lag_criterion : dict
        Criterion value at each candidate lag order.
    """

    stat: float
    lags: int
    nobs: int
    trend: Trend
    lag_method: str
    critical_values: dict[float, float]
    lag_criterion: dict[int, float] = field(default_factory=dict, repr=False)

    @property
    def null_hypothesis(self) -> str:
        return "the series has a unit root"

    def decision(self, alpha: float = 0.05) -> str:
        """``'stationary'`` or ``'unit_root'`` at level ``alpha``.

        Raises
        ------
        KeyError
            If no critical value is tabulated at ``alpha``.
        """
        return "stationary" if self.stat < self.critical_values[alpha] else "unit_root"

    def summary(self) -> str:
        """Readable one-block report."""
        cv = "  ".join(
            f"{int(a * 100)}%: {v:.4f}" for a, v in sorted(self.critical_values.items())
        )
        return (
            f"DF-GLS (Elliott, Rothenberg & Stock 1996) - trend '{self.trend}', "
            f"lags={self.lags} ({self.lag_method}), nobs={self.nobs}\n"
            f"  statistic = {self.stat:.4f}   decision (5%): "
            f"{self.decision(0.05)}\n"
            f"  critical values (left tail)   {cv}\n"
            f"  H0: {self.null_hypothesis}"
        )


def dfgls(
    y: ArrayLike,
    trend: Trend = "c",
    lags: int | None = None,
    method: LagMethod = "maic",
    max_lags: int | None = None,
) -> DFGLSResults:
    r"""Test for a unit root on GLS-detrended data.

    Parameters
    ----------
    y : array_like
        Series to test, shape ``(T,)``.
    trend : {'c', 'ct'}, default 'c'
        Deterministic terms. Use ``'ct'`` when the series clearly
        trends: testing a trending series under ``'c'`` will almost
        never reject, whatever the truth.
    lags : int, optional
        Fixed lag order. When omitted, chosen by ``method``.
    method : {'maic', 'mbic', 'aic', 'bic', 't-stat'}, default 'maic'
        Lag-selection criterion, ignored when ``lags`` is given.
    max_lags : int, optional
        Upper bound of the search. Defaults to the Schwert rule.

    Returns
    -------
    DFGLSResults

    Notes
    -----
    The null is a unit root and the test is **left-tailed**: a large
    negative statistic is evidence *against* the unit root. Failing to
    reject is not evidence of a unit root — it is absence of evidence,
    which is why the sequential level-then-difference protocol of
    :func:`pyardl.unitroot.report` exists.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.unitroot import dfgls
    >>> rng = np.random.default_rng(0)
    >>> res = dfgls(np.cumsum(rng.standard_normal(200)))
    >>> res.decision(0.05)
    'unit_root'
    """
    from pyardl.critical_values.ers1996 import dfgls_critical_values

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    detrended = gls_detrend(y_arr, trend)

    if lags is None:
        # Lag order chosen on the OLS-detrended series; see
        # select_lags for why, and for the measurement behind it.
        chosen, criterion = select_lags(
            ols_detrend(y_arr, trend), method=method, max_lags=max_lags
        )
        lag_method = method
    else:
        if lags < 0:
            raise ValueError(f"lags={lags} must be non-negative.")
        chosen, criterion, lag_method = lags, {}, "fixed"

    fit = adf_regression(detrended, chosen)
    return DFGLSResults(
        stat=fit.tstat,
        lags=chosen,
        nobs=fit.nobs,
        trend=trend,
        lag_method=lag_method,
        # Table indexed on the series length, the axis it was
        # simulated on — not on the ADF regression rows.
        critical_values=dfgls_critical_values(y_arr.size, trend),
        lag_criterion=criterion,
    )
