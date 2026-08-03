r"""M tests of Ng & Perron (2001).

Four statistics computed on the GLS-detrended series, sharing one
long-run variance estimate. Their purpose is not extra power for its own
sake but **size**: the classical ADF and Phillips-Perron tests reject far
too often when the errors carry a large negative moving-average
component, a configuration common in macroeconomic data.

Ng & Perron trace that failure to two causes and fix both: the long-run
variance must be estimated autoregressively rather than by a kernel, and
the lag order must be chosen by a modified criterion (see
:func:`pyardl.unitroot.select_lags`). Using their statistics with a
kernel variance, or with the plain AIC, gives back the distortion they
were built to remove.

The four statistics
-------------------
With :math:`\tilde y` the detrended series and :math:`s^2_{AR}` the
autoregressive long-run variance:

.. math::

    MZ_\alpha &= \frac{T^{-1}\tilde y_T^2 - s^2_{AR}}
                      {2 T^{-2} \sum_{t=1}^{T-1} \tilde y_t^2} \\
    MSB &= \left( \frac{T^{-2} \sum \tilde y_t^2}{s^2_{AR}}
          \right)^{1/2} \\
    MZ_t &= MZ_\alpha \times MSB

:math:`MZ_t` has the same limiting distribution as the DF-GLS statistic,
which gives a free internal cross-check on the critical values.

:math:`MPT` is the point-optimal statistic evaluated at the same local
alternative :math:`\bar c`.

All four reject for **small** values, though by different routes:
:math:`MZ_\alpha` and :math:`MZ_t` are large and negative under
stationarity, while :math:`MSB` and :math:`MPT` are positive and shrink
towards zero. The rule is therefore uniform — reject when the statistic
falls below its bound — and every critical value is a lower-tail
quantile.

References
----------
.. [1] Ng, S. & Perron, P. (2001). Lag length selection and the
       construction of unit root tests with good size and power.
       *Econometrica*, 69(6), 1519-1554.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pyardl.unitroot.gls import (
    CBAR,
    LagMethod,
    Trend,
    adf_regression,
    gls_detrend,
    ols_detrend,
    select_lags,
)

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

__all__ = ["ng_perron", "NgPerronResults", "M_STATISTICS"]

#: The four statistics, in reporting order. All are lower-tail: the null
#: of a unit root is rejected when the statistic falls below its bound.
M_STATISTICS: tuple[str, ...] = ("MZa", "MZt", "MSB", "MPT")


@dataclass(frozen=True)
class NgPerronResults:
    """Outcome of the four M tests.

    Attributes
    ----------
    stats : dict
        ``MZa``, ``MZt``, ``MSB``, ``MPT``.
    critical_values : dict
        Statistic name, then level, then bound.
    lags, nobs, trend, lag_method : ...
        Settings actually used.
    s2_ar : float
        The autoregressive long-run variance shared by the four
        statistics. Reported because every one of them is only as good
        as this estimate.
    lag_criterion : dict
        Criterion value at each candidate lag order.
    """

    stats: dict[str, float]
    critical_values: dict[str, dict[float, float]]
    lags: int
    nobs: int
    trend: Trend
    lag_method: str
    s2_ar: float
    lag_criterion: dict[int, float] = field(default_factory=dict, repr=False)

    def decision(self, statistic: str = "MZt", alpha: float = 0.05) -> str:
        """``'stationary'`` or ``'unit_root'`` for one statistic.

        Parameters
        ----------
        statistic : {'MZa', 'MZt', 'MSB', 'MPT'}, default 'MZt'
        alpha : float, default 0.05

        Notes
        -----
        All four statistics are lower-tail, so the comparison is the same
        for each: reject the unit root when the statistic falls below its
        bound.
        """
        if statistic not in M_STATISTICS:
            raise ValueError(f"statistic={statistic!r} is not one of {M_STATISTICS}.")
        value = self.stats[statistic]
        bound = self.critical_values[statistic][alpha]
        return "stationary" if value < bound else "unit_root"

    def summary(self) -> str:
        """Readable report of the four statistics and their verdicts."""
        lines = [
            f"Ng-Perron M tests (2001) - trend '{self.trend}', "
            f"lags={self.lags} ({self.lag_method}), nobs={self.nobs}",
            f"  long-run variance (autoregressive): {self.s2_ar:.4f}",
            "",
            f"  {'statistic':<10}{'value':>12}{'5% bound':>12}  decision (5%)",
        ]
        for name in M_STATISTICS:
            lines.append(
                f"  {name:<10}{self.stats[name]:>12.4f}"
                f"{self.critical_values[name][0.05]:>12.4f}  "
                f"{self.decision(name, 0.05)}"
            )
        lines.append("")
        lines.append("  H0: the series has a unit root (reject when below)")
        return "\n".join(lines)


def ng_perron(
    y: ArrayLike,
    trend: Trend = "c",
    lags: int | None = None,
    method: LagMethod = "maic",
    max_lags: int | None = None,
) -> NgPerronResults:
    r"""Run the four M tests of Ng & Perron (2001).

    Parameters
    ----------
    y : array_like
        Series to test, shape ``(T,)``.
    trend : {'c', 'ct'}, default 'c'
        Deterministic terms.
    lags : int, optional
        Fixed lag order for the long-run variance. Chosen by ``method``
        when omitted.
    method : {'maic', 'mbic', 'aic', 'bic', 't-stat'}, default 'maic'
        Lag-selection criterion. Leave it at ``'maic'`` unless you have
        a specific reason: it is the modified criterion that makes these
        statistics behave, and the plain AIC undoes the point of the
        test.
    max_lags : int, optional
        Upper bound of the search. Defaults to the Schwert rule.

    Returns
    -------
    NgPerronResults

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.unitroot import ng_perron
    >>> rng = np.random.default_rng(0)
    >>> res = ng_perron(np.cumsum(rng.standard_normal(200)))
    >>> res.decision("MZt")
    'unit_root'
    """
    from pyardl.critical_values.ngperron2001 import m_critical_values

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    detrended = gls_detrend(y_arr, trend)
    n = detrended.size

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
    stats = _m_statistics(detrended, fit.s2_ar, n, trend)

    return NgPerronResults(
        stats=stats,
        # Series length, as for DF-GLS: that is the axis the
        # table was simulated on.
        critical_values=m_critical_values(n, trend),
        lags=chosen,
        nobs=fit.nobs,
        trend=trend,
        lag_method=lag_method,
        s2_ar=fit.s2_ar,
        lag_criterion=criterion,
    )


def _m_statistics(
    detrended: FloatArray, s2_ar: float, n: int, trend: Trend
) -> dict[str, float]:
    """The four statistics from the detrended series and ``s2_ar``."""
    lagged = detrended[:-1]
    sum_sq = float(lagged @ lagged) / n**2
    last_sq = float(detrended[-1] ** 2) / n

    mza = (last_sq - s2_ar) / (2.0 * sum_sq)
    msb = float(np.sqrt(sum_sq / s2_ar))
    mzt = mza * msb

    cbar = CBAR[trend]
    if trend == "c":
        mpt = (cbar**2 * sum_sq - cbar * last_sq) / s2_ar
    else:
        mpt = (cbar**2 * sum_sq + (1.0 - cbar) * last_sq) / s2_ar

    return {"MZa": float(mza), "MZt": float(mzt), "MSB": float(msb), "MPT": float(mpt)}
