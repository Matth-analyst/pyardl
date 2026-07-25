r"""Response-surface critical values and approximate p-values.

Provides asymptotic critical values of the F statistic at any
significance level, plus approximate p-values at both bounds, following
the response-surface approach of Kripfganz & Schneider (2020).

The underlying material is the one shipped with ``statsmodels``, which
re-simulated the null distributions at very large scale (32 million
replications per configuration) and fitted its own p-value polynomials.
p-values are evaluated as

    p = 1 - Phi(c0 + c1*x + c2*x^2 [+ c3*x^3]),   x = log(F)

and critical values at arbitrary levels are obtained by numerically
inverting that strictly decreasing function. At the directly simulated
percentiles the quantile estimates are returned as they are.

Coverage: the F statistic, ``k = 1..10``, asymptotic. Anything outside
that raises an explicit error pointing to a source that does cover it.

References
----------
Kripfganz, S. & Schneider, D. C. (2020). "Response Surface Regressions
for Critical Value Bounds and Approximate p-values in Equilibrium
Correction Models", *Oxford Bulletin of Economics and Statistics*,
82(6), 1456-1481.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from statsmodels.tsa.ardl import pss_critical_values as _sm_pss

__all__ = ["crit_value_bounds", "pvalue_bounds", "MAX_K_KS"]

MAX_K_KS = 10
_PERCENTILE_ALPHAS = {0.10: 0, 0.05: 1, 0.01: 2, 0.001: 3}


def _check_coverage(stat: str, case: int, k: int) -> None:
    if stat != "F":
        raise ValueError(
            "Only the F statistic is covered here; use cv_source='pss' for t bounds."
        )
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be between 1 and 5, got {case}.")
    if not 1 <= k <= MAX_K_KS:
        raise ValueError(
            f"k={k} is outside the covered range (k = 1..{MAX_K_KS}); for "
            "k=0 use cv_source='pss' or the simulation engine."
        )


def _pvalue_one(stat: float, key: tuple[int, int, bool]) -> float:
    """Approximate p-value for one bound."""
    if stat <= 0:
        return 1.0
    x = np.log(stat)
    coefs = (
        _sm_pss.large_p[key] if stat <= _sm_pss.stat_star[key] else _sm_pss.small_p[key]
    )
    y = sum(c * x**i for i, c in enumerate(coefs))
    return float(1 - norm.cdf(y))


def pvalue_bounds(
    f_stat: float,
    case: int,
    k: int,
) -> tuple[float, float]:
    """Approximate p-values of the F statistic at both bounds.

    Parameters
    ----------
    f_stat : float
        Observed F statistic.
    case : int
        Deterministic case, 1 to 5.
    k : int
        Number of level regressors, 1 to 10.

    Returns
    -------
    tuple of float
        ``(p_i0, p_i1)``: the p-value if all regressors were I(0), and if
        all were I(1), with ``p_i0 <= p_i1``. Read them as follows:
        ``p_i1 <= alpha`` means cointegration, ``p_i0 > alpha`` means no
        rejection, and anything in between is the inconclusive zone.

    Examples
    --------
    >>> p_i0, p_i1 = pvalue_bounds(6.0, case=3, k=1)
    >>> p_i0 < 0.05 < p_i1  # rejected at both bounds
    False
    >>> round(p_i0, 3) < round(p_i1, 3) < 0.05
    True
    """
    _check_coverage("F", case, k)
    p_i0 = _pvalue_one(f_stat, (k, case, False))
    p_i1 = _pvalue_one(f_stat, (k, case, True))
    return p_i0, p_i1


def crit_value_bounds(
    case: int,
    k: int,
    alpha: float,
) -> tuple[float, float]:
    """Asymptotic F bounds at an arbitrary significance level.

    At the directly simulated levels (10%, 5%, 1%, 0.1%) the quantile
    estimates are returned as they are; at any other level the p-value
    function is inverted numerically.

    Examples
    --------
    >>> lo, up = crit_value_bounds(case=3, k=1, alpha=0.05)
    >>> round(lo, 2), round(up, 2)  # PSS published: (4.94, 5.73)
    (4.92, 5.72)
    """
    _check_coverage("F", case, k)
    if not 0.0005 < alpha < 0.25:
        raise ValueError(
            f"alpha={alpha} is outside the reliable range of the surfaces "
            "(0.0005 < alpha < 0.25)."
        )

    if alpha in _PERCENTILE_ALPHAS:
        idx = _PERCENTILE_ALPHAS[alpha]
        return (
            float(_sm_pss.crit_vals[(k, case, False)][idx]),
            float(_sm_pss.crit_vals[(k, case, True)][idx]),
        )

    out = []
    for i1 in (False, True):
        key = (k, case, i1)

        def objective(s: float, key: tuple[int, int, bool] = key) -> float:
            return _pvalue_one(s, key) - alpha

        out.append(float(brentq(objective, 0.05, 100.0)))
    return out[0], out[1]
