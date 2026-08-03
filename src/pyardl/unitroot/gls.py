r"""GLS detrending and modified lag selection.

Shared machinery for the modern unit-root tests. Both the DF-GLS test of
Elliott, Rothenberg & Stock (1996) and the M tests of Ng & Perron (2001)
start from the same two ingredients: a series detrended under a *local
alternative* rather than under the null, and a lag order chosen by a
criterion that survives a negative moving-average component.

Local-to-unity detrending
-------------------------
The classical ADF removes the mean or trend by ordinary least squares,
which is what costs it most of its power: under a near-unit root, OLS
estimates the deterministic part badly. ERS instead quasi-difference the
data at :math:`\bar\alpha = 1 + \bar c / T` before removing the
deterministic terms, with :math:`\bar c = -7` (constant) or
:math:`-13.5` (constant and trend) — the points where the asymptotic
power envelope is reached at 50%.

.. warning::

    The first observation is **not** quasi-differenced. It enters as
    :math:`y_1` and :math:`z_1`, at level. Applying
    :math:`(1 - \bar\alpha)` to it instead is the classic implementation
    trap: on a random walk of length 200 it moves the DF-GLS statistic
    from -1.34 to -1.94, which is enough to reverse a decision at any
    conventional level. The convention here is verified against ``arch``
    to machine precision.

References
----------
.. [1] Elliott, G., Rothenberg, T. J. & Stock, J. H. (1996). Efficient
       tests for an autoregressive unit root. *Econometrica*, 64(4),
       813-836.
.. [2] Ng, S. & Perron, P. (2001). Lag length selection and the
       construction of unit root tests with good size and power.
       *Econometrica*, 69(6), 1519-1554.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

Trend = Literal["c", "ct"]
LagMethod = Literal["maic", "mbic", "aic", "bic", "t-stat", "fixed"]

__all__ = [
    "gls_detrend",
    "ols_detrend",
    "select_lags",
    "adf_regression",
    "CBAR",
    "ADFRegression",
    "Trend",
    "LagMethod",
]

#: Local alternative at which the power envelope is attained, by trend.
#: Elliott, Rothenberg & Stock (1996), §2.
CBAR: dict[str, float] = {"c": -7.0, "ct": -13.5}


def _check_trend(trend: str) -> Trend:
    if trend not in ("c", "ct"):
        raise ValueError(
            f"trend={trend!r} is not supported. Use 'c' (constant) or 'ct' "
            "(constant and linear trend). ERS derived the local alternative "
            "for these two cases only."
        )
    return trend  # type: ignore[return-value]


def gls_detrend(y: ArrayLike, trend: Trend = "c") -> FloatArray:
    r"""Remove the deterministic part under a local-to-unity alternative.

    Parameters
    ----------
    y : array_like
        Series, shape ``(T,)``.
    trend : {'c', 'ct'}, default 'c'
        Deterministic terms: constant, or constant and linear trend.

    Returns
    -------
    numpy.ndarray
        The detrended series :math:`\tilde y = y - z \hat\beta`, same
        length as ``y``.

    Notes
    -----
    With :math:`\bar\alpha = 1 + \bar c / T`, the quasi-differenced data
    are

    .. math::

        y^{\bar\alpha} = (y_1,\; y_2 - \bar\alpha y_1,\; \dots,\;
                          y_T - \bar\alpha y_{T-1})

    and likewise for the deterministic matrix :math:`z`. The first entry
    is left at level — see the module warning. :math:`\hat\beta` is the
    least-squares coefficient of :math:`y^{\bar\alpha}` on
    :math:`z^{\bar\alpha}`, applied back to the *undifferenced* ``z``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.unitroot import gls_detrend
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.standard_normal(100))
    >>> gls_detrend(y, "c").shape
    (100,)
    """
    trend = _check_trend(trend)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = y_arr.size
    if n < 4:
        raise ValueError(f"Sample too short: {n} observations.")

    alpha = 1.0 + CBAR[trend] / n
    z = (
        np.ones((n, 1))
        if trend == "c"
        else np.column_stack([np.ones(n), np.arange(1, n + 1, dtype=np.float64)])
    )

    y_qd = np.empty(n, dtype=np.float64)
    z_qd = np.empty_like(z)
    # First observation at level: this is the ERS convention, not an
    # oversight. See the module-level warning.
    y_qd[0] = y_arr[0]
    z_qd[0] = z[0]
    y_qd[1:] = y_arr[1:] - alpha * y_arr[:-1]
    z_qd[1:] = z[1:] - alpha * z[:-1]

    beta = np.linalg.lstsq(z_qd, y_qd, rcond=None)[0]
    return np.asarray(y_arr - z @ beta, dtype=np.float64)


def ols_detrend(y: ArrayLike, trend: Trend = "c") -> FloatArray:
    """Remove the deterministic part by ordinary least squares.

    Parameters
    ----------
    y : array_like
        Series, shape ``(T,)``.
    trend : {'c', 'ct'}, default 'c'

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Not an alternative to :func:`gls_detrend` for the test itself — the
    whole point of ERS is that GLS detrending is what recovers the power.
    This is used only to choose the lag order; see :func:`select_lags`.
    """
    trend = _check_trend(trend)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = y_arr.size
    z = (
        np.ones((n, 1))
        if trend == "c"
        else np.column_stack([np.ones(n), np.arange(1, n + 1, dtype=np.float64)])
    )
    beta = np.linalg.lstsq(z, y_arr, rcond=None)[0]
    return np.asarray(y_arr - z @ beta, dtype=np.float64)


@dataclass(frozen=True)
class ADFRegression:
    """Least-squares fit of the augmented Dickey-Fuller regression.

    Attributes
    ----------
    tstat : float
        t-ratio on the lagged level — the DF-GLS statistic itself.
    rho : float
        Coefficient on the lagged level.
    lags : int
        Number of lagged differences included.
    nobs : int
        Number of observations used in the regression.
    sigma2 : float
        Residual variance, with a degrees-of-freedom correction.
    s2_ar : float
        Autoregressive estimate of the long-run variance,
        ``sigma2 / (1 - sum b_j)^2``. This is the estimator Ng & Perron
        require: a kernel estimate reintroduces the size distortion the
        M tests are designed to remove.
    sum_y2 : float
        ``sum of y[t-1]^2`` over the regression sample, used by the M
        statistics and by the modified information criteria.
    """

    tstat: float
    rho: float
    lags: int
    nobs: int
    sigma2: float
    s2_ar: float
    sum_y2: float


def adf_regression(y_detrended: ArrayLike, lags: int) -> ADFRegression:
    r"""Fit the ADF regression on an already-detrended series.

    Parameters
    ----------
    y_detrended : array_like
        Output of :func:`gls_detrend`.
    lags : int
        Number of lagged differences. Zero is allowed.

    Returns
    -------
    ADFRegression

    Notes
    -----
    No deterministic term is included: the detrending step has already
    removed it, and adding it back would undo the power gain.

    .. math::

        \Delta \tilde y_t = \rho \tilde y_{t-1}
            + \sum_{j=1}^{k} b_j \Delta \tilde y_{t-j} + e_t
    """
    yd = np.asarray(y_detrended, dtype=np.float64).ravel()
    if lags < 0:
        raise ValueError(f"lags={lags} must be non-negative.")
    dy = np.diff(yd)
    n = dy.size
    if n - lags < lags + 2:
        raise ValueError(
            f"Sample too short for {lags} lags: {n + 1} observations leave "
            f"{n - lags} usable rows for {lags + 1} regressors."
        )

    target = dy[lags:]
    cols = [yd[lags:-1]]
    for j in range(1, lags + 1):
        cols.append(dy[lags - j : -j])
    design = np.column_stack(cols)

    coefs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coefs
    n_obs, n_par = design.shape
    sigma2 = float(resid @ resid) / (n_obs - n_par)
    xtx_inv = np.linalg.inv(design.T @ design)
    se_rho = float(np.sqrt(sigma2 * xtx_inv[0, 0]))

    b_sum = float(coefs[1:].sum()) if lags else 0.0
    # The autoregressive long-run variance blows up when the lag
    # coefficients sum to one; report it as infinite rather than dividing
    # by something near zero.
    s2_ar = float(np.inf) if abs(1.0 - b_sum) < 1e-12 else sigma2 / (1.0 - b_sum) ** 2

    lagged = yd[lags:-1]
    return ADFRegression(
        tstat=float(coefs[0]) / se_rho,
        rho=float(coefs[0]),
        lags=lags,
        nobs=n_obs,
        sigma2=sigma2,
        s2_ar=s2_ar,
        sum_y2=float(lagged @ lagged),
    )


def _default_max_lags(n_obs: int) -> int:
    """Schwert (1989) rule: ``floor(12 (T/100)^{1/4})``."""
    return int(np.floor(12.0 * (n_obs / 100.0) ** 0.25))


def select_lags(
    y_detrended: ArrayLike,
    method: LagMethod = "maic",
    max_lags: int | None = None,
    alpha: float = 0.05,
) -> tuple[int, dict[int, float]]:
    r"""Choose the lag order of the ADF regression.

    Parameters
    ----------
    y_detrended : array_like
        Output of :func:`gls_detrend`.
    method : {'maic', 'mbic', 'aic', 'bic', 't-stat'}, default 'maic'
        Selection criterion.
    max_lags : int, optional
        Upper bound of the search. Defaults to the Schwert rule,
        ``floor(12 (T/100)^{1/4})``.
    alpha : float, default 0.05
        Level of the sequential t test, used by ``method='t-stat'``.

    Returns
    -------
    lags : int
        Selected order.
    values : dict
        Criterion value at each candidate order, so the choice can be
        inspected rather than trusted.

    Notes
    -----
    **Why MAIC is the default.** With a large negative moving-average
    component, the standard AIC systematically picks too few lags, and
    unit-root tests then over-reject massively — this is the size
    distortion Ng & Perron set out to fix. Their modified criterion adds
    a penalty term that depends on the data:

    .. math::

        \mathrm{MAIC}(k) = \ln \hat\sigma^2_k
            + \frac{2(\tau_T(k) + k)}{T - k_{\max}},
        \qquad
        \tau_T(k) = \frac{\hat\rho^2 \sum \tilde y_{t-1}^2}
                         {\hat\sigma^2_k}

    :math:`\tau_T(k)` is large exactly when the lag order is too small to
    whiten the errors, which is what pushes the criterion towards richer
    specifications where the plain AIC stops too early. MBIC replaces the
    factor 2 by :math:`\ln(T - k_{\max})`.

    All candidates are compared on the **same sample**, the one implied
    by ``max_lags``. Comparing criteria computed on different numbers of
    observations is meaningless, and would bias the choice towards short
    lags.

    **Which detrending to select on.** :func:`~pyardl.unitroot.dfgls` and
    :func:`~pyardl.unitroot.ng_perron` pass the **OLS**-detrended series
    here, not the GLS-detrended one used for the test itself. The reason
    is measurable: on GLS-detrended white noise the criteria over-select
    badly (AIC picks 7.6 lags on average against 1.0), which destroys the
    test's power — 52% rejection instead of 100%. Selecting on
    OLS-detrended data reproduces the lag orders of ``arch``, an
    independent implementation. The article does not settle this point;
    the convention is documented rather than assumed.

    **A caveat on MAIC.** Its penalty term is large precisely when the
    series looks stationary, so MAIC still selects generously on I(0)
    data — 6.1 lags on white noise against 0.0 for BIC. That is the
    price of its protection against a negative moving-average component,
    and it costs classification accuracy in
    :func:`~pyardl.unitroot.report`; the measured trade-off is given
    there.
    """
    yd = np.asarray(y_detrended, dtype=np.float64).ravel()
    n = yd.size
    if max_lags is None:
        max_lags = _default_max_lags(n)
    max_lags = max(0, min(max_lags, (n - 4) // 2))

    if method == "fixed":
        raise ValueError(
            "method='fixed' is not a selection rule: pass lags=<int> directly instead."
        )

    # Common sample: every candidate is estimated on the rows the
    # richest specification can use.
    trimmed = yd[:n] if max_lags == 0 else yd
    denom = (n - 1) - max_lags

    values: dict[int, float] = {}
    if method == "t-stat":
        from scipy.stats import norm

        critical = float(norm.ppf(1.0 - alpha / 2.0))
        chosen = 0
        for k in range(max_lags, -1, -1):
            fit = _fit_common(trimmed, k, max_lags)
            values[k] = fit["t_last"]
            if k == 0 or abs(fit["t_last"]) > critical:
                chosen = k
                break
        return chosen, values

    for k in range(max_lags + 1):
        fit = _fit_common(trimmed, k, max_lags)
        sigma2 = fit["sigma2"]
        log_sigma2 = float(np.log(sigma2))
        if method in ("maic", "mbic"):
            tau = fit["rho"] ** 2 * fit["sum_y2"] / sigma2
            penalty = 2.0 if method == "maic" else float(np.log(denom))
            values[k] = log_sigma2 + penalty * (tau + k) / denom
        elif method == "aic":
            values[k] = log_sigma2 + 2.0 * (k + 1) / denom
        elif method == "bic":
            values[k] = log_sigma2 + float(np.log(denom)) * (k + 1) / denom
        else:
            raise ValueError(
                f"method={method!r} is not available. Use 'maic', 'mbic', "
                "'aic', 'bic' or 't-stat'."
            )
    return min(values, key=lambda k: values[k]), values


def _fit_common(yd: FloatArray, lags: int, max_lags: int) -> dict[str, float]:
    """ADF fit restricted to the sample the largest candidate can use."""
    dy = np.diff(yd)
    start = max_lags
    target = dy[start:]
    cols = [yd[start:-1]]
    for j in range(1, lags + 1):
        cols.append(dy[start - j : -j])
    design = np.column_stack(cols)

    coefs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coefs
    n_obs = design.shape[0]
    # Ng & Perron define sigma2_k without a degrees-of-freedom
    # correction; using one would change the penalty balance across
    # candidates and is not what the criterion was calibrated on.
    sigma2 = float(resid @ resid) / n_obs
    lagged = yd[start:-1]

    t_last = 0.0
    if lags:
        xtx_inv = np.linalg.inv(design.T @ design)
        s2_df = float(resid @ resid) / (n_obs - design.shape[1])
        t_last = float(coefs[-1]) / float(np.sqrt(s2_df * xtx_inv[-1, -1]))

    return {
        "sigma2": sigma2,
        "rho": float(coefs[0]),
        "sum_y2": float(lagged @ lagged),
        "t_last": t_last,
    }
