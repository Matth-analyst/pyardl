"""Shared building blocks used across the library.

These helpers are imported by the model modules rather than reimplemented
locally, so that input validation and lag construction behave identically
everywhere.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning


def _delta_method(
    g: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64] | float],
    theta_hat: npt.NDArray[np.float64],
    v_hat: npt.NDArray[np.float64],
    *,
    step: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generic delta method: ``g(theta_hat)`` and ``grad(g)' V grad(g)``.

    The gradient of ``g`` is obtained by central finite differences, so no
    closed form is required. Modules that do have an analytical gradient use
    it instead; this helper serves as a generic fallback and as a
    cross-check in tests.

    Parameters
    ----------
    g : callable
        Function of the parameter vector whose variance is wanted.
    theta_hat : ndarray, shape (n_params,)
        Point estimate of the parameter vector.
    v_hat : ndarray, shape (n_params, n_params)
        Estimated covariance matrix of ``theta_hat``.
    step : float
        Relative step size of the central finite differences.

    Returns
    -------
    g_hat : ndarray
        ``g`` evaluated at ``theta_hat``, flattened to 1-D.
    cov_g : ndarray, shape (m, m)
        Covariance matrix of ``g(theta_hat)``.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.array([2.0, 0.5])
    >>> v = np.diag([0.01, 0.0004])
    >>> g = lambda t: np.array([t[0] / (1 - t[1])])
    >>> g_hat, cov_g = _delta_method(g, theta, v)
    >>> round(float(g_hat[0]), 6)
    4.0
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    n = theta_hat.shape[0]

    g0 = np.atleast_1d(np.asarray(g(theta_hat), dtype=np.float64))
    m = g0.shape[0]

    jac = np.empty((m, n), dtype=np.float64)
    for i in range(n):
        bump = np.zeros(n, dtype=np.float64)
        bump[i] = step * max(abs(theta_hat[i]), 1.0)
        theta_plus = (theta_hat + bump).astype(np.float64)
        theta_minus = (theta_hat - bump).astype(np.float64)
        g_plus = np.atleast_1d(np.asarray(g(theta_plus), dtype=np.float64))
        g_minus = np.atleast_1d(np.asarray(g(theta_minus), dtype=np.float64))
        jac[:, i] = (g_plus - g_minus) / (2 * bump[i])

    cov_g = (jac @ v_hat @ jac.T).astype(np.float64)
    return g0, cov_g


def lag_matrix(
    x: npt.ArrayLike, lags: int, *, first_lag: int = 0
) -> npt.NDArray[np.float64]:
    """Build the matrix of lagged values of a series.

    Parameters
    ----------
    x : array-like, shape (T,)
        Input series.
    lags : int
        Highest lag to include (must be at least ``first_lag``).
    first_lag : int
        Lowest lag to include: ``0`` gives columns
        ``x_t, ..., x_{t-lags}``; ``1`` gives ``x_{t-1}, ..., x_{t-lags}``,
        which is what an ARDL model needs for the lags of the dependent
        variable.

    Returns
    -------
    ndarray, shape (T - lags, lags - first_lag + 1)
        Column ``i`` holds ``x_{t - (first_lag + i)}``. The first ``lags``
        observations are dropped so that all columns are aligned.

    Examples
    --------
    >>> import numpy as np
    >>> lag_matrix(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    array([[3., 2., 1.],
           [4., 3., 2.]])
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x must be 1-D.")
    if lags < first_lag or first_lag < 0:
        raise ValueError("lags >= first_lag >= 0 is required.")
    t_len = arr.shape[0]
    if t_len <= lags:
        raise ValueError(f"Series is too short: T={t_len} <= lags={lags}.")
    cols = [arr[lags - i : t_len - i] for i in range(first_lag, lags + 1)]
    return np.column_stack(cols)


def diff(
    x: npt.ArrayLike,
    d: int = 1,
    D: int = 0,
    s: int = 4,
) -> npt.NDArray[np.float64] | pd.Series:
    r"""Apply the differencing operator :math:`(1-L)^d (1-L^s)^D`.

    Parameters
    ----------
    x : array-like or pandas.Series, shape (T,)
        Input series.
    d : int, default 1
        Order of ordinary differencing.
    D : int, default 0
        Order of seasonal differencing.
    s : int, default 4
        Seasonal period. Quarterly data by default, as in the
        consumption function of Davidson, Hendry, Srba & Yeo (1978).

    Returns
    -------
    ndarray or pandas.Series, shape (T - d - D*s,)
        The differenced series. A Series keeps the tail of the original
        index, so the result stays aligned with the dates it belongs to
        rather than silently shifting by ``d + D*s`` periods.

    Raises
    ------
    ValueError
        If ``d`` or ``D`` is negative, if ``s < 1``, or if the series is
        too short to survive the requested differencing.

    Notes
    -----
    The two operators commute, so the order of application does not
    matter; ordinary differencing is applied first here.

    Seasonal differencing is not a detrending convenience: on quarterly
    data :math:`\Delta_4 y_t = y_t - y_{t-4}` removes a fixed seasonal
    pattern *and* a unit root at the same time, which is why DHSY built
    their consumption function on it. Combining it with an ordinary
    difference, ``d=1, D=1``, removes a seasonal pattern that itself
    drifts.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.utils import diff
    >>> x = np.arange(8.0)
    >>> diff(x, d=1)
    array([1., 1., 1., 1., 1., 1., 1.])
    >>> diff(x, d=0, D=1, s=4)
    array([4., 4., 4., 4.])
    """
    if d < 0 or D < 0:
        raise ValueError(f"d={d} and D={D} must be non-negative.")
    if s < 1:
        raise ValueError(f"s={s} must be at least 1.")

    index = x.index if isinstance(x, pd.Series) else None
    name = x.name if isinstance(x, pd.Series) else None
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x must be 1-D.")

    dropped = d + D * s
    if arr.shape[0] <= dropped:
        raise ValueError(
            f"Series is too short: T={arr.shape[0]} observations cannot "
            f"survive (1-L)^{d} (1-L^{s})^{D}, which drops {dropped}."
        )

    out: npt.NDArray[np.float64] = arr
    for _ in range(d):
        out = np.asarray(out[1:] - out[:-1], dtype=np.float64)
    for _ in range(D):
        out = np.asarray(out[s:] - out[:-s], dtype=np.float64)

    if index is not None:
        return pd.Series(out, index=index[dropped:], name=name)
    return out


def check_series(
    y: npt.ArrayLike,
    x: npt.ArrayLike | None = None,
    *,
    min_obs: int = 15,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
    pd.Index | None,
    str,
    tuple[str, ...],
]:
    """Validate and normalise the input series shared by all estimators.

    Checks that ``y`` and ``x`` have the same length and non-zero variance,
    trims leading and trailing NaNs (with a warning) while rejecting
    internal NaNs, and warns when the sample is too small for asymptotic
    inference to be trustworthy. A pandas index is preserved when supplied.

    Parameters
    ----------
    y : array-like, shape (T,)
        Dependent variable.
    x : array-like, shape (T,) or (T, k), optional
        Regressors, as a Series, DataFrame or ndarray. Column names are
        taken from a DataFrame when available.
    min_obs : int
        Sample size below which a small-sample warning is issued.

    Returns
    -------
    y_arr : ndarray, shape (T',)
    x_arr : ndarray, shape (T', k) or None
    index : pandas.Index or None
        Index aligned on the retained sample, or ``None`` for plain arrays.
    y_name : str
    x_names : tuple of str

    Raises
    ------
    ValueError
        If lengths differ, an internal NaN is found, or a series is
        constant.
    """
    y_name = getattr(y, "name", None) or "y"
    index = y.index if isinstance(y, pd.Series) else None

    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValueError("y must be 1-D.")

    x_names: tuple[str, ...] = ()
    x_arr: npt.NDArray[np.float64] | None = None
    if x is not None:
        if isinstance(x, pd.DataFrame):
            x_names = tuple(str(c) for c in x.columns)
            if index is None:
                index = x.index
        elif isinstance(x, pd.Series):
            x_names = (str(x.name) if x.name is not None else "x0",)
            if index is None:
                index = x.index
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if x_arr.ndim != 2:
            raise ValueError("x must be 1-D or 2-D.")
        if not x_names:
            x_names = tuple(f"x{j}" for j in range(x_arr.shape[1]))
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Incompatible lengths: y has {y_arr.shape[0]} observations, "
                f"x has {x_arr.shape[0]}."
            )

    # Leading/trailing NaNs are trimmed with a warning; internal NaNs are an
    # error, since silently dropping them would break the time ordering.
    stacked = y_arr[:, None] if x_arr is None else np.column_stack([y_arr, x_arr])
    valid = np.asarray(~np.isnan(stacked).any(axis=1), dtype=np.bool_)
    if not valid.all():
        first, last = (
            int(np.argmax(valid)),
            int(len(valid) - 1 - np.argmax(valid[::-1])),
        )
        if not valid[first : last + 1].all():
            raise ValueError(
                "Internal NaN detected; only leading and trailing NaNs can be "
                "trimmed automatically."
            )
        warnings.warn(
            f"Trimmed {int((~valid).sum())} leading/trailing observation(s) "
            "containing NaN.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )
        y_arr = y_arr[first : last + 1]
        if x_arr is not None:
            x_arr = x_arr[first : last + 1]
        if index is not None:
            index = index[first : last + 1]

    if y_arr.shape[0] < min_obs:
        warnings.warn(
            f"Very small sample (n={y_arr.shape[0]} < {min_obs}): asymptotic "
            "inference is not reliable.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )
    if np.var(y_arr) == 0.0:
        raise ValueError("y has zero variance.")
    if x_arr is not None and (np.var(x_arr, axis=0) == 0.0).any():
        raise ValueError("At least one column of x has zero variance.")

    return y_arr, x_arr, index, str(y_name), x_names
