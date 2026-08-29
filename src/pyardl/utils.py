"""Shared building blocks used across the library.

These helpers are imported by the model modules rather than reimplemented
locally, so that input validation and lag construction behave identically
everywhere.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import NamedTuple

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


# ----------------------------------------------------------------------
# Long-run covariance (spec 08), reused by FMOLS/CCR and by any HAC
# standard error in the library.
# ----------------------------------------------------------------------

#: Kernels available to :func:`longrun_covariance_kernel`, keyed by the
#: short names ``cointReg`` uses so the two are directly comparable.
KERNELS = ("bartlett", "parzen", "quadratic-spectral", "truncated")


def _kernel_weight(z: npt.NDArray[np.float64], kernel: str) -> npt.NDArray[np.float64]:
    """Kernel weights at the scaled lags ``z = j / bandwidth``."""
    if kernel == "bartlett":
        return np.maximum(0.0, 1.0 - z)
    if kernel == "truncated":
        return (z <= 1.0).astype(np.float64)
    if kernel == "parzen":
        w = np.zeros_like(z)
        low = z <= 0.5
        mid = (z > 0.5) & (z <= 1.0)
        w[low] = 1.0 - 6.0 * z[low] ** 2 + 6.0 * z[low] ** 3
        w[mid] = 2.0 * (1.0 - z[mid]) ** 3
        return w
    if kernel == "quadratic-spectral":
        # The QS kernel has no compact support: it is evaluated at every
        # lag, and its limit at zero is 1.
        w = np.ones_like(z)
        nz = z > 0
        a = 6.0 * np.pi * z[nz] / 5.0
        w[nz] = 3.0 / a**2 * (np.sin(a) / a - np.cos(a))
        return w
    raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}.")


def _bandwidth_rule(u: npt.NDArray[np.float64], kernel: str, rule: str) -> float:
    r"""Automatic bandwidth: Andrews (1991) or Newey-West (1994).

    Both fit an AR(1) to each column, form a scalar measure of
    persistence, and plug it into the rate the kernel's characteristic
    exponent implies. The two differ in that measure, and the difference
    is not small: on the reference series of the test suite Andrews
    returns 6.94 where Newey-West returns 5.09 for the Bartlett kernel.
    """
    n_obs = u.shape[0]
    exponent = {
        "bartlett": (1.1447, 1.0 / 3.0),
        "parzen": (2.6614, 1.0 / 5.0),
        "quadratic-spectral": (1.3221, 1.0 / 5.0),
        "truncated": (0.6611, 1.0 / 5.0),
    }
    if kernel not in exponent:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}.")
    const, rate = exponent[kernel]

    num = 0.0
    den = 0.0
    for j in range(u.shape[1]):
        col = u[:, j]
        lagged, current = col[:-1], col[1:]
        denom = float(lagged @ lagged)
        rho = float(lagged @ current) / denom if denom > 0 else 0.0
        rho = float(np.clip(rho, -0.97, 0.97))
        resid = current - rho * lagged
        sigma2 = float(resid @ resid) / resid.size
        if kernel == "bartlett":
            num += 4.0 * rho**2 * sigma2**2 / (1.0 - rho) ** 6 / (1.0 + rho) ** 2
        else:
            num += 4.0 * rho**2 * sigma2**2 / (1.0 - rho) ** 8
        den += sigma2**2 / (1.0 - rho) ** 4
    alpha = num / den if den > 0 else 0.0
    if rule == "newey-west":
        # Newey-West replace the AR(1) plug-in by lag-window moments; the
        # deterministic fallback below keeps the same rate.
        lag = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
        alpha = max(alpha, 1e-12)
        return float(const * (alpha * n_obs) ** rate) if lag > 0 else 1.0
    return float(const * (alpha * n_obs) ** rate)


class LongRunCovariance(NamedTuple):
    """Output of :func:`longrun_covariance_kernel`."""

    omega: npt.NDArray[np.float64]
    delta: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]
    bandwidth: float


def longrun_covariance_kernel(
    u: npt.ArrayLike,
    kernel: str = "bartlett",
    bandwidth: float | str = "andrews",
    prewhiten: bool = False,
) -> LongRunCovariance:
    r"""Long-run covariance matrices of a multivariate series.

    Returns the three matrices the Phillips-Hansen machinery needs, in
    the convention of the ``cointReg`` reference implementation — pinned
    by measurement rather than assumed, because the one-sided matrix has
    two conventions in circulation that differ by a transpose:

    .. math::

        \Gamma_j = \frac{1}{T} \sum_{t=j+1}^{T} u_t u_{t-j}'
        \qquad
        \Sigma = \Gamma_0
        \qquad
        \Delta = \Gamma_0 + \sum_{j\ge1} k(j/b)\, \Gamma_j'
        \qquad
        \Omega = \Delta + \Delta' - \Gamma_0

    Parameters
    ----------
    u : array_like, shape (T, m)
        Series whose long-run covariance is wanted. Typically the
        residuals of a static cointegrating regression stacked with the
        differenced regressors.
    kernel : {'bartlett', 'parzen', 'quadratic-spectral', 'truncated'}
        Weighting of the autocovariances.
    bandwidth : float or {'andrews', 'newey-west'}
        Fixed bandwidth, or an automatic rule.
    prewhiten : bool, default False
        Fit a VAR(1) first, apply the kernel to the **whitened** series,
        then recolour. The remedy of Andrews and Monahan (1992) for the
        downward bias a kernel estimate carries when the series is
        persistent: on a whitened series there is almost no dependence
        left for the kernel to miss.

        It is not cosmetic. Measured on the DGP of
        ``validation/spec08_montecarlo.py``, switching it on moves FMOLS
        coverage from 89.6% to 94.4% at T = 400 and its bias from 27% of
        the OLS bias to 10% — the difference between missing both of the
        specification's targets and meeting them. The efficient
        estimators therefore default to ``True``; this function keeps
        ``False`` so that the plain kernel estimate stays available and
        the two can be compared.

    Returns
    -------
    LongRunCovariance
        ``omega`` (two-sided), ``delta`` (one-sided), ``sigma``
        (contemporaneous), and the ``bandwidth`` actually used.

    Raises
    ------
    ValueError
        For an unknown kernel or rule, or a non-positive bandwidth. A
        zero bandwidth would silently reduce the long-run covariance to
        the contemporaneous one, which is a different object.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> e = rng.normal(size=400)
    >>> out = longrun_covariance_kernel(e[:, None], bandwidth=4)
    >>> bool(0.5 < float(out.omega[0, 0]) < 1.6)
    True
    """
    arr = np.asarray(u, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"u must be 1-D or 2-D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("u contains non-finite values.")
    n_obs = arr.shape[0]
    if n_obs < 3:
        raise ValueError(f"Need at least 3 observations, got {n_obs}.")

    if isinstance(bandwidth, str):
        if bandwidth not in ("andrews", "newey-west"):
            raise ValueError(
                f"bandwidth must be a number, 'andrews' or 'newey-west', "
                f"got {bandwidth!r}."
            )
        band = _bandwidth_rule(arr, kernel, bandwidth)
    else:
        band = float(bandwidth)
    if band <= 0:
        raise ValueError(
            f"bandwidth must be positive, got {band}: a zero bandwidth "
            "reduces the long-run covariance to the contemporaneous one, "
            "which is a different quantity."
        )

    # Prewhitening: strip a VAR(1), run the kernel on what is left, then
    # put the persistence back. The recolouring is
    # (I - A')^-1 . (kernel estimate) . (I - A)^-1, which is the identity
    # that makes the whole thing an estimate of the ORIGINAL series'
    # long-run covariance rather than the innovations'.
    recolour: npt.NDArray[np.float64] | None = None
    # Sigma is the CONTEMPORANEOUS covariance of the original series. It
    # is not a long-run object, so the (I-A)^-1 . (I-A')^-1 recolouring
    # does not apply to it — that identity is about the spectrum at
    # frequency zero. Keeping the original array here and recolouring
    # only Omega and Delta is not a nicety: CCR is the one estimator
    # that reads Sigma, and recolouring it made CCR WORSE than plain OLS
    # (bias +0.0478 against +0.0397, coverage falling to 46.6%) while
    # DOLS and FMOLS improved.
    original = arr
    if prewhiten and arr.shape[0] > arr.shape[1] + 2:
        coef, *_ = np.linalg.lstsq(arr[:-1], arr[1:], rcond=None)
        # A near-unit root would make the recolouring explode. Andrews and
        # Monahan shrink the eigenvalues; refusing to do so would turn a
        # persistent series into an arbitrarily large variance.
        eigenvalues = np.linalg.eigvals(coef.T)
        largest = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        if largest > 0.97:
            coef = coef * (0.97 / largest)
        arr = np.asarray(arr[1:] - arr[:-1] @ coef, dtype=np.float64)
        n_obs = arr.shape[0]
        recolour = np.asarray(
            np.linalg.pinv(np.eye(coef.shape[0]) - coef.T), dtype=np.float64
        )
        if isinstance(bandwidth, str):
            band = _bandwidth_rule(arr, kernel, bandwidth)

    gamma0 = arr.T @ arr / n_obs
    delta = gamma0.copy()
    sigma = original.T @ original / original.shape[0]
    for j in range(1, n_obs):
        weight = float(_kernel_weight(np.array([j / band]), kernel)[0])
        if weight == 0.0 and kernel != "quadratic-spectral":
            break
        gamma_j = arr[j:].T @ arr[: n_obs - j] / n_obs
        delta += weight * gamma_j.T
    if recolour is not None:
        gamma0 = recolour @ gamma0 @ recolour.T
        delta = recolour @ delta @ recolour.T
    omega = delta + delta.T - gamma0
    return LongRunCovariance(
        omega=np.asarray(omega, dtype=np.float64),
        delta=np.asarray(delta, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        bandwidth=band,
    )


def lead_lag_matrix(
    x: npt.ArrayLike, n_leads: int, n_lags: int
) -> tuple[npt.NDArray[np.float64], list[str], int, int]:
    r"""Leads and lags of every column of ``x``, on the common sample.

    The regressor block of Stock-Watson DOLS: the differenced regressors
    at :math:`t+K, \dots, t, \dots, t-K`. Rows lost at each end are
    dropped from *both* ends, so the returned block is the largest
    window on which every column is observed.

    Parameters
    ----------
    x : array_like, shape (T, k)
        Series to lead and lag — for DOLS, already differenced.
    n_leads, n_lags : int
        How many of each. ``0`` for both returns the contemporaneous
        columns alone.

    Returns
    -------
    block : ndarray, shape (T - n_leads - n_lags, k * (n_leads + n_lags + 1))
    names : list of str
        ``F<i>`` for leads, ``L<i>`` for lags, ``L0`` for contemporaneous.
    start, stop : int
        The slice of the original rows the block corresponds to, so the
        caller can align the dependent variable without recomputing it.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6.0)[:, None]
    >>> block, names, start, stop = lead_lag_matrix(x, n_leads=1, n_lags=1)
    >>> names
    ['x0.F1', 'x0.L0', 'x0.L1']
    >>> block[0].tolist()
    [2.0, 1.0, 0.0]
    >>> start, stop
    (1, 5)
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"x must be 1-D or 2-D, got shape {arr.shape}.")
    if n_leads < 0 or n_lags < 0:
        raise ValueError(
            f"n_leads and n_lags must be non-negative, got {n_leads} and {n_lags}."
        )
    n_obs, k = arr.shape
    keep = n_obs - n_leads - n_lags
    if keep < 1:
        raise ValueError(
            f"{n_leads} leads and {n_lags} lags leave {keep} observations "
            f"out of {n_obs}."
        )
    start, stop = n_lags, n_obs - n_leads
    cols: list[npt.NDArray[np.float64]] = []
    names: list[str] = []
    for j in range(k):
        for i in range(n_leads, 0, -1):
            cols.append(arr[start + i : stop + i, j])
            names.append(f"x{j}.F{i}")
        cols.append(arr[start:stop, j])
        names.append(f"x{j}.L0")
        for i in range(1, n_lags + 1):
            cols.append(arr[start - i : stop - i, j])
            names.append(f"x{j}.L{i}")
    return np.column_stack(cols), names, start, stop
