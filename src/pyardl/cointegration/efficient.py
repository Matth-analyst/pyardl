r"""Efficient single-equation long-run estimators (spec 08).

Static OLS on cointegrated variables is consistent — Engle and Granger's
superconsistency — and its **inference is not**. Three things break it
at once: the regressors are correlated with the equation error
(endogeneity), the error is serially correlated, and the two interact to
leave a second-order bias that does not vanish at the usual rate. The
`t` statistics that come out of a static regression therefore have a
non-standard distribution, and reading them against a Student table
overstates significance.

Three estimators repair this, by three different routes, and pyardl
implements all three because applied referees ask for them as a
robustness block next to the ARDL long run.

**DOLS** (Stock & Watson) adds leads *and* lags of :math:`\Delta x` to
the static regression. The leads are the point: they absorb the feedback
from the error onto future regressor changes, which is exactly the
endogeneity. What remains is serial correlation in the residual, handled
by a HAC standard error. Conceptually the simplest, and the one to reach
for first.

**FMOLS** (Phillips & Hansen) leaves the regression alone and corrects
the *data* and the *estimator*: the dependent variable is purged of the
part explained by the regressor innovations, and an explicit bias term
is subtracted. It needs the long-run covariance matrices, which is where
:func:`~pyardl.utils.longrun_covariance_kernel` comes in.

**CCR** (Park) transforms both :math:`y` and :math:`x` so that ordinary
least squares on the transformed data is already efficient. Same
ingredients as FMOLS, applied earlier.

All three deliver a long-run coefficient whose `t` is asymptotically
standard normal, which is the whole point: the number can be compared to
1.96 without apology.

References
----------
.. [1] Phillips, P. C. B. & Hansen, B. E. (1990). Statistical inference
       in instrumental variables regression with I(1) processes.
       *Review of Economic Studies*, 57(1), 99-125.
.. [2] Stock, J. H. & Watson, M. W. (1993). A simple estimator of
       cointegrating vectors in higher order integrated systems.
       *Econometrica*, 61(4), 783-820.
.. [3] Park, J. Y. (1992). Canonical cointegrating regressions.
       *Econometrica*, 60(1), 119-143.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.utils import (
    LongRunCovariance,
    check_series,
    lead_lag_matrix,
    longrun_covariance_kernel,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]

__all__ = [
    "EfficientLongRunResults",
    "ccr",
    "compare_longrun",
    "default_dols_lags",
    "dols",
    "fmols",
]

DetType = Literal["none", "const", "trend"]


def default_dols_lags(n_obs: int) -> int:
    r"""Leads and lags for DOLS: :math:`\lfloor T^{1/3} \rfloor`.

    Computed with :func:`numpy.cbrt`, not ``n_obs ** (1/3)``: at a
    perfect cube the power form lands just below the integer and the
    floor loses one — ``64 ** (1/3) == 3.99999999999999956``. The same
    trap as the cross-sectional lag rule of spec 24.

    Examples
    --------
    >>> [default_dols_lags(t) for t in (27, 64, 125, 1000)]
    [3, 4, 5, 10]
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be positive, got {n_obs}.")
    return int(np.floor(np.cbrt(float(n_obs))))


def _deterministic(
    n_obs: int, det: DetType, offset: int = 0
) -> tuple[FloatArray, list[str]]:
    if det not in ("none", "const", "trend"):
        raise ValueError(f"det must be 'none', 'const' or 'trend', got {det!r}.")
    cols: list[FloatArray] = []
    names: list[str] = []
    if det in ("const", "trend"):
        cols.append(np.ones(n_obs))
        names.append("const")
    if det == "trend":
        cols.append(np.arange(1.0 + offset, n_obs + 1.0 + offset, dtype=np.float64))
        names.append("trend")
    if not cols:
        return np.empty((n_obs, 0)), []
    return np.column_stack(cols), names


@dataclass(frozen=True)
class EfficientLongRunResults:
    """Long-run coefficients from DOLS, FMOLS or CCR.

    Attributes
    ----------
    longrun : pandas.DataFrame
        One row per regressor: ``theta``, ``se``, ``t``, ``pvalue`` and
        the 95% bounds. The ``t`` is asymptotically **standard normal**,
        which is what these estimators buy over static OLS.
    method : str
        ``'DOLS'``, ``'FMOLS'`` or ``'CCR'``.
    deterministic : pandas.Series
        Estimated deterministic terms, kept out of ``longrun`` so that a
        comparison table across methods lines up on the regressors.
    omega_uv : float
        The conditional long-run variance of the equation error. The
        scale of every standard error above.
    """

    longrun: pd.DataFrame
    method: str
    deterministic: pd.Series
    nobs: int
    omega_uv: float
    bandwidth: float | None = None
    kernel: str | None = None
    n_leads: int | None = None
    n_lags: int | None = None
    resid: pd.Series = field(default_factory=pd.Series, repr=False)

    def summary(self) -> str:
        """Publication-style table."""
        lines = [f"{self.method} long-run estimates - {self.nobs} observations"]
        if self.method == "DOLS":
            lines.append(
                f"  {self.n_leads} lead(s) and {self.n_lags} lag(s) of the "
                f"differenced regressors; HAC standard errors "
                f"({self.kernel}, bandwidth {self.bandwidth:.4g})"
            )
        else:
            lines.append(
                f"  long-run covariance: {self.kernel} kernel, "
                f"bandwidth {self.bandwidth:.4g}"
            )
        lines += [
            "  t statistics are asymptotically standard normal",
            "",
            f"    {'':<10}{'theta':>12}{'se':>12}{'z':>10}{'p':>10}",
        ]
        for name, row in self.longrun.iterrows():
            lines.append(
                f"    {str(name):<10}{row['theta']:>12.4f}{row['se']:>12.4f}"
                f"{row['t']:>10.3f}{row['pvalue']:>10.4f}"
            )
        return "\n".join(lines)


def _table(theta: FloatArray, se: FloatArray, names: Sequence[str]) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, theta / se, np.nan)
    crit = float(stats.norm.ppf(0.975))
    return pd.DataFrame(
        {
            "theta": theta,
            "se": se,
            "t": z,
            "pvalue": 2.0 * stats.norm.sf(np.abs(z)),
            "ci_lower": theta - crit * se,
            "ci_upper": theta + crit * se,
        },
        index=pd.Index(list(names), name="regressor"),
    )


def _prepare(
    y: npt.ArrayLike, x: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    y_arr, x_arr, _, _, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError(
            "These estimators need at least one regressor: with none there "
            "is no long-run relationship to estimate."
        )
    return y_arr, x_arr, x_names


def dols(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    det: DetType = "const",
    n_leads: int | None = None,
    n_lags: int | None = None,
    kernel: str = "bartlett",
    bandwidth: float | str = "andrews",
) -> EfficientLongRunResults:
    r"""Dynamic OLS (Stock & Watson 1993).

    .. math::

        y_t = d_t + \theta' x_t
              + \sum_{i=-K}^{K} \gamma_i' \Delta x_{t+i} + u_t

    The **leads** are what removes the endogeneity: they absorb the
    feedback running from the equation error to future changes in the
    regressors. Omit them and the estimator is static OLS with extra
    columns.

    Parameters
    ----------
    y, x : array_like
        Dependent variable and I(1) regressors.
    det : {'none', 'const', 'trend'}
        Deterministic terms.
    n_leads, n_lags : int, optional
        Defaults to :func:`default_dols_lags` for both.
    kernel, bandwidth
        HAC settings for the standard errors; see
        :func:`~pyardl.utils.longrun_covariance_kernel`.

    Returns
    -------
    EfficientLongRunResults

    Examples
    --------
    >>> from pyardl.datasets import load_denmark
    >>> d = load_denmark()
    >>> res = dols(d["LRM"], d[["LRY", "IBO", "IDE"]],
    ...            n_leads=2, n_lags=2, bandwidth=5)
    >>> res.method
    'DOLS'
    >>> round(float(res.longrun.loc["LRY", "theta"]), 6)
    1.221423
    """
    y_arr, x_arr, x_names = _prepare(y, x)
    n_obs = y_arr.shape[0]
    leads = default_dols_lags(n_obs) if n_leads is None else int(n_leads)
    lags = default_dols_lags(n_obs) if n_lags is None else int(n_lags)
    if leads < 0 or lags < 0:
        raise ValueError(
            f"n_leads and n_lags must be non-negative, got {leads} and {lags}."
        )

    dx = np.diff(x_arr, axis=0)
    block, _, start, stop = lead_lag_matrix(dx, leads, lags)
    # dx row i corresponds to observation i+1 of the level series.
    lo, hi = start + 1, stop + 1
    y_use = y_arr[lo:hi]
    x_use = x_arr[lo:hi]
    det_block, det_names = _deterministic(y_use.size, det, offset=lo)

    design = np.column_stack([c for c in (det_block, x_use, block) if c.size])
    beta, *_ = np.linalg.lstsq(design, y_use, rcond=None)
    resid = y_use - design @ beta
    n_use = y_use.size

    # HAC covariance of the whole coefficient vector, then the block for
    # theta. The long-run variance of u is what makes the t standard.
    lrv = longrun_covariance_kernel(resid[:, None], kernel=kernel, bandwidth=bandwidth)
    omega_u = float(lrv.omega[0, 0])
    xtx_inv = np.linalg.pinv(design.T @ design)
    # Omega is already the (1/T)-normalised long-run variance, so the
    # sandwich is Omega_u (X'X)^-1 with NO extra factor of T. Putting one
    # there inflates every standard error by sqrt(T) — which looks
    # plausible on its own and is off by a factor of seven at T = 50.
    cov = omega_u * xtx_inv
    n_det = det_block.shape[1]
    k = x_use.shape[1]
    sl = slice(n_det, n_det + k)
    theta = np.asarray(beta[sl], dtype=np.float64)
    se = np.asarray(np.sqrt(np.diag(cov)[sl]), dtype=np.float64)

    return EfficientLongRunResults(
        longrun=_table(theta, se, x_names),
        method="DOLS",
        deterministic=pd.Series(beta[:n_det], index=pd.Index(det_names, name="term")),
        nobs=n_use,
        omega_uv=omega_u,
        bandwidth=lrv.bandwidth,
        kernel=kernel,
        n_leads=leads,
        n_lags=lags,
        resid=pd.Series(resid),
    )


def _fm_pieces(
    y_arr: FloatArray,
    x_arr: FloatArray,
    det: DetType,
    kernel: str,
    bandwidth: float | str,
) -> tuple[FloatArray, FloatArray, FloatArray, LongRunCovariance, int]:
    """Static regression, its residuals, and the long-run covariances."""
    n_obs = y_arr.shape[0]
    det_block, _ = _deterministic(n_obs, det)
    static = np.column_stack([c for c in (det_block, x_arr) if c.size])
    beta0, *_ = np.linalg.lstsq(static, y_arr, rcond=None)
    u_hat = np.asarray(y_arr - static @ beta0, dtype=np.float64)
    v_hat = np.asarray(np.diff(x_arr, axis=0), dtype=np.float64)
    # u and v must share dates: v_t is defined from t = 2 onwards.
    stacked = np.column_stack([u_hat[1:], v_hat])
    lrv = longrun_covariance_kernel(stacked, kernel=kernel, bandwidth=bandwidth)
    return u_hat, v_hat, static, lrv, det_block.shape[1]


def fmols(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    det: DetType = "const",
    kernel: str = "bartlett",
    bandwidth: float | str = "andrews",
) -> EfficientLongRunResults:
    r"""Fully modified OLS (Phillips & Hansen 1990).

    Two corrections applied to a static regression:

    1. **Endogeneity**, by purging the dependent variable of the part
       the regressor innovations explain in the long run:
       :math:`y^+ = y - \Omega_{uv}\Omega_{vv}^{-1} \hat v`.
    2. **Second-order bias**, by subtracting an explicit term built from
       the one-sided long-run covariance:
       :math:`\lambda^+ = \Delta_{vu} - \Delta_{vv}\Omega_{vv}^{-1}\Omega_{vu}`.

    Parameters
    ----------
    y, x : array_like
    det : {'none', 'const', 'trend'}
    kernel, bandwidth
        Passed to :func:`~pyardl.utils.longrun_covariance_kernel`.

    Returns
    -------
    EfficientLongRunResults

    Examples
    --------
    >>> from pyardl.datasets import load_denmark
    >>> d = load_denmark()
    >>> res = fmols(d["LRM"], d[["LRY", "IBO", "IDE"]], bandwidth=5)
    >>> round(float(res.longrun.loc["LRY", "theta"]), 6)
    1.290357
    """
    y_arr, x_arr, x_names = _prepare(y, x)
    _, v_hat, static, lrv, n_det = _fm_pieces(y_arr, x_arr, det, kernel, bandwidth)
    omega = np.asarray(lrv.omega, dtype=np.float64)
    delta = np.asarray(lrv.delta, dtype=np.float64)

    k = x_arr.shape[1]
    omega_vv = omega[1:, 1:]
    omega_uv = omega[0:1, 1:]
    omega_vu = omega[1:, 0:1]
    delta_vv = delta[1:, 1:]
    delta_vu = delta[1:, 0:1]
    inv_vv = np.linalg.pinv(omega_vv)

    # Correction 1 — endogeneity. Defined only where v is, so the
    # regression runs on T-1 rows.
    y_plus = y_arr[1:] - (v_hat @ inv_vv @ omega_vu).ravel()
    design = static[1:]
    n_use = y_plus.size

    # Correction 2 — second-order bias. The scale is the FULL sample
    # size, not the T-1 rows the regression uses. That one-observation
    # difference is not cosmetic: it moves theta by 3.1e-02 here, which
    # is a hundred times the agreement it buys. Pinned against cointReg
    # by solving for the lambda+ its published coefficients imply — the
    # ratio to the T-1 version came out at exactly 55/54 on all three
    # coefficients, which is what identified the convention.
    n_full = y_arr.shape[0]
    lam_plus = (delta_vu - delta_vv @ inv_vv @ omega_vu).ravel()
    rhs = design.T @ y_plus
    rhs[n_det:] -= n_full * lam_plus

    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ rhs

    # Omega is already (1/T)-normalised, so the sandwich carries no
    # extra factor of T.
    omega_u_v = float(omega[0, 0] - (omega_uv @ inv_vv @ omega_vu).ravel()[0])
    cov = omega_u_v * xtx_inv
    sl = slice(n_det, n_det + k)
    theta = np.asarray(beta[sl], dtype=np.float64)
    se = np.asarray(np.sqrt(np.maximum(np.diag(cov)[sl], 0.0)), dtype=np.float64)

    det_names = _deterministic(1, det)[1]
    return EfficientLongRunResults(
        longrun=_table(theta, se, x_names),
        method="FMOLS",
        deterministic=pd.Series(beta[:n_det], index=pd.Index(det_names, name="term")),
        nobs=n_use,
        omega_uv=omega_u_v,
        bandwidth=lrv.bandwidth,
        kernel=kernel,
        resid=pd.Series(y_plus - design @ beta),
    )


def ccr(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    det: DetType = "const",
    kernel: str = "bartlett",
    bandwidth: float | str = "andrews",
) -> EfficientLongRunResults:
    r"""Canonical cointegrating regression (Park 1992).

    Same ingredients as FMOLS, applied one step earlier: instead of
    correcting the estimator, CCR transforms :math:`y` and :math:`x` so
    that ordinary least squares on the transformed data is already
    efficient. Asymptotically equivalent to FMOLS; in finite samples the
    two differ, which is exactly why reporting both is informative.

    Examples
    --------
    >>> from pyardl.datasets import load_denmark
    >>> d = load_denmark()
    >>> res = ccr(d["LRM"], d[["LRY", "IBO", "IDE"]], bandwidth=5)
    >>> res.method
    'CCR'
    >>> bool(np.isfinite(res.longrun["theta"]).all())
    True
    """
    y_arr, x_arr, x_names = _prepare(y, x)
    u_hat, v_hat, static, lrv, n_det = _fm_pieces(y_arr, x_arr, det, kernel, bandwidth)
    omega = np.asarray(lrv.omega, dtype=np.float64)
    delta = np.asarray(lrv.delta, dtype=np.float64)
    sigma = np.asarray(lrv.sigma, dtype=np.float64)

    k = x_arr.shape[1]
    inv_vv = np.linalg.pinv(omega[1:, 1:])
    inv_sigma = np.linalg.pinv(sigma)

    # Park's transformation, applied to the sample where v exists.
    # z_t = (u_t, v_t); delta_2 is the block of the one-sided covariance
    # that multiplies v.
    #
    # NOTE the transpose. The one-sided matrix has two conventions in
    # circulation that differ by exactly this, and FMOLS and CCR do NOT
    # use the same one — they come from papers with different notation.
    # `longrun_covariance_kernel` returns the convention Phillips-Hansen
    # needs (pinned against cointReg to 2.2e-16); Park's transformation
    # needs the other. Measured on 400 replications of an endogenous
    # cointegrated DGP: with the FMOLS convention CCR removes almost
    # nothing (bias +0.0370 against +0.0384 for plain OLS at T = 400),
    # with the transpose it removes most of it (+0.0103), which is where
    # FMOLS sits too (+0.0108). Asymptotic equivalence says they should
    # agree — the wrong transpose made them disagree, and that is what
    # exposed it.
    z = np.column_stack([u_hat[1:], v_hat])
    delta_2 = delta.T[:, 1:]
    shift = z @ (inv_sigma @ delta_2)

    beta_static, *_ = np.linalg.lstsq(static, y_arr, rcond=None)
    theta_static = beta_static[n_det:]
    gamma = (inv_vv @ omega[1:, 0:1]).ravel()

    x_star = x_arr[1:] - shift
    y_star = y_arr[1:] - shift @ theta_static - v_hat @ gamma

    det_block, det_names = _deterministic(y_star.size, det, offset=1)
    design = np.column_stack([c for c in (det_block, x_star) if c.size])
    beta, *_ = np.linalg.lstsq(design, y_star, rcond=None)

    omega_u_v = float(
        omega[0, 0] - (omega[0:1, 1:] @ inv_vv @ omega[1:, 0:1]).ravel()[0]
    )
    xtx_inv = np.linalg.pinv(design.T @ design)
    cov = omega_u_v * xtx_inv
    sl = slice(n_det, n_det + k)
    theta = np.asarray(beta[sl], dtype=np.float64)
    se = np.asarray(np.sqrt(np.maximum(np.diag(cov)[sl], 0.0)), dtype=np.float64)

    return EfficientLongRunResults(
        longrun=_table(theta, se, x_names),
        method="CCR",
        deterministic=pd.Series(beta[:n_det], index=pd.Index(det_names, name="term")),
        nobs=int(y_star.size),
        omega_uv=omega_u_v,
        bandwidth=lrv.bandwidth,
        kernel=kernel,
        resid=pd.Series(y_star - design @ beta),
    )


def compare_longrun(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    ardl_results: object | None = None,
    det: DetType = "const",
    kernel: str = "bartlett",
    bandwidth: float | str = "andrews",
    n_leads: int | None = None,
    n_lags: int | None = None,
) -> pd.DataFrame:
    """Long-run coefficients from every available estimator, side by side.

    The robustness block applied papers report. When the ARDL long run
    and the three efficient estimators agree, the conclusion rests on
    something other than one specification; when they disagree, the
    disagreement is the result.

    Parameters
    ----------
    y, x : array_like
    ardl_results : ARDLResults, optional
        A fitted ARDL whose ``longrun`` is added as a row block. Passing
        it is the point of the function, but it is optional so the three
        efficient estimators can be compared on their own.
    det, kernel, bandwidth, n_leads, n_lags
        Passed through.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(method, regressor)``, columns ``theta``, ``se``,
        ``t``.

    Examples
    --------
    >>> from pyardl.datasets import load_denmark
    >>> d = load_denmark()
    >>> table = compare_longrun(d["LRM"], d[["LRY", "IBO", "IDE"]], bandwidth=5)
    >>> sorted(set(table.index.get_level_values("method")))
    ['CCR', 'DOLS', 'FMOLS']
    """
    frames: list[pd.DataFrame] = []
    if ardl_results is not None:
        block = getattr(ardl_results, "longrun", None)
        if block is None:
            raise ValueError(
                "ardl_results has no `longrun` attribute; pass a fitted "
                "ARDLResults or None."
            )
        part = block[["theta", "se"]].copy()
        part["t"] = part["theta"] / part["se"]
        part["method"] = "ARDL"
        # ARDLResults.longrun does not name its index, and the efficient
        # estimators do. Naming it here rather than assuming keeps the
        # two blocks stackable.
        part.index = pd.Index(list(part.index), name="regressor")
        frames.append(part.reset_index().set_index(["method", "regressor"]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyardlMethodologyWarning)
        runs = [
            dols(
                y,
                x,
                det=det,
                n_leads=n_leads,
                n_lags=n_lags,
                kernel=kernel,
                bandwidth=bandwidth,
            ),
            fmols(y, x, det=det, kernel=kernel, bandwidth=bandwidth),
            ccr(y, x, det=det, kernel=kernel, bandwidth=bandwidth),
        ]
    for res in runs:
        label = res.method
        part = res.longrun[["theta", "se", "t"]].copy()
        part["method"] = label
        frames.append(part.reset_index().set_index(["method", "regressor"]))

    return pd.concat(frames)[["theta", "se", "t"]]
