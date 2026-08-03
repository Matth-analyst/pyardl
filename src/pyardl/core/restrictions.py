r"""Testing restrictions on the long-run coefficients.

The methodological contribution of Davidson, Hendry, Srba & Yeo (1978)
was not a new estimator but a discipline: write the model so that the
economics is testable, then test it. Their consumption function is built
around one such restriction — a long-run elasticity of consumption to
income equal to one, which turns the level term into the ratio
:math:`\log(C/Y)`.

That is what this module provides for any fitted ARDL: a Wald test of
:math:`R\theta = r` on the long-run coefficients, and the option to
re-estimate with the restriction imposed.

Why both
--------
The Wald test asks whether the data reject the restriction. Imposing it
answers a different question — what the model looks like once you accept
it. The two are complementary, and the second is what DHSY actually did:
having failed to reject unit elasticity, they wrote the model in terms
of the ratio and gained a degree of freedom plus an interpretable error
correction term.

References
----------
.. [1] Davidson, J. E. H., Hendry, D. F., Srba, F. & Yeo, S. (1978).
       Econometric modelling of the aggregate time-series relationship
       between consumers' expenditure and income in the United Kingdom.
       *The Economic Journal*, 88(352), 661-692.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyardl.core.transforms import longrun_coefs, longrun_covariance
from pyardl.utils import lag_matrix

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    from pyardl.core.ardl import ARDLResults

    FloatArray = NDArray[np.float64]

__all__ = ["LongRunRestrictionResults", "longrun_restriction"]


@dataclass(frozen=True)
class LongRunRestrictionResults:
    """Outcome of a test of ``R theta = r``.

    Attributes
    ----------
    statistic : float
        The Wald statistic.
    df : int
        Degrees of freedom, the rank of ``R``.
    pvalue : float
        Right-tail probability under the chi-squared null.
    theta : pandas.Series
        The unrestricted long-run coefficients.
    discrepancy : numpy.ndarray
        ``R theta - r``, so the direction of the violation is visible and
        not just its size.
    f_statistic, f_pvalue : float or None
        Regression F test of the same restriction, available only when
        ``impose=True``. It compares the restricted and unrestricted
        residual sums of squares.
    ssr_restricted, ssr_unrestricted : float or None
        The two sums of squares, for inspection.
    restricted_params : pandas.Series or None
        Coefficients of the restricted error-correction model.
    """

    statistic: float
    df: int
    pvalue: float
    theta: pd.Series
    discrepancy: FloatArray
    f_statistic: float | None = None
    f_pvalue: float | None = None
    ssr_restricted: float | None = None
    ssr_unrestricted: float | None = None
    restricted_params: pd.Series | None = None

    def decision(self, alpha: float = 0.05) -> str:
        """``'reject'`` or ``'not_rejected'`` at level ``alpha``.

        Notes
        -----
        ``'not_rejected'`` rather than ``'accept'``: failing to reject a
        restriction is not evidence that it holds, especially on the
        short samples this literature works with.
        """
        return "reject" if self.pvalue < alpha else "not_rejected"

    def summary(self) -> str:
        """Readable report of the test."""
        lines = [
            f"Long-run restriction test - Wald chi2({self.df}) = "
            f"{self.statistic:.4f}, p = {self.pvalue:.4f}",
            f"  decision (5%): {self.decision(0.05)}",
            f"  R.theta - r = {np.array2string(self.discrepancy, precision=4)}",
        ]
        if self.f_statistic is not None:
            assert self.f_pvalue is not None
            lines += [
                "",
                f"  imposed: F = {self.f_statistic:.4f}, p = {self.f_pvalue:.4f}",
                f"  SSR unrestricted = {self.ssr_unrestricted:.6f}, "
                f"restricted = {self.ssr_restricted:.6f}",
            ]
        return "\n".join(lines)


def _uecm_design(
    res: ARDLResults, ratio_with: str | None = None
) -> tuple[FloatArray, FloatArray, list[str]]:
    r"""Build the error-correction design matrix of a fitted ARDL.

    Parameters
    ----------
    res : ARDLResults
        The fitted model.
    ratio_with : str, optional
        Name of the regressor whose long-run coefficient is set to one.
        Its level column is then folded into the lagged level of ``y``,
        giving the single regressor :math:`y_{t-1} - x_{j,t-1}`.

    Returns
    -------
    design, y_dep, names

    Notes
    -----
    The unrestricted version of this design is an exact
    reparameterisation of the ARDL: same sample, same residuals, same
    sum of squares. That identity is what makes the F test below
    legitimate, and it is verified by a test rather than assumed.
    """
    model = res.model
    y, x = model._y, model._x
    n = y.shape[0]
    hb = model.hold_back

    cols: list[FloatArray] = []
    names: list[str] = []

    if model.det in ("const", "trend"):
        cols.append(np.ones(n - hb))
        names.append("const")
    if model.det == "trend":
        cols.append(np.arange(hb + 1, n + 1, dtype=np.float64))
        names.append("trend")

    # Level terms.
    y_lag1 = lag_matrix(y, hb, first_lag=1)[:, 0]
    level_names = list(model._x_names)
    ratio_index = None
    if ratio_with is not None:
        if ratio_with not in level_names:
            raise ValueError(
                f"ratio_with={ratio_with!r} is not a regressor of this model. "
                f"Available: {level_names}."
            )
        ratio_index = level_names.index(ratio_with)

    if ratio_index is None:
        cols.append(y_lag1)
        names.append(f"{model._y_name}.L1")
    else:
        assert x is not None
        x_lag1 = lag_matrix(x[:, ratio_index], hb, first_lag=1)[:, 0]
        cols.append(np.asarray(y_lag1 - x_lag1, dtype=np.float64))
        names.append(f"({model._y_name}-{ratio_with}).L1")

    if x is not None:
        for j, name in enumerate(model._x_names):
            if j == ratio_index:
                continue
            # q_j = 0 keeps the contemporaneous level, as everywhere else
            # in the library.
            first = 1 if model.q[j] > 0 else 0
            cols.append(lag_matrix(x[:, j], hb, first_lag=first)[:, 0])
            names.append(f"{name}.L{first}")

    # Short-run terms.
    dy = np.diff(y)
    for i in range(1, model.p):
        cols.append(lag_matrix(dy, hb - 1, first_lag=i)[:, 0])
        names.append(f"D.{model._y_name}.L{i}")

    if x is not None:
        for j, name in enumerate(model._x_names):
            dx = np.diff(x[:, j])
            for i in range(model.q[j]):
                cols.append(lag_matrix(dx, hb - 1, first_lag=i)[:, 0])
                names.append(f"D.{name}.L{i}")

    if model._fixed is not None:
        cols.extend(model._fixed[hb:].T)
        names.extend(model._fixed_names)

    design = np.column_stack(cols)
    y_dep = dy[hb - 1 :]
    return design, y_dep, names


def longrun_restriction(
    res: ARDLResults,
    r_matrix: ArrayLike,
    value: ArrayLike | float = 0.0,
    impose: bool = False,
) -> LongRunRestrictionResults:
    r"""Wald test of ``R theta = r`` on the long-run coefficients.

    Parameters
    ----------
    res : ARDLResults
        A fitted model.
    r_matrix : array_like
        Restriction matrix ``R``, shape ``(q, k)`` with ``k`` the number
        of regressors. A 1-D array is read as a single restriction.
    value : array_like or float, default 0.0
        Right-hand side ``r``.
    impose : bool, default False
        Also re-estimate the error-correction model with the restriction
        imposed, and report the regression F test. Currently supported
        only for the homogeneity restriction ``theta_j = 1`` on a single
        regressor — the DHSY case, where the level term collapses to the
        ratio ``y - x_j``.

    Returns
    -------
    LongRunRestrictionResults

    Raises
    ------
    ValueError
        If the shapes do not match, if ``R`` has no full row rank, or if
        ``impose=True`` is asked for a restriction that is not of the
        supported form. Nothing is silently approximated.

    Notes
    -----
    The covariance of :math:`\hat\theta` comes from the delta method with
    the analytical gradient, the same one the standard errors in
    ``.longrun`` are built on, so the test and the reported standard
    errors cannot disagree.

    .. math::

        W = (R\hat\theta - r)'
            \left[ R \hat V_\theta R' \right]^{-1}
            (R\hat\theta - r) \sim \chi^2(\mathrm{rank}\, R)

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.core.ardl import ARDL
    >>> rng = np.random.default_rng(0)
    >>> x = pd.Series(np.cumsum(rng.standard_normal(200)), name="x")
    >>> y = pd.Series(np.zeros(200), name="y")
    >>> for t in range(1, 200):
    ...     y.iloc[t] = 0.6 * y.iloc[t - 1] + 0.4 * x.iloc[t] + rng.standard_normal()
    >>> res = ARDL(y, x, order=(1, 1))._fit()
    >>> out = res.test_longrun_restriction([[1.0]], 1.0)
    >>> out.df
    1
    """
    from scipy.stats import chi2
    from scipy.stats import f as f_dist

    params = res.ardl_params
    theta = longrun_coefs(params)
    cov_theta = longrun_covariance(params)

    r_arr = np.atleast_2d(np.asarray(r_matrix, dtype=np.float64))
    k = theta.size
    if r_arr.shape[1] != k:
        raise ValueError(
            f"R has {r_arr.shape[1]} columns but the model has {k} long-run "
            f"coefficients {list(theta.index)}."
        )
    q = r_arr.shape[0]
    rhs = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if rhs.size == 1 and q > 1:
        rhs = np.repeat(rhs, q)
    if rhs.size != q:
        raise ValueError(f"r has {rhs.size} entries but R has {q} rows.")
    if np.linalg.matrix_rank(r_arr) < q:
        raise ValueError(
            "R does not have full row rank: the restrictions are redundant "
            "or contradictory."
        )

    discrepancy = r_arr @ theta.to_numpy() - rhs
    middle = r_arr @ cov_theta @ r_arr.T
    stat = float(discrepancy @ np.linalg.solve(middle, discrepancy))
    pvalue = float(chi2.sf(stat, q))

    f_stat = f_p = ssr_r = ssr_u = None
    restricted_params = None

    if impose:
        target = _homogeneity_target(r_arr, rhs, theta)
        design_u, y_dep, _ = _uecm_design(res)
        design_r, _, names_r = _uecm_design(res, ratio_with=target)

        beta_u = np.linalg.lstsq(design_u, y_dep, rcond=None)[0]
        resid_u = y_dep - design_u @ beta_u
        ssr_u = float(resid_u @ resid_u)

        beta_r = np.linalg.lstsq(design_r, y_dep, rcond=None)[0]
        resid_r = y_dep - design_r @ beta_r
        ssr_r = float(resid_r @ resid_r)

        df_resid = design_u.shape[0] - design_u.shape[1]
        f_stat = float(((ssr_r - ssr_u) / q) / (ssr_u / df_resid))
        f_p = float(f_dist.sf(f_stat, q, df_resid))
        restricted_params = pd.Series(beta_r, index=names_r, name="coef")

    return LongRunRestrictionResults(
        statistic=stat,
        df=q,
        pvalue=pvalue,
        theta=theta,
        discrepancy=discrepancy,
        f_statistic=f_stat,
        f_pvalue=f_p,
        ssr_restricted=ssr_r,
        ssr_unrestricted=ssr_u,
        restricted_params=restricted_params,
    )


def _homogeneity_target(r_arr: FloatArray, rhs: FloatArray, theta: pd.Series) -> str:
    """Name of the regressor whose long-run coefficient is set to one.

    Imposition is only implemented for the restriction that makes the
    level term a ratio. Anything else raises rather than quietly
    imposing something different from what was tested.
    """
    if r_arr.shape[0] != 1:
        raise ValueError(
            "impose=True supports a single restriction. Joint restrictions "
            "can be tested (impose=False) but not imposed, because the "
            "error-correction form has no ratio representation for them."
        )
    row = r_arr[0]
    nonzero = np.flatnonzero(np.abs(row) > 1e-12)
    if nonzero.size != 1 or not np.isclose(row[nonzero[0]], 1.0):
        raise ValueError(
            "impose=True supports the homogeneity restriction "
            "theta_j = 1 only, i.e. a row of R with a single entry equal "
            "to 1. Received "
            f"{np.array2string(row, precision=4)}."
        )
    if not np.isclose(rhs[0], 1.0):
        raise ValueError(
            f"impose=True requires r = 1 (unit long-run elasticity), got "
            f"{rhs[0]}. A different value has no ratio representation."
        )
    return str(theta.index[int(nonzero[0])])
