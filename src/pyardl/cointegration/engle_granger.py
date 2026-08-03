r"""Two-step cointegration test of Engle & Granger (1987).

The idea is direct: if two I(1) series are cointegrated, some linear
combination of them is stationary. So estimate that combination by least
squares, then test its residuals for a unit root.

.. math::

    \text{step 1:}\quad y_t &= \text{det}_t + x_t'\beta + u_t \\
    \text{step 2:}\quad \Delta \hat u_t &= \rho \hat u_{t-1}
        + \sum_i \xi_i \Delta \hat u_{t-i} + e_t

The null is **no cointegration**, so a large negative t-ratio on
:math:`\rho` is evidence *for* a long-run relationship.

Why this is not the library's main tool
---------------------------------------
Engle-Granger is here as a point of comparison and because it is what
much of the literature reports. Three limitations are structural, not
fixable, and each of them is a reason the bounds test exists:

- **The normalisation is arbitrary.** Regressing ``y`` on ``x`` and
  regressing ``x`` on ``y`` are different tests, and they can disagree.
  Nothing in the method says which is right.
- **Only one relationship can be found.** With three or more variables
  several cointegrating vectors may exist; this procedure sees at most
  one and gives no warning.
- **Every series must be I(1).** A mixture of I(0) and I(1) regressors
  invalidates the test. Establishing that beforehand is itself a
  sequence of tests, each with its own error rate.

The bounds test of Pesaran, Shin & Smith drops the third restriction
entirely, which is why :func:`pyardl.bounds.bounds_test` is the
recommended route.

References
----------
.. [1] Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error
       correction: representation, estimation, and testing.
       *Econometrica*, 55(2), 251-276.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyardl.critical_values.mackinnon import eg_critical_values, eg_pvalue
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.unitroot.gls import adf_regression, select_lags
from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

EGTrend = Literal["n", "c", "ct", "ctt"]

__all__ = ["engle_granger", "EGResults"]


@dataclass(frozen=True)
class EGResults:
    """Outcome of an Engle-Granger test.

    Attributes
    ----------
    stat : float
        Step-two t-ratio. Left-tailed.
    pvalue : float
        Approximate asymptotic p-value.
    critical_values : dict
        Level to critical value, at 10%, 5% and 1%.
    longrun_params : pandas.Series
        Coefficients of the first-step regression. **Not for inference**
        — see :attr:`longrun_params` notes below.
    resid : pandas.Series
        First-step residuals, the estimated equilibrium error.
    lags : int
        Lag order of the step-two regression.
    nobs : int
        Observations in the first-step regression.
    trend : str
        Deterministic terms of the first step.
    ecm : pandas.DataFrame or None
        Second-step error-correction model, when ``fit_ecm=True``.
    """

    stat: float
    pvalue: float
    critical_values: dict[float, float]
    longrun_params: pd.Series
    resid: pd.Series
    lags: int
    nobs: int
    n_vars: int
    trend: EGTrend
    ecm: pd.DataFrame | None = field(default=None, repr=False)

    def decision(self, alpha: float = 0.05) -> str:
        """``'cointegration'`` or ``'no_cointegration'`` at ``alpha``.

        Notes
        -----
        Unlike the bounds test, this one has a single critical value, so
        the verdict is binary. That is not a virtue: the certainty is
        bought by assuming every series is I(1), an assumption the test
        does not check.
        """
        if np.isnan(self.critical_values[alpha]):
            raise ValueError(
                f"No critical value at alpha={alpha} for trend="
                f"{self.trend!r}; the decision cannot be taken."
            )
        return (
            "cointegration"
            if self.stat < self.critical_values[alpha]
            else "no_cointegration"
        )

    def summary(self) -> str:
        """Readable report of the test."""
        cv = "  ".join(
            f"{int(a * 100)}%: {v:.4f}" for a, v in sorted(self.critical_values.items())
        )
        lines = [
            f"Engle-Granger test (1987) - trend '{self.trend}', "
            f"{self.n_vars} variables, lags={self.lags}, nobs={self.nobs}",
            f"  statistic = {self.stat:.4f}   p-value = {self.pvalue:.4f}",
            f"  decision (5%): {self.decision(0.05)}",
            f"  critical values (left tail)   {cv}",
            "  H0: no cointegration",
            "",
            "  Long-run coefficients (point estimates only, no inference):",
        ]
        lines += [
            f"    {name:<12}{value: .4f}" for name, value in self.longrun_params.items()
        ]
        return "\n".join(lines)


def engle_granger(
    y: ArrayLike,
    x: ArrayLike,
    trend: EGTrend = "c",
    max_lags: int | None = None,
    ic: str = "aic",
    fit_ecm: bool = False,
) -> EGResults:
    r"""Test for cointegration by the two-step procedure.

    Parameters
    ----------
    y : array_like
        Dependent variable, shape ``(T,)``. Which variable goes here is
        a choice the method does not make for you, and the answer can
        change with it — see the module notes.
    x : array_like
        Regressors, shape ``(T, k)``.
    trend : {'n', 'c', 'ct', 'ctt'}, default 'c'
        Deterministic terms of the first-step regression. The null
        distribution depends on them.
    max_lags : int, optional
        Upper bound for the step-two lag search. Defaults to the Schwert
        rule.
    ic : {'aic', 'bic', 't-stat', 'maic', 'mbic'}, default 'aic'
        Criterion for the step-two lag order.
    fit_ecm : bool, default False
        Also estimate the second-step error-correction model,
        :math:`\Delta y_t = \alpha \hat u_{t-1} + \Delta x_t'\gamma
        + \varepsilon_t`.

    Returns
    -------
    EGResults

    Warns
    -----
    PyardlMethodologyWarning
        When the first-step fit is almost perfect, which makes the
        residuals numerically meaningless; and when ``trend='n'``, for
        which no critical values were published.

    Notes
    -----
    **Do not run inference on the first step.** Its coefficients are
    super-consistent — they converge faster than usual — but their
    distribution is non-standard and they carry a second-order bias that
    does not vanish at the usual rate. The reported standard errors of an
    ordinary regression are simply wrong here. For long-run inference,
    use the ARDL route (:attr:`pyardl.core.ardl.ARDLResults.longrun`),
    where the delta method applies.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.cointegration import engle_granger
    >>> rng = np.random.default_rng(0)
    >>> x = np.cumsum(rng.standard_normal(200))
    >>> y = 1.5 * x + rng.standard_normal(200)
    >>> engle_granger(y, x).decision(0.05)
    'cointegration'
    """
    y_arr, x_arr, index, y_name, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("Engle-Granger needs at least one regressor.")
    n_obs, k = x_arr.shape
    n_vars = k + 1

    cols = [x_arr]
    names = list(x_names)
    if trend in ("c", "ct", "ctt"):
        cols.append(np.ones((n_obs, 1)))
        names.append("const")
    if trend in ("ct", "ctt"):
        cols.append(np.arange(1, n_obs + 1, dtype=np.float64)[:, None])
        names.append("trend")
    if trend == "ctt":
        squared = np.arange(1, n_obs + 1, dtype=np.float64) ** 2
        cols.append(np.asarray(squared, dtype=np.float64)[:, None])
        names.append("trend2")
    elif trend not in ("n", "c", "ct"):
        raise ValueError(f"trend={trend!r} must be 'n', 'c', 'ct' or 'ctt'.")

    design = np.column_stack(cols)
    beta, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    resid: FloatArray = np.asarray(y_arr - design @ beta, dtype=np.float64)

    tss = float(((y_arr - y_arr.mean()) ** 2).sum())
    if tss > 0 and float(resid @ resid) / tss < 1e-12:
        warnings.warn(
            "The first-step regression fits almost perfectly: the residuals "
            "are rounding error and the test carries no information. This "
            "usually means one regressor is a linear combination of the "
            "others, or of y.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    chosen, _ = select_lags(resid, method=ic, max_lags=max_lags)  # type: ignore[arg-type]
    fit = adf_regression(resid, chosen)

    ecm = None
    if fit_ecm:
        ecm = _fit_ecm(y_arr, x_arr, resid, x_names, index)

    resid_series = pd.Series(
        resid, index=index if index is not None else None, name=f"{y_name}.resid"
    )
    return EGResults(
        stat=fit.tstat,
        pvalue=eg_pvalue(fit.tstat, n_vars, trend),
        critical_values=eg_critical_values(n_vars, n_obs, trend),
        longrun_params=pd.Series(beta, index=names, name="coef"),
        resid=resid_series,
        lags=chosen,
        nobs=n_obs,
        n_vars=n_vars,
        trend=trend,
        ecm=ecm,
    )


def _fit_ecm(
    y: FloatArray,
    x: FloatArray,
    resid: FloatArray,
    x_names: tuple[str, ...],
    index: pd.Index | None,
) -> pd.DataFrame:
    """Second-step error-correction model.

    The lagged equilibrium error enters as a regressor, and its
    coefficient is the adjustment speed. Its standard error is treated as
    ordinary here, which is legitimate: because the first-step estimate
    converges faster, the estimation error it injects is asymptotically
    negligible in this regression.
    """
    dy = np.diff(y)
    dx = np.diff(x, axis=0)
    lagged = resid[:-1]

    design = np.column_stack([lagged, dx])
    names = ["ecm.L1"] + [f"D.{n}" for n in x_names]

    beta, _, _, _ = np.linalg.lstsq(design, dy, rcond=None)
    err = dy - design @ beta
    n, m = design.shape
    sigma2 = float(err @ err) / (n - m)
    xtx_inv = np.linalg.inv(design.T @ design)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))

    from scipy.stats import t as t_dist

    tvalues = beta / se
    return pd.DataFrame(
        {
            "coef": beta,
            "se": se,
            "t": tvalues,
            "pvalue": 2 * t_dist.sf(np.abs(tvalues), n - m),
        },
        index=names,
    )
