r"""Koyck (1954): the geometric distributed lag, and why OLS cannot fit it.

An infinite distributed lag with geometrically declining weights,

.. math::

    y_t = \alpha + \sum_{i=0}^{\infty} \beta_0 \lambda^i x_{t-i}
          + \varepsilon_t, \qquad 0 < \lambda < 1,

cannot be estimated as written. Koyck's transformation — take
:math:`y_t - \lambda y_{t-1}` and watch the infinite sum telescope —
turns it into three parameters and one lag:

.. math::

    y_t = \alpha(1-\lambda) + \beta_0 x_t + \lambda y_{t-1} + u_t,
    \qquad u_t = \varepsilon_t - \lambda \varepsilon_{t-1}.

**And that transformation is exactly what breaks least squares.** The
error :math:`u_t` contains :math:`\varepsilon_{t-1}`, which is also
inside :math:`y_{t-1}`, so
:math:`\operatorname{Cov}(y_{t-1}, u_t) = -\lambda \sigma^2 \neq 0`.
OLS on the transformed equation is not merely inefficient: it is
**inconsistent**, and the bias does not shrink with the sample.

This module therefore offers three estimators and defaults to the one
that works. ``"ols"`` is kept — with a warning that fires every time —
because seeing the bias is more instructive than being protected from
it, and because a reader comparing pyardl against a textbook example
computed by OLS needs to be able to reproduce it.

References
----------
.. [1] Koyck, L. M. (1954). *Distributed Lags and Investment Analysis*.
       North-Holland.
.. [2] Liviatan, N. (1963). Consistent estimation of distributed lags.
       *International Economic Review*, 4(1), 44-52.
.. [3] Durbin, J. (1970). Testing for serial correlation in least-squares
       regression when some of the regressors are lagged dependent
       variables. *Econometrica*, 38(3), 410-421.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from pyardl.exceptions import DegenerateCaseWarning, PyardlMethodologyWarning
from pyardl.utils import check_series

__all__ = ["KoyckModel", "KoyckResults"]

FloatArray = npt.NDArray[np.float64]
KoyckMethod = Literal["ols", "iv", "ml"]

_PARAM_NAMES = ("alpha", "beta0", "lam")


def _ols(y: FloatArray, design: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Least squares with the covariance and the residual variance."""
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coefs
    n, k = design.shape
    sigma2 = float(resid @ resid) / (n - k)
    xtx_inv = np.linalg.pinv(design.T @ design)
    return coefs.astype(np.float64), (sigma2 * xtx_inv).astype(np.float64), sigma2


def _structural(coefs: FloatArray, cov: FloatArray) -> tuple[FloatArray, FloatArray]:
    r"""Map the regression parameters to :math:`(\alpha, \beta_0, \lambda)`.

    The regression is ``y = c + b x + rho y_{-1}``, so
    :math:`\lambda = \rho`, :math:`\beta_0 = b` and
    :math:`\alpha = c / (1 - \rho)`. Only the intercept is a non-linear
    function of the regression parameters, and its variance follows from
    the Jacobian below rather than from a numerical gradient — the
    closed form is short enough that approximating it would be a
    needless source of error.
    """
    c, b, rho = (float(v) for v in coefs)
    if abs(1.0 - rho) < 1e-12:
        alpha = float("nan")
        jac = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
    else:
        alpha = c / (1.0 - rho)
        jac = np.array(
            [
                [1.0 / (1.0 - rho), 0.0, c / (1.0 - rho) ** 2],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    params = np.array([alpha, b, rho], dtype=np.float64)
    return params, (jac @ cov @ jac.T).astype(np.float64)


class KoyckModel:
    r"""Geometric distributed lag estimated on the Koyck transformation.

    Parameters
    ----------
    y : array-like, shape (T,)
        Dependent variable.
    x : array-like, shape (T,)
        The single distributed-lag regressor.
    method : {"iv", "ols", "ml"}, default "iv"
        ``"iv"`` instruments the lagged dependent variable with
        :math:`x_{t-1}` (Liviatan). ``"ml"`` estimates the MA(1) error
        jointly, exploiting the cross-restriction that its coefficient is
        :math:`-\lambda`, the same lambda as the dynamics. ``"ols"`` is
        inconsistent and says so, loudly, on every fit.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x = rng.normal(size=n)
    >>> e = rng.normal(scale=0.5, size=n)
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 2.0 * 0.4 + 0.8 * x[t] + 0.6 * y[t - 1] + e[t] - 0.6 * e[t - 1]
    >>> res = KoyckModel(pd.Series(y, name="y"), pd.Series(x, name="x")).fit()
    >>> res.params.round(3)
    alpha    1.999
    beta0    0.854
    lam      0.601
    Name: coef, dtype: float64

    The truth is ``(2.0, 0.8, 0.6)``. Refitting the same data with
    ``method="ols"`` returns ``lam = 0.473`` — the inconsistency is not
    a footnote, it is a quarter of the parameter.
    """

    def __init__(
        self,
        y: npt.ArrayLike,
        x: npt.ArrayLike,
        method: KoyckMethod = "iv",
    ) -> None:
        if method not in ("ols", "iv", "ml"):
            raise ValueError(f'method must be "ols", "iv" or "ml", got {method!r}.')
        self.method: KoyckMethod = method

        y_arr, x_arr, index, y_name, x_names = check_series(y, x)
        if x_arr is None or x_arr.shape[1] != 1:
            raise ValueError(
                "The Koyck model takes exactly one distributed-lag "
                "regressor; use ARDL for several."
            )
        self._y = y_arr
        self._x = x_arr[:, 0]
        self._index = index
        self._y_name = y_name
        self._x_name = x_names[0]

    # ------------------------------------------------------------------
    def _design(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``y``, ``[1, x_t, y_{t-1}]`` and the instrument matrix."""
        y, x = self._y, self._x
        y_dep = y[1:]
        ones = np.ones(y_dep.shape[0], dtype=np.float64)
        design = np.column_stack([ones, x[1:], y[:-1]])
        instruments = np.column_stack([ones, x[1:], x[:-1]])
        return y_dep, design, instruments

    def _first_stage_f(self, y_lag: FloatArray, instruments: FloatArray) -> float:
        """F on the excluded instrument in the first-stage regression.

        Just-identified with one endogenous regressor, so this is the
        squared t on :math:`x_{t-1}`. Below 10 the IV estimate is not to
        be trusted, and the model says so rather than reporting a
        standard error that assumes otherwise.
        """
        coefs, cov, _ = _ols(y_lag, instruments)
        se = float(np.sqrt(cov[2, 2]))
        return float("inf") if se == 0.0 else float((coefs[2] / se) ** 2)

    def _fit_ols(self) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
        y_dep, design, _ = self._design()
        coefs, cov, _ = _ols(y_dep, design)
        warnings.warn(
            "method='ols' is INCONSISTENT on the Koyck transformation: "
            "the error u_t = e_t - lambda e_{t-1} contains e_{t-1}, which "
            "is also inside y_{t-1}, so Cov(y_{t-1}, u_t) = -lambda*sigma^2. "
            "The bias does not vanish as T grows. Use method='iv' or 'ml' "
            "for inference; keep 'ols' to reproduce a textbook figure.",
            PyardlMethodologyWarning,
            stacklevel=3,
        )
        resid = y_dep - design @ coefs
        return coefs, cov, resid, {}

    def _fit_iv(self) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
        y_dep, design, instruments = self._design()
        zx = instruments.T @ design
        coefs = np.linalg.solve(zx, instruments.T @ y_dep)
        resid = y_dep - design @ coefs
        n, k = design.shape
        sigma2 = float(resid @ resid) / (n - k)
        zx_inv = np.linalg.pinv(zx)
        cov = sigma2 * (zx_inv @ (instruments.T @ instruments) @ zx_inv.T)

        f_stat = self._first_stage_f(self._y[:-1], instruments)
        if f_stat < 10.0:
            warnings.warn(
                f"Weak instrument: the first-stage F on x_(t-1) is "
                f"{f_stat:.2f} (< 10). x_(t-1) explains too little of "
                "y_(t-1) for the IV estimate to be reliable; its "
                "standard error understates the true uncertainty.",
                PyardlMethodologyWarning,
                stacklevel=3,
            )
        return (
            coefs.astype(np.float64),
            cov.astype(np.float64),
            resid.astype(np.float64),
            {"first_stage_f": f_stat},
        )

    def _fit_ml(self) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
        from scipy.optimize import minimize
        from statsmodels.tools.numdiff import approx_hess

        y_dep, design, _ = self._design()
        y_lag, x_cur = design[:, 2], design[:, 1]
        n = y_dep.shape[0]

        def _innovations(theta: FloatArray) -> FloatArray:
            alpha, beta0, zeta = theta
            lam = 1.0 / (1.0 + np.exp(-zeta))
            eps = np.zeros(n, dtype=np.float64)
            base = y_dep - alpha * (1.0 - lam) - beta0 * x_cur - lam * y_lag
            # eps_t = base_t + lambda * eps_{t-1}, started at eps_0 = 0.
            # The recursion is what imposes the cross-restriction: the
            # MA(1) coefficient is not free, it IS -lambda.
            for t in range(1, n):
                eps[t] = base[t] + lam * eps[t - 1]
            eps[0] = base[0]
            return eps

        def _neg_loglike(theta: FloatArray) -> float:
            eps = _innovations(np.asarray(theta, dtype=np.float64))
            ssr = float(eps @ eps)
            if not np.isfinite(ssr) or ssr <= 0.0:
                return 1e12
            return float(0.5 * n * (np.log(2 * np.pi * ssr / n) + 1.0))

        # Started at the IV estimate: consistent, so the optimiser begins
        # inside the basin rather than hunting for it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyardlMethodologyWarning)
            iv_coefs, _, _, _ = self._fit_iv()
        iv_struct, _ = _structural(iv_coefs, np.eye(3))
        lam0 = float(np.clip(iv_struct[2], 0.02, 0.98))
        start = np.array(
            [
                float(iv_struct[0]) if np.isfinite(iv_struct[0]) else 0.0,
                float(iv_struct[1]),
                float(np.log(lam0 / (1.0 - lam0))),
            ],
            dtype=np.float64,
        )
        opt = minimize(_neg_loglike, start, method="L-BFGS-B")
        theta = np.asarray(opt.x, dtype=np.float64)

        hess = np.asarray(approx_hess(theta, _neg_loglike), dtype=np.float64)
        try:
            v_theta = np.linalg.inv(hess)
        except np.linalg.LinAlgError:  # pragma: no cover - singular Hessian
            v_theta = np.full((3, 3), np.nan)

        alpha, beta0, zeta = (float(v) for v in theta)
        lam = 1.0 / (1.0 + np.exp(-zeta))
        # From (alpha, beta0, zeta) to the regression parameters
        # (c, b, rho) = (alpha(1-lam), beta0, lam), so that everything
        # downstream sees the same three columns whatever the estimator.
        jac = np.array(
            [
                [1.0 - lam, 0.0, -alpha * lam * (1.0 - lam)],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, lam * (1.0 - lam)],
            ],
            dtype=np.float64,
        )
        coefs = np.array([alpha * (1.0 - lam), beta0, lam], dtype=np.float64)
        cov = jac @ v_theta @ jac.T
        eps = _innovations(theta)
        if not bool(opt.success):
            warnings.warn(
                f"ML did not converge ({opt.message}). The IV estimate is "
                "consistent and is the honest fallback; results are "
                "reported with converged=False rather than silently.",
                PyardlMethodologyWarning,
                stacklevel=3,
            )
        return (
            coefs,
            cov.astype(np.float64),
            eps.astype(np.float64),
            {"converged": bool(opt.success), "llf": -float(opt.fun)},
        )

    # ------------------------------------------------------------------
    def fit(self) -> KoyckResults:
        """Estimate the model and return the results object."""
        runner = {"ols": self._fit_ols, "iv": self._fit_iv, "ml": self._fit_ml}[
            self.method
        ]
        coefs, cov, resid, extra = runner()
        params, param_cov = _structural(coefs, cov)

        lam = float(params[2])
        if not 0.0 < lam < 1.0:
            warnings.warn(
                f"lambda_hat = {lam:.4f} is outside (0, 1): the geometric "
                "lag structure the model assumes is rejected by the data. "
                "The long-run multiplier, the mean lag and the median lag "
                "are reported as NaN, because they are not defined here.",
                DegenerateCaseWarning,
                stacklevel=2,
            )
        elif lam > 0.98:
            warnings.warn(
                f"lambda_hat = {lam:.4f} > 0.98: the implied memory is "
                "nearly infinite and the long-run multiplier "
                "beta0/(1-lambda) is very poorly determined. This is what "
                "a unit root looks like from inside a Koyck model - test "
                "for one (pyardl.unitroot, pyardl.bounds) before reading "
                "the long run.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )
        return KoyckResults(
            model=self,
            _params=params,
            _cov=param_cov,
            _reg_params=coefs,
            _resid=resid,
            extra=extra,
        )


@dataclass(frozen=True)
class KoyckResults:
    r"""Outcome of a :class:`KoyckModel` fit."""

    model: KoyckModel
    _params: FloatArray = field(repr=False)
    _cov: FloatArray = field(repr=False)
    _reg_params: FloatArray = field(repr=False)
    _resid: FloatArray = field(repr=False)
    extra: dict[str, Any] = field(default_factory=dict)

    # -------------------------- basics --------------------------------
    @property
    def params(self) -> pd.Series:
        r"""The structural :math:`(\alpha, \beta_0, \lambda)`."""
        return pd.Series(self._params, index=list(_PARAM_NAMES), name="coef")

    @property
    def bse(self) -> pd.Series:
        return pd.Series(
            np.sqrt(np.diag(self._cov)), index=list(_PARAM_NAMES), name="se"
        )

    @property
    def tvalues(self) -> pd.Series:
        return pd.Series(self._params / self.bse.to_numpy(), index=list(_PARAM_NAMES))

    @property
    def nobs(self) -> int:
        return int(self._resid.shape[0])

    @property
    def resid(self) -> pd.Series:
        return pd.Series(self._resid, name="resid")

    @property
    def lam(self) -> float:
        return float(self._params[2])

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Normal confidence intervals for the three parameters."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}.")
        z = float(stats.norm.ppf(1 - alpha / 2))
        se = self.bse.to_numpy()
        return pd.DataFrame(
            {"lower": self._params - z * se, "upper": self._params + z * se},
            index=list(_PARAM_NAMES),
        )

    # -------------------------- multipliers ---------------------------
    @property
    def _geometric(self) -> bool:
        return bool(0.0 < self.lam < 1.0)

    @property
    def impact_multiplier(self) -> float:
        r"""Effect of a unit rise in ``x`` within the period: :math:`\beta_0`."""
        return float(self._params[1])

    @property
    def longrun_multiplier(self) -> float:
        r""":math:`\beta_0 / (1 - \lambda)`, or ``nan`` outside (0, 1)."""
        if not self._geometric:
            return float("nan")
        return float(self._params[1] / (1.0 - self.lam))

    @property
    def mean_lag(self) -> float:
        r""":math:`\lambda / (1 - \lambda)`, the weighted average delay."""
        if not self._geometric:
            return float("nan")
        return float(self.lam / (1.0 - self.lam))

    @property
    def median_lag(self) -> float:
        r""":math:`\ln(0.5) / \ln(\lambda)`: when half the effect has landed."""
        if not self._geometric:
            return float("nan")
        return float(np.log(0.5) / np.log(self.lam))

    def _multiplier_se(self, name: str) -> float:
        """Delta-method standard error of one derived quantity.

        The gradients are analytical. A numerical gradient would be
        available through ``pyardl.utils._delta_method`` and the tests
        check the two agree; the closed forms are used here because they
        are exact and cost nothing.
        """
        if not self._geometric:
            return float("nan")
        beta0, lam = float(self._params[1]), self.lam
        grad = np.zeros(3, dtype=np.float64)
        if name == "impact":
            grad[1] = 1.0
        elif name == "longrun":
            grad[1] = 1.0 / (1.0 - lam)
            grad[2] = beta0 / (1.0 - lam) ** 2
        elif name == "mean_lag":
            grad[2] = 1.0 / (1.0 - lam) ** 2
        elif name == "median_lag":
            grad[2] = -np.log(0.5) / (lam * np.log(lam) ** 2)
        else:  # pragma: no cover - guarded by the caller
            raise KeyError(name)
        return float(np.sqrt(grad @ self._cov @ grad))

    def interim_multiplier(self, h: int) -> float:
        r"""Cumulated effect after ``h`` periods:
        :math:`\beta_0 (1 - \lambda^{h+1}) / (1 - \lambda)`."""
        if h < 0:
            raise ValueError(f"h must be non-negative, got {h}.")
        if not self._geometric:
            return float("nan")
        lam = self.lam
        return float(self._params[1] * (1.0 - lam ** (h + 1)) / (1.0 - lam))

    def lag_weights(self, h: int) -> pd.Series:
        r"""The weights themselves: :math:`\beta_i = \beta_0 \lambda^i`."""
        if h < 0:
            raise ValueError(f"h must be non-negative, got {h}.")
        lam = self.lam
        weights = self._params[1] * lam ** np.arange(h + 1)
        return pd.Series(weights, index=pd.RangeIndex(h + 1, name="lag"), name="beta")

    def multipliers(self) -> pd.DataFrame:
        """The four derived quantities with their delta-method errors."""
        rows = {
            "impact": (self.impact_multiplier, self._multiplier_se("impact")),
            "longrun": (self.longrun_multiplier, self._multiplier_se("longrun")),
            "mean_lag": (self.mean_lag, self._multiplier_se("mean_lag")),
            "median_lag": (self.median_lag, self._multiplier_se("median_lag")),
        }
        return pd.DataFrame(
            {
                "value": [v for v, _ in rows.values()],
                "se": [s for _, s in rows.values()],
            },
            index=list(rows),
        )

    # -------------------------- diagnostics ---------------------------
    def diagnostics(self) -> pd.DataFrame:
        r"""Residual tests, with Durbin's h in place of Durbin-Watson.

        The Durbin-Watson statistic is **biased towards 2** — that is,
        towards "no autocorrelation" — when a lagged dependent variable
        sits among the regressors, which is precisely the Koyck case.
        Durbin's `h` replaces it.

        `h` is undefined when ``n * var(lambda_hat) >= 1``: the square
        root turns negative. That is not a rare pathology, it happens
        whenever lambda is imprecisely estimated. The method then falls
        back to Durbin's alternative test — regress the residuals on
        their own lag and the regressors, read the `t` — and the index
        says which of the two was used, rather than leaving the reader
        to guess from a NaN.

        **Read the autocorrelation tests against the right null.** Under
        ``"ols"`` and ``"iv"`` the residuals are :math:`u_t`, which is
        MA(1) *by construction* — that is what the Koyck transformation
        does to the error. Rejecting white noise there confirms the
        model rather than contradicting it. On the DGP of the class
        docstring, Ljung-Box gives ``p = 0.0000`` for IV and
        ``p = 0.9025`` for ML.

        Only under ``"ml"`` are the residuals the structural
        :math:`arepsilon_t`, and only there does a rejection mean
        something is wrong: it says the cross-restriction "the MA(1)
        coefficient equals :math:`-\lambda`" does not hold, which is
        the geometric assumption itself failing.
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
        from statsmodels.stats.stattools import jarque_bera

        resid = self._resid
        n = resid.shape[0]
        var_lam = float(self._cov[2, 2])

        rho1 = float(np.corrcoef(resid[1:], resid[:-1])[0, 1])
        denom = 1.0 - n * var_lam
        if denom > 0.0:
            h_stat = rho1 * np.sqrt(n / denom)
            h_p = 2.0 * float(stats.norm.sf(abs(h_stat)))
            h_name = "durbin_h"
        else:
            _, design, _ = self.model._design()
            aug = np.column_stack([design[1:], resid[:-1]])
            coefs, cov, _ = _ols(resid[1:], aug)
            t_stat = float(coefs[-1] / np.sqrt(cov[-1, -1]))
            h_stat = t_stat
            h_p = 2.0 * float(stats.t.sf(abs(t_stat), n - 1 - aug.shape[1]))
            h_name = "durbin_alternative"

        lb = acorr_ljungbox(resid, lags=[max(1, min(10, n // 5))])
        jb_stat, jb_p, _, _ = jarque_bera(resid)
        _, design, _ = self.model._design()
        bp = het_breuschpagan(resid, design)

        return pd.DataFrame(
            {
                "statistic": [h_stat, float(lb["lb_stat"].iloc[0]), jb_stat, bp[0]],
                "pvalue": [h_p, float(lb["lb_pvalue"].iloc[0]), jb_p, bp[1]],
            },
            index=[h_name, "ljung_box", "jarque_bera", "breusch_pagan"],
        )

    # -------------------------- forecast ------------------------------
    def forecast(self, x_future: npt.ArrayLike) -> pd.DataFrame:
        r"""Recursive forecast, with the MA(1) forecast variance.

        The point forecast iterates
        :math:`\hat y_{T+h} = \hat\alpha(1-\hat\lambda)
        + \hat\beta_0 x_{T+h} + \hat\lambda \hat y_{T+h-1}` from the last
        observed ``y``.

        The interval accounts for the error being MA(1), not white
        noise: at ``h = 1`` its variance is
        :math:`\sigma^2 (1 + \lambda^2)`, and beyond that the recursion
        carries the past forecast error forward,
        :math:`v_h = \sigma^2(1+\lambda^2) + \lambda^2 v_{h-1}`. Treating
        it as white noise would understate the interval by exactly the
        term the Koyck transformation created.

        Parameters
        ----------
        x_future : array-like, shape (H,)
            The path of ``x`` over the forecast horizon. Required: there
            is nothing sensible to assume in its place.
        """
        path = np.atleast_1d(np.asarray(x_future, dtype=np.float64))
        if path.ndim != 1 or path.size == 0:
            raise ValueError(
                "x_future must be a non-empty 1-D path for the regressor: "
                "a Koyck forecast is conditional on it, and there is no "
                "defensible default."
            )
        alpha, beta0, lam = (float(v) for v in self._params)
        sigma2 = float(self._resid @ self._resid) / (self.nobs - 3)

        last_y = float(self.model._y[-1])
        point = np.empty(path.size, dtype=np.float64)
        var = np.empty(path.size, dtype=np.float64)
        previous, prev_var = last_y, 0.0
        for h, x_h in enumerate(path):
            previous = alpha * (1.0 - lam) + beta0 * x_h + lam * previous
            prev_var = sigma2 * (1.0 + lam**2) + lam**2 * prev_var
            point[h] = previous
            var[h] = prev_var
        se = np.sqrt(var)
        return pd.DataFrame(
            {
                "forecast": point,
                "se": se,
                "lower": point - 1.959963984540054 * se,
                "upper": point + 1.959963984540054 * se,
            },
            index=pd.RangeIndex(1, path.size + 1, name="horizon"),
        )

    # -------------------------- bridges -------------------------------
    def to_ardl(self) -> Any:
        """The equivalent ARDL(1, 0), fitted on the same data.

        The Koyck transformation *is* an ARDL(1, 0) with an intercept, so
        this is not an approximation: with ``method="ols"`` the two give
        the same three regression coefficients to machine precision, and
        the test suite checks it. What the ARDL object cannot know is
        that the error is MA(1) by construction — which is the whole
        reason ``"ols"`` is the wrong estimator here.
        """
        from pyardl.core.ardl import ARDL

        y = pd.Series(self.model._y, name=self.model._y_name)
        x = pd.DataFrame({self.model._x_name: self.model._x})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyardlMethodologyWarning)
            return ARDL(y, x, order=(1, 0), det="const").fit()

    def summary(self) -> str:
        """Publication-style report of the fit."""
        method = {
            "ols": "OLS (INCONSISTENT - see the warning)",
            "iv": "IV, instrument x_(t-1) (Liviatan)",
            "ml": "conditional ML with the MA(1) cross-restriction",
        }[self.model.method]
        lines = [
            f"Koyck geometric distributed lag - {self.nobs} observations",
            f"  method: {method}",
        ]
        if "first_stage_f" in self.extra:
            first_stage = self.extra["first_stage_f"]
            lines.append(f"  first-stage F on x_(t-1): {first_stage:.2f}")
        if "converged" in self.extra:
            lines.append(f"  converged: {self.extra['converged']}")
        lines += [
            "",
            f"{'':>14}{'coef':>12}{'se':>12}{'t':>10}",
        ]
        for name in _PARAM_NAMES:
            lines.append(
                f"    {name:<10}{self.params[name]:>12.4f}"
                f"{self.bse[name]:>12.4f}{self.tvalues[name]:>10.3f}"
            )
        lines += ["", f"{'':>14}{'value':>12}{'se':>12}"]
        for name, row in self.multipliers().iterrows():
            lines.append(f"    {name:<10}{row['value']:>12.4f}{row['se']:>12.4f}")
        return "\n".join(lines)
