r"""Bayesian ARDL: Minnesota prior on short-run dynamics, posterior long-run.

Every other estimator in the library is frequentist: coefficients are
point estimates, uncertainty comes from the asymptotic normal law or
from bootstrap resampling (specs 14/16). This module answers a
different, complementary need:

- incorporating **prior information** that distant lags matter less
  than recent ones — the Minnesota prior (Litterman 1986), designed for
  BVARs and adapted here to the single-equation UECM structure;
- obtaining a **full posterior distribution** of the long-run
  coefficient :math:`\theta = -\gamma/\lambda` by direct simulation
  (every posterior draw of :math:`(\gamma,\lambda)` gives a draw of
  :math:`\theta`), instead of the delta method's first-order
  approximation (:meth:`pyardl.core.ardl.ARDLResults.longrun`) — a
  distribution that can reveal thick tails the delta method's normal
  approximation cannot, particularly when :math:`\lambda` is weakly
  identified (close to zero).

Like specs 30/34/36/41, "Bayesian ARDL" has no single founding paper —
this module adapts the Minnesota prior to the ARDL/UECM structure the
rest of the library uses.

**Prior scope, deliberately narrow.** Only the short-run dynamic terms
(:math:`\psi_i` on :math:`\Delta y_{t-i}`, :math:`\omega_{j,i}` on
:math:`\Delta x_{j,t-i}`) carry the informative Minnesota prior, with
variance shrinking as :math:`\tau^2/i^d`. The deterministic terms, the
error-correction coefficient :math:`\lambda` and the long-run level
regressors :math:`x_{j,t-1}` get a diffuse (weakly informative, large
fixed variance) prior — the same non-penalisation discipline as
regularisation (:mod:`pyardl.regularized`, spec 41 Section 2.1), for the
same reason: never let a short-run shrinkage choice bias the long-run
reading.

**Deviation from classical Minnesota (documented, not silent).** The
classical Minnesota prior for VARs centers the coefficient on a
variable's own first lag at 1 (a random-walk prior). Here the
persistence role is already carried entirely by :math:`\lambda`
(diffuse prior, never shrunk), and :math:`\Delta y_{t-i}` represents
short-run noise around the error-correction mechanism, not the level's
own persistence — there is no analogous "first own lag" to center at 1.
Every informative coefficient is centered at prior mean zero instead.
See ``docs/DEVIATIONS.md``.

Estimation is closed-form (conjugate normal-inverse-gamma posterior
under a Gaussian likelihood and independent normal priors per
coefficient) — no MCMC, consistent with the spec's "implement the
closed form first" guidance. The posterior mean is obtained by an
**augmented least squares** solve (stacking the prior's precision as
extra pseudo-observations and calling ``numpy.linalg.lstsq``), the same
discipline as the rest of the library's rule against ``inv(X'X)`` for a
main estimation step; only the small ``k x k`` triangular factor from
that same QR decomposition is reused (not inverted) to draw posterior
samples.

References
----------
.. [1] Litterman, R. B. (1986). Forecasting with Bayesian vector
       autoregressions: five years of experience. *Journal of Business
       & Economic Statistics*, 4(1), 25-38.
.. [2] Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and
       conditional projection using realistic prior distributions.
       *Econometric Reviews*, 3(1), 1-100.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import special
from scipy.linalg import solve_triangular

from pyardl.core.ardl import ARDL
from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

Det = Literal["none", "const", "trend"]
TauChoice = float | Literal["cv", "evidence"]

__all__ = ["BayesianARDL", "BayesianARDLResults"]

_DIFFUSE_VARIANCE = 1.0e8


def _build_design(
    y: FloatArray, x: FloatArray, p: int, q: int, det: Det
) -> tuple[FloatArray, FloatArray, list[str], FloatArray, FloatArray]:
    """UECM design at order (p, q); returns design, target, names, and two masks.

    Mirrors :func:`pyardl.regularized.model._build_design`'s column
    layout (deterministic terms, ``y.L1``, ``x{j}.L1`` levels, then the
    short-run ``D.y.Li`` / ``D.x{j}.Li`` terms) but with the Bayesian
    module's own, narrower notion of "informative": only the short-run
    terms are candidates for the Minnesota shrinkage (spec 42 Section
    2.1) — the level regressors ``x{j}.L1`` are diffuse here, unlike in
    :mod:`pyardl.regularized` where they are shrinkage candidates too
    (spec 41 Section 2.1 explicitly regularises them).

    Returns
    -------
    design, target, names : as usual
    is_informative : 1.0 for short-run terms, 0.0 otherwise
    lag_index : the lag ``i`` for informative terms (for ``tau^2/i^d``),
        0.0 for diffuse terms (unused)
    """
    k = x.shape[1]
    start = max(p, q, 1)
    n = y.shape[0]
    dy = np.diff(y)
    dx = np.diff(x, axis=0)
    target = dy[start - 1 :]

    cols: list[FloatArray] = []
    names: list[str] = []
    informative: list[float] = []
    lag_index: list[float] = []

    if det in ("const", "trend"):
        cols.append(np.ones_like(target))
        names.append("const")
        informative.append(0.0)
        lag_index.append(0.0)
    if det == "trend":
        cols.append(np.arange(1, target.shape[0] + 1, dtype=np.float64))
        names.append("trend")
        informative.append(0.0)
        lag_index.append(0.0)

    cols.append(y[start - 1 : n - 1])
    names.append("y.L1")
    informative.append(0.0)
    lag_index.append(0.0)

    for j in range(k):
        cols.append(x[start - 1 : n - 1, j])
        names.append(f"x{j}.L1")
        informative.append(0.0)
        lag_index.append(0.0)

    for i in range(1, p):
        cols.append(dy[start - i - 1 : n - i - 1])
        names.append(f"D.y.L{i}")
        informative.append(1.0)
        lag_index.append(float(i))
    for j in range(k):
        for i in range(q):
            cols.append(dx[start - i - 1 : n - i - 1, j])
            names.append(f"D.x{j}.L{i + 1}")
            informative.append(1.0)
            lag_index.append(float(i + 1))

    design = np.column_stack(cols)
    return (
        design,
        target,
        names,
        np.array(informative, dtype=np.float64),
        np.array(lag_index, dtype=np.float64),
    )


def _prior_variance(
    is_informative: FloatArray, lag_index: FloatArray, tau: float, decay: float
) -> FloatArray:
    """Diagonal prior variance: ``tau^2/i^decay`` informative, else diffuse."""
    variance = np.where(
        is_informative > 0.0,
        tau**2 / np.power(np.maximum(lag_index, 1.0), decay),
        _DIFFUSE_VARIANCE,
    )
    return np.asarray(variance, dtype=np.float64)


def _posterior_fit(
    design: FloatArray,
    target: FloatArray,
    v0: FloatArray,
    a0: float,
    b0: float,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Conjugate normal-inverse-gamma posterior, prior mean zero, diagonal V0.

    Returns ``(posterior_mean, r_factor, an, bn)`` where ``r_factor`` is
    the upper-triangular QR factor of the augmented design (so that
    ``V0^{-1} + X'X = r_factor' @ r_factor``) — reused both to draw
    posterior samples and to score the marginal likelihood, without
    ever forming or inverting ``X'X`` directly (CLAUDE.md rule 1).
    """
    n, k = design.shape
    prior_std_inv = 1.0 / np.sqrt(v0)
    augmented_x = np.vstack([design, np.diag(prior_std_inv)])
    augmented_y = np.concatenate([target, np.zeros(k)])

    posterior_mean, _, _, _ = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)
    _, r_factor = np.linalg.qr(augmented_x)

    resid = target - design @ posterior_mean
    prior_penalty = np.sum((posterior_mean**2) / v0)
    an = a0 + n / 2.0
    bn = b0 + 0.5 * (float(resid @ resid) + float(prior_penalty))
    return posterior_mean, r_factor, an, bn


def _log_marginal_likelihood(
    design: FloatArray,
    target: FloatArray,
    v0: FloatArray,
    a0: float,
    b0: float,
) -> float:
    """Log evidence of the conjugate NIG model, for hyperparameter selection."""
    n = design.shape[0]
    _, r_factor, an, bn = _posterior_fit(design, target, v0, a0, b0)
    log_det_v0inv = -float(np.sum(np.log(v0)))
    log_det_vninv = 2.0 * float(np.sum(np.log(np.abs(np.diag(r_factor)))))
    return float(
        -n / 2.0 * np.log(2.0 * np.pi)
        + 0.5 * log_det_v0inv
        - 0.5 * log_det_vninv
        + a0 * np.log(b0)
        - an * np.log(bn)
        + float(special.gammaln(an))
        - float(special.gammaln(a0))
    )


def _rolling_cv_error(
    y: FloatArray,
    x: FloatArray,
    p: int,
    q: int,
    det: Det,
    tau: float,
    decay: float,
    a0: float,
    b0: float,
    n_splits: int,
    horizon: int,
) -> float:
    """Mean squared one-step forecast error at the posterior mean, rolling-origin."""
    design, target, _, is_informative, lag_index = _build_design(y, x, p, q, det)
    v0 = _prior_variance(is_informative, lag_index, tau, decay)
    n_obs = design.shape[0]
    min_train = max(20, design.shape[1] + 5)
    if n_obs - min_train < n_splits * horizon:
        n_splits = max(1, (n_obs - min_train) // max(horizon, 1))
    if n_splits < 1:
        return np.inf

    errors = []
    step = max((n_obs - min_train) // n_splits, 1)
    for s in range(n_splits):
        t_end = min_train + s * step
        if t_end + horizon > n_obs:
            break
        posterior_mean, _, _, _ = _posterior_fit(
            design[:t_end], target[:t_end], v0, a0, b0
        )
        pred = design[t_end : t_end + horizon] @ posterior_mean
        err = target[t_end : t_end + horizon] - pred
        errors.append(float(np.mean(err**2)))
    return float(np.mean(errors)) if errors else np.inf


@dataclass(frozen=True)
class BayesianARDLResults:
    """Outcome of :class:`BayesianARDL`.

    Attributes
    ----------
    posterior_params : pandas.DataFrame
        ``n_draws`` rows, one column per UECM design term (``const``,
        ``y.L1``, ``x{j}.L1``, ``D.y.Li``, ``D.x{j}.Li``).
    posterior_mean, posterior_std : pandas.Series
    longrun_posterior : pandas.DataFrame
        ``n_draws`` rows, one column per regressor:
        :math:`\\theta^{(s)} = -\\gamma^{(s)}_j/\\lambda^{(s)}`.
    tau, decay : float
        Hyperparameters used for the fit (the selected value when
        ``tau="cv"`` or ``"evidence"``).
    tau_grid, tau_criterion : ndarray or None
        The grid searched and its criterion (CV error or negative log
        evidence) when ``tau`` was not a fixed float.
    names : list of str
        Design term names, in ``posterior_params`` column order.
    x_names : list of str
        Regressor names, in ``longrun_posterior`` column order.
    """

    posterior_params: pd.DataFrame
    posterior_mean: pd.Series
    posterior_std: pd.Series
    longrun_posterior: pd.DataFrame
    tau: float
    decay: float
    names: list[str] = field(repr=False)
    x_names: list[str] = field(repr=False)
    _y: FloatArray = field(repr=False)
    _x: FloatArray = field(repr=False)
    _det: Det = field(repr=False)
    _p: int = field(repr=False)
    _q: int = field(repr=False)
    tau_grid: FloatArray | None = None
    tau_criterion: FloatArray | None = None

    def credible_interval(self, alpha: float = 0.05) -> pd.DataFrame:
        """Empirical ``(alpha/2, 1-alpha/2)`` quantiles, params and long-run theta.

        Returns
        -------
        pandas.DataFrame
            Indexed by term name (design terms first, then
            ``theta.<regressor>``), columns ``lower`` and ``upper``.
        """
        lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
        rows = {}
        for name in self.posterior_params.columns:
            draws = self.posterior_params[name]
            rows[name] = (draws.quantile(lo), draws.quantile(hi))
        for name in self.longrun_posterior.columns:
            draws = self.longrun_posterior[name]
            rows[f"theta.{name}"] = (draws.quantile(lo), draws.quantile(hi))
        return pd.DataFrame.from_dict(rows, orient="index", columns=["lower", "upper"])

    def compare_to_delta_method(self, alpha: float = 0.05) -> pd.DataFrame:
        """Bayesian credible interval vs. delta-method CI, side by side.

        Fits a plain frequentist :class:`pyardl.core.ardl.ARDL` at the
        same order and refers to :attr:`~pyardl.core.ardl.ARDLResults.longrun`
        (spec 03) for the delta-method interval. The two need not agree
        — this is a diagnostic (spec 42 Section 2.3), not a claim that
        either interval is "more correct", and is documented as
        degrading in particular when the adjustment speed
        :math:`\\lambda` is weakly identified.
        """
        x_df = pd.DataFrame(self._x, columns=self.x_names)
        freq = ARDL(self._y, x_df, order=(self._p, self._q), det=self._det).fit()
        longrun = freq.longrun
        z = float(special.ndtri(1.0 - alpha / 2.0))

        bayes_ci = self.credible_interval(alpha)
        rows = []
        for name in self.x_names:
            b_lo = bayes_ci.loc[f"theta.{name}", "lower"]
            b_hi = bayes_ci.loc[f"theta.{name}", "upper"]
            theta_hat = longrun.loc[name, "theta"]
            se = longrun.loc[name, "se"]
            rows.append(
                {
                    "regressor": name,
                    "bayes_lower": b_lo,
                    "bayes_upper": b_hi,
                    "delta_lower": theta_hat - z * se,
                    "delta_upper": theta_hat + z * se,
                }
            )
        return pd.DataFrame(rows).set_index("regressor")

    def summary(self) -> str:
        """Readable report of the posterior."""
        lines = [
            f"Bayesian ARDL (Minnesota prior) - tau={self.tau:.6g}, "
            f"decay={self.decay:.4g}, {self.posterior_params.shape[0]} draws",
            "",
            "  Posterior mean (std):",
        ]
        for name in self.names:
            lines.append(
                f"    {name:<14}{self.posterior_mean[name]: .4f}"
                f"   ({self.posterior_std[name]:.4f})"
            )
        lines.append("\n  Long-run theta, posterior mean (std):")
        for name in self.x_names:
            draws = self.longrun_posterior[name]
            lines.append(f"    {name:<14}{draws.mean(): .4f}   ({draws.std():.4f})")
        return "\n".join(lines)


class BayesianARDL:
    r"""ARDL/UECM with a Minnesota prior on short-run coefficients.

    Parameters
    ----------
    y, x : array_like
        As in :class:`pyardl.core.ardl.ARDL`.
    order : tuple of (int, int)
        ``(p, q)``, the same uniform-lag convention as
        :func:`pyardl.regularized.select_order_regularized` (``q``
        applied to every regressor).
    det : {'none', 'const', 'trend'}, default 'const'
    prior : {'minnesota'}, default 'minnesota'
        The only prior family implemented.
    tau : float, "cv" or "evidence", default "cv"
        Global prior scale. ``"cv"`` selects it by rolling-origin
        cross-validation (mean squared one-step forecast error at the
        posterior mean); ``"evidence"`` maximises the closed-form log
        marginal likelihood. Both search a fixed grid at the given
        ``decay``, never jointly with ``decay`` (a documented scope
        simplification, see ``docs/DEVIATIONS.md``).
    decay : float, default 1.0
        Minnesota decay rate ``d``: informative-term prior variance is
        ``tau^2 / lag^d``.
    n_draws : int, default 5000
        Posterior draws.
    seed : int or numpy.random.Generator, optional
        Seeds :class:`numpy.random.Generator` (CLAUDE.md rule 2).
    a0, b0 : float, default 1e-3
        Weakly informative inverse-gamma hyperparameters for
        :math:`\sigma^2` (a standard diffuse default, not tuned).
    tau_grid_size : int, default 15
    cv_folds, cv_horizon : int
        As in :func:`pyardl.regularized.select_order_regularized`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.bayesian import BayesianARDL
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = y[t - 1] - 0.5 * (y[t - 1] - 1.5 * x.iloc[t - 1])
    ...     y[t] += rng.standard_normal() * 0.3
    >>> res = BayesianARDL(y, pd.DataFrame({"x": x}), order=(1, 1),
    ...                     tau=0.5, n_draws=500, seed=0).fit()
    >>> res.posterior_params.shape[0]
    500
    """

    def __init__(
        self,
        y: ArrayLike,
        x: ArrayLike,
        order: tuple[int, int],
        det: Det = "const",
        prior: Literal["minnesota"] = "minnesota",
        tau: TauChoice = "cv",
        decay: float = 1.0,
        n_draws: int = 5000,
        seed: int | np.random.Generator | None = None,
        a0: float = 1e-3,
        b0: float = 1e-3,
        tau_grid_size: int = 15,
        cv_folds: int = 5,
        cv_horizon: int = 5,
    ) -> None:
        if prior != "minnesota":
            raise ValueError(f"prior={prior!r}: only 'minnesota' is implemented.")
        if det not in ("none", "const", "trend"):
            raise ValueError(f"det={det!r} must be 'none', 'const' or 'trend'.")
        p, q = order
        if p < 1:
            raise ValueError(
                "order[0] (p) must be >= 1: no error-correction term otherwise."
            )

        self.y = y
        self.x = x
        self.p = int(p)
        self.q = int(q)
        self.det: Det = det
        self.tau = tau
        self.decay = float(decay)
        self.n_draws = int(n_draws)
        self.seed = seed
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.tau_grid_size = int(tau_grid_size)
        self.cv_folds = int(cv_folds)
        self.cv_horizon = int(cv_horizon)

    def fit(self) -> BayesianARDLResults:
        """Draw from the conjugate posterior.

        Returns
        -------
        BayesianARDLResults
        """
        y_arr, x_arr, _, _, x_names = check_series(self.y, self.x)
        if x_arr is None:
            raise ValueError("BayesianARDL needs at least one regressor.")

        design, target, names, is_informative, lag_index = _build_design(
            y_arr, x_arr, self.p, self.q, self.det
        )

        scale = float(np.std(target)) * max(
            float(np.std(design[:, is_informative > 0]))
            if np.any(is_informative > 0)
            else 1.0,
            1e-6,
        )
        tau_max = max(scale, 1e-3)
        tau_grid: FloatArray = np.asarray(
            np.geomspace(tau_max * 1e-2, tau_max * 10, self.tau_grid_size),
            dtype=np.float64,
        )

        tau_criterion: FloatArray | None = None
        if self.tau == "cv":
            errors = np.array(
                [
                    _rolling_cv_error(
                        y_arr,
                        x_arr,
                        self.p,
                        self.q,
                        self.det,
                        float(t),
                        self.decay,
                        self.a0,
                        self.b0,
                        self.cv_folds,
                        self.cv_horizon,
                    )
                    for t in tau_grid
                ]
            )
            tau_criterion = errors
            tau_selected = float(tau_grid[int(np.argmin(errors))])
        elif self.tau == "evidence":
            neg_log_ev = np.array(
                [
                    -_log_marginal_likelihood(
                        design,
                        target,
                        _prior_variance(
                            is_informative, lag_index, float(t), self.decay
                        ),
                        self.a0,
                        self.b0,
                    )
                    for t in tau_grid
                ]
            )
            tau_criterion = neg_log_ev
            tau_selected = float(tau_grid[int(np.argmin(neg_log_ev))])
        else:
            tau_selected = float(self.tau)

        v0 = _prior_variance(is_informative, lag_index, tau_selected, self.decay)
        posterior_mean, r_factor, an, bn = _posterior_fit(
            design, target, v0, self.a0, self.b0
        )

        rng = np.random.default_rng(self.seed)
        k = design.shape[1]
        sigma2_draws = bn / rng.gamma(shape=an, scale=1.0, size=self.n_draws)
        z_draws = rng.standard_normal(size=(self.n_draws, k))
        beta_perturb = solve_triangular(r_factor, z_draws.T, lower=False).T
        theta_draws = (
            posterior_mean[np.newaxis, :]
            + beta_perturb * np.sqrt(sigma2_draws)[:, None]
        )

        posterior_params = pd.DataFrame(theta_draws, columns=names)
        posterior_mean_s = posterior_params.mean(axis=0)
        posterior_std_s = posterior_params.std(axis=0)

        idx_lambda = names.index("y.L1")
        lam_draws = theta_draws[:, idx_lambda]
        longrun_cols = {}
        for j, xname in enumerate(x_names):
            idx_gamma = names.index(f"x{j}.L1")
            longrun_cols[xname] = -theta_draws[:, idx_gamma] / lam_draws
        longrun_posterior = pd.DataFrame(longrun_cols)

        return BayesianARDLResults(
            posterior_params=posterior_params,
            posterior_mean=posterior_mean_s,
            posterior_std=posterior_std_s,
            longrun_posterior=longrun_posterior,
            tau=tau_selected,
            decay=self.decay,
            names=names,
            x_names=list(x_names),
            _y=y_arr,
            _x=x_arr,
            _det=self.det,
            _p=self.p,
            _q=self.q,
            tau_grid=tau_grid if self.tau in ("cv", "evidence") else None,
            tau_criterion=tau_criterion,
        )
