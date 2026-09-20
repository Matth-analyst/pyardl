r"""Hansen (1999, 2000) — threshold regression with an estimated threshold.

NARDL (:mod:`pyardl.nardl`) decomposes a regressor around a threshold
**fixed a priori** (zero, by default) on the regressor's own change.
Hansen asks a different question: what *is* the threshold, estimated
from the data rather than chosen by the user? Two models that must not
be confused (a frequent trap in the applied literature):

- **NARDL**: the threshold is on the *change* of a regressor, and the
  regime depends on the sign of :math:`\Delta x_t` itself — asymmetry of
  response to a rise/fall in a regressor.
- **Threshold regression (this module)**: the threshold is on a
  **transition variable** :math:`q_t` (which can be :math:`x` itself, a
  third variable, or a lagged :math:`y` — self-exciting TAR in the
  strict sense), and the regime depends on where :math:`q_{t-d}` falls
  relative to an **unknown** threshold :math:`\gamma`, estimated jointly
  with each regime's coefficients.

Also distinct from Gregory-Hansen / Bai-Perron
(:mod:`pyardl.cointegration`), whose threshold is a *date*, not a
generic transition variable — the three share the same algorithmic
family (grid search plus the "problem of Davies" for inference) without
being substitutable.

References
----------
.. [1] Hansen, B. E. (2000). Sample splitting and threshold estimation.
       *Econometrica*, 68(3), 575-603.
.. [2] Hansen, B. E. (1999). Threshold effects in non-dynamic panels:
       estimation, testing, and inference. *Journal of Econometrics*,
       93(2), 345-368.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyardl.utils import check_regressors_allow_constant, check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

__all__ = ["threshold_ardl", "ThresholdARDLResults"]


@dataclass(frozen=True)
class ThresholdARDLResults:
    """Outcome of a Hansen threshold-regression search.

    Attributes
    ----------
    gamma_hat : float
        Estimated threshold, minimising the pooled residual sum of
        squares over the search grid.
    gamma_ci : tuple of float
        Asymptotic no-rejection confidence region for ``gamma_hat``
        (Hansen 2000), from the shape of ``SSR(gamma)`` near its
        minimum — not a symmetric interval around a point estimate.
    regime_params : tuple of pandas.Series
        OLS coefficients of regime 1 (:math:`q_{t-d} \\le \\hat\\gamma`)
        and regime 2 (:math:`q_{t-d} > \\hat\\gamma`).
    ssr : float
        Residual sum of squares of the two-regime model at
        ``gamma_hat``.
    grid : pandas.DataFrame
        Columns ``gamma`` and ``ssr`` for every candidate threshold
        tested, so the margin of the retained threshold is visible.
    linearity_stat : float
        sup-Wald(-type F) statistic of :math:`H_0: \\beta_1 = \\beta_2`
        (no threshold effect).
    linearity_pvalue : float
        Bootstrap p-value of ``linearity_stat`` (Hansen 1996): residuals
        of the restricted linear fit are resampled with the design held
        fixed, and the same grid search is rerun on each replicate.
    delay : int
        Lag ``d`` of the transition variable.
    trim : float
        Fraction of the transition variable's empirical distribution
        excluded at each end of the search grid.
    n_boot : int
        Bootstrap replications for ``linearity_pvalue``.
    seed : int or None
        Seed of the bootstrap generator.
    """

    gamma_hat: float
    gamma_ci: tuple[float, float]
    regime_params: tuple[pd.Series, pd.Series]
    ssr: float
    grid: pd.DataFrame
    linearity_stat: float
    linearity_pvalue: float
    delay: int
    trim: float
    n_boot: int
    seed: int | None

    def decision(self, alpha: float = 0.05) -> str:
        """``'threshold'`` or ``'linear'`` at ``alpha``, from the bootstrap p-value."""
        return "threshold" if self.linearity_pvalue < alpha else "linear"

    def summary(self) -> str:
        """Readable report of the search."""
        lines = [
            f"Hansen threshold regression (1999, 2000) - delay={self.delay}, "
            f"trim={self.trim}",
            f"  gamma_hat = {self.gamma_hat:.4f}   90% no-rejection region: "
            f"[{self.gamma_ci[0]:.4f}, {self.gamma_ci[1]:.4f}]",
            f"  SSR = {self.ssr:.4f}",
            f"  Linearity test: stat={self.linearity_stat:.4f}  "
            f"p-value={self.linearity_pvalue:.4f} ({self.n_boot} bootstrap reps)",
            f"  decision (5%): {self.decision(0.05)}",
            "  H0: beta1 = beta2 (no threshold effect)",
        ]
        return "\n".join(lines)


def _regime_ssr_and_beta(
    y: FloatArray, x: FloatArray, mask: NDArray[np.bool_]
) -> tuple[float, FloatArray, int]:
    """OLS on one regime; returns SSR, coefficients, and observation count."""
    y_r, x_r = y[mask], x[mask]
    beta, _, _, _ = np.linalg.lstsq(x_r, y_r, rcond=None)
    resid = y_r - x_r @ beta
    return float(resid @ resid), np.asarray(beta, dtype=np.float64), int(mask.sum())


def _grid_search(
    y: FloatArray, x: FloatArray, q: FloatArray, candidates: FloatArray, min_regime: int
) -> tuple[FloatArray, int]:
    """SSR(gamma) over the grid; returns the array and the argmin index."""
    ssr = np.full(candidates.shape[0], np.inf, dtype=np.float64)
    for i, gamma in enumerate(candidates):
        mask1 = q <= gamma
        n1, n2 = int(mask1.sum()), int((~mask1).sum())
        if n1 < min_regime or n2 < min_regime:
            continue
        ssr1, _, _ = _regime_ssr_and_beta(y, x, mask1)
        ssr2, _, _ = _regime_ssr_and_beta(y, x, ~mask1)
        ssr[i] = ssr1 + ssr2
    return ssr, int(np.argmin(ssr))


def _linearity_stat(ssr_linear: float, ssr_threshold: float, n: int, k: int) -> float:
    """sup-F: reduction in SSR from splitting into two regimes."""
    dof = n - 2 * k
    if dof <= 0 or ssr_threshold <= 0:
        return 0.0
    stat = ((ssr_linear - ssr_threshold) / k) / (ssr_threshold / dof)
    return max(float(stat), 0.0)


def threshold_ardl(
    y: ArrayLike,
    x: ArrayLike,
    transition: ArrayLike,
    delay: int = 1,
    trim: float = 0.15,
    n_boot: int = 999,
    seed: int | None = None,
) -> ThresholdARDLResults:
    r"""Estimate a two-regime threshold regression and test linearity.

    Parameters
    ----------
    y : array_like
        Dependent variable, shape ``(T,)``.
    x : array_like
        Regressors, shape ``(T, k)``. No constant is added
        automatically — include one explicitly, as for
        :func:`pyardl.cointegration.bai_perron.bai_perron`.
    transition : array_like
        Transition variable :math:`q_t`, shape ``(T,)``. Common choices:
        a regressor itself, :math:`y_{t-1}` (self-exciting TAR), or a
        third series.
    delay : int, default 1
        Lag ``d``: the regime at time ``t`` is set by
        :math:`q_{t-d}`.
    trim : float, default 0.15
        Fraction of :math:`q`'s empirical distribution excluded at each
        end of the threshold search grid — never search the whole
        support, the same discipline as
        :func:`pyardl.cointegration.gregory_hansen.gregory_hansen` and
        :func:`pyardl.cointegration.bai_perron.bai_perron`.
    n_boot : int, default 999
        Bootstrap replications for the linearity test's p-value.
    seed : int, optional
        Seed for the bootstrap's :class:`numpy.random.Generator`.

    Returns
    -------
    ThresholdARDLResults

    Notes
    -----
    The sup-Wald statistic testing :math:`\beta_1 = \beta_2` has no
    standard asymptotic distribution — :math:`\gamma` is identified only
    under the alternative (the "problem of Davies"), so its critical
    values come from a residual bootstrap under the linear (restricted)
    model (Hansen 1996): the design (:math:`x`, :math:`q`) is held
    fixed, only the linear fit's residuals are resampled.

    ``gamma_ci`` is built from Hansen's (2000) likelihood-ratio-based
    no-rejection region, not a symmetric normal interval — see
    ``docs/QUESTIONS.md`` for the exact constant's verification status.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.threshold import threshold_ardl
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> q = rng.standard_normal(n)
    >>> const = np.ones(n)
    >>> beta = np.where(q < 0.0, 2.0, -1.0)
    >>> y = beta + rng.standard_normal(n) * 0.3
    >>> res = threshold_ardl(y, const.reshape(-1, 1), q, delay=0, n_boot=99, seed=0)
    >>> res.decision(0.05)
    'threshold'
    """
    if not 0.0 < trim < 0.5:
        raise ValueError(f"trim={trim} must be in (0, 0.5).")
    if delay < 0:
        raise ValueError(f"delay={delay} must be non-negative.")

    y_arr, _, _, _, _ = check_series(y, None)
    x_arr, x_names = check_regressors_allow_constant(x, y_arr.shape[0])
    q_arr = np.asarray(transition, dtype=np.float64).ravel()
    if q_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"transition must have the same length as y: got {q_arr.shape[0]} "
            f"against {y_arr.shape[0]}."
        )

    if delay > 0:
        y_arr = y_arr[delay:]
        x_arr = x_arr[delay:]
        q_use = q_arr[:-delay]
    else:
        q_use = q_arr
    n = y_arr.shape[0]
    k = x_arr.shape[1]

    min_regime = max(int(np.ceil(trim * n)), k + 1)
    sorted_q = np.sort(np.unique(q_use))
    lo = int(np.floor(trim * sorted_q.size))
    hi = int(np.ceil((1.0 - trim) * sorted_q.size))
    candidates = sorted_q[lo:hi]
    if candidates.size == 0:
        raise ValueError(
            "trim leaves no candidate threshold; lower trim or use more data."
        )

    ssr_grid, best_idx = _grid_search(y_arr, x_arr, q_use, candidates, min_regime)
    if not np.isfinite(ssr_grid[best_idx]):
        raise ValueError(
            "No candidate threshold leaves both regimes with at least "
            f"{min_regime} observations. Lower trim or use more data."
        )
    gamma_hat = float(candidates[best_idx])
    ssr_threshold = float(ssr_grid[best_idx])

    mask1 = q_use <= gamma_hat
    ssr1, beta1, _ = _regime_ssr_and_beta(y_arr, x_arr, mask1)
    ssr2, beta2, _ = _regime_ssr_and_beta(y_arr, x_arr, ~mask1)
    regime_params = (
        pd.Series(beta1, index=x_names, name="regime1"),
        pd.Series(beta2, index=x_names, name="regime2"),
    )

    beta_linear, _, _, _ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
    resid_linear = y_arr - x_arr @ beta_linear
    ssr_linear = float(resid_linear @ resid_linear)
    fitted_linear = x_arr @ beta_linear

    linearity_stat = _linearity_stat(ssr_linear, ssr_threshold, n, k)

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        y_b = fitted_linear + rng.choice(resid_linear, size=n, replace=True)
        beta_b, _, _, _ = np.linalg.lstsq(x_arr, y_b, rcond=None)
        resid_b = y_b - x_arr @ beta_b
        ssr_linear_b = float(resid_b @ resid_b)
        ssr_grid_b, best_idx_b = _grid_search(y_b, x_arr, q_use, candidates, min_regime)
        if np.isfinite(ssr_grid_b[best_idx_b]):
            boot_stats[b] = _linearity_stat(
                ssr_linear_b, float(ssr_grid_b[best_idx_b]), n, k
            )
        else:
            boot_stats[b] = 0.0
    linearity_pvalue = float((boot_stats >= linearity_stat).mean())

    # Hansen (2000) LR-based no-rejection region: gamma is retained at
    # 100(1-alpha)% confidence when LR(gamma) <= c(alpha), c derived
    # from the limiting distribution of the LR sequence (see module
    # notes / docs/QUESTIONS.md for its verification status).
    sigma2 = ssr_threshold / n
    lr = (ssr_grid - ssr_threshold) / sigma2
    c_alpha = -2.0 * np.log(1.0 - np.sqrt(1.0 - 0.10))
    in_region = candidates[lr <= c_alpha]
    gamma_ci = (
        (float(in_region.min()), float(in_region.max()))
        if in_region.size
        else (gamma_hat, gamma_hat)
    )

    grid = pd.DataFrame({"gamma": candidates, "ssr": ssr_grid})

    return ThresholdARDLResults(
        gamma_hat=gamma_hat,
        gamma_ci=gamma_ci,
        regime_params=regime_params,
        ssr=ssr_threshold,
        grid=grid,
        linearity_stat=linearity_stat,
        linearity_pvalue=linearity_pvalue,
        delay=delay,
        trim=trim,
        n_boot=n_boot,
        seed=seed,
    )
