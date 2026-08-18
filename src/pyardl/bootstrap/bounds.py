r"""Bootstrap version of the bounds test (McNown, Sam & Goh 2018).

The classical bounds test compares a statistic against two critical
values — one assuming every regressor is I(0), one assuming every
regressor is I(1) — because the true distribution depends on integration
orders nobody knows. When the statistic falls between them the answer is
*inconclusive*, and on short samples that happens often.

The bootstrap removes the ambiguity at its source. Instead of bracketing
the distribution, it **builds it**: regenerate the data many times under
a null that is true by construction, recompute the statistic each time,
and read the critical value off the resulting distribution. Because the
regenerated data inherit the integration orders, the dynamics and the
error covariance of the sample at hand, the critical value is specific
to that sample and there is nothing left to bracket.

What you gain, and what it costs
--------------------------------
The verdict becomes binary and the size is better on short samples. The
cost is computation: every replication re-estimates the model, so the
work is ``B`` times a full fit. The default ``B = 2999`` follows the
article.

The bounds are still reported alongside, because they are what the
literature quotes and because a disagreement between the two is
informative.

References
----------
.. [1] McNown, R., Sam, C. Y. & Goh, S. K. (2018). Bootstrapping the
       autoregressive distributed lag test for cointegration.
       *Applied Economics*, 50(13), 1509-1521.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyardl.bootstrap.batch import batch_uecm_statistics
from pyardl.bootstrap.dgp import estimate_null_dgp, simulate_paths
from pyardl.bootstrap.resample import ResampleScheme, resample_residuals
from pyardl.bounds.pss import bounds_test
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    from pyardl.bounds.pss import BoundsTestResults

    FloatArray = NDArray[np.float64]

__all__ = ["bootstrap_bounds_test", "BootstrapBoundsResults"]

_ALPHAS = (0.10, 0.05, 0.01)

# Block size: bounds the memory held by the regenerated paths
# (chunk x periods x equations x 8 bytes) without giving up the gain from
# vectorising, which saturates well below this.
_CHUNK = 256


@dataclass(frozen=True)
class BootstrapBoundsResults:
    """Outcome of a bootstrap bounds test.

    Attributes
    ----------
    f_stat, t_stat : float
        The observed statistics, identical to those of the classical
        test on the same specification.
    f_critical, t_critical : dict
        Bootstrap critical values by level. The ``t`` bounds are
        lower-tail quantiles, the ``F`` bounds upper-tail.
    f_pvalue, t_pvalue : float
        Bootstrap p-values, ``(1 + #{as extreme}) / (B + 1)``.
    decision_f, decision_t, decision_joint : str
        Binary verdicts — there is no inconclusive zone here.
    classical : BoundsTestResults
        The classical test on the same specification, for comparison.
    n_boot, n_failed : int
        Replications requested and replications discarded because the
        regenerated sample could not be estimated.
    seed, resample, var_order, burn_in : ...
        Everything needed to reproduce the run exactly.
    distribution : pandas.DataFrame or None
        The simulated statistics, when ``store_distribution=True``.
    """

    f_stat: float
    t_stat: float
    f_critical: dict[float, float]
    t_critical: dict[float, float]
    f_pvalue: float
    t_pvalue: float
    classical: BoundsTestResults
    n_boot: int
    n_failed: int
    seed: int
    resample: ResampleScheme
    var_order: int
    burn_in: int
    case: int
    order: tuple[int, tuple[int, ...]]
    distribution: pd.DataFrame | None = field(default=None, repr=False)

    def decision_f(self, alpha: float = 0.05) -> str:
        """``'cointegration'`` or ``'no_cointegration'`` on the F test."""
        return (
            "cointegration"
            if self.f_stat > self.f_critical[alpha]
            else "no_cointegration"
        )

    def decision_t(self, alpha: float = 0.05) -> str:
        """Same, on the left-tailed t test."""
        return (
            "cointegration"
            if self.t_stat < self.t_critical[alpha]
            else "no_cointegration"
        )

    def decision_joint(self, alpha: float = 0.05) -> str:
        """Joint verdict, with the same reading as the classical test.

        Both tests must agree. When the F rejects and the t does not, the
        level terms are jointly significant while the dependent variable
        shows no pull back to equilibrium: that is a degenerate case, not
        cointegration.
        """
        f_rej = self.decision_f(alpha) == "cointegration"
        t_rej = self.decision_t(alpha) == "cointegration"
        if f_rej and t_rej:
            return "cointegration"
        if f_rej and not t_rej:
            return "degenerate_suspicion"
        return "no_cointegration"

    def summary(self) -> str:
        """Readable report, with the classical bounds for comparison."""
        p, q = self.order
        lines = [
            f"Bootstrap bounds test (McNown, Sam & Goh 2018) - case "
            f"{self.case}, B={self.n_boot}, resample='{self.resample}', "
            f"seed={self.seed}",
            "",
            f"F_overall = {self.f_stat:.4f}   bootstrap p = {self.f_pvalue:.4f}"
            f"   decision (5%): {self.decision_f(0.05)}",
            f"t_BDM     = {self.t_stat:.4f}   bootstrap p = {self.t_pvalue:.4f}"
            f"   decision (5%): {self.decision_t(0.05)}",
            f"joint decision (F and t): {self.decision_joint(0.05)}",
            "",
            "  bootstrap critical values",
            f"  {'alpha':>7}{'F':>12}{'t':>12}",
        ]
        for a in _ALPHAS:
            lines.append(
                f"  {a:>7}{self.f_critical[a]:>12.4f}{self.t_critical[a]:>12.4f}"
            )
        classical = self.classical
        lines += [
            "",
            "  for comparison, the classical bounds at 5%: "
            f"F in [{classical.bounds.loc[0.05, 'F_I0']:.3f}, "
            f"{classical.bounds.loc[0.05, 'F_I1']:.3f}]"
            f" -> {classical.decision_f}",
        ]
        if self.n_failed:
            lines.append(
                f"  note: {self.n_failed} replication(s) discarded as unestimable."
            )
        return "\n".join(lines)


def bootstrap_bounds_test(
    y: ArrayLike,
    x: ArrayLike,
    case: int = 3,
    order: tuple[int, int | dict[str, int]] | None = None,
    n_boot: int = 2999,
    resample: ResampleScheme = "iid",
    seed: int | None = None,
    var_order: int = 1,
    burn_in: int = 50,
    store_distribution: bool = False,
    **kwargs: object,
) -> BootstrapBoundsResults:
    r"""Bounds test with bootstrap critical values.

    Parameters
    ----------
    y : array_like
        Dependent variable, shape ``(T,)``.
    x : array_like
        Regressors, shape ``(T, k)``.
    case : int, default 3
        Deterministic case, 1 to 5, as in the classical test.
    order : tuple, optional
        ``(p, q)``. Selected automatically when omitted, by the same
        route as :func:`pyardl.bounds.bounds_test`.
    n_boot : int, default 2999
        Number of bootstrap replications. The article's default.
    resample : {'iid', 'wild'}, default 'iid'
        Residual resampling scheme. Use ``'wild'`` when the residuals
        look heteroskedastic.
    seed : int, optional
        Seed of the generator. **Recorded on the result**, and drawn
        from entropy when omitted so that a run is always reproducible
        after the fact.
    var_order : int, default 1
        Lag order of the marginal VAR that regenerates the regressors.
    burn_in : int, default 50
        Initial periods discarded from each regenerated path.
    store_distribution : bool, default False
        Keep the simulated statistics on the result.
    **kwargs
        Passed through to the classical test (``ic``, ``max_p``,
        ``max_q``, ``fixed_regressors``, ``cv_source``).

    Returns
    -------
    BootstrapBoundsResults

    Warns
    -----
    PyardlMethodologyWarning
        When replications had to be discarded as unestimable, or when
        ``n_boot`` is too small for the requested levels to mean
        anything.

    Notes
    -----
    The p-value is :math:`(1 + \#\{F^* \ge F\}) / (B + 1)`. The added
    unit is not cosmetic: it keeps the p-value strictly positive and
    makes the test exact at levels of the form :math:`(m+1)/(B+1)`.
    Reporting a bootstrap p-value of exactly zero would claim more
    resolution than ``B`` replications can provide.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.bootstrap import bootstrap_bounds_test
    >>> rng = np.random.default_rng(0)
    >>> x = np.cumsum(rng.standard_normal((80, 1)), axis=0)
    >>> y = np.zeros(80)
    >>> for t in range(1, 80):
    ...     y[t] = y[t - 1] - 0.5 * (y[t - 1] - x[t - 1, 0]) + rng.standard_normal()
    >>> res = bootstrap_bounds_test(y, x, n_boot=99, seed=1)
    >>> res.decision_f(0.05)
    'cointegration'
    """
    if n_boot < 99:
        raise ValueError(
            f"n_boot={n_boot} is too small: with fewer than 99 replications "
            "the 1% quantile is not identified."
        )
    if seed is None:
        entropy = np.random.SeedSequence().entropy
        seed = int(entropy) % (2**63) if isinstance(entropy, int) else 0
    rng = np.random.default_rng(seed)

    classical = bounds_test(y, x, case=case, order=order, **kwargs)  # type: ignore[arg-type]

    y_arr, x_arr, _, y_name, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("The bootstrap bounds test needs at least one regressor.")

    # BoundsTestResults.order carries q as a dict keyed by regressor name;
    # the estimator wants a tuple aligned on the column order of x.
    p_order, q_map = classical.order
    q_order = tuple(int(q_map[name]) for name in x_names)

    dgp = estimate_null_dgp(
        y_arr, x_arr, p=p_order, q=q_order, case=case, var_order=var_order
    )

    n_obs = y_arr.shape[0]
    f_star = np.empty(n_boot, dtype=np.float64)
    t_star = np.empty(n_boot, dtype=np.float64)
    kept = 0
    n_failed = 0

    # The innovations are still drawn one replication at a time, in the
    # same order as before: the generator stream is unchanged, so a given
    # seed still yields exactly the same critical values. Drawing is ~2%
    # of the cost; it is the regeneration that is batched.
    n_periods = burn_in + n_obs
    n_eq = 1 + dgp.n_regressors
    chunk = max(1, min(n_boot, _CHUNK))

    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        block = np.empty((size, n_periods, n_eq), dtype=np.float64)
        for i in range(size):
            block[i] = resample_residuals(dgp.residuals, n_periods, rng, resample)

        y_block, x_block = simulate_paths(
            dgp, block, y0=y_arr[0], x0=x_arr[0], burn_in=burn_in
        )
        f_block, t_block, ok = batch_uecm_statistics(
            y_block, x_block, p_order, q_order, case
        )
        n_ok = int(ok.sum())
        f_star[kept : kept + n_ok] = f_block[ok]
        t_star[kept : kept + n_ok] = t_block[ok]
        kept += n_ok
        n_failed += size - n_ok
        done += size

    if kept < n_boot // 2:
        raise ValueError(
            f"Only {kept} of {n_boot} replications could be estimated. The "
            "specification is probably too rich for the sample."
        )
    f_kept = np.asarray(f_star[:kept], dtype=np.float64)
    t_kept = np.asarray(t_star[:kept], dtype=np.float64)

    if n_failed:
        warnings.warn(
            f"{n_failed} of {n_boot} bootstrap replications were discarded as "
            "unestimable. The critical values rest on the remaining "
            f"{kept}. A large share here means the model is close to "
            "singular on resampled data.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    f_crit = {a: float(np.quantile(f_kept, 1.0 - a)) for a in _ALPHAS}
    t_crit = {a: float(np.quantile(t_kept, a)) for a in _ALPHAS}
    f_p = float((1 + np.sum(f_kept >= classical.f_stat)) / (kept + 1))
    t_p = float((1 + np.sum(t_kept <= classical.t_stat)) / (kept + 1))

    distribution = (
        pd.DataFrame({"F": f_kept, "t": t_kept}) if store_distribution else None
    )

    return BootstrapBoundsResults(
        f_stat=classical.f_stat,
        t_stat=classical.t_stat,
        f_critical=f_crit,
        t_critical=t_crit,
        f_pvalue=f_p,
        t_pvalue=t_p,
        classical=classical,
        n_boot=kept,
        n_failed=n_failed,
        seed=seed,
        resample=resample,
        var_order=var_order,
        burn_in=burn_in,
        case=case,
        order=(p_order, q_order),
        distribution=distribution,
    )
