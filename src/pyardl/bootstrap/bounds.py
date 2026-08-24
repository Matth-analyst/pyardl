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
from pyardl.bounds.classification import Classification, classify
from pyardl.bounds.pss import _wald_f_indep, bounds_test
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
    f_stat, t_stat, f_indep_stat : float
        The three observed statistics, identical to those of the
        classical test on the same specification.
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
    f_indep_stat: float
    f_critical: dict[float, float]
    t_critical: dict[float, float]
    f_indep_critical: dict[float, float]
    f_pvalue: float
    t_pvalue: float
    f_indep_pvalue: float
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

    def decision_indep(self, alpha: float = 0.05) -> str:
        """Verdict of the third test: the regressors' levels."""
        return (
            "cointegration"
            if self.f_indep_stat > self.f_indep_critical[alpha]
            else "no_cointegration"
        )

    def classification(self, alpha: float = 0.05) -> tuple[Classification, str]:
        """Three-test verdict of Sam, McNown & Goh (2019), and its reason.

        The bootstrap leaves no inconclusive zone, so only four outcomes
        can occur here: ``cointegration``, ``degenerate_1``,
        ``degenerate_2`` and ``no_cointegration`` — plus
        ``inconclusive`` on the contradictory combinations, which stay
        possible because three tests can disagree even when each is
        decisive on its own.
        """
        return classify(
            self.decision_f(alpha),
            self.decision_t(alpha),
            self.decision_indep(alpha),
        )

    def comparison(self, alpha: float = 0.05) -> pd.DataFrame:
        """The three tests, bootstrap against classical bounds.

        Spec 16 §2.3: the two routes are reported side by side, because
        a disagreement between them is itself a result. The bootstrap
        leaves no inconclusive zone; the bounds do, and the row that
        differs is the one worth looking at.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level of both verdicts.

        Returns
        -------
        pandas.DataFrame
            One row per test: the statistic, the bootstrap critical
            value, p-value and verdict, then the classical I(0)/I(1)
            bounds and their verdict. ``NaN`` and ``"unavailable"``
            where the classical route tabulates nothing.
        """
        cls = self.classical
        rows = []
        for _name, stat, crit, pval, boot_dec, lo_col, up_col, cls_dec in (
            (
                "F_overall",
                self.f_stat,
                self.f_critical[alpha],
                self.f_pvalue,
                self.decision_f(alpha),
                "F_I0",
                "F_I1",
                cls.decision_f,
            ),
            (
                "t_BDM",
                self.t_stat,
                self.t_critical[alpha],
                self.t_pvalue,
                self.decision_t(alpha),
                "t_I0",
                "t_I1",
                cls.decision_t,
            ),
            (
                "F_indep",
                self.f_indep_stat,
                self.f_indep_critical[alpha],
                self.f_indep_pvalue,
                self.decision_indep(alpha),
                "F_indep_I0",
                "F_indep_I1",
                cls.decision_indep,
            ),
        ):
            lo = float(cls.bounds.loc[alpha, lo_col])
            up = float(cls.bounds.loc[alpha, up_col])
            rows.append(
                {
                    "statistic": float(stat),
                    "boot_cv": float(crit),
                    "boot_p": float(pval),
                    "boot_decision": boot_dec,
                    "bound_I0": lo,
                    "bound_I1": up,
                    "bound_decision": (
                        cls_dec if cls_dec is not None else "unavailable"
                    ),
                }
            )
        frame = pd.DataFrame(rows, index=["F_overall", "t_BDM", "F_indep"])
        frame.index.name = f"alpha={alpha}"
        return frame

    def agrees_with_bounds(self, alpha: float = 0.05) -> bool:
        """Whether the two routes reach the same classification.

        A disagreement is not an error: the bootstrap has no
        inconclusive zone and the bounds do, so the two can legitimately
        differ. It is a reason to report both rather than to pick one.
        """
        return self.classification(alpha)[0] == self.classical.classification()[0]

    def summary(self) -> str:
        """Readable report, with the classical bounds for comparison."""
        p, q = self.order
        lines = [
            f"Bootstrap bounds test (McNown, Sam & Goh 2018) - case "
            f"{self.case}, B={self.n_boot}, resample='{self.resample}', "
            f"seed={self.seed}"
            + ("" if self.classical.conditional else ", UNCONDITIONAL"),
            "",
            f"F_overall = {self.f_stat:.4f}   bootstrap p = {self.f_pvalue:.4f}"
            f"   decision (5%): {self.decision_f(0.05)}",
            f"t_BDM     = {self.t_stat:.4f}   bootstrap p = {self.t_pvalue:.4f}"
            f"   decision (5%): {self.decision_t(0.05)}",
            f"F_indep   = {self.f_indep_stat:.4f}   bootstrap p = "
            f"{self.f_indep_pvalue:.4f}   decision (5%): "
            f"{self.decision_indep(0.05)}",
            "",
        ]
        label, reason = self.classification(0.05)
        lines += [
            f"  CLASSIFICATION (5%): {label}",
            f"  {reason}",
            "",
            "  bootstrap critical values",
            f"  {'alpha':>7}{'F':>12}{'t':>12}{'F_indep':>12}",
        ]
        for a in _ALPHAS:
            lines.append(
                f"  {a:>7}{self.f_critical[a]:>12.4f}{self.t_critical[a]:>12.4f}"
                f"{self.f_indep_critical[a]:>12.4f}"
            )
        # Spec 16 §2.3 — les deux routes en regard, test par test.
        comp = self.comparison(0.05)
        lines += [
            "",
            "  bootstrap against classical bounds (5%)",
            f"  {'test':>10}{'stat':>10}{'boot cv':>10}{'boot p':>9}"
            f"{'I(0)':>9}{'I(1)':>9}  {'boot':<17}{'bounds':<17}",
        ]
        for name, row in comp.iterrows():
            lo = (
                "     -   " if np.isnan(row["bound_I0"]) else f"{row['bound_I0']:>9.3f}"
            )
            up = (
                "     -   " if np.isnan(row["bound_I1"]) else f"{row['bound_I1']:>9.3f}"
            )
            lines.append(
                f"  {name:>10}{row['statistic']:>10.4f}{row['boot_cv']:>10.4f}"
                f"{row['boot_p']:>9.4f}{lo}{up}  "
                f"{row['boot_decision']:<17}{row['bound_decision']:<17}"
            )
        cls_label = self.classical.classification()[0]
        lines += [
            "",
            f"  classification: bootstrap -> {label}, bounds -> {cls_label}"
            + ("" if self.agrees_with_bounds(0.05) else "   (THEY DISAGREE)"),
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
    conditional: bool = True,
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

    classical = bounds_test(
        y,
        x,
        case=case,
        order=order,
        conditional=conditional,
        **kwargs,  # type: ignore[arg-type]
    )

    y_arr, x_arr, _, y_name, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("The bootstrap bounds test needs at least one regressor.")

    # BoundsTestResults.order carries q as a dict keyed by regressor name;
    # the estimator wants a tuple aligned on the column order of x.
    p_order, q_map = classical.order
    q_order = tuple(int(q_map[name]) for name in x_names)

    dgp = estimate_null_dgp(
        y_arr,
        x_arr,
        p=p_order,
        q=q_order,
        case=case,
        var_order=var_order,
        conditional=conditional,
    )

    n_obs = y_arr.shape[0]
    f_star = np.empty(n_boot, dtype=np.float64)
    t_star = np.empty(n_boot, dtype=np.float64)
    i_star = np.empty(n_boot, dtype=np.float64)
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
        f_block, t_block, i_block, _, ok = batch_uecm_statistics(
            y_block, x_block, p_order, q_order, case, conditional
        )
        n_ok = int(ok.sum())
        f_star[kept : kept + n_ok] = f_block[ok]
        t_star[kept : kept + n_ok] = t_block[ok]
        i_star[kept : kept + n_ok] = i_block[ok]
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
    i_kept = np.asarray(i_star[:kept], dtype=np.float64)

    if n_failed:
        warnings.warn(
            f"{n_failed} of {n_boot} bootstrap replications were discarded as "
            "unestimable. The critical values rest on the remaining "
            f"{kept}. A large share here means the model is close to "
            "singular on resampled data.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    f_indep_obs = _wald_f_indep(classical._fit)
    f_crit = {a: float(np.quantile(f_kept, 1.0 - a)) for a in _ALPHAS}
    t_crit = {a: float(np.quantile(t_kept, a)) for a in _ALPHAS}
    i_crit = {a: float(np.quantile(i_kept, 1.0 - a)) for a in _ALPHAS}
    f_p = float((1 + np.sum(f_kept >= classical.f_stat)) / (kept + 1))
    t_p = float((1 + np.sum(t_kept <= classical.t_stat)) / (kept + 1))
    i_p = float((1 + np.sum(i_kept >= f_indep_obs)) / (kept + 1))

    distribution = (
        pd.DataFrame({"F": f_kept, "t": t_kept, "F_indep": i_kept})
        if store_distribution
        else None
    )

    return BootstrapBoundsResults(
        f_stat=classical.f_stat,
        t_stat=classical.t_stat,
        f_indep_stat=f_indep_obs,
        f_critical=f_crit,
        t_critical=t_crit,
        f_indep_critical=i_crit,
        f_pvalue=f_p,
        t_pvalue=t_p,
        f_indep_pvalue=i_p,
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
