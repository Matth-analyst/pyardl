r"""Mean-Group NARDL: asymmetric long-run coefficients, averaged across individuals.

Spec 30: not a founding paper's method but the combination `pyardl`
retains for what the applied literature calls "Panel NARDL" — NARDL
(:mod:`pyardl.nardl`, Shin, Yu & Greenwood-Nimmo 2014) composed with the
Mean-Group aggregator (:mod:`pyardl.panel.mg`, Pesaran & Smith 1995).

**No new estimator.** Every individual is fit with
:class:`pyardl.nardl.NARDL` exactly as in the single-series case; this
module owns only the panel bookkeeping and the aggregation, reusing
:func:`pyardl.panel.mg._aggregate` unchanged.

The one genuine trap, not present in Mean-Group's symmetric case:
averaging and subtracting do not commute under heterogeneity. Averaging
:math:`\hat\theta^+_i` and :math:`\hat\theta^-_i` **separately** across
individuals, then comparing the two group means, is not the same
quantity as averaging each individual's own
:math:`\hat\theta^+_i - \hat\theta^-_i` and testing whether that average
differs from zero — and the group asymmetry test here needs the correct
one: two independent group means (individuals are independent under the
standard Mean-Group assumption, but :math:`\hat\theta^+_i` and
:math:`\hat\theta^-_i` from the *same* individual are correlated), so
the variance of the difference is the sum of the two group variances,
not the between-individual variance of a pre-differenced quantity.

References
----------
.. [1] Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). Modelling
       asymmetric cointegration and dynamic multipliers in a nonlinear
       ARDL framework. In *Festschrift in Honor of Peter Schmidt*.
.. [2] Pesaran, M. H. & Smith, R. (1995). Estimating long-run
       relationships from dynamic heterogeneous panels. *Journal of
       Econometrics*, 68(1), 79-113.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.nardl.model import NARDL
from pyardl.panel.container import PanelData, panel_from_frame
from pyardl.panel.mg import _aggregate

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from pyardl.nardl.decompose import Threshold
    from pyardl.nardl.model import AsymLags, NARDLResults

__all__ = ["MeanGroupNARDL", "MeanGroupNARDLResults"]

Aggregator = Literal["mean", "median", "trimmed"]
Decomposition = Literal["per_individual", "pooled"]


@dataclass(frozen=True)
class MeanGroupNARDLResults:
    """Mean-Group NARDL estimates, with every individual NARDL fit kept.

    Attributes
    ----------
    longrun_asym : pandas.DataFrame
        One row per decomposed regressor: ``theta_pos``/``theta_neg``
        (the group means, aggregated **separately** — never their
        difference averaged directly, see the module notes),
        ``se_pos``/``se_neg`` (between-individual), and ``diff``,
        ``se_diff``, ``t_diff``, ``pvalue_diff`` for
        :math:`H_0: \\theta^+_{MG} = \\theta^-_{MG}` — the variance of
        the difference of two *independent* group means, i.e. the sum
        of the two group variances (spec 30 §2.3), not a delta method
        on an individual ratio.
    individual : dict
        ``{key: NARDLResults}`` — every individual NARDL fit.
    theta_pos_i, theta_neg_i : pandas.DataFrame
        Individual long-run coefficients that were averaged, one column
        per decomposed regressor.
    lambda_i : pandas.Series
        Individual adjustment speeds.
    panel : PanelData
    aggregator : str
    failed : dict
        ``{key: reason}`` for individuals whose NARDL could not be fitted.
    """

    longrun_asym: pd.DataFrame
    individual: dict[object, NARDLResults] = field(repr=False)
    theta_pos_i: pd.DataFrame = field(repr=False)
    theta_neg_i: pd.DataFrame = field(repr=False)
    lambda_i: pd.Series = field(repr=False)
    panel: PanelData = field(repr=False)
    aggregator: str = "mean"
    n_effective: int = 0
    failed: dict[object, str] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        """Individuals actually averaged (excludes :attr:`failed`)."""
        return int(self.theta_pos_i.shape[0])

    @property
    def non_adjusting(self) -> pd.Index:
        """Individuals whose estimated adjustment speed is not negative.

        Same discipline as :attr:`pyardl.panel.mg.MeanGroupResults.non_adjusting`:
        kept in the average (dropping them would select on the outcome)
        but named, since their :math:`\\hat\\theta^\\pm_i` is not a
        long-run coefficient in the sense being averaged.
        """
        return self.lambda_i.index[self.lambda_i >= 0]

    def share_asymmetric(self, alpha: float = 0.05) -> pd.Series:
        """Share of individuals whose OWN long-run asymmetry is significant.

        Distinguishes "the average asymmetry is non-zero" (what
        :attr:`longrun_asym` answers) from "asymmetry is widespread
        across the panel" (what this answers) — a Mean-Group point
        estimate alone cannot tell the two apart: a panel where half the
        individuals are strongly asymmetric in opposite directions and
        half are exactly symmetric can average to the same
        :attr:`longrun_asym` as one where every individual is mildly
        asymmetric in the same direction.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level of each individual's own
            ``longrun_theta`` Wald test (:meth:`NARDLResults.asymmetry_tests`).

        Returns
        -------
        pandas.Series
            Indexed by regressor, the fraction of individuals (among
            those successfully fitted) whose individual long-run
            asymmetry test rejects at ``alpha``.
        """
        shares: dict[str, float] = {}
        for base in self.longrun_asym.index:
            sig = 0
            for fit in self.individual.values():
                tests = fit.asymmetry_tests()
                row = tests.loc[(base, "longrun_theta")]
                if row["pvalue"] < alpha:
                    sig += 1
            shares[base] = sig / len(self.individual)
        return pd.Series(shares, name="share_asymmetric")

    def heterogeneity(self) -> pd.DataFrame:
        """Inter-individual dispersion of :math:`\\hat\\theta^+_i - \\hat\\theta^-_i`.

        Returns
        -------
        pandas.DataFrame
        """
        diff_i = self.theta_pos_i - self.theta_neg_i
        desc = diff_i.describe().T[["mean", "std", "min", "50%", "max"]]
        desc.columns = ["mean", "sd", "min", "median", "max"]
        return desc

    def summary(self) -> str:
        """Publication-style summary of the group asymmetric estimates."""
        lines = [
            f"Mean-Group NARDL - {self.n_units} individuals, "
            f"aggregator={self.aggregator}",
            f"  dependent: {self.panel.y_name}   "
            f"{'unbalanced' if self.panel.unbalanced else 'balanced'} panel",
            "  standard errors: BETWEEN-individual dispersion "
            "(not pooled within-individual)",
            "",
            "  Asymmetric long-run coefficients",
            f"    {'':<10}{'theta+':>10}{'theta-':>10}{'diff':>10}"
            f"{'t(diff)':>10}{'p(diff)':>10}",
        ]
        for name, row in self.longrun_asym.iterrows():
            lines.append(
                f"    {str(name):<10}{row['theta_pos']:>10.4f}"
                f"{row['theta_neg']:>10.4f}{row['diff']:>10.4f}"
                f"{row['t_diff']:>10.3f}{row['pvalue_diff']:>10.4f}"
            )
        n_bad = len(self.non_adjusting)
        if n_bad:
            lines.append(
                f"  WARNING {n_bad} of {self.n_units} individuals have "
                "lambda_i >= 0: excluded from no group, but their theta_i "
                "is not a long-run coefficient in the sense being averaged."
            )
        if self.failed:
            lines.append(f"  {len(self.failed)} individual(s) could not be fitted:")
            lines.extend(f"    {k!r}: {v}" for k, v in self.failed.items())
        return "\n".join(lines)


class MeanGroupNARDL:
    """Mean-Group estimator for a panel of asymmetric (NARDL) individuals.

    Notes
    -----
    A Monte Carlo measurement (``validation/spec30_montecarlo.py``)
    found the group asymmetry test's p-value (``pvalue_diff`` in
    :attr:`MeanGroupNARDLResults.longrun_asym`) conservative under a
    symmetric DGP (0% rejection at a nominal 5%, N=20, T=100) — the
    variance formula (sum of the two group variances) likely omits a
    non-negligible covariance term between :math:`\\hat\\theta^+_{MG}`
    and :math:`\\hat\\theta^-_{MG}`. See ``docs/QUESTIONS.md``; do not
    treat ``pvalue_diff`` as calibrated without revalidation.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel.
    y : str
    X : sequence of str
    asym : sequence of str
        Regressors to decompose into positive/negative partial sums, as
        in :class:`pyardl.nardl.NARDL`.
    id, time : str
    order : tuple or {'auto'}
        Applied to every individual, as in
        :class:`pyardl.panel.mg.MeanGroup`.
    decomposition : {'per_individual', 'pooled'}, default 'per_individual'
        Whether ``x+``/``x-`` are built from each individual's own
        :math:`\\Delta x_{i,t}` (the default — consistent with MG/PMG
        already estimating individual by individual) or from a single
        pair built on the pooled panel before refiltering by individual
        (relevant only when a *common* threshold has an economic
        meaning; spec 30 §2.2 names this the exception, not the
        default). ``'pooled'`` is **not implemented** in this version —
        see ``docs/DEVIATIONS.md``.
    case : int, default 3
        PSS deterministic case, applied to every individual.
    threshold : float or {'mean'}, default 0.0
        Decomposition threshold, as in :class:`pyardl.nardl.NARDL`.
    aggregator, trim, min_obs, max_p, max_q, ic, asym_lags :
        As in :class:`pyardl.panel.mg.MeanGroup` /
        :class:`pyardl.nardl.NARDL`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(15):
    ...     lam = -0.5 + 0.05 * rng.normal()
    ...     x = np.cumsum(rng.normal(size=80))
    ...     y = np.zeros(80)
    ...     for t in range(1, 80):
    ...         resid = rng.normal(scale=0.3)
    ...         y[t] = y[t - 1] + lam * (y[t - 1] - 2.0 * x[t - 1]) + resid
    ...     rows.append(
    ...         pd.DataFrame({"id": i, "t": np.arange(80), "y": y, "x": x})
    ...     )
    >>> panel = pd.concat(rows, ignore_index=True)
    >>> res = MeanGroupNARDL(panel, y="y", X=["x"], asym=["x"], id="id",
    ...                       time="t", order=(1, 1)).fit()
    >>> res.n_units
    15
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        X: Sequence[str],
        asym: Sequence[str],
        id: str,  # noqa: A002 - matches the spec's public API
        time: str,
        order: tuple[int, int | dict[str, int]] | Literal["auto"] = (1, 1),
        decomposition: Decomposition = "per_individual",
        case: int = 3,
        threshold: Threshold = 0.0,
        aggregator: Aggregator = "mean",
        trim: float = 0.1,
        min_obs: int = 15,
        max_p: int = 4,
        max_q: int = 4,
        ic: Literal["aic", "bic", "hq"] = "aic",
        asym_lags: AsymLags = "paired",
    ) -> None:
        if decomposition == "pooled":
            raise NotImplementedError(
                "decomposition='pooled' is not implemented: spec 30 §2.2 "
                "documents it as the exception (a single pair of partial "
                "sums built on the pooled panel, relevant only when a "
                "common threshold has an economic meaning), not the "
                "default this project has prioritised implementing. Use "
                "decomposition='per_individual' (the default). See "
                "docs/DEVIATIONS.md."
            )
        if decomposition != "per_individual":
            raise ValueError(
                f"decomposition={decomposition!r} must be 'per_individual' "
                "('pooled' is not implemented)."
            )
        if aggregator not in ("mean", "median", "trimmed"):
            raise ValueError(
                f"aggregator must be 'mean', 'median' or 'trimmed', got {aggregator!r}."
            )
        if not 0.0 <= trim < 0.5:
            raise ValueError(f"trim must be in [0, 0.5), got {trim}.")
        if not asym:
            raise ValueError(
                "asym is empty: a MeanGroupNARDL with nothing decomposed is "
                "a MeanGroup (pyardl.panel.MeanGroup)."
            )

        self.panel = panel_from_frame(
            df, y=y, x=list(X), id_col=id, time_col=time, min_obs=min_obs
        )
        self.asym = tuple(str(a) for a in asym)
        self.order = order
        self.case = int(case)
        self.threshold: Threshold = threshold
        self.max_p = int(max_p)
        self.max_q = int(max_q)
        self.ic = ic
        self.asym_lags: AsymLags = asym_lags
        self.aggregator: Aggregator = aggregator
        self.trim = float(trim)

    def _fit_one(self, y: pd.Series, x: pd.DataFrame) -> NARDLResults:
        return NARDL(
            y,
            x,
            asym=self.asym,
            order=self.order,
            case=self.case,
            threshold=self.threshold,
            max_p=self.max_p,
            max_q=self.max_q,
            ic=self.ic,
            asym_lags=self.asym_lags,
        ).fit()

    def fit(self) -> MeanGroupNARDLResults:
        """Estimate every individual NARDL, then average theta+ and theta- separately.

        Returns
        -------
        MeanGroupNARDLResults

        Raises
        ------
        ValueError
            If fewer than two individuals can be fitted.
        """
        fits: dict[object, NARDLResults] = {}
        failed: dict[object, str] = {}
        pos_rows: dict[object, pd.Series] = {}
        neg_rows: dict[object, pd.Series] = {}
        lambda_rows: dict[object, float] = {}

        for unit in self.panel:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = self._fit_one(unit.y, unit.x)
                    longrun = fit.longrun_asym
            except (ValueError, np.linalg.LinAlgError) as exc:
                failed[unit.key] = f"{type(exc).__name__}: {exc}"
                continue
            if not np.all(np.isfinite(longrun[["theta_pos", "theta_neg"]].to_numpy())):
                failed[unit.key] = "non-finite long-run coefficients"
                continue
            fits[unit.key] = fit
            pos_rows[unit.key] = longrun["theta_pos"]
            neg_rows[unit.key] = longrun["theta_neg"]
            lambda_rows[unit.key] = float(fit.lam)

        if len(fits) < 2:
            raise ValueError(
                f"Only {len(fits)} individual(s) could be fitted, and the "
                "Mean-Group standard error is the dispersion ACROSS "
                f"individuals: with fewer than two there is none. Failures: "
                f"{failed}."
            )
        if failed:
            warnings.warn(
                f"{len(failed)} individual(s) could not be fitted and are "
                f"absent from the group average: {failed}.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

        keys = list(fits)
        index = pd.Index(keys, name="id")
        theta_pos_i = pd.DataFrame(
            [pos_rows[k].to_numpy() for k in keys], index=index, columns=self.asym
        )
        theta_neg_i = pd.DataFrame(
            [neg_rows[k].to_numpy() for k in keys], index=index, columns=self.asym
        )
        lambda_i = pd.Series([lambda_rows[k] for k in keys], index=index, name="lambda")

        pos, var_pos, n_eff = _aggregate(
            theta_pos_i.to_numpy(), self.aggregator, self.trim
        )
        neg, var_neg, _ = _aggregate(theta_neg_i.to_numpy(), self.aggregator, self.trim)
        se_pos = np.sqrt(var_pos)
        se_neg = np.sqrt(var_neg)

        # H0: theta+_MG = theta-_MG. Two INDEPENDENT group means (spec 30
        # §2.3): the variance of the difference is the sum of the two
        # group variances, not a delta method on a per-individual ratio.
        diff = pos - neg
        se_diff = np.sqrt(var_pos + var_neg)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_diff = np.where(se_diff > 0, diff / se_diff, np.nan)
        dof = max(n_eff - 1, 1)
        pvalue_diff = 2.0 * stats.t.sf(np.abs(t_diff), dof)

        longrun_asym = pd.DataFrame(
            {
                "theta_pos": pos,
                "se_pos": se_pos,
                "theta_neg": neg,
                "se_neg": se_neg,
                "diff": diff,
                "se_diff": se_diff,
                "t_diff": t_diff,
                "pvalue_diff": pvalue_diff,
            },
            index=pd.Index(list(self.asym), name="variable"),
        )

        n_bad = int((lambda_i >= 0).sum())
        if n_bad:
            warnings.warn(
                f"{n_bad} of {len(fits)} individuals have lambda_i >= 0: "
                "they do not error-correct, so their theta_i is not a "
                "long-run coefficient in the sense being averaged. Kept — "
                "dropping them would select on the outcome. See "
                "res.non_adjusting.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

        return MeanGroupNARDLResults(
            longrun_asym=longrun_asym,
            individual=fits,
            theta_pos_i=theta_pos_i,
            theta_neg_i=theta_neg_i,
            lambda_i=lambda_i,
            panel=self.panel,
            aggregator=self.aggregator,
            n_effective=n_eff,
            failed=failed,
        )
