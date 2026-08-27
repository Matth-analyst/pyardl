r"""Mean Group estimation for heterogeneous dynamic panels (spec 22).

The result Pesaran and Smith (1995) established is negative and it is
the reason this module exists: in a dynamic panel whose coefficients
differ across individuals, **pooled estimators are inconsistent even as
both N and T go to infinity**. Forcing a common dynamic when the
dynamics differ pushes the heterogeneity into the error term, where it
becomes serial correlation correlated with the lagged dependent
variable — and that contaminates every coefficient, including the
long-run ones. More data does not help, because the misspecification
does not shrink.

Their fix is almost embarrassingly simple. Estimate each individual's
ARDL on its own, then average:

.. math::

    \hat\theta_{MG} = \frac{1}{N} \sum_{i=1}^{N} \hat\theta_i

Each :math:`\hat\theta_i` is consistent as :math:`T_i \to \infty`, so
their average is consistent as both dimensions grow, with no homogeneity
assumption at all.

The inference is where it gets counter-intuitive
------------------------------------------------
Every individual fit hands back a perfectly good standard error for its
own :math:`\hat\theta_i`. It is tempting — and wrong — to combine those
into a standard error for the average.

The variance of :math:`\hat\theta_{MG}` comes from the **dispersion of
the estimates across individuals**:

.. math::

    \widehat{V}(\hat\theta_{MG})
      = \frac{1}{N(N-1)} \sum_{i=1}^{N} (\hat\theta_i - \hat\theta_{MG})^2

The reason is that :math:`\theta_i` is itself random — a draw from the
distribution of long-run coefficients across individuals — and
:math:`\hat\theta_{MG}` estimates the *mean of that distribution*. Its
sampling variability is therefore dominated by how much the
:math:`\theta_i` genuinely differ, not by how precisely each one was
measured. Pool the within-individual standard errors instead and you
estimate the precision of a common coefficient that does not exist; the
interval is far too narrow and the coverage collapses. That is measured,
not asserted: see ``validation/spec22_montecarlo.py`` and the panel
documentation page.

References
----------
.. [1] Pesaran, M. H. & Smith, R. (1995). Estimating long-run
       relationships from dynamic heterogeneous panels. *Journal of
       Econometrics*, 68(1), 79-113.
.. [2] Pesaran, M. H., Shin, Y. & Smith, R. P. (1999). Pooled mean group
       estimation of dynamic heterogeneous panels. *Journal of the
       American Statistical Association*, 94(446), 621-634.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from pyardl.core.ardl import ARDL
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.panel.container import PanelData, panel_from_frame

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import numpy.typing as npt

    from pyardl.core.ardl import ARDLResults

    FloatArray = npt.NDArray[np.float64]

__all__ = ["MeanGroup", "MeanGroupResults"]

Aggregator = Literal["mean", "median", "trimmed"]
DetType = Literal["none", "const", "trend"]


def _aggregate(
    values: FloatArray, how: Aggregator, trim: float
) -> tuple[FloatArray, FloatArray, int]:
    """Group estimate, its between-individual variance, and the N used.

    Returns the point estimate, the variance **of the estimate** (already
    divided by the effective N, so it is a variance of a mean, not of the
    population), and that effective N.

    All three aggregators share one principle: the variance comes from
    how much the individual estimates DIFFER, never from how precisely
    each was measured.
    """
    n_all, k = values.shape
    if how == "mean":
        point = values.mean(axis=0)
        var = np.sum((values - point) ** 2, axis=0) / (n_all * (n_all - 1))
        return point, var, n_all

    if how == "median":
        point = np.median(values, axis=0)
        # The median's sampling variance is pi/2 times the mean's under
        # normality. Reporting the mean's formula here would understate
        # it by 25%, so the factor is applied rather than ignored.
        var = (np.sum((values - point) ** 2, axis=0) / (n_all * (n_all - 1))) * (
            np.pi / 2
        )
        return point, var, n_all

    n_trim = int(np.floor(trim * n_all))
    if 2 * n_trim >= n_all:
        raise ValueError(
            f"trim={trim} removes {2 * n_trim} of {n_all} individuals, "
            "leaving nothing to average."
        )
    point = np.empty(k)
    var = np.empty(k)
    for j in range(k):
        col = np.sort(values[:, j])
        col = col[n_trim : n_all - n_trim] if n_trim else col
        m = col.shape[0]
        point[j] = col.mean()
        var[j] = np.sum((col - point[j]) ** 2) / (m * (m - 1))
    return point, var, n_all - 2 * n_trim


@dataclass(frozen=True)
class MeanGroupResults:
    """Mean Group estimates, with the individual fits kept.

    Attributes
    ----------
    longrun : pandas.DataFrame
        One row per regressor: ``theta`` (the group mean), ``se``,
        ``t``, ``pvalue`` and the confidence bounds. The standard error
        is the **between-individual** one; see the module docstring for
        why the within-individual ones are not pooled.
    coefficients : pandas.DataFrame
        Group mean of the **raw ARDL coefficients**, same construction.
        This is the short-run half of the picture, and it is what
        ``plm::pmg(model="mg")`` reports, which makes it the natural
        object for cross-implementation checks. Empty when the orders
        were selected per individual, because coefficients from
        different specifications are not the same quantity and averaging
        them would produce a number with no interpretation.
    adjustment : pandas.Series
        Group mean of the error-correction coefficients, its
        between-individual standard error, and the share of individuals
        that do not adjust.
    individual : dict
        ``{key: ARDLResults}`` — every individual fit, kept so a group
        average can always be traced back to what produced it.
    theta_i : pandas.DataFrame
        The individual long-run coefficients that were averaged.
    lambda_i : pandas.Series
        The individual adjustment speeds.
    n_units : int
        Individuals in the average.
    failed : dict
        ``{key: reason}`` for individuals whose ARDL could not be fitted.
    """

    longrun: pd.DataFrame
    coefficients: pd.DataFrame
    adjustment: pd.Series
    theta_i: pd.DataFrame
    coef_i: pd.DataFrame
    lambda_i: pd.Series
    orders: pd.DataFrame
    individual: dict[object, ARDLResults] = field(repr=False)
    panel: PanelData = field(repr=False)
    aggregator: str = "mean"
    n_effective: int = 0
    failed: dict[object, str] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return int(self.theta_i.shape[0])

    @property
    def non_adjusting(self) -> pd.Index:
        """Individuals whose estimated adjustment speed is not negative.

        Error correction requires :math:`\\lambda_i < 0`. An individual
        with :math:`\\hat\\lambda_i \\ge 0` is not converging back to any
        long-run relation, and its :math:`\\hat\\theta_i` is not a
        long-run coefficient in the sense being averaged. They are kept
        in the average by default — dropping them would select on the
        outcome — but they are named, and their share is reported.
        """
        return self.lambda_i.index[self.lambda_i >= 0]

    def heterogeneity(self) -> pd.DataFrame:
        """Spread of the individual estimates, per coefficient.

        The table that says whether pooling would have been defensible:
        a small dispersion relative to the mean is the case where MG and
        a pooled estimator agree, and a large one is the case Pesaran
        and Smith wrote the paper about.
        """
        desc = self.theta_i.describe().T[["mean", "std", "min", "50%", "max"]]
        desc.columns = ["mean", "sd", "min", "median", "max"]
        with np.errstate(divide="ignore", invalid="ignore"):
            desc["cv"] = np.abs(desc["sd"] / desc["mean"])
        return desc

    def summary(self, alpha: float = 0.05) -> str:
        """Publication-style summary of the group estimates."""
        lines = [
            f"Mean Group estimation (Pesaran & Smith 1995) - "
            f"{self.n_units} individuals",
            f"  dependent: {self.panel.y_name}   "
            f"{'unbalanced' if self.panel.unbalanced else 'balanced'} panel, "
            f"T from {int(self.panel.sample_sizes.min())} to "
            f"{int(self.panel.sample_sizes.max())}",
            f"  aggregator: {self.aggregator} over {self.n_effective} individuals",
            "  standard errors: BETWEEN-individual dispersion "
            "(not pooled within-individual)",
            "",
            "  Long-run coefficients",
            f"    {'':<12}{'theta':>12}{'se':>12}{'t':>10}{'p':>10}",
        ]
        for name, row in self.longrun.iterrows():
            lines.append(
                f"    {str(name):<12}{row['theta']:>12.4f}{row['se']:>12.4f}"
                f"{row['t']:>10.3f}{row['pvalue']:>10.4f}"
            )
        lines += [
            "",
            f"  Adjustment speed: {self.adjustment['lambda']:.4f} "
            f"(se {self.adjustment['se']:.4f})",
        ]
        n_bad = len(self.non_adjusting)
        if n_bad:
            lines.append(
                f"  WARNING {n_bad} of {self.n_units} individuals have "
                f"lambda_i >= 0 ({list(self.non_adjusting)[:5]}"
                f"{'...' if n_bad > 5 else ''}): they do not error-correct, "
                "so their theta_i is not a long-run coefficient in the "
                "sense being averaged."
            )
        if self.failed:
            lines.append(f"  {len(self.failed)} individual(s) could not be fitted:")
            lines.extend(f"    {k!r}: {v}" for k, v in self.failed.items())
        if self.panel.excluded:
            lines.append(
                f"  {len(self.panel.excluded)} individual(s) excluded at "
                f"validation: {self.panel.excluded}"
            )
        del alpha
        return "\n".join(lines)

    def plot_heterogeneity(self, ax: object = None):  # type: ignore[no-untyped-def]
        """Plot the distribution of the individual long-run coefficients.

        Requires ``matplotlib``, which is an optional dependency.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "plot_heterogeneity requires matplotlib, an optional "
                "dependency. Install it with: pip install matplotlib"
            ) from exc
        k = self.theta_i.shape[1]
        if ax is None:  # pragma: no cover - trivial plumbing
            _, ax = plt.subplots(1, k, figsize=(5 * k, 4), squeeze=False)
            ax = ax[0]
        axes = np.atleast_1d(np.asarray(ax, dtype=object))
        for j, name in enumerate(self.theta_i.columns):
            a = axes[j]
            a.hist(self.theta_i[name], bins="auto", alpha=0.75)
            a.axvline(
                self.longrun.loc[name, "theta"],
                color="black",
                linestyle="--",
                label="MG",
            )
            a.set_title(f"theta_i: {name}")
            a.legend()
        return axes


class MeanGroup:
    """Mean Group estimator for a heterogeneous dynamic panel.

    Estimates an ARDL per individual and averages the coefficients. The
    heavy lifting is done by :class:`pyardl.ARDL`; this class owns the
    panel bookkeeping and the aggregation.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel: one row per (individual, period).
    y : str
        Dependent-variable column.
    X : sequence of str
        Regressor columns.
    id : str
        Individual identifier column.
    time : str
        Time column.
    order : tuple or {'auto'}
        ``(p, q)`` applied to every individual, or ``'auto'`` to select
        the orders **per individual** by information criterion. Both
        modes are offered because both appear in the literature and they
        answer different questions: a common order makes the individual
        coefficients directly comparable, per-individual selection lets
        each dynamic be what the data say it is.
    det : {'none', 'const', 'trend'}
        Deterministic terms in each individual regression.
    max_p, max_q : int
        Search bounds when ``order='auto'``.
    ic : {'aic', 'bic', 'hq'}
        Information criterion for per-individual selection.
    aggregator : {'mean', 'median', 'trimmed'}
        How to combine the individual estimates. ``'mean'`` is the
        estimator of Pesaran and Smith. ``'median'`` and ``'trimmed'``
        are the small-N robustifications the paper's own discussion
        suggests: a single individual with an explosive dynamic can move
        a mean of twenty by a lot, and the group estimate would then
        describe that one individual rather than the group.
    trim : float
        Fraction removed from each tail when ``aggregator='trimmed'``.
    min_obs : int
        Individuals with fewer usable rows are excluded.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(12):
    ...     theta = 0.8 + 0.1 * rng.normal()
    ...     lam = -0.4 + 0.05 * rng.normal()
    ...     x = np.cumsum(rng.normal(size=60))
    ...     y = np.zeros(60)
    ...     for t in range(1, 60):
    ...         y[t] = y[t-1] + lam * (y[t-1] - theta * x[t-1]) + rng.normal(scale=.3)
    ...     rows.append(pd.DataFrame({"id": i, "t": np.arange(60),
    ...                               "y": y, "x": x}))
    >>> panel = pd.concat(rows, ignore_index=True)
    >>> res = MeanGroup(panel, y="y", X=["x"], id="id", time="t",
    ...                 order=(1, 1)).fit()
    >>> res.n_units
    12
    >>> bool(res.adjustment["lambda"] < 0)
    True
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        X: Sequence[str],
        id: str,  # noqa: A002 - matches the spec's public API
        time: str,
        order: tuple[int, int | dict[str, int]] | Literal["auto"] = (1, 1),
        det: DetType = "const",
        max_p: int = 4,
        max_q: int = 4,
        ic: Literal["aic", "bic", "hq"] = "aic",
        aggregator: Aggregator = "mean",
        trim: float = 0.1,
        min_obs: int = 15,
    ) -> None:
        if aggregator not in ("mean", "median", "trimmed"):
            raise ValueError(
                f"aggregator must be 'mean', 'median' or 'trimmed', got {aggregator!r}."
            )
        if not 0.0 <= trim < 0.5:
            raise ValueError(f"trim must be in [0, 0.5), got {trim}.")
        if det not in ("none", "const", "trend"):
            raise ValueError(f"det must be 'none', 'const' or 'trend', got {det!r}.")
        if order != "auto":
            p_req = int(order[0])
            if p_req < 1:
                raise ValueError(
                    f"p must be at least 1 for an error-correction "
                    f"representation, got {p_req}: with p=0 there is no "
                    "lagged dependent variable and so no adjustment speed."
                )

        self.panel = panel_from_frame(
            df, y=y, x=list(X), id_col=id, time_col=time, min_obs=min_obs
        )
        self.order = order
        self.det: DetType = det
        self.max_p = int(max_p)
        self.max_q = int(max_q)
        self.ic = ic
        self.aggregator: Aggregator = aggregator
        self.trim = float(trim)

    def _fit_one(self, unit: object, y: pd.Series, x: pd.DataFrame) -> ARDLResults:
        """Fit one individual, selecting its order when asked to."""
        del unit
        if self.order == "auto":
            selection = ARDL.select_order(
                y,
                x,
                max_p=self.max_p,
                max_q=self.max_q,
                ic=self.ic,
                det=self.det,
                min_p=1,
            )
            return selection.best_model
        p, q = self.order
        return ARDL(y, x, order=(int(p), q), det=self.det).fit()

    def fit(self) -> MeanGroupResults:
        """Estimate every individual, then average.

        Returns
        -------
        MeanGroupResults

        Raises
        ------
        ValueError
            If fewer than two individuals can be fitted. With one there
            is no between-individual dispersion, so no standard error
            exists — and reporting a point estimate with no interval
            would invite it to be read as if it had one.
        """
        fits: dict[object, ARDLResults] = {}
        failed: dict[object, str] = {}
        theta_rows: dict[object, pd.Series] = {}
        coef_rows: dict[object, pd.Series] = {}
        lambda_rows: dict[object, float] = {}
        order_rows: dict[object, dict[str, int]] = {}

        for unit in self.panel:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = self._fit_one(unit.key, unit.y, unit.x)
                    longrun = fit.longrun
                    adjustment = fit.adjustment
            except (ValueError, np.linalg.LinAlgError) as exc:
                failed[unit.key] = f"{type(exc).__name__}: {exc}"
                continue
            if not np.all(np.isfinite(longrun["theta"].to_numpy())):
                failed[unit.key] = "non-finite long-run coefficients"
                continue
            fits[unit.key] = fit
            theta_rows[unit.key] = longrun["theta"]
            coef_rows[unit.key] = fit.params
            lambda_rows[unit.key] = float(adjustment["lambda"])
            order_rows[unit.key] = {
                "p": int(fit.model.p),
                **{
                    f"q[{n}]": int(v)
                    for n, v in zip(self.panel.x_names, fit.model.q, strict=True)
                },
            }

        if len(fits) < 2:
            raise ValueError(
                f"Only {len(fits)} individual(s) could be fitted, and the Mean "
                "Group standard error is the dispersion ACROSS individuals: "
                "with fewer than two there is none, so no interval exists. "
                f"Failures: {failed}."
            )
        if failed:
            warnings.warn(
                f"{len(failed)} individual(s) could not be fitted and are "
                f"absent from the group average: {failed}. The reported N is "
                f"{len(fits)}.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

        keys = list(fits)
        index = pd.Index(keys, name="id")
        theta_i = pd.DataFrame(
            [theta_rows[k].to_numpy() for k in keys],
            index=index,
            columns=list(self.panel.x_names),
        )
        lambda_i = pd.Series([lambda_rows[k] for k in keys], index=index, name="lambda")
        orders = pd.DataFrame([order_rows[k] for k in keys], index=index)

        # Raw coefficients are averaged only when every individual was
        # given the same specification. Under per-individual selection
        # the columns do not line up, and a mean over whichever names
        # happen to coincide would be a number about nothing.
        common_spec = orders.nunique().eq(1).all()
        names = list(coef_rows[keys[0]].index)
        if common_spec and all(list(coef_rows[k].index) == names for k in keys):
            coef_i = pd.DataFrame(
                [coef_rows[k].to_numpy() for k in keys], index=index, columns=names
            )
        else:
            coef_i = pd.DataFrame(index=index)

        theta, var_theta, n_eff = _aggregate(
            theta_i.to_numpy(), self.aggregator, self.trim
        )
        lam, var_lam, _ = _aggregate(
            lambda_i.to_numpy()[:, None], self.aggregator, self.trim
        )

        se = np.sqrt(var_theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(se > 0, theta / se, np.nan)
        # The reference distribution is the t with N-1 degrees of freedom,
        # not the normal: the variance is estimated from N individual
        # estimates, and at the N of a typical macro panel the difference
        # is not cosmetic.
        dof = max(n_eff - 1, 1)
        pvalue = 2.0 * stats.t.sf(np.abs(tstat), dof)
        crit = float(stats.t.ppf(0.975, dof))
        longrun = pd.DataFrame(
            {
                "theta": theta,
                "se": se,
                "t": tstat,
                "pvalue": pvalue,
                "ci_lower": theta - crit * se,
                "ci_upper": theta + crit * se,
            },
            index=pd.Index(list(self.panel.x_names), name="regressor"),
        )

        if coef_i.shape[1]:
            c_point, c_var, _ = _aggregate(
                coef_i.to_numpy(), self.aggregator, self.trim
            )
            c_se = np.sqrt(c_var)
            with np.errstate(divide="ignore", invalid="ignore"):
                c_t = np.where(c_se > 0, c_point / c_se, np.nan)
            coefficients = pd.DataFrame(
                {
                    "coef": c_point,
                    "se": c_se,
                    "t": c_t,
                    "pvalue": 2.0 * stats.t.sf(np.abs(c_t), dof),
                },
                index=pd.Index(list(coef_i.columns), name="term"),
            )
        else:
            coefficients = pd.DataFrame(
                columns=["coef", "se", "t", "pvalue"],
                index=pd.Index([], name="term"),
            )

        n_bad = int((lambda_i >= 0).sum())
        adjustment = pd.Series(
            {
                "lambda": float(lam[0]),
                "se": float(np.sqrt(var_lam[0])),
                "share_non_adjusting": n_bad / len(fits),
            },
            name="adjustment",
        )
        if n_bad:
            warnings.warn(
                f"{n_bad} of {len(fits)} individuals have lambda_i >= 0: they "
                "do not error-correct, so their theta_i is not a long-run "
                "coefficient in the sense being averaged. They are kept — "
                "dropping them would select on the outcome — but the group "
                "estimate mixes two regimes. See res.non_adjusting.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

        return MeanGroupResults(
            longrun=longrun,
            coefficients=coefficients,
            adjustment=adjustment,
            theta_i=theta_i,
            coef_i=coef_i,
            lambda_i=lambda_i,
            orders=orders,
            individual=fits,
            panel=self.panel,
            aggregator=self.aggregator,
            n_effective=n_eff,
            failed=failed,
        )
