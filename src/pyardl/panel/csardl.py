r"""CS-ARDL and CS-DL: panels with common factors (spec 24).

Specs 22 and 23 estimate each individual on its own and average. That
is exactly right when individuals are independent, and wrong when they
are not — a world cycle or a commodity price hits everyone, the omitted
factor correlates with the regressors, and every :math:`\hat\theta_i`
is biased before the averaging starts.

Chudik and Pesaran augment each individual's regression with the
**cross-sectional averages** of the variables, which span the unobserved
factor space asymptotically. Two estimators come out of it:

**CS-ARDL** keeps the full dynamic specification and adds the averages
plus :math:`p_z` of their lags. The long run is then rebuilt from the
short-run coefficients, exactly as in spec 03:

.. math:: \hat\theta_i = \frac{\sum_j \hat\beta_{ij}}{1 - \sum_s \hat\phi_{is}}

**CS-DL** skips the dynamics. It regresses :math:`y_{it}` on
:math:`x_{it}`, a few lagged *differences* of :math:`x`, and the
averages; the coefficient on :math:`x_{it}` **is** the long-run
coefficient, with no ratio to form. That buys robustness — a
misspecified lag order cannot distort a ratio that is never computed —
at the price of assuming the adjustment is fast enough for the
truncation to be innocuous.

Both aggregate across individuals the Mean Group way, so the standard
error is the **between-individual** dispersion of spec 22, not anything
pooled from within.

Why the collinearity handling is spelled out
--------------------------------------------
With :math:`k+1` averages, :math:`p_z` lags of each, and a modest
:math:`T`, the individual design is often rank-deficient. Dropping
columns is unavoidable; doing it *by whatever the linear algebra
happens to prefer today* is not. A pivoted QR on one platform and a
different BLAS on another would silently keep different columns and
report different long-run coefficients from the same data.

So the rule here is fixed and stated: columns are examined **left to
right in a declared order** — deterministic terms, own lags, own
regressors, then the cross-sectional averages from contemporaneous to
most-lagged — and a column is dropped when it adds nothing to the rank
of what precedes it. The averages are last on purpose: they are the
approximation, so when something must go, it should be the
approximation rather than the model. Every drop is recorded in
``res.dropped_columns``.

References
----------
.. [1] Chudik, A. & Pesaran, M. H. (2015). Common correlated effects
       estimation of heterogeneous dynamic panel data models with weakly
       exogenous regressors. *Journal of Econometrics*, 188(2), 393-420.
.. [2] Chudik, A., Mohaddes, K., Pesaran, M. H. & Raissi, M. (2016).
       Long-run effects in large heterogeneous panel data models with
       cross-sectionally correlated errors. *Advances in Econometrics*,
       36, 85-135.
.. [3] Pesaran, M. H. (2006). Estimation and inference in large
       heterogeneous panels with a multifactor error structure.
       *Econometrica*, 74(4), 967-1012.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.panel.container import PanelData, panel_from_frame
from pyardl.panel.crosssection import (
    CDResult,
    cd_test,
    cross_section_averages,
    default_cs_lags,
)
from pyardl.panel.mg import _aggregate

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]

__all__ = ["CSDL", "CSARDL", "CSDLResults", "CSARDLResults"]

DetType = Literal["none", "const", "trend"]

#: Relative tolerance of the rank test used to drop collinear columns.
#: Fixed rather than left to ``lstsq``'s default so the same data give
#: the same design on every platform.
RANK_TOL = 1e-9


def _select_independent(
    design: FloatArray, names: Sequence[str]
) -> tuple[FloatArray, list[str], list[str]]:
    """Keep a maximal independent set of columns, left to right.

    Deterministic by construction: columns are considered in the order
    given, and one is kept only if it raises the rank of those already
    kept. The caller orders the columns so that the model comes before
    the approximation, which makes the *dropped* set the least
    informative one rather than an accident of the solver.
    """
    kept_idx: list[int] = []
    kept: list[str] = []
    dropped: list[str] = []
    scale = float(np.linalg.norm(design)) or 1.0
    for j, name in enumerate(names):
        trial = design[:, [*kept_idx, j]]
        if np.linalg.matrix_rank(trial, tol=RANK_TOL * scale) == len(kept_idx) + 1:
            kept_idx.append(j)
            kept.append(name)
        else:
            dropped.append(name)
    return design[:, kept_idx], kept, dropped


@dataclass(frozen=True)
class _Fit:
    """One individual's augmented regression."""

    key: object
    params: pd.Series
    resid: pd.Series
    nobs: int
    dropped: list[str]


def _mg_table(values: pd.DataFrame, names: Sequence[str]) -> tuple[pd.DataFrame, int]:
    """Mean Group table from a frame of individual estimates."""
    point, var, n_eff = _aggregate(values.to_numpy(dtype=np.float64), "mean", 0.0)
    se = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, point / se, np.nan)
    dof = max(values.shape[0] - 1, 1)
    crit = float(stats.t.ppf(0.975, dof))
    table = pd.DataFrame(
        {
            "theta": point,
            "se": se,
            "t": tstat,
            "pvalue": 2.0 * stats.t.sf(np.abs(tstat), dof),
            "ci_lower": point - crit * se,
            "ci_upper": point + crit * se,
        },
        index=pd.Index(list(names), name="regressor"),
    )
    return table, n_eff


@dataclass(frozen=True)
class _CSResultsBase:
    """What CS-ARDL and CS-DL share."""

    longrun: pd.DataFrame
    theta_i: pd.DataFrame
    residuals: pd.DataFrame = field(repr=False)
    individual: dict[object, _Fit] = field(repr=False)
    panel: PanelData = field(repr=False)
    cs_lags: int
    dropped_columns: dict[object, list[str]] = field(default_factory=dict)
    failed: dict[object, str] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return int(self.theta_i.shape[0])

    def cd_test(self, min_overlap: int = 5) -> CDResult:
        """Pesaran's CD test on the residuals of the augmented model.

        Run *after* the augmentation, so a failure to reject is the good
        outcome: it says the cross-sectional averages absorbed the
        common factor. A rejection says they did not — more lags, or
        more factors than the averages can span.
        """
        return cd_test(self.residuals, min_overlap=min_overlap)

    def heterogeneity(self) -> pd.DataFrame:
        """Spread of the individual long-run estimates."""
        desc = self.theta_i.describe().T[["mean", "std", "min", "50%", "max"]]
        desc.columns = ["mean", "sd", "min", "median", "max"]
        with np.errstate(divide="ignore", invalid="ignore"):
            desc["cv"] = np.abs(desc["sd"] / desc["mean"])
        return desc


@dataclass(frozen=True)
class CSARDLResults(_CSResultsBase):
    """CS-ARDL estimates."""

    adjustment: pd.Series = field(default_factory=pd.Series)
    lambda_i: pd.Series = field(default_factory=pd.Series)
    order: tuple[int, int] = (1, 1)

    @property
    def non_adjusting(self) -> pd.Index:
        return self.lambda_i.index[self.lambda_i >= 0]

    def summary(self) -> str:
        lines = [
            f"CS-ARDL (Chudik & Pesaran 2015) - {self.n_units} individuals",
            f"  ARDL{self.order} augmented with cross-sectional averages and "
            f"{self.cs_lags} lag(s) of them",
            "  standard errors: BETWEEN-individual dispersion (Mean Group)",
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
            f"  Mean adjustment speed: {self.adjustment['lambda']:.4f} "
            f"(se {self.adjustment['se']:.4f})",
        ]
        n_bad = len(self.non_adjusting)
        if n_bad:
            lines.append(
                f"  WARNING {n_bad} of {self.n_units} individuals have "
                "lambda_i >= 0: they do not error-correct, so their theta_i "
                "is not a long-run coefficient in the sense being averaged."
            )
        if self.dropped_columns:
            lines.append(
                f"  {len(self.dropped_columns)} individual(s) had collinear "
                "columns removed; see res.dropped_columns."
            )
        if self.failed:
            lines.append(f"  {len(self.failed)} not fitted: {self.failed}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CSDLResults(_CSResultsBase):
    """CS-DL estimates."""

    trunc_lags: int = 0

    def summary(self) -> str:
        lines = [
            f"CS-DL (Chudik, Mohaddes, Pesaran & Raissi 2016) - "
            f"{self.n_units} individuals",
            f"  y on x, {self.trunc_lags} lagged difference(s) of x, and "
            f"cross-sectional averages with {self.cs_lags} lag(s)",
            "  standard errors: BETWEEN-individual dispersion (Mean Group)",
            "",
            "  Long-run coefficients (read directly off x, no ratio formed)",
            f"    {'':<12}{'theta':>12}{'se':>12}{'t':>10}{'p':>10}",
        ]
        for name, row in self.longrun.iterrows():
            lines.append(
                f"    {str(name):<12}{row['theta']:>12.4f}{row['se']:>12.4f}"
                f"{row['t']:>10.3f}{row['pvalue']:>10.4f}"
            )
        lines += [
            "",
            "  No adjustment speed: CS-DL estimates the long run without "
            "the dynamics, which is what makes it robust to a misspecified "
            "lag order and what stops it from saying how fast adjustment is.",
        ]
        if self.dropped_columns:
            lines.append(
                f"  {len(self.dropped_columns)} individual(s) had collinear "
                "columns removed; see res.dropped_columns."
            )
        if self.failed:
            lines.append(f"  {len(self.failed)} not fitted: {self.failed}")
        return "\n".join(lines)


def _prepare(
    df: pd.DataFrame,
    y: str,
    x: Sequence[str],
    id_col: str,
    time_col: str,
    cs_lags: int | Literal["auto"],
    weights: str | None,
    min_obs: int,
) -> tuple[PanelData, pd.DataFrame, int]:
    """Validate the panel and build the cross-sectional averages once."""
    panel = panel_from_frame(
        df, y=y, x=list(x), id_col=id_col, time_col=time_col, min_obs=min_obs
    )
    n_periods = int(df[time_col].nunique())
    lags = default_cs_lags(n_periods) if cs_lags == "auto" else int(cs_lags)
    if lags < 0:
        raise ValueError(f"cs_lags must be non-negative, got {lags}.")
    averages = cross_section_averages(
        df,
        [y, *x],
        id_col=id_col,
        time_col=time_col,
        lags=lags,
        weights=weights,
    )
    return panel, averages, lags


def _fit_individual(
    key: object,
    target: pd.Series,
    columns: dict[str, pd.Series],
) -> _Fit:
    """One augmented OLS, with the deterministic collinearity rule."""
    frame = pd.DataFrame(columns)
    frame["__y__"] = target
    frame = frame.dropna()
    if frame.shape[0] <= frame.shape[1]:
        raise ValueError(
            f"{frame.shape[0]} usable observations for {frame.shape[1] - 1} "
            "regressors after adding the cross-sectional averages."
        )
    names = [c for c in frame.columns if c != "__y__"]
    design = frame[names].to_numpy(dtype=np.float64)
    yv = frame["__y__"].to_numpy(dtype=np.float64)
    design, kept, dropped = _select_independent(design, names)
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    resid = yv - design @ beta
    return _Fit(
        key=key,
        params=pd.Series(beta, index=pd.Index(kept, name="term")),
        resid=pd.Series(resid, index=frame.index),
        nobs=int(frame.shape[0]),
        dropped=dropped,
    )


def _assemble(
    fits: dict[object, _Fit],
    theta_rows: dict[object, FloatArray],
    x_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = list(fits)
    index = pd.Index(keys, name="id")
    theta_i = pd.DataFrame(
        [theta_rows[k] for k in keys], index=index, columns=list(x_names)
    )
    residuals = pd.DataFrame({k: fits[k].resid for k in keys})
    table, _ = _mg_table(theta_i, x_names)
    return table, theta_i, residuals


class CSARDL:
    """CS-ARDL: the dynamic panel, augmented with cross-sectional averages.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel.
    y, X, id, time : str / sequence of str
        Column names, as in :class:`~pyardl.panel.MeanGroup`.
    order : tuple, default (1, 1)
        ``(p, q)`` of the individual ARDL, common to all individuals.
    cs_lags : int or 'auto', default 'auto'
        Lags of the cross-sectional averages. ``'auto'`` uses
        ``floor(T**(1/3))`` — see
        :func:`~pyardl.panel.crosssection.default_cs_lags`.
    det : {'none', 'const', 'trend'}
        Deterministic terms in each individual equation.
    weights : str, optional
        Column of weights for the averages. Equal weights by default.
    min_obs : int
        Individuals with fewer usable rows are excluded.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> T, N = 60, 12
    >>> f = np.cumsum(rng.normal(size=T))          # facteur commun
    >>> rows = []
    >>> for i in range(N):
    ...     g = 0.5 + rng.normal(scale=0.2)
    ...     x = np.cumsum(rng.normal(size=T)) + g * f
    ...     y = np.zeros(T)
    ...     for t in range(1, T):
    ...         y[t] = (y[t-1] - 0.4 * (y[t-1] - 0.8 * x[t-1])
    ...                 + g * (f[t] - f[t-1]) + rng.normal(scale=0.3))
    ...     rows.append(pd.DataFrame({"id": i, "t": np.arange(T),
    ...                               "y": y, "x": x}))
    >>> res = CSARDL(pd.concat(rows, ignore_index=True), y="y", X=["x"],
    ...              id="id", time="t", order=(1, 1)).fit()
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
        order: tuple[int, int] = (1, 1),
        cs_lags: int | Literal["auto"] = "auto",
        det: DetType = "const",
        weights: str | None = None,
        min_obs: int = 15,
    ) -> None:
        if det not in ("none", "const", "trend"):
            raise ValueError(f"det must be 'none', 'const' or 'trend', got {det!r}.")
        p, q = int(order[0]), int(order[1])
        if p < 1:
            raise ValueError(
                f"p must be at least 1 for an error-correction representation, got {p}."
            )
        if q < 0:
            raise ValueError(f"q must be non-negative, got {q}.")
        self.df = df
        self.y_name, self.x_names = y, tuple(str(c) for c in X)
        self.id_col, self.time_col = id, time
        self.p, self.q = p, q
        self.det: DetType = det
        self.panel, self.averages, self.cs_lags = _prepare(
            df, y, list(X), id, time, cs_lags, weights, min_obs
        )

    def fit(self) -> CSARDLResults:
        """Estimate each individual, then average the long-run coefficients."""
        fits: dict[object, _Fit] = {}
        failed: dict[object, str] = {}
        theta_rows: dict[object, FloatArray] = {}
        lambda_rows: dict[object, float] = {}
        dropped: dict[object, list[str]] = {}

        for unit in self.panel:
            cols: dict[str, pd.Series] = {}
            if self.det in ("const", "trend"):
                cols["const"] = pd.Series(1.0, index=unit.y.index)
            if self.det == "trend":
                cols["trend"] = pd.Series(
                    np.arange(1.0, unit.y.size + 1.0), index=unit.y.index
                )
            for s in range(1, self.p + 1):
                cols[f"{self.y_name}.L{s}"] = unit.y.shift(s)
            for name in self.x_names:
                for s in range(self.q + 1):
                    cols[f"{name}.L{s}"] = unit.x[name].shift(s)
            # The averages come LAST: when the rank rule has to drop
            # something, it should drop the approximation, not the model.
            for col in self.averages.columns:
                if col == "cs_count":
                    continue
                cols[col] = self.averages[col].reindex(unit.y.index)

            try:
                fit = _fit_individual(unit.key, unit.y, cols)
            except (ValueError, np.linalg.LinAlgError) as exc:
                failed[unit.key] = f"{type(exc).__name__}: {exc}"
                continue

            phi = np.array(
                [
                    fit.params.get(f"{self.y_name}.L{s}", 0.0)
                    for s in range(1, self.p + 1)
                ]
            )
            denom = 1.0 - float(phi.sum())
            if abs(denom) < 1e-10:
                failed[unit.key] = (
                    "the autoregressive coefficients sum to one: the long-run "
                    "coefficient is a ratio with a zero denominator, so no "
                    "long run exists for this individual."
                )
                continue
            theta = np.array(
                [
                    sum(fit.params.get(f"{n}.L{s}", 0.0) for s in range(self.q + 1))
                    / denom
                    for n in self.x_names
                ]
            )
            if not np.all(np.isfinite(theta)):
                failed[unit.key] = "non-finite long-run coefficients"
                continue
            fits[unit.key] = fit
            theta_rows[unit.key] = theta
            lambda_rows[unit.key] = -denom
            if fit.dropped:
                dropped[unit.key] = fit.dropped

        if len(fits) < 2:
            raise ValueError(
                f"Only {len(fits)} individual(s) could be fitted, and the Mean "
                "Group standard error is the dispersion ACROSS individuals: "
                f"with fewer than two there is none. Failures: {failed}."
            )
        if failed:
            warnings.warn(
                f"{len(failed)} individual(s) absent from the group average: "
                f"{failed}. The reported N is {len(fits)}.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

        table, theta_i, residuals = _assemble(fits, theta_rows, self.x_names)
        keys = list(fits)
        lam = np.array([lambda_rows[k] for k in keys])
        n = lam.size
        lambda_i = pd.Series(lam, index=pd.Index(keys, name="id"), name="lambda")
        adjustment = pd.Series(
            {
                "lambda": float(lam.mean()),
                "se": float(np.sqrt(np.sum((lam - lam.mean()) ** 2) / (n * (n - 1)))),
                "share_non_adjusting": float((lam >= 0).mean()),
            },
            name="adjustment",
        )
        if int((lam >= 0).sum()):
            warnings.warn(
                f"{int((lam >= 0).sum())} of {n} individuals have lambda_i >= 0: "
                "they do not error-correct, so their theta_i is not a long-run "
                "coefficient in the sense being averaged.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )
        return CSARDLResults(
            longrun=table,
            theta_i=theta_i,
            residuals=residuals,
            individual=fits,
            panel=self.panel,
            cs_lags=self.cs_lags,
            dropped_columns=dropped,
            failed=failed,
            adjustment=adjustment,
            lambda_i=lambda_i,
            order=(self.p, self.q),
        )


class CSDL:
    """CS-DL: the long run without the dynamics.

    Regresses :math:`y_{it}` on :math:`x_{it}`, ``trunc_lags`` lagged
    differences of :math:`x`, and the cross-sectional averages. The
    coefficient on :math:`x_{it}` **is** the long-run coefficient — no
    ratio of estimated dynamics is ever formed, which is precisely what
    makes it robust to getting the lag order wrong.

    The price is stated rather than hidden: the truncation is innocuous
    only if adjustment is fast enough relative to ``trunc_lags``, and
    CS-DL cannot report an adjustment speed at all.

    Parameters
    ----------
    trunc_lags : int or 'auto', default 'auto'
        Lagged differences of ``x``. ``'auto'`` uses the same
        ``floor(T**(1/3))`` rule as the averages.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> T, N = 60, 12
    >>> f = np.cumsum(rng.normal(size=T))
    >>> rows = []
    >>> for i in range(N):
    ...     g = 0.5 + rng.normal(scale=0.2)
    ...     x = np.cumsum(rng.normal(size=T)) + g * f
    ...     y = 0.8 * x + g * f + rng.normal(scale=0.3, size=T)
    ...     rows.append(pd.DataFrame({"id": i, "t": np.arange(T),
    ...                               "y": y, "x": x}))
    >>> res = CSDL(pd.concat(rows, ignore_index=True), y="y", X=["x"],
    ...            id="id", time="t").fit()
    >>> res.n_units
    12
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        X: Sequence[str],
        id: str,  # noqa: A002
        time: str,
        trunc_lags: int | Literal["auto"] = "auto",
        cs_lags: int | Literal["auto"] = "auto",
        det: DetType = "const",
        weights: str | None = None,
        min_obs: int = 15,
    ) -> None:
        if det not in ("none", "const", "trend"):
            raise ValueError(f"det must be 'none', 'const' or 'trend', got {det!r}.")
        self.df = df
        self.y_name, self.x_names = y, tuple(str(c) for c in X)
        self.id_col, self.time_col = id, time
        self.det: DetType = det
        self.panel, self.averages, self.cs_lags = _prepare(
            df, y, list(X), id, time, cs_lags, weights, min_obs
        )
        n_periods = int(df[time].nunique())
        self.trunc_lags = (
            default_cs_lags(n_periods) if trunc_lags == "auto" else int(trunc_lags)
        )
        if self.trunc_lags < 0:
            raise ValueError(f"trunc_lags must be non-negative, got {self.trunc_lags}.")

    def fit(self) -> CSDLResults:
        """Estimate each individual, then average."""
        fits: dict[object, _Fit] = {}
        failed: dict[object, str] = {}
        theta_rows: dict[object, FloatArray] = {}
        dropped: dict[object, list[str]] = {}

        for unit in self.panel:
            cols: dict[str, pd.Series] = {}
            if self.det in ("const", "trend"):
                cols["const"] = pd.Series(1.0, index=unit.y.index)
            if self.det == "trend":
                cols["trend"] = pd.Series(
                    np.arange(1.0, unit.y.size + 1.0), index=unit.y.index
                )
            for name in self.x_names:
                cols[name] = unit.x[name]
                dx = unit.x[name].diff()
                for s in range(self.trunc_lags):
                    cols[f"D.{name}.L{s}"] = dx.shift(s)
            for col in self.averages.columns:
                if col == "cs_count":
                    continue
                cols[col] = self.averages[col].reindex(unit.y.index)

            try:
                fit = _fit_individual(unit.key, unit.y, cols)
            except (ValueError, np.linalg.LinAlgError) as exc:
                failed[unit.key] = f"{type(exc).__name__}: {exc}"
                continue
            missing = [n for n in self.x_names if n not in fit.params.index]
            if missing:
                failed[unit.key] = (
                    f"the level of {missing} was dropped as collinear, so the "
                    "long-run coefficient it carries cannot be read off."
                )
                continue
            theta = np.array([float(fit.params[n]) for n in self.x_names])
            if not np.all(np.isfinite(theta)):
                failed[unit.key] = "non-finite long-run coefficients"
                continue
            fits[unit.key] = fit
            theta_rows[unit.key] = theta
            if fit.dropped:
                dropped[unit.key] = fit.dropped

        if len(fits) < 2:
            raise ValueError(
                f"Only {len(fits)} individual(s) could be fitted; the Mean "
                "Group standard error needs at least two. "
                f"Failures: {failed}."
            )
        if failed:
            warnings.warn(
                f"{len(failed)} individual(s) absent from the group average: "
                f"{failed}. The reported N is {len(fits)}.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )
        table, theta_i, residuals = _assemble(fits, theta_rows, self.x_names)
        return CSDLResults(
            longrun=table,
            theta_i=theta_i,
            residuals=residuals,
            individual=fits,
            panel=self.panel,
            cs_lags=self.cs_lags,
            dropped_columns=dropped,
            failed=failed,
            trunc_lags=self.trunc_lags,
        )
