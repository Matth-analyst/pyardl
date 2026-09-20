r"""Gregory & Hansen (1996) cointegration test with an endogenous break.

Engle-Granger (:mod:`pyardl.cointegration.engle_granger`) assumes the
cointegrating vector is stable over the whole sample. If the relationship
actually shifts regime partway through, the static step-one regression
averages both regimes, the residual looks non-stationary, and the test
wrongly concludes there is no cointegration. Gregory-Hansen answers by
leaving the break fraction unknown and estimating it from the data: a
grid search over candidate break fractions, an Engle-Granger step-two ADF
test at each one, and the test statistic is the smallest (most negative)
ADF found over the grid.

No new numerical engine is introduced: step two is exactly the ADF
regression of Engle-Granger (spec 06 §2.2), reused unchanged inside a
loop over candidate breaks; only the step-one design matrix changes (it
gains regime-shift regressors), and the critical values do, because a
supremum over a grid of unknown break dates has a different — and
non-standard — asymptotic distribution than a single ADF statistic (the
same "problem of Davies" already met at the Fourier frequency pre-test,
:mod:`pyardl.fourier`).

References
----------
.. [1] Gregory, A. W. & Hansen, B. E. (1996). Residual-based tests for
       cointegration in models with regime shifts. *Journal of
       Econometrics*, 70(1), 99-126.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyardl.bootstrap.dgp import _fit_marginal_var
from pyardl.unitroot.gls import adf_regression, select_lags
from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

GHModel = Literal["C", "C/T", "C/S", "C/S/T"]
CVSource = Literal["bootstrap"]

__all__ = ["gregory_hansen", "GregoryHansenResults"]


@dataclass(frozen=True)
class GregoryHansenResults:
    """Outcome of a Gregory-Hansen test.

    Attributes
    ----------
    statistic : float
        ADF* = the minimum step-two ADF t-ratio over the break grid.
        Left-tailed, like Engle-Granger.
    break_fraction : float
        :math:`\\hat\\tau`, the break point as a fraction of the sample
        that minimises the step-two ADF statistic.
    break_index : int
        Observation index of the estimated break (0-based; the break
        dummy is 1 for observations strictly after this index).
    break_date : object or None
        Value of the original index at ``break_index``, when the input
        carried a pandas index; ``None`` otherwise.
    critical_values : dict
        Level to critical value, at 10%, 5% and 1%, from ``cv_source``.
    model : str
        Break specification used ('C', 'C/T', 'C/S' or 'C/S/T').
    cv_source : str
        How the critical values were obtained. Only ``'bootstrap'`` is
        implemented — see the module notes.
    grid : pandas.DataFrame
        Columns ``tau`` and ``adf_stat`` for every break fraction tested,
        so the margin of the retained break is visible, not just the
        minimum.
    lags : int
        Lag order of the step-two ADF regression at the retained break.
    n_boot : int
        Bootstrap replications used for the critical values.
    seed : int or None
        Seed passed to the bootstrap generator.
    """

    statistic: float
    break_fraction: float
    break_index: int
    break_date: object | None
    critical_values: dict[float, float]
    model: GHModel
    cv_source: CVSource
    grid: pd.DataFrame
    lags: int
    n_boot: int
    seed: int | None

    def decision(self, alpha: float = 0.05) -> str:
        """``'cointegration'`` or ``'no_cointegration'`` at ``alpha``.

        As with Engle-Granger, rejection of the null (no cointegration)
        happens when the statistic falls below (more negative than) the
        critical value — but here the null distribution already accounts
        for the search over the break grid, so ``ADF*`` must never be
        compared to an ordinary Engle-Granger/MacKinnon critical value.
        """
        if alpha not in self.critical_values or np.isnan(self.critical_values[alpha]):
            raise ValueError(
                f"No critical value at alpha={alpha} for model="
                f"{self.model!r}, cv_source={self.cv_source!r}."
            )
        return (
            "cointegration"
            if self.statistic < self.critical_values[alpha]
            else "no_cointegration"
        )

    def summary(self) -> str:
        """Readable report of the test."""
        cv = "  ".join(
            f"{int(a * 100)}%: {v:.4f}" for a, v in sorted(self.critical_values.items())
        )
        date_txt = f" ({self.break_date})" if self.break_date is not None else ""
        lines = [
            f"Gregory-Hansen test (1996) - model '{self.model}', "
            f"cv_source={self.cv_source}, lags={self.lags}",
            f"  statistic (ADF*) = {self.statistic:.4f}",
            f"  break fraction (tau_hat) = {self.break_fraction:.4f}, "
            f"index={self.break_index}{date_txt}",
            f"  decision (5%): {self.decision(0.05)}",
            f"  critical values (left tail, {self.n_boot} bootstrap reps)   {cv}",
            "  H0: no cointegration (with any single break)",
        ]
        return "\n".join(lines)


def _design_columns(
    n: int, break_idx: int, x: FloatArray, model: GHModel
) -> tuple[FloatArray, list[str]]:
    """Step-one design matrix for a candidate break at ``break_idx``.

    ``phi[t] = 1`` for observations strictly after ``break_idx`` (0-based),
    matching :math:`\\phi_t(\\tau) = 1\\{t > \\lfloor n\\tau \\rfloor\\}`.
    """
    phi = (np.arange(n) > break_idx).astype(np.float64)
    cols: list[FloatArray] = [np.ones(n, dtype=np.float64), phi]
    names = ["const", "phi"]
    if model in ("C/T", "C/S/T"):
        cols.append(np.arange(1, n + 1, dtype=np.float64))
        names.append("trend")
    k = x.shape[1]
    for j in range(k):
        cols.append(x[:, j])
        names.append(f"x{j}")
    if model in ("C/S", "C/S/T"):
        for j in range(k):
            cols.append(np.asarray(phi * x[:, j], dtype=np.float64))
            names.append(f"phi.x{j}")
    return np.column_stack(cols), names


def _grid_search(
    y: FloatArray,
    x: FloatArray,
    model: GHModel,
    break_indices: NDArray[np.int64],
    max_lags: int | None,
    ic: str,
) -> tuple[FloatArray, int, int]:
    """Run the step-one/step-two loop over every candidate break.

    Returns the array of ADF statistics per candidate, the index into
    ``break_indices`` of the minimum, and the lag order used there.
    """
    n = y.shape[0]
    stats = np.empty(break_indices.shape[0], dtype=np.float64)
    lags_at: list[int] = []
    for i, b in enumerate(break_indices):
        design, _ = _design_columns(n, int(b), x, model)
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        chosen, _ = select_lags(resid, method=ic, max_lags=max_lags)  # type: ignore[arg-type]
        fit = adf_regression(resid, chosen)
        stats[i] = fit.tstat
        lags_at.append(int(chosen))
    best = int(np.argmin(stats))
    return stats, best, lags_at[best]


def _bootstrap_null_replicate(
    rng: np.random.Generator,
    const: FloatArray,
    ar: FloatArray,
    resid: FloatArray,
    start_values: FloatArray,
    n: int,
    burn_in: int,
) -> FloatArray:
    """One simulated path of the joint system ``[y, x]`` under H0.

    The null of Gregory-Hansen is "no cointegration, with or without a
    break": ``y`` and ``x`` are jointly integrated but with no long-run
    relationship tying them together. A VAR fitted to the *differences*
    of the stacked system ``[y, x]``, with no error-correction term,
    encodes exactly that by construction — the same principle
    :func:`pyardl.bootstrap.dgp._fit_marginal_var` already applies to the
    regressors alone for the bounds-test bootstrap, reused here unchanged
    on the whole system since Gregory-Hansen's null has no conditional
    equation to keep separate.
    """
    order, k, _ = ar.shape
    total = n - 1 + burn_in
    draws = rng.integers(0, resid.shape[0], size=total)
    innovations = resid[draws]

    dz = np.zeros((total, k), dtype=np.float64)
    history = np.tile(start_values, (order, 1)) if order > 0 else np.empty((0, k))
    for t in range(total):
        val = const.copy()
        for i in range(order):
            val = val + ar[i] @ history[-(i + 1)]
        val = val + innovations[t]
        dz[t] = val
        if order > 0:
            history = np.vstack([history, val])[1:]

    dz_kept = dz[burn_in:]
    z0 = np.asarray(start_values, dtype=np.float64)
    z_path = np.vstack([z0, z0 + np.cumsum(dz_kept, axis=0)])
    return z_path


def gregory_hansen(
    y: ArrayLike,
    x: ArrayLike,
    model: GHModel = "C",
    trim: float = 0.15,
    max_lags: int | None = None,
    ic: str = "aic",
    cv_source: CVSource = "bootstrap",
    var_order: int = 1,
    n_boot: int = 999,
    seed: int | None = None,
) -> GregoryHansenResults:
    r"""Test for cointegration with an unknown, single regime shift.

    Parameters
    ----------
    y : array_like
        Dependent variable, shape ``(T,)``.
    x : array_like
        Regressors, shape ``(T, k)``.
    model : {'C', 'C/T', 'C/S', 'C/S/T'}, default 'C'
        Break specification (Gregory & Hansen 1996, section 2): level
        shift only ('C'), level shift plus trend ('C/T'), level and slope
        shift ('C/S'), or all three ('C/S/T'). All four are implemented;
        there is no silent fallback to a single case.
    trim : float, default 0.15
        Fraction of the sample excluded at each end of the break-fraction
        grid, so the two regimes each keep enough observations for
        step-two ADF to be estimable.
    max_lags : int, optional
        Upper bound for the step-two lag search at each candidate break.
        Defaults to the Schwert rule (see
        :func:`pyardl.unitroot.gls.select_lags`).
    ic : str, default 'aic'
        Criterion for the step-two lag order, passed through to
        :func:`pyardl.unitroot.gls.select_lags`.
    cv_source : {'bootstrap'}, default 'bootstrap'
        How the critical values of ``ADF*`` are obtained. A supremum over
        the break grid has a non-standard asymptotic distribution
        (the same "problem of Davies" as the Fourier frequency pre-test),
        so it cannot reuse Engle-Granger/MacKinnon critical values.
        The published Gregory-Hansen asymptotic table (their Table 1) is
        **not yet encoded**: pyardl does not currently hold that table
        with a verified provenance (CLAUDE.md rule 9 — no critical-value
        table is written from memory), so ``cv_source='table'`` is not
        implemented. See ``docs/QUESTIONS.md``.
    var_order : int, default 1
        Lag order of the joint VAR-in-differences the bootstrap null
        model is fitted on (see :func:`_bootstrap_null_replicate`).
    n_boot : int, default 999
        Bootstrap replications for the critical values.
    seed : int, optional
        Seed for the bootstrap's :class:`numpy.random.Generator`. Logged
        on the result for reproducibility.

    Returns
    -------
    GregoryHansenResults

    Notes
    -----
    A break *searched for* on the data, rather than fixed a priori,
    changes the law of the test statistic: never compare ``ADF*`` to an
    ordinary single-break-free ADF table. This is a single-break test —
    for multiple breaks, see :func:`pyardl.cointegration.bai_perron`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.cointegration import gregory_hansen
    >>> rng = np.random.default_rng(0)
    >>> n = 150
    >>> x = np.cumsum(rng.standard_normal(n))
    >>> u = rng.standard_normal(n) * 0.5
    >>> beta = np.where(np.arange(n) > 75, 2.5, 1.0)
    >>> y = beta * x + u
    >>> res = gregory_hansen(y, x, model="C/S", n_boot=99, seed=0)
    >>> res.decision(0.05)
    'cointegration'
    """
    valid_models: tuple[GHModel, ...] = ("C", "C/T", "C/S", "C/S/T")
    if model not in valid_models:
        raise ValueError(f"model={model!r} must be one of {valid_models}.")
    if cv_source != "bootstrap":
        raise NotImplementedError(
            "cv_source='table' is not implemented: pyardl does not hold a "
            "verified copy of the Gregory-Hansen (1996) published critical "
            "value table (CLAUDE.md rule 9 forbids encoding a table from "
            "memory). Use cv_source='bootstrap' (the default). See "
            "docs/QUESTIONS.md."
        )
    if not 0.0 < trim < 0.5:
        raise ValueError(f"trim={trim} must be in (0, 0.5).")

    y_arr, x_arr, index, y_name, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("Gregory-Hansen needs at least one regressor.")
    n = y_arr.shape[0]

    lo = int(np.ceil(trim * n))
    hi = int(np.floor((1.0 - trim) * n))
    if hi - lo < 2:
        raise ValueError(
            f"Sample too short for trim={trim} on n={n} observations: the "
            f"break grid has fewer than 2 candidates."
        )
    break_indices = np.arange(lo, hi + 1)

    stats, best, best_lags = _grid_search(
        y_arr, x_arr, model, break_indices, max_lags, ic
    )
    best_break = int(break_indices[best])
    tau_hat = (best_break + 1) / n
    statistic = float(stats[best])

    grid = pd.DataFrame(
        {"tau": (break_indices + 1) / n, "adf_stat": stats}, index=break_indices
    )
    break_date = index[best_break] if index is not None else None

    rng = np.random.default_rng(seed)
    stacked = np.column_stack([y_arr, x_arr])
    d_stacked = np.diff(stacked, axis=0)
    const, ar, resid = _fit_marginal_var(d_stacked, var_order)
    resid = resid - resid.mean(axis=0)

    boot_stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        path = _bootstrap_null_replicate(
            rng, const, ar, resid, stacked[0], n, burn_in=50
        )
        y_b, x_b = path[:, 0], path[:, 1:]
        boot_stats_b, _, _ = _grid_search(y_b, x_b, model, break_indices, max_lags, ic)
        boot_stats[b] = boot_stats_b.min()

    critical_values = {
        alpha: float(np.quantile(boot_stats, alpha)) for alpha in (0.01, 0.05, 0.10)
    }

    return GregoryHansenResults(
        statistic=statistic,
        break_fraction=tau_hat,
        break_index=best_break,
        break_date=break_date,
        critical_values=critical_values,
        model=model,
        cv_source=cv_source,
        grid=grid,
        lags=best_lags,
        n_boot=n_boot,
        seed=seed,
    )
