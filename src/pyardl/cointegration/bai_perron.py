r"""Bai & Perron (1998, 2003) — multiple structural breaks.

Gregory-Hansen (:mod:`pyardl.cointegration.gregory_hansen`) detects at
most one break. A macro series spanning decades often crosses several
regimes, and a single tested break where there are two or three
under-counts them. Bai-Perron generalises the question: how many breaks,
and where, estimated jointly by minimising the residual sum of squares
over every admissible partition of the sample.

The model estimated within each segment is an ordinary least-squares
regression, exactly as in Gregory-Hansen — what is new here is the
search algorithm over partitions (dynamic programming) and the inference
on their number, not a new pointwise estimator.

Scope of this implementation
-----------------------------
Spec 31 §2.4 distinguishes two uses, deliberately not to be conflated:
(a) a **diagnostic** on any already-specified regression, testing
stability beyond what CUSUM/CUSUMSQ can date or count; (b) a
generalisation of Gregory-Hansen to *m* breaks in a cointegration
setting (spec 29 becoming its m=1 special case). This module implements
(a) only — it assumes the residuals of the fitted segments are the
object of interest and does not build the I(1)-specific bootstrap null
that (b) would need. Applying it to Engle-Granger's step-one residuals
as a multi-break cointegration test (spec 31 §2.4b) is future work; see
``docs/DEVIATIONS.md``.

Spec 31 §2.3 lists three routes to choose the number of breaks: sequential
sup-F tests, UDmax/WDmax, and information criteria. Only the first and
third are implemented here. UDmax/WDmax and the tabulated Bai & Perron
(1998) critical values for the sequential test are **not implemented**:
this module never encodes a table it cannot verify (CLAUDE.md rule 9),
so ``selection='sequential'`` uses a bootstrap null instead of the
published table, and ``selection='udmax'`` raises ``NotImplementedError``
outright rather than shipping an unweighted approximation under a name
that promises the real (weighted) test. See ``docs/QUESTIONS.md`` and
``docs/DEVIATIONS.md``.

References
----------
.. [1] Bai, J. & Perron, P. (1998). Estimating and testing linear models
       with multiple structural changes. *Econometrica*, 66(1), 47-78.
.. [2] Bai, J. & Perron, P. (2003). Computation and analysis of multiple
       structural change models. *Journal of Applied Econometrics*,
       18(1), 1-22.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

Selection = Literal["ic", "sequential"]
ICMethod = Literal["aic", "bic"]

__all__ = ["bai_perron", "BaiPerronResults"]


@dataclass(frozen=True)
class BaiPerronResults:
    """Outcome of a Bai-Perron multiple structural-change search.

    Attributes
    ----------
    n_breaks : int
        Selected number of breaks, :math:`\\hat m`.
    break_indices : tuple of int
        0-based indices of the last observation of each segment except
        the final one (so segment ``j`` runs from ``break_indices[j-1]``
        inclusive-plus-one to ``break_indices[j]`` inclusive).
    break_dates : tuple or None
        Corresponding values of the original index, when the input
        carried a pandas index; ``None`` otherwise.
    break_fractions : tuple of float
        Break points as fractions of the sample.
    segment_params : list of pandas.Series
        OLS coefficients :math:`\\hat\\beta_j` of each of the
        :math:`\\hat m + 1` segments.
    ssr : float
        Residual sum of squares of the selected partition.
    ic_values : dict
        Number of breaks (0 to ``max_breaks``) to its information
        criterion value, so the choice can be inspected rather than
        trusted.
    sequential_tests : dict or None
        When ``selection='sequential'``: number of existing breaks
        :math:`l` to ``{'stat', 'pvalue', 'break_added'}`` for every step
        actually run. ``None`` under ``selection='ic'``.
    selection : str
        Route used to choose :math:`\\hat m`.
    ic : str or None
        Criterion used when ``selection='ic'``.
    max_breaks : int
        Upper bound on the number of breaks searched.
    trim : float
        Minimum segment length, as a fraction of the sample.
    n_boot : int or None
        Bootstrap replications used by ``selection='sequential'``.
    seed : int or None
        Seed of the bootstrap generator.
    """

    n_breaks: int
    break_indices: tuple[int, ...]
    break_dates: tuple[object, ...] | None
    break_fractions: tuple[float, ...]
    segment_params: list[pd.Series]
    ssr: float
    ic_values: dict[int, float]
    sequential_tests: dict[int, dict[str, float]] | None
    selection: Selection
    ic: ICMethod | None
    max_breaks: int
    trim: float
    n_boot: int | None = field(default=None)
    seed: int | None = field(default=None)

    def summary(self) -> str:
        """Readable report of the search."""
        lines = [
            f"Bai-Perron multiple structural changes (1998, 2003) - "
            f"selection={self.selection}, trim={self.trim}, "
            f"max_breaks={self.max_breaks}",
            f"  n_breaks (m_hat) = {self.n_breaks}",
        ]
        if self.break_fractions:
            frac_txt = ", ".join(f"{f:.3f}" for f in self.break_fractions)
            lines.append(f"  break fractions = [{frac_txt}]")
        lines.append(f"  SSR({self.n_breaks}) = {self.ssr:.4f}")
        if self.ic is not None:
            ic_txt = "  ".join(
                f"m={m}: {v:.4f}" for m, v in sorted(self.ic_values.items())
            )
            lines.append(f"  {self.ic.upper()}(m)   {ic_txt}")
        if self.sequential_tests:
            lines.append("  Sequential sup-F(l+1|l) tests (bootstrap p-values):")
            for level, res in sorted(self.sequential_tests.items()):
                lines.append(
                    f"    l={level}: stat={res['stat']:.4f}  "
                    f"p-value={res['pvalue']:.4f}  "
                    f"break_added={'yes' if res['break_added'] else 'no'}"
                )
        return "\n".join(lines)


def _check_regressors(x: ArrayLike, n_y: int) -> tuple[FloatArray, list[str]]:
    """Validate ``x`` without :func:`pyardl.utils.check_series`'s ban on
    a zero-variance column — a constant is a legitimate regressor here.
    """
    names: list[str]
    if isinstance(x, pd.DataFrame):
        names = [str(c) for c in x.columns]
    elif isinstance(x, pd.Series):
        names = [str(x.name) if x.name is not None else "x0"]
    else:
        names = []
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if x_arr.ndim != 2:
        raise ValueError("x must be 1-D or 2-D.")
    if not names:
        names = [f"x{j}" for j in range(x_arr.shape[1])]
    if x_arr.shape[0] != n_y:
        raise ValueError(
            f"Incompatible lengths: y has {n_y} observations, x has {x_arr.shape[0]}."
        )
    if np.isnan(x_arr).any():
        raise ValueError("x must not contain NaN.")
    return x_arr, names


def _segment_ssr_and_beta(
    y: FloatArray, x: FloatArray, i: int, j: int
) -> tuple[float, FloatArray]:
    """OLS on the half-open segment ``[i, j)`` — SSR and coefficients."""
    y_seg = y[i:j]
    x_seg = x[i:j]
    beta, _, _, _ = np.linalg.lstsq(x_seg, y_seg, rcond=None)
    resid = y_seg - x_seg @ beta
    return float(resid @ resid), np.asarray(beta, dtype=np.float64)


def _dp_partitions(
    y: FloatArray, x: FloatArray, h: int, max_breaks: int
) -> tuple[dict[int, float], dict[int, tuple[int, ...]]]:
    """Bai-Perron dynamic programme: optimal SSR and breaks for each m.

    ``dp[m][T]`` is the minimal cumulative SSR of partitioning ``[0, T)``
    into ``m + 1`` segments, each at least ``h`` observations long. This
    is the O(T^2) recursion of Bai & Perron (2003) section 3: solving it
    once for every ``m`` up to ``max_breaks`` is what makes the search
    tractable beyond a handful of breaks, not a statistical shortcut.
    """
    n = y.shape[0]
    ssr_cache: dict[tuple[int, int], float] = {}

    def ssr(i: int, j: int) -> float:
        key = (i, j)
        cached = ssr_cache.get(key)
        if cached is None:
            cached, _ = _segment_ssr_and_beta(y, x, i, j)
            ssr_cache[key] = cached
        return cached

    # dp[0][T] : single segment [0, T).
    dp: list[dict[int, float]] = [{}]
    back: list[dict[int, int]] = [{}]
    for t in range(h, n + 1):
        dp[0][t] = ssr(0, t)

    for m in range(1, max_breaks + 1):
        dp.append({})
        back.append({})
        min_t = (m + 1) * h
        for t in range(min_t, n + 1):
            best_val = np.inf
            best_b = -1
            for b in range(m * h, t - h + 1):
                if b not in dp[m - 1]:
                    continue
                val = dp[m - 1][b] + ssr(b, t)
                if val < best_val:
                    best_val = val
                    best_b = b
            if best_b >= 0:
                dp[m][t] = best_val
                back[m][t] = best_b

    best_ssr = {m: dp[m][n] for m in range(max_breaks + 1) if n in dp[m]}
    best_breaks: dict[int, tuple[int, ...]] = {}
    for m in best_ssr:
        breaks: list[int] = []
        t = n
        for level in range(m, 0, -1):
            b = back[level][t]
            breaks.append(b)
            t = b
        best_breaks[m] = tuple(reversed(breaks))
    return best_ssr, best_breaks


def _segment_params(
    y: FloatArray, x: FloatArray, breaks: tuple[int, ...], names: list[str]
) -> list[pd.Series]:
    n = y.shape[0]
    bounds = [0, *breaks, n]
    params = []
    for i in range(len(bounds) - 1):
        _, beta = _segment_ssr_and_beta(y, x, bounds[i], bounds[i + 1])
        params.append(pd.Series(beta, index=names, name=f"segment_{i}"))
    return params


def _information_criterion(
    ssr: float, n: int, n_params: int, method: ICMethod
) -> float:
    """Standard AIC/BIC on ``SSR(m)``.

    Notes
    -----
    Spec 31 §2.3.3 names Yao's (1988) *modified* BIC and the Liu-Wu-Zidek
    (1997) criterion specifically. Their exact penalty terms are not
    reproduced here with a verified source (CLAUDE.md rule 9 forbids
    writing a formula's constants from memory without one) — this
    function implements the textbook AIC/BIC on ``SSR(m)/n`` instead,
    which selects consistently but is not guaranteed to coincide with
    the modified variants in finite samples. Logged as a known deviation
    in ``docs/QUESTIONS.md``.
    """
    penalty = 2.0 if method == "aic" else float(np.log(n))
    return float(n * np.log(ssr / n) + penalty * n_params)


def _bootstrap_supf(
    rng: np.random.Generator,
    y_fitted: FloatArray,
    resid: FloatArray,
    x: FloatArray,
    h: int,
    existing_breaks: tuple[int, ...],
    observed_stat: float,
    n_boot: int,
) -> float:
    """Bootstrap p-value of the sup-F(l+1|l) statistic.

    Residual (pairs-free) bootstrap under H0 "l breaks are enough": the
    fitted values of the l-break model are held fixed, residuals are
    resampled with replacement and added back, and the same
    add-one-break search is rerun on the bootstrap sample. This is the
    classic residual bootstrap for a fixed-design piecewise regression —
    no I(1) structure is assumed, unlike the Gregory-Hansen null
    (:mod:`pyardl.cointegration.gregory_hansen`), because Bai-Perron here
    is applied as a diagnostic on an already-stationary regression.
    """
    n = y_fitted.shape[0]
    exceed = 0
    for _ in range(n_boot):
        y_b = y_fitted + rng.choice(resid, size=n, replace=True)
        stat_b = _add_one_break_stat(y_b, x, h, existing_breaks)
        if stat_b >= observed_stat:
            exceed += 1
    return (exceed + 1) / (n_boot + 1)


def _add_one_break_stat(
    y: FloatArray, x: FloatArray, h: int, existing_breaks: tuple[int, ...]
) -> float:
    """sup-F(l+1|l): best single extra break added to any existing segment."""
    n, k = x.shape
    bounds = [0, *existing_breaks, n]
    ssr_l = 0.0
    for i in range(len(bounds) - 1):
        seg_ssr, _ = _segment_ssr_and_beta(y, x, bounds[i], bounds[i + 1])
        ssr_l += seg_ssr

    best_reduction = -np.inf
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        seg_ssr, _ = _segment_ssr_and_beta(y, x, lo, hi)
        for b in range(lo + h, hi - h + 1):
            left, _ = _segment_ssr_and_beta(y, x, lo, b)
            right, _ = _segment_ssr_and_beta(y, x, b, hi)
            reduction = seg_ssr - (left + right)
            if reduction > best_reduction:
                best_reduction = reduction

    if best_reduction <= 0:
        return 0.0
    dof = n - (len(bounds)) * k - k
    if dof <= 0 or ssr_l <= best_reduction:
        return 0.0
    return float(best_reduction / (ssr_l - best_reduction) * dof / k)


def bai_perron(
    y: ArrayLike,
    x: ArrayLike,
    max_breaks: int = 5,
    trim: float = 0.15,
    selection: Selection = "ic",
    ic: ICMethod = "bic",
    n_boot: int = 199,
    alpha: float = 0.05,
    seed: int | None = None,
) -> BaiPerronResults:
    r"""Search for multiple structural breaks by SSR minimisation.

    Parameters
    ----------
    y : array_like
        Dependent variable, shape ``(T,)``.
    x : array_like
        Regressors, shape ``(T, k)``. No constant is added automatically
        — include one explicitly if the model needs it, exactly as the
        spec's model :math:`y_t = x_t'\beta_j + u_t` is written.
    max_breaks : int, default 5
        Upper bound on the number of breaks searched.
    trim : float, default 0.15
        Minimum segment length, as a fraction of the sample (each
        segment must hold at least ``ceil(trim * T)`` observations, and
        never fewer than ``k`` — see the module notes on the O(T^2)
        dynamic programme).
    selection : {'ic', 'sequential'}, default 'ic'
        Route used to choose the number of breaks.
        ``'ic'`` minimises an information criterion over ``SSR(m)``,
        ``m = 0, ..., max_breaks`` (see ``ic``).
        ``'sequential'`` starts at 0 breaks and, at each step, tests
        whether adding one more break to the current partition reduces
        SSR by more than chance, using a bootstrap p-value (**not** the
        tabulated Bai & Perron 1998 critical values — see the module
        notes); stops the first time the test fails to reject at
        ``alpha``. ``'udmax'`` (spec 31 §2.3.2) is not implemented; see
        the module notes.
    ic : {'aic', 'bic'}, default 'bic'
        Criterion for ``selection='ic'``. See the module notes: this is
        the textbook AIC/BIC, not the modified variants the spec names.
    n_boot : int, default 199
        Bootstrap replications per step of ``selection='sequential'``.
        The smallest attainable p-value is ``1 / (n_boot + 1)``: pick
        ``n_boot`` well above ``1 / alpha - 1``, or the test can never
        reject even when every replicate is beaten.
    alpha : float, default 0.05
        Significance level of the sequential stopping rule.
    seed : int, optional
        Seed for the bootstrap's :class:`numpy.random.Generator`.

    Returns
    -------
    BaiPerronResults

    Notes
    -----
    A break is estimated only as the boundary between two OLS-fitted
    segments — it is assumed sharp, not a gradual transition (see
    :mod:`pyardl.fourier` or the smooth-transition extension for that),
    and the regressors are assumed exogenous to the break dates
    themselves.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.cointegration import bai_perron
    >>> rng = np.random.default_rng(0)
    >>> n = 240
    >>> const = np.ones(n)
    >>> beta = np.where(np.arange(n) < 80, 1.0, np.where(np.arange(n) < 160, 3.0, -1.0))
    >>> y = beta + rng.standard_normal(n) * 0.3
    >>> res = bai_perron(y, const.reshape(-1, 1), max_breaks=4, selection="ic", seed=0)
    >>> res.n_breaks
    2
    """
    if selection == "udmax":  # type: ignore[comparison-overlap]
        raise NotImplementedError(
            "selection='udmax' is not implemented: the weighted UDmax/WDmax "
            "test needs the Bai-Perron (1998) tabulated critical values/"
            "weights, which pyardl does not hold with a verified provenance "
            "(CLAUDE.md rule 9). Use selection='ic' or 'sequential'. See "
            "docs/QUESTIONS.md."
        )
    if selection not in ("ic", "sequential"):
        raise ValueError(f"selection={selection!r} must be 'ic' or 'sequential'.")
    if not 0.0 < trim < 0.5:
        raise ValueError(f"trim={trim} must be in (0, 0.5).")
    if max_breaks < 1:
        raise ValueError(f"max_breaks={max_breaks} must be >= 1.")

    # check_series() rejects a zero-variance regressor column, which is
    # the right default for a long-run cointegrating vector (spec 08/10)
    # but wrong here: unlike Gregory-Hansen, Bai-Perron's model explicitly
    # allows x to carry a constant (spec 31 §2.1). y is still validated
    # through check_series; x gets its own, lighter check.
    y_arr, _, index, y_name, _ = check_series(y, None)
    x_arr, x_names = _check_regressors(x, y_arr.shape[0])
    n, k = x_arr.shape
    h = max(int(np.ceil(trim * n)), k + 1)
    if (max_breaks + 1) * h > n:
        raise ValueError(
            f"Sample too short: max_breaks={max_breaks} at trim={trim} "
            f"(segment length >= {h}) needs at least {(max_breaks + 1) * h} "
            f"observations, got {n}."
        )

    ssr_by_m, breaks_by_m = _dp_partitions(y_arr, x_arr, h, max_breaks)

    if selection == "ic":
        ic_values = {
            m: _information_criterion(ssr_by_m[m], n, (m + 1) * k, ic) for m in ssr_by_m
        }
        m_hat = min(ic_values, key=lambda m: ic_values[m])
        sequential_tests_opt: dict[int, dict[str, float]] | None = None
    else:
        rng = np.random.default_rng(seed)
        sequential_results: dict[int, dict[str, float]] = {}
        m_hat = 0
        for level in range(max_breaks):
            current_breaks = breaks_by_m[level]
            bounds = [0, *current_breaks, n]
            fitted = np.empty(n, dtype=np.float64)
            resid = np.empty(n, dtype=np.float64)
            for i in range(len(bounds) - 1):
                lo, hi = bounds[i], bounds[i + 1]
                _, b = _segment_ssr_and_beta(y_arr, x_arr, lo, hi)
                fitted[lo:hi] = x_arr[lo:hi] @ b
                resid[lo:hi] = y_arr[lo:hi] - fitted[lo:hi]
            stat = _add_one_break_stat(y_arr, x_arr, h, current_breaks)
            pvalue = _bootstrap_supf(
                rng, fitted, resid, x_arr, h, current_breaks, stat, n_boot
            )
            accept = pvalue < alpha
            sequential_results[level] = {
                "stat": stat,
                "pvalue": pvalue,
                "break_added": float(accept),
            }
            if not accept:
                break
            m_hat = level + 1
        ic_values = {
            m: _information_criterion(ssr_by_m[m], n, (m + 1) * k, "bic")
            for m in ssr_by_m
        }
        sequential_tests_opt = sequential_results

    breaks = breaks_by_m[m_hat]
    names = list(x_names)
    return BaiPerronResults(
        n_breaks=m_hat,
        break_indices=breaks,
        break_dates=tuple(index[b] for b in breaks) if index is not None else None,
        break_fractions=tuple((b + 1) / n for b in breaks),
        segment_params=_segment_params(y_arr, x_arr, breaks, names),
        ssr=ssr_by_m[m_hat],
        ic_values=ic_values,
        sequential_tests=sequential_tests_opt,
        selection=selection,
        ic=ic if selection == "ic" else None,
        max_breaks=max_breaks,
        trim=trim,
        n_boot=n_boot if selection == "sequential" else None,
        seed=seed,
    )
