r"""Parameter-constancy tests of Brown, Durbin & Evans (1975).

An ARDL fitted over a long sample assumes its coefficients did not move.
That assumption is rarely stated and almost never checked, yet a break in
the middle of the sample invalidates every long-run coefficient the model
produces. These two tests check it, and applied work is expected to
report both.

Both are built on the **recursive residuals**. Estimate the model on the
first :math:`t-1` observations, predict :math:`y_t`, and standardise the
prediction error:

.. math::

    w_t = \frac{y_t - x_t' b_{t-1}}
               {\sqrt{1 + x_t' (X'X)_{t-1}^{-1} x_t}},
    \qquad t = k+1, \dots, T

Under constant parameters the :math:`w_t` are i.i.d.
:math:`N(0, \sigma^2)`, whatever the data look like. A break makes them
drift or change scale, and each test looks for one of those two
symptoms:

- **CUSUM** cumulates the residuals, so it reacts to a shift in the
  *mean* of the coefficients — a slow drift shows as a departing path.
- **CUSUM of squares** cumulates their squares, so it reacts to a change
  in *variance*. It catches breaks the CUSUM is blind to.

Reporting one without the other leaves half the failure modes untested,
and the gap is wider than it looks. A break in the slope on a
zero-mean regressor leaves the recursive residuals centred on zero: the
CUSUM path stays flat and the test sees nothing, while the inflated
variance pushes the CUSUM of squares straight out of its band. The two
tests are not two views of the same thing.

References
----------
.. [1] Brown, R. L., Durbin, J. & Evans, J. M. (1975). Techniques for
       testing the constancy of regression relationships over time.
       *Journal of the Royal Statistical Society B*, 37(2), 149-192.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyardl.critical_values.bde1975 import cusum_a, cusumsq_c0

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]
    IntArray = NDArray[np.intp]

__all__ = [
    "recursive_residuals",
    "cusum",
    "cusumsq",
    "stability_tests",
    "plot_cusum",
    "plot_cusumsq",
    "CUSUMResults",
]


def recursive_residuals(y: ArrayLike, x: ArrayLike) -> FloatArray:
    r"""Standardised one-step-ahead prediction errors of a linear model.

    Parameters
    ----------
    y : array_like
        Dependent variable, shape ``(T,)``.
    x : array_like
        Design matrix, shape ``(T, k)``, deterministic terms included.

    Returns
    -------
    numpy.ndarray
        The ``T - k`` recursive residuals :math:`w_{k+1}, \dots, w_T`,
        i.i.d. :math:`N(0, \sigma^2)` under constant parameters.

    Raises
    ------
    ValueError
        If the sample is too short to leave any recursive residual, or if
        the first ``k`` rows are collinear, which leaves the recursion
        without a starting point.

    Notes
    -----
    The recursion updates :math:`(X'X)^{-1}` in place by the
    Sherman-Morrison identity instead of re-estimating the model
    :math:`T-k` times:

    .. math::

        f_t &= 1 + x_t' A_{t-1} x_t \\
        b_t &= b_{t-1} + A_{t-1} x_t (y_t - x_t' b_{t-1}) / f_t \\
        A_t &= A_{t-1} - A_{t-1} x_t x_t' A_{t-1} / f_t

    This is what makes the test usable on long series: the cost is linear
    in ``T`` rather than cubic. The result is exact, not an approximation
    — it agrees with a full re-estimation to machine precision.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.diagnostics import recursive_residuals
    >>> rng = np.random.default_rng(0)
    >>> x = np.column_stack([np.ones(50), rng.standard_normal(50)])
    >>> y = x @ np.array([1.0, 0.5]) + rng.standard_normal(50)
    >>> w = recursive_residuals(y, x)
    >>> w.shape
    (48,)
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Incompatible lengths: y has {y_arr.shape[0]} observations, "
            f"x has {x_arr.shape[0]}."
        )
    n_obs, k = x_arr.shape
    if n_obs <= k:
        raise ValueError(
            f"Sample too short: {n_obs} observations for {k} regressors "
            "leaves no recursive residual."
        )

    x0 = x_arr[:k]
    if np.linalg.matrix_rank(x0) < k:
        raise ValueError(
            f"The first {k} rows of the design matrix are collinear, so the "
            "recursion has no starting point. Reorder the observations or "
            "drop the redundant regressor."
        )

    a_inv = np.linalg.inv(x0.T @ x0)
    beta = a_inv @ (x0.T @ y_arr[:k])

    w = np.empty(n_obs - k, dtype=np.float64)
    for i in range(n_obs - k):
        xi = x_arr[k + i]
        a_xi = a_inv @ xi
        f_t = 1.0 + float(xi @ a_xi)
        err = float(y_arr[k + i] - xi @ beta)
        w[i] = err / np.sqrt(f_t)
        beta = beta + a_xi * (err / f_t)
        a_inv = a_inv - np.outer(a_xi, a_xi) / f_t
    return w


@dataclass(frozen=True)
class CUSUMResults:
    """Outcome of a CUSUM or CUSUM-of-squares test.

    Attributes
    ----------
    test : str
        ``"cusum"`` or ``"cusumsq"``.
    statistic : numpy.ndarray
        The cumulated path, one value per recursive residual.
    lower, upper : numpy.ndarray
        The rejection boundaries, aligned with ``statistic``.
    stable : bool
        ``True`` when the path stays inside the boundaries throughout.
    max_excess : float
        Largest excursion beyond a boundary, in the units of the
        statistic. Zero when the path never leaves the band. It measures
        how far the model is from stability, not merely whether it
        crossed.
    crossings : numpy.ndarray
        Positions, in the original sample, where the path lies outside
        the band. Reading them locates *when* the break happened, which
        the boolean alone cannot.
    alpha : float
        Significance level used for the boundaries.
    nobs, n_recursive, k : int
        Sample size, number of recursive residuals, number of regressors.
    """

    test: Literal["cusum", "cusumsq"]
    statistic: FloatArray
    lower: FloatArray
    upper: FloatArray
    stable: bool
    max_excess: float
    crossings: IntArray
    alpha: float
    nobs: int
    n_recursive: int
    k: int
    index: pd.Index | None = field(default=None, repr=False)

    def summary(self) -> str:
        """One-line verdict, with the location of the first crossing."""
        name = "CUSUM" if self.test == "cusum" else "CUSUM of squares"
        verdict = "stable" if self.stable else "UNSTABLE"
        line = (
            f"{name} ({int(self.alpha * 100)}%): {verdict}"
            f"  [n={self.n_recursive}, k={self.k}]"
        )
        if not self.stable:
            first = self.crossings[0]
            label = (
                str(self.index[int(first)])
                if self.index is not None
                else f"observation {int(first) + 1}"
            )
            line += (
                f"\n  first crossing at {label}"
                f", max excess = {self.max_excess:.4f}"
                f", {len(self.crossings)} point(s) outside the band"
            )
        return line


def _prepare(
    y: ArrayLike, x: ArrayLike
) -> tuple[FloatArray, int, int, pd.Index | None, float]:
    index = y.index if isinstance(y, pd.Series) else None
    w = recursive_residuals(y, x)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    # Scale against which a residual sum counts as numerically zero. A
    # model that fits perfectly leaves residuals of the order of rounding
    # error, never exactly zero, so an ``== 0`` guard would let pure
    # floating-point noise through and report it as a statistic.
    scale = float(y_arr @ y_arr) * float(np.finfo(np.float64).eps) * y_arr.size
    return w, x_arr.shape[0], x_arr.shape[1], index, scale


def _reject_degenerate(total: float, scale: float, statistic: str) -> None:
    if not np.isfinite(total) or total <= scale:
        raise ValueError(
            f"The recursive residuals are numerically zero (sum of squares "
            f"{total:.3e} against a rounding-error scale of {scale:.3e}): the "
            f"{statistic} statistic carries no information. This happens when "
            "the model fits exactly, typically because a regressor "
            "reproduces the dependent variable."
        )


def cusum(y: ArrayLike, x: ArrayLike, alpha: float = 0.05) -> CUSUMResults:
    r"""CUSUM test for constancy of the regression coefficients.

    Parameters
    ----------
    y : array_like
        Dependent variable, shape ``(T,)``. A pandas Series keeps its
        index, so crossings are reported as dates.
    x : array_like
        Design matrix, shape ``(T, k)``.
    alpha : float, default 0.05
        Significance level. One of 0.10, 0.05, 0.01 — the only levels
        Brown, Durbin & Evans tabulated.

    Returns
    -------
    CUSUMResults

    Notes
    -----
    The statistic is
    :math:`W_t = \sum_{s \le t} w_s / \hat\sigma_w`, compared with the
    lines :math:`\pm [a\sqrt{n} + 2a(t-k)/\sqrt{n}]`, where
    :math:`n = T-k` and :math:`\hat\sigma_w` is the standard deviation of
    the recursive residuals.

    The boundaries widen with :math:`t`, which is not a detail: a random
    walk spreads as :math:`\sqrt{t}`, so constant-width bands would
    reject far too often late in the sample.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.diagnostics import cusum
    >>> rng = np.random.default_rng(1)
    >>> x = np.column_stack([np.ones(120), rng.standard_normal(120)])
    >>> y = x @ np.array([1.0, 0.5]) + rng.standard_normal(120)
    >>> cusum(y, x).stable
    True
    """
    w, n_obs, k, index, scale = _prepare(y, x)
    n = w.size
    _reject_degenerate(float(w @ w), scale, "CUSUM")
    sigma = float(np.std(w, ddof=1))
    stat = np.asarray(np.cumsum(w) / sigma, dtype=np.float64)

    a = cusum_a(alpha)
    steps = np.arange(1, n + 1, dtype=np.float64)
    half_width = np.asarray(
        a * np.sqrt(n) + 2.0 * a * steps / np.sqrt(n), dtype=np.float64
    )
    lower, upper = -half_width, half_width

    outside = (stat > upper) | (stat < lower)
    excess = np.maximum(stat - upper, lower - stat)
    return CUSUMResults(
        test="cusum",
        statistic=stat,
        lower=lower,
        upper=upper,
        stable=not bool(outside.any()),
        max_excess=float(max(0.0, excess.max())),
        crossings=(np.flatnonzero(outside) + k).astype(np.intp),
        alpha=alpha,
        nobs=n_obs,
        n_recursive=n,
        k=k,
        index=index,
    )


def cusumsq(y: ArrayLike, x: ArrayLike, alpha: float = 0.05) -> CUSUMResults:
    r"""CUSUM-of-squares test for constancy of the error variance.

    Parameters
    ----------
    y : array_like
        Dependent variable, shape ``(T,)``.
    x : array_like
        Design matrix, shape ``(T, k)``.
    alpha : float, default 0.05
        Significance level. One of 0.10, 0.05, 0.01.

    Returns
    -------
    CUSUMResults

    Notes
    -----
    The statistic is
    :math:`S_t = \sum_{s \le t} w_s^2 / \sum_{s \le n} w_s^2`, compared
    with :math:`(t-k)/(T-k) \pm c_0`. It runs from 0 to 1 by
    construction, so a departure means the squared residuals accumulated
    faster or slower than a constant variance would allow.

    This test complements rather than duplicates the CUSUM: a pure
    variance break leaves the CUSUM path near zero while pushing this one
    out of its band.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.diagnostics import cusumsq
    >>> rng = np.random.default_rng(1)
    >>> x = np.column_stack([np.ones(120), rng.standard_normal(120)])
    >>> y = x @ np.array([1.0, 0.5]) + rng.standard_normal(120)
    >>> cusumsq(y, x).stable
    True
    """
    w, n_obs, k, index, scale = _prepare(y, x)
    n = w.size
    total = float(w @ w)
    _reject_degenerate(total, scale, "CUSUM-of-squares")
    stat = np.asarray(np.cumsum(w**2) / total, dtype=np.float64)

    expected = np.arange(1, n + 1, dtype=np.float64) / n
    c0 = cusumsq_c0(n, alpha)
    lower = np.asarray(expected - c0, dtype=np.float64)
    upper = np.asarray(expected + c0, dtype=np.float64)

    outside = (stat > upper) | (stat < lower)
    excess = np.maximum(stat - upper, lower - stat)
    return CUSUMResults(
        test="cusumsq",
        statistic=stat,
        lower=lower,
        upper=upper,
        stable=not bool(outside.any()),
        max_excess=float(max(0.0, excess.max())),
        crossings=(np.flatnonzero(outside) + k).astype(np.intp),
        alpha=alpha,
        nobs=n_obs,
        n_recursive=n,
        k=k,
        index=index,
    )


def stability_tests(y: ArrayLike, x: ArrayLike, alpha: float = 0.05) -> pd.DataFrame:
    """Run both stability tests and tabulate the verdicts.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like
        Design matrix.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pandas.DataFrame
        Rows ``CUSUM`` and ``CUSUM-of-squares``; columns ``stable``,
        ``max_excess`` and ``first_crossing`` (``NaN`` when stable).

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.diagnostics import stability_tests
    >>> rng = np.random.default_rng(2)
    >>> x = np.column_stack([np.ones(150), rng.standard_normal(150)])
    >>> y = x @ np.array([1.0, 0.5]) + rng.standard_normal(150)
    >>> stability_tests(y, x)["stable"].tolist()
    [True, True]
    """
    results = (cusum(y, x, alpha), cusumsq(y, x, alpha))
    return pd.DataFrame(
        {
            "stable": [r.stable for r in results],
            "max_excess": [r.max_excess for r in results],
            "first_crossing": [
                float(r.crossings[0]) if r.crossings.size else np.nan for r in results
            ],
        },
        index=pd.Index(["CUSUM", "CUSUM-of-squares"], name="test"),
    )


def _plot(res: CUSUMResults, title: str, ylabel: str):  # type: ignore[no-untyped-def]
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Plotting requires matplotlib, an optional dependency. "
            "Install it with: pip install 'pyardl[plot]'"
        ) from exc

    positions = np.arange(res.k, res.k + res.n_recursive)
    xs = res.index[positions] if res.index is not None else positions
    fig, ax = plt.subplots()
    ax.plot(xs, res.statistic, label=ylabel, color="black")
    ax.plot(xs, res.upper, linestyle="--", color="red")
    ax.plot(
        xs,
        res.lower,
        linestyle="--",
        color="red",
        label=f"{int(res.alpha * 100)}% bounds",
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig


def plot_cusum(res: CUSUMResults):  # type: ignore[no-untyped-def]
    """Plot the CUSUM path with its rejection boundaries.

    Parameters
    ----------
    res : CUSUMResults
        Output of :func:`cusum`.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``res`` comes from :func:`cusumsq` instead.
    ImportError
        If matplotlib, an optional dependency, is not installed.
    """
    if res.test != "cusum":
        raise ValueError("plot_cusum expects the output of cusum().")
    return _plot(res, "CUSUM test for parameter stability", "CUSUM")


def plot_cusumsq(res: CUSUMResults):  # type: ignore[no-untyped-def]
    """Plot the CUSUM-of-squares path with its rejection boundaries.

    Parameters
    ----------
    res : CUSUMResults
        Output of :func:`cusumsq`.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``res`` comes from :func:`cusum` instead.
    ImportError
        If matplotlib, an optional dependency, is not installed.
    """
    if res.test != "cusumsq":
        raise ValueError("plot_cusumsq expects the output of cusumsq().")
    return _plot(
        res, "CUSUM-of-squares test for parameter stability", "CUSUM of squares"
    )
