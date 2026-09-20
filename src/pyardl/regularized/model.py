r"""Regularized ARDL: order selection by penalisation.

Tibshirani (1996) LASSO and Zou & Hastie (2005) Elastic Net, adapted to
the UECM structure.

``ARDL.select_order`` (:mod:`pyardl.core.ardl`, spec 05) chooses ``p``
and ``q`` by exhaustive or sequential search on a grid, re-estimating a
complete model at every candidate — expensive when the number of
regressors ``k`` is large (the grid grows as ``(max_q+1)^k`` for
``search="grid"``). Regularisation is the alternative: fit **one**
model at the maximal order with a penalty that shrinks irrelevant lag
coefficients to zero, instead of choosing the order by discrete search.

Like specs 30/34/36, "regularized ARDL" has no single founding
paper — this module adapts LASSO/Elastic Net penalisation to the
ARDL/UECM structure the rest of the library uses.

The one methodological point that matters more than the algorithm:
weights are **block-differentiated**. The deterministic terms and the
error-correction coefficient :math:`\lambda y_{t-1}` are **never**
penalised (a zero weight) — penalising the adjustment speed itself
would bias the long-run reading, exactly the quantity this whole
library exists to get right. Only the short-run lag terms, and
optionally the long-run levels themselves, are candidates for shrinkage.

Estimation is a plain cyclic coordinate descent (Friedman, Hastie &
Tibshirani 2010) implemented directly with NumPy — no new runtime
dependency added for it, consistent with the project's dependency
policy (``scikit-learn``/``glmnet`` are used only in the external
validation script, never at runtime).

References
----------
.. [1] Tibshirani, R. (1996). Regression shrinkage and selection via
       the lasso. *Journal of the Royal Statistical Society, Series B*,
       58(1), 267-288.
.. [2] Zou, H. & Hastie, T. (2005). Regularization and variable
       selection via the elastic net. *Journal of the Royal Statistical
       Society, Series B*, 67(2), 301-320.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyardl.core.ardl import ARDL
from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    from pyardl.core.ardl import ARDLResults

    FloatArray = NDArray[np.float64]

Method = Literal["lasso", "elastic_net"]
Det = Literal["none", "const", "trend"]

__all__ = ["select_order_regularized", "RegularizedOrderResults"]

_ZERO_TOL = 1e-8


def _build_design(
    y: FloatArray, x: FloatArray, max_p: int, max_q: int, det: Det
) -> tuple[FloatArray, FloatArray, list[str], FloatArray]:
    """UECM design at the maximal order; returns design, target, names, weight mask.

    weight mask: 0.0 for never-penalised columns (deterministic terms,
    the error-correction term y.L1), 1.0 for every other column
    (short-run lags and the levels x.L1 — spec 41 §2.1).
    """
    k = x.shape[1]
    start = max(max_p, max_q, 1)
    n = y.shape[0]
    dy = np.diff(y)
    dx = np.diff(x, axis=0)
    target = dy[start - 1 :]

    cols: list[FloatArray] = []
    names: list[str] = []
    weights: list[float] = []

    if det in ("const", "trend"):
        cols.append(np.ones_like(target))
        names.append("const")
        weights.append(0.0)
    if det == "trend":
        cols.append(np.arange(1, target.shape[0] + 1, dtype=np.float64))
        names.append("trend")
        weights.append(0.0)

    cols.append(y[start - 1 : n - 1])
    names.append("y.L1")
    weights.append(0.0)

    for j in range(k):
        cols.append(x[start - 1 : n - 1, j])
        names.append(f"x{j}.L1")
        weights.append(1.0)

    for i in range(1, max_p):
        cols.append(dy[start - i - 1 : n - i - 1])
        names.append(f"D.y.L{i}")
        weights.append(1.0)
    for j in range(k):
        for i in range(max_q):
            cols.append(dx[start - i - 1 : n - i - 1, j])
            names.append(f"D.x{j}.L{i + 1}")
            weights.append(1.0)

    design = np.column_stack(cols)
    return design, target, names, np.array(weights, dtype=np.float64)


def _coordinate_descent(
    design: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 500,
    tol: float = 1e-6,
    theta_init: FloatArray | None = None,
) -> FloatArray:
    """Cyclic coordinate descent for weighted Elastic Net.

    Minimises ``(1/2n)||y - X theta||^2 + alpha * sum_j w_j *
    [l1_ratio*|theta_j| + (1-l1_ratio)/2 * theta_j^2]``. ``w_j = 0``
    columns (deterministic terms, the error-correction term) are never
    shrunk — an ordinary least-squares update, every iteration.

    ``theta_init`` warm-starts the descent (standard practice when
    tracing a path over a grid of close alphas, e.g. glmnet) — a
    performance choice, not an approximation: the fixed point the
    iteration converges to does not depend on the starting value.
    """
    n = design.shape[0]
    n_params = design.shape[1]
    theta = (
        theta_init.copy()
        if theta_init is not None
        else np.zeros(n_params, dtype=np.float64)
    )
    col_sq = np.sum(design**2, axis=0)
    resid = target - design @ theta

    for _ in range(max_iter):
        theta_old = theta.copy()
        for j in range(n_params):
            if col_sq[j] < 1e-12:
                continue
            resid += design[:, j] * theta[j]
            rho = design[:, j] @ resid
            if weights[j] == 0.0:
                theta[j] = rho / col_sq[j]
            else:
                penalty_l1 = alpha * weights[j] * l1_ratio * n
                penalty_l2 = alpha * weights[j] * (1.0 - l1_ratio) * n
                soft = np.sign(rho) * max(abs(rho) - penalty_l1, 0.0)
                theta[j] = soft / (col_sq[j] + penalty_l2)
            resid -= design[:, j] * theta[j]
        if np.max(np.abs(theta - theta_old)) < tol:
            break
    return theta


def _deduce_order(
    theta: FloatArray, names: list[str], max_p: int, max_q: int, k: int
) -> tuple[int, dict[str, int]]:
    """Retro-deduce (p, {name: q_j}) from which lag coefficients survived."""
    nonzero = {
        name for name, val in zip(names, theta, strict=True) if abs(val) > _ZERO_TOL
    }

    p = 1
    for i in range(1, max_p):
        if f"D.y.L{i}" in nonzero:
            p = i + 1

    q_map: dict[str, int] = {}
    for j in range(k):
        name = f"x{j}"
        q = 0
        for i in range(1, max_q + 1):
            if f"D.x{j}.L{i}" in nonzero:
                q = i
        q_map[name] = q
    return p, q_map


@dataclass(frozen=True)
class RegularizedOrderResults:
    """Outcome of :func:`select_order_regularized`.

    Attributes
    ----------
    alpha_selected : float
    coefficients_path : pandas.DataFrame
        Indexed by alpha (descending grid), one column per design term —
        :math:`\\theta(\\alpha)`, for plotting the regularisation path.
    selected_order : tuple
        ``(p, {name: q_j})``, retro-deduced from which coefficients
        survived at ``alpha_selected``.
    best_model : ARDLResults
        :class:`pyardl.core.ardl.ARDL` re-estimated by plain OLS at
        ``selected_order`` — **never** with the penalty: the
        regularisation path chooses the order, it never produces the
        reported coefficients (spec 41 §3, a documented discipline, not
        an implementation detail).
    method : str
    l1_ratio : float
    alpha_grid : ndarray
    cv_errors : pandas.Series or None
        Mean rolling-origin CV error per alpha, when ``alpha='cv'``.
    """

    alpha_selected: float
    coefficients_path: pd.DataFrame
    selected_order: tuple[int, dict[str, int]]
    best_model: ARDLResults = field(repr=False)
    method: Method
    l1_ratio: float
    alpha_grid: FloatArray
    cv_errors: pd.Series | None = None

    def summary(self) -> str:
        """Readable report."""
        p, q_map = self.selected_order
        lines = [
            f"Regularized ARDL order selection ({self.method}, "
            f"l1_ratio={self.l1_ratio}) - alpha_selected={self.alpha_selected:.6g}",
            f"  selected_order: p={p}, q={q_map}",
            "  best_model re-estimated by plain OLS at this order (no penalty)",
        ]
        return "\n".join(lines)

    def plot_regularization_path(self, ax: object = None):  # type: ignore[no-untyped-def]
        """Plot theta(alpha) over the grid. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "plot_regularization_path requires matplotlib, an optional "
                "dependency. Install it with: pip install matplotlib"
            ) from exc
        if ax is None:  # pragma: no cover - trivial plumbing
            _, ax = plt.subplots(figsize=(7, 4))
        axes: object = ax
        for name in self.coefficients_path.columns:
            axes.plot(  # type: ignore[attr-defined]
                self.coefficients_path.index, self.coefficients_path[name], label=name
            )
        axes.set_xscale("log")  # type: ignore[attr-defined]
        axes.set_xlabel("alpha")  # type: ignore[attr-defined]
        axes.set_ylabel("theta(alpha)")  # type: ignore[attr-defined]
        axes.axvline(self.alpha_selected, color="black", linestyle="--")  # type: ignore[attr-defined]
        return axes


def _rolling_cv_error(
    y: FloatArray,
    x: FloatArray,
    max_p: int,
    max_q: int,
    det: Det,
    alpha: float,
    l1_ratio: float,
    n_splits: int,
    horizon: int,
) -> float:
    """Mean squared one-step forecast error over rolling-origin splits."""
    design, target, names, weights = _build_design(y, x, max_p, max_q, det)
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
        theta = _coordinate_descent(
            design[:t_end], target[:t_end], weights, alpha, l1_ratio
        )
        pred = design[t_end : t_end + horizon] @ theta
        err = target[t_end : t_end + horizon] - pred
        errors.append(float(np.mean(err**2)))
    del names
    return float(np.mean(errors)) if errors else np.inf


def select_order_regularized(
    y: ArrayLike,
    x: ArrayLike,
    max_p: int = 4,
    max_q: int = 4,
    method: Method = "elastic_net",
    l1_ratio: float = 1.0,
    alpha: float | Literal["cv"] = "cv",
    alpha_grid_size: int = 20,
    cv_folds: int = 5,
    cv_horizon: int = 5,
    det: Det = "const",
) -> RegularizedOrderResults:
    r"""Choose an ARDL's lag order by penalised least squares.

    Parameters
    ----------
    y, x : array_like
        As in :class:`pyardl.core.ardl.ARDL`.
    max_p, max_q : int, default 4
        Maximal order the penalised fit starts from.
    method : {'elastic_net', 'lasso'}, default 'elastic_net'
        ``'lasso'`` is the ``l1_ratio=1.0`` special case, offered
        directly since pure LASSO's instability under correlated
        regressors (spec 41 §2.2) is exactly what Elastic Net is for.
    l1_ratio : float, default 1.0
        Elastic Net mixing parameter, in ``[0, 1]``. Ignored (forced to
        1.0) when ``method='lasso'``.
    alpha : float or "cv", default "cv"
        Penalty strength. ``"cv"`` selects it by rolling-origin
        (expanding window) cross-validation — never a random split,
        which would break the time structure (spec 41 §2.3).
    alpha_grid_size : int, default 20
        Points in the log-spaced alpha grid.
    cv_folds : int, default 5
        Rolling-origin splits when ``alpha='cv'``.
    cv_horizon : int, default 5
        Forecast horizon evaluated at each split.
    det : {'none', 'const', 'trend'}, default 'const'

    Returns
    -------
    RegularizedOrderResults

    Notes
    -----
    Classical inference (standard errors, Wald tests) on a model chosen
    by regularisation and then re-estimated by OLS ignores the
    selection's own uncertainty ("post-selection inference", a
    distinct literature not covered here) — do not present
    ``best_model``'s standard errors as valid in the same sense as an
    order fixed a priori.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.regularized import select_order_regularized
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = y[t - 1] - 0.5 * (y[t - 1] - 1.5 * x.iloc[t - 1])
    ...     y[t] += rng.standard_normal() * 0.3
    >>> res = select_order_regularized(y, pd.DataFrame({"x": x}), max_p=3, max_q=3,
    ...                                 method="lasso", alpha=0.01)
    >>> res.selected_order[0] >= 1
    True
    """
    if method not in ("lasso", "elastic_net"):
        raise ValueError(f"method={method!r} must be 'lasso' or 'elastic_net'.")
    if method == "lasso":
        l1_ratio = 1.0
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError(f"l1_ratio={l1_ratio} must be in [0, 1].")
    if det not in ("none", "const", "trend"):
        raise ValueError(f"det={det!r} must be 'none', 'const' or 'trend'.")

    y_arr, x_arr, _, _, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("select_order_regularized needs at least one regressor.")
    k = x_arr.shape[1]

    design, target, names, weights = _build_design(y_arr, x_arr, max_p, max_q, det)

    scale = float(np.max(np.abs(design.T @ target))) / max(design.shape[0], 1)
    alpha_max = max(scale, 1e-6)
    alpha_grid = np.asarray(
        np.geomspace(alpha_max, alpha_max * 1e-4, alpha_grid_size), dtype=np.float64
    )

    cv_errors_series: pd.Series | None = None
    if alpha == "cv":
        cv_errors = [
            _rolling_cv_error(
                y_arr,
                x_arr,
                max_p,
                max_q,
                det,
                float(a),
                l1_ratio,
                cv_folds,
                cv_horizon,
            )
            for a in alpha_grid
        ]
        cv_errors_series = pd.Series(cv_errors, index=alpha_grid, name="cv_error")
        alpha_selected = float(alpha_grid[int(np.argmin(cv_errors))])
    else:
        alpha_selected = float(alpha)

    path = np.empty((alpha_grid.shape[0], design.shape[1]), dtype=np.float64)
    warm: FloatArray | None = None
    theta_at_selected: FloatArray | None = None
    for i, a in enumerate(alpha_grid):
        warm = _coordinate_descent(
            design, target, weights, float(a), l1_ratio, theta_init=warm
        )
        path[i] = warm
        if np.isclose(float(a), alpha_selected):
            theta_at_selected = warm
    coefficients_path = pd.DataFrame(
        path, index=pd.Index(alpha_grid, name="alpha"), columns=names
    )

    theta_selected = (
        theta_at_selected
        if theta_at_selected is not None
        else _coordinate_descent(design, target, weights, alpha_selected, l1_ratio)
    )
    p_sel, q_map = _deduce_order(theta_selected, names, max_p, max_q, k)

    x_df = pd.DataFrame(x_arr, columns=list(x_names))
    q_by_name = {name: q_map[f"x{j}"] for j, name in enumerate(x_names)}
    best_model = ARDL(y_arr, x_df, order=(p_sel, q_by_name), det=det).fit()

    return RegularizedOrderResults(
        alpha_selected=alpha_selected,
        coefficients_path=coefficients_path,
        selected_order=(p_sel, q_by_name),
        best_model=best_model,
        method=method,
        l1_ratio=l1_ratio,
        alpha_grid=alpha_grid,
        cv_errors=cv_errors_series,
    )
