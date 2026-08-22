r"""Quantile regression, and the tolerance it has to be run at.

Quantile regression minimises the *check loss*

.. math::

    \sum_t \rho_\tau(y_t - x_t'\beta), \qquad
    \rho_\tau(u) = u\,(\tau - \mathbf{1}\{u < 0\}),

which is a linear program: the optimum is a vertex, and it is exact.
That makes the estimator easy to check — unlike least squares there is no
closed form to compare against, but there *is* a number that cannot be
beaten, and any candidate can be scored on it.

**Which matters here, because the obvious tool gets it wrong by default.**
``statsmodels``' iteratively reweighted least squares stops on a
convergence tolerance of ``1e-6``, and on ordinary data that is not
enough: measured against the exact linear-programming optimum, its
default settings miss the loss by up to 3.4e-03 and the coefficients by
up to 2.6e-02. Nothing warns — the fit returns, the numbers look
reasonable, and they are 5% wrong.

Tightening the tolerance fixes it: at ``p_tol = 1e-10`` the same
estimates land within 3e-6 of the optimum, for between 1.3 and 2.6 times
the running time. That is the configuration this module uses, and
:func:`check_loss` is what the tests use to hold it to account.

The linear program is available through :func:`quantile_regression_lp`,
but it is **not** the estimator: it is roughly six times slower, and its
role is to be the oracle a fast method is checked against.

References
----------
.. [1] Koenker, R. & Bassett, G. (1978). Regression quantiles.
       *Econometrica*, 46(1), 33-50.
.. [2] Koenker, R. (2005). *Quantile Regression*. Cambridge University
       Press.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

__all__ = [
    "P_TOL",
    "MAX_ITER",
    "check_loss",
    "quantile_regression",
    "quantile_regression_lp",
]

#: Convergence tolerance of the IRLS solver. Not a default taken on
#: trust: see the module docstring for what the library's default costs.
P_TOL = 1e-10

#: Iteration cap. Raised well above the solver's own default so that the
#: tolerance, not the cap, is what stops the iteration.
MAX_ITER = 5000

FloatArray = npt.NDArray[np.float64]


def check_loss(y: FloatArray, x: FloatArray, beta: FloatArray, tau: float) -> float:
    r"""The quantile loss :math:`\sum_t \rho_\tau(y_t - x_t'\beta)`.

    The quantity quantile regression minimises, and therefore the only
    honest way to compare two candidate solutions: the one with the
    lower loss is the better estimate, whatever route produced it.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1.0, 2.0, 3.0])
    >>> x = np.ones((3, 1))
    >>> float(check_loss(y, x, np.array([2.0]), 0.5))
    1.0
    """
    resid = np.asarray(y, dtype=np.float64) - np.asarray(x, dtype=np.float64) @ beta
    return float(np.sum(np.where(resid >= 0, tau * resid, (tau - 1.0) * resid)))


def quantile_regression(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    tau: float,
    p_tol: float = P_TOL,
    max_iter: int = MAX_ITER,
) -> tuple[FloatArray, FloatArray]:
    r"""Estimate one quantile regression, at a tolerance that converges.

    Parameters
    ----------
    y : array_like, shape (T,)
    x : array_like, shape (T, k)
        Design matrix, intercept included if the model has one.
    tau : float
        Quantile, strictly inside ``(0, 1)``.
    p_tol : float
        Convergence tolerance. The default is **not** the solver's own;
        see the module docstring.
    max_iter : int
        Iteration cap.

    Returns
    -------
    params : numpy.ndarray, shape (k,)
    cov : numpy.ndarray, shape (k, k)
        Covariance from the kernel (sparsity) estimator, for inference at
        this quantile alone. Joint inference *across* quantiles needs the
        cross-quantile covariance, which this does not give — see
        :mod:`pyardl.qardl.model`.

    Raises
    ------
    ValueError
        If ``tau`` is not strictly between 0 and 1, or the design cannot
        support an estimate.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.column_stack([np.ones(200), rng.normal(size=200)])
    >>> y = x @ np.array([1.0, 2.0]) + rng.normal(size=200)
    >>> params, _ = quantile_regression(y, x, 0.5)
    >>> bool(abs(params[1] - 2.0) < 0.2)
    True
    """
    from statsmodels.regression.quantile_regression import QuantReg

    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must lie strictly in (0, 1), got {tau}.")
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be two-dimensional, got shape {x_arr.shape}.")
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(
            f"y has {y_arr.shape[0]} observations and x has {x_arr.shape[0]}."
        )
    if x_arr.shape[0] <= x_arr.shape[1]:
        raise ValueError(
            f"{x_arr.shape[0]} observations for {x_arr.shape[1]} parameters: "
            "the quantile regression is not identified."
        )

    with warnings.catch_warnings():
        # The solver warns when it stops on the iteration cap. That is
        # worth knowing, and it is re-raised below as our own warning
        # rather than left as a message about someone else's internals.
        warnings.simplefilter("always")
        caught: list[warnings.WarningMessage] = []
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            fitted = QuantReg(y_arr, x_arr).fit(q=tau, max_iter=max_iter, p_tol=p_tol)
            caught.extend(record)

    if any("Maximum number of iterations" in str(w.message) for w in caught):
        from pyardl.exceptions import PyardlMethodologyWarning

        warnings.warn(
            f"The quantile regression at tau={tau:.3f} stopped on its "
            f"iteration cap ({max_iter}) rather than on its tolerance. The "
            "estimate may not sit at the optimum; compare it against "
            "quantile_regression_lp before relying on it.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    params = np.asarray(fitted.params, dtype=np.float64)
    cov = np.asarray(fitted.cov_params(), dtype=np.float64)
    return params, cov


def quantile_regression_lp(
    y: npt.ArrayLike, x: npt.ArrayLike, tau: float
) -> FloatArray:
    r"""Exact quantile regression, as a linear program.

    Writing the residual as :math:`u - v` with :math:`u, v \ge 0` turns
    the check loss into a linear objective, and the problem into a linear
    program that HiGHS solves to optimality. There is no tolerance to
    choose and no iteration to cut short.

    This is the **oracle**, not the estimator: it is about six times
    slower than the iterative route, so it earns its place in tests and
    in diagnosis, not on the path a user waits on.

    Returns
    -------
    numpy.ndarray, shape (k,)

    Raises
    ------
    RuntimeError
        If the solver does not reach optimality. An unsolved linear
        program has no meaningful answer, and returning its last iterate
        would look exactly like a solution.

    Examples
    --------
    The median of five points, with only an intercept in the design, is
    the middle observation — and unlike an even sample it is unique, so
    the vertex the solver lands on is not a matter of tie-breaking.

    >>> import numpy as np
    >>> y = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    >>> x = np.ones((5, 1))
    >>> float(np.round(quantile_regression_lp(y, x, 0.5)[0], 6))
    3.0
    """
    from scipy import sparse
    from scipy.optimize import linprog

    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must lie strictly in (0, 1), got {tau}.")
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    n, k = x_arr.shape

    cost = np.concatenate([np.zeros(k), tau * np.ones(n), (1.0 - tau) * np.ones(n)])
    design = sparse.hstack(
        [
            sparse.csc_matrix(x_arr),
            sparse.eye(n, format="csc"),
            -sparse.eye(n, format="csc"),
        ],
        format="csc",
    )
    bounds = [(None, None)] * k + [(0.0, None)] * (2 * n)
    result = linprog(cost, A_eq=design, b_eq=y_arr, bounds=bounds, method="highs")
    if not result.success:  # pragma: no cover - HiGHS solves this LP class
        raise RuntimeError(
            f"The linear program did not solve at tau={tau}: {result.message}"
        )
    return np.asarray(result.x[:k], dtype=np.float64)
