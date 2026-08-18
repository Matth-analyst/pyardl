r"""The data-generating process the bootstrap regenerates.

To build the null distribution of the bounds test, the bootstrap has to
generate data in which the null is **true by construction** — series
that are genuinely not cointegrated — while keeping everything else
about the data: the short-run dynamics, the deterministic terms, the
variance and the cross-equation correlation of the innovations.

That requires two models estimated on the observed data.

The conditional model, under the null
-------------------------------------
The unrestricted error-correction model is

.. math::

    \Delta y_t = \text{det}_t + \lambda y_{t-1} + \sum_j \gamma_j x_{j,t-1}
        + \sum_i \psi_i \Delta y_{t-i}
        + \sum_j \sum_i \omega_{j,i} \Delta x_{j,t-i} + \varepsilon_t

The null of the overall F test is :math:`\lambda = \gamma_1 = \dots =
\gamma_k = 0`. Imposing it means **deleting the level terms** — the
model becomes a regression in differences only. That is the model whose
coefficients and residuals drive the bootstrap.

Note what this is *not*: it is not the unrestricted model with its level
coefficients set to zero after the fact. The short-run coefficients are
re-estimated under the restriction, because that is the model the null
actually describes.

The marginal model for the regressors
-------------------------------------
The regressors cannot be held fixed across bootstrap replications: they
are I(1), and the statistic's distribution depends on their stochastic
behaviour. They are therefore regenerated from their own model, a VAR in
first differences,

.. math::

    \Delta x_t = \mu + \sum_{i=1}^{r} A_i \Delta x_{t-i} + \eta_t

which reproduces integrated regressors with the observed short-run
dependence, and no cointegration among them by construction.

Why the two residual vectors travel together
--------------------------------------------
:math:`\varepsilon_t` and :math:`\eta_t` are contemporaneously
correlated in the data — that is what weak exogeneity is *about*, and it
is rarely exactly zero. The bootstrap resamples the stacked vector
:math:`(\varepsilon_t, \eta_t')` by date, never equation by equation.

References
----------
.. [1] McNown, R., Sam, C. Y. & Goh, S. K. (2018). Bootstrapping the
       autoregressive distributed lag test for cointegration.
       *Applied Economics*, 50(13), 1509-1521.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]

__all__ = ["NullDGP", "estimate_null_dgp", "simulate_path", "simulate_paths"]


@dataclass(frozen=True)
class NullDGP:
    """Everything needed to regenerate data under the null.

    Attributes
    ----------
    y_const : float
        Intercept of the conditional equation (zero when the case has
        no unrestricted intercept).
    y_trend : float
        Trend coefficient, zero unless the case carries one.
    psi : numpy.ndarray, shape (p-1,)
        Coefficients on the lagged differences of ``y``.
    omega : tuple of numpy.ndarray
        Per regressor, the coefficients on its current and lagged
        differences.
    x_const : numpy.ndarray, shape (k,)
        Intercepts of the marginal VAR — the drift of the regressors.
    x_ar : numpy.ndarray, shape (r, k, k)
        VAR coefficient matrices of the marginal model.
    residuals : numpy.ndarray, shape (n, 1 + k)
        Centred residuals, column 0 the conditional equation, the rest
        the marginal block. Rows are dates and are resampled as a unit.
    p, q : ...
        The lag structure the conditional equation was built with.
    start : int
        Number of initial observations the estimation had to drop.
    """

    y_const: float
    y_trend: float
    psi: FloatArray
    omega: tuple[FloatArray, ...]
    x_const: FloatArray
    x_ar: FloatArray
    residuals: FloatArray
    p: int
    q: tuple[int, ...]
    start: int

    @property
    def n_regressors(self) -> int:
        return self.x_const.size

    @property
    def var_order(self) -> int:
        return int(self.x_ar.shape[0])


def _fit_marginal_var(
    dx: FloatArray, order: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Least-squares VAR on the differenced regressors."""
    n, k = dx.shape
    if order < 0:
        raise ValueError(f"var_order={order} must be non-negative.")
    if n - order < order * k + 1 + 1:
        raise ValueError(
            f"Sample too short for a VAR({order}) on {k} regressors: "
            f"{n - order} usable rows."
        )

    target = dx[order:]
    blocks: list[FloatArray] = [np.ones((n - order, 1), dtype=np.float64)]
    for i in range(1, order + 1):
        blocks.append(np.asarray(dx[order - i : n - i], dtype=np.float64))
    design = np.column_stack(blocks)

    coefs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coefs

    const = np.asarray(coefs[0], dtype=np.float64)
    ar = np.zeros((order, k, k), dtype=np.float64)
    for i in range(order):
        # Row block i holds A_{i+1}', so it is transposed back.
        ar[i] = coefs[1 + i * k : 1 + (i + 1) * k].T
    return const, ar, np.asarray(resid, dtype=np.float64)


def estimate_null_dgp(
    y: FloatArray,
    x: FloatArray,
    p: int,
    q: tuple[int, ...],
    case: int,
    var_order: int = 1,
) -> NullDGP:
    r"""Estimate the null model and the marginal model of the regressors.

    Parameters
    ----------
    y : numpy.ndarray, shape (T,)
    x : numpy.ndarray, shape (T, k)
    p, q : ...
        Lag orders of the unrestricted error-correction model.
    case : int
        Deterministic case, 1 to 5. Only the *unrestricted* deterministic
        terms are carried into the null model: under cases 2 and 4 the
        restricted term belongs to the tested vector and is therefore
        removed along with the level terms.
    var_order : int, default 1
        Lag order of the marginal VAR on the differenced regressors.

    Returns
    -------
    NullDGP

    Notes
    -----
    The residuals of both blocks are centred before being stored. An
    uncentred residual would inject a drift into every regenerated path,
    biasing the bootstrap distribution.
    """
    n, k = x.shape
    start = max([p, *q]) if q else p
    dy = np.diff(y)
    dx = np.diff(x, axis=0)

    cols: list[FloatArray] = []
    n_rows = n - start
    has_const = case in (3, 4, 5)
    has_trend = case == 5
    if has_const:
        cols.append(np.ones(n_rows))
    if has_trend:
        cols.append(np.arange(start + 1, n + 1, dtype=np.float64))

    for i in range(1, p):
        cols.append(dy[start - i - 1 : n - i - 1])
    for j in range(k):
        for i in range(q[j]):
            cols.append(dx[start - i - 1 : n - i - 1, j])

    target = dy[start - 1 :]
    if cols:
        design = np.column_stack(cols)
        coefs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        eps = target - design @ coefs
    else:
        # No regressor at all under the null: the difference of y is
        # then pure noise around zero.
        coefs = np.zeros(0)
        eps = target.copy()

    pos = 0
    y_const = float(coefs[pos]) if has_const else 0.0
    pos += int(has_const)
    y_trend = float(coefs[pos]) if has_trend else 0.0
    pos += int(has_trend)
    psi = np.asarray(coefs[pos : pos + (p - 1)], dtype=np.float64)
    pos += p - 1
    omega: list[FloatArray] = []
    for j in range(k):
        omega.append(np.asarray(coefs[pos : pos + q[j]], dtype=np.float64))
        pos += q[j]

    x_const, x_ar, eta = _fit_marginal_var(dx, var_order)

    # Align the two residual blocks on their common dates, then centre.
    n_common = min(eps.size, eta.shape[0])
    stacked = np.column_stack([eps[-n_common:], eta[-n_common:]])
    stacked = stacked - stacked.mean(axis=0)

    return NullDGP(
        y_const=y_const,
        y_trend=y_trend,
        psi=psi,
        omega=tuple(omega),
        x_const=x_const,
        x_ar=x_ar,
        residuals=np.asarray(stacked, dtype=np.float64),
        p=p,
        q=q,
        start=start,
    )


def simulate_paths(
    dgp: NullDGP,
    innovations: FloatArray,
    y0: float,
    x0: FloatArray,
    burn_in: int = 50,
) -> tuple[FloatArray, FloatArray]:
    r"""Regenerate many bootstrap samples at once.

    Parameters
    ----------
    dgp : NullDGP
        Output of :func:`estimate_null_dgp`.
    innovations : numpy.ndarray, shape (B, burn_in + T, 1 + k)
        Resampled residual rows, one plane per replication. Column 0 of
        the last axis feeds the conditional equation, the rest the
        marginal block.
    y0 : float
    x0 : numpy.ndarray, shape (k,)
        Initial values, taken from the observed data.
    burn_in : int, default 50
        Initial periods discarded from every path.

    Returns
    -------
    y_star : numpy.ndarray, shape (B, T)
    x_star : numpy.ndarray, shape (B, T, k)

    Notes
    -----
    The recursion is sequential in ``t`` — period ``t`` needs period
    ``t-1`` — but **independent across replications**. So the loop runs
    over ``T`` periods with every replication advanced together, instead
    of over ``B x T`` scalar steps. The arithmetic is identical; only
    the interpreter overhead disappears.

    The trend of the null model, when the deterministic case carries one,
    is indexed so that the first *kept* observation sits at ``t = 1``,
    matching the estimation. Burn-in periods therefore take
    non-positive trend values — the same straight line extended
    backwards, which is the only choice that leaves no discontinuity at
    the join.
    """
    inn = np.asarray(innovations, dtype=np.float64)
    if inn.ndim != 3:
        raise ValueError(
            f"innovations must be 3-D (B, periods, 1+k), got shape {inn.shape}."
        )
    n_rep, n_total, n_eq = inn.shape
    k = dgp.n_regressors
    if n_eq != 1 + k:
        raise ValueError(
            f"innovations has {n_eq} columns for {k} regressors; expected {1 + k}."
        )
    if n_total - burn_in <= 0:
        raise ValueError(f"burn_in={burn_in} leaves no observation out of {n_total}.")

    r = dgp.var_order
    lag_max = max(dgp.p, max(dgp.q, default=0), r, 1)

    dx = np.zeros((n_rep, n_total, k), dtype=np.float64)
    dy = np.zeros((n_rep, n_total), dtype=np.float64)

    for t in range(lag_max, n_total):
        acc = np.broadcast_to(dgp.x_const, (n_rep, k)).copy()
        for i in range(r):
            acc += dx[:, t - i - 1] @ dgp.x_ar[i].T
        dx[:, t] = acc + inn[:, t, 1:]

        val = np.full(n_rep, dgp.y_const, dtype=np.float64)
        if dgp.y_trend:
            val += dgp.y_trend * float(t - burn_in + 1)
        for i in range(1, dgp.p):
            val += dgp.psi[i - 1] * dy[:, t - i]
        for j in range(k):
            for i in range(dgp.q[j]):
                val += dgp.omega[j][i] * dx[:, t - i, j]
        dy[:, t] = val + inn[:, t, 0]

    x_star = np.asarray(
        np.cumsum(dx[:, burn_in:], axis=1) + np.asarray(x0, dtype=np.float64),
        dtype=np.float64,
    )
    y_star = np.asarray(
        np.cumsum(dy[:, burn_in:], axis=1) + float(y0), dtype=np.float64
    )
    return y_star, x_star


def simulate_path(
    dgp: NullDGP,
    innovations: FloatArray,
    y0: float,
    x0: FloatArray,
    burn_in: int = 50,
) -> tuple[FloatArray, FloatArray]:
    """Regenerate a single bootstrap sample.

    Thin wrapper over :func:`simulate_paths` with one replication. The
    single-path and many-path routes therefore cannot drift apart: there
    is only one recursion in the code.

    Parameters
    ----------
    dgp : NullDGP
    innovations : numpy.ndarray, shape (burn_in + T, 1 + k)
    y0, x0 : ...
    burn_in : int, default 50

    Returns
    -------
    y_star : numpy.ndarray, shape (T,)
    x_star : numpy.ndarray, shape (T, k)
    """
    inn = np.asarray(innovations, dtype=np.float64)
    if inn.ndim != 2:
        raise ValueError("innovations must be 2-D (periods, 1+k).")
    y_star, x_star = simulate_paths(dgp, inn[None, ...], y0, x0, burn_in)
    return y_star[0], x_star[0]
