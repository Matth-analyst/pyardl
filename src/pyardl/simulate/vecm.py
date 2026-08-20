r"""A single VECM simulator for every Monte Carlo study in the library.

Before this module, each specification carried its own data-generating
code. That is how two studies end up disagreeing for reasons nobody can
locate: not because the estimators differ, but because the *data* did.
Everything now goes through one generator, so a disagreement between two
validation studies is a disagreement about estimators.

The system is the VECM

.. math::

    \Delta z_t = \Pi z_{t-1} + \sum_{i=1}^{s} \Gamma_i \Delta z_{t-i}
                 + d_t + \varepsilon_t,
    \qquad \Pi = \alpha \beta',

where :math:`z_t = (y_t, x_{1t}, \dots, x_{kt})'`. Writing :math:`\Pi`
as :math:`\alpha\beta'` is what makes the rank *chosen* rather than
hoped for: ``beta`` holds the long-run relations, one per column, and
``alpha`` the speeds at which each equation adjusts to them. The rank of
:math:`\Pi` is the number of columns, by construction.

The two degeneracies of the bounds literature are first-class options,
because a framework that claims to detect them has to be shown data that
actually contains them:

**Type 1** — :math:`\lambda \ne 0`, :math:`\gamma = 0`. The relation
involves ``y`` alone: it is stationary around a constant while the
regressors wander off on their own.

**Type 2** — :math:`\gamma \ne 0`, :math:`\lambda = 0`. The relation
holds among the regressors, and the ``y`` equation responds to it
without ``y`` itself appearing in it. The level terms of the ``x`` are
jointly significant in the ``y`` equation, yet nothing pulls ``y`` back.

References
----------
.. [1] Bertelli, S., Vacca, G. & Zoia, M. (2022). Bootstrap cointegration
       tests in ARDL models. *Economic Modelling*, 116, 105987.
.. [2] Johansen, S. (1991). Estimation and hypothesis testing of
       cointegration vectors in Gaussian vector autoregressive models.
       *Econometrica*, 59(6), 1551-1580.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]

__all__ = ["VECMSimulation", "degenerate_system", "vecm_ardl"]

Degeneracy = Literal[1, 2]


@dataclass(frozen=True)
class VECMSimulation:
    """Simulated system, together with the parameters that produced it.

    Attributes
    ----------
    data : pandas.DataFrame
        The series, ``y`` first then the regressors.
    alpha, beta : numpy.ndarray
        Adjustment speeds and cointegrating vectors, ``(n, r)`` each.
    pi : numpy.ndarray
        ``alpha @ beta.T``, kept because it is what the estimators see.
    gammas : tuple of numpy.ndarray
        Short-run matrices, one per lag.
    rank : int
        Number of cointegrating relations. Read off ``pi`` rather than
        off the number of columns of ``beta``: passing a zero ``alpha``
        or a zero ``beta`` gives a system whose ``Pi`` has rank 0, and
        reporting the column count there would claim a relation the data
        do not contain.
    case : int
        Deterministic case the data were generated under.
    seed : int
        Recorded, always — a simulated sample nobody can regenerate is
        not evidence.
    """

    data: pd.DataFrame
    alpha: FloatArray
    beta: FloatArray
    pi: FloatArray
    gammas: tuple[FloatArray, ...]
    rank: int
    case: int
    seed: int
    sigma: FloatArray = field(repr=False)

    @property
    def y(self) -> pd.Series:
        """The dependent variable, first column by convention."""
        return self.data.iloc[:, 0]

    @property
    def x(self) -> pd.DataFrame:
        """The regressors."""
        return self.data.iloc[:, 1:]

    @property
    def lam(self) -> float:
        r"""The true :math:`\lambda`: coefficient of ``y`` in the ``y``
        equation of :math:`\Pi`."""
        return float(self.pi[0, 0])

    @property
    def gamma_true(self) -> FloatArray:
        r"""The true :math:`\gamma`: coefficients of the regressors'
        levels in the ``y`` equation."""
        return np.asarray(self.pi[0, 1:], dtype=np.float64)


def _validate_case(case: int) -> tuple[bool, bool]:
    """Return (has_const, has_trend) for a deterministic case."""
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be 1..5 (PSS numbering), got {case}.")
    return case in (2, 3, 4, 5), case in (4, 5)


def vecm_ardl(
    n_obs: int,
    alpha: ArrayLike,
    beta: ArrayLike,
    gammas: Sequence[ArrayLike] = (),
    case: int = 3,
    sigma: ArrayLike | None = None,
    const: ArrayLike | None = None,
    trend: ArrayLike | None = None,
    seed: int | None = None,
    burn_in: int = 100,
    names: Sequence[str] | None = None,
) -> VECMSimulation:
    r"""Generate a system with a chosen cointegration rank.

    Parameters
    ----------
    n_obs : int
        Number of observations to keep, after ``burn_in``.
    alpha, beta : array_like
        Adjustment speeds and cointegrating vectors, ``(n, r)`` each,
        where ``n = 1 + k``. The rank is ``r``, by construction.
    gammas : sequence of array_like, optional
        Short-run matrices ``(n, n)``, one per lag.
    case : int, default 3
        Deterministic case in the PSS numbering.
    sigma : array_like, optional
        Covariance of the innovations, ``(n, n)``. Defaults to the
        identity. A non-diagonal ``sigma`` is what makes the conditional
        and unconditional models differ, so it is worth setting
        deliberately.
    const, trend : array_like, optional
        Deterministic vectors, ``(n,)``. Default to zeros when the case
        carries the term, and are **refused** when it does not.
    seed : int, optional
        Drawn from entropy and recorded when omitted.
    burn_in : int, default 100
        Initial periods discarded.
    names : sequence of str, optional
        Column names; defaults to ``y, x1, ..., xk``.

    Returns
    -------
    VECMSimulation

    Raises
    ------
    ValueError
        On any shape mismatch, or on a deterministic term the requested
        case does not carry. Nothing is silently dropped or resized.

    Notes
    -----
    Stability is **not** checked: an explosive system is a legitimate
    thing to simulate — to verify that a test never calls it
    cointegration, for instance. What is guaranteed is that what comes
    back is what the parameters describe.

    Examples
    --------
    >>> import numpy as np
    >>> sim = vecm_ardl(200, alpha=[[-0.4], [0.0]], beta=[[1.0], [-1.0]], seed=0)
    >>> sim.rank
    1
    >>> float(np.round(sim.lam, 4))
    -0.4
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be at least 2, got {n_obs}.")
    if burn_in < 0:
        raise ValueError(f"burn_in must be non-negative, got {burn_in}.")

    a = np.atleast_2d(np.asarray(alpha, dtype=np.float64))
    b = np.atleast_2d(np.asarray(beta, dtype=np.float64))
    if a.shape != b.shape:
        raise ValueError(
            f"alpha has shape {a.shape} and beta {b.shape}; both must be "
            "(n, r) with the same n and r."
        )
    n_var = a.shape[0]
    if n_var < 2:
        raise ValueError(
            f"The system needs at least two variables, got {n_var}. With one "
            "there is no long-run relationship to simulate."
        )

    has_const, has_trend = _validate_case(case)

    gam = tuple(np.asarray(g, dtype=np.float64) for g in gammas)
    for i, g in enumerate(gam):
        if g.shape != (n_var, n_var):
            raise ValueError(
                f"gammas[{i}] has shape {g.shape}, expected ({n_var}, {n_var})."
            )

    cov = np.eye(n_var) if sigma is None else np.asarray(sigma, dtype=np.float64)
    if cov.shape != (n_var, n_var):
        raise ValueError(f"sigma has shape {cov.shape}, expected ({n_var}, {n_var}).")

    def _det(vec: ArrayLike | None, carried: bool, label: str) -> FloatArray:
        if vec is None:
            return np.zeros(n_var)
        if not carried:
            raise ValueError(
                f"case {case} carries no {label}, but one was given. Pick the "
                "case that describes the model you mean rather than passing a "
                "term it does not have."
            )
        arr = np.asarray(vec, dtype=np.float64).ravel()
        if arr.size != n_var:
            raise ValueError(f"{label} has {arr.size} elements, expected {n_var}.")
        return arr

    mu = _det(const, has_const, "constant")
    tau = _det(trend, has_trend, "trend")

    if seed is None:
        entropy = np.random.SeedSequence().entropy
        seed = int(entropy) % (2**63) if isinstance(entropy, int) else 0
    rng = np.random.default_rng(seed)

    pi = np.asarray(a @ b.T, dtype=np.float64)
    # The rank the DATA carry, which is the rank of Pi — not the number
    # of columns supplied. A zero alpha or beta yields rank 0.
    rank = int(np.linalg.matrix_rank(pi))
    n_total = burn_in + n_obs
    lag_max = max(len(gam), 1)

    eps = rng.multivariate_normal(np.zeros(n_var), cov, size=n_total)
    z = np.zeros((n_total, n_var))
    dz = np.zeros((n_total, n_var))
    for t in range(lag_max, n_total):
        acc = pi @ z[t - 1] + eps[t]
        if has_const:
            acc = acc + mu
        if has_trend:
            # Indexed so the first KEPT observation sits at t = 1, the
            # convention the estimators use.
            acc = acc + tau * float(t - burn_in + 1)
        for i, g in enumerate(gam):
            acc = acc + g @ dz[t - i - 1]
        dz[t] = acc
        z[t] = z[t - 1] + acc

    if names is None:
        labels = ["y"] + [f"x{j}" for j in range(1, n_var)]
    else:
        labels = [str(v) for v in names]
        if len(labels) != n_var:
            raise ValueError(f"names has {len(labels)} entries, expected {n_var}.")

    return VECMSimulation(
        data=pd.DataFrame(z[burn_in:], columns=labels),
        alpha=a,
        beta=b,
        pi=pi,
        gammas=gam,
        rank=rank,
        case=case,
        seed=int(seed),
        sigma=cov,
    )


def degenerate_system(
    kind: Degeneracy | None,
    k: int = 1,
    speed: float = -0.4,
) -> tuple[FloatArray, FloatArray]:
    r"""Build ``(alpha, beta)`` for one of the canonical systems.

    The cases the three-test framework has to tell apart, written once so
    every study uses the same ones.

    Parameters
    ----------
    kind : {1, 2, None}
        ``1`` for a type 1 degeneracy (:math:`\lambda \ne 0`,
        :math:`\gamma = 0`), ``2`` for type 2 (:math:`\gamma \ne 0`,
        :math:`\lambda = 0`), ``None`` for genuine cointegration. For no
        cointegration at all there is no relation to build: pass a rank-0
        system to :func:`vecm_ardl` directly.
    k : int, default 1
        Number of regressors.
    speed : float, default -0.4
        Adjustment speed of the ``y`` equation. Must be negative — a
        non-negative speed is not slow adjustment, it is no adjustment.

    Returns
    -------
    alpha, beta : numpy.ndarray, shape (1 + k, 1)

    Examples
    --------
    >>> alpha, beta = degenerate_system(1, k=1)
    >>> (alpha @ beta.T)[0]          # y adjusts, x carries nothing
    array([-0.4,  0. ])
    >>> alpha, beta = degenerate_system(2, k=1)
    >>> (alpha @ beta.T)[0]          # x levels matter, y does not adjust
    array([ 0. , -0.4])
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")
    if speed >= 0:
        raise ValueError(
            f"speed must be negative to pull back towards equilibrium, got {speed}."
        )
    n_var = 1 + k
    alpha = np.zeros((n_var, 1))
    beta = np.zeros((n_var, 1))
    alpha[0, 0] = speed

    if kind == 1:
        # The relation involves y alone: stationary around a constant,
        # while the regressors wander off independently.
        beta[0, 0] = 1.0
    elif kind == 2:
        # The relation holds among the regressors; the y equation
        # responds to it, but y does not appear in it, so nothing pulls
        # y back to anything.
        beta[1, 0] = 1.0
        if k >= 2:
            beta[2, 0] = -1.0
    elif kind is None:
        beta[0, 0] = 1.0
        beta[1:, 0] = -1.0
    else:
        raise ValueError(f"kind must be 1, 2 or None, got {kind!r}.")
    return alpha, beta
