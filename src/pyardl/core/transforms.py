r"""Exact algebra between the ARDL and error-correction representations.

An ARDL(p, q_1, ..., q_k) model

    y_t = alpha + delta*t + sum_i phi_i y_{t-i}
          + sum_j sum_i beta_{j,i} x_{j,t-i} + eps_t

can be rewritten *exactly* — no approximation, identical residuals — as an
error-correction model (ECM):

    Δy_t = alpha + delta*t + lam*y_{t-1} + sum_j gamma_j x_{j,t-1}
           + sum_i psi_i Δy_{t-i} + sum_j sum_i omega_{j,i} Δx_{j,t-i} + eps_t

The two parameterisations are linked by:

    lam = -(1 - sum_i phi_i)
    gamma_j = sum_i beta_{j,i}
    psi_i = -sum_{m=i+1}^{p} phi_m,             i = 1, ..., p-1
    omega_{j,0} = beta_{j,0}
    omega_{j,i} = -sum_{m=i+1}^{q_j} beta_{j,m}, i = 1, ..., q_j-1
    theta_j = -gamma_j / lam                     (long-run coefficient)

``lam`` is the speed of adjustment: the fraction of last period's
disequilibrium corrected each period. ``theta_j`` is the long-run
elasticity of y with respect to x_j.

References
----------
Sargan, J. D. (1964). "Wages and Prices in the United Kingdom: A Study in
Econometric Methodology", in *Econometric Analysis for National Economic
Planning*, Butterworths.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.exceptions import DegenerateCaseWarning

_LAMBDA_TOL = 1e-8

FloatArray = npt.NDArray[np.float64]


def _as_float_array(a: npt.ArrayLike) -> FloatArray:
    return np.asarray(a, dtype=np.float64)


@dataclass(frozen=True)
class ARDLParams:
    """Parameters of an ARDL(p, q_1, ..., q_k) model.

    Parameters
    ----------
    p : int
        Autoregressive order, i.e. number of lags of y (at least 1).
    q : tuple of int
        Lag order of each regressor x_j (may be 0).
    phi : ndarray, shape (p,)
        Coefficients phi_1, ..., phi_p on the lags of y.
    beta : tuple of ndarray
        ``beta[j]`` has shape ``(q_j + 1,)`` and holds
        beta_{j,0}, ..., beta_{j,q_j}.
    const : float
        Intercept (0.0 if the model has none).
    trend : float
        Linear trend coefficient (0.0 if the model has none).
    has_const, has_trend : bool
        Whether the intercept and trend are actually part of the estimated
        model. They determine the layout of the parameter vector used by
        ``cov_params``.
    x_names : tuple of str, optional
        Names of the regressors, used for display.
    cov_params : ndarray, optional
        Covariance matrix of the full parameter vector, ordered as
        :meth:`param_vector` (const?, trend?, phi, beta[0], beta[1], ...).
    """

    p: int
    q: tuple[int, ...]
    phi: FloatArray
    beta: tuple[FloatArray, ...]
    const: float = 0.0
    trend: float = 0.0
    has_const: bool = True
    has_trend: bool = False
    x_names: tuple[str, ...] | None = None
    cov_params: FloatArray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "phi", _as_float_array(self.phi))
        object.__setattr__(self, "beta", tuple(_as_float_array(b) for b in self.beta))
        if self.p < 1:
            raise ValueError("p must be >= 1 (the model needs at least y_{t-1}).")
        if self.phi.shape != (self.p,):
            raise ValueError(f"phi must have shape ({self.p},).")
        if len(self.q) != len(self.beta):
            raise ValueError("q and beta must have the same length (k).")
        for j, (qj, bj) in enumerate(zip(self.q, self.beta, strict=True)):
            if qj < 0:
                raise ValueError(f"q[{j}] must be >= 0.")
            if bj.shape != (qj + 1,):
                raise ValueError(f"beta[{j}] must have shape ({qj + 1},).")
        if self.x_names is not None and len(self.x_names) != len(self.q):
            raise ValueError("x_names must have the same length as q.")

    @property
    def k(self) -> int:
        """Number of x regressors."""
        return len(self.q)

    def param_vector(self) -> FloatArray:
        """Stack the parameters as const?, trend?, phi, beta[0], beta[1], ...

        This ordering is the contract used by ``cov_params`` and by
        :func:`longrun_covariance`.
        """
        parts: list[FloatArray] = []
        if self.has_const:
            parts.append(np.array([self.const]))
        if self.has_trend:
            parts.append(np.array([self.trend]))
        parts.append(self.phi)
        parts.extend(self.beta)
        return np.concatenate(parts)


@dataclass(frozen=True)
class ECMParams:
    """Parameters of the equivalent error-correction representation.

    Parameters
    ----------
    p, q
        See :class:`ARDLParams`.
    lam : float
        Speed of adjustment, ``lam = -(1 - sum phi_i)``.
    gamma : ndarray, shape (k,)
        Level coefficients on x_{j,t-1}.
    psi : ndarray, shape (p-1,)
        Coefficients on Δy_{t-i}, i = 1, ..., p-1.
    omega : tuple of ndarray
        ``omega[j]`` has shape ``(q_j,)`` and holds the coefficients on
        Δx_{j,t-i}, i = 0, ..., q_j - 1.

        When ``q_j = 0`` the array is empty: regressor x_j has no
        short-run dynamics of its own, and ``gamma_j`` then multiplies the
        *contemporaneous* level x_{j,t} rather than x_{j,t-1}. Giving it a
        lagged level instead would add a degree of freedom the original
        ARDL does not have, and the two representations would no longer
        share the same residuals. This matches the behaviour of Stata's
        ``ardl``; ``statsmodels.tsa.ardl.UECM`` rejects ``q_j = 0``
        altogether, whereas pyardl supports it.
    """

    p: int
    q: tuple[int, ...]
    lam: float
    gamma: FloatArray
    psi: FloatArray
    omega: tuple[FloatArray, ...]
    const: float = 0.0
    trend: float = 0.0
    has_const: bool = True
    has_trend: bool = False
    x_names: tuple[str, ...] | None = None
    cov_params: FloatArray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma", _as_float_array(self.gamma))
        object.__setattr__(self, "psi", _as_float_array(self.psi))
        object.__setattr__(self, "omega", tuple(_as_float_array(o) for o in self.omega))
        if self.p < 1:
            raise ValueError("p must be >= 1.")
        if self.psi.shape != (max(self.p - 1, 0),):
            raise ValueError(f"psi must have shape ({max(self.p - 1, 0)},).")
        if len(self.q) != len(self.omega) or len(self.q) != self.gamma.shape[0]:
            raise ValueError("q, gamma and omega must have the same length (k).")
        for j, (qj, oj) in enumerate(zip(self.q, self.omega, strict=True)):
            if oj.shape != (qj,):
                raise ValueError(f"omega[{j}] must have shape ({qj},).")

    @property
    def k(self) -> int:
        """Number of x regressors."""
        return len(self.q)


def ardl_to_ecm(params: ARDLParams) -> ECMParams:
    """Convert ARDL parameters to their error-correction counterpart.

    The reparameterisation is exact: fitting either representation on the
    same data yields identical residuals and sum of squared residuals.

    Examples
    --------
    >>> import numpy as np
    >>> p = ARDLParams(p=1, q=(1,), phi=np.array([0.5]), beta=(np.array([0.3, 0.2]),))
    >>> ecm = ardl_to_ecm(p)
    >>> round(ecm.lam, 6)
    -0.5
    >>> round(float(ecm.gamma[0]), 6)
    0.5
    """
    lam = -(1.0 - float(np.sum(params.phi)))

    # psi_i = -sum_{m=i+1}^{p} phi_m, i = 1..p-1  (s[i] = sum(phi[i:]), 0-indexed)
    s = np.cumsum(params.phi[::-1])[::-1]
    psi = -s[1:] if params.p > 1 else np.array([], dtype=np.float64)

    gamma = np.empty(params.k, dtype=np.float64)
    omega: list[FloatArray] = []
    for j, (qj, bj) in enumerate(zip(params.q, params.beta, strict=True)):
        gamma[j] = float(np.sum(bj))
        if qj == 0:
            # No short-run dynamics of its own: gamma_j will multiply the
            # contemporaneous x_{j,t}, not x_{j,t-1} (see ECMParams.omega).
            omega.append(np.array([], dtype=np.float64))
            continue
        # c_i = sum_{m=i}^{qj} beta_{j,m}, 0-indexed, length qj+1
        c = np.cumsum(bj[::-1])[::-1]
        c1 = c[1]
        if qj >= 2:
            omega_j = np.concatenate(([c[0] - c1], -c[2:]))
        else:
            # qj == 1: a single term, omega_{j,0} = c_0 - c_1
            omega_j = np.array([c[0] - c1])
        omega.append(omega_j)

    return ECMParams(
        p=params.p,
        q=params.q,
        lam=lam,
        gamma=gamma,
        psi=psi,
        omega=tuple(omega),
        const=params.const,
        trend=params.trend,
        has_const=params.has_const,
        has_trend=params.has_trend,
        x_names=params.x_names,
    )


def ecm_to_ardl(params: ECMParams) -> ARDLParams:
    """Convert error-correction parameters back to the ARDL form.

    Exact inverse of :func:`ardl_to_ecm`; the mapping is triangular and is
    solved by cumulative sums.

    Examples
    --------
    >>> import numpy as np
    >>> e = ECMParams(
    ...     p=1, q=(1,), lam=-0.5, gamma=np.array([0.5]),
    ...     psi=np.array([]), omega=(np.array([0.3]),),
    ... )
    >>> a = ecm_to_ardl(e)
    >>> round(float(a.phi[0]), 6)
    0.5
    """
    p = params.p
    # d_1 = 1 + lam; d_i = -psi_{i-1} for i = 2..p; d_{p+1} = 0
    d = np.concatenate(([1.0 + params.lam], -params.psi, [0.0]))
    phi = -np.diff(d)  # length p

    beta: list[FloatArray] = []
    for qj, gamma_j, omega_j in zip(params.q, params.gamma, params.omega, strict=True):
        if qj == 0:
            beta.append(np.array([gamma_j]))
            continue
        # c_0 = gamma_j; c_1 = gamma_j - omega_{j,0}; c_i = -omega_{j,i-1}
        c1 = gamma_j - omega_j[0]
        if qj >= 2:
            c = np.concatenate(([gamma_j, c1], -omega_j[1:], [0.0]))
        else:
            c = np.array([gamma_j, c1, 0.0])
        beta.append(-np.diff(c))

    return ARDLParams(
        p=p,
        q=params.q,
        phi=phi,
        beta=tuple(beta),
        const=params.const,
        trend=params.trend,
        has_const=params.has_const,
        has_trend=params.has_trend,
        x_names=params.x_names,
    )


def speed_of_adjustment(params: ARDLParams) -> float:
    """Return the speed of adjustment ``lam = -(1 - sum phi_i)``.

    A value in ``]-1, 0[`` means the system converges back to its long-run
    equilibrium; the closer to -1, the faster the correction.
    """
    return -(1.0 - float(np.sum(params.phi)))


def longrun_coefs(params: ARDLParams, *, tol: float = _LAMBDA_TOL) -> pd.Series:
    """Long-run coefficients ``theta_j = sum_i beta_{j,i} / (1 - sum_i phi_i)``.

    If there is no error-correction force (``|lam| < tol``) the long-run
    coefficients are not defined: the function returns NaNs and issues a
    :class:`~pyardl.exceptions.DegenerateCaseWarning`.

    Returns
    -------
    pandas.Series
        One long-run coefficient per regressor, indexed by regressor name.
    """
    lam = speed_of_adjustment(params)
    names = params.x_names or tuple(f"x{j}" for j in range(params.k))
    if abs(lam) < tol:
        warnings.warn(
            "lambda is ~0: there is no error-correction force, so long-run "
            "coefficients are not defined.",
            DegenerateCaseWarning,
            stacklevel=2,
        )
        return pd.Series(np.full(params.k, np.nan), index=names, name="theta")
    denom = 1.0 - np.sum(params.phi)
    theta = np.array([float(np.sum(b)) for b in params.beta]) / denom
    return pd.Series(theta, index=names, name="theta")


def longrun_covariance(params: ARDLParams, v: FloatArray | None = None) -> FloatArray:
    """Covariance matrix of the long-run coefficients, by the delta method.

    Uses the analytical gradient

        d(theta_j)/d(beta_{j,i}) = 1 / (1 - sum phi)
        d(theta_j)/d(phi_i)      = theta_j / (1 - sum phi)

    with zero entries for the parameters of the other regressors and for
    the deterministic terms.

    Parameters
    ----------
    params : ARDLParams
        Must carry ``cov_params`` (the covariance matrix of
        ``params.param_vector()``) unless ``v`` is given.
    v : ndarray, optional
        Covariance matrix to use instead of ``params.cov_params``.

    Returns
    -------
    ndarray, shape (k, k)
        Covariance matrix of ``(theta_0, ..., theta_{k-1})``.
    """
    v_hat = v if v is not None else params.cov_params
    if v_hat is None:
        raise ValueError(
            "cov_params is required, either on params or through the v argument."
        )

    denom = 1.0 - float(np.sum(params.phi))
    theta = np.array([float(np.sum(b)) for b in params.beta]) / denom

    n_lead = (1 if params.has_const else 0) + (1 if params.has_trend else 0)
    p, k = params.p, params.k
    n_params = v_hat.shape[0]
    jac = np.zeros((k, n_params), dtype=np.float64)

    phi_slice = slice(n_lead, n_lead + p)
    offset = n_lead + p
    beta_slices = []
    for b in params.beta:
        beta_slices.append(slice(offset, offset + b.shape[0]))
        offset += b.shape[0]

    for j in range(k):
        jac[j, phi_slice] = theta[j] / denom
        jac[j, beta_slices[j]] = 1.0 / denom

    result = jac @ v_hat @ jac.T
    return result.astype(np.float64)


def half_life(params: ARDLParams) -> float:
    """Half-life of the return to equilibrium: ``ln(0.5) / ln(1 + lam)``.

    This is the number of periods needed to absorb half of a shock. It is
    only meaningful when ``-1 < lam < 0``; otherwise the function returns
    NaN and issues a :class:`~pyardl.exceptions.DegenerateCaseWarning`.
    """
    lam = speed_of_adjustment(params)
    if not (-1.0 < lam < 0.0):
        warnings.warn(
            "half_life is undefined: lambda lies outside (-1, 0), so there is "
            "no geometric convergence to a long-run equilibrium.",
            DegenerateCaseWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(np.log(0.5) / np.log(1.0 + lam))
