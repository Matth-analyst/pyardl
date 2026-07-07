r"""Algèbre exacte ARDL <-> ECM (spec 03 — Sargan 1964).

Réécriture exacte d'un modèle ARDL(p, q_1, ..., q_k) :

    y_t = alpha + delta*t + sum_i phi_i y_{t-i}
          + sum_j sum_i beta_{j,i} x_{j,t-i} + eps_t

sous forme de mécanisme à correction d'erreur (ECM) :

    Δy_t = alpha + delta*t + lam*y_{t-1} + sum_j gamma_j x_{j,t-1}
           + sum_i psi_i Δy_{t-i} + sum_j sum_i omega_{j,i} Δx_{j,t-i} + eps_t

Formules de passage (spec 03 §2.2, dérivées par sommation par parties,
voir ``docs/QUESTIONS.md`` pour le traitement du cas limite q_j = 0) :

    lam = -(1 - sum_i phi_i)
    gamma_j = sum_i beta_{j,i}
    psi_i = -sum_{m=i+1}^{p} phi_m,            i = 1, ..., p-1
    omega_{j,0} = beta_{j,0}
    omega_{j,i} = -sum_{m=i+1}^{q_j} beta_{j,m}, i = 1, ..., q_j-1
    theta_j = -gamma_j / lam = sum_i beta_{j,i} / (1 - sum_i phi_i)   (long terme)

Les transformations sont vectorisées par sommes cumulées inversées
(``np.cumsum``), sans boucle Python sur l'ordre des retards — seule une
boucle sur les k régresseurs (peu nombreux) subsiste, chaque régresseur
ayant son propre ordre q_j.

Références
----------
Sargan, J. D. (1964). "Wages and Prices in the United Kingdom: A Study in
Econometric Methodology". Clé BibTeX : ``sargan1964wages``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from ardlpy.exceptions import DegenerateCaseWarning

_LAMBDA_TOL = 1e-8

FloatArray = npt.NDArray[np.float64]


def _as_float_array(a: npt.ArrayLike) -> FloatArray:
    return np.asarray(a, dtype=np.float64)


@dataclass(frozen=True)
class ARDLParams:
    """Paramètres d'un modèle ARDL(p, q_1, ..., q_k).

    Parameters
    ----------
    p : int
        Ordre autorégressif (nombre de retards de y), p >= 1.
    q : tuple of int
        Ordre des retards de chaque régresseur x_j, q_j >= 0.
    phi : ndarray, shape (p,)
        Coefficients phi_1, ..., phi_p (retards de y).
    beta : tuple of ndarray
        beta[j] a la forme (q_j + 1,) : coefficients beta_{j,0}, ..., beta_{j,q_j}.
    const : float
        Constante (0.0 si absente).
    trend : float
        Coefficient de tendance linéaire (0.0 si absente).
    has_const, has_trend : bool
        Présence effective de la constante / tendance dans le modèle estimé
        (contrôle l'ordre du vecteur de paramètres pour ``cov_params``).
    x_names : tuple of str, optional
        Noms des régresseurs x (longueur k), pour l'affichage.
    cov_params : ndarray, optional
        Matrice de covariance du vecteur de paramètres complet, dans
        l'ordre défini par :func:`param_vector` (const?, trend?, phi,
        beta[0], beta[1], ...).
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
            raise ValueError("p doit être >= 1 (au moins y_{t-1}).")
        if self.phi.shape != (self.p,):
            raise ValueError(f"phi doit avoir la forme ({self.p},).")
        if len(self.q) != len(self.beta):
            raise ValueError("q et beta doivent avoir la même longueur (k).")
        for j, (qj, bj) in enumerate(zip(self.q, self.beta, strict=True)):
            if qj < 0:
                raise ValueError(f"q[{j}] doit être >= 0.")
            if bj.shape != (qj + 1,):
                raise ValueError(f"beta[{j}] doit avoir la forme ({qj + 1},).")
        if self.x_names is not None and len(self.x_names) != len(self.q):
            raise ValueError("x_names doit avoir la même longueur que q.")

    @property
    def k(self) -> int:
        """Nombre de régresseurs x."""
        return len(self.q)

    def param_vector(self) -> FloatArray:
        """Vecteur de paramètres empilé, ordre : const?, trend?, phi, beta[0], ...

        Cet ordre est le contrat utilisé par ``cov_params`` et par
        :func:`longrun_covariance` (spec 03 §3.2).
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
    """Paramètres de la forme ECM équivalente (spec 03 §2.2).

    Parameters
    ----------
    p, q : voir :class:`ARDLParams`.
    lam : float
        Vitesse d'ajustement lambda = -(1 - sum phi_i).
    gamma : ndarray, shape (k,)
        Coefficients de niveau x_{j,t-1}.
    psi : ndarray, shape (p-1,)
        Coefficients des Δy_{t-i}, i = 1, ..., p-1.
    omega : tuple of ndarray
        omega[j] a la forme (q_j,) : coefficients des Δx_{j,t-i},
        i = 0, ..., q_j - 1. Si q_j = 0, omega[j] est vide : le régresseur
        x_j n'a pas de dynamique de court terme propre et gamma_j
        multiplie alors x_{j,t} (contemporain) et non x_{j,t-1} dans la
        régression ECM — cf. docs/QUESTIONS.md pour la justification
        (dimension du sous-espace engendré) et spec 05 pour la
        construction de la matrice de dessin correspondante.

        Cette convention q_j = 0 est celle de Stata ``ardl`` (qui
        accepte q_j = 0 et fait entrer x_{j,t} contemporain dans la
        partie de niveau de l'EC) ; ``statsmodels.tsa.ardl.UECM``, en
        revanche, refuse q_j = 0 à la construction (``ValueError: All
        included exog variables must have a lag length >= 1``) — ardlpy
        le supporte.
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
            raise ValueError("p doit être >= 1.")
        if self.psi.shape != (max(self.p - 1, 0),):
            raise ValueError(f"psi doit avoir la forme ({max(self.p - 1, 0)},).")
        if len(self.q) != len(self.omega) or len(self.q) != self.gamma.shape[0]:
            raise ValueError("q, gamma et omega doivent avoir la même longueur (k).")
        for j, (qj, oj) in enumerate(zip(self.q, self.omega, strict=True)):
            if oj.shape != (qj,):
                raise ValueError(f"omega[{j}] doit avoir la forme ({qj},).")

    @property
    def k(self) -> int:
        return len(self.q)


def ardl_to_ecm(params: ARDLParams) -> ECMParams:
    """Transforme des paramètres ARDL en paramètres ECM (reparamétrisation exacte).

    Voir le module docstring pour les formules ; vectorisé via
    ``np.cumsum`` (spec 03 §3, point 1).

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

    # psi_i = -sum_{m=i+1}^{p} phi_m, i = 1..p-1  (S[i] = sum(phi[i:]), 0-indexed)
    s = np.cumsum(params.phi[::-1])[::-1]
    psi = -s[1:] if params.p > 1 else np.array([], dtype=np.float64)

    gamma = np.empty(params.k, dtype=np.float64)
    omega: list[FloatArray] = []
    for j, (qj, bj) in enumerate(zip(params.q, params.beta, strict=True)):
        gamma[j] = float(np.sum(bj))
        if qj == 0:
            # Pas de dynamique de court terme propre : gamma_j multipliera
            # x_{j,t} (contemporain) dans la régression, pas x_{j,t-1}
            # (sinon sur-paramétrisation, cf. docs/QUESTIONS.md).
            omega.append(np.array([], dtype=np.float64))
            continue
        # C_i = sum_{m=i}^{qj} beta_{j,m}, 0-indexed, longueur qj+1
        c = np.cumsum(bj[::-1])[::-1]
        c1 = c[1]
        if qj >= 2:
            omega_j = np.concatenate(([c[0] - c1], -c[2:]))
        else:
            # qj == 1 : un seul terme omega_{j,0} = C_0 - C_1
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
    """Transforme des paramètres ECM en paramètres ARDL (inversion séquentielle).

    Système triangulaire résolu par sommes cumulées (spec 03 §2.3).

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
    # D_1 = 1 + lam ; D_i = -psi_{i-1} pour i = 2..p ; D_{p+1} = 0
    d = np.concatenate(([1.0 + params.lam], -params.psi, [0.0]))
    phi = -np.diff(d)  # longueur p

    beta: list[FloatArray] = []
    for qj, gamma_j, omega_j in zip(params.q, params.gamma, params.omega, strict=True):
        if qj == 0:
            beta.append(np.array([gamma_j]))
            continue
        # C_0 = gamma_j ; C_1 = gamma_j - omega_{j,0} ; C_i = -omega_{j,i-1}, i=2..qj
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
    """Vitesse d'ajustement lambda = -(1 - sum phi_i)."""
    return -(1.0 - float(np.sum(params.phi)))


def longrun_coefs(params: ARDLParams, *, tol: float = _LAMBDA_TOL) -> pd.Series:
    """Coefficients de long terme theta_j = sum_i beta_{j,i} / (1 - sum_i phi_i).

    Émet :class:`~ardlpy.exceptions.DegenerateCaseWarning` et renvoie des
    NaN si |lambda| < tol (absence de force de rappel, spec 03 §3.4).
    """
    lam = speed_of_adjustment(params)
    names = params.x_names or tuple(f"x{j}" for j in range(params.k))
    if abs(lam) < tol:
        warnings.warn(
            "lambda ~ 0 : pas de force de rappel, les coefficients de long "
            "terme ne sont pas définis (cf. specs 14-15, dégénérescences).",
            DegenerateCaseWarning,
            stacklevel=2,
        )
        return pd.Series(np.full(params.k, np.nan), index=names, name="theta")
    denom = 1.0 - np.sum(params.phi)
    theta = np.array([float(np.sum(b)) for b in params.beta]) / denom
    return pd.Series(theta, index=names, name="theta")


def longrun_covariance(params: ARDLParams, v: FloatArray | None = None) -> FloatArray:
    """Covariance des coefficients de long terme theta_j par méthode delta.

    Gradient analytique (spec 03 §3, point 2) :
    d(theta_j)/d(beta_{j,i}) = 1 / (1 - sum phi) ;
    d(theta_j)/d(phi_i) = theta_j / (1 - sum phi) ;
    0 pour les paramètres des autres régresseurs et pour const/trend.

    Parameters
    ----------
    params : ARDLParams
        Doit porter ``cov_params`` (matrice de covariance du vecteur
        ``params.param_vector()``) si ``v`` n'est pas fourni.
    v : ndarray, optional
        Matrice de covariance à utiliser à la place de ``params.cov_params``.

    Returns
    -------
    ndarray, shape (k, k)
        Matrice de covariance de theta = (theta_0, ..., theta_{k-1}).
    """
    v_hat = v if v is not None else params.cov_params
    if v_hat is None:
        raise ValueError("cov_params requis (sur params ou en argument v).")

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
    """Demi-vie du retour à l'équilibre : ln(0.5) / ln(1 + lambda).

    Valide uniquement si -1 < lambda < 0 (spec 03 §3, point 3) ; NaN +
    :class:`~ardlpy.exceptions.DegenerateCaseWarning` sinon.
    """
    lam = speed_of_adjustment(params)
    if not (-1.0 < lam < 0.0):
        warnings.warn(
            "half_life non défini : lambda hors de (-1, 0), pas de "
            "convergence géométrique vers l'équilibre de long terme.",
            DegenerateCaseWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(np.log(0.5) / np.log(1.0 + lam))
