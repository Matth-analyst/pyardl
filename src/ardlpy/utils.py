"""Briques transversales réutilisées par tous les modules (spec 01 §3, 00_INDEX.md).

Ne pas dupliquer : toute fonction ici doit être importée, jamais réécrite
localement dans un module de modèle.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from ardlpy.exceptions import ArdlpyMethodologyWarning


def _delta_method(
    g: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64] | float],
    theta_hat: npt.NDArray[np.float64],
    v_hat: npt.NDArray[np.float64],
    *,
    step: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Méthode delta générique : g(theta_hat), Var(g) = grad(g)' V_hat grad(g).

    Le gradient de ``g`` est calculé par différences finies centrées
    (pas de forme analytique requise), ce qui en fait un helper générique
    réutilisable dans toute la bibliothèque (spec 01 §3). Les modules qui
    disposent d'un gradient analytique (ex. spec 03 §3.2) doivent
    l'utiliser en priorité et ne recourir à ce helper que pour la
    vérification croisée en test.

    Parameters
    ----------
    g : Callable[[ndarray], ndarray | float]
        Fonction (vectorisée ou scalaire) des paramètres, ``g(theta_hat)``
        donne la quantité dérivée dont on veut la variance.
    theta_hat : ndarray, shape (n_params,)
        Estimation ponctuelle du vecteur de paramètres.
    v_hat : ndarray, shape (n_params, n_params)
        Matrice de covariance estimée de ``theta_hat``.
    step : float
        Pas des différences finies centrées.

    Returns
    -------
    g_hat : ndarray
        Valeur de ``g`` évaluée en ``theta_hat`` (aplatie en 1D).
    cov_g : ndarray, shape (m, m)
        Matrice de covariance de ``g(theta_hat)`` (m = dimension de sortie
        de ``g``).

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.array([2.0, 0.5])
    >>> v = np.diag([0.01, 0.0004])
    >>> g = lambda t: np.array([t[0] / (1 - t[1])])
    >>> g_hat, cov_g = _delta_method(g, theta, v)
    >>> round(float(g_hat[0]), 6)
    4.0
    """
    theta_hat = np.asarray(theta_hat, dtype=np.float64)
    n = theta_hat.shape[0]

    g0 = np.atleast_1d(np.asarray(g(theta_hat), dtype=np.float64))
    m = g0.shape[0]

    jac = np.empty((m, n), dtype=np.float64)
    for i in range(n):
        bump = np.zeros(n, dtype=np.float64)
        bump[i] = step * max(abs(theta_hat[i]), 1.0)
        theta_plus = (theta_hat + bump).astype(np.float64)
        theta_minus = (theta_hat - bump).astype(np.float64)
        g_plus = np.atleast_1d(np.asarray(g(theta_plus), dtype=np.float64))
        g_minus = np.atleast_1d(np.asarray(g(theta_minus), dtype=np.float64))
        jac[:, i] = (g_plus - g_minus) / (2 * bump[i])

    cov_g = (jac @ v_hat @ jac.T).astype(np.float64)
    return g0, cov_g


def lag_matrix(
    x: npt.ArrayLike, lags: int, *, first_lag: int = 0
) -> npt.NDArray[np.float64]:
    """Matrice des retards de x (spec 02 §2, building block du cœur ARDL).

    Parameters
    ----------
    x : array-like, shape (T,)
        Série d'entrée.
    lags : int
        Retard maximal (>= first_lag).
    first_lag : int
        Premier retard inclus : 0 (défaut, colonnes x_t, ..., x_{t-lags})
        ou 1 (colonnes x_{t-1}, ..., x_{t-lags} — cas des retards de y
        dans un ARDL).

    Returns
    -------
    ndarray, shape (T - lags, lags - first_lag + 1)
        Colonne i = x_{t - (first_lag + i)}, alignée sur t = lags..T-1
        (0-indexé) : les ``lags`` premières observations sont perdues.

    Examples
    --------
    >>> import numpy as np
    >>> lag_matrix(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    array([[3., 2., 1.],
           [4., 3., 2.]])
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x doit être 1D.")
    if lags < first_lag or first_lag < 0:
        raise ValueError("lags >= first_lag >= 0 requis.")
    t_len = arr.shape[0]
    if t_len <= lags:
        raise ValueError(f"Série trop courte : T={t_len} <= lags={lags}.")
    cols = [arr[lags - i : t_len - i] for i in range(first_lag, lags + 1)]
    return np.column_stack(cols)


def check_series(
    y: npt.ArrayLike,
    x: npt.ArrayLike | None = None,
    *,
    min_obs: int = 15,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
    pd.Index | None,
    str,
    tuple[str, ...],
]:
    """Validation d'entrées commune à toute la bibliothèque (spec 01 §6).

    Vérifie : longueurs égales, pas de NaN interne (NaN de bord -> trim
    avec warning), n >= ``min_obs`` sinon warning petit échantillon,
    variance non nulle. Préserve l'index temporel pandas si fourni.

    Parameters
    ----------
    y : array-like, shape (T,)
        Variable dépendante.
    x : array-like, shape (T,) ou (T, k), optional
        Régresseurs (Series, DataFrame ou ndarray).
    min_obs : int
        Seuil du warning petit échantillon.

    Returns
    -------
    y_arr : ndarray, shape (T',)
    x_arr : ndarray, shape (T', k) ou None
    index : pd.Index ou None
        Index pandas aligné sur l'échantillon conservé (None si les
        entrées sont des ndarrays nus).
    y_name : str
    x_names : tuple of str

    Raises
    ------
    ValueError
        Longueurs différentes, NaN interne, ou variance nulle.
    """
    y_name = getattr(y, "name", None) or "y"
    index = y.index if isinstance(y, pd.Series) else None

    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValueError("y doit être 1D.")

    x_names: tuple[str, ...] = ()
    x_arr: npt.NDArray[np.float64] | None = None
    if x is not None:
        if isinstance(x, pd.DataFrame):
            x_names = tuple(str(c) for c in x.columns)
            if index is None:
                index = x.index
        elif isinstance(x, pd.Series):
            x_names = (str(x.name) if x.name is not None else "x0",)
            if index is None:
                index = x.index
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if x_arr.ndim != 2:
            raise ValueError("x doit être 1D ou 2D.")
        if not x_names:
            x_names = tuple(f"x{j}" for j in range(x_arr.shape[1]))
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Longueurs incompatibles : y a {y_arr.shape[0]} "
                f"observations, x en a {x_arr.shape[0]}."
            )

    # NaN de bord -> trim avec warning ; NaN interne -> erreur.
    stacked = y_arr[:, None] if x_arr is None else np.column_stack([y_arr, x_arr])
    valid = np.asarray(~np.isnan(stacked).any(axis=1), dtype=np.bool_)
    if not valid.all():
        first, last = (
            int(np.argmax(valid)),
            int(len(valid) - 1 - np.argmax(valid[::-1])),
        )
        if not valid[first : last + 1].all():
            raise ValueError("NaN interne détecté (seuls les NaN de bord sont trimés).")
        warnings.warn(
            f"NaN de bord : {int((~valid).sum())} observation(s) retirée(s).",
            ArdlpyMethodologyWarning,
            stacklevel=2,
        )
        y_arr = y_arr[first : last + 1]
        if x_arr is not None:
            x_arr = x_arr[first : last + 1]
        if index is not None:
            index = index[first : last + 1]

    if y_arr.shape[0] < min_obs:
        warnings.warn(
            f"Échantillon très petit (n={y_arr.shape[0]} < {min_obs}) : "
            "l'inférence asymptotique n'est pas fiable.",
            ArdlpyMethodologyWarning,
            stacklevel=2,
        )
    if np.var(y_arr) == 0.0:
        raise ValueError("y a une variance nulle.")
    if x_arr is not None and (np.var(x_arr, axis=0) == 0.0).any():
        raise ValueError("Au moins une colonne de x a une variance nulle.")

    return y_arr, x_arr, index, str(y_name), x_names
