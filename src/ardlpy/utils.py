"""Briques transversales réutilisées par tous les modules (spec 01 §3, 00_INDEX.md).

Ne pas dupliquer : toute fonction ici doit être importée, jamais réécrite
localement dans un module de modèle.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


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
