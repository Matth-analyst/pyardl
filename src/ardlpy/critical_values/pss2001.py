"""Bornes asymptotiques du bounds test PSS 2001 (spec 10 §4).

Tables CI(i)-(v) (statistique F) et CII(i)/(iii)/(v) (statistique t) de
Pesaran, Shin & Smith (2001), pp. 300-303. Provenance détaillée, seconde
source de recoupement et limitations de couverture : voir PROVENANCE.md
dans ce dossier — NE PAS modifier ces nombres sans mettre PROVENANCE.md
à jour.

Couverture : cas I-V, k = 0..10, seuils 10 %, 5 %, 2.5 %, 1 %. Les
seuils 10/5/1 % sont les valeurs PUBLIÉES par PSS 2001 (servies à
l'identique — fonction : reproduction de la littérature) ; le seuil
2.5 % provient du moteur de simulation interne (spec 12,
``pss2001_p025.py``, provenance distincte — marqué needs_review jusqu'à
vérification contre l'article original). Toute combinaison non couverte
lève une exception explicite — jamais de
substitution silencieuse.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["get_bounds", "LEVELS", "MAX_K"]

LEVELS: tuple[float, ...] = (0.10, 0.05, 0.01)
MAX_K = 10

# Chaque table : shape (11, 3, 2) = (k = 0..10, seuil ∈ {10%, 5%, 1%},
# borne ∈ {I(0), I(1)}). Source : PSS 2001, tables CI/CII (PROVENANCE.md).

_F = {
    # Table CI(i) — cas I : ni constante ni tendance
    1: [
        [(3.00, 3.00), (4.20, 4.20), (7.17, 7.17)],
        [(2.44, 3.28), (3.15, 4.11), (4.81, 6.02)],
        [(2.17, 3.19), (2.72, 3.83), (3.88, 5.30)],
        [(2.01, 3.10), (2.45, 3.63), (3.42, 4.84)],
        [(1.90, 3.01), (2.26, 3.48), (3.07, 4.44)],
        [(1.81, 2.93), (2.14, 3.34), (2.82, 4.21)],
        [(1.75, 2.87), (2.04, 3.24), (2.66, 4.05)],
        [(1.70, 2.83), (1.97, 3.18), (2.54, 3.91)],
        [(1.66, 2.79), (1.91, 3.11), (2.45, 3.79)],
        [(1.63, 2.75), (1.86, 3.05), (2.34, 3.68)],
        [(1.60, 2.72), (1.82, 2.99), (2.26, 3.60)],
    ],
    # Table CI(ii) — cas II : constante restreinte, pas de tendance
    2: [
        [(3.80, 3.80), (4.60, 4.60), (6.44, 6.44)],
        [(3.02, 3.51), (3.62, 4.16), (4.94, 5.58)],
        [(2.63, 3.35), (3.10, 3.87), (4.13, 5.00)],
        [(2.37, 3.20), (2.79, 3.67), (3.65, 4.66)],
        [(2.20, 3.09), (2.56, 3.49), (3.29, 4.37)],
        [(2.08, 3.00), (2.39, 3.38), (3.06, 4.15)],
        [(1.99, 2.94), (2.27, 3.28), (2.88, 3.99)],
        [(1.92, 2.89), (2.17, 3.21), (2.73, 3.90)],
        [(1.85, 2.85), (2.11, 3.15), (2.62, 3.77)],
        [(1.80, 2.80), (2.04, 3.08), (2.50, 3.68)],
        [(1.76, 2.77), (1.98, 3.04), (2.41, 3.61)],
    ],
    # Table CI(iii) — cas III : constante non restreinte, pas de tendance
    3: [
        [(6.58, 6.58), (8.21, 8.21), (11.79, 11.79)],
        [(4.04, 4.78), (4.94, 5.73), (6.84, 7.84)],
        [(3.17, 4.14), (3.79, 4.85), (5.15, 6.36)],
        [(2.72, 3.77), (3.23, 4.35), (4.29, 5.61)],
        [(2.45, 3.52), (2.86, 4.01), (3.74, 5.06)],
        [(2.26, 3.35), (2.62, 3.79), (3.41, 4.68)],
        [(2.12, 3.23), (2.45, 3.61), (3.15, 4.43)],
        [(2.03, 3.13), (2.32, 3.50), (2.96, 4.26)],
        [(1.95, 3.06), (2.22, 3.39), (2.79, 4.10)],
        [(1.88, 2.99), (2.14, 3.30), (2.65, 3.97)],
        [(1.83, 2.94), (2.06, 3.24), (2.54, 3.86)],
    ],
    # Table CI(iv) — cas IV : constante non restreinte, tendance restreinte
    4: [
        [(5.37, 5.37), (6.29, 6.29), (8.26, 8.26)],
        [(4.05, 4.49), (4.68, 5.15), (6.10, 6.73)],
        [(3.38, 4.02), (3.88, 4.61), (4.99, 5.85)],
        [(2.97, 3.74), (3.38, 4.23), (4.30, 5.23)],
        [(2.68, 3.53), (3.05, 3.97), (3.81, 4.92)],
        [(2.49, 3.38), (2.81, 3.76), (3.50, 4.63)],
        [(2.33, 3.25), (2.63, 3.62), (3.27, 4.39)],
        [(2.22, 3.17), (2.50, 3.50), (3.07, 4.23)],
        [(2.13, 3.09), (2.38, 3.41), (2.93, 4.06)],
        [(2.05, 3.02), (2.30, 3.33), (2.79, 3.93)],
        [(1.98, 2.97), (2.21, 3.25), (2.68, 3.84)],
    ],
    # Table CI(v) — cas V : constante et tendance non restreintes
    5: [
        [(9.81, 9.81), (11.64, 11.64), (15.73, 15.73)],
        [(5.59, 6.26), (6.56, 7.30), (8.74, 9.63)],
        [(4.19, 5.06), (4.87, 5.85), (6.34, 7.52)],
        [(3.47, 4.45), (4.01, 5.07), (5.17, 6.36)],
        [(3.03, 4.06), (3.47, 4.57), (4.40, 5.72)],
        [(2.75, 3.79), (3.12, 4.25), (3.93, 5.23)],
        [(2.53, 3.59), (2.87, 4.00), (3.60, 4.90)],
        [(2.38, 3.45), (2.69, 3.83), (3.34, 4.63)],
        [(2.26, 3.34), (2.55, 3.68), (3.15, 4.43)],
        [(2.16, 3.24), (2.43, 3.56), (2.97, 4.24)],
        [(2.07, 3.16), (2.33, 3.46), (2.84, 4.10)],
    ],
}

_T = {
    # Table CII(i) — cas I
    1: [
        [(-1.62, -1.62), (-1.95, -1.95), (-2.58, -2.58)],
        [(-1.62, -2.28), (-1.95, -2.60), (-2.58, -3.22)],
        [(-1.62, -2.68), (-1.95, -3.02), (-2.58, -3.66)],
        [(-1.62, -3.00), (-1.95, -3.33), (-2.58, -3.97)],
        [(-1.62, -3.26), (-1.95, -3.60), (-2.58, -4.23)],
        [(-1.62, -3.49), (-1.95, -3.83), (-2.58, -4.44)],
        [(-1.62, -3.70), (-1.95, -4.04), (-2.58, -4.67)],
        [(-1.62, -3.90), (-1.95, -4.23), (-2.58, -4.88)],
        [(-1.62, -4.09), (-1.95, -4.43), (-2.58, -5.07)],
        [(-1.62, -4.26), (-1.95, -4.61), (-2.58, -5.25)],
        [(-1.62, -4.42), (-1.95, -4.76), (-2.58, -5.44)],
    ],
    # Table CII(iii) — cas III
    3: [
        [(-2.57, -2.57), (-2.86, -2.86), (-3.43, -3.43)],
        [(-2.57, -2.91), (-2.86, -3.22), (-3.43, -3.82)],
        [(-2.57, -3.21), (-2.86, -3.53), (-3.43, -4.10)],
        [(-2.57, -3.46), (-2.86, -3.78), (-3.43, -4.37)],
        [(-2.57, -3.66), (-2.86, -3.99), (-3.43, -4.60)],
        [(-2.57, -3.86), (-2.86, -4.19), (-3.43, -4.79)],
        [(-2.57, -4.04), (-2.86, -4.38), (-3.43, -4.99)],
        [(-2.57, -4.23), (-2.86, -4.57), (-3.43, -5.19)],
        [(-2.57, -4.40), (-2.86, -4.72), (-3.43, -5.37)],
        [(-2.57, -4.56), (-2.86, -4.88), (-3.43, -5.54)],
        [(-2.57, -4.69), (-2.86, -5.03), (-3.43, -5.68)],
    ],
    # Table CII(v) — cas V
    5: [
        [(-3.13, -3.13), (-3.41, -3.41), (-3.96, -3.97)],
        [(-3.13, -3.40), (-3.41, -3.69), (-3.96, -4.26)],
        [(-3.13, -3.63), (-3.41, -3.95), (-3.96, -4.53)],
        [(-3.13, -3.84), (-3.41, -4.16), (-3.96, -4.73)],
        [(-3.13, -4.04), (-3.41, -4.36), (-3.96, -4.96)],
        [(-3.13, -4.21), (-3.41, -4.52), (-3.96, -5.13)],
        [(-3.13, -4.37), (-3.41, -4.69), (-3.96, -5.31)],
        [(-3.13, -4.53), (-3.41, -4.85), (-3.96, -5.49)],
        [(-3.13, -4.68), (-3.41, -5.01), (-3.96, -5.65)],
        [(-3.13, -4.82), (-3.41, -5.15), (-3.96, -5.79)],
        [(-3.13, -4.96), (-3.41, -5.29), (-3.96, -5.94)],
    ],
}

F_BOUNDS: dict[int, npt.NDArray[np.float64]] = {
    case: np.asarray(rows, dtype=np.float64) for case, rows in _F.items()
}
T_BOUNDS: dict[int, npt.NDArray[np.float64]] = {
    case: np.asarray(rows, dtype=np.float64) for case, rows in _T.items()
}


def get_bounds(
    stat: Literal["F", "t"],
    case: int,
    k: int,
    alpha: float,
) -> tuple[float, float]:
    """Bornes asymptotiques PSS 2001 pour le bounds test (spec 10 §4.2).

    Parameters
    ----------
    stat : {"F", "t"}
        Statistique testée : F_overall ou t_BDM.
    case : int
        Cas déterministe PSS (1 à 5). Le t n'est tabulé que pour les cas
        I, III et V (PSS 2001 ne publie pas de bornes t pour les cas à
        déterministes restreints).
    k : int
        Nombre de régresseurs de niveau (0 à 10).
    alpha : float
        Seuil : 0.10, 0.05 ou 0.01. Le seuil 2.5 % de l'article n'est
        pas encore couvert (dette spec 12, cf. PROVENANCE.md).

    Returns
    -------
    (lower, upper) : tuple of float
        Borne I(0) (inférieure) et borne I(1) (supérieure). Pour le t,
        « inférieure/supérieure » s'entend en valeur absolue : la borne
        I(1) est la plus négative (test unilatéral gauche).

    Raises
    ------
    ValueError
        Combinaison non couverte par les tables (jamais de substitution
        silencieuse — règle du projet).

    Examples
    --------
    >>> get_bounds("F", case=3, k=1, alpha=0.05)
    (4.94, 5.73)
    >>> get_bounds("t", case=3, k=1, alpha=0.05)
    (-2.86, -3.22)
    """
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case doit être dans 1..5, reçu {case}.")
    if not 0 <= k <= MAX_K:
        raise ValueError(
            f"k={k} hors des tables PSS 2001 (k = 0..{MAX_K}) ; pour k plus "
            "grand, utiliser le moteur de simulation (spec 12) ou les "
            "surfaces de réponse (spec 13)."
        )
    if stat not in ("F", "t"):
        raise ValueError(f'stat doit être "F" ou "t", reçu {stat!r}.')
    if stat == "t" and case not in T_BOUNDS:
        raise ValueError(
            f"PSS 2001 ne publie pas de bornes t pour le cas {case} "
            "(déterministes restreints) : la statistique t_BDM n'y est "
            "pas applicable — utiliser le F_overall (tables CI) ou les "
            "cas III/V."
        )

    if alpha == 0.025:
        # Seuil non transcrit des tables publiées : simulation interne
        # (spec 12, provenance et statut needs_review : PROVENANCE.md).
        from ardlpy.critical_values.pss2001_p025 import F_P025, T_P025

        cell = (F_P025 if stat == "F" else T_P025)[case][k]
        return float(cell[0]), float(cell[1])

    try:
        level_idx = LEVELS.index(alpha)
    except ValueError:
        raise ValueError(
            f"alpha={alpha} non couvert : seuils disponibles "
            f"{(*LEVELS, 0.025)} (cf. PROVENANCE.md)."
        ) from None

    table = F_BOUNDS[case] if stat == "F" else T_BOUNDS[case]
    lower, upper = table[k, level_idx]
    return float(lower), float(upper)
