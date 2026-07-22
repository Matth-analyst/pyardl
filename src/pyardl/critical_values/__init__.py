"""Valeurs critiques des bounds tests (specs 10, 12, 13).

Toute table encodée cite sa source exacte dans PROVENANCE.md et est recoupée par une seconde source ou par le
moteur de simulation interne (spec 12).

Politique et hiérarchie des sources (spec 12 §2.4, arbitrage
2026-07-07) :

- ``"pss"`` : bornes asymptotiques PSS 2001, servies À L'IDENTIQUE des
  valeurs publiées — leur fonction est la REPRODUCTION DE LA
  LITTÉRATURE. Elles portent l'erreur MC d'origine de l'article
  (40 000 réplications ; ~±0.05 aux seuils usuels, jusqu'à ~±0.15 dans
  la queue à 1 % — quantification : PROVENANCE.md). Exception : le
  seuil 2.5 % provient du moteur interne (non transcrit).
- ``"narayan"`` : bornes petits échantillons de Narayan 2005
  (30 <= T <= 80, cas II/III/V, k <= 7, F seulement), valeurs publiées
  à l'identique — recommandé sur données annuelles courtes.
- ``"kripfganz"`` : surfaces de réponse (spec 13, voie A1 via
  statsmodels) — source PAR DÉFAUT et RECOMMANDÉE : plus précise que
  les tables publiées (32M de réplications par configuration contre
  40k), tout seuil alpha, p-values aux deux bornes
  (:func:`pvalue_bounds`). Limites voie A1 : asymptotique (ajustement
  fini-T -> voie A2/B), F seulement, k = 1..10.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from pyardl.critical_values.ks2020 import crit_value_bounds as _ks_bounds
from pyardl.critical_values.ks2020 import pvalue_bounds
from pyardl.critical_values.narayan2005 import (
    F_NARAYAN,
    MAX_K_NARAYAN,
    T_GRID,
)
from pyardl.critical_values.pss2001 import get_bounds as _pss_bounds
from pyardl.critical_values.simulate import SimulatedBounds, simulate_bounds
from pyardl.exceptions import PyardlMethodologyWarning

__all__ = ["get_bounds", "pvalue_bounds", "simulate_bounds", "SimulatedBounds"]

_NARAYAN_LEVELS = {0.10: 0, 0.05: 1, 0.01: 2}


def _narayan_bounds(
    stat: str, case: int, k: int, alpha: float, t_obs: int
) -> tuple[float, float]:
    """Bornes de Narayan 2005 avec interpolation linéaire en T
    (spec 12 §2.1)."""
    if stat != "F":
        raise ValueError(
            "Narayan 2005 ne publie que des bornes F (pas de t_BDM) : "
            'utiliser cv_source="pss" pour le t, ou les surfaces de '
            "réponse (spec 13)."
        )
    if case in (1, 4):
        raise ValueError(
            f"Narayan 2005 ne couvre pas le cas {case} (cas II, III et V "
            'seulement) : utiliser cv_source="kripfganz" (spec 13) ou '
            '"pss" (asymptotique).'
        )
    if case not in (2, 3, 5):
        raise ValueError(f"case doit être dans 1..5, reçu {case}.")
    if k > MAX_K_NARAYAN:
        raise ValueError(
            f"k={k} hors des tables de Narayan (k <= {MAX_K_NARAYAN}) : "
            'utiliser cv_source="kripfganz" (spec 13) ou "pss".'
        )
    if alpha not in _NARAYAN_LEVELS:
        raise ValueError(
            f"alpha={alpha} non couvert par Narayan 2005 "
            f"(seuils {tuple(_NARAYAN_LEVELS)})."
        )

    if t_obs < T_GRID[0] or t_obs > T_GRID[-1]:
        warnings.warn(
            f"T={t_obs} hors de la plage des tables de Narayan "
            f"({T_GRID[0]}-{T_GRID[-1]}) : repli sur les bornes "
            "asymptotiques PSS 2001 (spec 12 §2.1).",
            PyardlMethodologyWarning,
            stacklevel=3,
        )
        return _pss_bounds("F", case, k, alpha)

    level = _NARAYAN_LEVELS[alpha]
    grid = np.asarray(T_GRID)
    hi = int(np.searchsorted(grid, t_obs))
    if grid[hi] == t_obs:
        cell = F_NARAYAN[t_obs][case][k, level]
        return float(cell[0]), float(cell[1])
    lo = hi - 1
    w = (t_obs - grid[lo]) / (grid[hi] - grid[lo])
    cell_lo = F_NARAYAN[int(grid[lo])][case][k, level]
    cell_hi = F_NARAYAN[int(grid[hi])][case][k, level]
    return (
        float((1 - w) * cell_lo[0] + w * cell_hi[0]),
        float((1 - w) * cell_lo[1] + w * cell_hi[1]),
    )


def get_bounds(
    stat: Literal["F", "t"],
    case: int,
    k: int,
    alpha: float,
    cv_source: Literal["pss", "narayan", "kripfganz"] = "pss",
    t_obs: int | None = None,
) -> tuple[float, float]:
    """Bornes du bounds test — interface commune (specs 10 §4.2, 12 §2.1).

    Parameters
    ----------
    stat, case, k, alpha
        Voir :func:`pyardl.critical_values.pss2001.get_bounds`.
    cv_source : {"pss", "narayan", "kripfganz"}
        Source des valeurs critiques (voir la politique du module).
    t_obs : int, optional
        Taille d'échantillon — requis pour ``"narayan"`` (interpolation
        linéaire entre les tailles tabulées 30..80 ; hors plage ->
        repli asymptotique + warning).

    Examples
    --------
    >>> get_bounds("F", case=3, k=1, alpha=0.05)
    (4.94, 5.73)
    >>> get_bounds("F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=40)
    (5.26, 6.16)
    """
    if cv_source == "pss":
        return _pss_bounds(stat, case, k, alpha)
    if cv_source == "narayan":
        if t_obs is None:
            raise ValueError('cv_source="narayan" requiert t_obs.')
        return _narayan_bounds(stat, case, k, alpha, t_obs)
    if cv_source == "kripfganz":
        # Voie A1 : asymptotique, F seulement (t -> exception explicite
        # de ks2020 pointant vers "pss") ; t_obs ignoré à ce stade
        # (ajustement fini-T : voie A2/B, cf. PROVENANCE.md).
        return _ks_bounds(case, k, alpha) if stat == "F" else _ks_t_error()
    raise ValueError(f"cv_source inconnu : {cv_source!r}.")


def _ks_t_error() -> tuple[float, float]:
    raise ValueError(
        'cv_source="kripfganz" (voie A1) ne couvre pas le t_BDM : '
        'utiliser cv_source="pss" pour les bornes t (surfaces t : voie '
        "A2/B, cf. PROVENANCE.md)."
    )
