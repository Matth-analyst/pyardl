r"""Surfaces de réponse Kripfganz-Schneider 2020 — voie A1 (spec 13 §2.1).

Valeurs critiques asymptotiques à tout seuil et p-values approchées
(les deux bornes) pour la statistique F du bounds test, via le matériel
embarqué dans ``statsmodels.tsa.ardl.pss_critical_values``.

Provenance (détail : PROVENANCE.md) : ce matériel n'est PAS une
redistribution des coefficients publiés par K&S — statsmodels a
RE-SIMULÉ les distributions (32 000 000 de réplications par
configuration, méthodologie PSS/K&S) et ajusté ses propres polynômes de
p-values asymptotiques. Licence BSD-3 de statsmodels, dépendance
runtime déjà requise : aucune redistribution par pyardl.

Forme fonctionnelle des p-values (docstring statsmodels) :

    p = 1 - Phi( c0 + c1*x + c2*x^2 [+ c3*x^3] ),  x = log(F)

avec bascule entre le polynôme d'ordre 3 (``large_p``, F <= stat_star)
et d'ordre 2 (``small_p``, F > stat_star). Les CV à seuil arbitraire
sont obtenus par inversion numérique (brentq) de cette fonction
strictement décroissante ; aux percentiles simulés (90/95/99/99.9), les
estimations ponctuelles des quantiles sont servies directement.

Limitations de la voie A1 (exceptions explicites, jamais de
substitution silencieuse — règle du projet) :

- asymptotique seulement : l'ajustement continu en T des surfaces
  complètes de K&S nécessite leurs coefficients Stata (voie A2, licence
  en cours de clarification) ou nos propres surfaces (voie B, v0.4+) ;
- statistique F seulement (pas de t dans le matériel statsmodels) ;
- k = 1..10 (pas de k = 0).

Références
----------
Kripfganz, S. & Schneider, D. C. (2020). "Response Surface Regressions
for Critical Value Bounds and Approximate p-values in Equilibrium
Correction Models", *Oxford Bulletin of Economics and Statistics*,
82(6), 1456-1481. Clé BibTeX : ``kripfganz2020response``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from statsmodels.tsa.ardl import pss_critical_values as _sm_pss

__all__ = ["crit_value_bounds", "pvalue_bounds", "MAX_K_KS"]

MAX_K_KS = 10
_PERCENTILE_ALPHAS = {0.10: 0, 0.05: 1, 0.01: 2, 0.001: 3}


def _check_coverage(stat: str, case: int, k: int) -> None:
    if stat != "F":
        raise ValueError(
            "Voie A1 (matériel statsmodels) : statistique F seulement — "
            'le t_BDM n\'y est pas couvert ; utiliser cv_source="pss" '
            "pour les bornes t (les surfaces t arriveront avec la voie "
            "A2 ou B, cf. PROVENANCE.md)."
        )
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case doit être dans 1..5, reçu {case}.")
    if not 1 <= k <= MAX_K_KS:
        raise ValueError(
            f"k={k} hors couverture Kripfganz-Schneider/statsmodels "
            f'(k = 1..{MAX_K_KS}) ; pour k=0, utiliser cv_source="pss" '
            "ou le moteur de simulation (spec 12)."
        )


def _pvalue_one(stat: float, key: tuple[int, int, bool]) -> float:
    """p-value approchée pour une borne (clé (k, case, I1))."""
    if stat <= 0:
        return 1.0
    x = np.log(stat)
    coefs = (
        _sm_pss.large_p[key] if stat <= _sm_pss.stat_star[key] else _sm_pss.small_p[key]
    )
    y = sum(c * x**i for i, c in enumerate(coefs))
    return float(1 - norm.cdf(y))


def pvalue_bounds(
    f_stat: float,
    case: int,
    k: int,
) -> tuple[float, float]:
    """p-values approchées du F_overall aux deux bornes (spec 13 §2.1.3).

    Parameters
    ----------
    f_stat : float
        Statistique F observée.
    case : int
        Cas déterministe PSS (1 à 5).
    k : int
        Nombre de régresseurs de niveau (1 à 10).

    Returns
    -------
    (p_i0, p_i1) : tuple of float
        p-value sous « tout I(0) » (borne inférieure) et sous
        « tout I(1) » (borne supérieure) — p_i0 <= p_i1. Lecture :
        p_i1 <= alpha -> cointégration ; p_i0 > alpha -> non-rejet ;
        alpha entre les deux -> zone non concluante (« inconclusive,
        p ∈ [p_I1, p_I0] »).

    Examples
    --------
    >>> p_i0, p_i1 = pvalue_bounds(6.0, case=3, k=1)
    >>> p_i0 < 0.05 < p_i1  # 4.94 < 5.73 < 6.0 -> rejet aux deux bornes
    False
    >>> round(p_i0, 3) < round(p_i1, 3) < 0.05
    True
    """
    _check_coverage("F", case, k)
    p_i0 = _pvalue_one(f_stat, (k, case, False))
    p_i1 = _pvalue_one(f_stat, (k, case, True))
    return p_i0, p_i1


def crit_value_bounds(
    case: int,
    k: int,
    alpha: float,
) -> tuple[float, float]:
    """CV asymptotiques (I0, I1) du F à un seuil quelconque (spec 13).

    Aux seuils simulés (10/5/1/0.1 %), sert les estimations ponctuelles
    des quantiles (32M de réplications) ; sinon, inverse numériquement
    la fonction de p-value (brentq).

    Examples
    --------
    >>> lo, up = crit_value_bounds(case=3, k=1, alpha=0.05)
    >>> round(lo, 2), round(up, 2)  # PSS publié : (4.94, 5.73)
    (4.92, 5.72)
    """
    _check_coverage("F", case, k)
    if not 0.0005 < alpha < 0.25:
        raise ValueError(
            f"alpha={alpha} hors du domaine fiable des surfaces "
            "(0.0005 < alpha < 0.25)."
        )

    if alpha in _PERCENTILE_ALPHAS:
        idx = _PERCENTILE_ALPHAS[alpha]
        return (
            float(_sm_pss.crit_vals[(k, case, False)][idx]),
            float(_sm_pss.crit_vals[(k, case, True)][idx]),
        )

    out = []
    for i1 in (False, True):
        key = (k, case, i1)

        def objective(s: float, key: tuple[int, int, bool] = key) -> float:
            return _pvalue_one(s, key) - alpha

        out.append(float(brentq(objective, 0.05, 100.0)))
    return out[0], out[1]
