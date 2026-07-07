"""Verrou spec 05 §3.2 / spec 02 §4 (piège de l'échantillon commun) :
TOUS les candidats de select_order sont estimés sur les MÊMES
observations, sinon les critères d'information ne sont pas comparables.
Écrit AVANT l'implémentation de select_order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ardlpy.core.ardl import ARDL


def _dgp(seed: int, n: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    """DGP ARDL(2, 1) : ordre vrai connu pour les tests de consistance."""
    rng = np.random.default_rng(seed)
    xv = rng.normal(size=n).cumsum()
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (
            0.5
            + 0.5 * y[t - 1]
            - 0.2 * y[t - 2]
            + 0.6 * xv[t]
            + 0.3 * xv[t - 1]
            + rng.normal(scale=0.4)
        )
    return pd.Series(y, name="y"), pd.DataFrame({"x": xv})


def test_all_candidates_on_common_sample() -> None:
    """Spec 05 §3.2 : le tableau de sélection rapporte le même nobs pour
    tous les candidats, égal à T - max(max_p, max_q)."""
    y, x = _dgp(seed=0)
    max_p, max_q = 4, 3
    sel = ARDL.select_order(y, x, max_p=max_p, max_q=max_q, ic="aic")

    assert sel.table["nobs"].nunique() == 1
    assert int(sel.table["nobs"].iloc[0]) == len(y) - max(max_p, max_q)


def test_ic_computed_on_common_sample_not_maximal() -> None:
    """L'AIC rapporté pour un candidat d'ordre faible doit être celui de
    l'échantillon COMMUN (hold_back = max), pas celui de son échantillon
    maximal propre — c'est exactement le piège que ce test verrouille."""
    y, x = _dgp(seed=1)
    max_p, max_q = 4, 3
    hold_back = max(max_p, max_q)
    sel = ARDL.select_order(y, x, max_p=max_p, max_q=max_q, ic="aic")

    row = sel.table[(sel.table["p"] == 1) & (sel.table["q_x"] == 0)].iloc[0]

    aic_common = ARDL(y, x, order=(1, 0), det="const", hold_back=hold_back).fit().aic
    aic_maximal = ARDL(y, x, order=(1, 0), det="const").fit().aic

    assert row["aic"] == pytest.approx(aic_common, abs=1e-8)
    assert abs(aic_common - aic_maximal) > 1e-6  # les deux diffèrent bien


def test_best_model_refit_on_maximal_sample() -> None:
    """Spec 05 §3.4 : post-sélection, le meilleur modèle est ré-estimé sur
    l'échantillon maximal de SON ordre (pas l'échantillon commun)."""
    y, x = _dgp(seed=2)
    sel = ARDL.select_order(y, x, max_p=4, max_q=3, ic="bic")
    p_best, q_best = sel.best_order
    expected_nobs = len(y) - max(p_best, max(q_best.values()))
    assert sel.best_model.nobs == expected_nobs


def test_table_sorted_by_ic() -> None:
    """Spec 05 §3.3 : sortie = tableau trié par le critère choisi."""
    y, x = _dgp(seed=3)
    sel = ARDL.select_order(y, x, max_p=3, max_q=2, ic="bic")
    assert (sel.table["bic"].diff().dropna() >= 0).all()
    # le meilleur ordre est bien la première ligne
    first = sel.table.iloc[0]
    p_best, q_best = sel.best_order
    assert int(first["p"]) == p_best
    assert int(first["q_x"]) == q_best["x"]


def test_grid_is_complete() -> None:
    """Grille p ∈ 1..max_p, q ∈ 0..max_q : (max_p) × (max_q+1) candidats."""
    y, x = _dgp(seed=4)
    sel = ARDL.select_order(y, x, max_p=3, max_q=2, ic="aic")
    assert len(sel.table) == 3 * 3


def test_per_variable_search_matches_grid_on_k1() -> None:
    """Avec k=1, la recherche per_variable doit trouver le même optimum
    que la grille complète (l'espace est identique)."""
    y, x = _dgp(seed=5)
    sel_grid = ARDL.select_order(y, x, max_p=4, max_q=3, ic="bic", search="grid")
    sel_pv = ARDL.select_order(y, x, max_p=4, max_q=3, ic="bic", search="per_variable")
    assert sel_grid.best_order == sel_pv.best_order


@pytest.mark.fast_mc
def test_bic_consistency_fast() -> None:
    """Spec 05 §6.3 (version CI) : BIC retrouve l'ordre vrai (2, 1) sur
    DGP connu, T=300 — majorité des 50 réplications."""
    hits = 0
    n_rep = 50
    for rep in range(n_rep):
        y, x = _dgp(seed=10_000 + rep)
        sel = ARDL.select_order(y, x, max_p=4, max_q=3, ic="bic")
        p_best, q_best = sel.best_order
        if p_best == 2 and q_best["x"] == 1:
            hits += 1
    assert hits / n_rep >= 0.7


@pytest.mark.slow
def test_bic_consistency_full() -> None:
    """Spec 05 §6.3 (version complète) : 500 réplications MC, T=300."""
    hits = 0
    n_rep = 500
    for rep in range(n_rep):
        y, x = _dgp(seed=20_000 + rep)
        sel = ARDL.select_order(y, x, max_p=4, max_q=3, ic="bic")
        p_best, q_best = sel.best_order
        if p_best == 2 and q_best["x"] == 1:
            hits += 1
    assert hits / n_rep >= 0.8
