"""Spec 12 — seuil 2.5 % des tables PSS (simulation interne).

Marqué ``needs_review`` : ces valeurs proviennent du moteur interne
(PROVENANCE.md), non des tables publiées — à vérifier contre l'article
original quand il sera consultable. Le test d'encadrement (arbitrage
2026-07-07) verrouille leur cohérence : chaque borne 2.5 % est
STRICTEMENT comprise entre les bornes 5 % et 1 % publiées de la même
cellule.
"""

from __future__ import annotations

import pytest

from pyardl.critical_values.pss2001 import MAX_K, T_BOUNDS, get_bounds


@pytest.mark.needs_review
class TestP025Encadrement:
    def test_f_bracketed_by_published_5_and_1_pct(self) -> None:
        for case in (1, 2, 3, 4, 5):
            for k in range(MAX_K + 1):
                lo5, up5 = get_bounds("F", case=case, k=k, alpha=0.05)
                lo25, up25 = get_bounds("F", case=case, k=k, alpha=0.025)
                lo1, up1 = get_bounds("F", case=case, k=k, alpha=0.01)
                assert lo5 < lo25 < lo1, f"F I0 c{case} k{k}: {lo5}/{lo25}/{lo1}"
                assert up5 < up25 < up1, f"F I1 c{case} k{k}: {up5}/{up25}/{up1}"

    def test_t_bracketed_by_published_5_and_1_pct(self) -> None:
        """t (gauche) : le 2.5 % est plus négatif que le 5 %, moins que
        le 1 %."""
        for case in T_BOUNDS:
            for k in range(MAX_K + 1):
                lo5, up5 = get_bounds("t", case=case, k=k, alpha=0.05)
                lo25, up25 = get_bounds("t", case=case, k=k, alpha=0.025)
                lo1, up1 = get_bounds("t", case=case, k=k, alpha=0.01)
                assert lo1 < lo25 < lo5, f"t I0 c{case} k{k}"
                assert up1 < up25 < up5, f"t I1 c{case} k{k}"

    def test_i0_leq_i1(self) -> None:
        for case in (1, 2, 3, 4, 5):
            for k in range(1, MAX_K + 1):
                lo, up = get_bounds("F", case=case, k=k, alpha=0.025)
                assert lo <= up

    def test_known_cell_close_to_ks(self) -> None:
        """Cas III, k=1, 2.5 % : Kripfganz-Schneider ne publie pas ce
        percentile dans statsmodels, mais la cellule doit rester dans
        la fourchette plausible dérivée des valeurs publiées (5 % :
        4.94/5.73 ; 1 % : 6.84/7.84)."""
        lo, up = get_bounds("F", case=3, k=1, alpha=0.025)
        assert 4.94 < lo < 6.84
        assert 5.73 < up < 7.84
