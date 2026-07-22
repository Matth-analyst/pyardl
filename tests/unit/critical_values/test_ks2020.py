"""Spec 13 §3 (voie A1) — surfaces de réponse K&S via statsmodels :
validation contre les valeurs publiées, cohérences internes (bornes PSS
de la spec 12, monotonies, OBS-4), aller-retour p-value/CV, couverture.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.critical_values import get_bounds, pvalue_bounds
from pyardl.critical_values.ks2020 import MAX_K_KS, crit_value_bounds

# ---------------------------------------------------------------------------
# Valeurs PUBLIÉES : Kripfganz & Schneider, "Response Surface Regressions
# for Critical Value Bounds and Approximate p-values in Equilibrium
# Correction Models", University of Exeter Discussion Paper 1901 (version
# ouverte de l'article OBES 2020), Appendix D — intercepts theta_{0,0}
# des surfaces = CV asymptotiques. Transcrites le 2026-07-10 depuis
# https://exetereconomics.github.io/RePEc/dpapers/DP1901.pdf.
#
# Tolérances ±0.03 (10 %/5 %) et ±0.06 (1 %) — PAS le 1e-3 de la spec
# §3.1 : la voie A1 utilise la RE-SIMULATION INDÉPENDANTE de statsmodels
# (32M réplications), pas les coefficients publiés. Deux effets : (a)
# deux estimateurs de haute précision du même quantile diffèrent de
# ~0.01-0.02 (designs de simulation différents) ; (b) les theta_{0,0}
# publiés sont l'intercept T->inf d'une surface extrapolée, alors que
# statsmodels simule directement — le léger biais fini-T résiduel est
# visible dans la queue à 1 % (~0.04 observé). Le 1e-3 s'appliquera à
# la voie A2 (mêmes coefficients). Déviation documentée dans
# docs/DEVIATIONS.md.
# ---------------------------------------------------------------------------
_PUBLISHED_THETA00 = {
    # (case, k, alpha, i1): valeur — Table 12, case (iii)
    (3, 1, 0.01, False): 6.8187,
    (3, 1, 0.05, False): 4.9055,
    (3, 1, 0.10, False): 4.0346,
    (3, 1, 0.01, True): 7.7358,
    (3, 1, 0.05, True): 5.7040,
    (3, 1, 0.10, True): 4.7675,
    (3, 2, 0.05, False): 3.7841,
    (3, 5, 0.05, False): 2.6202,
    (3, 10, 0.10, False): 1.8152,
}


class TestPublishedValues:
    """Spec 13 §3.1 : concordance avec les valeurs publiées (WP Exeter
    1901, tables de l'annexe D)."""

    @pytest.mark.parametrize(("case", "k", "alpha", "i1"), sorted(_PUBLISHED_THETA00))
    def test_asymptotic_cv_matches_published_theta00(
        self, case: int, k: int, alpha: float, i1: bool
    ) -> None:
        lo, up = crit_value_bounds(case=case, k=k, alpha=alpha)
        got = up if i1 else lo
        expected = _PUBLISHED_THETA00[(case, k, alpha, i1)]
        tol = 0.06 if alpha == 0.01 else 0.03  # cf. en-tête du module
        assert got == pytest.approx(expected, abs=tol), (
            f"case={case} k={k} alpha={alpha} I1={i1}"
        )


class TestInternalCoherence:
    """Spec 13 §3.2 + vigilance n°3 de l'arbitrage."""

    def test_asymptotic_cv_close_to_pss_tables(self) -> None:
        """CV(T→inf) -> bornes PSS validées en spec 12 (l'écart K&S-PSS
        est l'erreur MC des tables publiées, déjà quantifiée : ±0.15)."""
        for case in (1, 2, 3, 4, 5):
            for k in range(1, MAX_K_KS + 1):
                for alpha in (0.10, 0.05, 0.01):
                    ks_lo, ks_up = crit_value_bounds(case=case, k=k, alpha=alpha)
                    pss_lo, pss_up = get_bounds(
                        "F", case=case, k=k, alpha=alpha, cv_source="pss"
                    )
                    assert ks_lo == pytest.approx(pss_lo, abs=0.15)
                    assert ks_up == pytest.approx(pss_up, abs=0.15)

    def test_monotonic_in_k(self) -> None:
        for case in (1, 2, 3, 4, 5):
            for alpha in (0.10, 0.05, 0.025, 0.01):
                prev = None
                for k in range(1, MAX_K_KS + 1):
                    cell = crit_value_bounds(case=case, k=k, alpha=alpha)
                    if prev is not None:
                        assert cell[0] <= prev[0] + 0.02
                        assert cell[1] <= prev[1] + 0.02
                    prev = cell

    def test_monotonic_in_alpha(self) -> None:
        """Seuil plus strict -> CV plus grand (y compris au 2.5 %
        interpolé par inversion)."""
        for case in (1, 3, 5):
            for k in (1, 4, 8):
                values = [
                    crit_value_bounds(case=case, k=k, alpha=a)
                    for a in (0.10, 0.05, 0.025, 0.01)
                ]
                for prev, nxt in zip(values[:-1], values[1:], strict=True):
                    assert nxt[0] > prev[0]
                    assert nxt[1] > prev[1]

    def test_i0_leq_i1(self) -> None:
        for case in (1, 2, 3, 4, 5):
            for k in range(1, MAX_K_KS + 1):
                lo, up = crit_value_bounds(case=case, k=k, alpha=0.05)
                assert lo <= up + 1e-9

    def test_obs4_resolved_by_surface(self) -> None:
        """Vigilance n°3 : première confirmation d'usage du registre —
        la surface K&S redonne ~3.31 pour la cellule OBS-4 (cas II, k=2,
        10 % I(1)), là où PSS 2001 publie 3.35
        (docs/VALIDATION_OBSERVATIONS.md)."""
        _, up = crit_value_bounds(case=2, k=2, alpha=0.10)
        assert up == pytest.approx(3.31, abs=0.01)
        # et notre simulation interne (spec 12) concordait déjà : 3.299
        assert abs(up - 3.299) < 0.02

    def test_p025_consistent_with_spec12_internal_table(self) -> None:
        """Le seuil 2.5 % par inversion K&S recoupe la table interne de
        la spec 12 (simulation indépendante) à ±0.05."""
        from pyardl.critical_values.pss2001_p025 import F_P025

        for case in (1, 2, 3, 4, 5):
            for k in (1, 3, 7, 10):
                ks = crit_value_bounds(case=case, k=k, alpha=0.025)
                internal = F_P025[case][k]
                assert ks[0] == pytest.approx(internal[0], abs=0.05)
                assert ks[1] == pytest.approx(internal[1], abs=0.05)


class TestPvalues:
    def test_pvalue_at_cv_is_alpha_roundtrip(self) -> None:
        """Aller-retour CV -> p-value : p(CV(alpha)) = alpha (1e-8 aux
        seuils inversés, ~1e-3 aux percentiles ponctuels)."""
        for case in (2, 3, 5):
            for k in (1, 4):
                for alpha in (0.07, 0.025, 0.033):
                    lo, up = crit_value_bounds(case=case, k=k, alpha=alpha)
                    p_lo, p_up = pvalue_bounds(lo, case=case, k=k)
                    assert p_lo == pytest.approx(alpha, abs=1e-8)
                    p_lo2, p_up2 = pvalue_bounds(up, case=case, k=k)
                    assert p_up2 == pytest.approx(alpha, abs=1e-8)

    def test_pvalue_ordering_and_monotonicity(self) -> None:
        """p_I0 <= p_I1, et p décroît quand la stat croît."""
        for stat in (2.0, 4.0, 6.0, 9.0):
            p_i0, p_i1 = pvalue_bounds(stat, case=3, k=2)
            assert 0.0 <= p_i0 <= p_i1 <= 1.0
        stats = np.linspace(0.5, 15, 30)
        ps = [pvalue_bounds(float(s), case=3, k=2)[1] for s in stats]
        assert (np.diff(ps) <= 1e-12).all()

    def test_published_pss_bounds_give_expected_pvalues(self) -> None:
        """Les bornes 5 % publiées de PSS (4.94/5.73, cas III k=1)
        doivent avoir des p-values proches de 5 % de leur côté."""
        p_i0, _ = pvalue_bounds(4.94, case=3, k=1)
        _, p_i1 = pvalue_bounds(5.73, case=3, k=1)
        assert p_i0 == pytest.approx(0.05, abs=0.005)
        assert p_i1 == pytest.approx(0.05, abs=0.005)


class TestCoverageExceptions:
    """couverture manquante = exception explicite."""

    def test_t_stat_raises(self) -> None:
        with pytest.raises(ValueError, match="ne couvre pas le t_BDM"):
            get_bounds("t", case=3, k=1, alpha=0.05, cv_source="kripfganz")

    def test_k0_raises(self) -> None:
        with pytest.raises(ValueError, match="k=0"):
            crit_value_bounds(case=3, k=0, alpha=0.05)

    def test_k_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="hors couverture"):
            crit_value_bounds(case=3, k=11, alpha=0.05)

    def test_alpha_out_of_domain_raises(self) -> None:
        with pytest.raises(ValueError, match="domaine"):
            crit_value_bounds(case=3, k=1, alpha=0.5)

    def test_bad_case_raises(self) -> None:
        with pytest.raises(ValueError, match="case"):
            crit_value_bounds(case=0, k=1, alpha=0.05)
