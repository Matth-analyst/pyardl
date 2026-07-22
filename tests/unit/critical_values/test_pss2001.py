"""Recoupement des tables PSS 2001 encodées.

1. F (k=1..10) vs valeurs asymptotiques Kripfganz-Schneider 2020
   (statsmodels) — seconde source indépendante, tolérance ±0.15.
2. t, colonnes I(0) et ligne k=0 vs CV Dickey-Fuller asymptotiques
   MacKinnon (statsmodels) — tolérance ±0.03.
3. Colonnes I(1) du t (k>=1) : pas de seconde source accessible →
   monotonies structurelles seulement, recoupement simulation = dette
   spec 12 (docs/QUESTIONS.md), d'où le marquage needs_review.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.tsa.adfvalues import mackinnoncrit
from statsmodels.tsa.ardl import pss_critical_values as sm_pss

from pyardl.critical_values.pss2001 import (
    F_BOUNDS,
    LEVELS,
    MAX_K,
    T_BOUNDS,
    get_bounds,
)

_SM_PERCENTILE_IDX = {0.10: 0, 0.05: 1, 0.01: 2}  # (90, 95, 99, 99.9)


class TestCrossCheckKS:
    """F : PSS 2001 vs Kripfganz-Schneider asymptotique (statsmodels)."""

    @pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("alpha", LEVELS)
    def test_f_bounds_match_ks_within_tolerance(self, case: int, alpha: float) -> None:
        for k in range(1, MAX_K + 1):
            lower, upper = get_bounds("F", case=case, k=k, alpha=alpha)
            idx = _SM_PERCENTILE_IDX[alpha]
            ks_lower = sm_pss.crit_vals[(k, case, False)][idx]
            ks_upper = sm_pss.crit_vals[(k, case, True)][idx]
            assert lower == pytest.approx(ks_lower, abs=0.15), (
                f"F I(0) case={case} k={k} alpha={alpha}: PSS={lower} vs K&S={ks_lower}"
            )
            assert upper == pytest.approx(ks_upper, abs=0.15), (
                f"F I(1) case={case} k={k} alpha={alpha}: PSS={upper} vs K&S={ks_upper}"
            )


class TestCrossCheckDickeyFuller:
    """t : bornes I(0) = CV Dickey-Fuller asymptotiques (MacKinnon)."""

    @pytest.mark.parametrize(("case", "regression"), [(1, "n"), (3, "c"), (5, "ct")])
    @pytest.mark.parametrize("alpha", LEVELS)
    def test_t_i0_column_matches_df(
        self, case: int, regression: str, alpha: float
    ) -> None:
        df_cv = float(
            mackinnoncrit(N=1, regression=regression, nobs=np.inf)[
                {0.01: 0, 0.05: 1, 0.10: 2}[alpha]
            ]
        )
        for k in range(MAX_K + 1):
            lower, _ = get_bounds("t", case=case, k=k, alpha=alpha)
            assert lower == pytest.approx(df_cv, abs=0.03), (
                f"t I(0) case={case} k={k} alpha={alpha}: "
                f"PSS={lower} vs DF/MacKinnon={df_cv}"
            )

    @pytest.mark.parametrize(("case", "regression"), [(1, "n"), (3, "c"), (5, "ct")])
    def test_t_k0_upper_matches_df(self, case: int, regression: str) -> None:
        """k=0 : les deux bornes coïncident avec la distribution DF."""
        for alpha in LEVELS:
            lower, upper = get_bounds("t", case=case, k=0, alpha=alpha)
            assert upper == pytest.approx(lower, abs=0.011)


class TestStructuralMonotonicity:
    """Cohérences structurelles des tables encodées.

    (Marque ``needs_review`` levée le 2026-07-07 : les cellules sans
    seconde source — colonnes I(1) du t, lignes k=0 du F — sont
    désormais recoupées par le moteur de simulation interne, spec 12 :
    527/528 cellules dans le critère de 3 erreurs types combinées, cf.
    QUESTIONS.md et validation/results/spec12_pss_crosscheck.csv.)"""

    def test_f_upper_geq_lower_everywhere(self) -> None:
        for table in F_BOUNDS.values():
            assert (table[:, :, 1] >= table[:, :, 0]).all()

    def test_t_upper_leq_lower_everywhere(self) -> None:
        """Pour le t (gauche), la borne I(1) est plus négative."""
        for table in T_BOUNDS.values():
            assert (table[:, :, 1] <= table[:, :, 0] + 1e-12).all()

    def test_f_decreasing_in_k(self) -> None:
        """Les bornes F décroissent avec k (dilution des restrictions)."""
        for table in F_BOUNDS.values():
            assert (np.diff(table[1:, :, :], axis=0) <= 1e-12).all()

    def test_t_i1_decreasing_in_k(self) -> None:
        """La borne I(1) du t devient plus négative quand k augmente."""
        for table in T_BOUNDS.values():
            assert (np.diff(table[:, :, 1], axis=0) <= 1e-12).all()

    def test_stricter_level_more_extreme(self) -> None:
        for table in F_BOUNDS.values():
            assert (np.diff(table, axis=1) >= -1e-12).all()  # 10% < 5% < 1%
        for table in T_BOUNDS.values():
            assert (np.diff(table, axis=1) <= 1e-12).all()


class TestCoverageExceptions:
    """couverture manquante = exception explicite."""

    def test_t_case_ii_iv_raises(self) -> None:
        for case in (2, 4):
            with pytest.raises(ValueError, match="ne publie pas de bornes t"):
                get_bounds("t", case=case, k=1, alpha=0.05)

    def test_alpha_025_served_by_internal_simulation(self) -> None:
        """Depuis la spec 12, le seuil 2.5 % est servi (simulation
        interne, cf. test_p025.py) ; un alpha réellement inconnu lève
        toujours une exception explicite."""
        lo, up = get_bounds("F", case=3, k=1, alpha=0.025)
        assert 4.94 < lo < 6.84  # encadré par les 5 % et 1 % publiés
        with pytest.raises(ValueError, match="non couvert"):
            get_bounds("F", case=3, k=1, alpha=0.20)

    def test_k_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="hors des tables"):
            get_bounds("F", case=3, k=11, alpha=0.05)

    def test_bad_case_raises(self) -> None:
        with pytest.raises(ValueError, match="case"):
            get_bounds("F", case=6, k=1, alpha=0.05)

    def test_bad_stat_raises(self) -> None:
        with pytest.raises(ValueError, match="stat"):
            get_bounds("W", case=3, k=1, alpha=0.05)  # type: ignore[arg-type]


def test_known_reference_values() -> None:
    """Les valeurs les plus citées de la littérature (cas III, k=1)."""
    assert get_bounds("F", case=3, k=1, alpha=0.05) == (4.94, 5.73)
    assert get_bounds("F", case=3, k=1, alpha=0.10) == (4.04, 4.78)
    assert get_bounds("t", case=3, k=1, alpha=0.05) == (-2.86, -3.22)
