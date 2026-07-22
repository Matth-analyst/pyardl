"""Spec 12 §2.1/§3 — tables de Narayan 2005 : cohérences structurelles
(toutes les cellules), recoupement par le moteur interne (T=40, 60),
interpolation en T, exceptions de couverture.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.critical_values import get_bounds
from pyardl.critical_values.narayan2005 import F_NARAYAN, T_GRID
from pyardl.critical_values.pss2001 import get_bounds as pss_bounds
from pyardl.critical_values.simulate import simulate_bounds
from pyardl.exceptions import PyardlMethodologyWarning


class TestStructuralConsistency:
    """Vérifications sur TOUTES les cellules transcrites."""

    def test_i0_leq_i1(self) -> None:
        for cases in F_NARAYAN.values():
            for table in cases.values():
                assert (table[:, :, 0] <= table[:, :, 1] + 1e-12).all()

    def test_decreasing_in_k(self) -> None:
        """Tolérance 0.02 : les tables publiées de Narayan portent leur
        propre bruit MC (40k réplications) — une seule cellule non
        monotone dans toute la transcription : T=30, cas 2, 5 % I(1),
        k=6 (4.148) -> k=7 (4.163), +0.015, dans l'erreur MC de l'article.
        Valeur publiée conservée telle quelle (jamais de correction
        silencieuse)."""
        for cases in F_NARAYAN.values():
            for table in cases.values():
                assert (np.diff(table[1:], axis=0) <= 0.02).all()

    def test_stricter_level_larger(self) -> None:
        for cases in F_NARAYAN.values():
            for table in cases.values():
                assert (np.diff(table, axis=1) >= -1e-12).all()

    def test_decreasing_toward_asymptotic_in_t(self) -> None:
        """Les bornes petits échantillons décroissent (au bruit MC près)
        quand T croît, et restent au-dessus de l'asymptotique PSS."""
        for case in (2, 3, 5):
            for k in range(8):
                for lvl_idx, alpha in enumerate((0.10, 0.05, 0.01)):
                    values = [F_NARAYAN[t][case][k, lvl_idx, 1] for t in T_GRID]
                    # tendance décroissante (tolère le bruit MC de Narayan)
                    assert values[0] >= values[-1] - 0.05
                    _, pss_upper = pss_bounds("F", case, k, alpha)
                    assert values[-1] >= pss_upper - 0.15


@pytest.mark.fast_mc
class TestEngineCrossCheck:
    """Spec 12 §3.1 : le moteur reproduit des cellules publiées de
    Narayan (T=40, 60) — version CI à 20k sims, ±0.15."""

    @pytest.mark.parametrize(
        ("t_obs", "case", "k"), [(40, 3, 1), (40, 2, 2), (60, 5, 1), (60, 3, 3)]
    )
    def test_cells(self, t_obs: int, case: int, k: int) -> None:
        lo = simulate_bounds(
            case=case, k=k, t_obs=t_obs, n_sims=20_000, seed=400 + k, i1=False
        )
        up = simulate_bounds(
            case=case, k=k, t_obs=t_obs, n_sims=20_000, seed=500 + k, i1=True
        )
        for alpha in (0.10, 0.05):
            enc_lo, enc_up = get_bounds(
                "F", case=case, k=k, alpha=alpha, cv_source="narayan", t_obs=t_obs
            )
            assert lo.f_cv(alpha) == pytest.approx(enc_lo, abs=0.15)
            assert up.f_cv(alpha) == pytest.approx(enc_up, abs=0.15)


@pytest.mark.slow
@pytest.mark.parametrize(("t_obs", "case", "k"), [(40, 3, 1), (60, 2, 2), (60, 5, 3)])
def test_engine_crosscheck_full(t_obs: int, case: int, k: int) -> None:
    """Spec 12 §3.1 (nightly, critère révisé — même logique que pour les
    tables PSS, cf. note de révision de la spec) : cellules Narayan
    recoupées à n_sims=100k avec tolérance par cellule = 3 x SE combinée
    (Narayan 2005 : 40 000 réplications, comme PSS)."""
    from tests.unit.critical_values.test_simulate import _cell_tolerance

    n_sims = 100_000
    lo = simulate_bounds(
        case=case, k=k, t_obs=t_obs, n_sims=n_sims, seed=600 + k, i1=False
    )
    up = simulate_bounds(
        case=case, k=k, t_obs=t_obs, n_sims=n_sims, seed=700 + k, i1=True
    )
    for alpha in (0.10, 0.05, 0.01):
        enc_lo, enc_up = get_bounds(
            "F", case=case, k=k, alpha=alpha, cv_source="narayan", t_obs=t_obs
        )
        tol_lo = _cell_tolerance(lo.f_stats, 1 - alpha, n_sims)
        tol_up = _cell_tolerance(up.f_stats, 1 - alpha, n_sims)
        assert lo.f_cv(alpha) == pytest.approx(enc_lo, abs=tol_lo)
        assert up.f_cv(alpha) == pytest.approx(enc_up, abs=tol_up)


class TestInterpolationAndCoverage:
    def test_exact_grid_point(self) -> None:
        assert get_bounds(
            "F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=40
        ) == (5.26, 6.16)

    def test_interpolation_between_grid_points(self) -> None:
        """Spec 12 §3.2 : interpolation linéaire, monotonie en T."""
        lo40, up40 = get_bounds(
            "F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=40
        )
        lo42, up42 = get_bounds(
            "F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=42
        )
        lo45, up45 = get_bounds(
            "F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=45
        )
        assert min(up40, up45) <= up42 <= max(up40, up45)
        assert min(lo40, lo45) <= lo42 <= max(lo40, lo45)
        # poids 2/5 exact
        assert lo42 == pytest.approx(lo40 + (lo45 - lo40) * 2 / 5, abs=1e-12)

    def test_out_of_range_falls_back_to_pss_with_warning(self) -> None:
        """Spec 12 §3.2 : hors plage -> warning + asymptotique."""
        with pytest.warns(PyardlMethodologyWarning, match="hors de la plage"):
            got = get_bounds(
                "F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=200
            )
        assert got == pss_bounds("F", case=3, k=1, alpha=0.05)

    def test_uncovered_cases_raise(self) -> None:
        """Spec 12 §2.2 / §3.3 : cas I et IV -> exception documentée."""
        for case in (1, 4):
            with pytest.raises(ValueError, match="kripfganz"):
                get_bounds(
                    "F", case=case, k=1, alpha=0.05, cv_source="narayan", t_obs=40
                )

    def test_t_stat_raises(self) -> None:
        with pytest.raises(ValueError, match="bornes F"):
            get_bounds("t", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=40)

    def test_k_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="k <= 7"):
            get_bounds("F", case=3, k=8, alpha=0.05, cv_source="narayan", t_obs=40)

    def test_missing_t_obs_raises(self) -> None:
        with pytest.raises(ValueError, match="t_obs"):
            get_bounds("F", case=3, k=1, alpha=0.05, cv_source="narayan")

    def test_kripfganz_now_available(self) -> None:
        """Depuis la spec 13 (voie A1), kripfganz sert les CV F —
        proches de PSS à l'erreur MC des tables publiées près."""
        lo, up = get_bounds("F", case=3, k=1, alpha=0.05, cv_source="kripfganz")
        assert lo == pytest.approx(4.94, abs=0.15)
        assert up == pytest.approx(5.73, abs=0.15)
