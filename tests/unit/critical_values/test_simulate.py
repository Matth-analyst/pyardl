"""Spec 12 §3 — tests du moteur simulate_bounds.

La version fast_mc recoupe un échantillon de cellules PSS (dont des
cellules SANS seconde source : F k=0, t I(1) k>=1 — dette QUESTIONS.md
spec 10 §4) à n_sims modéré ; le recoupement intégral à n_sims=100k est
dans validation/spec12_montecarlo.py (nightly) et sa version pytest
marquée slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.critical_values.pss2001 import get_bounds
from pyardl.critical_values.simulate import simulate_bounds


class TestEngineBasics:
    def test_parameters_are_logged(self) -> None:
        """seed et paramètres journalisés."""
        sb = simulate_bounds(case=3, k=1, t_obs=100, n_sims=100, seed=123)
        assert sb.seed == 123
        assert sb.n_sims == 100
        assert sb.t_obs == 100
        assert sb.case == 3
        assert sb.k == 1
        assert sb.i1 is True
        assert sb.alphas == (0.10, 0.05, 0.025, 0.01)

    def test_reproducible_with_same_seed(self) -> None:
        a = simulate_bounds(case=3, k=1, t_obs=100, n_sims=500, seed=7)
        b = simulate_bounds(case=3, k=1, t_obs=100, n_sims=500, seed=7)
        np.testing.assert_array_equal(a.f_stats, b.f_stats)
        assert a.f_quantiles == b.f_quantiles

    def test_different_seed_differs(self) -> None:
        a = simulate_bounds(case=3, k=1, t_obs=100, n_sims=500, seed=7)
        b = simulate_bounds(case=3, k=1, t_obs=100, n_sims=500, seed=8)
        assert not np.array_equal(a.f_stats, b.f_stats)

    def test_chunk_is_logged_and_reproducible(self) -> None:
        """Le flux aléatoire est tiré par lots : chunk fait partie des
        paramètres de reproductibilité et est journalisé (règle 2).
        Même (seed, chunk) -> statistiques identiques ; les quantiles de
        découpages différents restent statistiquement compatibles."""
        a = simulate_bounds(case=2, k=2, t_obs=80, n_sims=2000, seed=5, chunk=500)
        b = simulate_bounds(case=2, k=2, t_obs=80, n_sims=2000, seed=5, chunk=500)
        c = simulate_bounds(case=2, k=2, t_obs=80, n_sims=2000, seed=5, chunk=64)
        assert a.chunk == 500 and c.chunk == 64
        np.testing.assert_array_equal(a.f_stats, b.f_stats)
        assert a.f_cv(0.05) == pytest.approx(c.f_cv(0.05), abs=0.5)

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="case"):
            simulate_bounds(case=0, k=1)
        with pytest.raises(ValueError, match="k must be >= 0"):
            simulate_bounds(case=3, k=-1)
        with pytest.raises(ValueError, match="t_obs"):
            simulate_bounds(case=3, k=1, t_obs=10)

    def test_k0_bounds_coincide(self) -> None:
        """k = 0 : pas de x, bornes I(0) et I(1) = même distribution
        (même seed -> mêmes tirages de y)."""
        up = simulate_bounds(case=3, k=0, t_obs=200, n_sims=400, seed=3, i1=True)
        lo = simulate_bounds(case=3, k=0, t_obs=200, n_sims=400, seed=3, i1=False)
        np.testing.assert_allclose(up.f_stats, lo.f_stats, atol=1e-10)


@pytest.mark.fast_mc
class TestCrossCheckSampleCells:
    """Recoupement d'un échantillon de cellules PSS 2001 (n_sims=20k,
    T=1000, tolérance ±0.12 — erreur MC des deux côtés)."""

    TOL_F = 0.12
    TOL_T = 0.06

    @pytest.mark.parametrize(
        ("case", "k", "alpha"), [(3, 1, 0.05), (2, 3, 0.05), (5, 2, 0.10)]
    )
    def test_f_cells_with_second_source(self, case: int, k: int, alpha: float) -> None:
        lower, upper = get_bounds("F", case=case, k=k, alpha=alpha)
        lo = simulate_bounds(case=case, k=k, n_sims=20_000, seed=100, i1=False)
        up = simulate_bounds(case=case, k=k, n_sims=20_000, seed=101, i1=True)
        assert lo.f_cv(alpha) == pytest.approx(lower, abs=self.TOL_F)
        assert up.f_cv(alpha) == pytest.approx(upper, abs=self.TOL_F)

    @pytest.mark.parametrize(("case", "k"), [(1, 0), (3, 0), (4, 0)])
    def test_f_k0_cells_debt(self, case: int, k: int) -> None:
        """Cellules k=0 du F : SANS seconde source avant cette spec
        (statsmodels/K&S commence à k=1) — dette QUESTIONS.md."""
        lower, upper = get_bounds("F", case=case, k=0, alpha=0.05)
        assert lower == upper  # k=0 : bornes confondues
        sim = simulate_bounds(case=case, k=0, n_sims=20_000, seed=102)
        assert sim.f_cv(0.05) == pytest.approx(upper, abs=self.TOL_F)

    @pytest.mark.parametrize(("case", "k"), [(3, 1), (5, 2), (1, 3)])
    def test_t_i1_cells_debt(self, case: int, k: int) -> None:
        """Colonnes I(1) du t (k>=1) : SANS seconde source avant cette
        spec — c'est le cœur de la dette QUESTIONS.md spec 10 §4."""
        _, upper = get_bounds("t", case=case, k=k, alpha=0.05)
        sim = simulate_bounds(case=case, k=k, n_sims=20_000, seed=103, i1=True)
        assert sim.t_cv(0.05) == pytest.approx(upper, abs=self.TOL_T)


def _cell_tolerance(stats: np.ndarray, p: float, n_sims: int) -> float:
    """Tolérance par cellule = 3 x SE combinée (PSS 40k + simulation) —
    critère dérivé de l'erreur MC des quantiles (PROVENANCE.md, note de
    révision spec 12 §3.1 ; dérivation : validation/spec12_mc_error.py)."""
    h = 0.005
    q_hi = np.quantile(stats, min(p + h, 1.0))
    q_lo = np.quantile(stats, max(p - h, 0.0))
    density = 2 * h / (q_hi - q_lo)
    se_pub = np.sqrt(p * (1 - p) / 40_000) / density
    se_sim = np.sqrt(p * (1 - p) / n_sims) / density
    return float(3.0 * np.hypot(se_pub, se_sim))


@pytest.mark.slow
def test_full_pss_crosscheck_slow() -> None:
    """Spec 12 §3.1 (nightly, critère révisé — note de révision de la
    spec) : recoupement de TOUTES les cellules F et t des tables PSS
    encodées, n_sims=100k, T=1000, tolérance par cellule = 3 x SE
    combinée. À 3σ sur 528 cellules, 0-3 dépassements fortuits sont
    attendus -> le test tolère jusqu'à 3 dépassements NON persistants.

    Version script (journal + CSV) : validation/spec12_montecarlo.py.
    """
    from pyardl.critical_values.pss2001 import LEVELS, MAX_K, T_BOUNDS

    n_sims = 100_000
    failures: list[str] = []
    for case in (1, 2, 3, 4, 5):
        for k in range(MAX_K + 1):
            lo = simulate_bounds(
                case=case, k=k, n_sims=n_sims, seed=10_000 + case * 100 + k, i1=False
            )
            up = simulate_bounds(
                case=case, k=k, n_sims=n_sims, seed=20_000 + case * 100 + k, i1=True
            )
            for alpha in LEVELS:
                f_lo, f_up = get_bounds("F", case=case, k=k, alpha=alpha)
                for sb, sim, enc, tag in (
                    (lo, lo.f_cv(alpha), f_lo, "I0"),
                    (up, up.f_cv(alpha), f_up, "I1"),
                ):
                    tol = _cell_tolerance(sb.f_stats, 1 - alpha, n_sims)
                    if abs(sim - enc) > tol:
                        failures.append(f"F {tag} c{case} k{k} a{alpha}")
                if case in T_BOUNDS:
                    t_lo, t_up = get_bounds("t", case=case, k=k, alpha=alpha)
                    for sb, sim, enc, tag in (
                        (lo, lo.t_cv(alpha), t_lo, "I0"),
                        (up, up.t_cv(alpha), t_up, "I1"),
                    ):
                        tol = _cell_tolerance(sb.t_stats, alpha, n_sims)
                        if abs(sim - enc) > tol:
                            failures.append(f"t {tag} c{case} k{k} a{alpha}")
    assert len(failures) <= 3, (
        f"{len(failures)} cellules hors critère 3σ (max 3 fortuits "
        f"attendus) : {failures[:20]}"
    )


@pytest.mark.needs_review
@pytest.mark.slow
def test_obs2_case1_k0_1pct_documented_state() -> None:
    """OBS-2 (docs/VALIDATION_OBSERVATIONS.md) : la cellule PSS cas I,
    k=0, 1 % (7.17 publié) porte le plus grand écart absolu du
    recoupement (~-0.22, convergé inter-seeds à ±0.02) — requalifié à
    ~2σ par le critère dérivé (queue épaisse -> SE publiée ~0.11), donc
    dans l'erreur MC attendue. Ce test verrouille l'état documenté :
    il ÉCHOUERA si l'écart disparaît (clore OBS-2) ou s'aggrave
    (ré-ouvrir l'enquête) ; needs_review jusqu'à vérification contre
    l'article original."""
    sim = simulate_bounds(case=1, k=0, n_sims=300_000, seed=111, i1=True)
    _, enc = get_bounds("F", case=1, k=0, alpha=0.01)
    gap = sim.f_cv(0.01) - enc
    assert -0.30 < gap < -0.15, f"écart {gap:.3f} hors de l'état documenté"
