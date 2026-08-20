"""Spec 16 §3.1 — le simulateur VECM.

Un simulateur ne se teste pas sur ce qu'il affiche mais sur ce que les
estimateurs y retrouvent : le rang injecté doit être retrouvé par le
test de Johansen (spec 07), et les dégénérescences injectées par la
classification à trois tests (spec 15). C'est le seul contrôle qui
attrape une erreur de construction du DGP.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyardl.bounds import bounds_test
from pyardl.cointegration import johansen
from pyardl.simulate import VECMSimulation, degenerate_system, vecm_ardl


class TestContract:
    """Ce qui revient est ce que les paramètres décrivent."""

    def test_shape_and_names(self) -> None:
        sim = vecm_ardl(150, alpha=[[-0.4], [0.0]], beta=[[1.0], [-1.0]], seed=0)
        assert isinstance(sim, VECMSimulation)
        assert sim.data.shape == (150, 2)
        assert list(sim.data.columns) == ["y", "x1"]
        assert sim.rank == 1

    def test_pi_is_alpha_beta_transpose(self) -> None:
        alpha = np.array([[-0.5], [0.1]])
        beta = np.array([[1.0], [-2.0]])
        sim = vecm_ardl(100, alpha=alpha, beta=beta, seed=1)
        assert sim.pi == pytest.approx(alpha @ beta.T)
        assert sim.lam == pytest.approx(-0.5)
        assert sim.gamma_true == pytest.approx([1.0])

    def test_seed_is_recorded_and_reproducible(self) -> None:
        a, b = degenerate_system(None, k=1)
        first = vecm_ardl(120, alpha=a, beta=b, seed=7)
        again = vecm_ardl(120, alpha=a, beta=b, seed=7)
        assert first.seed == 7
        assert np.array_equal(first.data.to_numpy(), again.data.to_numpy())

    def test_seed_is_recorded_when_omitted(self) -> None:
        """Un échantillon simulé que personne ne peut régénérer n'est
        pas une preuve : la seed tirée de l'entropie est journalisée."""
        a, b = degenerate_system(None, k=1)
        sim = vecm_ardl(50, alpha=a, beta=b)
        assert isinstance(sim.seed, int)
        repeat = vecm_ardl(50, alpha=a, beta=b, seed=sim.seed)
        assert np.array_equal(sim.data.to_numpy(), repeat.data.to_numpy())

    def test_different_seeds_give_different_data(self) -> None:
        a, b = degenerate_system(None, k=1)
        one = vecm_ardl(100, alpha=a, beta=b, seed=1)
        two = vecm_ardl(100, alpha=a, beta=b, seed=2)
        assert not np.array_equal(one.data.to_numpy(), two.data.to_numpy())

    def test_gammas_are_used(self) -> None:
        a, b = degenerate_system(None, k=1)
        plain = vecm_ardl(100, alpha=a, beta=b, seed=3)
        with_g = vecm_ardl(
            100, alpha=a, beta=b, gammas=[np.array([[0.5, 0.0], [0.0, 0.3]])], seed=3
        )
        assert not np.allclose(plain.data.to_numpy(), with_g.data.to_numpy())
        assert len(with_g.gammas) == 1

    def test_custom_names(self) -> None:
        a, b = degenerate_system(None, k=2)
        sim = vecm_ardl(60, alpha=a, beta=b, seed=4, names=["cons", "inc", "rate"])
        assert list(sim.data.columns) == ["cons", "inc", "rate"]
        assert list(sim.x.columns) == ["inc", "rate"]
        assert sim.y.name == "cons"


class TestDeterministics:
    """Les cas déterministes portent effectivement leur terme."""

    def test_constant_shifts_the_drift(self) -> None:
        a, b = degenerate_system(None, k=1)
        flat = vecm_ardl(200, alpha=a, beta=b, case=3, seed=11)
        drift = vecm_ardl(200, alpha=a, beta=b, case=3, const=[0.0, 0.5], seed=11)
        # Le regresseur derive : sa moyenne s'eloigne franchement.
        assert abs(drift.x.iloc[:, 0].mean()) > abs(flat.x.iloc[:, 0].mean())

    def test_trend_is_used_under_case_5(self) -> None:
        a, b = degenerate_system(None, k=1)
        flat = vecm_ardl(200, alpha=a, beta=b, case=5, seed=12)
        with_trend = vecm_ardl(200, alpha=a, beta=b, case=5, trend=[0.0, 0.05], seed=12)
        assert not np.allclose(flat.data.to_numpy(), with_trend.data.to_numpy())


class TestValidation:
    """Refus explicites — rien n'est redimensionné en silence."""

    def test_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="both must be"):
            vecm_ardl(100, alpha=[[-0.4], [0.0]], beta=[[1.0], [-1.0], [0.5]])

    def test_single_variable_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two variables"):
            vecm_ardl(100, alpha=[[-0.4]], beta=[[1.0]])

    def test_bad_case_is_refused(self) -> None:
        a, b = degenerate_system(None, k=1)
        with pytest.raises(ValueError, match="case must be"):
            vecm_ardl(100, alpha=a, beta=b, case=6)

    def test_deterministic_term_the_case_lacks_is_refused(self) -> None:
        """Passer une tendance à un cas qui n'en porte pas est une
        erreur de spécification, pas un détail à absorber."""
        a, b = degenerate_system(None, k=1)
        with pytest.raises(ValueError, match="carries no trend"):
            vecm_ardl(100, alpha=a, beta=b, case=3, trend=[0.1, 0.0])
        with pytest.raises(ValueError, match="carries no constant"):
            vecm_ardl(100, alpha=a, beta=b, case=1, const=[0.1, 0.0])

    def test_wrong_sized_deterministic_is_refused(self) -> None:
        a, b = degenerate_system(None, k=1)
        with pytest.raises(ValueError, match="expected 2"):
            vecm_ardl(100, alpha=a, beta=b, case=3, const=[0.1, 0.0, 0.2])

    def test_bad_gamma_shape_is_refused(self) -> None:
        a, b = degenerate_system(None, k=1)
        with pytest.raises(ValueError, match="gammas\\[0\\]"):
            vecm_ardl(100, alpha=a, beta=b, gammas=[np.eye(3)])

    def test_bad_sigma_shape_is_refused(self) -> None:
        a, b = degenerate_system(None, k=1)
        with pytest.raises(ValueError, match="sigma has shape"):
            vecm_ardl(100, alpha=a, beta=b, sigma=np.eye(3))

    def test_bad_names_length_is_refused(self) -> None:
        a, b = degenerate_system(None, k=1)
        with pytest.raises(ValueError, match="names has"):
            vecm_ardl(100, alpha=a, beta=b, names=["only_one"])

    def test_short_sample_and_negative_burn_in_are_refused(self) -> None:
        a, b = degenerate_system(None, k=1)
        with pytest.raises(ValueError, match="n_obs"):
            vecm_ardl(1, alpha=a, beta=b)
        with pytest.raises(ValueError, match="burn_in"):
            vecm_ardl(100, alpha=a, beta=b, burn_in=-1)


class TestCanonicalSystems:
    def test_degenerate_system_shapes(self) -> None:
        for kind in (None, 1, 2):
            a, b = degenerate_system(kind, k=2)
            assert a.shape == b.shape == (3, 1)

    def test_type_1_has_no_gamma(self) -> None:
        a, b = degenerate_system(1, k=2)
        pi = a @ b.T
        assert pi[0, 0] != 0
        assert pi[0, 1:] == pytest.approx([0.0, 0.0])

    def test_type_2_has_no_lambda(self) -> None:
        a, b = degenerate_system(2, k=2)
        pi = a @ b.T
        assert pi[0, 0] == pytest.approx(0.0)
        assert np.any(np.abs(pi[0, 1:]) > 0)

    def test_positive_speed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="speed must be negative"):
            degenerate_system(None, speed=0.3)

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            degenerate_system(3, k=1)  # type: ignore[arg-type]

    def test_k_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            degenerate_system(None, k=0)


class TestEstimatorsRecoverWhatWasInjected:
    """§3.1 — le contrôle qui compte."""

    @pytest.mark.parametrize("seed", range(3))
    def test_johansen_finds_the_injected_rank(self, seed: int) -> None:
        a, b = degenerate_system(None, k=2, speed=-0.5)
        sim = vecm_ardl(300, alpha=a, beta=b, seed=100 + seed)
        assert johansen(sim.data, det_order=0, k_ar_diff=1).selected_rank >= 1

    def test_a_zero_pi_is_reported_as_rank_zero(self) -> None:
        """Le rang annonce est celui de Pi, pas le nombre de colonnes
        fournies : un alpha nul ne cree aucune relation."""
        zero = np.zeros((3, 1))
        sim = vecm_ardl(300, alpha=zero, beta=[[1.0], [-1.0], [0.0]], seed=200)
        assert sim.rank == 0

    @pytest.mark.fast_mc
    def test_rank_zero_system_is_rarely_seen_as_cointegrated(self) -> None:
        """Le simulateur ne fabrique pas de relation : le taux de fausse
        detection reste celui du test de Johansen lui-meme (OBS-10),
        pas davantage."""
        zero = np.zeros((3, 1))
        hits = sum(
            johansen(
                vecm_ardl(300, alpha=zero, beta=zero, seed=200 + s).data,
                det_order=0,
                k_ar_diff=1,
            ).selected_rank
            > 0
            for s in range(100)
        )
        assert hits / 100 < 0.12

    @pytest.mark.parametrize("seed", range(3))
    def test_cointegration_is_classified_as_such(self, seed: int) -> None:
        a, b = degenerate_system(None, k=1, speed=-0.5)
        sim = vecm_ardl(300, alpha=a, beta=b, seed=300 + seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(sim.y, sim.x, case=3, order=(1, 1))
        assert res.classification()[0] == "cointegration"

    @pytest.mark.parametrize("seed", range(3))
    def test_type_1_is_never_called_cointegration(self, seed: int) -> None:
        a, b = degenerate_system(1, k=1, speed=-0.5)
        sim = vecm_ardl(300, alpha=a, beta=b, seed=400 + seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(sim.y, sim.x, case=3, order=(1, 1))
        assert res.classification()[0] != "cointegration"

    @pytest.mark.parametrize("seed", range(3))
    def test_type_2_is_never_called_cointegration(self, seed: int) -> None:
        a, b = degenerate_system(2, k=1, speed=-0.5)
        sim = vecm_ardl(300, alpha=a, beta=b, seed=500 + seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(sim.y, sim.x, case=3, order=(1, 1))
        assert res.classification()[0] != "cointegration"
