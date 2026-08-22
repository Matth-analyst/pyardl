"""Spec 18 - VERROU : l'estimation atteint l'optimum, pas ses environs.

La regression quantile n'a pas de forme fermee a laquelle se comparer,
mais elle a quelque chose de mieux : un optimum EXACT, celui du
programme lineaire equivalent. Toute estimation peut donc etre notee sur
la seule grandeur qui compte, la perte quantile.

Ce verrou existe parce que le solveur par defaut echoue a ce test. Avec
sa tolerance d'origine, statsmodels manque l'optimum de 3.4e-03 en perte
et de 2.6e-02 en coefficients, sans rien signaler.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.qardl.estimate import (
    P_TOL,
    check_loss,
    quantile_regression,
    quantile_regression_lp,
)

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
#: Ecart de perte tolere entre notre estimation et l'optimum exact.
LOSS_TOL = 1e-5


def _sample(seed: int, n: int = 200, k: int = 6):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    y = x @ rng.normal(size=k) + rng.normal(size=n)
    return y, x


class TestReachesTheOptimum:
    """Le verrou proprement dit."""

    @pytest.mark.parametrize("tau", TAUS)
    def test_loss_matches_the_linear_program(self, tau: float) -> None:
        y, x = _sample(seed=0)
        ours, _ = quantile_regression(y, x, tau)
        exact = quantile_regression_lp(y, x, tau)
        excess = check_loss(y, x, ours, tau) - check_loss(y, x, exact, tau)
        assert excess >= -1e-9, "aucune estimation ne peut battre l'optimum"
        assert excess < LOSS_TOL

    @pytest.mark.parametrize(("n", "k"), [(150, 4), (200, 8), (400, 10)])
    def test_across_sample_shapes(self, n: int, k: int) -> None:
        y, x = _sample(seed=1, n=n, k=k)
        for tau in (0.1, 0.5, 0.9):
            ours, _ = quantile_regression(y, x, tau)
            exact = quantile_regression_lp(y, x, tau)
            excess = check_loss(y, x, ours, tau) - check_loss(y, x, exact, tau)
            assert excess < LOSS_TOL, f"tau={tau}, n={n}, k={k}"

    def test_the_default_tolerance_is_materially_worse(self) -> None:
        """La raison d'etre du reglage, gardee sous test.

        La comparaison porte sur la PERTE, pas sur les coefficients :
        pres d'un optimum plat, deux jeux de coefficients eloignes
        peuvent valoir la meme chose, et c'est la perte qui dit lequel
        est le meilleur.

        Si une version future de statsmodels resserre sa tolerance par
        defaut, ce test le signalera — et le reglage explicite pourra
        etre reconsidere plutot que conserve par habitude.
        """
        configurations = ((0, 200, 6), (2, 300, 6), (3, 200, 6))
        worst_default = 0.0
        worst_ours = 0.0
        for seed, n, k in configurations:
            y, x = _sample(seed=seed, n=n, k=k)
            for tau in TAUS:
                exact = check_loss(y, x, quantile_regression_lp(y, x, tau), tau)
                loose, _ = quantile_regression(y, x, tau, p_tol=1e-6)
                ours, _ = quantile_regression(y, x, tau)
                worst_default = max(worst_default, check_loss(y, x, loose, tau) - exact)
                worst_ours = max(worst_ours, check_loss(y, x, ours, tau) - exact)
        assert worst_ours < LOSS_TOL
        assert worst_default > 100 * worst_ours, (
            "la tolerance par defaut atteint desormais l'optimum d'aussi "
            "pres que la notre : le reglage explicite merite d'etre revu"
        )

    def test_our_tolerance_is_tighter_than_the_solver_default(self) -> None:
        assert P_TOL < 1e-6


class TestCheckLoss:
    """La fonction objectif, verifiee sur des cas calculables a la main."""

    def test_median_of_three(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        x = np.ones((3, 1))
        # A la mediane, la perte vaut la somme des ecarts absolus / 2.
        assert check_loss(y, x, np.array([2.0]), 0.5) == pytest.approx(1.0)

    def test_asymmetric_weighting(self) -> None:
        """Un residu positif pese tau, un negatif pese 1 - tau."""
        y = np.array([1.0, -1.0])
        x = np.ones((2, 1))
        assert check_loss(y, x, np.array([0.0]), 0.9) == pytest.approx(0.9 + 0.1)

    def test_minimised_at_the_empirical_quantile(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.normal(size=501)
        x = np.ones((501, 1))
        for tau in (0.25, 0.5, 0.75):
            best = float(np.quantile(y, tau))
            here = check_loss(y, x, np.array([best]), tau)
            for shift in (-0.1, -0.01, 0.01, 0.1):
                assert check_loss(y, x, np.array([best + shift]), tau) >= here - 1e-9


class TestValidation:
    @pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.5])
    def test_tau_outside_the_open_interval(self, tau: float) -> None:
        y, x = _sample(seed=4, n=50, k=3)
        with pytest.raises(ValueError, match="strictly in"):
            quantile_regression(y, x, tau)
        with pytest.raises(ValueError, match="strictly in"):
            quantile_regression_lp(y, x, tau)

    def test_one_dimensional_design(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            quantile_regression(np.zeros(10), np.zeros(10), 0.5)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="observations"):
            quantile_regression(np.zeros(10), np.ones((9, 2)), 0.5)

    def test_more_parameters_than_observations(self) -> None:
        with pytest.raises(ValueError, match="not identified"):
            quantile_regression(np.zeros(3), np.ones((3, 4)), 0.5)


class TestIterationCap:
    """Quand le solveur s'arrete sur le compteur et non sur la tolerance."""

    def test_hitting_the_cap_warns(self) -> None:
        """Un arret sur le nombre d'iterations ne prouve rien sur
        l'optimalite. Le message de la dependance parle de ses internes ;
        le notre dit ce que l'utilisateur doit en faire."""
        from pyardl.exceptions import PyardlMethodologyWarning

        y, x = _sample(seed=7, n=200, k=6)
        with pytest.warns(PyardlMethodologyWarning, match="iteration cap"):
            quantile_regression(y, x, 0.5, max_iter=1)

    def test_a_normal_fit_does_not_warn(self) -> None:
        import warnings as _w

        from pyardl.exceptions import PyardlMethodologyWarning

        y, x = _sample(seed=8, n=200, k=6)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            quantile_regression(y, x, 0.5)
        assert not [
            w for w in caught if issubclass(w.category, PyardlMethodologyWarning)
        ]


class TestCovariance:
    def test_shape_and_symmetry(self) -> None:
        y, x = _sample(seed=5)
        _, cov = quantile_regression(y, x, 0.5)
        assert cov.shape == (x.shape[1], x.shape[1])
        assert np.allclose(cov, cov.T)
        assert np.all(np.diag(cov) > 0)

    def test_wider_in_the_tails(self) -> None:
        """L'information se rarefie dans les queues : l'incertitude y est
        plus grande qu'a la mediane."""
        y, x = _sample(seed=6, n=400)
        _, mid = quantile_regression(y, x, 0.5)
        _, tail = quantile_regression(y, x, 0.05)
        assert np.diag(tail).sum() > np.diag(mid).sum()
