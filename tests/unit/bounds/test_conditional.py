"""Spec 16 §2.1 — modèle conditionnel contre modèle inconditionnel.

L'UECM conditionnel inclut les Δx contemporains (le cadre de PSS) ;
l'inconditionnel les exclut. La convention retenue n'a pas été déduite
du texte : elle reproduit à l'identique la statistique inconditionnelle
que bootCT rapporte lui-même (voir tests/replication/test_spec16.py).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.bootstrap import bootstrap_bounds_test
from pyardl.bounds import bounds_test
from pyardl.bounds.pss import _estimate_uecm
from pyardl.simulate import degenerate_system, vecm_ardl


def _sample(seed: int, n: int = 200) -> tuple[pd.Series, pd.DataFrame]:
    """Innovations CORRÉLÉES : sans corrélation contemporaine, les deux
    modèles ne peuvent pas différer de façon intéressante."""
    a, b = degenerate_system(None, k=1, speed=-0.5)
    sigma = np.array([[1.0, 0.6], [0.6, 1.0]])
    sim = vecm_ardl(n, alpha=a, beta=b, sigma=sigma, seed=seed)
    return sim.y, sim.x


class TestDesign:
    """§2.1 — ce que la spécification retire, et ce qu'elle garde."""

    def test_unconditional_drops_only_the_contemporaneous_differences(self) -> None:
        y, x = _sample(seed=0)
        y_arr = y.to_numpy()
        x_arr = x.to_numpy()
        cond = _estimate_uecm(y_arr, x_arr, ("x1",), "y", 2, (2,), 3)
        uncond = _estimate_uecm(
            y_arr, x_arr, ("x1",), "y", 2, (2,), 3, conditional=False
        )
        removed = set(cond.names) - set(uncond.names)
        assert removed == {"D.x1.L0"}
        # Tout le reste est identique, colonne pour colonne et dans le
        # même ordre : c'est la seule différence entre les deux modèles.
        assert uncond.names == [n for n in cond.names if n != "D.x1.L0"]

    def test_level_terms_are_untouched(self) -> None:
        """Les niveaux testés ne bougent pas : le vecteur teste reste le
        même, sinon les deux modèles ne testeraient pas la même chose."""
        y, x = _sample(seed=1)
        cond = _estimate_uecm(y.to_numpy(), x.to_numpy(), ("x1",), "y", 2, (2,), 3)
        uncond = _estimate_uecm(
            y.to_numpy(), x.to_numpy(), ("x1",), "y", 2, (2,), 3, conditional=False
        )
        assert cond.tested == uncond.tested
        assert cond.lam_name == uncond.lam_name

    def test_q_of_one_leaves_no_short_run_term(self) -> None:
        """Avec q_j = 1, le seul terme de court terme du régresseur est
        le contemporain : l'inconditionnel n'en garde aucun."""
        y, x = _sample(seed=2)
        uncond = _estimate_uecm(
            y.to_numpy(), x.to_numpy(), ("x1",), "y", 1, (1,), 3, conditional=False
        )
        assert not [n for n in uncond.names if n.startswith("D.x1")]


class TestBoundsTestIntegration:
    def test_default_is_conditional(self) -> None:
        y, x = _sample(seed=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            default = bounds_test(y, x, case=3, order=(2, 2))
            explicit = bounds_test(y, x, case=3, order=(2, 2), conditional=True)
        assert default.conditional is True
        assert default.f_stat == pytest.approx(explicit.f_stat)

    def test_the_two_models_give_different_statistics(self) -> None:
        y, x = _sample(seed=4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cond = bounds_test(y, x, case=3, order=(2, 2))
            uncond = bounds_test(y, x, case=3, order=(2, 2), conditional=False)
        assert cond.f_stat != pytest.approx(uncond.f_stat)
        assert uncond.conditional is False

    def test_summary_says_which_model_was_used(self) -> None:
        """Un lecteur qui reçoit une sortie doit savoir laquelle des deux
        spécifications l'a produite."""
        y, x = _sample(seed=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cond = bounds_test(y, x, case=3, order=(2, 2))
            uncond = bounds_test(y, x, case=3, order=(2, 2), conditional=False)
        assert "UNCONDITIONAL" in uncond.summary()
        assert "UNCONDITIONAL" not in cond.summary()


class TestBootstrapIntegration:
    def test_bootstrap_carries_the_choice_through(self) -> None:
        """Le DGP nul et la statistique doivent porter la MÊME
        spécification : sinon la nulle simulée n'est pas la nulle
        testée."""
        y, x = _sample(seed=6, n=120)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bootstrap_bounds_test(
                y, x, case=3, order=(2, 2), n_boot=199, seed=1, conditional=False
            )
            classical = bounds_test(y, x, case=3, order=(2, 2), conditional=False)
        assert res.classical.conditional is False
        assert res.f_stat == pytest.approx(classical.f_stat)

    def test_conditional_and_unconditional_bootstraps_differ(self) -> None:
        y, x = _sample(seed=7, n=120)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cond = bootstrap_bounds_test(y, x, case=3, order=(2, 2), n_boot=199, seed=1)
            uncond = bootstrap_bounds_test(
                y, x, case=3, order=(2, 2), n_boot=199, seed=1, conditional=False
            )
        assert cond.f_critical[0.05] != pytest.approx(uncond.f_critical[0.05])

    def test_reproducibility_holds_in_the_unconditional_form(self) -> None:
        y, x = _sample(seed=8, n=120)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            one = bootstrap_bounds_test(
                y, x, case=3, order=(2, 2), n_boot=199, seed=42, conditional=False
            )
            two = bootstrap_bounds_test(
                y, x, case=3, order=(2, 2), n_boot=199, seed=42, conditional=False
            )
        assert one.f_critical == two.f_critical
        assert one.f_indep_critical == two.f_indep_critical
