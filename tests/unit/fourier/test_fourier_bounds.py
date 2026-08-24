"""Spec 20 §3 — plan de tests du Fourier-ADL.

Deux choses rendent les valeurs critiques non standards, pas une : les
regresseurs sont integres, ET la frequence est choisie sur les donnees.
La seconde est la lecon d'OBS-15, que cette spec doit heriter.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.fourier import fourier_bounds_test


def _cointegrated_with_break(
    seed: int, n: int = 120, size: float = 4.0, lam: float = -0.3
) -> tuple[pd.Series, pd.DataFrame]:
    """Cointegration VRAIE, avec une rupture lisse dans la constante."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1)
    x = np.cumsum(rng.normal(size=n))
    shift = size / (1 + np.exp(-0.1 * (t - n / 2)))
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = (
            y[i - 1]
            + lam * (y[i - 1] - x[i - 1] - shift[i - 1])
            + rng.normal(scale=0.4)
        )
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


def _independent(seed: int, n: int = 120) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return (
        pd.Series(np.cumsum(rng.normal(size=n)), name="y"),
        pd.DataFrame({"x": np.cumsum(rng.normal(size=n))}),
    )


class TestModel:
    """§2.1 — l'UECM augmente des sinusoides."""

    def test_fourier_terms_enter_the_design_not_the_tested_vector(self) -> None:
        """Les sinusoides sont deterministes : elles appartiennent a la
        specification, pas a la relation de niveau qu'on teste. Les
        mettre dans le vecteur teste changerait l'hypothese nulle."""
        y, x = _cointegrated_with_break(seed=0)
        res = fourier_bounds_test(y, x, n_sims=200, seed=1)
        names = list(res._fit.names)
        assert any(n.startswith("fourier") for n in names)
        assert not any(n.startswith("fourier") for n in res._fit.tested)

    def test_more_harmonics_add_two_columns_each(self) -> None:
        y, x = _cointegrated_with_break(seed=1)
        one = fourier_bounds_test(y, x, fourier_k=1, n_sims=200, seed=1)
        two = fourier_bounds_test(y, x, fourier_k=2, n_sims=200, seed=1)
        assert len(two._fit.names) == len(one._fit.names) + 2

    @pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
    def test_every_deterministic_case_builds(self, case: int) -> None:
        y, x = _cointegrated_with_break(seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fourier_bounds_test(y, x, case=case, n_sims=200, seed=1)
        assert np.isfinite(res.t_stat)


class TestCriticalValues:
    """§2.3 — la lecon de Davies, heritee de la spec 19."""

    def test_selection_changes_the_critical_values(self) -> None:
        """Chercher la frequence deplace la loi nulle. Une valeur
        critique calculee a frequence fixee ne s'applique pas au
        resultat d'une recherche."""
        y, x = _independent(seed=10)
        searched = fourier_bounds_test(y, x, n_sims=800, seed=1)
        fixed = fourier_bounds_test(y, x, freq=1.0, n_sims=800, seed=1)
        assert searched.freq_estimated is True
        assert fixed.freq_estimated is False
        # La recherche prend le meilleur de cinq : la queue gauche du t
        # s'etend, donc la valeur critique est PLUS negative.
        assert searched.critical[0.05] < fixed.critical[0.05]

    def test_summary_names_the_construction(self) -> None:
        y, x = _independent(seed=11)
        searched = fourier_bounds_test(y, x, n_sims=200, seed=1)
        fixed = fourier_bounds_test(y, x, freq=1.0, n_sims=200, seed=1)
        assert "inside the loop" in searched.summary()
        assert "at the fixed frequency" in fixed.summary()

    def test_critical_values_are_ordered_and_negative(self) -> None:
        """Test unilateral GAUCHE : rejeter exige un lambda negatif."""
        y, x = _independent(seed=12)
        res = fourier_bounds_test(y, x, n_sims=500, seed=1)
        assert res.critical[0.01] < res.critical[0.05] < res.critical[0.10] < 0

    def test_reproducible_with_a_seed(self) -> None:
        y, x = _cointegrated_with_break(seed=13)
        a = fourier_bounds_test(y, x, n_sims=300, seed=9)
        b = fourier_bounds_test(y, x, n_sims=300, seed=9)
        assert a.critical == b.critical
        assert a.pvalue == b.pvalue

    def test_seed_recorded_when_omitted(self) -> None:
        y, x = _cointegrated_with_break(seed=14)
        res = fourier_bounds_test(y, x, n_sims=200)
        again = fourier_bounds_test(y, x, n_sims=200, seed=res.seed)
        assert again.critical == res.critical

    def test_pvalue_is_never_exactly_zero(self) -> None:
        y, x = _cointegrated_with_break(seed=15)
        res = fourier_bounds_test(y, x, n_sims=200, seed=1)
        assert res.pvalue >= 1 / (res.n_sims + 1)


class TestDecisions:
    def test_cointegration_with_a_break_is_found(self) -> None:
        y, x = _cointegrated_with_break(seed=20, lam=-0.4, size=4.0)
        res = fourier_bounds_test(y, x, n_sims=500, seed=1)
        assert res.decision == "cointegration"

    def test_independent_walks_are_not(self) -> None:
        y, x = _independent(seed=21)
        res = fourier_bounds_test(y, x, n_sims=500, seed=1)
        assert res.decision == "no_cointegration"

    @pytest.mark.fast_mc
    def test_size_under_the_null(self) -> None:
        """H0 vraie : deux marches aleatoires independantes. Le taux de
        rejet doit rester au voisinage du seuil nominal."""
        rejects = 0
        for m in range(60):
            y, x = _independent(seed=300 + m)
            rejects += (
                fourier_bounds_test(y, x, n_sims=300, seed=1).decision
                == "cointegration"
            )
        assert rejects / 60 < 0.15


class TestPreTest:
    """§2.4 — le pre-test qui dit quand ce test est le mauvais outil."""

    def test_break_makes_the_terms_significant(self) -> None:
        y, x = _cointegrated_with_break(seed=30, size=6.0)
        res = fourier_bounds_test(y, x, n_sims=600, seed=1)
        assert res.fourier_is_warranted
        assert "right one" in res.recommendation

    def test_no_break_leaves_them_insignificant(self) -> None:
        """Sans rupture, les deux parametres de Fourier sont depenses
        pour rien, et la recommandation le dit.

        La premiere version de ce pre-test appelait le test de Fourier
        autonome sur y. Sa loi nulle est simulee sur du BRUIT BLANC,
        alors que y est INTEGREE : la statistique etait donc toujours
        enorme et le pre-test se declarait significatif dans 100 % des
        cas, rupture ou pas. Il compare desormais deux ajustements du
        MEME modele, lus contre la loi simulee dans la meme boucle.
        OBS-18.
        """
        y, x = _cointegrated_with_break(seed=31, size=0.0)
        res = fourier_bounds_test(y, x, n_sims=600, seed=1)
        assert not res.fourier_is_warranted
        assert "prefer pyardl.bounds.bounds_test" in res.recommendation
        assert "NOT significant" in res.summary()

    def test_the_pretest_is_read_against_its_own_null(self) -> None:
        """Le F du pre-test se compare a une loi simulee sur des donnees
        INTEGREES, pas sur du bruit blanc. Sur une serie sans rupture, la
        valeur critique implicite est donc bien au-dessus d'un F(2, dof)
        tabule."""
        from scipy.stats import f as f_dist

        y, x = _cointegrated_with_break(seed=33, size=0.0)
        res = fourier_bounds_test(y, x, n_sims=600, seed=1)
        tabulated = float(f_dist.ppf(0.95, 2, res.nobs - len(res._fit.names)))
        assert res.fourier_critical[0.05] > tabulated
        assert res.fourier_pvalue > 0.05

    def test_pretest_appears_in_the_summary(self) -> None:
        y, x = _cointegrated_with_break(seed=32)
        text = fourier_bounds_test(y, x, n_sims=200, seed=1).summary()
        assert "pre-test on the Fourier terms" in text


class TestSelectionTable:
    def test_table_is_kept_when_searched(self) -> None:
        y, x = _cointegrated_with_break(seed=40)
        res = fourier_bounds_test(y, x, n_sims=200, seed=1)
        assert res.selection is not None
        assert list(res.selection.columns) == ["freq", "ssr"]
        assert res.selection["ssr"].is_monotonic_increasing
        assert res.frequency == res.selection.loc[0, "freq"]

    def test_no_table_when_fixed(self) -> None:
        y, x = _cointegrated_with_break(seed=41)
        res = fourier_bounds_test(y, x, freq=2.0, n_sims=200, seed=1)
        assert res.selection is None
        assert res.frequency == 2.0

    def test_decision_is_robust_to_the_grid_step(self) -> None:
        """§3.3 — un pas de grille plus fin ne doit pas retourner le
        verdict sur une relation nette."""
        y, x = _cointegrated_with_break(seed=42, lam=-0.4, size=4.0)
        coarse = fourier_bounds_test(y, x, n_sims=400, seed=1)
        fine = fourier_bounds_test(
            y, x, grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), n_sims=400, seed=1
        )
        assert coarse.decision == fine.decision


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"case": 6}, "case must be"),
            ({"fourier_k": 0}, "fourier_k must be"),
            ({"alpha": 0.5}, "alpha must be one of"),
            ({"n_sims": 10}, "too few"),
            ({"grid": ()}, "no frequency to search"),
            ({"freq": "best"}, "must be a number or 'auto'"),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        y, x = _cointegrated_with_break(seed=50)
        with pytest.raises(ValueError, match=match):
            fourier_bounds_test(y, x, **kwargs)

    def test_no_regressor(self) -> None:
        with pytest.raises(ValueError, match="at least one regressor"):
            fourier_bounds_test(pd.Series(np.arange(50.0), name="y"), None)  # type: ignore[arg-type]
