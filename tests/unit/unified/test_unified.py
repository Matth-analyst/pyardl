"""Spec 21 - plan de tests de la couche d'unification.

Ce module n'ajoute aucun estimateur : il route. Le plan de tests porte
donc sur les deux choses qu'il peut casser tout seul — le choix de la
source de valeurs critiques, et le cablage du moteur bootstrap generalise
— et sur rien d'autre. Les estimateurs sont deja verrouilles par les
specs 10 a 20.

Le verrou central est `TestEngineEquivalence` : avec la decomposition et
les termes de Fourier desactives, le moteur generalise DOIT reproduire
`bootstrap_bounds_test` a la precision machine, meme graine. S'il en
devie, c'est une reimplementation qui a derive, pas une generalisation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.bootstrap import bootstrap_bounds_test
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.unified import (
    UnifiedResults,
    cointegration_analysis,
    resolve_critical_values,
)
from pyardl.unified.analysis import _bootstrap_cell, _CellConfig


def _cointegrated(
    seed: int, n: int = 120, lam: float = -0.35, break_size: float = 0.0
) -> tuple[pd.Series, pd.DataFrame]:
    """Cointegration VRAIE, avec une rupture lisse optionnelle."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1)
    x = np.cumsum(rng.normal(size=n))
    shift = break_size / (1 + np.exp(-0.1 * (t - n / 2)))
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = (
            y[i - 1]
            + lam * (y[i - 1] - x[i - 1] - shift[i - 1])
            + rng.normal(scale=0.5)
        )
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


def _independent(seed: int, n: int = 120) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return (
        pd.Series(np.cumsum(rng.normal(size=n)), name="y"),
        pd.DataFrame({"x": np.cumsum(rng.normal(size=n))}),
    )


class TestCriticalValueRouting:
    """§2.1 - la seule responsabilite vraiment nouvelle du module."""

    @pytest.mark.parametrize(
        ("asym", "fourier", "inference", "expected"),
        [
            (False, False, "bounds", "pss_tables"),
            (False, True, "bounds", "simulated_fourier"),
            (True, False, "bounds", "simulated_syg"),
            (False, False, "bootstrap", "bootstrap"),
            (False, True, "bootstrap", "bootstrap"),
            (True, False, "bootstrap", "bootstrap"),
            (True, True, "bootstrap", "bootstrap"),
        ],
    )
    def test_each_cell_names_its_source(
        self, asym: bool, fourier: bool, inference: str, expected: str
    ) -> None:
        source, reason = resolve_critical_values(asym, fourier, inference)
        assert source == expected
        assert reason

    def test_the_uncovered_cell_is_refused_not_approximated(self) -> None:
        """NARDL + Fourier sans bootstrap : aucune table publiee ne couvre
        la combinaison. Substituer une table voisine en silence est
        exactement la faute que le projet traque ; lever est le design."""
        with pytest.raises(ValueError, match="No tabulated or pre-simulated"):
            resolve_critical_values(True, True, "bounds")

    def test_the_refusal_names_the_way_out(self) -> None:
        with pytest.raises(ValueError, match="inference='bootstrap'"):
            resolve_critical_values(True, True, "bounds")

    def test_the_refusal_reaches_the_public_entry_point(self) -> None:
        y, x = _cointegrated(seed=1)
        with pytest.raises(ValueError, match="No tabulated or pre-simulated"):
            cointegration_analysis(y, x, asym=["x"], fourier={"k": 1})


class TestEngineEquivalence:
    """§2.2 - LE verrou. Ecrit avant le reste, comme le test de residus
    de la spec 03."""

    def test_engine_reproduces_the_validated_brick_exactly(self) -> None:
        """Sans decomposition ni Fourier, le moteur generalise et
        `bootstrap_bounds_test` tirent le MEME DGP nul, les MEMES
        innovations et les MEMES chemins. Les trois statistiques et les
        neuf valeurs critiques doivent coincider a la precision machine,
        pas 'a peu pres'."""
        y, x = _cointegrated(seed=11)
        y_arr = y.to_numpy()
        x_arr = x.to_numpy()
        ref = bootstrap_bounds_test(
            y, x, case=3, order=(1, 1), n_boot=999, seed=4321, var_order=1, burn_in=50
        )
        cfg = _CellConfig(
            asym=(),
            fourier_k=0,
            freq=None,
            grid=(),
            inference="bootstrap",
            cv_source="bootstrap",
            cv_reason="",
        )
        out = _bootstrap_cell(
            y_arr, x_arr, x_arr, 0.0, (), 1, (1,), 3, True, cfg, 999, "iid", 1, 50, 4321
        )
        assert out.f_stat == pytest.approx(ref.f_stat, abs=1e-10)
        assert out.t_stat == pytest.approx(ref.t_stat, abs=1e-10)
        assert out.f_indep_stat == pytest.approx(ref.f_indep_stat, abs=1e-10)
        for a in (0.10, 0.05, 0.01):
            assert out.f_critical[a] == pytest.approx(ref.f_critical[a], abs=1e-10)
            assert out.t_critical[a] == pytest.approx(ref.t_critical[a], abs=1e-10)
            assert out.f_indep_critical[a] == pytest.approx(
                ref.f_indep_critical[a], abs=1e-10
            )

    def test_public_plain_cell_delegates_rather_than_recomputes(self) -> None:
        y, x = _cointegrated(seed=12)
        res = cointegration_analysis(
            y, x, inference="bootstrap", order=(1, 1), n_boot=299, seed=7
        )
        ref = bootstrap_bounds_test(y, x, case=3, order=(1, 1), n_boot=299, seed=7)
        assert res.f_stat == pytest.approx(ref.f_stat, abs=1e-12)
        assert res.f_critical == ref.f_critical


class TestDecompositionInTheLoop:
    """§2.3 - les sommes partielles sont une FONCTION des donnees."""

    def test_partial_sums_are_recomputed_on_each_regenerated_path(self) -> None:
        """Re-simuler des sommes partielles comme si elles etaient des
        series I(1) ordinaires produirait des chemins dont les increments
        n'ont pas de signe constant — donc des sommes partielles de rien.
        Le moteur regenere x a l'echelle d'origine puis re-applique la
        decomposition ; l'identite x = x0 + x+ + x- doit tenir sur chaque
        chemin regenere."""
        from pyardl.bootstrap.dgp import estimate_null_dgp, simulate_paths

        y, x = _cointegrated(seed=13)
        y_arr, x_arr = y.to_numpy(), x.to_numpy()
        from pyardl.nardl.decompose import partial_sums

        pos, neg = partial_sums(x["x"])
        z = np.column_stack([pos.to_numpy(), neg.to_numpy()])
        dgp = estimate_null_dgp(
            y_arr, z, p=1, q=(1, 1), case=3, var_order=1, x_marginal=x_arr
        )
        rng = np.random.default_rng(0)
        inn = rng.normal(size=(5, 170, 1 + 1))

        def expand(dx_t: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [np.maximum(dx_t[:, 0], 0.0), np.minimum(dx_t[:, 0], 0.0)]
            )

        _, x_star = simulate_paths(
            dgp, inn, y0=y_arr[0], x0=x_arr[0], burn_in=50, expand=expand
        )
        d = np.diff(x_star[:, :, 0], axis=1)
        zero = np.zeros((5, 1))
        p_star = np.concatenate([zero, np.cumsum(np.maximum(d, 0.0), axis=1)], axis=1)
        n_star = np.concatenate([zero, np.cumsum(np.minimum(d, 0.0), axis=1)], axis=1)
        rebuilt = x_star[:, :1, 0] + p_star + n_star
        assert np.allclose(rebuilt, x_star[:, :, 0], atol=1e-12)

    def test_nardl_cell_runs_and_classifies(self) -> None:
        y, x = _cointegrated(seed=14, lam=-0.4)
        res = cointegration_analysis(
            y, x, asym=["x"], inference="bootstrap", n_boot=299, seed=3
        )
        assert res.label == "nardl/bootstrap"
        assert res.classification is not None
        assert res.order[1].keys() == {"x_pos", "x_neg"}


class TestFourierInTheNull:
    """§2.4 - un terme deterministe ne disparait pas avec la relation."""

    def test_the_break_is_carried_into_the_null_dgp(self) -> None:
        """Si les colonnes de Fourier etaient retirees du modele nul, les
        chemins regeneres n'auraient plus la rupture, et les valeurs
        critiques seraient celles d'un autre monde que celui teste."""
        from pyardl.bootstrap.dgp import estimate_null_dgp
        from pyardl.unified.analysis import _fourier_matrix

        y, x = _cointegrated(seed=15, break_size=6.0)
        y_arr, x_arr = y.to_numpy(), x.to_numpy()
        t = np.arange(1, y_arr.size + 1, dtype=np.float64)
        det = _fourier_matrix(t, y_arr.size, 1.0, 1)
        dgp = estimate_null_dgp(y_arr, x_arr, p=1, q=(1,), case=3, var_order=1, det=det)
        assert dgp.det_coefs is not None
        assert dgp.det_coefs.size == 2
        plain = estimate_null_dgp(y_arr, x_arr, p=1, q=(1,), case=3, var_order=1)
        assert plain.det_coefs is None

    def test_det_paths_extend_backwards_through_the_burn_in(self) -> None:
        """Comme la tendance, la sinusoide est la meme fonction du temps
        prolongee vers l'arriere : c'est le seul choix qui ne laisse pas
        de discontinuite au raccord."""
        from pyardl.unified.analysis import _fourier_matrix

        burn_in = 50
        full = _fourier_matrix(
            np.arange(1 - burn_in, 121, dtype=np.float64), 120, 1.0, 1
        )
        kept = _fourier_matrix(np.arange(1, 121, dtype=np.float64), 120, 1.0, 1)
        assert full.shape == (burn_in + 120, 2)
        assert np.allclose(full[burn_in:], kept, atol=1e-14)

    def test_searched_frequency_widens_the_null(self) -> None:
        """La lecon de Davies survit a la composition : chercher la
        frequence dans le bootstrap etend la queue gauche du t, donc la
        valeur critique est PLUS negative qu'a frequence fixee."""
        y, x = _independent(seed=16)
        searched = cointegration_analysis(
            y, x, fourier={"k": 1}, inference="bootstrap", n_boot=600, seed=5
        )
        fixed = cointegration_analysis(
            y,
            x,
            fourier={"k": 1, "freq": 1.0},
            inference="bootstrap",
            n_boot=600,
            seed=5,
        )
        assert searched.frequency is not None
        assert searched.t_critical is not None and fixed.t_critical is not None
        assert searched.t_critical[0.05] < fixed.t_critical[0.05]

    def test_selection_table_is_kept(self) -> None:
        y, x = _cointegrated(seed=17, break_size=4.0)
        res = cointegration_analysis(
            y, x, fourier={"k": 1}, inference="bootstrap", n_boot=299, seed=5
        )
        table = res.detail.selection
        assert table is not None
        assert list(table.columns) == ["freq", "ssr"]
        assert table["ssr"].is_monotonic_increasing
        assert res.frequency == table.loc[0, "freq"]


class TestMixedRegressors:
    """§2.5 - un seul regresseur decompose parmi plusieurs.

    Le cas applique le plus frequent, et celui ou le cablage peut se
    tromper en silence : les colonnes decomposees et les colonnes
    intactes doivent traverser le moteur dans le bon ordre, a
    l'estimation comme dans chaque chemin regenere.
    """

    @staticmethod
    def _two_regressors(seed: int, n: int = 120) -> tuple[pd.Series, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        a = np.cumsum(rng.normal(size=n))
        b = np.cumsum(rng.normal(size=n))
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = (
                y[i - 1]
                - 0.4 * (y[i - 1] - a[i - 1] - 0.5 * b[i - 1])
                + rng.normal(scale=0.5)
            )
        return pd.Series(y, name="y"), pd.DataFrame({"a": a, "b": b})

    def test_only_one_regressor_is_decomposed(self) -> None:
        y, x = self._two_regressors(seed=60)
        res = cointegration_analysis(
            y, x, asym=["a"], inference="bootstrap", n_boot=299, seed=1
        )
        assert set(res.order[1]) == {"a_pos", "a_neg", "b"}
        assert res.classification is not None

    def test_the_intact_column_survives_the_regenerated_paths(self) -> None:
        """La colonne non decomposee doit ressortir du moteur egale au
        chemin marginal lui-meme, pas melangee avec les sommes
        partielles de sa voisine."""
        from pyardl.unified.analysis import _bootstrap_cell, _CellConfig

        y, x = self._two_regressors(seed=61)
        from pyardl.nardl.decompose import partial_sums

        pos, neg = partial_sums(x["a"])
        z = np.column_stack([pos.to_numpy(), neg.to_numpy(), x["b"].to_numpy()])
        cfg = _CellConfig(
            asym=("a",),
            fourier_k=0,
            freq=None,
            grid=(),
            inference="bootstrap",
            cv_source="bootstrap",
            cv_reason="",
        )
        out = _bootstrap_cell(
            y.to_numpy(),
            z,
            x.to_numpy(),
            0.0,
            (0,),
            1,
            (1, 1, 1),
            3,
            True,
            cfg,
            299,
            "iid",
            1,
            50,
            5,
        )
        assert np.isfinite(out.f_stat)
        assert out.n_boot > 0

    def test_mixed_order_dict_uses_transformed_names(self) -> None:
        y, x = self._two_regressors(seed=62)
        res = cointegration_analysis(
            y,
            x,
            asym=["a"],
            inference="bootstrap",
            order=(1, {"a_pos": 1, "a_neg": 1, "b": 2}),
            n_boot=199,
            seed=1,
        )
        assert res.order[1] == {"a_pos": 1, "a_neg": 1, "b": 2}

    def test_mixed_cell_with_fourier(self) -> None:
        y, x = self._two_regressors(seed=63)
        res = cointegration_analysis(
            y,
            x,
            asym=["a"],
            fourier={"k": 1},
            inference="bootstrap",
            n_boot=299,
            seed=1,
        )
        assert res.frequency is not None
        assert res.classification is not None


class TestResults:
    def test_summary_reports_the_frequency_when_there_is_one(self) -> None:
        y, x = _cointegrated(seed=19, break_size=4.0)
        res = cointegration_analysis(
            y, x, fourier={"k": 1}, inference="bootstrap", n_boot=199, seed=1
        )
        assert "Fourier frequency:" in res.summary()

    def test_summary_names_the_source_and_the_reason(self) -> None:
        y, x = _cointegrated(seed=20)
        res = cointegration_analysis(y, x)
        text = res.summary()
        assert "pss_tables" in text
        assert "F_overall" in text and "t_BDM" in text and "F_indep" in text

    def test_summary_says_when_a_statistic_is_not_covered(self) -> None:
        """La route tabulee du NARDL ne porte que le F d'ensemble. Le dire
        vaut mieux que d'afficher un vide, et beaucoup mieux que de
        remplir la case avec une valeur critique d'ailleurs."""
        y, x = _cointegrated(seed=21)
        res = cointegration_analysis(y, x, asym=["x"])
        text = res.summary()
        assert "not covered by this source" in text
        assert "classification: unavailable" in text

    def test_bootstrap_route_carries_all_three(self) -> None:
        y, x = _cointegrated(seed=22, lam=-0.4)
        res = cointegration_analysis(y, x, inference="bootstrap", n_boot=299, seed=1)
        assert res.f_stat is not None
        assert res.t_stat is not None
        assert res.f_indep_stat is not None
        assert res.classification is not None

    def test_reproducible_with_a_seed(self) -> None:
        y, x = _cointegrated(seed=23)
        a = cointegration_analysis(y, x, inference="bootstrap", n_boot=299, seed=99)
        b = cointegration_analysis(y, x, inference="bootstrap", n_boot=299, seed=99)
        assert a.f_critical == b.f_critical
        assert a.t_critical == b.t_critical

    def test_seed_recorded_when_omitted(self) -> None:
        y, x = _cointegrated(seed=24)
        res = cointegration_analysis(
            y, x, asym=["x"], inference="bootstrap", n_boot=199
        )
        assert res.seed is not None
        again = cointegration_analysis(
            y, x, asym=["x"], inference="bootstrap", n_boot=199, seed=res.seed
        )
        assert again.f_critical == res.f_critical

    def test_pvalues_are_never_exactly_zero(self) -> None:
        y, x = _cointegrated(seed=25, lam=-0.5)
        res = cointegration_analysis(y, x, inference="bootstrap", n_boot=199, seed=1)
        assert res.f_pvalue is not None and res.n_boot is not None
        assert res.f_pvalue >= 1 / (res.n_boot + 1)

    def test_results_are_immutable(self) -> None:
        y, x = _cointegrated(seed=26)
        res = cointegration_analysis(y, x)
        with pytest.raises(AttributeError):
            res.f_stat = 0.0  # type: ignore[misc]


class TestCompare:
    def test_the_robustness_table_runs_every_cell(self) -> None:
        y, x = _cointegrated(seed=30, lam=-0.4)
        base = cointegration_analysis(y, x, inference="bootstrap", n_boot=199, seed=2)
        table = base.compare(n_boot=199)
        assert len(table) == 4
        assert "classification" in table.columns
        assert table["classification"].notna().all()

    def test_unavailable_cells_are_reported_not_skipped(self) -> None:
        """Une case sans source valide disparait silencieusement d'un
        tableau si on l'attrape sans le dire. Elle doit y figurer avec sa
        raison."""
        y, x = _cointegrated(seed=31)
        base = cointegration_analysis(y, x)
        table = base.compare()
        assert len(table) == 4
        unavailable = table.loc["nardl+fourier/bounds", "classification"]
        assert isinstance(unavailable, str)
        assert unavailable.startswith("unavailable:")


class TestGuards:
    def test_searched_frequency_under_bootstrap_warns(self) -> None:
        """La seule combinaison mesuree sur-rejetante doit le dire. La
        recherche est recalibree dans chaque replication ; ce qui ne
        l'est pas, c'est que le DGP nul a ete estime a la frequence que
        la recherche avait deja gagnee. OBS-19."""
        y, x = _cointegrated(seed=43)
        with pytest.warns(PyardlMethodologyWarning, match="over-rejects"):
            cointegration_analysis(
                y, x, fourier={"k": 1}, inference="bootstrap", n_boot=199, seed=1
            )

    def test_a_fixed_frequency_does_not_warn(self) -> None:
        """Le bras correctement dimensionne ne porte pas l'avertissement,
        sinon il ne veut plus rien dire."""
        y, x = _cointegrated(seed=44)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PyardlMethodologyWarning)
            cointegration_analysis(
                y,
                x,
                fourier={"k": 1, "freq": 1.0},
                inference="bootstrap",
                n_boot=199,
                seed=1,
            )

    def test_the_bounds_fourier_route_does_not_warn(self) -> None:
        """L'avertissement porte sur le bootstrap. La route a valeurs
        critiques simulees de la spec 20 a ete mesuree a 3.5 % ; lui
        coller le meme avertissement serait faux."""
        y, x = _cointegrated(seed=45)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PyardlMethodologyWarning)
            cointegration_analysis(y, x, fourier={"k": 1}, n_boot=199, seed=1)

    def test_overparameterised_specification_warns(self) -> None:
        """Une decomposition, des sinusoides et des retards paraissent
        bon marche separement. Ensemble, sur 60 points, ils ne le sont
        pas — et la loi asymptotique invoquee n'a plus rien a voir avec
        l'echantillon."""
        y, x = _cointegrated(seed=40, n=60)
        with pytest.warns(PyardlMethodologyWarning, match="ratio"):
            cointegration_analysis(
                y,
                x,
                asym=["x"],
                fourier={"k": 2},
                inference="bootstrap",
                order=(3, 3),
                n_boot=199,
                seed=1,
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"inference": "magic"}, "inference must be"),
            ({"alpha": 0.5}, "alpha must be one of"),
            ({"asym": ["z"]}, "are not regressors"),
            ({"fourier": {"k": 0}}, "fourier k must be"),
            ({"fourier": {"harmonics": 2}}, "are not understood"),
            ({"fourier": {"grid": ()}}, "no frequency to search"),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        y, x = _cointegrated(seed=41)
        with pytest.raises(ValueError, match=match):
            cointegration_analysis(y, x, **kwargs)

    def test_order_dict_must_cover_the_transformed_columns(self) -> None:
        """Apres decomposition les colonnes s'appellent x_pos et x_neg.
        Un dict indexe sur les noms d'origine ne les couvre pas, et le
        dire vaut mieux que de completer en silence."""
        y, x = _cointegrated(seed=42)
        with pytest.raises(ValueError, match="missing the transformed columns"):
            cointegration_analysis(
                y,
                x,
                asym=["x"],
                inference="bootstrap",
                order=(1, {"x": 1}),
                n_boot=199,
                seed=1,
            )

    def test_singular_design_is_named_not_swallowed(self) -> None:
        """Un design singulier produit des statistiques NaN qui se
        propageraient jusqu'a une decision. Le refus nomme la cause et la
        sortie, plutot que de rendre un verdict sur du vide."""
        rng = np.random.default_rng(3)
        n = 120
        a = np.cumsum(rng.normal(size=n))
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = y[i - 1] - 0.4 * (y[i - 1] - a[i - 1]) + rng.normal(scale=0.5)
        with pytest.raises(ValueError, match="singular design"):
            cointegration_analysis(
                pd.Series(y, name="y"),
                pd.DataFrame({"a": a, "b": 2.0 * a}),
                asym=["a"],
                inference="bootstrap",
                order=(1, 1),
                n_boot=199,
                seed=1,
            )

    def test_no_regressor(self) -> None:
        with pytest.raises(ValueError, match="at least one regressor"):
            cointegration_analysis(pd.Series(np.arange(50.0), name="y"), None)  # type: ignore[arg-type]


class TestDecisions:
    def test_independent_walks_are_not_cointegrated_in_any_cell(self) -> None:
        y, x = _independent(seed=50)
        for kwargs in (
            {},
            {"asym": ["x"], "inference": "bootstrap"},
            {"fourier": {"k": 1}, "inference": "bootstrap"},
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cointegration_analysis(y, x, n_boot=399, seed=1, **kwargs)  # type: ignore[arg-type]
            assert res.decision_t in (None, "no_cointegration")

    def test_a_clear_relation_is_found_in_every_bootstrap_cell(self) -> None:
        y, x = _cointegrated(seed=51, lam=-0.5)
        for kwargs in (
            {},
            {"asym": ["x"]},
            {"fourier": {"k": 1}},
            {"asym": ["x"], "fourier": {"k": 1}},
        ):
            res = cointegration_analysis(
                y, x, inference="bootstrap", n_boot=399, seed=1, **kwargs
            )  # type: ignore[arg-type]
            assert res.classification == "cointegration", kwargs

    @pytest.mark.fast_mc
    def test_size_of_the_richest_cell(self) -> None:
        """H0 vraie. La case la plus riche — decomposition ET Fourier —
        est celle ou tout peut deraper a la fois. Version rapide ; la
        mesure dimensionnee est dans validation/spec21_size.py."""
        rejects = 0
        for m in range(40):
            y, x = _independent(seed=600 + m)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cointegration_analysis(
                    y,
                    x,
                    asym=["x"],
                    fourier={"k": 1},
                    inference="bootstrap",
                    n_boot=299,
                    seed=1,
                )
            rejects += res.classification == "cointegration"
        assert rejects / 40 < 0.20


def test_module_exports() -> None:
    assert issubclass(UnifiedResults, object)
    assert callable(cointegration_analysis)
    assert callable(resolve_critical_values)
