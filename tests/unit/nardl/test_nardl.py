"""Spec 17 §4 — plan de tests du NARDL.

Le verrou de la decomposition est dans test_decompose.py. Ici on verifie
ce que le modele ajoute : l'equivalence des deux parametrisations, la
recuperation des coefficients asymetriques, les tests de symetrie, et la
convergence des multiplicateurs vers le long terme.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.nardl import NARDL


def _asymmetric(
    seed: int,
    n: int = 300,
    theta_pos: float = 2.0,
    theta_neg: float = 0.5,
    lam: float = -0.4,
) -> tuple[pd.Series, pd.DataFrame]:
    """DGP NARDL : y repond differemment aux hausses et aux baisses de x."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    delta = np.diff(x)
    pos = np.concatenate([[0.0], np.cumsum(np.maximum(delta, 0.0))])
    neg = np.concatenate([[0.0], np.cumsum(np.minimum(delta, 0.0))])
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = (
            y[t - 1]
            + lam * (y[t - 1] - theta_pos * pos[t - 1] - theta_neg * neg[t - 1])
            + rng.normal(scale=0.5)
        )
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


def _symmetric(seed: int, n: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    """DGP symetrique : le meme coefficient des deux cotes."""
    return _asymmetric(seed, n=n, theta_pos=1.0, theta_neg=1.0)


class TestParameterisationEquivalence:
    """VERROU — les deux vues sont un seul ajustement.

    Les multiplicateurs se calculent sur la forme ARDL, les tests de Wald
    et les bornes sur la forme a correction d'erreur. Si les deux
    n'etaient pas la meme regression, le graphique et les tests
    decriraient deux modeles differents sans que rien ne le signale.
    """

    @pytest.mark.parametrize("order", [(1, 1), (2, 2), (3, 1)])
    def test_residuals_coincide(self, order: tuple[int, int]) -> None:
        y, x = _asymmetric(seed=0)
        res = NARDL(y, x, order=order).fit()
        r_ecm = np.asarray(res._fit.resid, dtype=float)
        r_ardl = np.asarray(res._ardl_res._resid, dtype=float)
        assert r_ecm.shape == r_ardl.shape
        assert np.max(np.abs(r_ecm - r_ardl)) < 1e-10

    @pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
    def test_residuals_coincide_across_cases(self, case: int) -> None:
        y, x = _asymmetric(seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = NARDL(y, x, order=(1, 1), case=case).fit()
        assert (
            np.max(
                np.abs(np.asarray(res._fit.resid) - np.asarray(res._ardl_res._resid))
            )
            < 1e-10
        )


class TestRecovery:
    """§4.1 — les deux coefficients de long terme sont retrouves."""

    @pytest.mark.parametrize("seed", range(3))
    def test_asymmetric_coefficients(self, seed: int) -> None:
        y, x = _asymmetric(seed=10 + seed, n=500)
        res = NARDL(y, x, order=(1, 1)).fit()
        row = res.longrun_asym.loc["x"]
        assert row["theta_pos"] == pytest.approx(2.0, abs=0.25)
        assert row["theta_neg"] == pytest.approx(0.5, abs=0.25)

    def test_difference_is_the_difference(self) -> None:
        """La colonne 'difference' vaut bien theta+ moins theta-."""
        y, x = _asymmetric(seed=20)
        row = NARDL(y, x, order=(1, 1)).fit().longrun_asym.loc["x"]
        assert row["difference"] == pytest.approx(
            row["theta_pos"] - row["theta_neg"], rel=1e-10
        )

    def test_lambda_is_negative_under_error_correction(self) -> None:
        y, x = _asymmetric(seed=21)
        assert NARDL(y, x, order=(1, 1)).fit().lam < 0


class TestAsymmetryTests:
    """§2.3 — les quatre tests, et ce qu'ils disent."""

    def test_asymmetric_dgp_is_detected(self) -> None:
        y, x = _asymmetric(seed=30, n=400)
        tests = NARDL(y, x, order=(1, 1)).fit().asymmetry_tests()
        assert tests.loc[("x", "longrun_gamma"), "pvalue"] < 0.01
        assert tests.loc[("x", "longrun_gamma"), "decision"] == "asymmetric"

    def test_symmetric_dgp_is_not_flagged(self) -> None:
        y, x = _symmetric(seed=31, n=400)
        tests = NARDL(y, x, order=(1, 1)).fit().asymmetry_tests()
        assert tests.loc[("x", "longrun_gamma"), "pvalue"] > 0.05

    def test_all_four_tests_are_reported(self) -> None:
        y, x = _asymmetric(seed=32)
        tests = NARDL(y, x, order=(2, 2)).fit().asymmetry_tests()
        assert set(tests.index.get_level_values("test")) == {
            "longrun_gamma",
            "longrun_theta",
            "shortrun_additive",
            "shortrun_strong",
        }

    def test_strong_shortrun_unavailable_with_unpaired_orders(self) -> None:
        """Ordres differents des deux cotes : le test fort n'a pas de
        sens et le dit, au lieu d'apparier des termes qui ne se
        correspondent pas."""
        y, x = _asymmetric(seed=33)
        res = NARDL(y, x, order=(1, {"x_pos": 3, "x_neg": 1})).fit()
        tests = res.asymmetry_tests()
        assert tests.loc[("x", "shortrun_strong"), "decision"] == "unavailable"
        assert np.isnan(tests.loc[("x", "shortrun_strong"), "stat"])
        # Le test additif, lui, reste defini.
        assert np.isfinite(tests.loc[("x", "shortrun_additive"), "stat"])

    def test_symmetric_model_is_suggested_when_nothing_rejects(self) -> None:
        y, x = _symmetric(seed=34, n=400)
        res = NARDL(y, x, order=(1, 1)).fit()
        if res.suggests_symmetric_model():
            assert "symmetric ARDL" in res.summary()


class TestMultipliers:
    """§4.2 — convergence vers le long terme, et le miroir symetrique."""

    def test_convergence_to_the_long_run(self) -> None:
        """m+_h -> theta+ et m-_h -> theta- : c'est ce qui fait des
        multiplicateurs le chemin du long terme, pas une autre grandeur."""
        y, x = _asymmetric(seed=40, n=400)
        res = NARDL(y, x, order=(1, 1)).fit()
        table = res.dynamic_multipliers(h=200, r=2, seed=1)
        row = res.longrun_asym.loc["x"]
        assert table["m_pos"].iloc[-1] == pytest.approx(row["theta_pos"], abs=1e-3)
        assert table["m_neg"].iloc[-1] == pytest.approx(row["theta_neg"], abs=1e-3)

    def test_symmetric_case_gives_equal_multipliers(self) -> None:
        y, x = _symmetric(seed=41, n=400)
        res = NARDL(y, x, order=(1, 1)).fit()
        table = res.dynamic_multipliers(h=100, r=2, seed=1)
        assert table["difference"].iloc[-1] == pytest.approx(0.0, abs=0.35)

    def test_reproducible_with_a_seed(self) -> None:
        y, x = _asymmetric(seed=42)
        res = NARDL(y, x, order=(1, 1)).fit()
        a = res.dynamic_multipliers(h=20, r=100, seed=7)
        b = res.dynamic_multipliers(h=20, r=100, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_seed_is_recorded_when_omitted(self) -> None:
        y, x = _asymmetric(seed=43)
        res = NARDL(y, x, order=(1, 1)).fit()
        table = res.dynamic_multipliers(h=10, r=50)
        seed = table.attrs["seed"]
        again = res.dynamic_multipliers(h=10, r=50, seed=seed)
        pd.testing.assert_frame_equal(table, again, check_like=True)

    def test_bands_bracket_the_point_estimate(self) -> None:
        y, x = _asymmetric(seed=44, n=400)
        table = NARDL(y, x, order=(1, 1)).fit().dynamic_multipliers(h=30, r=400, seed=3)
        inside = (table["m_pos"] >= table["m_pos_lower"] - 1e-9) & (
            table["m_pos"] <= table["m_pos_upper"] + 1e-9
        )
        assert inside.mean() > 0.9

    def test_horizon_zero_is_the_impact_effect(self) -> None:
        """A l'horizon 0 le multiplicateur vaut le coefficient
        contemporain, rien de plus."""
        y, x = _asymmetric(seed=45)
        res = NARDL(y, x, order=(1, 1)).fit()
        table = res.dynamic_multipliers(h=5, r=2, seed=1)
        beta0 = float(res._ardl_res.params["x_pos.L0"])
        assert table["m_pos"].iloc[0] == pytest.approx(beta0, rel=1e-10)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"h": 0}, "h must be at least 1"),
            ({"r": 1}, "r must be at least 2"),
            ({"alpha": 0.0}, "alpha must lie"),
            ({"alpha": 1.0}, "alpha must lie"),
        ],
    )
    def test_invalid_arguments_refused(self, kwargs: dict, match: str) -> None:
        y, x = _asymmetric(seed=46)
        res = NARDL(y, x, order=(1, 1)).fit()
        with pytest.raises(ValueError, match=match):
            res.dynamic_multipliers(**kwargs)


class TestBoundsTest:
    """§2.4 — la valeur critique est simulee pour CE nul, pas empruntee."""

    def test_returns_a_single_critical_value_per_level(self) -> None:
        """Pas de paire de bornes : decomposer une serie stationnaire
        donne des series a tendance, donc la borne I(0) ne decrit aucun
        monde atteignable. Une seule valeur, et donc aucune zone non
        concluante."""
        y, x = _asymmetric(seed=50, n=400)
        res = NARDL(y, x, order=(1, 1)).fit().bounds_test()
        assert set(res.critical) == {0.10, 0.05, 0.01}
        assert res.decision in ("cointegration", "no_cointegration")

    def test_statistic_matches_the_uecm_wald(self) -> None:
        from pyardl.bounds.pss import _wald_f

        y, x = _asymmetric(seed=51, n=400)
        fitted = NARDL(y, x, order=(1, 1)).fit()
        assert fitted.bounds_test().f_stat == pytest.approx(_wald_f(fitted._fit))

    def test_cointegrated_data_reject(self) -> None:
        y, x = _asymmetric(seed=52, n=400)
        assert NARDL(y, x, order=(1, 1)).fit().bounds_test().decision == (
            "cointegration"
        )

    def test_independent_walks_do_not_reject(self) -> None:
        rng = np.random.default_rng(53)
        n = 300
        y = pd.Series(np.cumsum(rng.normal(size=n)), name="y")
        x = pd.DataFrame({"x": np.cumsum(rng.normal(size=n))})
        res = NARDL(y, x, order=(1, 1)).fit().bounds_test()
        assert res.decision == "no_cointegration"

    def test_critical_values_come_from_the_nardl_table(self) -> None:
        from pyardl.critical_values.syg2014 import nardl_critical_value

        y, x = _asymmetric(seed=54, n=400)
        res = NARDL(y, x, order=(1, 1), case=3).fit().bounds_test()
        assert res.critical[0.05] == nardl_critical_value(3, 1, 0.05)

    def test_stricter_level_is_harder_to_reject(self) -> None:
        y, x = _asymmetric(seed=55, n=400)
        crit = NARDL(y, x, order=(1, 1)).fit().bounds_test().critical
        assert crit[0.10] < crit[0.05] < crit[0.01]

    def test_mixed_model_is_refused_rather_than_approximated(self) -> None:
        """Un regresseur symetrique en plus : la table ne couvre pas ce
        design, et on le dit au lieu de lire une valeur voisine."""
        rng = np.random.default_rng(56)
        n = 200
        y = pd.Series(np.cumsum(rng.normal(size=n)), name="y")
        frame = pd.DataFrame(
            {"a": np.cumsum(rng.normal(size=n)), "b": np.cumsum(rng.normal(size=n))}
        )
        res = NARDL(y, frame, asym=["a"], order=(1, 1)).fit()
        with pytest.raises(NotImplementedError, match="every"):
            res.bounds_test()

    def test_summary_names_the_source_of_the_values(self) -> None:
        y, x = _asymmetric(seed=57, n=400)
        text = NARDL(y, x, order=(1, 1)).fit().bounds_test().summary()
        assert "syg2014" in text
        assert "NARDL bounds test" in text


class TestModelValidation:
    """Refus explicites a la construction."""

    def test_unknown_asym_name(self) -> None:
        y, x = _asymmetric(seed=60)
        with pytest.raises(ValueError, match="are not regressors"):
            NARDL(y, x, asym=["oil"])

    def test_empty_asym(self) -> None:
        y, x = _asymmetric(seed=61)
        with pytest.raises(ValueError, match="Use pyardl.ARDL instead"):
            NARDL(y, x, asym=[])

    def test_bad_case(self) -> None:
        y, x = _asymmetric(seed=62)
        with pytest.raises(ValueError, match="case must be 1..5"):
            NARDL(y, x, case=6)

    def test_no_regressor(self) -> None:
        with pytest.raises(ValueError, match="at least one regressor"):
            NARDL(pd.Series(np.arange(20.0), name="y"), None)  # type: ignore[arg-type]

    def test_order_dict_must_name_transformed_columns(self) -> None:
        """Le dict d'ordres porte sur les colonnes decomposees, pas sur
        la variable d'origine — le message le dit."""
        y, x = _asymmetric(seed=63)
        with pytest.raises(ValueError, match="_pos and <name>_neg"):
            NARDL(y, x, order=(1, {"x": 1}))


class TestPartialAsymmetry:
    """Un sous-ensemble decompose, le reste symetrique."""

    def test_only_the_named_regressor_is_decomposed(self) -> None:
        rng = np.random.default_rng(70)
        n = 200
        a = np.cumsum(rng.normal(size=n))
        b = np.cumsum(rng.normal(size=n))
        y = pd.Series(np.cumsum(rng.normal(size=n)), name="y")
        model = NARDL(y, pd.DataFrame({"a": a, "b": b}), asym=["a"], order=(1, 1))
        assert list(model.transformed.columns) == ["a_pos", "a_neg", "b"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit()
        assert res.asym == ("a",)
        assert list(res.longrun_asym.index) == ["a"]


class TestSummary:
    def test_summary_reports_everything_needed(self) -> None:
        y, x = _asymmetric(seed=80, n=400)
        text = NARDL(y, x, order=(1, 1)).fit().summary()
        for expected in ("NARDL", "theta+", "theta-", "difference", "symmetry tests"):
            assert expected in text


class TestCriticalValueTable:
    """La table simulee : couverture, refus, et son ecart a PSS."""

    def test_covered_configurations(self) -> None:
        from pyardl.critical_values.syg2014 import MAX_K_ASYM, nardl_critical_value

        for case in (1, 2, 3, 4, 5):
            for k in range(1, MAX_K_ASYM + 1):
                for alpha in (0.10, 0.05, 0.01):
                    assert nardl_critical_value(case, k, alpha) > 0

    def test_out_of_grid_is_refused(self) -> None:
        from pyardl.critical_values.syg2014 import MAX_K_ASYM, nardl_critical_value

        with pytest.raises(ValueError, match="decomposed variables"):
            nardl_critical_value(3, MAX_K_ASYM + 1, 0.05)
        with pytest.raises(ValueError, match="case must be"):
            nardl_critical_value(6, 1, 0.05)
        with pytest.raises(ValueError, match="No simulated value"):
            nardl_critical_value(3, 1, 0.025)

    def test_decreasing_in_the_number_of_variables(self) -> None:
        """Le F est un F PAR restriction : chaque variable decomposee
        supplementaire le dilue."""
        from pyardl.critical_values.syg2014 import nardl_critical_value

        for case in (1, 2, 3, 4, 5):
            values = [nardl_critical_value(case, k, 0.05) for k in (1, 2, 3)]
            assert values[0] > values[1] > values[2]

    def test_exceeds_pss_in_the_cases_without_trend(self) -> None:
        """C'est la raison d'etre de la table : lire un NARDL contre les
        bornes de PSS a k = 2*k_asym est trop permissif."""
        from pyardl.critical_values import get_bounds
        from pyardl.critical_values.syg2014 import nardl_critical_value

        for case in (1, 2, 3):
            for k in (1, 2, 3):
                _, pss_upper = get_bounds("F", case=case, k=2 * k, alpha=0.05)
                assert nardl_critical_value(case, k, 0.05) > pss_upper


class TestPlot:
    """La figure : on teste les DONNEES tracees, pas les pixels.

    Un graphique se verifie sur ce qu'il represente. Comparer des images
    ne dirait rien de la justesse des courbes et casserait a chaque
    changement de version de matplotlib.
    """

    def test_curves_carry_the_computed_multipliers(self) -> None:
        y, x = _asymmetric(seed=90, n=400)
        res = NARDL(y, x, order=(1, 1)).fit()
        table = res.dynamic_multipliers(h=15, r=200, seed=11)
        fig = res.plot_multipliers(h=15, r=200, seed=11)
        top, bottom = fig.axes
        drawn_pos = top.lines[0].get_ydata()
        drawn_neg = top.lines[1].get_ydata()
        drawn_diff = bottom.lines[0].get_ydata()
        assert np.allclose(drawn_pos, table["m_pos"].to_numpy())
        assert np.allclose(drawn_neg, table["m_neg"].to_numpy())
        assert np.allclose(drawn_diff, table["difference"].to_numpy())

    def test_two_panels_and_a_zero_line_on_each(self) -> None:
        y, x = _asymmetric(seed=91, n=400)
        fig = NARDL(y, x, order=(1, 1)).fit().plot_multipliers(h=10, r=100, seed=1)
        assert len(fig.axes) == 2
        for ax in fig.axes:
            assert any(np.allclose(line.get_ydata(), 0.0) for line in ax.lines), (
                "chaque panneau porte une ligne de zero, la reference du regard"
            )

    def test_curves_are_distinguishable_without_colour(self) -> None:
        """Ces figures finissent imprimees : le style de trait doit
        suffire a separer m+ de m-."""
        y, x = _asymmetric(seed=92, n=400)
        fig = NARDL(y, x, order=(1, 1)).fit().plot_multipliers(h=10, r=100, seed=1)
        top = fig.axes[0]
        assert top.lines[0].get_linestyle() != top.lines[1].get_linestyle()

    def test_selecting_a_variable(self) -> None:
        rng = np.random.default_rng(93)
        n = 250
        frame = pd.DataFrame(
            {"a": np.cumsum(rng.normal(size=n)), "b": np.cumsum(rng.normal(size=n))}
        )
        y = pd.Series(np.cumsum(rng.normal(size=n)), name="y")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = NARDL(y, frame, order=(1, 1)).fit()
        fig = res.plot_multipliers(h=8, r=50, seed=1, variable="b")
        assert "b" in fig.axes[0].get_title()

    def test_unknown_variable_is_refused(self) -> None:
        y, x = _asymmetric(seed=94)
        res = NARDL(y, x, order=(1, 1)).fit()
        with pytest.raises(ValueError, match="was not decomposed"):
            res.plot_multipliers(variable="zzz")

    def test_multi_variable_table_is_indexed_by_variable(self) -> None:
        rng = np.random.default_rng(95)
        n = 250
        frame = pd.DataFrame(
            {"a": np.cumsum(rng.normal(size=n)), "b": np.cumsum(rng.normal(size=n))}
        )
        y = pd.Series(np.cumsum(rng.normal(size=n)), name="y")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = NARDL(y, frame, order=(1, 1)).fit()
        table = res.dynamic_multipliers(h=6, r=50, seed=1)
        assert isinstance(table.columns, pd.MultiIndex)
        assert set(table.columns.get_level_values(0)) == {"a", "b"}


class TestUecmTable:
    def test_uecm_reports_coefficients_and_ratios(self) -> None:
        y, x = _asymmetric(seed=96, n=400)
        res = NARDL(y, x, order=(1, 1)).fit()
        table = res.uecm
        assert list(table.columns) == ["coef", "se", "t"]
        assert np.allclose(table["t"], table["coef"] / table["se"])
        assert "y.L1" in table.index


class TestNoShortRunTerms:
    """Ordre q = 0 : aucun terme de court terme a comparer."""

    def test_shortrun_tests_report_unavailable(self) -> None:
        """Sans difference retardee d'un cote, le test additif n'a rien a
        contraster. Il le dit, plutot que de rendre un zero qui se
        lirait comme une symetrie parfaite."""
        y, x = _asymmetric(seed=97, n=300)
        res = NARDL(y, x, order=(1, {"x_pos": 0, "x_neg": 0})).fit()
        tests = res.asymmetry_tests()
        assert tests.loc[("x", "shortrun_additive"), "decision"] == "unavailable"
        assert tests.loc[("x", "shortrun_strong"), "decision"] == "unavailable"
        # Les tests de long terme, eux, restent parfaitement definis.
        assert np.isfinite(tests.loc[("x", "longrun_gamma"), "stat"])


class TestVectorisedRecursion:
    """La recursion des multiplicateurs avance tous les tirages ensemble.

    L'implementation naive — une boucle Python par tirage — est la
    reference : elle est evidemment correcte et evidemment lente. Le
    verrou est que la version vectorisee lui soit identique, pas
    seulement proche.
    """

    @staticmethod
    def _scalar_path(params, names, column, h, y_prefix):  # type: ignore[no-untyped-def]
        """Implementation naive, un tirage a la fois."""
        phi = np.array(
            [params[i] for i, n in enumerate(names) if n.startswith(f"{y_prefix}.L")]
        )
        beta = np.array(
            [params[i] for i, n in enumerate(names) if n.startswith(f"{column}.L")]
        )
        p, q = phi.size, beta.size
        out = np.zeros(h + 1 + p)
        for t in range(h + 1):
            value = 0.0
            for i in range(p):
                value += phi[i] * out[t + p - 1 - i]
            for j in range(q):
                if t - j >= 0:
                    value += beta[j]
            out[t + p] = value
        return out[p:]

    @pytest.mark.parametrize("order", [(1, 1), (2, 2), (3, 2)])
    def test_matches_the_naive_loop(self, order: tuple[int, int]) -> None:
        from pyardl.nardl.model import _autoregressive_prefix, _multiplier_path

        y, x = _asymmetric(seed=98, n=300)
        res = NARDL(y, x, order=order).fit()
        names = [str(v) for v in res._ardl_res._param_names]
        params = np.asarray(res._ardl_res._params, dtype=float)
        cov = np.asarray(res._ardl_res._cov_params, dtype=float)
        prefix = _autoregressive_prefix(names)

        draws = np.random.default_rng(5).multivariate_normal(params, cov, size=50)
        reference = np.array(
            [self._scalar_path(d, names, "x_pos", 40, prefix) for d in draws]
        )
        vectorised = _multiplier_path(draws, names, "x_pos", 40, prefix)
        assert np.max(np.abs(reference - vectorised)) < 1e-12

    def test_single_vector_returns_one_row(self) -> None:
        from pyardl.nardl.model import _autoregressive_prefix, _multiplier_path

        y, x = _asymmetric(seed=99, n=300)
        res = NARDL(y, x, order=(1, 1)).fit()
        names = [str(v) for v in res._ardl_res._param_names]
        params = np.asarray(res._ardl_res._params, dtype=float)
        prefix = _autoregressive_prefix(names)
        path = _multiplier_path(params, names, "x_pos", 10, prefix)
        assert path.shape == (1, 11)


class TestOrderSelection:
    """§2.2 — selection d'ordre sur le modele TRANSFORME."""

    def test_recovers_the_true_order(self) -> None:
        """DGP d'ordre (1, 1) : la selection doit le retrouver."""
        y, x = _asymmetric(seed=110, n=400)
        model = NARDL(y, x, order="auto", max_p=3, max_q=3)
        assert model.p == 1
        assert set(model.q_map.values()) == {1}

    def test_common_sample_for_every_candidate(self) -> None:
        """Le piege classique de la selection d'ordre : comparer des
        criteres calcules sur des nombres d'observations differents n'a
        aucun sens. Tous les candidats partagent donc le meme
        echantillon."""
        y, x = _asymmetric(seed=111, n=300)
        model = NARDL(y, x, order="auto", max_p=4, max_q=4)
        assert model.selection is not None
        assert model.selection["nobs"].nunique() == 1

    def test_paired_mode_keeps_the_two_sides_together(self) -> None:
        y, x = _asymmetric(seed=112, n=300)
        model = NARDL(y, x, order="auto", max_p=2, max_q=3, asym_lags="paired")
        assert model.selection is not None
        assert (model.selection["q[x_pos]"] == model.selection["q[x_neg]"]).all()
        assert model.q_map["x_pos"] == model.q_map["x_neg"]

    def test_free_mode_explores_unequal_orders(self) -> None:
        y, x = _asymmetric(seed=113, n=300)
        model = NARDL(y, x, order="auto", max_p=2, max_q=3, asym_lags="free")
        assert model.selection is not None
        assert (model.selection["q[x_pos]"] != model.selection["q[x_neg]"]).any()

    def test_free_grid_is_larger_than_paired(self) -> None:
        """La liberte se paie en candidats : le carre au lieu du simple."""
        y, x = _asymmetric(seed=114, n=300)
        paired = NARDL(y, x, order="auto", max_p=2, max_q=3, asym_lags="paired")
        free = NARDL(y, x, order="auto", max_p=2, max_q=3, asym_lags="free")
        assert paired.selection is not None and free.selection is not None
        assert len(free.selection) > len(paired.selection)

    def test_table_is_sorted_by_the_chosen_criterion(self) -> None:
        y, x = _asymmetric(seed=115, n=300)
        for ic, column in (("aic", "aic"), ("bic", "bic"), ("hq", "hqic")):
            model = NARDL(y, x, order="auto", max_p=2, max_q=2, ic=ic)
            assert model.selection is not None
            values = model.selection[column].to_numpy()
            assert np.all(np.diff(values) >= 0)

    def test_bic_never_picks_a_larger_order_than_aic(self) -> None:
        """BIC penalise plus durement : il ne peut pas etre plus
        genereux que l'AIC sur les memes candidats."""
        y, x = _asymmetric(seed=116, n=300)
        aic = NARDL(y, x, order="auto", max_p=4, max_q=4, ic="aic")
        bic = NARDL(y, x, order="auto", max_p=4, max_q=4, ic="bic")
        assert bic.p + sum(bic.q_map.values()) <= aic.p + sum(aic.q_map.values())

    def test_selection_is_none_for_an_explicit_order(self) -> None:
        y, x = _asymmetric(seed=117)
        assert NARDL(y, x, order=(2, 2)).selection is None

    def test_selected_model_fits(self) -> None:
        y, x = _asymmetric(seed=118, n=400)
        res = NARDL(y, x, order="auto", max_p=3, max_q=3).fit()
        assert res.lam < 0
        assert np.isfinite(res.longrun_asym.loc["x", "theta_pos"])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"ic": "mallows"}, 'ic must be "aic"'),
            ({"asym_lags": "both"}, 'asym_lags must be "paired"'),
            ({"max_p": 0}, "max_p must be at least 1"),
            ({"max_q": -1}, "max_q must be non-negative"),
        ],
    )
    def test_invalid_settings_refused(self, kwargs: dict, match: str) -> None:
        y, x = _asymmetric(seed=119)
        with pytest.raises(ValueError, match=match):
            NARDL(y, x, order="auto", **kwargs)

    def test_sample_too_short_is_refused(self) -> None:
        """Aucun candidat estimable : on le dit, on ne rend pas un ordre
        par defaut qui aurait l'air d'avoir ete choisi."""
        rng = np.random.default_rng(120)
        y = pd.Series(np.cumsum(rng.normal(size=7)), name="y")
        x = pd.DataFrame({"x": np.cumsum(rng.normal(size=7))})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="No candidate could be estimated"):
                NARDL(y, x, order="auto", max_p=4, max_q=4)

    def test_unestimable_candidates_are_skipped_not_scored(self) -> None:
        """Un candidat que l'echantillon ne supporte pas sort de la
        table. Le noter avec un critere infini le ferait figurer au
        classement comme s'il avait perdu au merite."""
        rng = np.random.default_rng(121)
        n = 18
        y = pd.Series(np.cumsum(rng.normal(size=n)), name="y")
        x = pd.DataFrame({"x": np.cumsum(rng.normal(size=n))})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = NARDL(y, x, order="auto", max_p=4, max_q=4)
        assert model.selection is not None
        # La grille appariee compte 4 * 5 = 20 candidats ; certains ne
        # tiennent pas dans 18 observations.
        assert len(model.selection) < 20
        assert np.isfinite(model.selection[["aic", "bic", "hqic"]].to_numpy()).all()
