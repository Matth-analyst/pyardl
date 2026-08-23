"""Spec 18 §4 — plan de tests du QARDL.

Le verrou de l'estimateur est dans test_estimate.py. Ici on verifie ce
que le modele ajoute : le meme design que le reste de la bibliotheque,
theta(tau) et son garde-fou, les tests joints, et la composition avec la
decomposition asymetrique.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.qardl import QARDL

TAUS = (0.25, 0.5, 0.75)


def _homogeneous(seed: int, n: int = 250, theta: float = 1.0, lam: float = -0.4):  # type: ignore[no-untyped-def]
    """Memes coefficients a tous les quantiles : theta(tau) est plat."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] + lam * (y[t - 1] - theta * x[t - 1]) + rng.normal(scale=0.5)
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


def _no_relation(seed: int, n: int = 250):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    return (
        pd.Series(np.cumsum(rng.normal(size=n)), name="y"),
        pd.DataFrame({"x": np.cumsum(rng.normal(size=n))}),
    )


class TestSharedDesign:
    """VERROU — le QARDL estime LE modele de la bibliotheque.

    Si le design differait de celui du test des bornes, un QARDL a la
    mediane et un ARDL ne seraient pas deux lectures d'une meme
    specification mais deux modeles voisins, et toute comparaison entre
    eux serait trompeuse.
    """

    def test_columns_match_the_bounds_test_design(self) -> None:
        from pyardl.bounds.pss import _estimate_uecm

        y, x = _homogeneous(seed=0)
        res = QARDL(y, x, order=(2, 2), taus=(0.5,)).fit(inference="kernel")
        reference = _estimate_uecm(y.to_numpy(), x.to_numpy(), ("x",), "y", 2, (2,), 3)
        assert list(res.names) == list(reference.names)
        assert res.nobs == reference.nobs

    @pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
    def test_every_deterministic_case_builds(self, case: int) -> None:
        y, x = _homogeneous(seed=1)
        res = QARDL(y, x, order=(1, 1), taus=(0.5,), case=case).fit(inference="kernel")
        assert res.coefficients.shape[0] == 1


class TestRecovery:
    """§4.1 et §4.2 — ce que les estimations doivent retrouver."""

    def test_theta_is_flat_on_a_homogeneous_dgp(self) -> None:
        y, x = _homogeneous(seed=10, n=400, theta=1.0)
        res = QARDL(y, x, order=(1, 1), taus=(0.25, 0.5, 0.75)).fit(inference="kernel")
        theta = res.longrun()["x"]
        assert np.all(np.abs(theta - 1.0) < 0.25)
        assert theta.max() - theta.min() < 0.25

    def test_median_matches_the_mean_regression_on_gaussian_data(self) -> None:
        """A tau = 0.5 sur des innovations symetriques, la regression
        quantile et les moindres carres estiment la meme chose."""
        from pyardl.bounds import bounds_test

        y, x = _homogeneous(seed=11, n=500)
        res = QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(inference="kernel")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols = bounds_test(y, x, case=3, order=(1, 1))
        lam_q = float(res.lam.iloc[0])
        lam_ols = float(ols.uecm.loc["y.L1", "coef"])
        assert lam_q == pytest.approx(lam_ols, abs=0.15)

    def test_adjustment_is_negative(self) -> None:
        y, x = _homogeneous(seed=12, n=400)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="kernel")
        assert np.all(res.lam.to_numpy() < 0)


class TestLongRun:
    """theta(tau), et le garde-fou sur lambda proche de zero."""

    def test_theta_equals_minus_gamma_over_lambda(self) -> None:
        y, x = _homogeneous(seed=20, n=300)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="kernel")
        coef = res.coefficients
        expected = -coef["x.L1"] / coef["y.L1"]
        assert np.allclose(res.longrun()["x"], expected)

    def test_bands_bracket_the_estimate(self) -> None:
        y, x = _homogeneous(seed=21, n=300)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(
            inference="mbb", n_boot=60, seed=1
        )
        table = res.longrun()
        assert np.all(table["x_lower"] <= table["x"] + 1e-9)
        assert np.all(table["x"] <= table["x_upper"] + 1e-9)

    def test_vanishing_lambda_gives_nan_not_a_huge_number(self) -> None:
        """Un theta lu sur un lambda nul est un artefact de division. La
        bibliotheque le dit au lieu de rendre 1e12."""
        y, x = _homogeneous(seed=22, n=200)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="kernel")
        forged = res._params.copy()
        forged[1, res._index_of(res._fit.lam_name)] = 0.0
        object.__setattr__(res, "_params", forged)
        with pytest.warns(PyardlMethodologyWarning, match="indistinguishable"):
            table = res.longrun()
        assert np.isnan(table["x"].iloc[1])
        assert np.isfinite(table["x"].iloc[0])

    def test_restricting_to_one_variable(self) -> None:
        y, x = _homogeneous(seed=23, n=200)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="kernel")
        assert list(res.longrun("x").columns) == ["x", "x_lower", "x_upper"]


class TestJointTests:
    """§2.4 — les tests qui ont besoin de la loi JOINTE."""

    def test_constancy_needs_the_bootstrap(self) -> None:
        y, x = _homogeneous(seed=30, n=200)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="kernel")
        with pytest.raises(ValueError, match="inference='mbb'"):
            res.wald_constancy()
        with pytest.raises(ValueError, match="inference='mbb'"):
            res.symmetry_test()

    def test_constancy_reports_a_verdict(self) -> None:
        y, x = _homogeneous(seed=31, n=250)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(
            inference="mbb", n_boot=80, seed=1
        )
        row = res.wald_constancy().loc["x"]
        assert row["df"] == len(TAUS) - 1
        assert row["decision"] in ("constant", "varies with tau", "unavailable")

    def test_symmetry_needs_mirror_pairs(self) -> None:
        """Une grille sans paire (tau, 1-tau) ne peut pas repondre : on
        le dit plutot que d'apparier des quantiles qui ne se
        correspondent pas."""
        y, x = _homogeneous(seed=32, n=200)
        res = QARDL(y, x, order=(1, 1), taus=(0.2, 0.5, 0.6)).fit(
            inference="mbb", n_boot=40, seed=1
        )
        with pytest.raises(ValueError, match="mirror pair"):
            res.symmetry_test()

    def test_symmetry_runs_on_a_mirror_grid(self) -> None:
        y, x = _homogeneous(seed=33, n=250)
        res = QARDL(y, x, order=(1, 1), taus=(0.2, 0.5, 0.8)).fit(
            inference="mbb", n_boot=80, seed=1
        )
        row = res.symmetry_test().loc["x"]
        assert row["df"] == 1  # une seule paire miroir : (0.2, 0.8)


class TestCointegrationTest:
    """§2.4b — le t sur lambda(tau), a valeurs critiques bootstrap."""

    def test_cointegrated_data_reject(self) -> None:
        y, x = _homogeneous(seed=40, n=300)
        res = QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(inference="kernel")
        out = res.cointegration_test(tau=0.5, n_boot=60, seed=1)
        assert out["decision"] == "cointegration"
        assert out["t_stat"] < out["cv_5"]

    def test_independent_walks_do_not_reject(self) -> None:
        y, x = _no_relation(seed=41, n=250)
        res = QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(inference="kernel")
        out = res.cointegration_test(tau=0.5, n_boot=60, seed=1)
        assert out["decision"] == "no_cointegration"

    def test_left_tailed(self) -> None:
        """Les valeurs critiques sont dans la queue GAUCHE : rejeter
        exige un lambda negatif, un vrai rappel vers l'equilibre."""
        y, x = _homogeneous(seed=42, n=250)
        res = QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(inference="kernel")
        out = res.cointegration_test(tau=0.5, n_boot=60, seed=1)
        assert out["cv_1"] < out["cv_5"] < out["cv_10"] < 0

    def test_unknown_tau_is_refused(self) -> None:
        y, x = _homogeneous(seed=43, n=200)
        res = QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(inference="kernel")
        with pytest.raises(ValueError, match="was not estimated"):
            res.cointegration_test(tau=0.9)

    def test_reproducible_with_a_seed(self) -> None:
        y, x = _homogeneous(seed=44, n=200)
        res = QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(inference="kernel")
        a = res.cointegration_test(tau=0.5, n_boot=40, seed=7)
        b = res.cointegration_test(tau=0.5, n_boot=40, seed=7)
        assert a["cv_5"] == b["cv_5"]


class TestReproducibility:
    def test_same_seed_same_draws(self) -> None:
        y, x = _homogeneous(seed=50, n=200)
        kwargs = {"inference": "mbb", "n_boot": 40, "seed": 3}
        a = QARDL(y, x, order=(1, 1), taus=TAUS).fit(**kwargs)  # type: ignore[arg-type]
        b = QARDL(y, x, order=(1, 1), taus=TAUS).fit(**kwargs)  # type: ignore[arg-type]
        assert np.array_equal(a._draws, b._draws)

    def test_seed_recorded_when_omitted(self) -> None:
        y, x = _homogeneous(seed=51, n=200)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="mbb", n_boot=30)
        assert isinstance(res.seed, int)
        again = QARDL(y, x, order=(1, 1), taus=TAUS).fit(
            inference="mbb", n_boot=30, seed=res.seed
        )
        assert np.array_equal(res._draws, again._draws)

    def test_block_length_defaults_to_the_cube_root(self) -> None:
        y, x = _homogeneous(seed=52, n=200)
        res = QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(
            inference="mbb", n_boot=20, seed=1
        )
        assert res.block_length == int(np.ceil(res.nobs ** (1 / 3)))


class TestQNARDL:
    """§2.5 — la composition avec la decomposition asymetrique."""

    def test_decomposes_the_named_regressor(self) -> None:
        y, x = _homogeneous(seed=60, n=250)
        res = QARDL(y, x, order=(1, 1), taus=TAUS, asym=["x"]).fit(inference="kernel")
        assert set(res.regressors) == {"x_pos", "x_neg"}
        table = res.longrun()
        assert "x_pos" in table.columns
        assert "x_neg" in table.columns

    def test_unknown_asym_name(self) -> None:
        y, x = _homogeneous(seed=61, n=200)
        with pytest.raises(ValueError, match="are not regressors"):
            QARDL(y, x, order=(1, 1), asym=["oil"])

    def test_partial_asymmetry(self) -> None:
        rng = np.random.default_rng(62)
        n = 250
        frame = pd.DataFrame(
            {"a": np.cumsum(rng.normal(size=n)), "b": np.cumsum(rng.normal(size=n))}
        )
        y = pd.Series(np.cumsum(rng.normal(size=n)), name="y")
        res = QARDL(y, frame, order=(1, 1), taus=(0.5,), asym=["a"]).fit(
            inference="kernel"
        )
        assert set(res.regressors) == {"a_pos", "a_neg", "b"}


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"taus": ()}, "no quantile"),
            ({"taus": (0.0, 0.5)}, "strictly in"),
            ({"taus": (0.5, 0.5)}, "duplicates"),
            ({"case": 6}, "case must be"),
        ],
    )
    def test_bad_settings(self, kwargs: dict, match: str) -> None:
        y, x = _homogeneous(seed=70, n=150)
        with pytest.raises(ValueError, match=match):
            QARDL(y, x, order=(1, 1), **kwargs)

    def test_no_regressor(self) -> None:
        with pytest.raises(ValueError, match="at least one regressor"):
            QARDL(pd.Series(np.arange(30.0), name="y"), None)  # type: ignore[arg-type]

    def test_bad_inference(self) -> None:
        y, x = _homogeneous(seed=71, n=150)
        with pytest.raises(ValueError, match="inference must be"):
            QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(inference="jackknife")  # type: ignore[arg-type]

    def test_too_few_replications(self) -> None:
        y, x = _homogeneous(seed=72, n=150)
        with pytest.raises(ValueError, match="n_boot must be at least 2"):
            QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(inference="mbb", n_boot=1)

    def test_block_longer_than_the_sample(self) -> None:
        y, x = _homogeneous(seed=73, n=150)
        with pytest.raises(ValueError, match="does not fit"):
            QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(
                inference="mbb", n_boot=5, block_length=10_000
            )

    def test_order_dict_must_cover_every_column(self) -> None:
        y, x = _homogeneous(seed=74, n=150)
        with pytest.raises(ValueError, match="missing"):
            QARDL(y, x, order=(1, {"z": 1}), taus=(0.5,))

    def test_unknown_coefficient_name(self) -> None:
        y, x = _homogeneous(seed=75, n=200)
        res = QARDL(y, x, order=(1, 1), taus=(0.5,)).fit(inference="kernel")
        with pytest.raises(KeyError, match="not a coefficient"):
            res._index_of("nope")
        with pytest.raises(KeyError, match="No level term"):
            res._level_name("nope")


class TestSummary:
    def test_reports_the_grid_and_the_tests(self) -> None:
        y, x = _homogeneous(seed=80, n=250)
        text = (
            QARDL(y, x, order=(1, 1), taus=TAUS)
            .fit(inference="mbb", n_boot=60, seed=1)
            .summary()
        )
        assert "QARDL" in text
        assert "lambda" in text
        assert "constancy" in text

    def test_kernel_summary_says_what_is_missing(self) -> None:
        y, x = _homogeneous(seed=81, n=200)
        text = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="kernel").summary()
        assert "joint tests unavailable" in text


class TestPlot:
    """On teste les donnees tracees, pas les pixels."""

    def test_curves_carry_the_estimates(self) -> None:
        y, x = _homogeneous(seed=90, n=250)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(
            inference="mbb", n_boot=40, seed=1
        )
        fig = res.plot_coefficients()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = res.longrun()
        assert np.allclose(fig.axes[0].lines[0].get_ydata(), table["x"].to_numpy())
        assert np.allclose(fig.axes[1].lines[0].get_ydata(), res.lam.to_numpy())

    def test_lambda_panel_can_be_dropped(self) -> None:
        y, x = _homogeneous(seed=91, n=200)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="kernel")
        assert len(res.plot_coefficients(show_lambda=False).axes) == 1

    def test_zero_line_on_every_panel(self) -> None:
        y, x = _homogeneous(seed=92, n=200)
        fig = (
            QARDL(y, x, order=(1, 1), taus=TAUS)
            .fit(inference="kernel")
            .plot_coefficients()
        )
        for ax in fig.axes:
            assert any(np.allclose(line.get_ydata(), 0.0) for line in ax.lines)


class TestDegenerateBranches:
    """Les chemins ou une quantite n'est pas definie, et le disent."""

    def test_delta_se_is_nan_when_lambda_vanishes(self) -> None:
        """Sans denominateur, le ratio n'a pas de variance non plus.
        Lineariser autour de zero renverrait un nombre, et ce nombre
        n'aurait aucun sens."""
        y, x = _homogeneous(seed=100, n=200)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="kernel")
        forged = res._params.copy()
        forged[0, res._index_of(res._fit.lam_name)] = 0.0
        object.__setattr__(res, "_params", forged)
        se = res._delta_se("x")
        assert np.isnan(se[0])
        assert np.isfinite(se[1])

    def test_contrast_test_refuses_an_undefined_point_estimate(self) -> None:
        from pyardl.qardl.model import _contrast_test

        point = np.array([1.0, np.nan, 1.2])
        drawn = np.random.default_rng(0).normal(size=(50, 3))
        stat, dof, pvalue = _contrast_test(point, drawn, None, [(1, 0), (2, 0)])
        assert np.isnan(stat) and np.isnan(pvalue) and dof == 2

    def test_contrast_test_refuses_fewer_draws_than_contrasts(self) -> None:
        """Moins de tirages utilisables que de contrastes : la covariance
        serait singuliere par construction."""
        from pyardl.qardl.model import _contrast_test

        point = np.array([1.0, 1.1, 1.2])
        drawn = np.random.default_rng(0).normal(size=(2, 3))
        stat, _, pvalue = _contrast_test(point, drawn, None, [(1, 0), (2, 0)])
        assert np.isnan(stat) and np.isnan(pvalue)

    def test_order_dict_is_accepted_when_complete(self) -> None:
        y, x = _homogeneous(seed=101, n=200)
        model = QARDL(y, x, order=(2, {"x": 1}), taus=(0.5,))
        assert model.q_tuple == (1,)
        assert model.p == 2

    def test_verdict_reports_unavailable_on_a_missing_pvalue(self) -> None:
        """Un test qui n'a pas pu tourner ne rend pas 'symmetric' par
        defaut : il rend 'unavailable'. Un verdict par defaut se lirait
        comme un resultat."""
        from pyardl.qardl.model import _verdict

        assert _verdict(float("nan"), "asymmetric", "symmetric") == "unavailable"
        assert _verdict(0.01, "asymmetric", "symmetric") == "asymmetric"
        assert _verdict(0.50, "asymmetric", "symmetric") == "symmetric"


class TestCalibration:
    """Ce a quoi la statistique est comparee — un choix mesure."""

    def test_both_calibrations_run(self) -> None:
        y, x = _homogeneous(seed=110, n=250)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(
            inference="mbb", n_boot=80, seed=1
        )
        for cal in ("chi2", "mbb", "null"):
            row = res.wald_constancy(calibration=cal).loc["x"]
            assert np.isfinite(row["stat"])
            assert 0.0 < row["pvalue"] <= 1.0

    def test_the_statistic_does_not_depend_on_the_calibration(self) -> None:
        """Seule la reference change, pas la quantite mesuree."""
        y, x = _homogeneous(seed=111, n=250)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(
            inference="mbb", n_boot=80, seed=1
        )
        a = res.wald_constancy(calibration="chi2").loc["x", "stat"]
        b = res.wald_constancy(calibration="null").loc["x", "stat"]
        assert a == pytest.approx(b)

    def test_bootstrap_pvalue_is_never_exactly_zero(self) -> None:
        """(1 + #)/(B + 1) : B tirages ne resolvent pas plus que
        1/(B+1), et annoncer p = 0 pretendrait le contraire."""
        y, x = _homogeneous(seed=112, n=250)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(
            inference="mbb", n_boot=50, seed=1
        )
        assert res.wald_constancy().loc["x", "pvalue"] >= 1 / 51

    def test_symmetry_accepts_the_calibration_too(self) -> None:
        y, x = _homogeneous(seed=113, n=250)
        res = QARDL(y, x, order=(1, 1), taus=(0.2, 0.5, 0.8)).fit(
            inference="mbb", n_boot=60, seed=1
        )
        assert np.isfinite(res.symmetry_test(calibration="chi2").loc["x", "pvalue"])

    def test_null_draws_are_computed_once(self) -> None:
        """Les deux tests joints partagent les memes tirages : les
        recalculer donnerait deux p-values differentes pour une meme
        donnee, en plus de doubler le cout."""
        y, x = _homogeneous(seed=114, n=200)
        res = QARDL(y, x, order=(1, 1), taus=(0.2, 0.5, 0.8)).fit(
            inference="mbb", n_boot=40, seed=1
        )
        first = res._null_draws()
        second = res._null_draws()
        assert first is not None
        assert first is second

    def test_no_null_draws_without_the_bootstrap(self) -> None:
        y, x = _homogeneous(seed=115, n=200)
        res = QARDL(y, x, order=(1, 1), taus=TAUS).fit(inference="kernel")
        assert res._null_draws() is None

    def test_null_calibration_needs_its_draws(self) -> None:
        """Sans tirages sous la nulle, la p-value n'est pas calculable :
        on rend NaN plutot qu'un nombre issu d'une autre reference."""
        from pyardl.qardl.model import _contrast_test

        point = np.array([1.0, 1.1, 1.2])
        drawn = np.random.default_rng(0).normal(size=(60, 3))
        stat, _, pvalue = _contrast_test(point, drawn, None, [(1, 0), (2, 0)], "null")
        assert np.isfinite(stat) and np.isnan(pvalue)

    def test_too_few_null_draws_gives_no_pvalue(self) -> None:
        from pyardl.qardl.model import _contrast_test

        point = np.array([1.0, 1.1, 1.2])
        drawn = np.random.default_rng(0).normal(size=(60, 3))
        null = np.random.default_rng(1).normal(size=(2, 3))
        stat, _, pvalue = _contrast_test(point, drawn, null, [(1, 0), (2, 0)], "null")
        assert np.isfinite(stat) and np.isnan(pvalue)
