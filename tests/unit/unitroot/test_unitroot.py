"""Spec 27 §3 — plan de tests ERS / Ng-Perron.

Verrou du module (§3.3) : la statistique DF-GLS maison doit être
IDENTIQUE à celle du package `arch`. C'est ce verrou qui fixe la
convention de première observation du dé-trending GLS, source d'erreur
n°1 de cette spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.critical_values.ers1996 import dfgls_critical_values
from pyardl.critical_values.ngperron2001 import m_critical_values
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.unitroot import (
    CBAR,
    adf_regression,
    dfgls,
    gls_detrend,
    integration_order,
    ng_perron,
    report,
    select_lags,
)

arch_unitroot = pytest.importorskip("arch.unitroot")


def _random_walk(n: int, seed: int) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def _stationary(n: int, seed: int, rho: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + e[t]
    return y


def _ma_error(n: int, seed: int, theta: float = -0.8) -> np.ndarray:
    """Marche aléatoire à innovations MA(1) NÉGATIVE.

    Le cas qui met en défaut l'ADF standard et motive tout Ng-Perron.
    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 1)
    return np.cumsum(e[1:] + theta * e[:-1])


class TestDFGLSLock:
    """§3.3 — verrou : concordance exacte avec arch."""

    @pytest.mark.parametrize("n", [50, 120, 300])
    @pytest.mark.parametrize("trend", ["c", "ct"])
    @pytest.mark.parametrize("lags", [0, 2, 5])
    def test_statistic_matches_arch_1e8(self, n: int, trend: str, lags: int) -> None:
        """Statistique DF-GLS maison == arch à 1e-8."""
        y = _random_walk(n, seed=n + lags)
        mine = dfgls(y, trend=trend, lags=lags).stat
        reference = arch_unitroot.DFGLS(y, trend=trend, lags=lags).stat
        assert abs(mine - reference) < 1e-8

    def test_detrending_first_observation_convention(self) -> None:
        """La première observation reste AU NIVEAU, pas quasi-différenciée.

        Piège n°1 de la spec : quasi-différencier y_1 déplace la
        statistique de -1.34 à -1.94 sur une marche aléatoire de longueur
        200, largement de quoi renverser une décision. Ce test compare
        les deux conventions et verrouille la bonne.
        """
        y = _random_walk(200, seed=0)
        n = y.size
        alpha = 1.0 + CBAR["c"] / n

        # Convention FAUSSE, écrite explicitement pour être rejetée.
        z = np.ones((n, 1))
        y_bad, z_bad = np.empty(n), np.empty_like(z)
        y_bad[0] = (1 - alpha) * y[0]
        z_bad[0] = (1 - alpha) * z[0]
        y_bad[1:] = y[1:] - alpha * y[:-1]
        z_bad[1:] = z[1:] - alpha * z[:-1]
        beta_bad = np.linalg.lstsq(z_bad, y_bad, rcond=None)[0]
        wrong = y - z @ beta_bad

        good = gls_detrend(y, "c")
        assert not np.allclose(good, wrong)

        t_good = adf_regression(good, 4).tstat
        t_wrong = adf_regression(wrong, 4).tstat
        assert abs(t_good - arch_unitroot.DFGLS(y, trend="c", lags=4).stat) < 1e-8
        assert abs(t_wrong - t_good) > 0.5

    def test_detrended_series_keeps_length(self) -> None:
        y = _random_walk(80, seed=3)
        assert gls_detrend(y, "c").shape == (80,)
        assert gls_detrend(y, "ct").shape == (80,)

    def test_cbar_values(self) -> None:
        assert CBAR["c"] == -7.0
        assert CBAR["ct"] == -13.5


class TestLagSelection:
    """§3.2 — MAIC contre AIC en présence de MA négative."""

    def test_maic_selects_more_lags_than_aic_under_negative_ma(self) -> None:
        """L'expérience clé de Ng-Perron.

        Avec une MA(1) fortement négative, l'AIC sous-sélectionne : c'est
        cette sous-sélection qui produit la distorsion de taille. Le MAIC
        retient des ordres plus riches.
        """
        maic_total = aic_total = 0
        for seed in range(20):
            yd = gls_detrend(_ma_error(200, seed), "c")
            maic_total += select_lags(yd, method="maic")[0]
            aic_total += select_lags(yd, method="aic")[0]
        assert maic_total > aic_total

    def test_maic_reduces_size_distortion(self) -> None:
        """La conséquence pratique : moins de faux rejets.

        Sous H0 (racine unitaire) avec MA(1) négative, l'ADF sur retards
        choisis par AIC sur-rejette massivement. Le MAIC ramène le taux
        vers le nominal. C'est LA raison d'être du critère modifié.
        """
        rej_maic = rej_aic = 0
        reps = 120
        for seed in range(reps):
            y = _ma_error(200, 5000 + seed)
            rej_maic += dfgls(y, method="maic").decision(0.05) == "stationary"
            rej_aic += dfgls(y, method="aic").decision(0.05) == "stationary"
        assert rej_aic > rej_maic
        assert rej_maic / reps < 0.30

    def test_criterion_values_exposed(self) -> None:
        """Le choix est inspectable, pas un oracle."""
        yd = gls_detrend(_random_walk(150, 7), "c")
        chosen, values = select_lags(yd, method="maic", max_lags=6)
        assert set(values) == set(range(7))
        assert chosen == min(values, key=lambda k: values[k])

    def test_common_sample_across_candidates(self) -> None:
        """Tous les candidats sur le MÊME échantillon.

        Comparer des critères calculés sur des nombres d'observations
        différents n'a pas de sens et biaise le choix vers les ordres
        courts — même piège qu'en sélection d'ordre ARDL.
        """
        yd = gls_detrend(_random_walk(120, 11), "c")
        _, v6 = select_lags(yd, method="aic", max_lags=6)
        _, v3 = select_lags(yd, method="aic", max_lags=3)
        # Les valeurs pour k <= 3 diffèrent entre les deux appels
        # justement parce que l'échantillon commun change avec max_lags.
        assert v6[3] != v3[3]

    @pytest.mark.parametrize("method", ["maic", "mbic", "aic", "bic", "t-stat"])
    def test_all_methods_return_valid_order(self, method: str) -> None:
        yd = gls_detrend(_random_walk(200, 13), "c")
        chosen, _ = select_lags(yd, method=method, max_lags=8)
        assert 0 <= chosen <= 8

    def test_fixed_is_not_a_selection_rule(self) -> None:
        yd = gls_detrend(_random_walk(100, 17), "c")
        with pytest.raises(ValueError, match="not a selection rule"):
            select_lags(yd, method="fixed")

    def test_unknown_method_raises(self) -> None:
        yd = gls_detrend(_random_walk(100, 19), "c")
        with pytest.raises(ValueError, match="not available"):
            select_lags(yd, method="nope")  # type: ignore[arg-type]


class TestPower:
    """§3.1 — taille et puissance."""

    @pytest.mark.fast_mc
    def test_size_under_null(self) -> None:
        rej = 0
        reps = 200
        for seed in range(reps):
            rej += (
                dfgls(_random_walk(150, 20_000 + seed), lags=0).decision(0.05)
                == "stationary"
            )
        assert 0.01 <= rej / reps <= 0.12

    @pytest.mark.fast_mc
    def test_more_powerful_than_adf(self) -> None:
        """DF-GLS plus puissant que l'ADF classique, à taille comparable.

        C'est l'apport d'ERS : le dé-trending sous l'alternative locale
        récupère la puissance que la démoyennisation OLS détruit.
        """
        rej_gls = rej_adf = 0
        reps = 150
        for seed in range(reps):
            y = _stationary(120, 30_000 + seed, rho=0.9)
            rej_gls += dfgls(y, lags=0).decision(0.05) == "stationary"
            rej_adf += arch_unitroot.ADF(y, trend="c", lags=0).pvalue < 0.05
        assert rej_gls > rej_adf

    def test_stationary_series_rejects(self) -> None:
        y = _stationary(300, 42, rho=0.3)
        assert dfgls(y).decision(0.05) == "stationary"
        assert ng_perron(y).decision("MZt", 0.05) == "stationary"

    def test_random_walk_does_not_reject(self) -> None:
        y = _random_walk(300, 43)
        assert dfgls(y).decision(0.05) == "unit_root"
        assert ng_perron(y).decision("MZt", 0.05) == "unit_root"


class TestNgPerron:
    """§2.2 — les quatre statistiques M."""

    def test_mzt_equals_mza_times_msb(self) -> None:
        """Identité algébrique exacte, pas approchée."""
        for seed in (1, 2, 3):
            res = ng_perron(_random_walk(200, seed))
            assert res.stats["MZt"] == pytest.approx(
                res.stats["MZa"] * res.stats["MSB"], rel=1e-12
            )

    def test_all_four_present(self) -> None:
        res = ng_perron(_random_walk(150, 4))
        assert set(res.stats) == {"MZa", "MZt", "MSB", "MPT"}

    def test_msb_and_mpt_positive(self) -> None:
        res = ng_perron(_random_walk(200, 5))
        assert res.stats["MSB"] > 0
        assert res.stats["MPT"] > 0

    def test_unknown_statistic_raises(self) -> None:
        res = ng_perron(_random_walk(100, 6))
        with pytest.raises(ValueError, match="not one of"):
            res.decision("MZb")

    def test_summary_lists_four_verdicts(self) -> None:
        text = ng_perron(_random_walk(200, 7)).summary()
        for name in ("MZa", "MZt", "MSB", "MPT"):
            assert name in text
        assert "reject when below" in text

    def test_s2_ar_reported(self) -> None:
        """La variance de long terme est exposée : les quatre
        statistiques n'en valent pas plus qu'elle."""
        res = ng_perron(_random_walk(200, 8))
        assert res.s2_ar > 0
        assert "long-run variance" in res.summary()


class TestCriticalValues:
    """§2.1/§2.2 — provenance et couverture des deux tables."""

    def test_dfgls_cv_monotone_in_level(self) -> None:
        cv = dfgls_critical_values(200, "c")
        assert cv[0.01] < cv[0.05] < cv[0.10]

    def test_dfgls_trend_case_more_negative(self) -> None:
        """Le cas tendance exige une statistique plus extrême."""
        assert (
            dfgls_critical_values(200, "ct")[0.05]
            < (dfgls_critical_values(200, "c")[0.05])
        )

    def test_dfgls_cv_interpolation(self) -> None:
        low = dfgls_critical_values(200, "c")[0.05]
        high = dfgls_critical_values(250, "c")[0.05]
        mid = dfgls_critical_values(220, "c")[0.05]
        assert min(low, high) <= mid <= max(low, high)

    def test_cv_outside_grid_warns(self) -> None:
        with pytest.warns(PyardlMethodologyWarning, match="outside the simulated"):
            dfgls_critical_values(5, "c")
        with pytest.warns(PyardlMethodologyWarning, match="outside the simulated"):
            m_critical_values(100_000, "ct")

    def test_bad_trend_raises(self) -> None:
        with pytest.raises(ValueError, match="'c' or 'ct'"):
            dfgls_critical_values(200, "n")
        with pytest.raises(ValueError, match="'c' or 'ct'"):
            m_critical_values(200, "n")

    def test_mzt_converges_to_dfgls_bounds(self) -> None:
        """Recoupement INTERNE, faute de seconde implémentation.

        MZt partage la loi asymptotique du DF-GLS. Les deux tables étant
        simulées séparément, leur convergence quand T croît est une
        preuve non triviale que ni l'une ni l'autre ne repose sur une
        formule fausse.
        """
        for trend in ("c", "ct"):
            gaps = [
                abs(
                    m_critical_values(n, trend)["MZt"][0.05]
                    - dfgls_critical_values(n, trend)[0.05]
                )
                for n in (100, 500, 2000)
            ]
            assert gaps[0] > gaps[1] > gaps[2]
            assert gaps[-1] < 0.02

    def test_dfgls_cv_agree_with_arch_for_large_samples(self) -> None:
        """Recoupement EXTERNE, là où la seconde source est fiable.

        Les surfaces de réponse de arch s'écartent des nôtres à T = 50
        (voir OBS-6, tranché par une expérience de taille en faveur de
        nos valeurs) ; à partir de T = 100 les deux sources concordent
        dans le bruit de simulation.
        """
        from arch.unitroot.critical_values.dfgls import dfgls_cv_approx

        for trend in ("c", "ct"):
            for n in (100, 200, 500, 1000):
                for j, a in enumerate((0.01, 0.05, 0.10)):
                    mine = dfgls_critical_values(n, trend)[a]
                    ref = float(np.polyval(dfgls_cv_approx[trend][j][::-1], 1.0 / n))
                    assert abs(mine - ref) < 0.02


class TestReport:
    """§2.3 — rapport séquentiel et garde-fou I(2)."""

    def test_i1_series_classified(self) -> None:
        assert integration_order(_random_walk(300, 50))["order"] == "I(1)"

    def test_i0_series_classified(self) -> None:
        assert integration_order(_stationary(300, 51, rho=0.2))["order"] == "I(0)"

    def test_screening_defaults_to_bic(self) -> None:
        """Le défaut du dépistage est BIC, celui des tests ciblés MAIC."""
        import inspect

        from pyardl.unitroot import dfgls as dfgls_fn
        from pyardl.unitroot import ng_perron as np_fn

        assert inspect.signature(report).parameters["method"].default == "bic"
        assert (
            inspect.signature(integration_order).parameters["method"].default == "bic"
        )
        assert inspect.signature(dfgls_fn).parameters["method"].default == "maic"
        assert inspect.signature(np_fn).parameters["method"].default == "maic"

    def test_maic_loses_classification_accuracy_on_clean_data(self) -> None:
        """Arbitrage mesuré entre MAIC et BIC dans le rapport séquentiel.

        MAIC protège contre la MA négative — sa raison d'être — mais
        sur-sélectionne sur données stationnaires, ce qui coûte de la
        puissance à l'étape « différence » et produit de fausses
        suspicions d'I(2). Sur DGP propre, BIC classe mieux. Le test fige
        le constat pour qu'une régression future soit visible.
        """
        rates = {}
        for method in ("maic", "bic"):
            ok = sum(
                integration_order(_random_walk(250, 3000 + s), method=method)["order"]
                == "I(1)"
                for s in range(20)
            )
            rates[method] = ok / 20
        assert rates["bic"] >= rates["maic"]
        assert rates["bic"] >= 0.90

    def test_i2_series_flagged(self) -> None:
        """Série I(2) simulée : double intégration -> suspicion."""
        y = np.cumsum(np.cumsum(np.random.default_rng(52).standard_normal(300)))
        assert integration_order(y)["order"] == "I(2)-suspect"

    def test_report_frame(self) -> None:
        rng = np.random.default_rng(53)
        df = pd.DataFrame(
            {
                "rw": np.cumsum(rng.standard_normal(250)),
                "stat": _stationary(250, 54, rho=0.2),
            }
        )
        table = report(df)
        assert list(table.index) == ["rw", "stat"]
        assert table.loc["rw", "order"] == "I(1)"
        assert table.loc["stat", "order"] == "I(0)"

    def test_report_warns_on_i2(self) -> None:
        rng = np.random.default_rng(55)
        df = pd.DataFrame({"i2": np.cumsum(np.cumsum(rng.standard_normal(300)))})
        with pytest.warns(PyardlMethodologyWarning, match="I\\(2\\) suspected"):
            report(df)

    def test_report_accepts_series(self) -> None:
        s = pd.Series(_random_walk(200, 56), name="x")
        assert report(s).shape[0] == 1

    def test_i0_skips_difference_test(self) -> None:
        """Pas de test sur la différence quand le niveau rejette déjà."""
        out = integration_order(_stationary(300, 57, rho=0.2))
        assert np.isnan(out["dfgls_diff"])
        assert out["decision_diff"] == ""


class TestValidation:
    """Validation des entrées."""

    def test_bad_trend_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            gls_detrend(_random_walk(50, 60), "n")  # type: ignore[arg-type]

    def test_sample_too_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            gls_detrend(np.array([1.0, 2.0]), "c")

    def test_negative_lags_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            dfgls(_random_walk(100, 61), lags=-1)
        with pytest.raises(ValueError, match="non-negative"):
            ng_perron(_random_walk(100, 62), lags=-1)
        with pytest.raises(ValueError, match="non-negative"):
            adf_regression(gls_detrend(_random_walk(100, 63), "c"), -2)

    def test_too_many_lags_for_sample(self) -> None:
        with pytest.raises(ValueError, match="Sample too short"):
            adf_regression(gls_detrend(_random_walk(20, 64), "c"), 15)

    def test_dfgls_summary(self) -> None:
        text = dfgls(_random_walk(200, 65)).summary()
        assert "DF-GLS" in text
        assert "left tail" in text
        assert "unit root" in text

    def test_fixed_lags_reported(self) -> None:
        res = dfgls(_random_walk(150, 66), lags=3)
        assert res.lags == 3
        assert res.lag_method == "fixed"
        assert res.lag_criterion == {}


class TestCoverageGaps:
    """Branches restantes : retards fixes et cas tendance du MPT."""

    def test_ng_perron_fixed_lags(self) -> None:
        res = ng_perron(_random_walk(200, 70), lags=2)
        assert res.lags == 2
        assert res.lag_method == "fixed"
        assert res.lag_criterion == {}

    def test_mpt_trend_branch_differs(self) -> None:
        """Le MPT n'a pas la même formule sous 'c' et sous 'ct'.

        Le terme en y_T change de signe et de coefficient avec le cas
        déterministe ; les deux branches sont donc exercées séparément.
        """
        y = _random_walk(200, 71)
        assert (
            ng_perron(y, trend="c").stats["MPT"]
            != (ng_perron(y, trend="ct").stats["MPT"])
        )
        assert ng_perron(y, trend="ct").stats["MPT"] > 0
