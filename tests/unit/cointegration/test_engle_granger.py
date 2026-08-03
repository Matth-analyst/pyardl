"""Spec 06 §5 — plan de tests Engle-Granger.

Verrou du module (§5.2) : la statistique et la p-value doivent coïncider
avec `statsmodels.tsa.stattools.coint` à 1e-8. Le test isole la
STATISTIQUE de la convention de sélection de retards, en imposant le
même max_lags des deux côtés : sinon on mesurerait un écart de règle
d'arrondi et non un écart de calcul.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from pyardl.cointegration import engle_granger
from pyardl.critical_values.mackinnon import (
    EG_MAX_VARS,
    eg_critical_values,
    eg_pvalue,
)
from pyardl.exceptions import PyardlMethodologyWarning


def _schwert_ceil(n: int) -> int:
    """max_lags selon la convention statsmodels (arrondi au-dessus)."""
    return int(np.ceil(12.0 * (n / 100.0) ** 0.25))


def _cointegrated(n: int, k: int, seed: int, sigma: float = 1.0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, k)), axis=0)
    y = x @ np.ones(k) + sigma * rng.standard_normal(n)
    return y, x


def _independent_walks(n: int, k: int, seed: int):  # type: ignore[no-untyped-def]
    """Marches aléatoires indépendantes : H0 exactement vraie."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)), np.cumsum(
        rng.standard_normal((n, k)), axis=0
    )


class TestStatisticLock:
    """§5.2 — verrou : concordance avec statsmodels."""

    @pytest.mark.parametrize("trend", ["c", "ct", "ctt"])
    @pytest.mark.parametrize("k", [1, 2, 3])
    @pytest.mark.parametrize("n", [100, 200])
    def test_matches_statsmodels_coint_1e8(self, trend: str, k: int, n: int) -> None:
        y, x = _cointegrated(n, k, seed=n + k + len(trend), sigma=1.5)
        mine = engle_granger(y, x, trend=trend, max_lags=_schwert_ceil(n))
        ref = coint(y, x, trend=trend, autolag="aic")
        assert abs(mine.stat - ref[0]) < 1e-8
        assert abs(mine.pvalue - ref[1]) < 1e-8

    def test_critical_values_match_statsmodels(self) -> None:
        y, x = _cointegrated(200, 2, seed=5)
        mine = engle_granger(y, x, trend="c", max_lags=_schwert_ceil(200))
        ref = coint(y, x, trend="c", autolag="aic")
        for i, alpha in enumerate((0.01, 0.05, 0.10)):
            # statsmodels evalue la surface a nobs-1 ; nous a nobs.
            assert abs(mine.critical_values[alpha] - ref[2][i]) < 0.02

    def test_maxlag_convention_documented(self) -> None:
        """La règle de Schwert : nous arrondissons EN DESSOUS.

        Schwert (1989) définit floor(12 (T/100)^{1/4}) ; statsmodels
        arrondit au-dessus. Sur une quasi-égalité de l'AIC, ce retard de
        plus déplace l'échantillon commun et peut renverser le choix —
        d'où le max_lags explicite dans les tests de concordance.
        """
        from pyardl.unitroot.gls import _default_max_lags

        for n in (100, 200, 500):
            assert _default_max_lags(n) == int(np.floor(12.0 * (n / 100.0) ** 0.25))
            assert _default_max_lags(n) <= _schwert_ceil(n)

    def test_first_step_residuals_match_ols(self) -> None:
        """§5.3 — les résidus d'étape 1 sont ceux d'une OLS."""
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tsa.tsatools import add_trend

        y, x = _cointegrated(200, 2, seed=6)
        mine = engle_granger(y, x, trend="c")
        ref = OLS(y, add_trend(x, trend="c", prepend=False)).fit().resid
        assert np.max(np.abs(mine.resid.to_numpy() - ref)) < 1e-10


class TestSizeAndPower:
    """§5.1 — taille sous H0, puissance sous cointégration."""

    @pytest.mark.fast_mc
    def test_size_under_independent_walks(self) -> None:
        """Marches indépendantes -> taux de rejet proche du nominal."""
        rejections = 0
        reps = 200
        for seed in range(reps):
            y, x = _independent_walks(150, 1, seed=40_000 + seed)
            out = engle_granger(y, x)
            rejections += out.decision(0.05) == "cointegration"
        assert 0.01 <= rejections / reps <= 0.12

    @pytest.mark.fast_mc
    def test_power_under_cointegration(self) -> None:
        rejections = 0
        reps = 150
        for seed in range(reps):
            y, x = _cointegrated(150, 1, seed=41_000 + seed, sigma=1.0)
            rejections += engle_granger(y, x).decision(0.05) == "cointegration"
        assert rejections / reps > 0.80

    @pytest.mark.slow
    def test_size_full(self) -> None:
        rejections = 0
        reps = 1000
        for seed in range(reps):
            y, x = _independent_walks(150, 1, seed=42_000 + seed)
            rejections += engle_granger(y, x).decision(0.05) == "cointegration"
        assert 0.02 <= rejections / reps <= 0.10

    def test_more_regressors_shift_critical_values_left(self) -> None:
        """Estimer plus de coefficients à l'étape 1 durcit le test.

        C'est toute la raison d'être des surfaces de MacKinnon : utiliser
        les valeurs de Dickey-Fuller ici sur-rejetterait, d'autant plus
        que k est grand.
        """
        cvs = [eg_critical_values(n_vars=k, nobs=200)[0.05] for k in (2, 3, 4, 5)]
        assert all(a > b for a, b in zip(cvs[:-1], cvs[1:], strict=True))


class TestCriticalValues:
    """§2.3 — couverture et provenance des surfaces."""

    def test_levels_ordered(self) -> None:
        cv = eg_critical_values(n_vars=2, nobs=200)
        assert cv[0.01] < cv[0.05] < cv[0.10]

    def test_trend_case_more_negative(self) -> None:
        assert (
            eg_critical_values(2, 200, "ct")[0.05]
            < eg_critical_values(2, 200, "c")[0.05]
        )

    def test_no_trend_case_unavailable(self) -> None:
        """trend='n' : pas de surface publiée -> NaN + warning, jamais
        une valeur empruntée à un cas voisin."""
        with pytest.warns(PyardlMethodologyWarning, match="no cointegration"):
            cv = eg_critical_values(2, 200, "n")
        assert all(np.isnan(v) for v in cv.values())

    def test_decision_refuses_without_critical_value(self) -> None:
        y, x = _cointegrated(150, 1, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = engle_granger(y, x, trend="n")
        with pytest.raises(ValueError, match="No critical value"):
            res.decision(0.05)

    def test_coverage_errors(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            eg_critical_values(2, 200, "q")
        with pytest.raises(ValueError, match="outside the published"):
            eg_critical_values(EG_MAX_VARS + 1, 200)
        with pytest.raises(ValueError, match="outside the published"):
            eg_pvalue(-3.0, 0)

    def test_pvalue_monotone_in_statistic(self) -> None:
        values = [eg_pvalue(s, 2) for s in (-5.0, -4.0, -3.0, -2.0)]
        assert all(a < b for a, b in zip(values[:-1], values[1:], strict=True))

    def test_pvalue_matches_critical_value(self) -> None:
        """Cohérence interne : la p-value au seuil vaut le seuil."""
        for n_vars in (2, 3, 4):
            cv = eg_critical_values(n_vars, 100_000)[0.05]
            assert abs(eg_pvalue(cv, n_vars) - 0.05) < 0.005


class TestResults:
    """§3 — objet de résultats."""

    def test_longrun_params_named(self) -> None:
        y = pd.Series(
            np.cumsum(np.random.default_rng(8).standard_normal(200)), name="y"
        )
        x = pd.DataFrame(
            {"a": np.cumsum(np.random.default_rng(9).standard_normal(200))}
        )
        res = engle_granger(y, x)
        assert list(res.longrun_params.index) == ["a", "const"]

    def test_ecm_option(self) -> None:
        y, x = _cointegrated(200, 2, seed=10)
        res = engle_granger(y, x, fit_ecm=True)
        assert res.ecm is not None
        assert "ecm.L1" in res.ecm.index
        # La vitesse d'ajustement doit être négative sous cointégration.
        assert res.ecm.loc["ecm.L1", "coef"] < 0
        assert res.ecm.loc["ecm.L1", "pvalue"] < 0.05

    def test_ecm_absent_by_default(self) -> None:
        y, x = _cointegrated(150, 1, seed=11)
        assert engle_granger(y, x).ecm is None

    def test_summary_warns_against_first_step_inference(self) -> None:
        y, x = _cointegrated(200, 1, seed=12)
        text = engle_granger(y, x).summary()
        assert "Engle-Granger" in text
        assert "no inference" in text
        assert "H0: no cointegration" in text

    def test_resid_keeps_index(self) -> None:
        idx = pd.period_range("1980Q1", periods=200, freq="Q")
        y = pd.Series(
            np.cumsum(np.random.default_rng(13).standard_normal(200)), index=idx
        )
        x = pd.DataFrame(
            {"a": np.cumsum(np.random.default_rng(14).standard_normal(200))}, index=idx
        )
        res = engle_granger(y, x)
        assert res.resid.index.equals(idx)


class TestNormalisationMatters:
    """§4 — la limite structurelle, démontrée plutôt qu'affirmée."""

    def test_swapping_y_and_x_changes_the_test(self) -> None:
        """Régresser y sur x et x sur y sont deux tests différents.

        Rien dans la méthode ne dit lequel est le bon. C'est la première
        des trois raisons pour lesquelles le bounds test existe.
        """
        gaps = []
        for seed in range(15):
            y, x = _cointegrated(150, 1, seed=600 + seed, sigma=2.0)
            forward = engle_granger(y, x).stat
            backward = engle_granger(x[:, 0], y[:, None]).stat
            gaps.append(abs(forward - backward))
        assert max(gaps) > 0.1

    def test_perfect_fit_warns(self) -> None:
        rng = np.random.default_rng(15)
        x = np.cumsum(rng.standard_normal((150, 1)), axis=0)
        y = 2.0 * x[:, 0]  # relation exacte
        with pytest.warns(PyardlMethodologyWarning, match="almost perfectly"):
            engle_granger(y, x)


class TestValidation:
    def test_no_regressor_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one regressor"):
            engle_granger(np.arange(50.0), None)  # type: ignore[arg-type]

    def test_bad_trend(self) -> None:
        y, x = _cointegrated(100, 1, seed=16)
        with pytest.raises(ValueError, match="must be 'n', 'c', 'ct' or 'ctt'"):
            engle_granger(y, x, trend="q")  # type: ignore[arg-type]
