"""Spec 10 — plan de tests §7 (hors verrou §7.2, cf.
test_pss_wald_equivalence.py) : concordance statsmodels, statut à trois
états, Monte Carlo taille/puissance, cas limites.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.ardl import UECM as SM_UECM

from ardlpy.bounds import bounds_test
from ardlpy.bounds.pss import _classify
from ardlpy.exceptions import ArdlpyMethodologyWarning, DegenerateCaseWarning


def _dgp_cointegrated(
    seed: int, n: int = 200, k: int = 1
) -> tuple[pd.Series, pd.DataFrame]:
    """DGP cointégré : y rappelé vers x (lambda = -0.3)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=(n, k)), axis=0)
    y = np.zeros(n)
    for t in range(1, n):
        eq = 1.0 + x[t - 1].sum()
        y[t] = y[t - 1] - 0.3 * (y[t - 1] - eq) + rng.normal(scale=0.5)
    return (
        pd.Series(y, name="y"),
        pd.DataFrame({f"x{j}": x[:, j] for j in range(k)}),
    )


def _dgp_no_cointegration(
    seed: int, n: int = 200, k: int = 1
) -> tuple[pd.Series, pd.DataFrame]:
    """DGP sous H0 : y marche aléatoire indépendante des x I(1)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=(n, k)), axis=0)
    y = np.cumsum(rng.normal(size=n))
    return (
        pd.Series(y, name="y"),
        pd.DataFrame({f"x{j}": x[:, j] for j in range(k)}),
    )


class TestStatsmodelsConcordance:
    """Spec 10 §7.3 : mêmes F que statsmodels UECM.bounds_test, 5 cas."""

    @pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
    def test_f_stat_matches_statsmodels(self, case: int) -> None:
        y, x = _dgp_cointegrated(seed=100 + case)
        p, q = 2, 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(y, x, case=case, order=(p, q))
        # statsmodels : trend selon le cas, le test refit en interne
        trend = {1: "n", 2: "c", 3: "c", 4: "ct", 5: "ct"}[case]
        sm_fit = SM_UECM(y, lags=p, exog=x, order={"x0": q}, trend=trend).fit()
        sm_bt = sm_fit.bounds_test(case=case)
        assert res.f_stat == pytest.approx(sm_bt.stat, abs=1e-8)


class TestThreeStateDecision:
    """Spec 10 §7.4 : statut inconclusive explicite, jamais un booléen."""

    def test_classify_f_three_states(self) -> None:
        lower, upper = 4.94, 5.73  # cas III, k=1, 5%
        assert _classify(6.0, lower, upper, left_tail=False) == "cointegration"
        assert _classify(4.0, lower, upper, left_tail=False) == "no_cointegration"
        assert _classify(5.3, lower, upper, left_tail=False) == "inconclusive"

    def test_classify_t_left_tail_three_states(self) -> None:
        lower, upper = -2.86, -3.22  # cas III, k=1, 5%
        assert _classify(-3.5, lower, upper, left_tail=True) == "cointegration"
        assert _classify(-2.0, lower, upper, left_tail=True) == "no_cointegration"
        assert _classify(-3.0, lower, upper, left_tail=True) == "inconclusive"

    def test_decisions_are_strings_not_bool(self) -> None:
        y, x = _dgp_cointegrated(seed=0)
        res = bounds_test(y, x, case=3, order=(2, 1))
        assert isinstance(res.decision_f, str)
        assert res.decision_f in (
            "cointegration",
            "no_cointegration",
            "inconclusive",
        )
        assert not isinstance(res.decision_f, bool)

    def test_cointegrated_dgp_detected(self) -> None:
        y, x = _dgp_cointegrated(seed=1, n=400)
        res = bounds_test(y, x, case=3, order=(2, 1))
        assert res.decision_f == "cointegration"
        assert res.decision_t == "cointegration"

    def test_no_cointegration_dgp(self) -> None:
        y, x = _dgp_no_cointegration(seed=2, n=400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(y, x, case=3, order=(2, 1))
        assert res.decision_f in ("no_cointegration", "inconclusive")


class TestCasesAndGuards:
    def test_t_decision_none_for_cases_ii_iv_with_warning(self) -> None:
        y, x = _dgp_cointegrated(seed=3)
        for case in (2, 4):
            with pytest.warns(ArdlpyMethodologyWarning, match="ne tabule pas"):
                res = bounds_test(y, x, case=case, order=(2, 1))
            assert res.decision_t is None
            assert np.isfinite(res.t_stat)  # la stat est calculée quand même

    def test_positive_lambda_warns_and_no_cointegration_t(self) -> None:
        """t unilatéral gauche exige lambda_hat < 0."""
        rng = np.random.default_rng(4)
        n = 150
        x = np.cumsum(rng.normal(size=n))
        y = np.zeros(n)
        for t in range(1, n):  # dérive explosive : lambda_hat > 0
            y[t] = 1.03 * y[t - 1] + 0.1 * x[t] + rng.normal(scale=0.3)
        with pytest.warns(DegenerateCaseWarning, match="lambda_hat"):
            res = bounds_test(
                pd.Series(y, name="y"),
                pd.DataFrame({"x": x}),
                case=3,
                order=(1, 1),
            )
        assert res.decision_t == "no_cointegration"

    def test_q_zero_regressor_enters_level_vector_contemporaneously(
        self,
    ) -> None:
        """Note spec 03 -> spec 10 (docs/QUESTIONS.md) : q_j=0 -> niveau
        contemporain x_{j,t} dans le vecteur testé."""
        y, x = _dgp_cointegrated(seed=5, k=2)
        res = bounds_test(y, x, case=3, order=(2, {"x0": 1, "x1": 0}))
        assert "x1.L0" in res.uecm.index  # contemporain
        assert "x1.L1" not in res.uecm.index
        assert "D.x1.L0" not in res.uecm.index  # pas de Δ distinct
        assert np.isfinite(res.f_stat)

    def test_cv_source_not_implemented(self) -> None:
        y, x = _dgp_cointegrated(seed=6)
        with pytest.raises(NotImplementedError, match="spec 13"):
            bounds_test(y, x, case=3, order=(1, 1), cv_source="kripfganz")
        with pytest.raises(NotImplementedError, match="spec 12"):
            bounds_test(y, x, case=3, order=(1, 1), cv_source="narayan")

    def test_bad_case_raises(self) -> None:
        y, x = _dgp_cointegrated(seed=7)
        with pytest.raises(ValueError, match="case"):
            bounds_test(y, x, case=0, order=(1, 1))

    def test_order_none_triggers_selection(self) -> None:
        y, x = _dgp_cointegrated(seed=8, n=300)
        res = bounds_test(y, x, case=3, ic="bic", max_p=3, max_q=2)
        p, q = res.order
        assert 1 <= p <= 3
        assert 0 <= q["x0"] <= 2

    def test_autocorrelation_guard_fires(self) -> None:
        """Spec 09 §2.2 appliqué au bounds test."""
        rng = np.random.default_rng(9)
        n = 300
        x = np.cumsum(rng.normal(size=n))
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.8 * u[t - 1] + rng.normal(scale=0.3)
        y = pd.Series(0.5 * x + u, name="y")
        with pytest.warns(ArdlpyMethodologyWarning, match="Ljung-Box"):
            bounds_test(y, pd.DataFrame({"x": x}), case=3, order=(1, 0))


class TestPresentation:
    def test_summary_contains_key_elements(self) -> None:
        y, x = _dgp_cointegrated(seed=10)
        res = bounds_test(y, x, case=3, order=(2, 1))
        s = res.summary()
        for token in ("cas 3", "F_overall", "t_BDM", "F_I0", "F_I1"):
            assert token in s

    def test_bounds_table_has_three_levels(self) -> None:
        y, x = _dgp_cointegrated(seed=11)
        res = bounds_test(y, x, case=3, order=(2, 1))
        assert list(res.bounds.index) == [0.10, 0.05, 0.01]

    def test_diagnostics_frame(self) -> None:
        y, x = _dgp_cointegrated(seed=12)
        res = bounds_test(y, x, case=3, order=(2, 1))
        diag = res.diagnostics()
        assert "Jarque-Bera" in diag.index


def _mc_rejection_rates(
    n_rep: int, n: int, case: int, seed0: int
) -> tuple[float, float]:
    """Taux de rejet (borne I(1), 5 %) sous H0 et sous cointégration."""
    rej_h0 = rej_h1 = 0
    for rep in range(n_rep):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, x = _dgp_no_cointegration(seed=seed0 + rep, n=n)
            res0 = bounds_test(y, x, case=case, order=(2, 1))
            if res0.decision_f == "cointegration":
                rej_h0 += 1
            y, x = _dgp_cointegrated(seed=seed0 + 50_000 + rep, n=n)
            res1 = bounds_test(y, x, case=case, order=(2, 1))
            if res1.decision_f == "cointegration":
                rej_h1 += 1
    return rej_h0 / n_rep, rej_h1 / n_rep


@pytest.mark.fast_mc
@pytest.mark.parametrize("case", [3, 5])
def test_size_and_power_fast(case: int) -> None:
    """Spec 10 §7.1 (version CI, 200 réplications) : taille aux bornes
    cohérente (rejet à la borne I(1) <= ~7.5 % sous H0 tout-I(1)) et
    puissance élevée sous cointégration (T=200)."""
    size, power = _mc_rejection_rates(n_rep=200, n=200, case=case, seed0=1000)
    assert size <= 0.075, f"taille empirique {size:.3f} > 7.5 % (cas {case})"
    assert power >= 0.60, f"puissance {power:.3f} < 60 % (cas {case})"


@pytest.mark.slow
@pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
def test_size_and_power_full(case: int) -> None:
    """Spec 10 §7.1 (version complète, 1000 réplications, tous les cas) :
    taille <= 6.5 % à la borne I(1) sous H0 ; puissance croissante en T."""
    size_200, power_200 = _mc_rejection_rates(n_rep=1000, n=200, case=case, seed0=3000)
    assert size_200 <= 0.065, f"taille {size_200:.3f} (cas {case})"
    _, power_400 = _mc_rejection_rates(n_rep=1000, n=400, case=case, seed0=4000)
    assert power_400 >= power_200, "puissance non croissante en T"
    assert power_400 >= 0.85
