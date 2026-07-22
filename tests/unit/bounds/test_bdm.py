"""Spec 11 (Banerjee-Dolado-Mestre 1998) — plan de tests §3 + ajouts §2 :
unilatéralité stricte, décision jointe F+t, règle « IC sur lambda
seulement après cointégration établie ». Écrits avant l'implémentation
de decision_joint / adjustment.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.ardl import UECM as SM_UECM

from pyardl.bounds import bounds_test
from pyardl.bounds.pss import _joint_decision
from pyardl.exceptions import DegenerateCaseWarning, PyardlMethodologyWarning


def _dgp_cointegrated(seed: int, n: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] - 0.3 * (y[t - 1] - (1.0 + x[t - 1])) + rng.normal(scale=0.5)
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


def _dgp_explosive(seed: int, n: int = 150) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 1.04 * y[t - 1] + 0.1 * x[t] + rng.normal(scale=0.3)
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


def _dgp_no_cointegration(seed: int, n: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return (
        pd.Series(np.cumsum(rng.normal(size=n)), name="y"),
        pd.DataFrame({"x": np.cumsum(rng.normal(size=n))}),
    )


class TestJointDecisionLogic:
    """Spec 11 §2.3 : table de décision jointe sur cas construits."""

    def test_both_reject_is_cointegration(self) -> None:
        assert _joint_decision("cointegration", "cointegration") == "cointegration"

    def test_f_rejects_t_does_not_is_degenerate_suspicion(self) -> None:
        """F rejette mais pas t -> suspicion de dégénérescence de type 1
        (gamma seuls significatifs), renvoi vers la spec 15."""
        assert (
            _joint_decision("cointegration", "no_cointegration")
            == "degenerate_suspicion"
        )
        assert _joint_decision("cointegration", "inconclusive") == (
            "degenerate_suspicion"
        )

    def test_both_fail_is_no_cointegration(self) -> None:
        assert (
            _joint_decision("no_cointegration", "no_cointegration")
            == "no_cointegration"
        )

    def test_other_discordances_are_inconclusive(self) -> None:
        assert _joint_decision("inconclusive", "cointegration") == "inconclusive"
        assert _joint_decision("no_cointegration", "cointegration") == "inconclusive"
        assert _joint_decision("inconclusive", "inconclusive") == "inconclusive"
        assert _joint_decision("inconclusive", "no_cointegration") == "inconclusive"

    def test_t_unavailable_gives_none(self) -> None:
        """Cas II/IV : pas de t tabulé -> pas de décision jointe."""
        assert _joint_decision("cointegration", None) is None


class TestJointDecisionIntegration:
    def test_cointegrated_dgp_joint_cointegration(self) -> None:
        y, x = _dgp_cointegrated(seed=0, n=400)
        res = bounds_test(y, x, case=3, order=(2, 1))
        assert res.decision_joint == "cointegration"

    def test_cases_ii_iv_joint_is_none(self) -> None:
        y, x = _dgp_cointegrated(seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(y, x, case=2, order=(2, 1))
        assert res.decision_joint is None

    def test_degenerate_suspicion_emits_warning_pointing_to_spec15(self) -> None:
        """DGP dégénéré de type 1 : relation en niveaux portée par x seul
        (lambda = 0, gamma != 0) -> F peut rejeter, t ne doit pas."""
        rng = np.random.default_rng(2)
        n = 400
        x = np.cumsum(rng.normal(size=n))
        # Δy_t = 0.5 x_{t-1} + eps : gamma significatif, pas de rappel en y
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = y[t - 1] + 0.5 * x[t - 1] + rng.normal(scale=0.5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = bounds_test(
                pd.Series(y, name="y"), pd.DataFrame({"x": x}), case=3, order=(1, 1)
            )
        if res.decision_joint == "degenerate_suspicion":
            messages = [str(w.message) for w in caught]
            assert any("spec 15" in m for m in messages)
            assert any("dégénérescence" in m for m in messages)

    def test_summary_reports_joint_decision(self) -> None:
        y, x = _dgp_cointegrated(seed=3)
        res = bounds_test(y, x, case=3, order=(2, 1))
        assert "décision jointe" in res.summary()


class TestUnilaterality:
    """Spec 11 §3.1 : DGP explosif -> JAMAIS « cointegration »."""

    @pytest.mark.parametrize("seed", range(5))
    def test_explosive_never_cointegration(self, seed: int) -> None:
        y, x = _dgp_explosive(seed=100 + seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(y, x, case=3, order=(1, 1))
        assert res.decision_t != "cointegration"
        assert res.decision_joint != "cointegration"

    def test_explosive_warns_degenerate_case(self) -> None:
        y, x = _dgp_explosive(seed=200)
        with pytest.warns(DegenerateCaseWarning, match="lambda_hat"):
            bounds_test(y, x, case=3, order=(1, 1))


class TestAdjustmentCIRule:
    """Spec 11 §2.4 + piège connu : IC sur lambda affiché seulement
    après cointégration établie (décision jointe)."""

    def test_ci_available_when_cointegrated(self) -> None:
        y, x = _dgp_cointegrated(seed=10, n=400)
        res = bounds_test(y, x, case=3, order=(2, 1))
        assert res.decision_joint == "cointegration"
        adj = res.adjustment()
        assert adj["lambda"] < 0
        assert np.isfinite(adj["ci_lower"]) and np.isfinite(adj["ci_upper"])
        assert adj["ci_lower"] < adj["lambda"] < adj["ci_upper"]

    def test_ci_masked_when_not_cointegrated(self) -> None:
        y, x = _dgp_no_cointegration(seed=11)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(y, x, case=3, order=(2, 1))
        assert res.decision_joint != "cointegration"
        with pytest.warns(PyardlMethodologyWarning, match="cointégration"):
            adj = res.adjustment()
        assert np.isfinite(adj["lambda"])  # l'estimée reste consultable
        assert np.isnan(adj["ci_lower"]) and np.isnan(adj["ci_upper"])

    def test_ci_masked_for_cases_without_t(self) -> None:
        """Cas II/IV : cointégration non établissable par la logique
        jointe -> IC masqué."""
        y, x = _dgp_cointegrated(seed=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(y, x, case=4, order=(2, 1))
        with pytest.warns(PyardlMethodologyWarning, match="cointégration"):
            adj = res.adjustment()
        assert np.isnan(adj["ci_lower"])


class TestStatsmodelsConcordance:
    """Spec 11 §3.3 (volet statsmodels) : le t_BDM = t du coefficient
    y.L1 de l'UECM statsmodels, à 1e-6. (Volet R bounds_t_test :
    validation/external/spec11_bounds_t_test.R, données danoises.)"""

    @pytest.mark.parametrize("case", [3, 5])
    def test_t_stat_matches_statsmodels_uecm(self, case: int) -> None:
        y, x = _dgp_cointegrated(seed=20)
        trend = {3: "c", 5: "ct"}[case]
        res = bounds_test(y, x, case=case, order=(2, 2))
        sm_fit = SM_UECM(y, lags=2, exog=x, order={"x": 2}, trend=trend).fit()
        assert res.t_stat == pytest.approx(float(sm_fit.tvalues["y.L1"]), abs=1e-6)


def _mc_t_rejection(n_rep: int, n: int, case: int, seed0: int) -> tuple[float, float]:
    """Taux de rejet du t seul à la borne I(1) (5 %) sous H0 / sous H1."""
    rej_h0 = rej_h1 = 0
    for rep in range(n_rep):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, x = _dgp_no_cointegration(seed=seed0 + rep, n=n)
            if bounds_test(y, x, case=case, order=(2, 1)).decision_t == (
                "cointegration"
            ):
                rej_h0 += 1
            y, x = _dgp_cointegrated(seed=seed0 + 70_000 + rep, n=n)
            if bounds_test(y, x, case=case, order=(2, 1)).decision_t == (
                "cointegration"
            ):
                rej_h1 += 1
    return rej_h0 / n_rep, rej_h1 / n_rep


@pytest.mark.fast_mc
@pytest.mark.parametrize("case", [3, 5])
def test_t_size_and_power_fast(case: int) -> None:
    """Spec 11 §3.2 (version CI, 200 réplications) : taille du t seul à
    la borne I(1) <= ~7.5 % sous H0 tout-I(1) ; puissance correcte."""
    size, power = _mc_t_rejection(n_rep=200, n=300, case=case, seed0=5000)
    assert size <= 0.075, f"taille t {size:.3f} (cas {case})"
    assert power >= 0.60, f"puissance t {power:.3f} (cas {case})"


@pytest.mark.slow
@pytest.mark.parametrize("case", [3, 5])
def test_t_size_and_power_full(case: int) -> None:
    """Spec 11 §3.2 (version complète, 1000 réplications, cas III et V)."""
    size, power_300 = _mc_t_rejection(n_rep=1000, n=300, case=case, seed0=6000)
    assert size <= 0.065, f"taille t {size:.3f} (cas {case})"
    _, power_500 = _mc_t_rejection(n_rep=1000, n=500, case=case, seed0=7000)
    assert power_500 >= power_300
    assert power_500 >= 0.85
