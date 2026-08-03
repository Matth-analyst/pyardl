"""Spec 05 — plan de tests §6 (hors verrous, cf. test_ardl_statsmodels.py
et test_ardl_select_order.py) : cohérence interne, stabilité, pont
spec 03, garde-fou spec 09, covariances robustes, GETS.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pyardl.core.ardl import ARDL
from pyardl.core.transforms import ardl_to_ecm
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.utils import lag_matrix


def _dgp_clean(seed: int, n: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    """DGP ARDL(1,1) propre (erreurs iid) : le garde-fou ne doit PAS tirer."""
    rng = np.random.default_rng(seed)
    xv = rng.normal(size=n).cumsum()
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = (
            0.3 + 0.5 * y[t - 1] + 0.6 * xv[t] + 0.2 * xv[t - 1] + rng.normal(scale=0.4)
        )
    return pd.Series(y, name="y"), pd.DataFrame({"x": xv})


class TestInternalConsistency:
    def test_ardl_0_q_equals_fdl_ols(self) -> None:
        """Spec 05 §6.1 : ARDL(0, q) ≡ FDL libre = OLS sur lag_matrix.

        (Le croisement avec AlmonModel/KoyckModel attend les specs 01-02,
        v0.5 ; le FDL libre est une OLS directe, testable dès maintenant.)
        """
        y, x = _dgp_clean(seed=0)
        q = 2
        res = ARDL(y, x, order=(0, q), det="const")._fit()

        xl = lag_matrix(x["x"].to_numpy(), q)
        design = np.column_stack([np.ones(xl.shape[0]), xl])
        coefs, *_ = np.linalg.lstsq(design, y.to_numpy()[q:], rcond=None)
        np.testing.assert_allclose(res.params.values, coefs, atol=1e-10)

    def test_pure_ar_without_x(self) -> None:
        """AR(p) pur : x=None, order=p entier."""
        y, _ = _dgp_clean(seed=1)
        res = ARDL(y, order=2)._fit()
        assert res.params.index.tolist() == ["const", "y.L1", "y.L2"]

    def test_det_none(self) -> None:
        y, x = _dgp_clean(seed=2)
        res = ARDL(y, x, order=(1, 1), det="none")._fit()
        assert "const" not in res.params.index

    def test_hold_back_too_small_raises(self) -> None:
        y, x = _dgp_clean(seed=3)
        with pytest.raises(ValueError, match="hold_back"):
            ARDL(y, x, order=(2, 3), hold_back=2)

    def test_seasonal_dummies_available(self) -> None:
        """Les dummies saisonnières existent désormais (spec 04 §2.3).

        Ce test remplace celui qui verrouillait le NotImplementedError :
        la fonctionnalité est implémentée, pas contournée.
        """
        y, x = _dgp_clean(seed=4)
        res = ARDL(y, x, order=(1, 1), seasonal=True, seasonal_periods=4)._fit()
        assert sum("season" in n for n in res.params.index) == 3

    def test_fixed_regressors_match_statsmodels(self) -> None:
        from statsmodels.tsa.ardl import ARDL as SM_ARDL

        y, x = _dgp_clean(seed=5)
        rng = np.random.default_rng(99)
        z = rng.normal(size=(len(y), 1))
        res = ARDL(y, x, order=(1, 1), fixed_regressors=z)._fit()
        sm_res = SM_ARDL(y, lags=1, exog=x, order=1, trend="c", fixed=z).fit()
        np.testing.assert_allclose(res.params.values, sm_res.params.values, atol=1e-10)


class TestSpec03Bridge:
    """Point de vigilance n°2 : les objets de fit() sont consommables par
    la spec 03 sans conversion manuelle."""

    def test_ardl_params_feeds_ardl_to_ecm_directly(self) -> None:
        y, x = _dgp_clean(seed=10)
        res = ARDL(y, x, order=(2, 1)).fit()
        ecm = ardl_to_ecm(res.ardl_params)  # aucune conversion manuelle
        assert ecm.p == 2
        assert ecm.q == (1,)

    def test_to_ecm_matches_manual_ecm_regression(self) -> None:
        """Spec 05 §6.2 : équivalence ARDL/ECM au niveau de l'objet
        résultat (verrou spec 03 §6.1.2 rejoué via l'API)."""
        y, x = _dgp_clean(seed=11)
        res = ARDL(y, x, order=(2, 2)).fit()
        ecm = res.to_ecm()

        yv, xv = y.to_numpy(), x["x"].to_numpy()
        n, start = len(yv), 2
        dy, dx = np.diff(yv), np.diff(xv)
        design = np.column_stack(
            [
                np.ones(n - start),
                yv[start - 1 : n - 1],
                xv[start - 1 : n - 1],
                dy[start - 2 : n - 2],
                dx[start - 1 : n - 1],
                dx[start - 2 : n - 2],
            ]
        )
        coefs, *_ = np.linalg.lstsq(design, dy[start - 1 :], rcond=None)

        assert ecm.lam == pytest.approx(coefs[1], abs=1e-8)
        assert ecm.gamma[0] == pytest.approx(coefs[2], abs=1e-8)
        assert ecm.psi[0] == pytest.approx(coefs[3], abs=1e-8)
        np.testing.assert_allclose(ecm.omega[0], coefs[4:6], atol=1e-8)

    def test_longrun_and_adjustment_views(self) -> None:
        y, x = _dgp_clean(seed=12)
        res = ARDL(y, x, order=(1, 1)).fit()
        lr = res.longrun
        assert list(lr.columns) == ["theta", "se"]
        assert np.isfinite(lr["theta"].iloc[0])
        adj = res.adjustment
        assert adj["lambda"] < 0  # force de rappel présente sur ce DGP
        assert np.isfinite(adj["half_life"])

    def test_to_ecm_raises_for_p0(self) -> None:
        y, x = _dgp_clean(seed=13)
        res = ARDL(y, x, order=(0, 1))._fit()
        with pytest.raises(ValueError, match="p=0"):
            res.to_ecm()


class TestStability:
    def test_explosive_dgp_is_stable_false(self) -> None:
        """Spec 05 §6.4 : DGP explosif -> is_stable=False + warning."""
        rng = np.random.default_rng(20)
        n = 150
        xv = rng.normal(size=n)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 1.05 * y[t - 1] + 0.1 * xv[t] + rng.normal(scale=0.1)
        res = ARDL(pd.Series(y, name="y"), pd.DataFrame({"x": xv}), order=(1, 0))._fit()
        with pytest.warns(PyardlMethodologyWarning, match="Unstable dynamics"):
            assert res.is_stable is False

    def test_stationary_dgp_is_stable_true(self) -> None:
        y, x = _dgp_clean(seed=21)
        res = ARDL(y, x, order=(1, 1))._fit()
        assert res.is_stable is True

    def test_ar_roots_match_phi(self) -> None:
        """Racine unique 1/phi pour un AR(1)."""
        y, x = _dgp_clean(seed=22)
        res = ARDL(y, x, order=(1, 1))._fit()
        phi = float(res.params["y.L1"])
        assert res.ar_roots[0] == pytest.approx(1.0 / phi, rel=1e-10)


class TestSpec09Guard:
    """Spec 09 §2.2 : Ljung-Box automatique après fit."""

    def test_warns_on_underspecified_ar_errors(self) -> None:
        """DGP à erreurs AR(1), modèle volontairement sous-spécifié."""
        rng = np.random.default_rng(30)
        n = 400
        xv = rng.normal(size=n).cumsum()
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.8 * u[t - 1] + rng.normal(scale=0.3)
        y = pd.Series(0.5 * xv + u, name="y")
        with pytest.warns(PyardlMethodologyWarning, match="Ljung-Box"):
            ARDL(y, pd.DataFrame({"x": xv}), order=(1, 0)).fit()

    def test_silent_on_clean_dgp(self) -> None:
        """Pas de warning sur le DGP propre correctement spécifié."""
        y, x = _dgp_clean(seed=31)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PyardlMethodologyWarning)
            ARDL(y, x, order=(1, 1)).fit()


class TestRobustCovariances:
    """Spec 05 §2.2 : cov_type ∈ {HC0-3, HAC} — concordance avec
    statsmodels OLS sur le même design, à 1e-10."""

    @pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
    def test_hc_match_statsmodels_ols(self, cov_type: str) -> None:
        y, x = _dgp_clean(seed=40)
        model = ARDL(y, x, order=(1, 1))
        res = model._fit(cov_type=cov_type)  # type: ignore[arg-type]
        design, y_dep, _ = model._build_design()
        sm_res = sm.OLS(y_dep, design).fit(cov_type=cov_type)
        np.testing.assert_allclose(res.bse.values, sm_res.bse, rtol=0, atol=1e-10)

    def test_hac_match_statsmodels_ols(self) -> None:
        y, x = _dgp_clean(seed=41)
        model = ARDL(y, x, order=(1, 1))
        res = model._fit(cov_type="HAC", cov_kwds={"nlags": 4})
        design, y_dep, _ = model._build_design()
        sm_res = sm.OLS(y_dep, design).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False}
        )
        np.testing.assert_allclose(res.bse.values, sm_res.bse, rtol=0, atol=1e-10)

    def test_unknown_cov_type_raises(self) -> None:
        y, x = _dgp_clean(seed=42)
        with pytest.raises(ValueError, match="cov_type"):
            ARDL(y, x, order=(1, 1))._fit(cov_type="HC9")  # type: ignore[arg-type]


class TestGETS:
    @pytest.mark.needs_review
    def test_reduction_from_general_recovers_parsimonious_model(self) -> None:
        """Spec 05 §4 : GETS depuis (4, 3) sur DGP ARDL(2, 1) réduit les
        ordres sans casser les diagnostics.

        Marqué needs_review : la spec ne précise pas si l'élimination
        peut créer des « trous » dans la structure des retards ;
        interprétation contiguë retenue (cf. docs/QUESTIONS.md).
        """
        rng = np.random.default_rng(50)
        n = 500
        xv = rng.normal(size=n).cumsum()
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = (
                0.5
                + 0.5 * y[t - 1]
                - 0.2 * y[t - 2]
                + 0.6 * xv[t]
                + 0.3 * xv[t - 1]
                + rng.normal(scale=0.4)
            )
        gets = ARDL.gets(
            pd.Series(y, name="y"), pd.DataFrame({"x": xv}), max_p=4, max_q=3
        )
        p_final, q_final = gets.final_order
        assert p_final <= 4 and q_final["x"] <= 3
        assert p_final >= 2  # ne sur-réduit pas la vraie dynamique
        assert q_final["x"] >= 1
        # chemin journalisé et cohérent
        path = gets.reduction_path
        assert set(path.columns) >= {
            "dropped",
            "pvalue",
            "ljungbox_p",
            "cumulative_f_p",
            "accepted",
        }
        assert len(path) >= 1  # au moins une réduction tentée depuis (4,3)

    def test_reduction_path_stops_at_significant_lag(self) -> None:
        """Le retard terminal significatif n'est jamais éliminé."""
        rng = np.random.default_rng(51)
        n = 500
        xv = rng.normal(size=n).cumsum()
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 0.6 * y[t - 1] + 0.8 * xv[t] + rng.normal(scale=0.3)
        gets = ARDL.gets(
            pd.Series(y, name="y"), pd.DataFrame({"x": xv}), max_p=2, max_q=2
        )
        p_final, q_final = gets.final_order
        assert p_final >= 1
        # les réductions acceptées ont toutes p-value > alpha
        accepted = gets.reduction_path[gets.reduction_path["accepted"]]
        assert (accepted["pvalue"] > 0.05).all()


class TestPresentation:
    def test_summary_contains_key_stats(self) -> None:
        y, x = _dgp_clean(seed=60)
        res = ARDL(y, x, order=(1, 1))._fit()
        s = res.summary()
        for token in ("ARDL(1; x:1)", "nobs=", "AIC=", "y.L1", "x.L0"):
            assert token in s

    def test_diagnostics_frame(self) -> None:
        y, x = _dgp_clean(seed=61)
        res = ARDL(y, x, order=(1, 1))._fit()
        diag = res.diagnostics()
        assert "Jarque-Bera" in diag.index
        assert ((diag["pvalue"] >= 0) & (diag["pvalue"] <= 1)).all()

    def test_selection_top(self) -> None:
        y, x = _dgp_clean(seed=62)
        sel = ARDL.select_order(y, x, max_p=2, max_q=1, ic="aic")
        assert len(sel.top(3)) == 3
