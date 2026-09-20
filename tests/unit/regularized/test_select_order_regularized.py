"""Spec 41 §5 — plan de tests ARDL régularisé.

La calibration Monte Carlo de la récupération de l'ordre vrai (§5.1,
« dans la majorité des réplications ») est laissée à un futur
validation/spec41_montecarlo.py. Ici : contrat d'API, cohérence
descendante alpha->0 (§5.2, verrou exact), et LASSO vs Elastic Net sur
régresseurs corrélés (§5.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.core import ARDL
from pyardl.regularized import select_order_regularized


def _sparse_ardl(n: int = 250, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] - 0.5 * (y[t - 1] - 1.5 * x.iloc[t - 1])
        y[t] += rng.standard_normal() * 0.3
    return y, pd.DataFrame({"x": x})


class TestAPIAndValidation:
    def test_invalid_method_raises(self) -> None:
        y, x = _sparse_ardl()
        with pytest.raises(ValueError, match="method"):
            select_order_regularized(y, x, method="bogus")  # type: ignore[arg-type]

    def test_invalid_l1_ratio_raises(self) -> None:
        y, x = _sparse_ardl()
        with pytest.raises(ValueError, match="l1_ratio"):
            select_order_regularized(y, x, l1_ratio=1.5)

    def test_invalid_det_raises(self) -> None:
        y, x = _sparse_ardl()
        with pytest.raises(ValueError, match="det"):
            select_order_regularized(y, x, det="bogus")  # type: ignore[arg-type]

    def test_lasso_forces_l1_ratio_one(self) -> None:
        y, x = _sparse_ardl()
        res = select_order_regularized(y, x, method="lasso", l1_ratio=0.3, alpha=0.01)
        assert res.l1_ratio == 1.0

    def test_summary_contains_key_fields(self) -> None:
        y, x = _sparse_ardl()
        res = select_order_regularized(y, x, max_p=3, max_q=3, alpha=0.01)
        text = res.summary()
        assert "Regularized ARDL" in text
        assert "selected_order" in text

    def test_coefficients_path_shape(self) -> None:
        y, x = _sparse_ardl()
        res = select_order_regularized(
            y, x, max_p=3, max_q=3, alpha=0.01, alpha_grid_size=8
        )
        assert res.coefficients_path.shape[0] == 8

    def test_cv_mode_runs_and_reports_errors(self) -> None:
        y, x = _sparse_ardl(n=150)
        res = select_order_regularized(
            y, x, max_p=3, max_q=3, alpha="cv", alpha_grid_size=6, cv_folds=3
        )
        assert res.cv_errors is not None
        assert len(res.cv_errors) == 6


class TestDownstreamConsistency:
    """Spec 41 §5.2 — alpha -> 0 coïncide avec l'OLS complet."""

    def test_near_zero_alpha_matches_full_ols(self) -> None:
        y, x = _sparse_ardl(seed=2)
        res = select_order_regularized(
            y, x, max_p=3, max_q=3, method="lasso", alpha=1e-10
        )
        assert res.selected_order == (3, {"x": 3})
        ols_full = ARDL(y, x, order=(3, {"x": 3})).fit()
        for name in ols_full.params.index:
            assert res.best_model.params[name] == pytest.approx(
                ols_full.params[name], abs=1e-4
            )


class TestSparseRecovery:
    """Spec 41 §5.1 — un ordre parcimonieux connu est retrouvé pour alpha suffisant."""

    def test_large_enough_alpha_recovers_order_one(self) -> None:
        y, x = _sparse_ardl(seed=3)
        res = select_order_regularized(
            y, x, max_p=4, max_q=4, method="lasso", alpha=0.1
        )
        p, q_map = res.selected_order
        assert p == 1
        assert q_map["x"] <= 2


class TestElasticNetVsLasso:
    """Spec 41 §5.3 — deux régresseurs de niveau corrélés : Elastic Net plus stable."""

    def test_elastic_net_keeps_correlated_pair_more_often(self) -> None:
        n = 200
        both_kept_lasso = 0
        both_kept_enet = 0
        n_reps = 8
        for i in range(n_reps):
            rng_i = np.random.default_rng(4 + i)
            x1 = rng_i.standard_normal(n).cumsum()
            x2 = x1 + 0.05 * rng_i.standard_normal(n)  # rho > 0.9
            y = np.zeros(n)
            for t in range(1, n):
                y[t] = y[t - 1] - 0.5 * (y[t - 1] - 1.0 * x1[t - 1] - 1.0 * x2[t - 1])
                y[t] += rng_i.standard_normal() * 0.3
            xdf = pd.DataFrame({"x1": x1, "x2": x2})

            res_l = select_order_regularized(
                y, xdf, max_p=2, max_q=2, method="lasso", alpha=0.05
            )
            beta_l = res_l.best_model.longrun["theta"]
            both_kept_lasso += int((beta_l.abs() > 1e-6).all())

            res_e = select_order_regularized(
                y, xdf, max_p=2, max_q=2, method="elastic_net", l1_ratio=0.3, alpha=0.05
            )
            beta_e = res_e.best_model.longrun["theta"]
            both_kept_enet += int((beta_e.abs() > 1e-6).all())

        assert both_kept_enet >= both_kept_lasso


class TestPlot:
    def test_plot_regularization_path_runs(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        y, x = _sparse_ardl(seed=5)
        res = select_order_regularized(y, x, max_p=2, max_q=2, alpha=0.01)
        ax = res.plot_regularization_path()
        assert ax is not None
