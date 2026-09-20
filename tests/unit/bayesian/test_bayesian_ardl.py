"""Tests for pyardl.bayesian.BayesianARDL (Spec 42 — Minnesota prior)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.bayesian import BayesianARDL
from pyardl.bayesian.model import _build_design, _posterior_fit, _prior_variance


def _dgp(
    rng: np.random.Generator,
    n: int,
    lam: float = -0.5,
    theta: float = 1.5,
    n_lags: int = 3,
    distant_coefs_zero: bool = False,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], pd.DataFrame]:
    x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
    y = np.zeros(n)
    short_run = np.zeros(n_lags)
    if not distant_coefs_zero:
        short_run[:] = [0.3, -0.15, 0.08][:n_lags]
    for t in range(n_lags + 1, n):
        dy_lags = sum(
            short_run[i] * (y[t - i - 1] - y[t - i - 2]) for i in range(n_lags)
        )
        y[t] = y[t - 1] + lam * (y[t - 1] - theta * x.iloc[t - 1]) + dy_lags
        y[t] += rng.standard_normal() * 0.3
    return y, pd.DataFrame({"x": x})


class TestAPI:
    def test_requires_p_at_least_1(self) -> None:
        rng = np.random.default_rng(0)
        y, x = _dgp(rng, 100)
        with pytest.raises(ValueError, match="p.*>= 1"):
            BayesianARDL(y, x, order=(0, 1))

    def test_rejects_non_minnesota_prior(self) -> None:
        rng = np.random.default_rng(0)
        y, x = _dgp(rng, 100)
        with pytest.raises(ValueError, match="minnesota"):
            BayesianARDL(y, x, order=(1, 1), prior="jeffreys")  # type: ignore[arg-type]

    def test_rejects_invalid_det(self) -> None:
        rng = np.random.default_rng(0)
        y, x = _dgp(rng, 100)
        with pytest.raises(ValueError, match="det"):
            BayesianARDL(y, x, order=(1, 1), det="bogus")  # type: ignore[arg-type]

    def test_requires_at_least_one_regressor(self) -> None:
        rng = np.random.default_rng(0)
        y, _ = _dgp(rng, 100)
        with pytest.raises(ValueError, match="regressor"):
            BayesianARDL(y, None, order=(1, 1)).fit()  # type: ignore[arg-type]

    def test_trend_deterministic(self) -> None:
        rng = np.random.default_rng(9)
        y, x = _dgp(rng, 150, n_lags=1)
        res = BayesianARDL(
            y, x, order=(1, 1), det="trend", tau=0.3, n_draws=20, seed=9
        ).fit()
        assert "trend" in res.names

    def test_result_shape(self) -> None:
        rng = np.random.default_rng(1)
        y, x = _dgp(rng, 200)
        res = BayesianARDL(y, x, order=(2, 2), tau=0.5, n_draws=300, seed=1).fit()
        assert res.posterior_params.shape == (300, len(res.names))
        assert res.longrun_posterior.shape == (300, 1)
        assert set(res.longrun_posterior.columns) == {"x"}
        assert isinstance(res.summary(), str)
        assert "tau=" in res.summary()


class TestDiffusePriorMatchesOLS:
    """Spec 42 §5.1 — tau -> infinity recovers the frequentist estimator."""

    def test_diffuse_prior_posterior_mean_matches_ols(self) -> None:
        """``posterior_mean`` is a *sample* mean over ``n_draws`` draws, so
        this compares it to the exact analytic posterior mean directly
        (via ``_posterior_fit``, the same closed form the fit uses
        internally) rather than to OLS through a noisy Monte Carlo
        average — a large ``n_draws`` would also work but this is exact
        and instantaneous.
        """
        rng = np.random.default_rng(2)
        y, x = _dgp(rng, 300, n_lags=2)
        res = BayesianARDL(y, x, order=(2, 2), tau=1e8, n_draws=10, seed=2).fit()

        y_arr = y.astype(np.float64)
        x_arr = x.to_numpy(dtype=np.float64)
        design, target, names, is_informative, lag_index = _build_design(
            y_arr, x_arr, 2, 2, "const"
        )
        ols_theta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        v0 = _prior_variance(is_informative, lag_index, tau=1e8, decay=1.0)
        analytic_posterior_mean, _, _, _ = _posterior_fit(
            design, target, v0, 1e-3, 1e-3
        )

        assert names == res.names
        np.testing.assert_allclose(
            analytic_posterior_mean, ols_theta, atol=1e-6, rtol=1e-6
        )


class TestInformativePriorShrinkage:
    """Spec 42 §5.2 — informative prior beats OLS in MSE when distant lags are zero.

    Monte Carlo sized per CLAUDE.md rule 10: 300 replications gives a
    standard error on a mean-squared-error ratio comparison well under
    the effect being measured here (shrinkage toward a correctly-zero
    coefficient should dominate OLS noise by a wide margin, not a
    marginal one — this is not a size/power measurement near a boundary
    the way spec 16's bootstrap calibration was).
    """

    def test_bayesian_posterior_mean_has_lower_mse_than_ols(self) -> None:
        rng = np.random.default_rng(3)
        n_rep = 300
        n = 120
        true_theta = 1.5
        bayes_errors = []
        ols_errors = []
        for _ in range(n_rep):
            y, x = _dgp(rng, n, theta=true_theta, n_lags=3, distant_coefs_zero=True)
            res = BayesianARDL(
                y,
                x,
                order=(3, 1),
                tau=0.05,
                decay=2.0,
                n_draws=50,
                seed=int(rng.integers(1 << 30)),
            ).fit()
            bayes_errors.append((res.longrun_posterior["x"].mean() - true_theta) ** 2)

            y_arr = y.astype(np.float64)
            x_arr = x.to_numpy(dtype=np.float64)
            design, target, names, _, _ = _build_design(y_arr, x_arr, 3, 1, "const")
            ols_theta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            lam_hat = ols_theta[names.index("y.L1")]
            gamma_hat = ols_theta[names.index("x0.L1")]
            ols_errors.append((-gamma_hat / lam_hat - true_theta) ** 2)

        assert np.mean(bayes_errors) < np.mean(ols_errors)


class TestWeaklyIdentifiedLongRun:
    """Spec 42 §5.3 — diagnostic comparison under near-zero lambda.

    Not a judgement of which interval is "right": only checks that the
    comparison mechanism runs and reports a measurable difference in
    this known-difficult case.
    """

    def test_comparison_runs_and_reports_a_difference(self) -> None:
        rng = np.random.default_rng(4)
        y, x = _dgp(rng, 300, lam=-0.03, theta=1.5, n_lags=1)
        res = BayesianARDL(y, x, order=(1, 1), tau=0.3, n_draws=2000, seed=4).fit()
        comparison = res.compare_to_delta_method()

        assert "x" in comparison.index
        bayes_width = (
            comparison.loc["x", "bayes_upper"] - comparison.loc["x", "bayes_lower"]
        )
        delta_width = (
            comparison.loc["x", "delta_upper"] - comparison.loc["x", "delta_lower"]
        )
        assert bayes_width > 0
        assert delta_width > 0


class TestConjugateFormulaManualCheck:
    """Spec 42 §5.4 — external validation fallback: hand-derived NIG posterior.

    Single-regressor, no intercept, diffuse-everything design so the
    closed-form posterior can be derived by hand and checked to 1e-10
    (CLAUDE.md tolerance for algebraic identities).
    """

    def test_matches_hand_derived_posterior_mean_and_variance(self) -> None:
        rng = np.random.default_rng(5)
        n = 50
        x_col = rng.standard_normal(n)
        beta_true = 2.0
        y_col = beta_true * x_col + rng.standard_normal(n) * 0.5
        design = x_col.reshape(-1, 1)
        target = y_col

        tau2 = 4.0
        v0 = np.array([tau2])
        a0, b0 = 2.0, 1.0

        xtx = float(design.T @ design)
        xty = float(design.T @ target)
        vn_manual = 1.0 / (1.0 / tau2 + xtx)
        mean_manual = vn_manual * xty
        an_manual = a0 + n / 2.0
        bn_manual = b0 + 0.5 * (float(target @ target) - mean_manual**2 / vn_manual)

        posterior_mean, r_factor, an, bn = _posterior_fit(design, target, v0, a0, b0)
        vn_from_r = 1.0 / float(r_factor[0, 0] ** 2)

        assert posterior_mean[0] == pytest.approx(mean_manual, abs=1e-10)
        assert vn_from_r == pytest.approx(vn_manual, abs=1e-10)
        assert an == pytest.approx(an_manual, abs=1e-10)
        assert bn == pytest.approx(bn_manual, abs=1e-8)


class TestTauSelection:
    def test_tau_cv_selects_from_grid(self) -> None:
        rng = np.random.default_rng(6)
        y, x = _dgp(rng, 150, n_lags=1)
        res = BayesianARDL(
            y, x, order=(1, 1), tau="cv", n_draws=20, seed=6, tau_grid_size=6
        ).fit()
        assert res.tau_grid is not None
        assert res.tau in [float(v) for v in res.tau_grid]

    def test_tau_evidence_selects_from_grid(self) -> None:
        rng = np.random.default_rng(7)
        y, x = _dgp(rng, 150, n_lags=1)
        res = BayesianARDL(
            y, x, order=(1, 1), tau="evidence", n_draws=20, seed=7, tau_grid_size=6
        ).fit()
        assert res.tau_grid is not None
        assert res.tau in [float(v) for v in res.tau_grid]

    def test_fixed_float_tau_has_no_grid(self) -> None:
        rng = np.random.default_rng(8)
        y, x = _dgp(rng, 150, n_lags=1)
        res = BayesianARDL(y, x, order=(1, 1), tau=0.4, n_draws=20, seed=8).fit()
        assert res.tau == pytest.approx(0.4)
        assert res.tau_grid is None


class TestPriorVarianceHelper:
    def test_diffuse_terms_get_large_fixed_variance(self) -> None:
        is_informative = np.array([0.0, 0.0, 1.0, 1.0])
        lag_index = np.array([0.0, 0.0, 1.0, 2.0])
        v0 = _prior_variance(is_informative, lag_index, tau=1.0, decay=1.0)
        assert v0[0] == pytest.approx(1.0e8)
        assert v0[1] == pytest.approx(1.0e8)
        assert v0[2] == pytest.approx(1.0)
        assert v0[3] == pytest.approx(0.5)

    def test_decay_shrinks_more_at_distant_lags(self) -> None:
        is_informative = np.array([1.0, 1.0, 1.0])
        lag_index = np.array([1.0, 2.0, 3.0])
        v0 = _prior_variance(is_informative, lag_index, tau=1.0, decay=2.0)
        assert v0[0] > v0[1] > v0[2]
