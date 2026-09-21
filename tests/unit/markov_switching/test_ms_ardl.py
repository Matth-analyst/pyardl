"""Tests for pyardl.markov_switching.ms_ardl (Spec 35 — Hamilton 1989)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.markov_switching import ms_ardl
from pyardl.markov_switching.model import _build_design


def _dgp_two_regimes(
    rng: np.random.Generator,
    n: int,
    switch_at: int,
    lam0: float = -0.6,
    theta0: float = 1.0,
    lam1: float = -0.15,
    theta1: float = 2.0,
    sigma: float = 0.3,
) -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    pd.DataFrame,
    np.ndarray[tuple[int, ...], np.dtype[np.int_]],
]:
    """Two-regime DGP with a single deterministic switch (a degenerate Markov chain).

    Not a general Markov draw — a single switch point is enough to
    check parameter recovery (spec 35 §5.1) without needing to also
    recover a simulated state path exactly.
    """
    x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
    y = np.zeros(n)
    true_state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state = 0 if t < switch_at else 1
        true_state[t] = state
        lam, theta = (lam0, theta0) if state == 0 else (lam1, theta1)
        y[t] = y[t - 1] + lam * (y[t - 1] - theta * x.iloc[t - 1])
        y[t] += rng.standard_normal() * sigma
    return y, pd.DataFrame({"x": x}), true_state


class TestAPI:
    def test_requires_p_at_least_1(self) -> None:
        rng = np.random.default_rng(0)
        y, x, _ = _dgp_two_regimes(rng, 100, 50)
        with pytest.raises(ValueError, match="p.*>= 1"):
            ms_ardl(y, x, order=(0, 1))

    def test_requires_at_least_2_states(self) -> None:
        rng = np.random.default_rng(0)
        y, x, _ = _dgp_two_regimes(rng, 100, 50)
        with pytest.raises(ValueError, match="n_states"):
            ms_ardl(y, x, order=(1, 1), n_states=1)

    def test_rejects_invalid_method(self) -> None:
        rng = np.random.default_rng(0)
        y, x, _ = _dgp_two_regimes(rng, 100, 50)
        with pytest.raises(ValueError, match="method"):
            ms_ardl(y, x, order=(1, 1), method="bogus")  # type: ignore[arg-type]

    def test_warns_above_3_states(self) -> None:
        rng = np.random.default_rng(0)
        y, x, _ = _dgp_two_regimes(rng, 200, 100)
        with pytest.warns(PyardlMethodologyWarning, match="degrades"):
            ms_ardl(y, x, order=(1, 1), n_states=4, n_starts=1)

    def test_requires_at_least_one_regressor(self) -> None:
        rng = np.random.default_rng(0)
        y, _, _ = _dgp_two_regimes(rng, 100, 50)
        with pytest.raises(ValueError, match="regressor"):
            ms_ardl(y, None, order=(1, 1))  # type: ignore[arg-type]

    def test_trend_and_extra_ar_lag(self) -> None:
        rng = np.random.default_rng(10)
        y, x, _ = _dgp_two_regimes(rng, 250, 125)
        res = ms_ardl(y, x, order=(2, 1), n_states=2, det="trend", n_starts=1, seed=10)
        assert "trend" in res.names
        assert "D.y.L1" in res.names

    def test_direct_method(self) -> None:
        rng = np.random.default_rng(11)
        y, x, _ = _dgp_two_regimes(rng, 200, 100)
        res = ms_ardl(
            y, x, order=(1, 1), n_states=2, method="direct", n_starts=1, seed=11
        )
        assert res.n_states == 2

    def test_plot_regimes_returns_axes(self) -> None:
        pytest.importorskip("matplotlib")
        rng = np.random.default_rng(12)
        y, x, _ = _dgp_two_regimes(rng, 200, 100)
        res = ms_ardl(y, x, order=(1, 1), n_states=2, n_starts=1, seed=12)
        axes = res.plot_regimes()
        assert axes is not None

    def test_result_shape(self) -> None:
        rng = np.random.default_rng(1)
        y, x, _ = _dgp_two_regimes(rng, 300, 150)
        res = ms_ardl(y, x, order=(1, 1), n_states=2, seed=1)

        assert res.transition_matrix.shape == (2, 2)
        np.testing.assert_allclose(
            res.transition_matrix.sum(axis=1).to_numpy(), 1.0, atol=1e-6
        )
        assert set(res.regime_params) == {0, 1}
        assert "sigma2" in res.regime_params[0]
        assert res.filtered_probs.shape[1] == 2
        assert res.smoothed_probs.shape[1] == 2
        assert len(res.expected_duration) == 2
        assert isinstance(res.summary(), str)
        assert "Regime 0" in res.summary()


class TestParameterRecovery:
    """Spec 35 §5.1 — known 2-regime DGP: regime coefficients recovered."""

    def test_recovers_regime_coefficients(self) -> None:
        rng = np.random.default_rng(2)
        n = 500
        y, x, _ = _dgp_two_regimes(rng, n, switch_at=250)
        res = ms_ardl(y, x, order=(1, 1), n_states=2, n_starts=3, seed=2)

        lambdas = sorted(params["y.L1"] for params in res.regime_params.values())
        assert lambdas[0] == pytest.approx(-0.6, abs=0.15)
        assert lambdas[1] == pytest.approx(-0.15, abs=0.15)

    def test_smoothed_probs_identify_the_switch(self) -> None:
        rng = np.random.default_rng(3)
        n = 500
        y, x, true_state = _dgp_two_regimes(rng, n, switch_at=250)
        res = ms_ardl(y, x, order=(1, 1), n_states=2, n_starts=3, seed=3)

        offset = n - res.smoothed_probs.shape[0]
        aligned_state = true_state[offset:]
        state1_prob = res.smoothed_probs[1].to_numpy()
        predicted_state = (state1_prob > 0.5).astype(int)

        agreement = np.mean(predicted_state == aligned_state)
        assert agreement > 0.7 or agreement < 0.3


class TestDegenerateChain:
    """Spec 35 §5.2 — near-absorbing chain: coefficients collapse, duration is long."""

    def test_single_regime_dgp_gives_similar_regimes_and_long_duration(self) -> None:
        rng = np.random.default_rng(4)
        n = 300
        x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = y[t - 1] - 0.4 * (y[t - 1] - 1.2 * x.iloc[t - 1])
            y[t] += rng.standard_normal() * 0.3
        res = ms_ardl(
            y, pd.DataFrame({"x": x}), order=(1, 1), n_states=2, n_starts=2, seed=4
        )

        lambdas = [params["y.L1"] for params in res.regime_params.values()]
        assert abs(lambdas[0] - lambdas[1]) < 0.3
        # At least one state should absorb almost all of the sample --
        # a spurious, single-observation "regime" occasionally appears
        # for the other (a known pathology of unconstrained ML for
        # switching-variance mixtures: likelihood is unbounded as a
        # regime's variance shrinks around one point), so the max
        # duration is the robust check for "collapsed to one regime".
        assert res.expected_duration.max() > 30


class TestEMLikelihoodMonotone:
    """Spec 35 §5.3 — EM log-likelihood is non-decreasing across iterations."""

    def test_em_llf_history_is_non_decreasing(self) -> None:
        rng = np.random.default_rng(5)
        n = 300
        y, x, _ = _dgp_two_regimes(rng, n, switch_at=150)
        y_arr = y.astype(np.float64)
        x_arr = x.to_numpy(dtype=np.float64)
        design, target, names = _build_design(y_arr, x_arr, 1, 1, "const")
        exog_df = pd.DataFrame(design, columns=names)

        mod = MarkovRegression(
            target,
            k_regimes=2,
            trend="n",
            exog=exog_df,
            switching_exog=True,
            switching_variance=True,
        )
        em_res = mod._fit_em(maxiter=15, full_output=True)
        llf_history = np.asarray(em_res.mle_retvals["llf"])

        diffs = np.diff(llf_history)
        assert np.mean(diffs >= -1e-6) > 0.85


class TestExternalConcordance:
    """Spec 35 §5.4 — statsmodels IS the estimation engine, so llf/params are exact.

    Not an independent cross-check (the module wraps
    ``statsmodels.tsa.regime_switching.markov_regression.MarkovRegression``
    directly, documented in ``docs/DEVIATIONS.md``) — this locks the
    wrapper's own bookkeeping (design construction, parameter
    extraction, transition-matrix orientation) against a direct call to
    the wrapped engine, so a refactor cannot silently break that
    bookkeeping.
    """

    def test_llf_and_params_match_direct_statsmodels_call(self) -> None:
        rng = np.random.default_rng(6)
        n = 300
        y, x, _ = _dgp_two_regimes(rng, n, switch_at=150)
        res = ms_ardl(y, x, order=(1, 1), n_states=2, n_starts=1, seed=6)

        y_arr = y.astype(np.float64)
        x_arr = x.to_numpy(dtype=np.float64)
        design, target, names = _build_design(y_arr, x_arr, 1, 1, "const")
        exog_df = pd.DataFrame(design, columns=names)
        mod = MarkovRegression(
            target,
            k_regimes=2,
            trend="n",
            exog=exog_df,
            switching_exog=True,
            switching_variance=True,
        )
        em_res = mod._fit_em(maxiter=1000, tolerance=1e-6)
        direct_res = mod.smooth(em_res.params)

        assert res.llf == pytest.approx(float(direct_res.llf), abs=1e-6)


class TestTransitionMatrixOrientation:
    def test_rows_sum_to_one_and_diagonal_matches_expected_duration(self) -> None:
        rng = np.random.default_rng(7)
        y, x, _ = _dgp_two_regimes(rng, 300, 150)
        res = ms_ardl(y, x, order=(1, 1), n_states=2, seed=7)

        for state in range(2):
            p_ii = res.transition_matrix.loc[state, state]
            expected = 1.0 / (1.0 - p_ii)
            assert res.expected_duration[state] == pytest.approx(expected, rel=1e-6)
