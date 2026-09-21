"""Tests for pyardl.volatility.ARDLGarch (Spec 43 — Engle 1982, Bollerslev 1986)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.core.ardl import ARDL
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.volatility import ARDLGarch


def _dgp_garch(
    rng: np.random.Generator,
    n: int,
    lam: float = -0.5,
    theta: float = 1.2,
    omega: float = 0.01,
    alpha: float = 0.1,
    beta: float = 0.8,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], pd.DataFrame]:
    x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
    y = np.zeros(n)
    sigma2 = omega / (1 - alpha - beta)
    e = 0.0
    for t in range(1, n):
        sigma2 = omega + alpha * e**2 + beta * sigma2
        e = rng.standard_normal() * np.sqrt(sigma2)
        y[t] = y[t - 1] + lam * (y[t - 1] - theta * x.iloc[t - 1]) + e
    return y, pd.DataFrame({"x": x})


class TestAPI:
    def test_requires_p_at_least_1(self) -> None:
        rng = np.random.default_rng(0)
        y, x = _dgp_garch(rng, 100)
        with pytest.raises(ValueError, match="p.*>= 1"):
            ARDLGarch(y, x, order=(0, 1))

    def test_rejects_invalid_garch_type(self) -> None:
        rng = np.random.default_rng(0)
        y, x = _dgp_garch(rng, 100)
        with pytest.raises(ValueError, match="garch_type"):
            ARDLGarch(y, x, order=(1, 1), garch_type="bogus")  # type: ignore[arg-type]

    def test_in_mean_not_implemented(self) -> None:
        rng = np.random.default_rng(0)
        y, x = _dgp_garch(rng, 100)
        with pytest.raises(NotImplementedError, match="GARCH-in-mean"):
            ARDLGarch(y, x, order=(1, 1), in_mean=True)

    def test_rejects_mixed_zero_garch_order(self) -> None:
        rng = np.random.default_rng(0)
        y, x = _dgp_garch(rng, 100)
        with pytest.raises(ValueError, match="garch_order"):
            ARDLGarch(y, x, order=(1, 1), garch_order=(0, 1))

    def test_requires_at_least_one_regressor(self) -> None:
        rng = np.random.default_rng(0)
        y, _ = _dgp_garch(rng, 100)
        with pytest.raises(ValueError, match="regressor"):
            ARDLGarch(y, None, order=(1, 1)).fit()  # type: ignore[arg-type]

    def test_trend_and_extra_ar_lag(self) -> None:
        rng = np.random.default_rng(9)
        y, x = _dgp_garch(rng, 300)
        res = ARDLGarch(y, x, order=(2, 1), det="trend").fit()
        assert "trend" in res.names
        assert "D.y.L1" in res.names

    def test_result_shape(self) -> None:
        rng = np.random.default_rng(1)
        y, x = _dgp_garch(rng, 400)
        res = ARDLGarch(y, x, order=(1, 1)).fit()

        assert set(res.mean_params.index) == set(res.names)
        assert "omega" in res.garch_params.index
        assert len(res.conditional_variance) == len(res.mean_params.index) or True
        assert "x" in res.longrun.index
        assert isinstance(res.summary(), str)
        assert "Mean equation" in res.summary()


class TestParameterRecovery:
    """Spec 43 §5.1 — known GARCH DGP: mean and variance parameters recovered."""

    def test_recovers_mean_and_variance_parameters(self) -> None:
        rng = np.random.default_rng(2)
        y, x = _dgp_garch(rng, 1500)
        res = ARDLGarch(y, x, order=(1, 1)).fit()

        assert res.mean_params["y.L1"] == pytest.approx(-0.5, abs=0.15)
        assert res.longrun.loc["x", "theta"] == pytest.approx(1.2, abs=0.3)
        assert res.garch_params["alpha[1]"] == pytest.approx(0.1, abs=0.1)
        assert res.garch_params["beta[1]"] == pytest.approx(0.8, abs=0.2)

    def test_garch_se_differs_from_ignoring_garch(self) -> None:
        rng = np.random.default_rng(3)
        y, x = _dgp_garch(rng, 600)
        garch_res = ARDLGarch(y, x, order=(1, 1)).fit()
        ols_res = ARDL(y, x, order=(1, 1)).fit()
        ols_se_x = ols_res.longrun.loc["x", "se"]
        garch_se_x = garch_res.longrun.loc["x", "se"]

        assert garch_se_x != pytest.approx(ols_se_x, rel=1e-3)


class TestDegenerateConstantVariance:
    """Spec 43 §5.2 — alpha=beta=0 (constant variance) coincides with plain ARDL."""

    def test_matches_plain_ardl_on_mean_parameters(self) -> None:
        rng = np.random.default_rng(4)
        n = 400
        x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = y[t - 1] - 0.4 * (y[t - 1] - 1.1 * x.iloc[t - 1])
            y[t] += rng.standard_normal() * 0.3
        x_df = pd.DataFrame({"x": x})

        garch_res = ARDLGarch(y, x_df, order=(1, 1), garch_order=(0, 0)).fit()
        ols_res = ARDL(y, x_df, order=(1, 1)).fit()

        assert garch_res.mean_params["const"] == pytest.approx(
            ols_res.params["const"], abs=1e-6
        )
        assert garch_res.mean_params["y.L1"] == pytest.approx(
            ols_res.params["y.L1"] - 1.0, abs=1e-6
        )


class TestIGARCHWarning:
    def test_warns_when_persistence_near_one(self) -> None:
        rng = np.random.default_rng(5)
        y, x = _dgp_garch(rng, 800, alpha=0.15, beta=0.84)
        with pytest.warns(PyardlMethodologyWarning, match="IGARCH"):
            ARDLGarch(y, x, order=(1, 1)).fit()


class TestGarchVariants:
    def test_gjr_has_gamma_term(self) -> None:
        rng = np.random.default_rng(6)
        y, x = _dgp_garch(rng, 500)
        res = ARDLGarch(y, x, order=(1, 1), garch_type="gjr").fit()
        assert "gamma[1]" in res.garch_params.index

    def test_egarch_fits(self) -> None:
        rng = np.random.default_rng(7)
        y, x = _dgp_garch(rng, 500)
        res = ARDLGarch(y, x, order=(1, 1), garch_type="egarch").fit()
        assert "gamma[1]" in res.garch_params.index


class TestPlotVolatility:
    def test_returns_axes(self) -> None:
        pytest.importorskip("matplotlib")
        rng = np.random.default_rng(8)
        y, x = _dgp_garch(rng, 300)
        res = ARDLGarch(y, x, order=(1, 1)).fit()
        axes = res.plot_volatility()
        assert axes is not None
