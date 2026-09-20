"""Spec 39 §5 — plan de tests GIRF/FEVD.

La portée de cette version (documentée dans le module et
docs/DEVIATIONS.md) couvre le cas linéaire — les tests 1, 2 et 4 de la
spec. Le test 3 (GIRF dépendante de l'historique sur un DGP NARDL avec
recomposition non linéaire à chaque pas simulé) demanderait le moteur
non implémenté ici et n'est donc pas testé.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.core.ardl import ARDL
from pyardl.simulate import dynardl_simulate, fevd, generalized_irf


def _linear_ardl(n: int = 200, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
    y = pd.Series(2.0 + 1.5 * x.to_numpy() + rng.standard_normal(n), name="y")
    return ARDL(y, pd.DataFrame({"x": x}), order=(1, 1)).fit()


class TestGIRFHistoryInvariance:
    """Spec 39 §5.1 — sur un ARDL linéaire, GIRF ne dépend pas de l'historique."""

    def test_all_histories_give_the_same_girf(self) -> None:
        res = _linear_ardl(seed=1)
        g = generalized_irf(res, "x", shock_size=1.0, h=20, r=30, n_histories=6, seed=2)
        vals = np.array([s.to_numpy() for s in g.girf_by_history.values()])
        for row in vals[1:]:
            np.testing.assert_allclose(row, vals[0], atol=1e-9)

    def test_explicit_histories_also_invariant(self) -> None:
        res = _linear_ardl(seed=1)
        histories = [{"x": 0.0}, {"x": 5.0}, {"x": -3.0}]
        g = generalized_irf(
            res, "x", shock_size=1.0, h=15, r=20, histories=histories, seed=2
        )
        vals = np.array([s.to_numpy() for s in g.girf_by_history.values()])
        np.testing.assert_allclose(vals[0], vals[1], atol=1e-9)
        np.testing.assert_allclose(vals[0], vals[2], atol=1e-9)


class TestGIRFMatchesDynardl:
    """Spec 39 §5.2 — GIRF coïncide avec dynardl_simulate sur un ARDL linéaire."""

    def test_girf_mean_matches_impulse_response(self) -> None:
        res = _linear_ardl(seed=3)
        g = generalized_irf(res, "x", shock_size=1.0, h=20, r=50, n_histories=5, seed=7)
        sim = dynardl_simulate(
            res, "x", shock_type="impulse", size=1.0, t0=0, horizon=20, r=50, seed=7
        )
        np.testing.assert_allclose(
            g.girf_mean.to_numpy(),
            sim.summary_df[("response", "point")].to_numpy(),
            atol=1e-8,
        )


class TestAPIAndValidation:
    def test_summary_contains_key_fields(self) -> None:
        res = _linear_ardl(seed=4)
        g = generalized_irf(res, "x", h=10, r=20, n_histories=3, seed=0)
        text = g.summary()
        assert "Generalized IRF" in text

    def test_fevd_invalid_h_raises(self) -> None:
        res = _linear_ardl(seed=4)
        with pytest.raises(ValueError, match="h"):
            fevd(res, "x", h=0)

    def test_fevd_invalid_shock_raises(self) -> None:
        res = _linear_ardl(seed=4)
        with pytest.raises(KeyError):
            fevd(res, "bogus")


class TestFEVD:
    """Spec 39 §5.4 — FEVD proche de 0 quand x n'a aucun effet sur y."""

    def test_shares_bounded(self) -> None:
        res = _linear_ardl(seed=5)
        out = fevd(res, "x", h=25)
        assert (out.shares >= 0.0).all()
        assert (out.shares <= 1.0).all()

    def test_no_effect_dgp_share_near_zero(self) -> None:
        rng = np.random.default_rng(6)
        n = 200
        x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
        y = pd.Series(2.0 + rng.standard_normal(n), name="y")
        res = ARDL(y, pd.DataFrame({"x": x}), order=(1, 1)).fit()
        out = fevd(res, "x", h=20)
        assert out.shares.iloc[-1] < 0.05

    def test_explicit_variance_used(self) -> None:
        res = _linear_ardl(seed=7)
        out = fevd(res, "x", h=10, x_shock_variance=2.5)
        assert out.x_shock_variance == pytest.approx(2.5)


class TestPlot:
    def test_plot_girf_runs(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        res = _linear_ardl(seed=8)
        g = generalized_irf(res, "x", h=10, r=20, n_histories=3, seed=0)
        ax = g.plot_girf()
        assert ax is not None

    def test_plot_fevd_runs(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        res = _linear_ardl(seed=8)
        out = fevd(res, "x", h=10)
        ax = out.plot_fevd()
        assert ax is not None
