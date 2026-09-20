"""Spec 37 §5 — plan de tests Pedroni/Westerlund.

Comme pour les autres specs à bootstrap coûteux (29/31/33), la
calibration de taille précise vit dans un futur
``validation/spec37_montecarlo.py`` (Étape 4). Ici : contrat d'API,
puissance sous cointégration, absence sous non-cointégration, et le
garde-fou CD (§2.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.panel import pedroni, westerlund


def _cointegrated_panel(n_units: int, n_obs: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        x = np.cumsum(rng.standard_normal(n_obs))
        u = np.zeros(n_obs)
        for t in range(1, n_obs):
            u[t] = -0.5 * u[t - 1] + rng.standard_normal() * 0.3
        y = 1.5 * x + u
        rows.append(pd.DataFrame({"id": i, "t": np.arange(n_obs), "y": y, "x": x}))
    return pd.concat(rows, ignore_index=True)


def _independent_panel(n_units: int, n_obs: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        x = np.cumsum(rng.standard_normal(n_obs))
        y = np.cumsum(rng.standard_normal(n_obs))
        rows.append(pd.DataFrame({"id": i, "t": np.arange(n_obs), "y": y, "x": x}))
    return pd.concat(rows, ignore_index=True)


class TestPedroniAPI:
    def test_invalid_det_raises(self) -> None:
        df = _independent_panel(6, 60, 0)
        with pytest.raises(ValueError, match="det"):
            pedroni(df, y="y", X=["x"], id="id", time="t", det="bogus")  # type: ignore[arg-type]

    def test_needs_two_units(self) -> None:
        df = _independent_panel(1, 60, 0)
        with pytest.raises(ValueError, match="two individuals"):
            pedroni(df, y="y", X=["x"], id="id", time="t")

    def test_invalid_decision_stat_raises(self) -> None:
        df = _independent_panel(6, 60, 0)
        res = pedroni(df, y="y", X=["x"], id="id", time="t", n_boot=9, seed=0)
        with pytest.raises(ValueError, match="stat"):
            res.decision("bogus")

    def test_summary_contains_both_statistics(self) -> None:
        df = _independent_panel(6, 60, 0)
        res = pedroni(df, y="y", X=["x"], id="id", time="t", n_boot=9, seed=0)
        text = res.summary()
        assert "panel_adf" in text
        assert "group_adf" in text


class TestWesterlundAPI:
    def test_invalid_det_raises(self) -> None:
        df = _independent_panel(6, 60, 0)
        with pytest.raises(ValueError, match="det"):
            westerlund(df, y="y", X=["x"], id="id", time="t", det="bogus")  # type: ignore[arg-type]

    def test_needs_two_units(self) -> None:
        df = _independent_panel(1, 60, 0)
        with pytest.raises(ValueError, match="two individuals"):
            westerlund(df, y="y", X=["x"], id="id", time="t")

    def test_cd_warning_set_when_pvalue_small(self) -> None:
        df = _independent_panel(6, 60, 0)
        with pytest.warns(match="cross-sectional dependence"):
            res = westerlund(
                df, y="y", X=["x"], id="id", time="t", n_boot=9, seed=0, cd_pvalue=0.001
            )
        assert res.cd_warning is not None

    def test_cd_warning_absent_when_pvalue_large(self) -> None:
        df = _independent_panel(6, 60, 0)
        res = westerlund(
            df, y="y", X=["x"], id="id", time="t", n_boot=9, seed=0, cd_pvalue=0.8
        )
        assert res.cd_warning is None


class TestPower:
    """Spec 37 §5.1 — DGP cointégré : les deux familles rejettent H0."""

    def test_cointegrated_dgp_rejects_no_cointegration(self) -> None:
        df = _cointegrated_panel(12, 100, 1)
        res_p = pedroni(df, y="y", X=["x"], id="id", time="t", n_boot=49, seed=1)
        res_w = westerlund(df, y="y", X=["x"], id="id", time="t", n_boot=49, seed=1)
        assert res_p.decision("panel_adf") == "cointegration"
        assert res_p.decision("group_adf") == "cointegration"
        assert res_w.decision("group_tau") == "cointegration"
        assert res_w.decision("panel_tau") == "cointegration"


class TestNoCointegration:
    """Spec 37 §5.2 — DGP sans cointégration : ne rejette pas (grossièrement)."""

    def test_independent_walks_group_adf_does_not_reject(self) -> None:
        df = _independent_panel(8, 80, 2)
        res = pedroni(df, y="y", X=["x"], id="id", time="t", n_boot=49, seed=2)
        assert res.decision("group_adf") == "no_cointegration"

    def test_independent_walks_westerlund_does_not_reject(self) -> None:
        df = _independent_panel(8, 80, 2)
        res = westerlund(df, y="y", X=["x"], id="id", time="t", n_boot=49, seed=2)
        assert res.decision("group_tau") == "no_cointegration"


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        df = _cointegrated_panel(6, 60, 3)
        r1 = pedroni(df, y="y", X=["x"], id="id", time="t", n_boot=19, seed=42)
        r2 = pedroni(df, y="y", X=["x"], id="id", time="t", n_boot=19, seed=42)
        assert r1.panel_adf == r2.panel_adf
        assert r1.critical_values == r2.critical_values
