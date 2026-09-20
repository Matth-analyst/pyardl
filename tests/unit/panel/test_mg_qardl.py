"""Spec 36 §5 — plan de tests Mean-Group QARDL.

Portée de cette version (documentée dans le module et
docs/DEVIATIONS.md) : seul le coeur agrégation-Mean-Group-par-tau
(spec 36 §2.1-2.2, inference='kernel') est implémenté. Les tests de
constance/symétrie groupe et l'agrégation MBB (§2.3-2.4) ne sont pas
implémentés, donc pas testés ici. La calibration de taille précise vit
dans un futur validation/spec36_montecarlo.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.panel import MeanGroupQARDL
from pyardl.qardl import QARDL


def _constant_theta_panel(
    n_units: int, n_obs: int, seed: int, theta: float = 1.5
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        x = np.cumsum(rng.standard_normal(n_obs))
        y = np.zeros(n_obs)
        for t in range(1, n_obs):
            resid = rng.standard_normal()
            y[t] = y[t - 1] - 0.4 * (y[t - 1] - theta * x[t - 1]) + resid
        rows.append(pd.DataFrame({"id": i, "t": np.arange(n_obs), "y": y, "x": x}))
    return pd.concat(rows, ignore_index=True)


class TestAPIAndValidation:
    def test_invalid_aggregator_raises(self) -> None:
        df = _constant_theta_panel(5, 80, 0)
        with pytest.raises(ValueError, match="aggregator"):
            MeanGroupQARDL(
                df,
                y="y",
                X=["x"],
                id="id",
                time="t",
                aggregator="bogus",  # type: ignore[arg-type]
            )

    def test_invalid_trim_raises(self) -> None:
        df = _constant_theta_panel(5, 80, 0)
        with pytest.raises(ValueError, match="trim"):
            MeanGroupQARDL(df, y="y", X=["x"], id="id", time="t", trim=0.6)

    def test_n_equals_1_raises(self) -> None:
        df = _constant_theta_panel(1, 80, 0)
        with pytest.raises(ValueError, match="fewer than two"):
            MeanGroupQARDL(df, y="y", X=["x"], id="id", time="t", order=(1, 1)).fit()

    def test_summary_contains_key_fields(self) -> None:
        df = _constant_theta_panel(8, 80, 1)
        res = MeanGroupQARDL(
            df,
            y="y",
            X=["x"],
            id="id",
            time="t",
            taus=(0.25, 0.5, 0.75),
            order=(1, 1),
        ).fit()
        text = res.summary()
        assert "Mean-Group QARDL" in text
        assert "0.50" in text

    def test_default_taus_used(self) -> None:
        df = _constant_theta_panel(8, 80, 1)
        res = MeanGroupQARDL(df, y="y", X=["x"], id="id", time="t", order=(1, 1)).fit()
        assert len(res.taus) >= 3


class TestRecovery:
    """Spec 36 §5.2 — DGP à effet constant sur tau : la surface est plate."""

    def test_constant_theta_recovered_at_every_tau(self) -> None:
        df = _constant_theta_panel(15, 150, 2, theta=1.5)
        res = MeanGroupQARDL(
            df,
            y="y",
            X=["x"],
            id="id",
            time="t",
            taus=(0.1, 0.25, 0.5, 0.75, 0.9),
            order=(1, 1),
        ).fit()
        table = res.longrun_mg["x"]
        assert (table["theta"] - 1.5).abs().max() < 0.15
        assert (table["pvalue"] < 0.01).all()


class TestDownstreamConsistency:
    """Spec 36 §5.3 — cohérence descendante et recoupement structurel (§5.4)."""

    def test_individual_matches_direct_qardl_call(self) -> None:
        df = _constant_theta_panel(6, 100, 3)
        res = MeanGroupQARDL(
            df,
            y="y",
            X=["x"],
            id="id",
            time="t",
            taus=(0.25, 0.5, 0.75),
            order=(1, 1),
        ).fit()
        unit0 = df[df["id"] == 0]
        direct = QARDL(
            unit0["y"], unit0[["x"]], order=(1, 1), taus=(0.25, 0.5, 0.75)
        ).fit(inference="kernel")
        packaged = res.individual[0]
        pd.testing.assert_frame_equal(
            direct.longrun(variable="x"), packaged.longrun(variable="x")
        )

    def test_group_mean_matches_manual_recomputation(self) -> None:
        df = _constant_theta_panel(8, 100, 4)
        taus = (0.25, 0.5, 0.75)
        res = MeanGroupQARDL(
            df, y="y", X=["x"], id="id", time="t", taus=taus, order=(1, 1)
        ).fit()

        manual = []
        for unit_id in sorted(df["id"].unique()):
            block = df[df["id"] == unit_id]
            fit = QARDL(block["y"], block[["x"]], order=(1, 1), taus=taus).fit(
                inference="kernel"
            )
            manual.append(fit.longrun(variable="x")["x"].to_numpy())
        manual_mean = np.mean(np.stack(manual), axis=0)
        np.testing.assert_allclose(res.longrun_mg["x"]["theta"].to_numpy(), manual_mean)


class TestDiagnostics:
    def test_failed_individual_reported(self) -> None:
        df = _constant_theta_panel(10, 100, 5)
        # Shrink unit 3 to a handful of rows: panel_from_frame's default
        # min_obs would exclude it outright, so min_obs is lowered here
        # to let it reach fit(), where QARDL then fails on too few
        # observations for the requested order — the failure path this
        # test targets, distinct from panel-level exclusion.
        keep = df["id"] != 3
        tiny = df[df["id"] == 3].head(3)
        df = pd.concat([df[keep], tiny], ignore_index=True)
        with pytest.warns(match="could not be fitted"):
            res = MeanGroupQARDL(
                df, y="y", X=["x"], id="id", time="t", order=(1, 1), min_obs=3
            ).fit()
        assert 3 in res.failed
        assert "could not be fitted" in res.summary()
