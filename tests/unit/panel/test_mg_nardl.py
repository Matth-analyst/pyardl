"""Spec 30 §5 — plan de tests Mean-Group NARDL.

La calibration de taille précise (§5.2, taux de rejet du test de
symétrie groupe sous un DGP symétrique) est laissée à un futur
``validation/spec30_montecarlo.py`` (Étape 4) — chaque réplication
relançant N estimations NARDL, un Monte Carlo complet coûte cher. Ici :
récupération de l'asymétrie commune (§5.1), hétérogénéité (§5.3),
cohérence descendante à N=1 (§5.4), et contrat d'API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.nardl import NARDL
from pyardl.panel import MeanGroupNARDL


def _asymmetric_panel(
    n_units: int,
    n_obs: int,
    seed: int,
    theta_pos: float = 2.0,
    theta_neg: float = 0.5,
    lam_mean: float = -0.5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        lam = lam_mean + 0.05 * rng.standard_normal()
        x = np.cumsum(rng.standard_normal(n_obs))
        dx = np.diff(x, prepend=x[0])
        xpos = np.cumsum(np.maximum(dx, 0.0))
        xneg = np.cumsum(np.minimum(dx, 0.0))
        y = np.zeros(n_obs)
        for t in range(1, n_obs):
            y[t] = (
                y[t - 1]
                + lam * (y[t - 1] - theta_pos * xpos[t - 1] - theta_neg * xneg[t - 1])
                + rng.normal(scale=0.2)
            )
        rows.append(pd.DataFrame({"id": i, "t": np.arange(n_obs), "y": y, "x": x}))
    return pd.concat(rows, ignore_index=True)


def _symmetric_panel(n_units: int, n_obs: int, seed: int) -> pd.DataFrame:
    return _asymmetric_panel(n_units, n_obs, seed, theta_pos=1.0, theta_neg=1.0)


class TestAPIAndValidation:
    def test_pooled_decomposition_not_implemented(self) -> None:
        df = _symmetric_panel(5, 40, 0)
        with pytest.raises(NotImplementedError, match="pooled"):
            MeanGroupNARDL(
                df,
                y="y",
                X=["x"],
                asym=["x"],
                id="id",
                time="t",
                decomposition="pooled",
            )

    def test_invalid_decomposition_raises(self) -> None:
        df = _symmetric_panel(5, 40, 0)
        with pytest.raises(ValueError, match="decomposition"):
            MeanGroupNARDL(
                df,
                y="y",
                X=["x"],
                asym=["x"],
                id="id",
                time="t",
                decomposition="bogus",  # type: ignore[arg-type]
            )

    def test_empty_asym_raises(self) -> None:
        df = _symmetric_panel(5, 40, 0)
        with pytest.raises(ValueError, match="asym"):
            MeanGroupNARDL(df, y="y", X=["x"], asym=[], id="id", time="t")

    def test_invalid_aggregator_raises(self) -> None:
        df = _symmetric_panel(5, 40, 0)
        with pytest.raises(ValueError, match="aggregator"):
            MeanGroupNARDL(
                df,
                y="y",
                X=["x"],
                asym=["x"],
                id="id",
                time="t",
                aggregator="bogus",  # type: ignore[arg-type]
            )

    def test_summary_and_heterogeneity_run(self) -> None:
        df = _asymmetric_panel(12, 60, 1)
        res = MeanGroupNARDL(
            df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
        ).fit()
        assert "Mean-Group NARDL" in res.summary()
        het = res.heterogeneity()
        assert "mean" in het.columns


class TestAsymmetryRecovery:
    """Spec 30 §5.1 — asymétrie commune retrouvée par le Mean-Group."""

    def test_common_asymmetry_recovered(self) -> None:
        df = _asymmetric_panel(20, 150, 2, theta_pos=2.0, theta_neg=0.5)
        res = MeanGroupNARDL(
            df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
        ).fit()
        row = res.longrun_asym.loc["x"]
        assert row["theta_pos"] == pytest.approx(2.0, abs=0.1)
        assert row["theta_neg"] == pytest.approx(0.5, abs=0.1)
        assert row["pvalue_diff"] < 0.01

    def test_symmetric_dgp_does_not_reject(self) -> None:
        df = _symmetric_panel(20, 150, 3)
        res = MeanGroupNARDL(
            df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
        ).fit()
        assert res.longrun_asym.loc["x", "pvalue_diff"] > 0.05


class TestHeterogeneity:
    """Spec 30 §2.5/§5.3 — share_asymmetric distingue moyenne et généralisation."""

    def test_share_asymmetric_high_under_common_asymmetry(self) -> None:
        df = _asymmetric_panel(15, 150, 4, theta_pos=3.0, theta_neg=-1.0)
        res = MeanGroupNARDL(
            df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
        ).fit()
        assert res.share_asymmetric().loc["x"] > 0.5

    def test_share_asymmetric_low_under_symmetric_dgp(self) -> None:
        df = _symmetric_panel(15, 150, 5)
        res = MeanGroupNARDL(
            df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
        ).fit()
        assert res.share_asymmetric().loc["x"] < 0.5


class TestDownstreamConsistency:
    """Spec 30 §5.4 — a N=1, MeanGroupNARDL doit degenerer proprement.

    Le Mean-Group lui-meme n'est pas defini a N=1 (pas de dispersion
    inter-individus, meme discipline que MeanGroup, spec 22) : le verrou
    ici est que le refus explicite arrive, et que sur exactement les
    memes donnees, l'appel direct a NARDL produit le meme theta_i que
    celui gardé dans res.individual quand N>=2.
    """

    def test_n_equals_1_raises(self) -> None:
        df = _asymmetric_panel(1, 100, 6)
        with pytest.raises(ValueError, match="fewer than two"):
            MeanGroupNARDL(
                df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
            ).fit()

    def test_individual_theta_i_matches_direct_nardl_call(self) -> None:
        df = _asymmetric_panel(6, 100, 7)
        res = MeanGroupNARDL(
            df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
        ).fit()
        unit0 = df[df["id"] == 0]
        direct = NARDL(unit0["y"], unit0[["x"]], asym=["x"], order=(1, 1)).fit()
        packaged = res.individual[0]
        pd.testing.assert_series_equal(
            direct.longrun_asym.loc["x"], packaged.longrun_asym.loc["x"]
        )


class TestDiagnostics:
    def test_non_adjusting_named_not_dropped(self) -> None:
        df = _asymmetric_panel(15, 100, 8)
        res = MeanGroupNARDL(
            df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
        ).fit()
        assert res.n_units == 15
        assert isinstance(res.non_adjusting, pd.Index)

    def test_failed_individual_reported(self) -> None:
        df = _asymmetric_panel(10, 100, 9)
        # Corrupt one individual with a constant y: NARDL should fail on it.
        df.loc[df["id"] == 3, "y"] = 0.0
        res = MeanGroupNARDL(
            df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
        ).fit()
        assert res.n_units <= 10


class TestStructuralRecoupment:
    """Spec 30 §5.5 — pas de référence externe unique connue pour Panel NARDL.

    Recoupement structurel seulement (même honnêteté que spec 24 pour la
    détection de rang) : régénérer à la main la moyenne group des
    theta_i individuels obtenus par des appels séparés à NARDL, et
    vérifier l'égalité à res.longrun_asym.
    """

    def test_group_mean_matches_manual_recomputation(self) -> None:
        df = _asymmetric_panel(10, 80, 10)
        res = MeanGroupNARDL(
            df, y="y", X=["x"], asym=["x"], id="id", time="t", order=(1, 1)
        ).fit()

        manual_pos = []
        manual_neg = []
        for unit_id in sorted(df["id"].unique()):
            block = df[df["id"] == unit_id]
            fit = NARDL(block["y"], block[["x"]], asym=["x"], order=(1, 1)).fit()
            manual_pos.append(fit.longrun_asym.loc["x", "theta_pos"])
            manual_neg.append(fit.longrun_asym.loc["x", "theta_neg"])

        assert res.longrun_asym.loc["x", "theta_pos"] == pytest.approx(
            float(np.mean(manual_pos))
        )
        assert res.longrun_asym.loc["x", "theta_neg"] == pytest.approx(
            float(np.mean(manual_neg))
        )
