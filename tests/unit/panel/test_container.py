"""Spec 22 §2.1 — le conteneur panel, brique commune aux specs 22/23/24.

Un panel casse produit des nombres plausibles. C'est la raison d'etre de
ce conteneur : chacune des choses qu'il refuse de faire en silence est
une facon d'obtenir un resultat qui a l'air juste et ne l'est pas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.panel import PanelData, panel_from_frame


def _frame(
    n_units: int = 4, n_obs: int = 40, seed: int = 0, start: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        rows.append(
            pd.DataFrame(
                {
                    "id": f"u{i}",
                    "t": np.arange(start, start + n_obs),
                    "y": np.cumsum(rng.normal(size=n_obs)),
                    "x": np.cumsum(rng.normal(size=n_obs)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _build(df: pd.DataFrame, **kw: object) -> PanelData:
    kw.setdefault("warn_short", False)
    return panel_from_frame(df, y="y", x=["x"], id_col="id", time_col="t", **kw)  # type: ignore[arg-type]


class TestSplitting:
    def test_one_unit_per_individual(self) -> None:
        panel = _build(_frame(n_units=5))
        assert panel.n_units == 5
        assert len(panel) == 5
        assert panel.keys == ("u0", "u1", "u2", "u3", "u4")

    def test_series_carry_names_and_time_index(self) -> None:
        panel = _build(_frame())
        unit = panel["u0"]
        assert unit.y.name == "y"
        assert list(unit.x.columns) == ["x"]
        assert unit.y.index.name == "t"
        assert unit.x.index.equals(unit.y.index)

    def test_lookup_by_key(self) -> None:
        panel = _build(_frame())
        assert panel["u2"].key == "u2"
        with pytest.raises(KeyError, match="not a retained individual"):
            panel["nope"]

    def test_iteration_order_is_the_estimation_order(self) -> None:
        panel = _build(_frame())
        assert [u.key for u in panel] == list(panel.keys)


class TestRowOrderIndependence:
    def test_shuffled_rows_give_the_same_panel(self) -> None:
        """§3.3 — les resultats doivent etre invariants a l'ordre des
        lignes. Un modele dynamique lit ses retards sur l'ordre des
        lignes : si le conteneur ne triait pas, un panel melange
        produirait des retards des mauvaises observations, et rien en
        aval ne pourrait le detecter."""
        df = _frame(seed=1)
        shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        a, b = _build(df), _build(shuffled)
        assert a.keys == b.keys
        for key in a.keys:
            pd.testing.assert_series_equal(a[key].y, b[key].y)
            pd.testing.assert_frame_equal(a[key].x, b[key].x)

    def test_reversed_time_is_restored(self) -> None:
        df = _frame(seed=2)
        reversed_df = df.iloc[::-1].reset_index(drop=True)
        panel = _build(reversed_df)
        assert panel["u0"].y.index.is_monotonic_increasing


class TestUnbalanced:
    def test_unbalanced_panel_is_accepted(self) -> None:
        """T_i variables : le cadre de Pesaran-Smith ne demande pas un
        panel cylindre, il demande T grand."""
        df = _frame(n_units=3, n_obs=40)
        df = df[~((df["id"] == "u1") & (df["t"] >= 30))]
        panel = _build(df)
        assert panel.n_units == 3
        assert panel.unbalanced
        assert panel.sample_sizes["u1"] == 30

    def test_balanced_panel_says_so(self) -> None:
        assert not _build(_frame()).unbalanced

    def test_sample_sizes_reported(self) -> None:
        panel = _build(_frame(n_units=3, n_obs=25))
        assert panel.sample_sizes.tolist() == [25, 25, 25]


class TestRefusals:
    def test_internal_gap_excludes_rather_than_bridges(self) -> None:
        """Un trou interne n'est pas un historique plus court. Combler
        le trou apparierait des observations distantes de deux periodes
        en les appelant un retard d'une periode."""
        df = _frame(n_units=3, n_obs=40)
        df.loc[(df["id"] == "u1") & (df["t"] == 20), "y"] = np.nan
        panel = _build(df)
        assert "u1" in panel.excluded
        assert "inside the sample" in panel.excluded["u1"]
        assert panel.n_units == 2

    def test_leading_nans_are_trimmed_not_fatal(self) -> None:
        df = _frame(n_units=2, n_obs=40)
        df.loc[(df["id"] == "u1") & (df["t"] < 3), "x"] = np.nan
        panel = _build(df)
        assert panel.n_units == 2
        assert panel.sample_sizes["u1"] == 37

    def test_duplicate_periods_are_refused(self) -> None:
        df = _frame(n_units=2, n_obs=40)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate periods"):
            _build(df)

    def test_constant_series_excluded_with_the_reason(self) -> None:
        df = _frame(n_units=3, n_obs=40)
        df.loc[df["id"] == "u2", "x"] = 1.0
        panel = _build(df)
        assert "constant series" in panel.excluded["u2"]

    def test_too_short_excluded_with_the_reason(self) -> None:
        df = _frame(n_units=3, n_obs=40)
        df = df[~((df["id"] == "u0") & (df["t"] >= 8))]
        panel = _build(df, min_obs=15)
        assert "below min_obs" in panel.excluded["u0"]

    def test_every_exclusion_is_accounted_for(self) -> None:
        """Le N d'un tableau de resultats doit toujours pouvoir se
        justifier : combien sont partis, lesquels, et pourquoi."""
        df = _frame(n_units=4, n_obs=40)
        df.loc[df["id"] == "u1", "y"] = 2.0
        df = df[~((df["id"] == "u2") & (df["t"] >= 5))]
        panel = _build(df)
        assert panel.n_units == 2
        assert set(panel.excluded) == {"u1", "u2"}
        assert all(isinstance(v, str) and v for v in panel.excluded.values())

    def test_nothing_survives_is_an_error(self) -> None:
        df = _frame(n_units=2, n_obs=40)
        df["x"] = 1.0
        with pytest.raises(ValueError, match="No individual survived"):
            _build(df)

    def test_missing_column_named(self) -> None:
        with pytest.raises(ValueError, match=r"\['gdp'\] are not in the DataFrame"):
            panel_from_frame(
                _frame(), y="gdp", x=["x"], id_col="id", time_col="t", warn_short=False
            )

    def test_no_regressor(self) -> None:
        with pytest.raises(ValueError, match="a panel ARDL with no regressor"):
            panel_from_frame(
                _frame(), y="y", x=[], id_col="id", time_col="t", warn_short=False
            )

    def test_not_a_dataframe(self) -> None:
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            panel_from_frame(
                [1, 2, 3],  # type: ignore[arg-type]
                y="y",
                x=["x"],
                id_col="id",
                time_col="t",
            )


class TestWarnings:
    def test_short_panel_warns_about_the_regime(self) -> None:
        """MG est convergent quand T grandit. A petit T chaque estimation
        individuelle porte le biais des panels dynamiques, et moyenner N
        estimations ne retire pas un biais qu'elles partagent."""
        with pytest.warns(PyardlMethodologyWarning, match="does not remove a bias"):
            panel_from_frame(
                _frame(n_units=3, n_obs=20),
                y="y",
                x=["x"],
                id_col="id",
                time_col="t",
            )

    def test_time_gaps_warn(self) -> None:
        df = _frame(n_units=2, n_obs=40)
        df = df[~((df["id"] == "u1") & (df["t"].isin([10, 11])))]
        with pytest.warns(PyardlMethodologyWarning, match="skips periods"):
            panel_from_frame(
                df, y="y", x=["x"], id_col="id", time_col="t", warn_short=False
            )

    def test_gaps_are_visible_on_the_container(self) -> None:
        df = _frame(n_units=2, n_obs=40)
        df = df[~((df["id"] == "u1") & (df["t"] == 10))]
        panel = _build(df)
        assert panel.units_with_time_gaps == ("u1",)
        assert not panel["u0"].has_time_gaps


class TestSummary:
    def test_summary_reports_shape_and_exclusions(self) -> None:
        df = _frame(n_units=3, n_obs=40)
        df.loc[df["id"] == "u2", "y"] = 3.0
        text = _build(df).summary()
        assert "2 individuals" in text
        assert "excluded 1" in text
        assert "u2" in text

    def test_summary_flags_gaps(self) -> None:
        df = _frame(n_units=2, n_obs=40)
        df = df[~((df["id"] == "u1") & (df["t"] == 10))]
        assert "time gaps" in _build(df).summary()
