"""Spec 24 §2.1/§2.4 — moyennes transversales et test CD.

Deux briques transversales, et deux facons distinctes de se tromper avec
elles. Les moyennes peuvent changer de SENS sans changer de nom quand le
panel n'est pas cylindre ; le test CD peut se lire a l'envers, puisque
la meme p-value veut dire des choses opposees avant et apres
augmentation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.panel import cd_test, cross_section_averages, default_cs_lags


def _panel(n_units: int = 6, n_obs: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        rows.append(
            pd.DataFrame(
                {
                    "id": f"u{i}",
                    "t": np.arange(n_obs),
                    "y": rng.normal(size=n_obs),
                    "x": rng.normal(size=n_obs),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class TestDefaultLags:
    def test_cube_root_rule(self) -> None:
        assert default_cs_lags(27) == 3
        assert default_cs_lags(125) == 5

    def test_perfect_cubes_are_not_lost_to_floating_point(self) -> None:
        """`t ** (1/3)` tombe JUSTE SOUS l'entier aux cubes parfaits, et
        le plancher perd alors un retard. Mesure : 64 ** (1/3) vaut
        3.99999999999999956 et 1000 ** (1/3) vaut 9.99999999999999822,
        donc la forme naive rendrait 3 et 9. Une liste de retards plus
        courte est une autre specification, pas un detail d'arrondi."""
        for t in (8, 27, 64, 125, 216, 343, 1000):
            naive = int(np.floor(t ** (1 / 3)))
            exact = default_cs_lags(t)
            assert exact == round(t ** (1 / 3))
            if t in (64, 216, 343, 1000):
                assert naive == exact - 1

    def test_refuses_a_non_positive_sample(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            default_cs_lags(0)


class TestAverages:
    def test_mean_over_individuals_at_each_date(self) -> None:
        df = pd.DataFrame(
            {"id": ["a", "a", "b", "b"], "t": [0, 1, 0, 1], "y": [1.0, 3.0, 3.0, 5.0]}
        )
        out = cross_section_averages(df, ["y"], "id", "t")
        assert out["cs_y"].tolist() == [2.0, 4.0]

    def test_lags_are_shifts_of_the_average(self) -> None:
        df = _panel()
        out = cross_section_averages(df, ["y", "x"], "id", "t", lags=2)
        assert list(out.columns) == [
            "cs_y",
            "cs_x",
            "cs_y_L1",
            "cs_x_L1",
            "cs_y_L2",
            "cs_x_L2",
            "cs_count",
        ]
        assert out["cs_y_L1"].iloc[1] == pytest.approx(out["cs_y"].iloc[0])
        assert np.isnan(out["cs_y_L1"].iloc[0])

    def test_count_is_reported(self) -> None:
        """Dans un panel non cylindre, une moyenne sur 40 pays et une
        moyenne sur 12 ne sont pas le meme objet. Le compte est la pour
        que la difference soit visible."""
        df = _panel(n_units=5, n_obs=20)
        df = df[~((df["id"] == "u4") & (df["t"] >= 10))]
        out = cross_section_averages(df, ["y"], "id", "t", warn_composition=False)
        assert out["cs_count"].iloc[0] == 5
        assert out["cs_count"].iloc[-1] == 4

    def test_unbalanced_averages_over_those_present(self) -> None:
        df = pd.DataFrame({"id": ["a", "a", "b"], "t": [0, 1, 0], "y": [1.0, 3.0, 3.0]})
        out = cross_section_averages(df, ["y"], "id", "t", warn_composition=False)
        assert out["cs_y"].tolist() == [2.0, 3.0]
        assert out["cs_count"].tolist() == [2, 1]

    def test_sharply_varying_composition_warns(self) -> None:
        df = _panel(n_units=6, n_obs=20)
        df = df[~(df["id"].isin(["u3", "u4", "u5"]) & (df["t"] >= 10))]
        with pytest.warns(PyardlMethodologyWarning, match="composition"):
            cross_section_averages(df, ["y"], "id", "t")

    def test_weighted_average(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "t": [0, 0],
                "y": [1.0, 3.0],
                "w": [3.0, 1.0],
            }
        )
        out = cross_section_averages(
            df, ["y"], "id", "t", weights="w", warn_composition=False
        )
        assert out["cs_y"].iloc[0] == pytest.approx(1.5)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"lags": -1}, "lags must be non-negative"),
            ({"variables": []}, "variables is empty"),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        df = _panel()
        call = {"variables": ["y"], "id_col": "id", "time_col": "t"}
        call.update(kwargs)
        with pytest.raises(ValueError, match=match):
            cross_section_averages(df, **call)  # type: ignore[arg-type]

    def test_missing_column_named(self) -> None:
        with pytest.raises(ValueError, match=r"\['gdp'\] are not in"):
            cross_section_averages(_panel(), ["gdp"], "id", "t")

    def test_non_positive_weights_refused(self) -> None:
        """Un poids nul retire un individu de la moyenne sans le dire, un
        poids negatif ne produit pas une moyenne du tout."""
        df = _panel(n_units=3, n_obs=5)
        df["w"] = 1.0
        df.loc[0, "w"] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            cross_section_averages(df, ["y"], "id", "t", weights="w")


class TestCDTest:
    def test_detects_a_common_factor(self) -> None:
        rng = np.random.default_rng(0)
        f = rng.normal(size=80)
        dep = pd.DataFrame({f"u{i}": f + 0.3 * rng.normal(size=80) for i in range(10)})
        result = cd_test(dep)
        assert result.rejects
        assert result.mean_abs_correlation > 0.5

    def test_silent_on_independent_residuals(self) -> None:
        rng = np.random.default_rng(1)
        ind = pd.DataFrame({f"u{i}": rng.normal(size=80) for i in range(10)})
        assert not cd_test(ind).rejects

    def test_pairs_are_matched_on_the_index_not_by_position(self) -> None:
        """Deux individus dont les fenetres different ne doivent pas voir
        leurs residus alignes par numero de ligne : ce serait correler
        des dates differentes."""
        rng = np.random.default_rng(2)
        f = rng.normal(size=60)
        a = pd.Series(f + 0.2 * rng.normal(size=60), index=range(60))
        # b decale : memes dates 20..59, mais pas les memes lignes.
        b = pd.Series((f + 0.2 * rng.normal(size=60))[20:], index=range(20, 60))
        aligned = cd_test({"a": a, "b": b})
        shifted = cd_test(
            {"a": a.reset_index(drop=True), "b": b.reset_index(drop=True)}
        )
        assert aligned.statistic != shifted.statistic
        assert aligned.mean_abs_correlation > shifted.mean_abs_correlation

    def test_summary_states_the_direction_before(self) -> None:
        """La meme p-value veut dire des choses opposees avant et apres
        augmentation ; le resume doit dire laquelle s'applique."""
        rng = np.random.default_rng(3)
        f = rng.normal(size=80)
        dep = pd.DataFrame({f"u{i}": f + 0.3 * rng.normal(size=80) for i in range(8)})
        text = cd_test(dep).summary(context="before")
        assert "augmentation is warranted" in text

    def test_summary_states_the_direction_after(self) -> None:
        rng = np.random.default_rng(4)
        f = rng.normal(size=80)
        dep = pd.DataFrame({f"u{i}": f + 0.3 * rng.normal(size=80) for i in range(8)})
        text = cd_test(dep).summary(context="after")
        assert "SURVIVES the augmentation" in text

    def test_summary_after_on_clean_residuals(self) -> None:
        rng = np.random.default_rng(5)
        ind = pd.DataFrame({f"u{i}": rng.normal(size=80) for i in range(8)})
        text = cd_test(ind).summary(context="after")
        assert "absorbed the dependence" in text

    def test_mean_absolute_correlation_separates_zero_from_cancelling(self) -> None:
        """Une statistique proche de zero peut venir de correlations
        nulles ou de correlations qui s'annulent. Les deux appellent des
        conclusions differentes, donc le module rend les deux nombres."""
        rng = np.random.default_rng(6)
        f = rng.normal(size=200)
        # Chargements de signes opposes : la somme s'annule, les
        # correlations individuelles restent fortes.
        cols = {}
        for i in range(8):
            sign = 1.0 if i % 2 == 0 else -1.0
            cols[f"u{i}"] = sign * f + 0.2 * rng.normal(size=200)
        result = cd_test(pd.DataFrame(cols))
        assert abs(result.statistic) < 30
        assert result.mean_abs_correlation > 0.8

    def test_constant_or_short_individuals_are_excluded_with_a_reason(self) -> None:
        rng = np.random.default_rng(7)
        frame = pd.DataFrame({f"u{i}": rng.normal(size=40) for i in range(5)})
        frame["flat"] = 1.0
        frame["short"] = np.nan
        frame.loc[:2, "short"] = rng.normal(size=3)
        result = cd_test(frame)
        assert "flat" in result.dropped
        assert "short" in result.dropped
        assert result.n_units == 5

    def test_the_fast_and_general_paths_agree(self) -> None:
        """Sans valeur manquante, la statistique se calcule en une seule
        matrice de correlation ; avec, il faut passer paire par paire.
        Les deux doivent rendre le MEME nombre, sinon la statistique
        dependrait de la presence d'un trou ailleurs dans le panel.

        Le chemin rapide existe pour une raison mesuree : a N = 60 le
        chemin general prenait 3.4 s et le rapide en prend 18 ms, soit
        pres de 200 fois moins. Un test CD trop lent n'est pas utilise,
        et un diagnostic qu'on n'execute pas ne protege de rien.
        """
        rng = np.random.default_rng(40)
        f = rng.normal(size=90)
        frame = pd.DataFrame({f"u{i}": f + 0.4 * rng.normal(size=90) for i in range(9)})

        fast = cd_test(frame)

        # Meme donnees, mais une ligne ENTIERE retiree : chaque paire a
        # exactement le meme recouvrement, donc la statistique doit etre
        # celle du frame complet ampute de cette date — c'est-a-dire
        # celle du chemin general sur le meme sous-echantillon.
        trimmed = frame.iloc[1:]
        with_hole = frame.copy()
        with_hole.iloc[0, :] = np.nan
        general = cd_test(with_hole)
        reference = cd_test(trimmed)
        assert general.statistic == pytest.approx(reference.statistic, rel=1e-12)
        assert general.n_pairs == reference.n_pairs
        assert fast.n_pairs == reference.n_pairs

    def test_needs_two_individuals(self) -> None:
        rng = np.random.default_rng(8)
        with pytest.raises(ValueError, match="at least two usable"):
            cd_test(pd.DataFrame({"u0": rng.normal(size=30)}))

    def test_no_overlap_is_an_error(self) -> None:
        a = pd.Series(np.arange(10.0), index=range(10))
        b = pd.Series(np.arange(10.0), index=range(100, 110))
        with pytest.raises(ValueError, match="No pair"):
            cd_test({"a": a, "b": b})

    def test_result_is_immutable(self) -> None:
        rng = np.random.default_rng(9)
        result = cd_test(pd.DataFrame({f"u{i}": rng.normal(size=40) for i in range(4)}))
        with pytest.raises(AttributeError):
            result.statistic = 0.0  # type: ignore[misc]
