"""Spec 17 §4.3 — VERROU : l'identite des sommes partielles.

Ecrit avant le modele. Toute la mecanique NARDL — estimation, tests de
Wald, bornes, multiplicateurs — repose sur cette decomposition. Une
erreur ici ne produirait pas une exception mais des resultats plausibles
et faux, donc elle se verrouille par une identite exacte, pas par
inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.nardl import (
    decomposition_error,
    multi_decomposition_error,
    partial_sums,
    partial_sums_multi,
)

TOL = 1e-12  # tolerance contractuelle de la spec 17 §4.3


class TestIdentity:
    """x = x_0 + x+ + x- a 1e-12, sur des series de toute nature."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_walk(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.normal(size=200))
        pos, neg = partial_sums(x)
        assert decomposition_error(x, pos, neg) < TOL

    @pytest.mark.parametrize("seed", range(5))
    def test_stationary_series(self, seed: int) -> None:
        rng = np.random.default_rng(100 + seed)
        x = rng.normal(size=150)
        pos, neg = partial_sums(x)
        assert decomposition_error(x, pos, neg) < TOL

    def test_trending_series(self) -> None:
        x = np.arange(100, dtype=float) * 0.7 + 3.0
        pos, neg = partial_sums(x)
        assert decomposition_error(x, pos, neg) < TOL

    def test_constant_series(self) -> None:
        """Aucune variation : les deux sommes partielles restent nulles."""
        x = np.full(50, 2.5)
        pos, neg = partial_sums(x)
        assert np.all(pos.to_numpy() == 0.0)
        assert np.all(neg.to_numpy() == 0.0)
        assert decomposition_error(x, pos, neg) < TOL

    def test_monotone_increasing_has_no_negative_part(self) -> None:
        x = np.cumsum(np.abs(np.random.default_rng(1).normal(size=80)) + 0.1)
        pos, neg = partial_sums(x)
        assert np.all(neg.to_numpy() == 0.0)
        assert decomposition_error(x, pos, neg) < TOL

    def test_large_scale_series(self) -> None:
        """Grandes valeurs : l'identite doit tenir en valeur RELATIVE.

        Une somme cumulee sur des nombres de l'ordre de 1e6 perd des
        chiffres significatifs ; la tolerance absolue de 1e-12 n'a alors
        plus de sens, et c'est l'erreur relative qui est verrouillee.
        """
        rng = np.random.default_rng(7)
        x = np.cumsum(rng.normal(scale=1e5, size=300)) + 1e6
        pos, neg = partial_sums(x)
        assert decomposition_error(x, pos, neg) / np.max(np.abs(x)) < TOL


class TestThreshold:
    """Le seuil non nul introduit une derive, et cela se dit."""

    def test_non_zero_threshold_breaks_the_plain_identity(self) -> None:
        rng = np.random.default_rng(3)
        x = np.cumsum(rng.normal(size=120))
        with pytest.warns(PyardlMethodologyWarning, match="linear drift"):
            pos, neg = partial_sums(x, threshold=0.25)
        # Sans le terme c*t, l'identite est massivement violee...
        assert decomposition_error(x, pos, neg, threshold=0.0) > 1.0
        # ... et exacte avec lui.
        assert decomposition_error(x, pos, neg, threshold=0.25) < TOL

    def test_mean_threshold_centres_the_increments(self) -> None:
        rng = np.random.default_rng(4)
        x = np.cumsum(rng.normal(loc=0.5, size=200))
        c = float(np.diff(x).mean())
        with pytest.warns(PyardlMethodologyWarning):
            pos, neg = partial_sums(x, threshold="mean")
        assert decomposition_error(x, pos, neg, threshold=c) < TOL
        # Increments centres : la derniere somme positive compense la negative.
        assert pos.iloc[-1] + neg.iloc[-1] == pytest.approx(0.0, abs=1e-10)

    def test_zero_threshold_is_silent(self) -> None:
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            partial_sums(np.arange(10.0))
        assert not [
            w for w in caught if issubclass(w.category, PyardlMethodologyWarning)
        ]


class TestStructure:
    """Forme, noms, index, monotonie."""

    def test_starts_at_zero(self) -> None:
        pos, neg = partial_sums(np.array([5.0, 7.0, 2.0]))
        assert pos.iloc[0] == 0.0
        assert neg.iloc[0] == 0.0

    def test_monotonicity(self) -> None:
        """x+ ne decroit jamais, x- ne croit jamais : c'est ce qui les
        rend interpretables comme cumul de hausses et de baisses."""
        rng = np.random.default_rng(5)
        pos, neg = partial_sums(np.cumsum(rng.normal(size=200)))
        assert np.all(np.diff(pos.to_numpy()) >= 0.0)
        assert np.all(np.diff(neg.to_numpy()) <= 0.0)

    def test_names_follow_the_series(self) -> None:
        pos, neg = partial_sums(pd.Series([1.0, 2.0, 1.5], name="oil"))
        assert pos.name == "oil_pos"
        assert neg.name == "oil_neg"

    def test_index_is_preserved(self) -> None:
        # pd.offsets.QuarterEnd() plutot que freq="QE" : cet alias
        # n'existe qu'a partir de pandas 2.2, alors que le projet
        # declare pandas>=2.1. L'objet offset, lui, se comporte a
        # l'identique de 2.1 a 3.0.
        idx = pd.date_range("2000-01-01", periods=6, freq=pd.offsets.QuarterEnd())
        pos, neg = partial_sums(pd.Series(np.arange(6.0), index=idx, name="p"))
        assert pos.index.equals(idx)
        assert neg.index.equals(idx)

    def test_explicit_name_overrides(self) -> None:
        pos, _ = partial_sums(pd.Series([1.0, 2.0], name="a"), name="b")
        assert pos.name == "b_pos"

    def test_unnamed_input_gets_a_default(self) -> None:
        pos, neg = partial_sums(np.array([1.0, 2.0]))
        assert (pos.name, neg.name) == ("x_pos", "x_neg")


class TestValidation:
    """Refus explicites."""

    def test_two_dimensional_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            partial_sums(np.zeros((10, 2)))

    def test_single_observation_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two observations"):
            partial_sums(np.array([1.0]))

    def test_nan_is_refused(self) -> None:
        """Un NaN empoisonnerait toutes les observations suivantes par
        la somme cumulee, sans rien signaler."""
        with pytest.raises(ValueError, match="NaN"):
            partial_sums(np.array([1.0, np.nan, 3.0]))

    def test_infinity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            partial_sums(np.array([1.0, np.inf, 3.0]))

    def test_mismatched_shapes_in_error_helper(self) -> None:
        with pytest.raises(ValueError, match="Shapes differ"):
            decomposition_error(np.zeros(5), np.zeros(4), np.zeros(5))


def _multi_regime_dgp(
    seed: int,
    n: int = 3000,
    c: float = 0.02,
    theta_small: float = 0.5,
    theta_large: float = 2.0,
    theta_neg: float = 1.0,
    lam: float = -0.5,
    noise: float = 0.15,
) -> tuple[np.ndarray, pd.DataFrame]:
    """DGP a quatre regimes : petites/grandes hausses, baisses (un seul
    seuil sur ce cote, deux bandes negatives par symetrie du helper)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(scale=0.04, size=n))
    bands = partial_sums_multi(x, thresholds=[c])
    y = np.zeros(n)
    for t in range(1, n):
        level = (
            theta_small * bands["x_pos_1"].iloc[t - 1]
            + theta_large * bands["x_pos_2"].iloc[t - 1]
            + theta_neg * bands["x_neg_1"].iloc[t - 1]
            + theta_neg * bands["x_neg_2"].iloc[t - 1]
        )
        y[t] = y[t - 1] + lam * (y[t - 1] - level) + rng.normal(scale=noise)
    return y, bands


class TestMultiIdentity:
    """x = x_0 + somme de toutes les bandes, a 1e-12."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_walk_one_threshold(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.normal(scale=0.03, size=200))
        bands = partial_sums_multi(x, thresholds=[0.01])
        assert multi_decomposition_error(x, bands) < TOL

    @pytest.mark.parametrize("seed", range(5))
    def test_random_walk_two_thresholds(self, seed: int) -> None:
        rng = np.random.default_rng(200 + seed)
        x = np.cumsum(rng.normal(scale=0.05, size=250))
        bands = partial_sums_multi(x, thresholds=[0.01, 0.03])
        assert multi_decomposition_error(x, bands) < TOL

    def test_stationary_series(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(size=150)
        bands = partial_sums_multi(x, thresholds=[0.5, 1.2])
        assert multi_decomposition_error(x, bands) < TOL

    def test_constant_series_gives_flat_bands(self) -> None:
        x = np.full(50, 2.5)
        bands = partial_sums_multi(x, thresholds=[0.1])
        assert np.all(bands.to_numpy() == 0.0)
        assert multi_decomposition_error(x, bands) < TOL


class TestMultiConsistency:
    """Le decoupage a un seuil retombe exactement sur la version binaire."""

    @pytest.mark.parametrize("seed", range(5))
    def test_one_threshold_recombines_to_partial_sums(self, seed: int) -> None:
        rng = np.random.default_rng(300 + seed)
        x = np.cumsum(rng.normal(size=150))
        pos, neg = partial_sums(x)
        bands = partial_sums_multi(x, thresholds=[0.3])
        recombined_pos = bands["x_pos_1"] + bands["x_pos_2"]
        recombined_neg = bands["x_neg_1"] + bands["x_neg_2"]
        assert np.max(np.abs(recombined_pos.to_numpy() - pos.to_numpy())) < 1e-10
        assert np.max(np.abs(recombined_neg.to_numpy() - neg.to_numpy())) < 1e-10

    def test_bounded_band_never_exceeds_its_width(self) -> None:
        """La bande bornee [c_{j-1}, c_j] ne peut jamais accumuler plus
        que c_j - c_{j-1} par periode : c'est ce qui la rend bornee."""
        rng = np.random.default_rng(9)
        x = np.cumsum(rng.normal(scale=0.5, size=300))
        bands = partial_sums_multi(x, thresholds=[0.05, 0.15])
        width = 0.15 - 0.05
        per_period = np.diff(bands["x_pos_2"].to_numpy())
        assert np.all(per_period <= width + 1e-12)
        assert np.all(per_period >= -1e-12)


class TestMultiRecovery:
    """Un DGP a coefficients distincts par bande les retrouve distincts."""

    def test_recovers_small_large_and_negative_thetas(self) -> None:
        y, bands = _multi_regime_dgp(seed=3)
        n = len(y)
        design = np.column_stack(
            [
                np.ones(n - 1),
                y[:-1],
                bands["x_pos_1"].iloc[:-1].to_numpy(),
                bands["x_pos_2"].iloc[:-1].to_numpy(),
                bands["x_neg_1"].iloc[:-1].to_numpy(),
                bands["x_neg_2"].iloc[:-1].to_numpy(),
            ]
        )
        coef, *_ = np.linalg.lstsq(design, y[1:], rcond=None)
        _, phi, g_small, g_large, g_neg1, g_neg2 = coef
        lam_hat = phi - 1.0
        theta_small = -g_small / lam_hat
        theta_large = -g_large / lam_hat
        theta_neg1 = -g_neg1 / lam_hat
        theta_neg2 = -g_neg2 / lam_hat

        assert theta_small == pytest.approx(0.5, abs=0.15)
        assert theta_large == pytest.approx(2.0, abs=0.2)
        assert theta_neg1 == pytest.approx(1.0, abs=0.15)
        assert theta_neg2 == pytest.approx(1.0, abs=0.15)
        # La distinction petite/grande hausse est bien identifiee, pas
        # seulement dans la marge d'erreur individuelle de chacune.
        assert theta_large > theta_small


class TestMultiStructure:
    """Noms, ordre des colonnes, index."""

    def test_column_names_and_order(self) -> None:
        bands = partial_sums_multi(
            pd.Series([1.0, 2.0, 1.5, 3.0], name="oil"), thresholds=[0.3, 0.8]
        )
        assert list(bands.columns) == [
            "oil_pos_1",
            "oil_pos_2",
            "oil_pos_3",
            "oil_neg_1",
            "oil_neg_2",
            "oil_neg_3",
        ]

    def test_starts_at_zero(self) -> None:
        bands = partial_sums_multi(np.array([5.0, 7.0, 2.0]), thresholds=[0.5])
        assert np.all(bands.iloc[0].to_numpy() == 0.0)

    def test_index_is_preserved(self) -> None:
        idx = pd.date_range("2000-01-01", periods=6, freq=pd.offsets.QuarterEnd())
        bands = partial_sums_multi(
            pd.Series(np.arange(6.0), index=idx, name="p"), thresholds=[1.0]
        )
        assert bands.index.equals(idx)

    def test_unnamed_input_gets_a_default(self) -> None:
        bands = partial_sums_multi(np.array([1.0, 2.0, 1.0]), thresholds=[0.3])
        assert bands.columns[0] == "x_pos_1"

    def test_explicit_name_overrides(self) -> None:
        bands = partial_sums_multi(
            pd.Series([1.0, 2.0], name="a"), thresholds=[0.3], name="b"
        )
        assert bands.columns[0] == "b_pos_1"


class TestMultiValidation:
    """Refus explicites, sans substitution silencieuse."""

    def test_empty_thresholds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            partial_sums_multi(np.array([1.0, 2.0, 3.0]), thresholds=[])

    def test_non_increasing_thresholds_are_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            partial_sums_multi(np.array([1.0, 2.0, 3.0]), thresholds=[0.5, 0.5])

    def test_decreasing_thresholds_are_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            partial_sums_multi(np.array([1.0, 2.0, 3.0]), thresholds=[0.8, 0.3])

    def test_non_positive_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            partial_sums_multi(np.array([1.0, 2.0, 3.0]), thresholds=[0.0, 0.5])

    def test_negative_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            partial_sums_multi(np.array([1.0, 2.0, 3.0]), thresholds=[-0.1])

    def test_two_dimensional_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            partial_sums_multi(np.zeros((10, 2)), thresholds=[0.1])

    def test_single_observation_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two observations"):
            partial_sums_multi(np.array([1.0]), thresholds=[0.1])

    def test_nan_is_refused(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            partial_sums_multi(np.array([1.0, np.nan, 3.0]), thresholds=[0.1])

    def test_mismatched_shapes_in_error_helper(self) -> None:
        with pytest.raises(ValueError, match="Shapes differ"):
            multi_decomposition_error(np.zeros(5), pd.DataFrame(np.zeros((4, 2))))
