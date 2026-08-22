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
from pyardl.nardl import decomposition_error, partial_sums

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
        idx = pd.date_range("2000-01-01", periods=6, freq="QE")
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
