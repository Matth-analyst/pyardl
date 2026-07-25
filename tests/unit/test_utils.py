"""Tests des briques transversales lag_matrix (spec 02 §2) et
check_series (spec 01 §6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.utils import check_series, lag_matrix


class TestLagMatrix:
    def test_basic_contemporaneous(self) -> None:
        x = np.arange(1.0, 6.0)
        out = lag_matrix(x, 2)
        expected = np.array([[3.0, 2.0, 1.0], [4.0, 3.0, 2.0], [5.0, 4.0, 3.0]])
        np.testing.assert_array_equal(out, expected)

    def test_first_lag_one(self) -> None:
        """Cas des retards de y dans un ARDL : colonnes x_{t-1}..x_{t-p}."""
        x = np.arange(1.0, 6.0)
        out = lag_matrix(x, 2, first_lag=1)
        expected = np.array([[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
        np.testing.assert_array_equal(out, expected)

    def test_zero_lags(self) -> None:
        x = np.arange(1.0, 4.0)
        out = lag_matrix(x, 0)
        np.testing.assert_array_equal(out, x[:, None])

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            lag_matrix(np.ones((3, 2)), 1)
        with pytest.raises(ValueError, match="first_lag"):
            lag_matrix(np.arange(5.0), 1, first_lag=2)
        with pytest.raises(ValueError, match="too short"):
            lag_matrix(np.arange(3.0), 3)


class TestCheckSeries:
    def test_passthrough_and_names(self) -> None:
        y = pd.Series(np.arange(20.0), name="conso")
        x = pd.DataFrame({"rev": np.arange(20.0) * 2, "px": np.arange(20.0) + 1})
        y_arr, x_arr, index, y_name, x_names = check_series(y, x)
        assert y_name == "conso"
        assert x_names == ("rev", "px")
        assert x_arr is not None and x_arr.shape == (20, 2)
        assert index is not None and len(index) == 20

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Incompatible lengths"):
            check_series(np.arange(10.0), np.arange(9.0))

    def test_edge_nan_trimmed_with_warning(self) -> None:
        """Spec 01 §6 : NaN de bord -> trim avec warning."""
        y = np.array([np.nan, 1.0, 2.0, 3.0] + list(np.linspace(4, 20, 17)))
        with pytest.warns(PyardlMethodologyWarning, match="Trimmed"):
            y_arr, *_ = check_series(y)
        assert y_arr.shape[0] == 20
        assert not np.isnan(y_arr).any()

    def test_internal_nan_raises(self) -> None:
        y = np.linspace(0, 19, 20)
        y[10] = np.nan
        with pytest.raises(ValueError, match="Internal NaN"):
            check_series(y)

    def test_small_sample_warning(self) -> None:
        with pytest.warns(PyardlMethodologyWarning, match="Very small sample"):
            check_series(np.arange(10.0))

    def test_zero_variance_raises(self) -> None:
        with pytest.raises(ValueError, match="zero variance"):
            check_series(np.ones(20))
        with pytest.raises(ValueError, match="zero variance"):
            check_series(np.arange(20.0), np.ones(20))
