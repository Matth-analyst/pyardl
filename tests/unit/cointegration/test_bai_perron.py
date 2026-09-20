"""Spec 31 §5 — plan de tests Bai-Perron.

Comme pour la spec 29, le test de taille/puissance à grand nombre de
réplications (§5.1, §5.2) vit dans ``validation/spec31_montecarlo.py``
(Étape 4). Ici : cohérence de l'algorithme de programmation dynamique,
contrat d'API, et une vérification ponctuelle de récupération des
ruptures sur un DGP simple (pas une étude de taille).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.cointegration import bai_perron
from pyardl.cointegration.bai_perron import _dp_partitions


def _two_breaks_dgp(n: int, seed: int, noise: float = 0.3):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    const = np.ones(n)
    level = np.where(
        np.arange(n) < n // 3, 1.0, np.where(np.arange(n) < 2 * n // 3, 3.0, -1.0)
    )
    y = level + rng.standard_normal(n) * noise
    return y, const.reshape(-1, 1)


def _no_break_dgp(n: int, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    const = np.ones(n)
    y = 1.0 + rng.standard_normal(n) * 0.5
    return y, const.reshape(-1, 1)


class TestAPIAndValidation:
    """Spec 31 §3 — contrat d'API et cas limites."""

    def test_udmax_not_implemented(self) -> None:
        y, x = _no_break_dgp(120, 0)
        with pytest.raises(NotImplementedError, match="udmax"):
            bai_perron(y, x, selection="udmax")  # type: ignore[arg-type]

    def test_invalid_selection_raises(self) -> None:
        y, x = _no_break_dgp(120, 0)
        with pytest.raises(ValueError, match="selection"):
            bai_perron(y, x, selection="bogus")  # type: ignore[arg-type]

    def test_invalid_trim_raises(self) -> None:
        y, x = _no_break_dgp(120, 0)
        with pytest.raises(ValueError, match="trim"):
            bai_perron(y, x, trim=0.6)

    def test_invalid_max_breaks_raises(self) -> None:
        y, x = _no_break_dgp(120, 0)
        with pytest.raises(ValueError, match="max_breaks"):
            bai_perron(y, x, max_breaks=0)

    def test_sample_too_short_raises(self) -> None:
        y, x = _no_break_dgp(10, 0)
        with pytest.raises(ValueError, match="Sample too short"):
            bai_perron(y, x, max_breaks=6)

    def test_needs_at_least_one_regressor(self) -> None:
        y = np.random.default_rng(0).standard_normal(120)
        with pytest.raises(ValueError):
            bai_perron(y, x=None)  # type: ignore[arg-type]

    def test_constant_regressor_allowed(self) -> None:
        """Contrairement à check_series, une constante seule est un x valide ici."""
        y, x = _no_break_dgp(120, 0)
        res = bai_perron(y, x, selection="ic")
        assert res.n_breaks >= 0

    def test_reproducible_with_seed_sequential(self) -> None:
        y, x = _two_breaks_dgp(150, 1)
        r1 = bai_perron(y, x, max_breaks=3, selection="sequential", n_boot=19, seed=7)
        r2 = bai_perron(y, x, max_breaks=3, selection="sequential", n_boot=19, seed=7)
        assert r1.n_breaks == r2.n_breaks
        assert r1.sequential_tests == r2.sequential_tests

    def test_summary_contains_key_fields(self) -> None:
        y, x = _no_break_dgp(120, 0)
        res = bai_perron(y, x, selection="ic")
        text = res.summary()
        assert "Bai-Perron" in text
        assert "n_breaks" in text

    def test_summary_with_breaks_and_sequential_tests(self) -> None:
        y, x = _two_breaks_dgp(240, 0, noise=0.3)
        res = bai_perron(y, x, max_breaks=4, selection="sequential", n_boot=49, seed=3)
        text = res.summary()
        assert "break fractions" in text
        assert "Sequential sup-F" in text

    def test_x_as_dataframe_and_series(self) -> None:
        import pandas as pd

        y, x = _no_break_dgp(120, 0)
        df = pd.DataFrame({"const": x[:, 0]})
        res_df = bai_perron(y, df, selection="ic")
        assert res_df.segment_params[0].index[0] == "const"

        series = pd.Series(x[:, 0], name="const")
        res_series = bai_perron(y, series, selection="ic")
        assert res_series.segment_params[0].index[0] == "const"

    def test_x_1d_array_accepted(self) -> None:
        y, x = _no_break_dgp(120, 0)
        res = bai_perron(y, x[:, 0], selection="ic")
        assert res.n_breaks >= 0

    def test_x_length_mismatch_raises(self) -> None:
        y, x = _no_break_dgp(120, 0)
        with pytest.raises(ValueError, match="Incompatible lengths"):
            bai_perron(y, x[:-5], selection="ic")

    def test_x_with_nan_raises(self) -> None:
        y, x = _no_break_dgp(120, 0)
        x = x.copy()
        x[5, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            bai_perron(y, x, selection="ic")


class TestDynamicProgramming:
    """Spec 31 §2.2/§5.1 — la recherche sur partitions retrouve les vraies ruptures."""

    def test_ssr_decreases_with_more_breaks(self) -> None:
        """SSR(m) décroît (faiblement) en m : chaque rupture ajoute des
        degrés de liberté au modèle, jamais moins."""
        y, x = _two_breaks_dgp(180, 2)
        h = int(np.ceil(0.15 * y.shape[0]))
        ssr_by_m, _ = _dp_partitions(y, x, h, max_breaks=4)
        values = [ssr_by_m[m] for m in sorted(ssr_by_m)]
        assert all(a >= b - 1e-9 for a, b in zip(values, values[1:], strict=False))

    def test_two_breaks_recovered_ic(self) -> None:
        y, x = _two_breaks_dgp(240, 0, noise=0.3)
        res = bai_perron(y, x, max_breaks=4, selection="ic", ic="bic")
        assert res.n_breaks == 2
        true_fractions = [1 / 3, 2 / 3]
        for est, true in zip(sorted(res.break_fractions), true_fractions, strict=True):
            assert abs(est - true) < 0.05

    def test_two_breaks_recovered_sequential(self) -> None:
        y, x = _two_breaks_dgp(240, 0, noise=0.3)
        res = bai_perron(y, x, max_breaks=4, selection="sequential", n_boot=49, seed=3)
        assert res.n_breaks == 2

    def test_no_break_dgp_selects_zero_ic(self) -> None:
        y, x = _no_break_dgp(180, 4)
        res = bai_perron(y, x, max_breaks=3, selection="ic")
        assert res.n_breaks == 0

    def test_segment_params_count_matches_breaks(self) -> None:
        y, x = _two_breaks_dgp(240, 3)
        res = bai_perron(y, x, max_breaks=4, selection="ic")
        assert len(res.segment_params) == res.n_breaks + 1
