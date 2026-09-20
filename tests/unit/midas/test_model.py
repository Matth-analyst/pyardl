"""Spec 40 §5 — plan de tests MIDAS-ARDL.

La calibration Monte Carlo précise n'est pas incluse ici (pas de test de
taille/puissance dans le plan de la spec — MIDAS-ARDL n'est pas un test
d'hypothèse, c'est un estimateur). Ici : contrat d'API, récupération de
theta/poids sur un DGP MIDAS connu, cohérence descendante m=1 (§5.2,
verrou exact), et comparaison à l'agrégation naïve (§5.3).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.core import ARDL
from pyardl.midas import MIDASARDL, midas_weights


def _midas_dgp(n_lf: int, m: int, seed: int, k_max: int = 8, theta=(0.0, -0.15)):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x_hf = np.cumsum(rng.standard_normal(n_lf * m))
    w_true = midas_weights(theta, k_max=k_max, form="almon_exp")
    z_true = np.full(n_lf, np.nan)
    for t in range(n_lf):
        idx = (t + 1) * m - 1
        if idx - k_max >= 0:
            z_true[t] = w_true @ x_hf[idx - k_max : idx + 1][::-1]
    y = np.zeros(n_lf)
    for t in range(1, n_lf):
        if not np.isnan(z_true[t - 1]):
            y[t] = y[t - 1] - 0.5 * (y[t - 1] - 1.5 * z_true[t - 1])
            y[t] += rng.standard_normal() * 0.3
        else:
            y[t] = y[t - 1]
    return y, x_hf, w_true


class TestMidasWeights:
    def test_sums_to_one(self) -> None:
        for form in ("almon_exp", "beta"):
            w = midas_weights([0.1, -0.2], k_max=10, form=form)  # type: ignore[arg-type]
            assert w.sum() == pytest.approx(1.0)

    def test_invalid_form_raises(self) -> None:
        with pytest.raises(ValueError, match="form"):
            midas_weights([0.0, 0.0], k_max=5, form="bogus")  # type: ignore[arg-type]

    def test_invalid_theta_length_raises(self) -> None:
        with pytest.raises(ValueError, match="theta"):
            midas_weights([0.0, 0.0, 0.0], k_max=5)

    def test_negative_k_max_raises(self) -> None:
        with pytest.raises(ValueError, match="k_max"):
            midas_weights([0.0, 0.0], k_max=-1)

    def test_decreasing_weights_for_negative_theta2(self) -> None:
        w = midas_weights([0.0, -0.3], k_max=8, form="almon_exp")
        assert np.all(np.diff(w) <= 0)


class TestAPIAndValidation:
    def test_invalid_freq_ratio_raises(self) -> None:
        y, x_hf, _ = _midas_dgp(50, 3, 0)
        with pytest.raises(ValueError, match="freq_ratio"):
            MIDASARDL(y, x_hf, freq_ratio=0)

    def test_mismatched_length_raises(self) -> None:
        y, x_hf, _ = _midas_dgp(50, 3, 0)
        with pytest.raises(ValueError, match="freq_ratio"):
            MIDASARDL(y, x_hf[:-5], freq_ratio=3)

    def test_invalid_form_raises(self) -> None:
        y, x_hf, _ = _midas_dgp(50, 3, 0)
        with pytest.raises(ValueError, match="form"):
            MIDASARDL(y, x_hf, freq_ratio=3, form="bogus")  # type: ignore[arg-type]

    def test_summary_contains_key_fields(self) -> None:
        y, x_hf, _ = _midas_dgp(80, 3, 1)
        res = MIDASARDL(y, x_hf, freq_ratio=3, k_max=6, order=(1, 1)).fit()
        text = res.summary()
        assert "MIDAS-ARDL" in text
        assert "theta_hat" in text


class TestParameterRecovery:
    """Spec 40 §5.1 — theta_hat et les poids retrouvent la vraie forme."""

    def test_longrun_recovered(self) -> None:
        y, x_hf, _ = _midas_dgp(200, 3, 2)
        res = MIDASARDL(y, x_hf, freq_ratio=3, k_max=8, order=(1, 1)).fit()
        assert res.longrun == pytest.approx(1.5, abs=0.3)

    def test_weight_shape_decreasing(self) -> None:
        y, x_hf, w_true = _midas_dgp(200, 3, 3)
        res = MIDASARDL(y, x_hf, freq_ratio=3, k_max=8, order=(1, 1)).fit()
        assert res.weights[0] > res.weights[-1]
        assert np.corrcoef(res.weights, w_true)[0, 1] > 0.8


class TestDownstreamConsistency:
    """Spec 40 §5.2 — m=1, k_max=0 : identité exacte avec ARDL."""

    def test_degenerate_case_matches_ardl(self) -> None:
        rng = np.random.default_rng(4)
        n = 200
        x = np.cumsum(rng.standard_normal(n))
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = y[t - 1] - 0.4 * (y[t - 1] - 1.5 * x[t - 1])
            y[t] += rng.standard_normal() * 0.3
        res = MIDASARDL(y, x, freq_ratio=1, k_max=0, order=(1, 1)).fit()
        ardl = ARDL(y, x, order=(1, 1)).fit()
        assert res.longrun == pytest.approx(
            float(ardl.longrun["theta"].iloc[0]), abs=1e-6
        )
        assert list(res.weights) == pytest.approx([1.0])


class TestNaiveAggregationComparison:
    """Spec 40 §5.3 — MIDAS bat l'agrégation naïve quand les poids décroissent vite."""

    def test_midas_beats_naive_averaging(self) -> None:
        y, x_hf, _ = _midas_dgp(200, 3, 5, theta=(0.0, -0.8))
        res = MIDASARDL(y, x_hf, freq_ratio=3, k_max=8, order=(1, 1)).fit()

        n_lf = y.shape[0]
        z_naive = np.array([x_hf[t * 3 : (t + 1) * 3].mean() for t in range(n_lf)])
        p, q = 1, 1
        start = max(p, q, 1)
        dy = np.diff(y)
        target = dy[start - 1 :]
        design = np.column_stack(
            [
                y[start - 1 : n_lf - 1],
                z_naive[start - 1 : n_lf - 1],
                np.ones_like(target),
            ]
        )
        beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        resid_naive = target - design @ beta
        ssr_naive = float(resid_naive @ resid_naive)

        assert res.ssr < ssr_naive
