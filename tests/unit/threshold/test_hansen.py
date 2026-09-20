"""Spec 32 §5 — plan de tests Hansen threshold regression.

Comme pour les specs 29/31/33, la calibration de taille précise à grand
nombre de réplications vit dans ``validation/spec32_montecarlo.py``
(Étape 4). Ici : contrat d'API, récupération du seuil, comparaison au
comportement NARDL, et le verrou explicite de la spec §5.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.threshold import threshold_ardl


def _threshold_dgp(  # type: ignore[no-untyped-def]
    n: int, seed: int, gamma: float = 0.0, beta1: float = 2.0, beta2: float = -1.0
):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n)
    const = np.ones(n)
    beta = np.where(q < gamma, beta1, beta2)
    y = beta + rng.standard_normal(n) * 0.3
    return y, const.reshape(-1, 1), q


def _linear_dgp(n: int, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n)
    const = np.ones(n)
    y = 1.5 + rng.standard_normal(n) * 0.5
    return y, const.reshape(-1, 1), q


class TestAPIAndValidation:
    """Spec 32 §3 — contrat d'API et cas limites."""

    def test_invalid_trim_raises(self) -> None:
        y, x, q = _linear_dgp(200, 0)
        with pytest.raises(ValueError, match="trim"):
            threshold_ardl(y, x, q, trim=0.6)

    def test_invalid_delay_raises(self) -> None:
        y, x, q = _linear_dgp(200, 0)
        with pytest.raises(ValueError, match="delay"):
            threshold_ardl(y, x, q, delay=-1)

    def test_mismatched_transition_length_raises(self) -> None:
        y, x, q = _linear_dgp(200, 0)
        with pytest.raises(ValueError, match="transition"):
            threshold_ardl(y, x, q[:-5])

    def test_needs_at_least_one_regressor(self) -> None:
        y, _, q = _linear_dgp(200, 0)
        with pytest.raises(ValueError):
            threshold_ardl(y, x=None, transition=q)  # type: ignore[arg-type]

    def test_constant_regressor_allowed(self) -> None:
        y, x, q = _linear_dgp(200, 0)
        res = threshold_ardl(y, x, q, n_boot=19, seed=0)
        assert res.regime_params[0].index[0] == "x0"

    def test_reproducible_with_seed(self) -> None:
        y, x, q = _threshold_dgp(200, 1)
        r1 = threshold_ardl(y, x, q, n_boot=29, seed=42)
        r2 = threshold_ardl(y, x, q, n_boot=29, seed=42)
        assert r1.gamma_hat == r2.gamma_hat
        assert r1.linearity_pvalue == r2.linearity_pvalue

    def test_summary_contains_key_fields(self) -> None:
        y, x, q = _threshold_dgp(200, 0)
        res = threshold_ardl(y, x, q, n_boot=19, seed=0)
        text = res.summary()
        assert "Hansen threshold" in text
        assert "gamma_hat" in text

    def test_grid_has_margin_visible(self) -> None:
        y, x, q = _threshold_dgp(200, 0)
        res = threshold_ardl(y, x, q, n_boot=5, seed=0)
        assert len(res.grid) > 1
        assert {"gamma", "ssr"} <= set(res.grid.columns)


class TestThresholdRecovery:
    """Spec 32 §5.1 — le seuil et l'effet de régime sont retrouvés."""

    def test_gamma_recovered_large_sample(self) -> None:
        y, x, q = _threshold_dgp(2000, 3, gamma=0.0)
        res = threshold_ardl(y, x, q, delay=0, n_boot=5, seed=3)
        assert abs(res.gamma_hat) < 0.05

    def test_linearity_rejected_with_high_power(self) -> None:
        y, x, q = _threshold_dgp(400, 4, gamma=0.0, beta1=3.0, beta2=-2.0)
        res = threshold_ardl(y, x, q, delay=0, n_boot=99, seed=4)
        assert res.decision(0.05) == "threshold"

    def test_linear_dgp_does_not_reject(self) -> None:
        y, x, q = _linear_dgp(300, 5)
        res = threshold_ardl(y, x, q, delay=0, n_boot=199, seed=5)
        assert res.decision(0.05) == "linear"

    def test_regime_params_match_true_dgp(self) -> None:
        y, x, q = _threshold_dgp(2000, 6, gamma=0.0, beta1=2.0, beta2=-1.0)
        res = threshold_ardl(y, x, q, delay=0, n_boot=5, seed=6)
        assert res.regime_params[0].iloc[0] == pytest.approx(2.0, abs=0.1)
        assert res.regime_params[1].iloc[0] == pytest.approx(-1.0, abs=0.1)


class TestDelay:
    """Spec 32 §2.1 — la variable de transition est décalée de `delay`."""

    def test_self_exciting_tar_with_delay(self) -> None:
        rng = np.random.default_rng(7)
        n = 500
        x = rng.standard_normal(n)
        const = np.ones(n)
        q = np.roll(x, 1)
        q[0] = 0.0
        beta = np.where(q < 0.0, 2.0, -1.0)
        y = beta + rng.standard_normal(n) * 0.3
        design = np.column_stack([const, x])
        res = threshold_ardl(y, design, x, delay=1, n_boot=19, seed=7)
        assert abs(res.gamma_hat) < 0.2
        assert res.decision(0.05) == "threshold"
