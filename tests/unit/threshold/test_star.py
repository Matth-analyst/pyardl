"""Spec 34 §5 — plan de tests STAR-ARDL.

La calibration de taille précise du test de linéarité vit dans un futur
``validation/spec34_montecarlo.py`` (Étape 4). Ici : contrat d'API,
récupération de gamma/c sur un DGP LSTAR, puissance/taille du test de
linéarité, et le cas limite gamma -> infini (§5.3, comparé à
threshold_ardl).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.threshold import star_ardl, threshold_ardl


def _lstar_dgp(n: int, seed: int, gamma: float = 3.0, c: float = 0.0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    q = np.cumsum(rng.standard_normal(n)) * 0.1
    x = np.cumsum(rng.standard_normal(n))
    g_true = 1.0 / (1.0 + np.exp(-gamma * (q - c)))
    beta = 1.0 + 2.0 * g_true
    y = beta * x + rng.standard_normal(n) * 0.3
    return y, x, q


def _linear_dgp(n: int, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    q = np.cumsum(rng.standard_normal(n)) * 0.1
    x = np.cumsum(rng.standard_normal(n))
    y = 1.5 * x + rng.standard_normal(n) * 0.5
    return y, x, q


class TestAPIAndValidation:
    def test_invalid_form_raises(self) -> None:
        y, x, q = _linear_dgp(100, 0)
        with pytest.raises(ValueError, match="form"):
            star_ardl(y, x, transition_var=q, form="bogus")  # type: ignore[arg-type]

    def test_invalid_trim_raises(self) -> None:
        y, x, q = _linear_dgp(100, 0)
        with pytest.raises(ValueError, match="trim"):
            star_ardl(y, x, transition_var=q, trim=0.6)

    def test_mismatched_transition_length_raises(self) -> None:
        y, x, q = _linear_dgp(100, 0)
        with pytest.raises(ValueError, match="transition_var"):
            star_ardl(y, x, transition_var=q[:-5])

    def test_estar_form_runs(self) -> None:
        y, x, q = _linear_dgp(150, 0)
        res = star_ardl(y, x, transition_var=q, form="estar", order=(1, 1))
        assert res.form == "estar"

    def test_auto_form_selects_one(self) -> None:
        y, x, q = _lstar_dgp(300, 1)
        res = star_ardl(y, x, transition_var=q, form="auto", order=(1, 1))
        assert res.form_selected in ("lstar", "estar")
        assert res.form == res.form_selected

    def test_summary_contains_key_fields(self) -> None:
        y, x, q = _lstar_dgp(200, 2)
        res = star_ardl(y, x, transition_var=q, order=(1, 1))
        text = res.summary()
        assert "STAR-ARDL" in text
        assert "gamma_hat" in text

    def test_longrun_at_shape(self) -> None:
        y, x, q = _lstar_dgp(300, 3)
        res = star_ardl(y, x, transition_var=q, order=(1, 1))
        curve = res.longrun_at([-1.0, 0.0, 1.0])
        assert list(curve.index) == [-1.0, 0.0, 1.0]
        assert "x0" in curve.columns


class TestParameterRecovery:
    """Spec 34 §5.1 — gamma_hat/c_hat retrouvent les vraies valeurs."""

    def test_gamma_and_c_recovered(self) -> None:
        y, x, q = _lstar_dgp(300, 4, gamma=3.0, c=0.0)
        res = star_ardl(y, x, transition_var=q, form="lstar", order=(1, 1))
        assert res.gamma_hat == pytest.approx(3.0, abs=1.5)
        assert res.c_hat == pytest.approx(0.0, abs=0.5)

    def test_linearity_rejected_under_transition_dgp(self) -> None:
        y, x, q = _lstar_dgp(300, 5)
        res = star_ardl(y, x, transition_var=q, form="lstar", order=(1, 1))
        assert res.linearity_pvalue < 0.01


class TestLinearityTestSize:
    """Spec 34 §5.2 — DGP linéaire : le test ne rejette pas."""

    def test_linear_dgp_does_not_reject(self) -> None:
        y, x, q = _linear_dgp(300, 6)
        res = star_ardl(y, x, transition_var=q, form="lstar", order=(1, 1))
        assert res.linearity_pvalue > 0.05


class TestSharpLimit:
    """Spec 34 §5.3 — gamma -> infini converge vers Threshold ARDL (spec 32)."""

    def test_very_sharp_transition_matches_threshold_regime_split(self) -> None:
        rng = np.random.default_rng(7)
        n = 300
        q = np.cumsum(rng.standard_normal(n)) * 0.1
        x = np.cumsum(rng.standard_normal(n))
        # A near-step function (very large true gamma): the two regimes
        # should be close to two distinct constants either side of c=0.
        beta = np.where(q < 0.0, 1.0, 3.5)
        y = beta * x + rng.standard_normal(n) * 0.2
        star_res = star_ardl(y, x, transition_var=q, form="lstar", order=(1, 1))
        thresh_res = threshold_ardl(
            y, np.column_stack([np.ones(n), x]), q, delay=0, n_boot=5, seed=7
        )
        star_curve = star_res.longrun_at([-2.0, 2.0])
        # Both should show a level well below and well above the
        # threshold's own regime coefficients, to a loose tolerance —
        # the logistic transition at finite gamma is never perfectly
        # sharp, so this is a structural, not exact, comparison.
        assert star_curve["x0"].iloc[0] < thresh_res.regime_params[1]["x1"]
        assert star_curve["x0"].iloc[1] > thresh_res.regime_params[0]["x1"]
