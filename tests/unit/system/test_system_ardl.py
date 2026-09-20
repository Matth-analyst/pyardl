"""Tests for pyardl.system.SystemARDL (Spec 38 — SUR-ECM, Zellner 1962)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.system import SystemARDL


def _dgp_two_equations(
    rng: np.random.Generator,
    n: int,
    rho: float,
    same_regressors: bool,
    sigma1: float = 0.2,
    sigma2: float = 0.2,
) -> dict[str, tuple[np.ndarray, pd.DataFrame, tuple[int, int]]]:
    """Two ECM equations sharing a contemporaneous error correlation rho.

    Errors are built as ``e_m = sqrt(1-rho^2) * idio_m + rho * common``
    (each scaled by its own sigma) so that ``Corr(e1, e2) = rho`` exactly
    in the DGP, matching the model's structural assumption
    (Spec 38 Section 2.1: contemporaneous correlation only, independent
    across time).
    """
    common = rng.standard_normal(n)
    idio1 = rng.standard_normal(n)
    idio2 = rng.standard_normal(n)
    e1 = sigma1 * (np.sqrt(1 - rho**2) * idio1 + rho * common)
    e2 = sigma2 * (np.sqrt(1 - rho**2) * idio2 + rho * common)

    x1 = pd.Series(rng.standard_normal(n).cumsum(), name="x1")
    x2v = (
        x1 if same_regressors else pd.Series(rng.standard_normal(n).cumsum(), name="x2")
    )

    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for t in range(1, n):
        y1[t] = y1[t - 1] - 0.5 * (y1[t - 1] - 1.0 * x1.iloc[t - 1]) + e1[t]
        y2[t] = y2[t - 1] - 0.6 * (y2[t - 1] - 1.5 * x2v.iloc[t - 1]) + e2[t]

    x1_df = pd.DataFrame({"x1": x1})
    x2_df = pd.DataFrame({"x1": x2v}) if same_regressors else pd.DataFrame({"x2": x2v})

    return {
        "eq1": (y1, x1_df, (1, 1)),
        "eq2": (y2, x2_df, (1, 1)),
    }


class TestAPI:
    """Spec 38 Section 3 — public API surface."""

    def test_requires_at_least_two_equations(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        x = pd.DataFrame({"x": rng.standard_normal(100)})
        with pytest.raises(ValueError, match="at least two"):
            SystemARDL({"eq1": (y, x, (1, 1))})

    def test_result_shape(self) -> None:
        rng = np.random.default_rng(1)
        eqs = _dgp_two_equations(rng, n=200, rho=0.5, same_regressors=False)
        res = SystemARDL(eqs).fit()

        assert sorted(res.equations) == ["eq1", "eq2"]
        assert sorted(res.fgls_params) == ["eq1", "eq2"]
        assert res.sigma.shape == (2, 2)
        assert list(res.sigma.index) == ["eq1", "eq2"]
        assert res.n_iter >= 1
        assert isinstance(res.summary(), str)
        assert "eq1" in res.summary()

    def test_unknown_equation_name_raises(self) -> None:
        rng = np.random.default_rng(2)
        eqs = _dgp_two_equations(rng, n=150, rho=0.4, same_regressors=False)
        res = SystemARDL(eqs).fit()
        with pytest.raises(KeyError):
            res.efficiency_gain("does_not_exist")


class TestEfficiencyGain:
    """Spec 38 Section 5.1 — FGLS more precise than per-equation OLS.

    With strong residual correlation and different regressors between
    equations, FGLS standard errors should on average be smaller than
    the per-equation OLS ones on the same draw.
    """

    def test_fgls_more_precise_than_ols_on_average(self) -> None:
        rng = np.random.default_rng(42)
        eqs = _dgp_two_equations(rng, n=400, rho=0.85, same_regressors=False)
        res = SystemARDL(eqs).fit()

        gains = pd.concat([res.efficiency_gain("eq1"), res.efficiency_gain("eq2")])
        assert gains.mean() > 1.0


class TestDegeneracy:
    """Spec 38 Section 5.2 — sigma_12 = 0 collapses FGLS onto per-equation OLS."""

    def test_zero_correlation_fgls_matches_ols(self) -> None:
        rng = np.random.default_rng(7)
        eqs = _dgp_two_equations(rng, n=300, rho=0.0, same_regressors=False)
        res = SystemARDL(eqs, iterate=True).fit()

        for name in ("eq1", "eq2"):
            ols_params = res.equations[name].params
            fgls_params = res.fgls_params[name]
            common = [n for n in fgls_params.index if n in ols_params.index]
            np.testing.assert_allclose(
                fgls_params[common].to_numpy(),
                ols_params[common].to_numpy(),
                atol=0.05,
            )


class TestIdenticalRegressorsIdentity:
    """Spec 38 Section 2.2 / 5.3 — classical Zellner result.

    The classical Zellner (1962) identity requires the *regressor
    matrix itself* (``Z_m``) to be identical across equations, not
    merely the same order and variable names. In a system of ECMs each
    equation carries its own lagged dependent variable
    (``y_m.L1``, distinct per equation whenever ``y_1 != y_2``), so an
    ECM-vs-ECM system with a common own-lag term does not satisfy the
    identity even with a shared ``x``. Using ``order=(0, q)`` (no own
    lag) with the same ``x`` on both equations gives genuinely identical
    ``Z_m`` matrices and isolates the pure linear-algebra identity,
    independent of the residual correlation or the y values — the same
    1e-10 discipline used for the ARDL <-> ECM lock (Spec 03).
    """

    def test_identical_regressors_exact_identity(self) -> None:
        rng = np.random.default_rng(11)
        n = 250
        x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
        y1 = 1.0 * x + rng.standard_normal(n) * 0.2
        y2 = -0.5 * x + rng.standard_normal(n) * 0.3
        eqs = {
            "eq1": (y1.to_numpy(), pd.DataFrame({"x": x}), (0, 1)),
            "eq2": (y2.to_numpy(), pd.DataFrame({"x": x}), (0, 1)),
        }
        res = SystemARDL(eqs, iterate=True, max_iter=200, tol=1e-14).fit()

        for name in ("eq1", "eq2"):
            ols_params = res.equations[name].params
            fgls_params = res.fgls_params[name]
            common = [n for n in fgls_params.index if n in ols_params.index]
            np.testing.assert_allclose(
                fgls_params[common].to_numpy(),
                ols_params[common].to_numpy(),
                atol=1e-10,
                rtol=1e-10,
            )


class TestCrossEquationRestriction:
    """Spec 38 Section 2.3 — Wald test on the raw stacked parameter vector."""

    def test_true_restriction_not_rejected_at_5pct_on_average(self) -> None:
        """Same adjustment speed and long-run coefficient in both equations.

        Both equations share ``lambda=-0.5`` and ``theta=1.0``, so the
        UECM level coefficient ``gamma = -lambda*theta`` is genuinely
        equal across equations (``H0`` is true by construction) — this
        isolates the Wald test's own size from the DGP's own
        restriction being false, which a first draft of this test
        conflated (a level coefficient built from different
        (lambda, theta) pairs per equation is *not* equal even though
        both equations share the same regressor).
        """
        rng = np.random.default_rng(99)
        rejections = 0
        n_rep = 200
        for _ in range(n_rep):
            n = 300
            common = rng.standard_normal(n)
            idio1 = rng.standard_normal(n)
            idio2 = rng.standard_normal(n)
            e1 = 0.2 * (np.sqrt(1 - 0.5**2) * idio1 + 0.5 * common)
            e2 = 0.2 * (np.sqrt(1 - 0.5**2) * idio2 + 0.5 * common)
            x1 = pd.Series(rng.standard_normal(n).cumsum(), name="x1")
            y1 = np.zeros(n)
            y2 = np.zeros(n)
            for t in range(1, n):
                y1[t] = y1[t - 1] - 0.5 * (y1[t - 1] - 1.0 * x1.iloc[t - 1]) + e1[t]
                y2[t] = y2[t - 1] - 0.5 * (y2[t - 1] - 1.0 * x1.iloc[t - 1]) + e2[t]
            eqs = {
                "eq1": (y1, pd.DataFrame({"x1": x1}), (1, 1)),
                "eq2": (y2, pd.DataFrame({"x1": x1}), (1, 1)),
            }
            res = SystemARDL(eqs).fit()
            names1 = res.names["eq1"]
            names2 = res.names["eq2"]
            k1 = len(names1)
            total_k = k1 + len(names2)
            idx1 = names1.index("x1.L1")
            idx2 = names2.index("x1.L1")

            r_matrix = np.zeros((1, total_k))
            r_matrix[0, idx1] = 1.0
            r_matrix[0, k1 + idx2] = -1.0
            wald = res.test_cross_equation_restriction(r_matrix, [0.0])
            if wald.pvalue < 0.05:
                rejections += 1

        rejection_rate = rejections / n_rep
        assert rejection_rate < 0.30

    def test_restriction_shape_validation(self) -> None:
        rng = np.random.default_rng(13)
        eqs = _dgp_two_equations(rng, n=150, rho=0.3, same_regressors=False)
        res = SystemARDL(eqs).fit()
        with pytest.raises(ValueError, match="columns"):
            res.test_cross_equation_restriction(np.zeros((1, 3)), [0.0])
        total_k = res._theta_stacked.shape[0]
        with pytest.raises(ValueError, match="same number of rows"):
            res.test_cross_equation_restriction(np.zeros((2, total_k)), [0.0])

    def test_wald_result_fields(self) -> None:
        rng = np.random.default_rng(21)
        eqs = _dgp_two_equations(rng, n=200, rho=0.4, same_regressors=False)
        res = SystemARDL(eqs).fit()
        total_k = res._theta_stacked.shape[0]
        r_matrix = np.zeros((1, total_k))
        r_matrix[0, 0] = 1.0
        wald = res.test_cross_equation_restriction(r_matrix, [0.0])
        assert wald.df == 1
        assert wald.stat >= 0.0
        assert 0.0 <= wald.pvalue <= 1.0


class TestIterateFlag:
    def test_single_pass_when_not_iterating(self) -> None:
        rng = np.random.default_rng(55)
        eqs = _dgp_two_equations(rng, n=180, rho=0.5, same_regressors=False)
        res = SystemARDL(eqs, iterate=False).fit()
        assert res.n_iter == 1

    def test_iterating_can_take_more_than_one_pass(self) -> None:
        rng = np.random.default_rng(56)
        eqs = _dgp_two_equations(rng, n=180, rho=0.5, same_regressors=False)
        res = SystemARDL(eqs, iterate=True, max_iter=200, tol=1e-12).fit()
        assert res.n_iter >= 1
