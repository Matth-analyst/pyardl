"""Spec 04 §3 — plan de tests DHSY.

Le verrou de ce module : le design ECM construit pour l'imposition doit
reproduire EXACTEMENT la régression ARDL (résidus identiques à 1e-10).
Sans lui, le test F comparerait deux modèles qui ne portent pas sur le
même échantillon, et le résultat n'aurait aucun sens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.core.ardl import ARDL
from pyardl.core.restrictions import _uecm_design
from pyardl.utils import diff


def _dgp_ecm(
    n: int,
    seed: int,
    theta: float = 1.0,
    lam: float = -0.4,
    sigma: float = 1.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """DGP à coefficient de long terme CONNU.

    y suit un ECM dont l'équilibre est y = theta * x : c'est ce theta que
    les tests d'homogénéité doivent retrouver.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = (
            y[t - 1]
            + lam * (y[t - 1] - theta * x[t - 1])
            + sigma * (rng.standard_normal())
        )
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


class TestUECMDesignLock:
    """Verrou : le design ECM == la régression ARDL."""

    @pytest.mark.parametrize("order", [(1, 1), (2, 1), (3, 2), (2, 0), (1, 3)])
    @pytest.mark.parametrize("det", ["const", "trend"])
    def test_uecm_reproduces_ardl_residuals(self, order, det) -> None:  # type: ignore[no-untyped-def]
        """Résidus identiques à 1e-10, donc même échantillon et même fit."""
        y, x = _dgp_ecm(200, seed=sum(order) + len(det))
        res = ARDL(y, x, order=order, det=det)._fit()

        design, y_dep, _ = _uecm_design(res)
        beta = np.linalg.lstsq(design, y_dep, rcond=None)[0]
        resid = y_dep - design @ beta

        assert np.max(np.abs(resid - res.resid.to_numpy())) < 1e-10
        assert abs(float(resid @ resid) - res.ssr) < 1e-8

    def test_ratio_design_folds_two_columns_into_one(self) -> None:
        """Sous la restriction, les deux niveaux deviennent un seul."""
        y, x = _dgp_ecm(200, seed=3)
        res = ARDL(y, x, order=(2, 2))._fit()
        free, _, names_free = _uecm_design(res)
        tied, _, names_tied = _uecm_design(res, ratio_with="x")
        assert tied.shape[1] == free.shape[1] - 1
        assert "(y-x).L1" in names_tied
        assert "y.L1" in names_free

    def test_unknown_ratio_regressor_raises(self) -> None:
        y, x = _dgp_ecm(150, seed=4)
        res = ARDL(y, x, order=(1, 1))._fit()
        with pytest.raises(ValueError, match="not a regressor"):
            _uecm_design(res, ratio_with="z")


class TestWaldSizeAndPower:
    """§3.1 — taille sous H0 vraie, puissance sous H0 fausse."""

    @pytest.mark.fast_mc
    def test_size_under_true_homogeneity(self) -> None:
        """theta = 1 vrai -> taux de rejet proche du nominal."""
        rejections = 0
        reps = 200
        for seed in range(reps):
            y, x = _dgp_ecm(200, seed=7000 + seed, theta=1.0)
            res = ARDL(y, x, order=(1, 1))._fit()
            out = res.test_longrun_restriction([[1.0]], 1.0)
            rejections += out.decision(0.05) == "reject"
        assert 0.01 <= rejections / reps <= 0.15

    @pytest.mark.fast_mc
    def test_power_increases_with_sample_size(self) -> None:
        """theta = 1.3 : le rejet devient plus fréquent quand T croît."""
        rates = []
        for n in (100, 400):
            rejections = 0
            reps = 100
            for seed in range(reps):
                y, x = _dgp_ecm(n, seed=8000 + seed, theta=1.3)
                res = ARDL(y, x, order=(1, 1))._fit()
                rejections += (
                    res.test_longrun_restriction([[1.0]], 1.0).decision(0.05)
                    == "reject"
                )
            rates.append(rejections / reps)
        assert rates[1] > rates[0]

    @pytest.mark.slow
    def test_size_full(self) -> None:
        rejections = 0
        reps = 1000
        for seed in range(reps):
            y, x = _dgp_ecm(200, seed=9000 + seed, theta=1.0)
            res = ARDL(y, x, order=(1, 1))._fit()
            rejections += (
                res.test_longrun_restriction([[1.0]], 1.0).decision(0.05) == "reject"
            )
        assert 0.02 <= rejections / reps <= 0.12


class TestImpose:
    """§3.2 — imposition de la restriction et test F."""

    def test_restricted_ssr_is_never_smaller(self) -> None:
        """Contrainte -> SSR >= SSR libre. Identité algébrique."""
        for seed in range(8):
            y, x = _dgp_ecm(200, seed=100 + seed)
            res = ARDL(y, x, order=(2, 2))._fit()
            out = res.test_longrun_restriction([[1.0]], 1.0, impose=True)
            assert out.ssr_restricted >= out.ssr_unrestricted - 1e-9

    def test_unrestricted_ssr_matches_the_ardl(self) -> None:
        """Le SSR non contraint du test F est celui du modèle ajusté."""
        y, x = _dgp_ecm(200, seed=11)
        res = ARDL(y, x, order=(2, 1))._fit()
        out = res.test_longrun_restriction([[1.0]], 1.0, impose=True)
        assert abs(out.ssr_unrestricted - res.ssr) < 1e-8

    def test_f_and_wald_agree_qualitatively(self) -> None:
        """Wald et F testent la même restriction : mêmes conclusions.

        Ils ne coïncident pas numériquement — le Wald est asymptotique et
        non linéaire dans les paramètres, le F est exact sous normalité —
        mais ils doivent trancher dans le même sens.
        """
        agree = 0
        for seed in range(20):
            y, x = _dgp_ecm(250, seed=200 + seed, theta=1.0)
            res = ARDL(y, x, order=(1, 1))._fit()
            out = res.test_longrun_restriction([[1.0]], 1.0, impose=True)
            agree += (out.pvalue < 0.05) == (out.f_pvalue < 0.05)
        assert agree >= 18

    def test_restricted_params_exposed(self) -> None:
        y, x = _dgp_ecm(200, seed=12)
        res = ARDL(y, x, order=(1, 1))._fit()
        out = res.test_longrun_restriction([[1.0]], 1.0, impose=True)
        assert out.restricted_params is not None
        assert "(y-x).L1" in out.restricted_params.index

    def test_impose_rejects_unsupported_restrictions(self) -> None:
        """Aucune imposition silencieuse d'autre chose que ce qui est testé."""
        y = pd.Series(
            np.cumsum(np.random.default_rng(13).standard_normal(200)), name="y"
        )
        x = pd.DataFrame(
            {
                "a": np.cumsum(np.random.default_rng(14).standard_normal(200)),
                "b": np.cumsum(np.random.default_rng(15).standard_normal(200)),
            }
        )
        res = ARDL(y, x, order=(1, 1))._fit()
        with pytest.raises(ValueError, match="single restriction"):
            res.test_longrun_restriction(
                [[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0], impose=True
            )
        with pytest.raises(ValueError, match="theta_j = 1 only"):
            res.test_longrun_restriction([[1.0, 1.0]], 1.0, impose=True)
        with pytest.raises(ValueError, match="requires r = 1"):
            res.test_longrun_restriction([[1.0, 0.0]], 0.5, impose=True)


class TestWaldMechanics:
    """Comportement du test lui-même."""

    def test_recovers_known_theta(self) -> None:
        """Sur un DGP à theta = 1, la restriction n'est pas rejetée."""
        y, x = _dgp_ecm(500, seed=21, theta=1.0)
        res = ARDL(y, x, order=(1, 1))._fit()
        out = res.test_longrun_restriction([[1.0]], 1.0)
        assert out.decision(0.05) == "not_rejected"
        assert abs(float(out.theta.iloc[0]) - 1.0) < 0.15

    def test_rejects_false_restriction(self) -> None:
        y, x = _dgp_ecm(500, seed=22, theta=1.0)
        res = ARDL(y, x, order=(1, 1))._fit()
        assert res.test_longrun_restriction([[1.0]], 2.0).decision(0.05) == "reject"

    def test_discrepancy_signed(self) -> None:
        """R.theta - r est renvoyé signé : le sens de l'écart est visible."""
        y, x = _dgp_ecm(300, seed=23, theta=1.0)
        res = ARDL(y, x, order=(1, 1))._fit()
        low = res.test_longrun_restriction([[1.0]], 0.5)
        high = res.test_longrun_restriction([[1.0]], 1.5)
        assert low.discrepancy[0] > 0
        assert high.discrepancy[0] < 0

    def test_one_dimensional_r_accepted(self) -> None:
        y, x = _dgp_ecm(200, seed=24)
        res = ARDL(y, x, order=(1, 1))._fit()
        assert res.test_longrun_restriction([1.0], 1.0).df == 1

    def test_joint_restriction_has_two_df(self) -> None:
        rng = np.random.default_rng(25)
        n = 300
        x = pd.DataFrame(
            {
                "a": np.cumsum(rng.standard_normal(n)),
                "b": np.cumsum(rng.standard_normal(n)),
            }
        )
        y = pd.Series(np.zeros(n), name="y")
        for t in range(1, n):
            y.iloc[t] = (
                y.iloc[t - 1]
                - 0.4 * (y.iloc[t - 1] - x["a"].iloc[t - 1] - x["b"].iloc[t - 1])
                + rng.standard_normal()
            )
        res = ARDL(y, x, order=(1, 1))._fit()
        out = res.test_longrun_restriction(np.eye(2), [1.0, 1.0])
        assert out.df == 2
        assert out.decision(0.05) == "not_rejected"

    def test_shape_errors(self) -> None:
        y, x = _dgp_ecm(200, seed=26)
        res = ARDL(y, x, order=(1, 1))._fit()
        with pytest.raises(ValueError, match="columns but the model has"):
            res.test_longrun_restriction([[1.0, 0.0]], 1.0)
        with pytest.raises(ValueError, match="entries but R has"):
            res.test_longrun_restriction([[1.0]], [1.0, 2.0])

    def test_rank_deficient_r_rejected(self) -> None:
        rng = np.random.default_rng(27)
        n = 250
        x = pd.DataFrame(
            {
                "a": np.cumsum(rng.standard_normal(n)),
                "b": np.cumsum(rng.standard_normal(n)),
            }
        )
        y = pd.Series(np.cumsum(rng.standard_normal(n)), name="y")
        res = ARDL(y, x, order=(1, 1))._fit()
        with pytest.raises(ValueError, match="full row rank"):
            res.test_longrun_restriction([[1.0, 0.0], [2.0, 0.0]], [1.0, 2.0])

    def test_summary(self) -> None:
        y, x = _dgp_ecm(200, seed=28)
        res = ARDL(y, x, order=(1, 1))._fit()
        text = res.test_longrun_restriction([[1.0]], 1.0, impose=True).summary()
        assert "Wald chi2(1)" in text
        assert "imposed" in text
        assert "SSR unrestricted" in text

    def test_not_rejected_wording(self) -> None:
        """« not_rejected », jamais « accept »."""
        y, x = _dgp_ecm(400, seed=29, theta=1.0)
        res = ARDL(y, x, order=(1, 1))._fit()
        assert res.test_longrun_restriction([[1.0]], 1.0).decision(0.05) == (
            "not_rejected"
        )


class TestDiff:
    """§3.3 — opérateur de différence, valeurs vérifiées à la main."""

    def test_ordinary_difference(self) -> None:
        assert np.allclose(diff(np.arange(5.0), d=1), np.ones(4))

    def test_seasonal_difference_hand_checked(self) -> None:
        """(d=0, D=1, s=4) sur une série connue."""
        x = np.array([1.0, 2.0, 4.0, 8.0, 3.0, 5.0, 9.0, 15.0])
        # x_t - x_{t-4} : 3-1, 5-2, 9-4, 15-8
        assert np.allclose(diff(x, d=0, D=1, s=4), [2.0, 3.0, 5.0, 7.0])

    def test_combined_difference_hand_checked(self) -> None:
        """(d=1, D=1, s=4), calculé à la main sur la même série."""
        x = np.array([1.0, 2.0, 4.0, 8.0, 3.0, 5.0, 9.0, 15.0])
        # Delta x : 1, 2, 4, -5, 2, 4, 6
        # puis Delta_4 : 2-1, 4-2, 6-4
        assert np.allclose(diff(x, d=1, D=1, s=4), [1.0, 2.0, 2.0])

    def test_operators_commute(self) -> None:
        rng = np.random.default_rng(31)
        x = rng.standard_normal(50)
        first = diff(diff(x, d=1, D=0), d=0, D=1, s=4)
        second = diff(diff(x, d=0, D=1, s=4), d=1, D=0)
        assert np.allclose(first, second)
        assert np.allclose(diff(x, d=1, D=1, s=4), first)

    def test_length_and_index_alignment(self) -> None:
        """L'index suit : le résultat reste collé à ses dates."""
        idx = pd.period_range("1980Q1", periods=40, freq="Q")
        s = pd.Series(np.arange(40.0), index=idx, name="c")
        out = diff(s, d=1, D=1, s=4)
        assert len(out) == 40 - 1 - 4
        assert out.index[0] == idx[5]
        assert out.index[-1] == idx[-1]
        assert out.name == "c"

    def test_zero_order_is_identity(self) -> None:
        x = np.arange(6.0)
        assert np.allclose(diff(x, d=0, D=0), x)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            diff(np.arange(10.0), d=-1)
        with pytest.raises(ValueError, match="at least 1"):
            diff(np.arange(10.0), D=1, s=0)
        with pytest.raises(ValueError, match="too short"):
            diff(np.arange(4.0), d=1, D=1, s=4)
        with pytest.raises(ValueError, match="1-D"):
            diff(np.ones((4, 2)))


class TestSeasonalDummies:
    """§2.3 — déterministes saisonniers."""

    def _seasonal_dgp(self, n: int = 120, seed: int = 41):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(seed)
        x = pd.Series(np.cumsum(rng.standard_normal(n)), name="x")
        pattern = np.tile([2.0, -1.0, 0.5, -1.5], n // 4)
        y = pd.Series(np.zeros(n), name="y")
        for t in range(1, n):
            y.iloc[t] = (
                0.5 * y.iloc[t - 1]
                + 0.4 * x.iloc[t]
                + pattern[t]
                + 0.3 * rng.standard_normal()
            )
        return y, x

    def test_dummy_trap_avoided_with_intercept(self) -> None:
        """s-1 dummies avec constante, s sans : jamais de colinéarité."""
        y, x = self._seasonal_dgp()
        with_const = ARDL(y, x, order=(1, 1), det="const", seasonal=True)._fit()
        without = ARDL(y, x, order=(1, 1), det="none", seasonal=True)._fit()
        assert sum("season" in n for n in with_const.params.index) == 3
        assert sum("season" in n for n in without.params.index) == 4

    def test_seasonality_captured(self) -> None:
        """Le modèle saisonnier explique mieux une saisonnalité réelle."""
        y, x = self._seasonal_dgp()
        plain = ARDL(y, x, order=(1, 1))._fit()
        seasonal = ARDL(y, x, order=(1, 1), seasonal=True)._fit()
        assert seasonal.rsquared > plain.rsquared
        assert seasonal.ssr < plain.ssr

    def test_phase_taken_from_original_series(self) -> None:
        """hold_back ne doit pas décaler les saisons.

        Si la phase était calculée sur l'échantillon tronqué, deux
        modèles d'ordres différents attribueraient la même observation à
        deux trimestres différents.
        """
        y, x = self._seasonal_dgp()
        short = ARDL(y, x, order=(1, 1), seasonal=True, hold_back=8)
        long_hb = ARDL(y, x, order=(1, 1), seasonal=True, hold_back=12)
        d_short, _, _ = short._build_design()
        d_long, _, _ = long_hb._build_design()
        cols = slice(1, 4)  # les trois dummies
        assert np.allclose(d_short[4:, cols], d_long[:, cols])

    def test_monthly_period(self) -> None:
        y, x = self._seasonal_dgp(n=144)
        res = ARDL(y, x, order=(1, 1), seasonal=True, seasonal_periods=12)._fit()
        assert sum("season" in n for n in res.params.index) == 11

    def test_invalid_period_rejected(self) -> None:
        y, x = self._seasonal_dgp()
        with pytest.raises(ValueError, match="at least 2"):
            ARDL(y, x, order=(1, 1), seasonal=True, seasonal_periods=1)

    def test_off_by_default(self) -> None:
        y, x = self._seasonal_dgp()
        res = ARDL(y, x, order=(1, 1))._fit()
        assert not any("season" in n for n in res.params.index)


class TestCoverageGaps:
    """Branches restantes du design ECM et du test."""

    def test_uecm_carries_fixed_regressors(self) -> None:
        """Les régresseurs fixes traversent la reparamétrisation.

        Ils n'appartiennent ni aux niveaux ni aux différences : s'ils
        étaient perdus, le design contraint et le design libre ne
        porteraient plus sur le même modèle et le test F serait faux.
        """
        y, x = _dgp_ecm(200, seed=51)
        dummy = np.zeros(200)
        dummy[100:] = 1.0
        res = ARDL(
            y, x, order=(1, 1), fixed_regressors=pd.DataFrame({"break": dummy})
        )._fit()
        design, y_dep, names = _uecm_design(res)
        assert "break" in names
        beta = np.linalg.lstsq(design, y_dep, rcond=None)[0]
        resid = y_dep - design @ beta
        assert np.max(np.abs(resid - res.resid.to_numpy())) < 1e-10

    def test_scalar_value_broadcast_over_restrictions(self) -> None:
        """Un r scalaire s'applique à toutes les lignes de R."""
        rng = np.random.default_rng(52)
        n = 250
        x = pd.DataFrame(
            {
                "a": np.cumsum(rng.standard_normal(n)),
                "b": np.cumsum(rng.standard_normal(n)),
            }
        )
        y = pd.Series(np.cumsum(rng.standard_normal(n)), name="y")
        res = ARDL(y, x, order=(1, 1))._fit()
        broadcast = res.test_longrun_restriction(np.eye(2), 1.0)
        explicit = res.test_longrun_restriction(np.eye(2), [1.0, 1.0])
        assert broadcast.statistic == pytest.approx(explicit.statistic)
        assert broadcast.df == 2
