"""Spec 08 §5 — DOLS, FMOLS, CCR et la covariance de long terme.

Ce que ces estimateurs achetent n'est pas la convergence — l'OLS statique
l'a deja — mais une INFERENCE valide. Le plan de tests porte donc autant
sur les erreurs types que sur les points, et le verrou est qu'un `t` doit
pouvoir etre lu contre 1.96 sans excuse.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.cointegration import ccr, compare_longrun, dols, fmols
from pyardl.cointegration.efficient import default_dols_lags
from pyardl.utils import lead_lag_matrix, longrun_covariance_kernel


def endogenous_dgp(
    n_obs: int = 300,
    seed: int = 0,
    theta: float = 1.5,
    rho: float = 0.6,
    endog: float = 0.8,
) -> tuple[pd.Series, pd.DataFrame]:
    """Cointegration avec endogeneite ET autocorrelation.

    Le choc qui pousse x entre aussi dans l'erreur de l'equation
    (`endog`), et cette erreur est persistante (`rho`) : les deux
    ingredients qui invalident l'inference de l'OLS statique.
    """
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n_obs)
    w = rng.normal(size=n_obs)
    x = np.cumsum(w)
    u = np.zeros(n_obs)
    for t in range(1, n_obs):
        u[t] = rho * u[t - 1] + e[t] + endog * w[t]
    y = 2.0 + theta * x + u
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


class TestLongRunCovariance:
    """§5.3 — la brique, et sa convergence."""

    def test_white_noise_recovers_the_true_variance(self) -> None:
        """Sur du bruit blanc il n'y a pas de dependance a agreger :
        Omega doit tendre vers la variance contemporaine."""
        rng = np.random.default_rng(0)
        e = rng.normal(scale=2.0, size=20_000)[:, None]
        out = longrun_covariance_kernel(e, bandwidth=8)
        assert out.omega[0, 0] == pytest.approx(4.0, rel=0.08)
        assert out.sigma[0, 0] == pytest.approx(4.0, rel=0.05)

    def test_ar1_inflates_the_long_run_variance(self) -> None:
        """Pour un AR(1) de coefficient phi, Omega = sigma2/(1-phi)^2 :
        une dependance positive AUGMENTE la variance de long terme, et
        c'est tout l'interet de ne pas utiliser la variance
        contemporaine."""
        rng = np.random.default_rng(1)
        n = 40_000
        phi = 0.5
        e = rng.normal(size=n)
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = phi * u[t - 1] + e[t]
        out = longrun_covariance_kernel(u[:, None], bandwidth=25)
        assert out.omega[0, 0] == pytest.approx(1.0 / (1 - phi) ** 2, rel=0.12)
        assert out.omega[0, 0] > 3.0 * out.sigma[0, 0] * 0.5

    @pytest.mark.parametrize(
        "kernel", ["bartlett", "parzen", "quadratic-spectral", "truncated"]
    )
    def test_every_kernel_runs_and_is_symmetric(self, kernel: str) -> None:
        rng = np.random.default_rng(2)
        u = rng.normal(size=(200, 3))
        out = longrun_covariance_kernel(u, kernel=kernel, bandwidth=6)
        assert out.omega == pytest.approx(out.omega.T, abs=1e-12)

    @pytest.mark.parametrize("rule", ["andrews", "newey-west"])
    def test_automatic_bandwidths_are_positive_and_differ(self, rule: str) -> None:
        rng = np.random.default_rng(3)
        n = 300
        u = np.zeros(n)
        e = rng.normal(size=n)
        for t in range(1, n):
            u[t] = 0.7 * u[t - 1] + e[t]
        out = longrun_covariance_kernel(u[:, None], bandwidth=rule)
        assert out.bandwidth > 0

    def test_a_zero_bandwidth_is_refused(self) -> None:
        """Une bande nulle reduit la covariance de long terme a la
        contemporaine : c'est une autre quantite, pas un cas limite."""
        rng = np.random.default_rng(4)
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            longrun_covariance_kernel(rng.normal(size=50)[:, None], bandwidth=0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"kernel": "gaussian"}, "kernel must be one of"),
            ({"bandwidth": "magic"}, "bandwidth must be"),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        rng = np.random.default_rng(5)
        with pytest.raises(ValueError, match=match):
            longrun_covariance_kernel(rng.normal(size=60)[:, None], **kwargs)

    def test_a_one_dimensional_series_is_accepted(self) -> None:
        """L'usage le plus courant est une seule serie de residus ; la
        forcer en colonne a l'appel serait une friction inutile."""
        rng = np.random.default_rng(6)
        flat = rng.normal(size=200)
        assert longrun_covariance_kernel(flat, bandwidth=5).omega.shape == (1, 1)

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            (np.zeros((4, 2, 2)), "1-D or 2-D"),
            (np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]), "non-finite"),
            (np.ones((2, 2)), "at least 3 observations"),
        ],
    )
    def test_input_refusals(self, bad: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            longrun_covariance_kernel(bad, bandwidth=2)


class TestPrewhitening:
    """Le composant qui manquait, et le bug que son ajout a revele."""

    def test_it_recovers_a_persistent_long_run_variance(self) -> None:
        """Pour un AR(1) de coefficient phi, Omega = sigma2/(1-phi)^2.
        Le noyau seul, a bande courte, sous-estime lourdement ; c'est
        exactement ce qui privait les intervalles de leur couverture."""
        rng = np.random.default_rng(50)
        n, phi = 4000, 0.8
        e = rng.normal(size=n)
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = phi * u[t - 1] + e[t]
        true_omega = 1.0 / (1 - phi) ** 2
        plain = longrun_covariance_kernel(u[:, None], bandwidth=6).omega[0, 0]
        white = longrun_covariance_kernel(
            u[:, None], bandwidth=6, prewhiten=True
        ).omega[0, 0]
        assert plain < 0.6 * true_omega
        assert abs(white - true_omega) / true_omega < 0.20
        assert abs(white - true_omega) < abs(plain - true_omega)

    def test_sigma_is_not_recoloured(self) -> None:
        """Sigma est la covariance CONTEMPORAINE : l'identite
        (I-A)^-1 . (I-A')^-1 porte sur le spectre a la frequence zero,
        donc sur Omega et Delta, pas sur elle.

        La recolorer a rendu CCR pire que l'OLS nue pendant que DOLS et
        FMOLS s'amelioraient. Ce test verifie que Sigma reste celle de la
        serie d'origine, blanchiment ou non."""
        rng = np.random.default_rng(51)
        n = 800
        u = np.zeros(n)
        e = rng.normal(size=n)
        for t in range(1, n):
            u[t] = 0.7 * u[t - 1] + e[t]
        plain = longrun_covariance_kernel(u[:, None], bandwidth=6)
        white = longrun_covariance_kernel(u[:, None], bandwidth=6, prewhiten=True)
        assert white.sigma == pytest.approx(plain.sigma, rel=1e-12)
        assert white.omega[0, 0] != pytest.approx(plain.omega[0, 0], rel=1e-6)

    @pytest.mark.parametrize("estimator", [dols, fmols, ccr])
    def test_the_estimators_prewhiten_by_default(self, estimator) -> None:  # type: ignore[no-untyped-def]
        """Le defaut est le comportement mesure comme correct. S'il
        changeait sans qu'on s'en apercoive, la couverture retomberait
        sous le nominal sans qu'aucun autre test ne bronche."""
        y, x = endogenous_dgp(n_obs=300, seed=52)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = estimator(y, x, bandwidth=6)
            off = estimator(y, x, bandwidth=6, prewhiten=False)
        assert res.prewhitened
        assert not off.prewhitened
        assert res.longrun.loc["x", "se"] != pytest.approx(
            off.longrun.loc["x", "se"], rel=1e-8
        )

    def test_a_near_unit_root_does_not_explode(self) -> None:
        """La recolorisation divise par (I-A) : sans bridage des valeurs
        propres, une racine proche de l'unite rendrait une variance
        arbitrairement grande plutot qu'une erreur."""
        rng = np.random.default_rng(53)
        n = 500
        u = np.cumsum(rng.normal(size=n))
        out = longrun_covariance_kernel(u[:, None], bandwidth=6, prewhiten=True)
        assert np.isfinite(out.omega).all()
        assert out.omega[0, 0] > 0


class TestLeadLagMatrix:
    def test_columns_are_ordered_leads_then_contemporaneous_then_lags(self) -> None:
        x = np.arange(6.0)[:, None]
        block, names, start, stop = lead_lag_matrix(x, n_leads=1, n_lags=1)
        assert names == ["x0.F1", "x0.L0", "x0.L1"]
        assert block[0].tolist() == [2.0, 1.0, 0.0]
        assert (start, stop) == (1, 5)

    def test_the_window_is_the_largest_common_sample(self) -> None:
        x = np.arange(20.0)[:, None]
        block, _, start, stop = lead_lag_matrix(x, n_leads=3, n_lags=2)
        assert block.shape == (15, 6)
        assert (start, stop) == (2, 17)

    def test_zero_leads_and_lags_gives_the_series_back(self) -> None:
        x = np.arange(5.0)[:, None]
        block, names, start, stop = lead_lag_matrix(x, 0, 0)
        assert names == ["x0.L0"]
        assert block.ravel().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert (start, stop) == (0, 5)

    def test_too_many_leaves_nothing(self) -> None:
        with pytest.raises(ValueError, match="leave"):
            lead_lag_matrix(np.arange(4.0)[:, None], n_leads=3, n_lags=3)

    def test_negative_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            lead_lag_matrix(np.arange(10.0)[:, None], n_leads=-1, n_lags=0)

    def test_a_one_dimensional_series_is_accepted(self) -> None:
        block, names, _, _ = lead_lag_matrix(np.arange(6.0), n_leads=0, n_lags=1)
        assert names == ["x0.L0", "x0.L1"]
        assert block.shape == (5, 2)

    def test_three_dimensional_is_refused(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            lead_lag_matrix(np.zeros((4, 2, 2)), n_leads=0, n_lags=0)


class TestDefaultLags:
    def test_cube_root_and_the_perfect_cube_trap(self) -> None:
        assert default_dols_lags(27) == 3
        assert default_dols_lags(64) == 4
        assert default_dols_lags(1000) == 10
        # La forme naive perd un retard aux cubes parfaits.
        assert int(np.floor(1000 ** (1 / 3))) == 9

    def test_refuses_a_non_positive_sample(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            default_dols_lags(0)


class TestEstimatorsRecoverTheta:
    @pytest.mark.parametrize("estimator", [dols, fmols, ccr])
    def test_each_recovers_the_true_coefficient(self, estimator) -> None:  # type: ignore[no-untyped-def]
        y, x = endogenous_dgp(n_obs=600, seed=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = estimator(y, x, bandwidth=8)
        assert res.longrun.loc["x", "theta"] == pytest.approx(1.5, abs=0.05)

    @pytest.mark.parametrize("estimator", [dols, fmols, ccr])
    def test_standard_errors_are_finite_and_positive(self, estimator) -> None:  # type: ignore[no-untyped-def]
        y, x = endogenous_dgp(n_obs=400, seed=11)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = estimator(y, x, bandwidth=6)
        assert np.isfinite(res.longrun["se"]).all()
        assert (res.longrun["se"] > 0).all()

    def test_ccr_and_fmols_converge_together(self) -> None:
        """Ils sont asymptotiquement equivalents. CCR n'a pas de
        reference externe — cointReg ne l'implemente pas — donc c'est
        par cette equivalence et par la convergence vers le vrai theta
        qu'il est verifie."""
        y, x = endogenous_dgp(n_obs=4000, seed=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = fmols(y, x, bandwidth=12)
            b = ccr(y, x, bandwidth=12)
        assert a.longrun.loc["x", "theta"] == pytest.approx(
            b.longrun.loc["x", "theta"], abs=0.02
        )

    def test_two_regressors(self) -> None:
        rng = np.random.default_rng(13)
        n = 500
        w1, w2, e = (rng.normal(size=n) for _ in range(3))
        x1, x2 = np.cumsum(w1), np.cumsum(w2)
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.5 * u[t - 1] + e[t] + 0.6 * w1[t]
        y = 1.0 + 0.8 * x1 - 0.5 * x2 + u
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fmols(
                pd.Series(y, name="y"), pd.DataFrame({"x1": x1, "x2": x2}), bandwidth=8
            )
        assert res.longrun.loc["x1", "theta"] == pytest.approx(0.8, abs=0.08)
        assert res.longrun.loc["x2", "theta"] == pytest.approx(-0.5, abs=0.08)


class TestDolsSpecifics:
    def test_leads_are_what_removes_the_endogeneity(self) -> None:
        """Sans avances, DOLS n'est qu'une OLS statique avec des colonnes
        en plus. Ce test le montre plutot que de l'affirmer : sur un DGP
        a forte endogeneite, la version sans avances doit etre plus
        eloignee du vrai theta."""
        y, x = endogenous_dgp(n_obs=300, seed=20, endog=1.5, rho=0.7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            without = dols(y, x, n_leads=0, n_lags=4, bandwidth=6)
            with_leads = dols(y, x, n_leads=4, n_lags=4, bandwidth=6)
        err_without = abs(without.longrun.loc["x", "theta"] - 1.5)
        err_with = abs(with_leads.longrun.loc["x", "theta"] - 1.5)
        assert err_with < err_without

    def test_automatic_lag_choice(self) -> None:
        y, x = endogenous_dgp(n_obs=216, seed=21)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = dols(y, x, bandwidth=6)
        assert res.n_leads == 6
        assert res.n_lags == 6

    def test_negative_lags_refused(self) -> None:
        y, x = endogenous_dgp(n_obs=100, seed=22)
        with pytest.raises(ValueError, match="non-negative"):
            dols(y, x, n_leads=-1)


class TestResults:
    def test_summary_names_the_method_and_the_distribution(self) -> None:
        y, x = endogenous_dgp(n_obs=300, seed=30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = fmols(y, x, bandwidth=6).summary()
        assert "FMOLS" in text
        assert "asymptotically standard normal" in text

    def test_dols_summary_reports_leads_and_lags(self) -> None:
        y, x = endogenous_dgp(n_obs=300, seed=31)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = dols(y, x, n_leads=3, n_lags=2, bandwidth=6).summary()
        assert "3 lead(s) and 2 lag(s)" in text

    def test_results_are_immutable(self) -> None:
        y, x = endogenous_dgp(n_obs=200, seed=32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fmols(y, x, bandwidth=6)
        with pytest.raises(AttributeError):
            res.method = "OLS"  # type: ignore[misc]

    @pytest.mark.parametrize("det", ["none", "const", "trend"])
    def test_every_deterministic_case_builds(self, det: str) -> None:
        y, x = endogenous_dgp(n_obs=300, seed=33)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fmols(y, x, det=det, bandwidth=6)
        assert np.isfinite(res.longrun.loc["x", "theta"])

    def test_bad_det_refused(self) -> None:
        y, x = endogenous_dgp(n_obs=100, seed=34)
        with pytest.raises(ValueError, match="det must be"):
            fmols(y, x, det="quadratic")  # type: ignore[arg-type]

    def test_no_regressor_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one regressor"):
            fmols(pd.Series(np.arange(50.0), name="y"), None)  # type: ignore[arg-type]


class TestCompareLongRun:
    def test_table_has_the_three_efficient_estimators(self) -> None:
        y, x = endogenous_dgp(n_obs=300, seed=40)
        table = compare_longrun(y, x, bandwidth=6)
        assert sorted(set(table.index.get_level_values("method"))) == [
            "CCR",
            "DOLS",
            "FMOLS",
        ]
        assert list(table.columns) == ["theta", "se", "t"]

    def test_ardl_block_is_added_when_given(self) -> None:
        from pyardl.core.ardl import ARDL

        y, x = endogenous_dgp(n_obs=300, seed=41)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARDL(y, x, order=(1, 1), det="const").fit()
            table = compare_longrun(y, x, ardl_results=fit, bandwidth=6)
        assert "ARDL" in set(table.index.get_level_values("method"))
        assert table.loc[("ARDL", "x"), "theta"] == pytest.approx(
            fit.longrun.loc["x", "theta"], rel=1e-12
        )

    def test_an_object_without_longrun_is_refused(self) -> None:
        y, x = endogenous_dgp(n_obs=200, seed=42)
        with pytest.raises(ValueError, match="no `longrun` attribute"):
            compare_longrun(y, x, ardl_results=object(), bandwidth=6)

    def test_the_four_methods_agree_on_a_clean_dgp(self) -> None:
        """Quand les quatre concordent, la conclusion ne repose pas sur
        une specification. C'est l'usage du tableau."""
        from pyardl.core.ardl import ARDL

        y, x = endogenous_dgp(n_obs=800, seed=43)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARDL(y, x, order=(1, 1), det="const").fit()
            table = compare_longrun(y, x, ardl_results=fit, bandwidth=10)
        spread = table.xs("x", level="regressor")["theta"]
        assert spread.max() - spread.min() < 0.15
