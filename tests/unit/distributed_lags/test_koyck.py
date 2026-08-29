"""Spec 01 §8.1 — Koyck (1954), et le biais qu'il faut voir plutot que cacher.

Le point de cette spec n'est pas qu'un estimateur marche : c'est que le
plus evident ne marche PAS. La transformation de Koyck cree
mecaniquement une correlation entre y_(t-1) et l'erreur, donc l'OLS y
est inconvergente. Les tests le verifient dans les deux sens : les
estimateurs corrects retrouvent la verite, et l'OLS ne la retrouve pas
— parce qu'un test qui n'exigerait que le premier passerait aussi avec
une implementation qui ignore silencieusement le probleme.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.distributed_lags import KoyckModel
from pyardl.exceptions import DegenerateCaseWarning, PyardlMethodologyWarning
from pyardl.utils import _delta_method

TRUE = {"alpha": 2.0, "beta0": 0.8, "lam": 0.6}


def _koyck_dgp(n: int, seed: int, lam: float = 0.6, sigma: float = 0.5):
    """Le vrai modele geometrique, ecrit sous sa forme transformee.

    L'erreur est MA(1) de coefficient -lambda : c'est ce que produit la
    transformation, et c'est la source de l'inconvergence de l'OLS. La
    simuler autrement reviendrait a tester un modele que personne
    n'estime.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e = rng.normal(scale=sigma, size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = (
            TRUE["alpha"] * (1 - lam)
            + TRUE["beta0"] * x[t]
            + lam * y[t - 1]
            + e[t]
            - lam * e[t - 1]
        )
    return pd.Series(y, name="y"), pd.Series(x, name="x")


def _fit(y, x, method: str = "iv"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KoyckModel(y, x, method=method).fit()


def _lam(y, x, method: str = "iv") -> float:
    return float(_fit(y, x, method).params["lam"])


class TestConsistency:
    """Spec 01 §8.1.1 — qui retrouve la verite, et qui ne la retrouve pas."""

    @staticmethod
    def _replicate(method: str, n: int, n_rep: int, seed0: int) -> tuple[float, float]:
        """Moyenne de lambda_hat sur `n_rep` echantillons, et son erreur MC.

        Un seul tirage ne mesure pas la convergence : a n = 5000 l'ecart
        type de lambda_hat vaut environ 0.018 sous IV, donc un tirage
        isole peut manquer la verite de 0.02 sans que rien ne soit
        casse. C'est arrive en ecrivant ce test, et la reaction correcte
        n'etait pas de relacher la tolerance mais de mesurer une
        moyenne.
        """
        values = np.empty(n_rep)
        for i in range(n_rep):
            values[i] = _lam(*_koyck_dgp(n=n, seed=seed0 + i), method)
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(n_rep))

    @pytest.mark.parametrize("method", ["iv", "ml"])
    def test_iv_and_ml_recover_the_truth(self, method: str) -> None:
        """Spec 01 8.1.1, mesure sur 30 replications.

        Dimensionnement (regle 10) : l'ecart a detecter est un biais de
        0.05 sur lambda, soit la moitie de celui de l'OLS. A 30
        replications l'erreur type de la moyenne vaut environ 0.003,
        donc 0.05 en represente une quinzaine : la question est tranchee
        sans ambiguite. La borne du test est posee a 0.015, soit
        environ cinq erreurs types.
        """
        mean, mc_se = self._replicate(method, n=2000, n_rep=30, seed0=1000)
        assert mc_se < 0.006
        assert abs(mean - TRUE["lam"]) < 0.015

    def test_ols_does_not(self) -> None:
        """Le biais DOIT etre la, et il doit etre du bon signe.

        Cov(y_(t-1), u_t) = -lambda*sigma^2 < 0, donc l'OLS sous-estime
        lambda. Un test qui se contenterait de « l'OLS est different »
        passerait aussi si l'erreur allait dans l'autre sens ; celui-ci
        verifie la direction que la theorie predit.
        """
        mean, mc_se = self._replicate("ols", n=2000, n_rep=30, seed0=1000)
        assert mean < TRUE["lam"] - 0.05
        assert (TRUE["lam"] - mean) / mc_se > 10.0

    def test_the_ols_bias_does_not_shrink_with_the_sample(self) -> None:
        """Inconvergence, pas imprecision : c'est la difference qui compte.

        Un estimateur seulement imprecis se rapproche de la verite quand
        T grandit. Celui-ci reste ou il est : biais mesure de 0.118 sur
        lambda a n = 2000 et 0.121 a n = 8000, pour une erreur type de
        moyenne de 0.002. Quadrupler l'echantillon ne rachete rien.
        """
        small, se_small = self._replicate("ols", n=2000, n_rep=20, seed0=1000)
        large, se_large = self._replicate("ols", n=8000, n_rep=20, seed0=1000)
        bias_small = TRUE["lam"] - small
        bias_large = TRUE["lam"] - large
        assert bias_small > 0.08
        assert bias_large > 0.08
        # Les deux biais sont indiscernables l'un de l'autre.
        assert abs(bias_large - bias_small) < 4.0 * np.hypot(se_small, se_large)

    def test_ols_warns_every_time(self) -> None:
        y, x = _koyck_dgp(n=300, seed=13)
        with pytest.warns(PyardlMethodologyWarning, match="INCONSISTENT"):
            KoyckModel(y, x, method="ols").fit()


class TestDeltaMethod:
    """Spec 01 §3 — les erreurs types des quantites derivees."""

    def test_analytical_gradients_match_the_numerical_helper(self) -> None:
        """Les gradients codes a la main contre le helper generique.

        `_delta_method` differencie numeriquement ; les formules du
        module sont analytiques. Les deux doivent coincider, et c'est ce
        qui protege les secondes d'une faute de derivation.
        """
        res = _fit(*_koyck_dgp(n=1000, seed=14))
        theta, v_hat = res._params, res._cov

        def longrun(t: np.ndarray) -> np.ndarray:
            return np.array([t[1] / (1.0 - t[2])])

        def mean_lag(t: np.ndarray) -> np.ndarray:
            return np.array([t[2] / (1.0 - t[2])])

        def median_lag(t: np.ndarray) -> np.ndarray:
            return np.array([np.log(0.5) / np.log(t[2])])

        table = res.multipliers()
        for name, fn in (
            ("longrun", longrun),
            ("mean_lag", mean_lag),
            ("median_lag", median_lag),
        ):
            _, cov_g = _delta_method(fn, theta, v_hat)
            assert float(table.loc[name, "se"]) == pytest.approx(
                float(np.sqrt(cov_g[0, 0])), rel=1e-4
            )

    @pytest.mark.fast_mc
    def test_delta_se_agrees_with_a_bootstrap(self) -> None:
        """Spec 01 §8.1.2 — la delta-methode contre un bootstrap.

        Le multiplicateur de long terme est un RATIO, beta0/(1-lambda),
        et la delta-methode le linearise. Elle est donc valide
        asymptotiquement, pas necessairement a n modeste. A n = 1000
        l'accord est a 1.4 % (mesure), bien dans les 10 % de la spec.
        """
        assert _bootstrap_gap(n_obs=1000, n_boot=400, seed=15) < 0.10

    def test_the_delta_method_understates_the_error_in_small_samples(self) -> None:
        """Et voici ou elle cesse d'etre valide — mesure, pas suppose.

        A n = 400 l'erreur type delta vaut 0.241 quand le bootstrap
        donne 0.309 : elle est **22 % trop petite**, donc un intervalle
        construit dessus sous-couvre. A n = 1000 et n = 3000 l'ecart
        tombe a 1.4 % et 1.6 %.

        Ce test verrouille le constat au lieu de le cacher : c'est une
        propriete de la linearisation d'un ratio, pas un defaut
        d'implementation, et l'utilisateur a besoin de savoir a partir
        de quelle taille l'intervalle se lit.
        """
        assert _bootstrap_gap(n_obs=400, n_boot=400, seed=15) > 0.15

    @pytest.mark.slow
    def test_delta_se_agrees_with_a_bootstrap_full(self) -> None:
        assert _bootstrap_gap(n_obs=1000, n_boot=1000, seed=15) < 0.10


def _bootstrap_gap(n_obs: int, n_boot: int, seed: int) -> float:
    """Ecart relatif entre l'erreur type delta et celle d'un bootstrap.

    Le reechantillonnage est PARAMETRIQUE : on retire des innovations
    sous le modele estime et on regenere y par la recursion. Un
    bootstrap par paires casserait la dependance temporelle, donc
    mesurerait la variabilite d'un autre estimateur que celui teste.
    """
    y, x = _koyck_dgp(n=n_obs, seed=seed)
    res = _fit(y, x)
    alpha, beta0, lam = (float(v) for v in res._params)
    sigma = float(np.std(res._resid, ddof=3))
    rng = np.random.default_rng(seed)
    x_arr = x.to_numpy()
    n = x_arr.size
    draws = np.empty(n_boot)
    for b in range(n_boot):
        e = rng.normal(scale=sigma, size=n)
        y_b = np.zeros(n)
        for t in range(1, n):
            y_b[t] = (
                alpha * (1 - lam)
                + beta0 * x_arr[t]
                + lam * y_b[t - 1]
                + e[t]
                - lam * e[t - 1]
            )
        draws[b] = _fit(pd.Series(y_b, name="y"), x).longrun_multiplier
    boot_se = float(np.std(draws[np.isfinite(draws)], ddof=1))
    delta_se = float(res.multipliers().loc["longrun", "se"])
    return abs(delta_se - boot_se) / boot_se


class TestMLRecursion:
    """Spec 01 §8.1.4 — les innovations recursives sont-elles blanches ?"""

    def test_ml_innovations_pass_ljung_box_and_iv_residuals_do_not(self) -> None:
        """Le contraste est le test, pas chacune des deux moities.

        Sous IV, les residus SONT le u_t = e_t - lambda e_(t-1) de la
        transformation : ils doivent etre autocorreles, et rejeter le
        bruit blanc confirme le modele au lieu de le contredire. Sous
        ML, la recursion a retire cette MA(1) et ce qui reste doit etre
        blanc. Verifier seulement le second laisserait passer une
        implementation ou la recursion ne fait rien.
        """
        y, x = _koyck_dgp(n=1000, seed=16)
        ml_p = float(_fit(y, x, "ml").diagnostics().loc["ljung_box", "pvalue"])
        iv_p = float(_fit(y, x, "iv").diagnostics().loc["ljung_box", "pvalue"])
        assert ml_p > 0.05
        assert iv_p < 0.01

    def test_ml_reports_convergence(self) -> None:
        res = _fit(*_koyck_dgp(n=600, seed=17), "ml")
        assert res.extra["converged"] is True
        assert np.isfinite(res.extra["llf"])


class TestDiagnostics:
    """Le h de Durbin, et son repli quand il n'est pas calculable."""

    def test_durbin_h_is_reported_when_defined(self) -> None:
        res = _fit(*_koyck_dgp(n=1000, seed=18))
        diag = res.diagnostics()
        assert "durbin_h" in diag.index
        assert np.isfinite(diag.loc["durbin_h", "statistic"])

    def test_alternative_test_when_h_is_undefined(self) -> None:
        """n * var(lambda) >= 1 : la racine devient negative.

        Ce n'est pas une pathologie rare — cela arrive des que lambda est
        mal estime. Le repli doit donc etre teste, pas seulement ecrit.
        """
        import dataclasses

        res = _fit(*_koyck_dgp(n=300, seed=19))
        broken_cov = res._cov.copy()
        broken_cov[2, 2] = 1.0  # n * var >> 1
        broken = dataclasses.replace(res, _cov=broken_cov)
        diag = broken.diagnostics()
        assert "durbin_alternative" in diag.index
        assert np.isfinite(diag.loc["durbin_alternative", "pvalue"])


class TestEdgeCases:
    """Spec 01 §7 — ce qui doit avertir, et ce qui doit refuser."""

    def test_lambda_near_one_warns_about_the_unit_root(self) -> None:
        """lambda_hat dans (0.98, 1) : le long terme explose sans que rien
        ne soit formellement degenere.

        La graine est choisie pour que l'estimation tombe dans cette
        bande etroite ; avec lambda vrai = 0.99 elle sort souvent
        au-dessus de 1, ce qui declenche l'autre avertissement (celui de
        la degenerescence) et ne testerait donc pas celui-ci.
        """
        y, x = _koyck_dgp(n=2000, seed=40, lam=0.985, sigma=0.2)
        with pytest.warns(PyardlMethodologyWarning, match="nearly infinite"):
            res = KoyckModel(y, x).fit()
        assert 0.98 < res.lam < 1.0

    def test_lambda_outside_the_unit_interval_gives_nan_multipliers(self) -> None:
        """Ne PAS tronquer : le modele geometrique est rejete, on le dit.

        Renvoyer lambda tronque a 0.999 produirait un multiplicateur de
        long terme enorme et parfaitement faux, avec l'apparence d'un
        resultat.
        """
        rng = np.random.default_rng(21)
        n = 300
        x = pd.Series(rng.normal(size=n), name="x")
        y = pd.Series(rng.normal(size=n), name="y")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = KoyckModel(y, x).fit()
        if not 0.0 < res.lam < 1.0:
            assert np.isnan(res.longrun_multiplier)
            assert np.isnan(res.mean_lag)
            assert np.isnan(res.median_lag)
            assert np.isnan(res.interim_multiplier(5))
        else:  # pragma: no cover - depends on the draw
            pytest.skip("this draw happened to land inside (0, 1)")

    def test_degenerate_lambda_warns(self) -> None:
        rng = np.random.default_rng(22)
        n = 200
        x = pd.Series(rng.normal(size=n), name="x")
        y = pd.Series(rng.normal(size=n), name="y")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = KoyckModel(y, x).fit()
        if not 0.0 < res.lam < 1.0:
            assert any(issubclass(w.category, DegenerateCaseWarning) for w in caught)

    def test_weak_instrument_warns(self) -> None:
        """x sans persistance : x_(t-1) instrumente mal y_(t-1)."""
        rng = np.random.default_rng(23)
        n = 60
        x = pd.Series(rng.normal(scale=0.01, size=n), name="x")
        y = pd.Series(rng.normal(size=n).cumsum(), name="y")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            KoyckModel(y, x).fit()
        assert any("Weak instrument" in str(w.message) for w in caught)

    def test_several_regressors_rejected(self) -> None:
        rng = np.random.default_rng(24)
        y = pd.Series(rng.normal(size=100), name="y")
        x = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
        with pytest.raises(ValueError, match="exactly one"):
            KoyckModel(y, x)

    def test_unknown_method(self) -> None:
        rng = np.random.default_rng(25)
        y = pd.Series(rng.normal(size=100), name="y")
        x = pd.Series(rng.normal(size=100), name="x")
        with pytest.raises(ValueError, match="method must be"):
            KoyckModel(y, x, method="gmm")  # type: ignore[arg-type]

    def test_small_sample_warns(self) -> None:
        rng = np.random.default_rng(26)
        y = pd.Series(rng.normal(size=10), name="y")
        x = pd.Series(rng.normal(size=10), name="x")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            KoyckModel(y, x)
        assert caught


class TestBridge:
    """Spec 01 §6 — le pont vers le coeur ARDL."""

    def test_ols_koyck_is_exactly_an_ardl_1_0(self) -> None:
        """Meme regression, donc memes coefficients a 1e-12.

        C'est le test qui garantit que la spec 01 n'est pas une
        reimplementation parallele du coeur : la transformation de Koyck
        EST un ARDL(1, 0), et la seule chose que l'objet ARDL ne sait
        pas est que l'erreur y est MA(1) par construction.
        """
        y, x = _koyck_dgp(n=400, seed=27)
        res = _fit(y, x, "ols")
        ardl = res.to_ardl()
        assert float(ardl.params["const"]) == pytest.approx(
            float(res._reg_params[0]), abs=1e-12
        )
        assert float(ardl.params["x.L0"]) == pytest.approx(
            float(res._reg_params[1]), abs=1e-12
        )
        assert float(ardl.params["y.L1"]) == pytest.approx(
            float(res._reg_params[2]), abs=1e-12
        )

    def test_longrun_multiplier_matches_the_ardl_long_run(self) -> None:
        """beta0/(1-lambda) est le prototype de theta = sum(beta)/(1-sum(phi))."""
        y, x = _koyck_dgp(n=400, seed=28)
        res = _fit(y, x, "ols")
        ardl = res.to_ardl()
        assert res.longrun_multiplier == pytest.approx(
            float(ardl.longrun.loc["x", "theta"]), rel=1e-10
        )


class TestPresentation:
    """Multiplicateurs, poids, prevision, resume."""

    def test_lag_weights_are_geometric_and_sum_to_the_long_run(self) -> None:
        res = _fit(*_koyck_dgp(n=800, seed=29))
        weights = res.lag_weights(200)
        assert float(weights.iloc[0]) == pytest.approx(res.impact_multiplier, rel=1e-12)
        ratios = weights.to_numpy()[1:] / weights.to_numpy()[:-1]
        assert np.allclose(ratios, res.lam, atol=1e-10)
        assert float(weights.sum()) == pytest.approx(res.longrun_multiplier, rel=1e-6)

    def test_interim_multiplier_converges_to_the_long_run(self) -> None:
        res = _fit(*_koyck_dgp(n=800, seed=30))
        assert res.interim_multiplier(0) == pytest.approx(
            res.impact_multiplier, rel=1e-12
        )
        assert res.interim_multiplier(500) == pytest.approx(
            res.longrun_multiplier, rel=1e-10
        )

    def test_forecast_variance_grows_and_accounts_for_the_ma1(self) -> None:
        """A h = 1 la variance vaut sigma^2 (1 + lambda^2), pas sigma^2.

        Ignorer le terme MA(1) — celui que la transformation de Koyck a
        cree — retrecirait l'intervalle d'exactement ce facteur.
        """
        res = _fit(*_koyck_dgp(n=500, seed=31))
        out = res.forecast([0.0, 0.0, 0.0])
        sigma2 = float(res._resid @ res._resid) / (res.nobs - 3)
        expected_1 = np.sqrt(sigma2 * (1 + res.lam**2))
        assert float(out["se"].iloc[0]) == pytest.approx(expected_1, rel=1e-12)
        assert out["se"].is_monotonic_increasing

    def test_forecast_requires_a_path(self) -> None:
        res = _fit(*_koyck_dgp(n=300, seed=32))
        with pytest.raises(ValueError, match="non-empty"):
            res.forecast([])

    def test_conf_int_and_summary(self) -> None:
        res = _fit(*_koyck_dgp(n=400, seed=33))
        ci = res.conf_int()
        assert (ci["lower"] < res.params).all()
        assert (ci["upper"] > res.params).all()
        with pytest.raises(ValueError, match="strictly in"):
            res.conf_int(alpha=1.5)
        text = res.summary()
        assert "Koyck" in text
        assert "first-stage F" in text
        assert "longrun" in text

    def test_negative_horizons_rejected(self) -> None:
        res = _fit(*_koyck_dgp(n=300, seed=34))
        with pytest.raises(ValueError, match="non-negative"):
            res.interim_multiplier(-1)
        with pytest.raises(ValueError, match="non-negative"):
            res.lag_weights(-1)
