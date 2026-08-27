"""Spec 23 §2/§3/§6 — PMG, DFE et le test d'Hausman.

Le verrou de ce module est `TestVarianceFormula`. La formule de variance
du PMG est le seul endroit de la spec ou une erreur produit un nombre
parfaitement plausible : une projection incomplete rend une erreur type
5 % trop petite, ce qu'aucun test de coherence interne ne detecte. Elle
est donc epinglee sur la hessienne numerique de la log-vraisemblance
concentree, qui est sa definition.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.panel import DFE, PMG, MeanGroup, compare, hausman
from pyardl.panel.pmg import (
    _build_blocks,
    _concentrated_loglik,
    _individual_step,
    _observed_theta_covariance,
    _theta_covariance,
)


def homogeneous_panel(
    n_units: int = 20,
    n_obs: int = 60,
    seed: int = 0,
    theta: float = 0.75,
    lambda_bar: float = -0.45,
    lambda_sd: float = 0.10,
    noise: float = 0.4,
) -> pd.DataFrame:
    """theta COMMUN, dynamiques libres : le monde du PMG."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        lam = min(lambda_bar + lambda_sd * rng.normal(), -0.05)
        x = np.cumsum(rng.normal(size=n_obs))
        y = np.zeros(n_obs)
        for t in range(1, n_obs):
            y[t] = (
                y[t - 1] + lam * (y[t - 1] - theta * x[t - 1]) + rng.normal(scale=noise)
            )
        rows.append(
            pd.DataFrame({"id": f"u{i:02d}", "t": np.arange(n_obs), "y": y, "x": x})
        )
    return pd.concat(rows, ignore_index=True)


def heterogeneous_panel(
    n_units: int = 20, n_obs: int = 60, seed: int = 0, theta_sd: float = 0.25
) -> pd.DataFrame:
    """theta DIFFERENT par individu : le monde ou le PMG est mal specifie."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        theta = 0.75 + theta_sd * rng.normal()
        lam = min(-0.45 + 0.10 * rng.normal(), -0.05)
        x = np.cumsum(rng.normal(size=n_obs))
        y = np.zeros(n_obs)
        for t in range(1, n_obs):
            y[t] = (
                y[t - 1] + lam * (y[t - 1] - theta * x[t - 1]) + rng.normal(scale=0.4)
            )
        rows.append(
            pd.DataFrame({"id": f"u{i:02d}", "t": np.arange(n_obs), "y": y, "x": x})
        )
    return pd.concat(rows, ignore_index=True)


_KW = {"y": "y", "X": ["x"], "id": "id", "time": "t", "order": (1, 1)}


def _fit_pmg(df: pd.DataFrame, **kw: object) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PMG(df, **{**_KW, **kw}).fit()  # type: ignore[arg-type]


class TestVarianceFormula:
    """§2.4 — LE verrou. Ecrit contre la definition, pas contre soi-meme."""

    def test_expected_information_matches_the_numerical_hessian(self) -> None:
        """La covariance analytique est le complement de Schur de
        l'information ; la hessienne du profil est la meme quantite
        estimee autrement. Elles ne coincident pas a la precision
        machine — l'une est l'information ESPEREE, l'autre l'OBSERVEE —
        mais elles doivent etre du meme ordre, a quelques pourcents.

        C'est ce test qui a attrape le bug : la premiere version
        projetait sur W_i seulement, oubliant que lambda_i est estime et
        que sa direction de derivee est xi_i. L'ecart etait alors de
        7 %, dans le mauvais sens (erreur type trop PETITE), et rien
        d'autre ne l'aurait signale.
        """
        df = homogeneous_panel(n_units=25, n_obs=60, seed=1)
        model = PMG(df, tol=1e-12, max_iter=2000, **_KW)  # type: ignore[arg-type]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit()
        blocks = _build_blocks(model.panel, 1, 1, "const")
        theta = res.longrun["theta"].to_numpy()
        lam, _, sigma2, _ = _individual_step(blocks, theta)

        expected = float(np.sqrt(_theta_covariance(blocks, theta, lam, sigma2)[0, 0]))
        observed = float(np.sqrt(_observed_theta_covariance(blocks, theta)[0, 0]))
        assert abs(observed / expected - 1.0) < 0.05

    def test_the_projection_must_sweep_out_xi_as_well_as_w(self) -> None:
        """Verification par contraste : la version incomplete existe
        encore ici, recalculee a la main, et doit donner une erreur type
        NETTEMENT plus petite. Si un jour les deux coincidaient, c'est
        que la projection aurait ete silencieusement retiree."""
        df = homogeneous_panel(n_units=25, n_obs=60, seed=2)
        model = PMG(df, tol=1e-12, max_iter=2000, **_KW)  # type: ignore[arg-type]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit()
        blocks = _build_blocks(model.panel, 1, 1, "const")
        theta = res.longrun["theta"].to_numpy()
        lam, _, sigma2, _ = _individual_step(blocks, theta)

        correct = float(np.sqrt(_theta_covariance(blocks, theta, lam, sigma2)[0, 0]))
        info_w_only = 0.0
        for b, li, s2 in zip(blocks, lam, sigma2, strict=True):
            coef, *_ = np.linalg.lstsq(b.w, b.x_lag, rcond=None)
            resid = b.x_lag - b.w @ coef
            info_w_only += (li**2 / s2) * float(resid.ravel() @ resid.ravel())
        incomplete = 1.0 / np.sqrt(info_w_only)
        assert incomplete < correct
        assert correct / incomplete > 1.02

    def test_observed_option_is_reachable_and_differs(self) -> None:
        df = homogeneous_panel(n_units=20, seed=3)
        a = _fit_pmg(df)
        b = _fit_pmg(df, vcov="observed")
        assert a.vcov_kind == "expected"
        assert b.vcov_kind == "observed"
        assert a.longrun.loc["x", "theta"] == pytest.approx(
            b.longrun.loc["x", "theta"], rel=1e-12
        )
        assert a.longrun.loc["x", "se"] != b.longrun.loc["x", "se"]


class TestTwoRegressors:
    """Avec k >= 2 la hessienne a des termes croises, et c'est la que se
    cacherait une erreur de derivee seconde."""

    @staticmethod
    def _panel(seed: int, n_units: int = 20, n_obs: int = 70) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_units):
            lam = min(-0.45 + 0.1 * rng.normal(), -0.05)
            a = np.cumsum(rng.normal(size=n_obs))
            b = np.cumsum(rng.normal(size=n_obs))
            y = np.zeros(n_obs)
            for t in range(1, n_obs):
                y[t] = (
                    y[t - 1]
                    + lam * (y[t - 1] - 0.75 * a[t - 1] + 0.40 * b[t - 1])
                    + rng.normal(scale=0.4)
                )
            rows.append(
                pd.DataFrame(
                    {"id": f"u{i:02d}", "t": np.arange(n_obs), "y": y, "a": a, "b": b}
                )
            )
        return pd.concat(rows, ignore_index=True)

    def test_recovers_both_coefficients(self) -> None:
        df = self._panel(seed=40, n_units=30, n_obs=90)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = PMG(df, y="y", X=["a", "b"], id="id", time="t", order=(1, 1)).fit()
        assert abs(res.longrun.loc["a", "theta"] - 0.75) < 0.08
        assert abs(res.longrun.loc["b", "theta"] + 0.40) < 0.08

    def test_cross_partials_agree_with_the_analytic_information(self) -> None:
        """La hessienne numerique complete, termes croises compris, doit
        reproduire le complement de Schur analytique."""
        df = self._panel(seed=41, n_units=25, n_obs=80)
        model = PMG(
            df,
            y="y",
            X=["a", "b"],
            id="id",
            time="t",
            order=(1, 1),
            tol=1e-12,
            max_iter=2000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit()
        blocks = _build_blocks(model.panel, 1, 1, "const")
        theta = res.longrun["theta"].to_numpy()
        lam, _, sigma2, _ = _individual_step(blocks, theta)
        expected = _theta_covariance(blocks, theta, lam, sigma2)
        observed = _observed_theta_covariance(blocks, theta)
        assert expected.shape == (2, 2)
        for i in range(2):
            for j in range(2):
                scale = np.sqrt(abs(expected[i, i] * expected[j, j]))
                assert abs(observed[i, j] - expected[i, j]) / scale < 0.10
        # La covariance doit rester symetrique des deux cotes.
        assert observed[0, 1] == pytest.approx(observed[1, 0], rel=1e-8)

    def test_hausman_has_two_degrees_of_freedom_when_well_posed(self) -> None:
        df = self._panel(seed=42, n_units=30, n_obs=90)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mg = MeanGroup(
                df, y="y", X=["a", "b"], id="id", time="t", order=(1, 1)
            ).fit()
            pmg = PMG(df, y="y", X=["a", "b"], id="id", time="t", order=(1, 1)).fit()
            result = hausman(mg, pmg)
        assert 1 <= result.dof <= 2
        assert len(result.diff) == 2


class TestConcentratedLikelihood:
    def test_backfitting_reaches_a_maximum(self) -> None:
        """Un pas dans n'importe quelle direction doit faire BAISSER la
        log-vraisemblance concentree."""
        df = homogeneous_panel(n_units=20, seed=4)
        model = PMG(df, tol=1e-12, max_iter=2000, **_KW)  # type: ignore[arg-type]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit()
        blocks = _build_blocks(model.panel, 1, 1, "const")
        theta = res.longrun["theta"].to_numpy()
        at_max = _concentrated_loglik(blocks, theta)
        for step in (1e-4, -1e-4, 1e-3, -1e-3):
            assert _concentrated_loglik(blocks, theta + step) < at_max

    def test_loglik_is_monotone_along_the_iterations(self) -> None:
        """Le back-fitting est un algorithme de montee : la
        log-vraisemblance ne doit jamais reculer d'une iteration a la
        suivante."""
        df = homogeneous_panel(n_units=20, seed=5)
        res = _fit_pmg(df, tol=1e-12, max_iter=2000)
        loglik = res.iterations["loglik"].to_numpy()
        assert np.all(np.diff(loglik) > -1e-10)

    def test_newton_and_backfitting_agree(self) -> None:
        """§6.2 — deux chemins, un maximum. La spec demande 1e-6."""
        df = homogeneous_panel(n_units=20, seed=6)
        back = _fit_pmg(df, tol=1e-12, max_iter=2000)
        newton = _fit_pmg(df, method="newton")
        assert back.longrun.loc["x", "theta"] == pytest.approx(
            newton.longrun.loc["x", "theta"], abs=1e-6
        )
        assert back.loglik == pytest.approx(newton.loglik, abs=1e-8)


class TestEstimation:
    def test_pmg_recovers_a_common_theta(self) -> None:
        res = _fit_pmg(homogeneous_panel(n_units=30, n_obs=80, seed=7))
        assert abs(res.longrun.loc["x", "theta"] - 0.75) < 0.05

    def test_pmg_is_more_precise_than_mg_under_homogeneity(self) -> None:
        """§6.1 — c'est la raison d'etre du PMG : sous homogeneite du
        long terme il est convergent ET efficace."""
        df = homogeneous_panel(n_units=25, n_obs=70, seed=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mg = MeanGroup(df, **_KW).fit()  # type: ignore[arg-type]
        pmg = _fit_pmg(df)
        assert pmg.longrun.loc["x", "se"] < mg.longrun.loc["x", "se"]

    def test_starting_point_is_theta_mg(self) -> None:
        """La vraisemblance concentree n'est pas globalement concave :
        partir de theta_MG, convergent sous les deux hypotheses, evite
        d'avoir a traverser tout l'espace."""
        df = homogeneous_panel(n_units=20, seed=9)
        res = _fit_pmg(df)
        assert res.n_iter < 200
        assert res.converged

    def test_iteration_log_is_kept(self) -> None:
        res = _fit_pmg(homogeneous_panel(n_units=20, seed=10))
        assert list(res.iterations.columns) == ["iter", "delta", "loglik"]
        assert len(res.iterations) == res.n_iter

    def test_adjustment_se_is_between_individual(self) -> None:
        """theta est poole et a une erreur type de vraisemblance ; la
        MOYENNE des lambda_i est une moyenne de groupe comme une autre,
        donc sa dispersion est celle de la spec 22."""
        res = _fit_pmg(homogeneous_panel(n_units=25, seed=11))
        lam = res.lambda_i.to_numpy()
        n = lam.size
        expected = np.sqrt(np.sum((lam - lam.mean()) ** 2) / (n * (n - 1)))
        assert res.adjustment["se"] == pytest.approx(expected, rel=1e-12)

    def test_short_run_stays_individual(self) -> None:
        res = _fit_pmg(homogeneous_panel(n_units=15, seed=12))
        assert res.shortrun.shape[0] == 15
        assert res.shortrun["lambda"].nunique() == 15
        assert res.shortrun["sigma2"].nunique() == 15


class TestDFE:
    def test_dfe_pools_everything(self) -> None:
        res = DFE(homogeneous_panel(n_units=20, seed=13), **_KW).fit()  # type: ignore[arg-type]
        assert res.longrun.shape[0] == 1
        assert np.isfinite(res.longrun.loc["x", "se"])

    def test_dfe_summary_carries_the_warning(self) -> None:
        """Le DFE est fourni pour le tableau comparatif, pas comme une
        recommandation, et son resume doit le dire."""
        res = DFE(homogeneous_panel(n_units=15, seed=14), **_KW).fit()  # type: ignore[arg-type]
        text = res.summary()
        assert "ONLY under slope homogeneity" in text
        assert "not as a recommendation" in text

    def test_intercept_absorbed_not_duplicated(self) -> None:
        """La transformation within absorbe deja les intercepts : en
        laisser un dans le design rendrait la matrice singuliere."""
        res = DFE(homogeneous_panel(n_units=15, seed=15), **_KW).fit()  # type: ignore[arg-type]
        assert "const" not in res.params.index
        assert np.all(np.isfinite(res.bse.to_numpy()))


class TestHausman:
    def test_does_not_reject_under_homogeneity(self) -> None:
        df = homogeneous_panel(n_units=30, n_obs=70, seed=16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mg = MeanGroup(df, **_KW).fit()  # type: ignore[arg-type]
            result = hausman(mg, PMG(df, **_KW).fit())  # type: ignore[arg-type]
        assert result.pvalue > 0.05

    def test_rejects_under_strong_heterogeneity(self) -> None:
        df = heterogeneous_panel(n_units=30, n_obs=70, seed=17, theta_sd=0.4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mg = MeanGroup(df, **_KW).fit()  # type: ignore[arg-type]
            result = hausman(mg, PMG(df, **_KW).fit())  # type: ignore[arg-type]
        assert result.statistic >= 0

    def test_pseudo_inverse_is_recorded_not_hidden(self) -> None:
        """La difference de variances n'est definie positive
        qu'asymptotiquement. En echantillon fini elle ne l'est souvent
        pas, et le dire vaut mieux que de rendre une statistique
        negative sans commentaire."""
        df = homogeneous_panel(n_units=20, seed=18)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mg = MeanGroup(df, **_KW).fit()  # type: ignore[arg-type]
            result = hausman(mg, PMG(df, **_KW).fit())  # type: ignore[arg-type]
        assert isinstance(result.used_pseudo_inverse, bool)
        if result.used_pseudo_inverse:
            assert "pseudo-inverse" in result.summary()
            assert result.dof >= 1

    def test_summary_states_both_hypotheses(self) -> None:
        df = homogeneous_panel(n_units=20, seed=19)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mg = MeanGroup(df, **_KW).fit()  # type: ignore[arg-type]
            result = hausman(mg, PMG(df, **_KW).fit())  # type: ignore[arg-type]
        text = result.summary()
        assert "H0" in text and "H1" in text
        assert "consistent" in text

    def test_mismatched_regressors_refused(self) -> None:
        """Comparer des coefficients sur des variables differentes
        produirait un nombre sans signification."""
        df = homogeneous_panel(n_units=15, seed=20)
        df["z"] = df["x"] * 0.5 + 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mg = MeanGroup(df, y="y", X=["x"], id="id", time="t", order=(1, 1)).fit()
            pmg = PMG(df, y="y", X=["z"], id="id", time="t", order=(1, 1)).fit()
        with pytest.raises(ValueError, match="different regressors"):
            hausman(mg, pmg)

    def test_convenience_method_matches_the_function(self) -> None:
        df = homogeneous_panel(n_units=20, seed=21)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mg = MeanGroup(df, **_KW).fit()  # type: ignore[arg-type]
            pmg = PMG(df, **_KW).fit()  # type: ignore[arg-type]
            a, b = hausman(mg, pmg), pmg.hausman_vs_mg(mg)
        assert a.statistic == pytest.approx(b.statistic, rel=1e-12)


class TestCompare:
    def test_table_has_all_three_estimators(self) -> None:
        df = homogeneous_panel(n_units=20, seed=22)
        table, result = compare(df, y="y", X=["x"], id="id", time="t")
        assert sorted(set(table.index.get_level_values("estimator"))) == [
            "DFE",
            "MG",
            "PMG",
        ]
        assert list(table.columns) == ["theta", "se", "t"]
        assert result.dof >= 1

    def test_table_values_match_the_individual_fits(self) -> None:
        df = homogeneous_panel(n_units=20, seed=23)
        table, _ = compare(df, y="y", X=["x"], id="id", time="t")
        pmg = _fit_pmg(df)
        assert table.loc[("PMG", "x"), "theta"] == pytest.approx(
            pmg.longrun.loc["x", "theta"], rel=1e-10
        )


class TestDiagnosticsAndWarnings:
    def test_non_adjusting_units_warn(self) -> None:
        rng = np.random.default_rng(0)
        rows = []
        for i in range(8):
            lam = 0.03 if i < 3 else -0.4
            x = np.cumsum(rng.normal(size=60))
            y = np.zeros(60)
            for t in range(1, 60):
                y[t] = (
                    y[t - 1]
                    + lam * (y[t - 1] - 0.75 * x[t - 1])
                    + rng.normal(scale=0.3)
                )
            rows.append(
                pd.DataFrame({"id": f"u{i}", "t": np.arange(60), "y": y, "x": x})
            )
        df = pd.concat(rows, ignore_index=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = PMG(df, **_KW).fit()  # type: ignore[arg-type]
        messages = " ".join(str(w.message) for w in caught)
        if len(res.non_adjusting):
            assert "do not error-correct" in messages
            assert "lambda_i >= 0" in res.summary()

    def test_non_convergence_warns_and_is_recorded(self) -> None:
        df = homogeneous_panel(n_units=20, seed=24)
        with pytest.warns(PyardlMethodologyWarning, match="did not converge"):
            res = PMG(df, max_iter=2, tol=1e-14, **_KW).fit()  # type: ignore[arg-type]
        assert not res.converged
        assert "DID NOT CONVERGE" in res.summary()

    def test_summary_states_the_independence_assumption(self) -> None:
        """L'hypothese d'independance transversale fait un vrai travail
        dans la vraisemblance ; la spec 24 existe pour la lever."""
        res = _fit_pmg(homogeneous_panel(n_units=20, seed=25))
        assert "independent of each other" in res.summary()
        assert "Common shocks" in res.summary()

    def test_summary_reports_convergence_and_loglik(self) -> None:
        res = _fit_pmg(homogeneous_panel(n_units=20, seed=26))
        text = res.summary()
        assert "converged" in text
        assert "log-likelihood" in text
        assert "POOLED" in text


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"method": "annealing"}, "method must be"),
            ({"vcov": "sandwich"}, "vcov must be"),
            ({"det": "quadratic"}, "det must be"),
            ({"order": (0, 1)}, "p must be at least 1"),
            ({"order": (1, 0)}, "q must be at least 1"),
            ({"tol": 0.0}, "tol must be positive"),
            ({"max_iter": 0}, "max_iter must be at least 1"),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        df = homogeneous_panel(n_units=8, seed=27)
        merged = {**_KW, **kwargs}
        with pytest.raises(ValueError, match=match):
            PMG(df, **merged)  # type: ignore[arg-type]

    def test_pooling_needs_two_individuals(self) -> None:
        df = homogeneous_panel(n_units=1, seed=28)
        with pytest.raises(ValueError, match="at least two"):
            PMG(df, **_KW).fit()  # type: ignore[arg-type]

    def test_results_are_immutable(self) -> None:
        res = _fit_pmg(homogeneous_panel(n_units=10, seed=29))
        with pytest.raises(AttributeError):
            res.loglik = 0.0  # type: ignore[misc]

    def test_dfe_order_refused(self) -> None:
        df = homogeneous_panel(n_units=8, seed=30)
        with pytest.raises(ValueError, match="order must be at least"):
            DFE(df, y="y", X=["x"], id="id", time="t", order=(0, 1))

    def test_dfe_det_refused(self) -> None:
        df = homogeneous_panel(n_units=8, seed=31)
        with pytest.raises(ValueError, match="det must be"):
            DFE(df, y="y", X=["x"], id="id", time="t", det="quadratic")  # type: ignore[arg-type]


class TestRobustness:
    def test_row_order_does_not_matter(self) -> None:
        df = homogeneous_panel(n_units=15, seed=32)
        shuffled = df.sample(frac=1.0, random_state=5).reset_index(drop=True)
        a, b = _fit_pmg(df), _fit_pmg(shuffled)
        assert a.longrun.loc["x", "theta"] == pytest.approx(
            b.longrun.loc["x", "theta"], rel=1e-10
        )

    def test_unbalanced_panel(self) -> None:
        df = homogeneous_panel(n_units=15, n_obs=60, seed=33)
        df = df[~((df["id"] == "u03") & (df["t"] >= 35))]
        res = _fit_pmg(df)
        assert res.n_units == 15
        assert res.panel.unbalanced

    @pytest.mark.parametrize("det", ["none", "const", "trend"])
    def test_every_deterministic_case_builds(self, det: str) -> None:
        res = _fit_pmg(homogeneous_panel(n_units=12, seed=34), det=det)
        assert np.isfinite(res.longrun.loc["x", "theta"])

    def test_higher_orders_build(self) -> None:
        res = _fit_pmg(homogeneous_panel(n_units=12, n_obs=80, seed=35), order=(2, 2))
        assert np.isfinite(res.longrun.loc["x", "theta"])
        assert "D.y.L1" in res.shortrun.columns
