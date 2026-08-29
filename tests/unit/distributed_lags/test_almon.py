"""Spec 02 §10.1 — Almon (1965) : la restriction, et le test de la restriction.

Un modele d'Almon produit toujours une distribution de retards lisse.
C'est ce qu'on lui a demande. Le seul test qui apprenne quelque chose
est donc celui qui la confronte au retard libre : la forme obtenue
est-elle dans les donnees, ou seulement dans l'hypothese ?

Les tests ci-dessous verifient les deux directions, comme pour la spec
01 : quand le polynome est vrai, l'estimateur le retrouve et le test ne
rejette pas ; quand il est faux, le test rejette. Une implementation qui
ne rejetterait jamais passerait la premiere moitie.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.distributed_lags import PDL, AlmonModel
from pyardl.distributed_lags.almon import _basis_matrix
from pyardl.exceptions import PyardlMethodologyWarning

Q, R = 8, 2
TRUE_WEIGHTS = np.array([0.1 * i * (Q - i) for i in range(Q + 1)])


def _almon_dgp(
    n: int, seed: int, weights: np.ndarray = TRUE_WEIGHTS, sigma: float = 0.3
):
    """y construit par convolution avec des poids donnes."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.convolve(x, weights)[:n] + 1.0 + rng.normal(scale=sigma, size=n)
    return pd.Series(y, name="y"), pd.Series(x, name="x")


def _fit(y, x, cov_type: str = "nonrobust", **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AlmonModel(y, x, **kwargs).fit(cov_type=cov_type)


class TestRecovery:
    """Spec 02 §10.1.1 — quand la restriction est vraie."""

    def test_weights_are_recovered(self) -> None:
        res = _fit(*_almon_dgp(n=2000, seed=1), q=Q, r=R)
        assert np.max(np.abs(res.lag_weights.to_numpy() - TRUE_WEIGHTS)) < 0.02

    def test_the_restriction_is_not_rejected_when_it_holds(self) -> None:
        res = _fit(*_almon_dgp(n=2000, seed=1), q=Q, r=R)
        assert float(res.polynomial_restriction_test()["pvalue"]) > 0.05

    def test_alias_pdl(self) -> None:
        assert PDL is AlmonModel

    @pytest.mark.fast_mc
    def test_reported_errors_match_the_sampling_dispersion(self) -> None:
        """Spec 02 §10.1.1 — l'erreur type annoncee est-elle la bonne ?

        Dimensionnement (regle 10) : sur 150 replications, l'erreur type
        d'un ecart type estime vaut environ 1/sqrt(2*150) = 5.8 % de
        lui-meme. La borne est donc posee a 20 %, soit trois fois ce
        bruit : le test attrape un facteur d'echelle faux (le genre
        d'erreur qu'une transformation H mal appliquee produit), pas une
        fluctuation.
        """
        weights = np.empty((150, Q + 1))
        for i in range(150):
            weights[i] = _fit(
                *_almon_dgp(n=300, seed=100 + i), q=Q, r=R
            ).lag_weights.to_numpy()
        empirical = weights.std(axis=0, ddof=1)
        reported = _fit(
            *_almon_dgp(n=300, seed=100), q=Q, r=R
        ).bse_lag_weights.to_numpy()
        assert np.max(np.abs(reported - empirical) / empirical) < 0.20


class TestRestrictionIsTested:
    """Spec 02 §10.1.5 — et quand elle est fausse ?"""

    def test_non_polynomial_weights_are_rejected(self) -> None:
        """Des poids en dents de scie : aucun polynome de degre 2 ne les suit."""
        jagged = np.array([1.0, -0.8, 1.2, -0.9, 1.1, -0.7, 1.0, -0.6, 0.9])
        res = _fit(*_almon_dgp(n=800, seed=2, weights=jagged), q=Q, r=R)
        assert float(res.polynomial_restriction_test()["pvalue"]) < 0.01

    def test_summary_says_so_when_the_restriction_is_rejected(self) -> None:
        jagged = np.array([1.0, -0.8, 1.2, -0.9, 1.1, -0.7, 1.0, -0.6, 0.9])
        res = _fit(*_almon_dgp(n=800, seed=2, weights=jagged), q=Q, r=R)
        assert "REJECTED" in res.summary()

    def test_raising_r_absorbs_a_more_complicated_shape(self) -> None:
        """Le remede que le message suggere doit fonctionner."""
        quartic = np.array(
            [0.02 * (i - 1) * (i - 3) * (i - 5) * (i - 7) for i in range(Q + 1)]
        )
        low = _fit(*_almon_dgp(n=1500, seed=3, weights=quartic), q=Q, r=2)
        high = _fit(*_almon_dgp(n=1500, seed=3, weights=quartic), q=Q, r=4)
        assert float(low.polynomial_restriction_test()["pvalue"]) < 0.01
        assert float(high.polynomial_restriction_test()["pvalue"]) > 0.05

    def test_lags_jointly_zero(self) -> None:
        res = _fit(*_almon_dgp(n=600, seed=4), q=Q, r=R)
        assert float(res.lags_are_jointly_zero()["pvalue"]) < 1e-8
        rng = np.random.default_rng(5)
        noise_y = pd.Series(rng.normal(size=400), name="y")
        noise_x = pd.Series(rng.normal(size=400), name="x")
        empty = _fit(noise_y, noise_x, q=Q, r=R)
        assert float(empty.lags_are_jointly_zero()["pvalue"]) > 0.05


class TestInvariance:
    """Spec 02 §10.1.2 et §10.1.3 — ce qui ne doit RIEN changer."""

    def test_the_two_bases_give_the_same_weights(self) -> None:
        """gamma change de sens avec la base, beta non.

        C'est ce qui autorise la base normalisee : si les deux
        parametrisations ne donnaient pas les memes beta, le
        reconditionnement changerait le modele au lieu de changer sa
        representation.
        """
        y, x = _almon_dgp(n=600, seed=6)
        power = _fit(y, x, q=10, r=4, basis="power")
        cheby = _fit(y, x, q=10, r=4, basis="chebyshev")
        assert np.allclose(
            power.lag_weights.to_numpy(), cheby.lag_weights.to_numpy(), atol=1e-8
        )
        assert np.allclose(
            power.bse_lag_weights.to_numpy(),
            cheby.bse_lag_weights.to_numpy(),
            atol=1e-8,
        )
        # ...et gamma, lui, differe : les deux bases ne sont pas la meme.
        assert not np.allclose(
            power.params_gamma.to_numpy(), cheby.params_gamma.to_numpy(), atol=1e-6
        )

    @pytest.mark.needs_review
    def test_r_equal_to_q_is_the_free_lag(self) -> None:
        """Spec 02 §10.1.2, teste sur l'algebre — voir docs/QUESTIONS.md.

        La spec se contredit : §9 exige `r >= q -> ValueError`, §10.1.2
        demande de verifier qu'a `r = q` le modele coincide avec le
        retard libre. L'API suit §9, qui est le contrat explicite ; le
        test verifie donc l'invariance la ou elle vit, dans l'algebre :
        avec H carree et inversible, regresser sur X_lag @ H puis
        remonter beta = H gamma redonne exactement l'OLS libre.

        C'est le controle qui compte, parce qu'il dit que H ne deforme
        rien : toute erreur dans la construction de la base ou dans le
        retour aux beta casserait cette egalite.
        """
        from pyardl.utils import lag_matrix

        y, x = _almon_dgp(n=400, seed=7)
        q = 6
        lags = lag_matrix(x.to_numpy(), q, first_lag=0)
        y_dep = y.to_numpy()[q:]
        design_free = np.column_stack([np.ones(y_dep.size), lags])
        beta_free, *_ = np.linalg.lstsq(design_free, y_dep, rcond=None)

        node = np.arange(q + 1, dtype=np.float64) / q
        h_mat = _basis_matrix(q, q, "power", node)
        assert h_mat.shape == (q + 1, q + 1)
        design_almon = np.column_stack([np.ones(y_dep.size), lags @ h_mat])
        gamma, *_ = np.linalg.lstsq(design_almon, y_dep, rcond=None)
        assert np.allclose(h_mat @ gamma[1:], beta_free[1:], atol=1e-8)
        assert float(gamma[0]) == pytest.approx(float(beta_free[0]), abs=1e-10)


class TestEndpoints:
    """Spec 02 §3 et §10.1.4 — les contraintes de bord."""

    def test_the_constraint_is_imposed_exactly(self) -> None:
        """beta_(q+1) = 0 doit tenir a la precision machine.

        La contrainte est imposee par reparametrisation sur le noyau,
        pas par penalisation : elle est donc exacte, et pas
        approximativement satisfaite.
        """
        y, x = _almon_dgp(n=600, seed=8)
        res = _fit(y, x, q=Q, r=3, endpoint="tail")
        node = np.array([(Q + 1.0) / Q])
        outside = _basis_matrix(Q, 3, "power", node)[0]
        gamma_full = res.extra["null_space"] @ res._coefs[1:]
        assert abs(float(outside @ gamma_full)) < 1e-10

    def test_both_endpoints(self) -> None:
        y, x = _almon_dgp(n=600, seed=9)
        res = _fit(y, x, q=Q, r=4, endpoint="both")
        assert res._coefs.size == 1 + (4 + 1) - 2

    def test_endpoint_test_does_not_reject_when_the_constraint_holds(self) -> None:
        """La contrainte porte sur i = -1, PAS sur i = 0.

        Piege qui m'a eu en ecrivant ce test : le polynome par defaut
        0.1 i (8 - i) s'annule bien en i = 0 et i = 8, mais la contrainte
        `head` demande beta_(-1) = 0, et il y vaut -0.9. Le test
        rejetait, a juste titre. Le DGP utilise ici est donc
        0.1 (i + 1) (8 - i), qui s'annule vraiment un cran avant la
        fenetre.
        """
        head_zero = np.array([0.1 * (i + 1) * (Q - i) for i in range(Q + 1)])
        y, x = _almon_dgp(n=2000, seed=10, weights=head_zero)
        res = _fit(y, x, q=Q, r=R, endpoint="head")
        assert float(res.endpoint_test()["pvalue"]) > 0.05

    def test_endpoint_test_rejects_when_it_does_not(self) -> None:
        """Des poids qui ne meurent pas au bord : la contrainte est fausse."""
        alive = np.array([0.2 * (i + 1) for i in range(Q + 1)])
        y, x = _almon_dgp(n=1500, seed=11, weights=alive)
        res = _fit(y, x, q=Q, r=R, endpoint="both")
        assert float(res.endpoint_test()["pvalue"]) < 0.01

    def test_endpoint_test_refuses_without_a_constraint(self) -> None:
        res = _fit(*_almon_dgp(n=400, seed=12), q=Q, r=R)
        with pytest.raises(ValueError, match="no constraint to test"):
            res.endpoint_test()


class TestSelectOrder:
    """Spec 02 §4 — l'echantillon commun, le piege classique."""

    def test_all_candidates_share_one_sample(self) -> None:
        """Sinon un q plus grand concourt sur moins d'observations.

        Ses criteres d'information ne sont alors plus comparables, et la
        selection recompense celui qui a jete les lignes les plus dures.
        Le test verifie l'invariant explicitement plutot que de faire
        confiance au code.
        """
        y, x = _almon_dgp(n=500, seed=13)
        table = AlmonModel.select_order(y, x, max_q=10, max_r=3)
        assert table["nobs"].nunique() == 1
        assert table["nobs"].iloc[0] == 500 - 10

    def test_it_finds_the_true_order(self) -> None:
        """Le degre exact, et une fenetre qui ne garde que ce qui porte.

        Le DGP a pour poids 0.1 i (8 - i), donc beta_8 = 0 EXACTEMENT :
        le huitieme retard n'apporte rien, et le BIC a raison de
        preferer q = 7 a q = 8. Attendre q = 8 etait mon erreur, pas
        celle du selecteur — la verite du DGP est un polynome de degre 2
        sur une fenetre effective de 7 retards.
        """
        y, x = _almon_dgp(n=800, seed=14)
        table = AlmonModel.select_order(y, x, max_q=10, max_r=3, ic="bic")
        assert int(table["r"].iloc[0]) == R
        assert int(table["q"].iloc[0]) in (Q - 1, Q)

    def test_argument_guards(self) -> None:
        y, x = _almon_dgp(n=300, seed=15)
        with pytest.raises(ValueError, match="ic must be"):
            AlmonModel.select_order(y, x, ic="mallows")
        with pytest.raises(ValueError, match="max_r"):
            AlmonModel.select_order(y, x, max_q=3, max_r=5)


class TestMultipliers:
    """Spec 02 §6 — et pourquoi l'erreur type du long terme est EXACTE ici."""

    def test_longrun_is_a_linear_form_so_its_error_is_exact(self) -> None:
        """Contrairement au Koyck, ce n'est pas un ratio.

        somme(beta) = iota' H gamma est lineaire en gamma, donc sa
        variance se transporte sans linearisation : aucune
        delta-methode, aucune approximation, et rien qui se degrade a
        petit echantillon.
        """
        res = _fit(*_almon_dgp(n=600, seed=16), q=Q, r=R)
        ones = np.ones(Q + 1)
        expected = float(np.sqrt(ones @ res._cov_weights @ ones))
        assert res.se_longrun_multiplier == pytest.approx(expected, rel=1e-12)
        assert res.longrun_multiplier == pytest.approx(
            float(res.lag_weights.sum()), rel=1e-12
        )

    def test_interim_multiplier_walks_from_impact_to_long_run(self) -> None:
        res = _fit(*_almon_dgp(n=600, seed=17), q=Q, r=R)
        first, _ = res.interim_multiplier(0)
        last, last_se = res.interim_multiplier(Q)
        assert first == pytest.approx(res.impact_multiplier, rel=1e-12)
        assert last == pytest.approx(res.longrun_multiplier, rel=1e-12)
        assert last_se == pytest.approx(res.se_longrun_multiplier, rel=1e-12)
        with pytest.raises(ValueError, match=r"\[0, q="):
            res.interim_multiplier(Q + 1)

    def test_mean_lag(self) -> None:
        res = _fit(*_almon_dgp(n=1500, seed=18), q=Q, r=R)
        value, se = res.mean_lag()
        truth = float(np.arange(Q + 1) @ TRUE_WEIGHTS / TRUE_WEIGHTS.sum())
        assert value == pytest.approx(truth, abs=0.05)
        assert se > 0

    def test_mean_lag_is_nan_when_the_weights_cancel(self) -> None:
        """Le delai moyen d'un effet dont la somme est nulle n'existe pas."""
        import dataclasses

        res = _fit(*_almon_dgp(n=400, seed=19), q=Q, r=R)
        zeroed = dataclasses.replace(res, _weights=np.zeros(Q + 1))
        value, se = zeroed.mean_lag()
        assert np.isnan(value) and np.isnan(se)


class TestDiagnosticsAndForecast:
    def test_durbin_watson_is_reported_here(self) -> None:
        """Valide dans ce modele : aucun y retarde parmi les regresseurs.

        C'est exactement la difference avec le Koyck, ou il est biaise
        vers 2 et remplace par le h de Durbin.
        """
        res = _fit(*_almon_dgp(n=600, seed=20), q=Q, r=R)
        diag = res.diagnostics()
        assert "durbin_watson" in diag.index
        assert 1.0 < float(diag.loc["durbin_watson", "statistic"]) < 3.0

    def test_forecast_matches_the_hand_computation(self) -> None:
        res = _fit(*_almon_dgp(n=500, seed=21), q=Q, r=R)
        path = np.array([1.0, 0.0, 0.0])
        out = res.forecast(path)
        history = np.concatenate([res.model._x[-Q:], path])
        expected = res.intercept + float(
            history[Q::-1][: Q + 1] @ res.lag_weights.to_numpy()
        )
        assert float(out["forecast"].iloc[0]) == pytest.approx(expected, rel=1e-12)
        assert len(out) == 3

    def test_forecast_requires_a_path(self) -> None:
        res = _fit(*_almon_dgp(n=400, seed=22), q=Q, r=R)
        with pytest.raises(ValueError, match="non-empty"):
            res.forecast([])

    def test_hac_covariance_changes_the_errors_not_the_weights(self) -> None:
        y, x = _almon_dgp(n=600, seed=23)
        plain = _fit(y, x, q=Q, r=R)
        hac = _fit(y, x, q=Q, r=R, cov_type="hac")
        assert np.allclose(
            plain.lag_weights.to_numpy(), hac.lag_weights.to_numpy(), atol=1e-12
        )
        assert not np.allclose(
            plain.bse_lag_weights.to_numpy(),
            hac.bse_lag_weights.to_numpy(),
            atol=1e-10,
        )

    def test_to_fdl_and_plot(self) -> None:
        plt = pytest.importorskip("matplotlib.pyplot")
        res = _fit(*_almon_dgp(n=600, seed=24), q=Q, r=R)
        free = res.to_fdl()
        assert list(free.columns) == ["beta", "se"]
        assert len(free) == Q + 1
        # Le retard libre est bruite, l'Almon lisse : c'est tout le sujet.
        assert free["se"].mean() > res.bse_lag_weights.mean()
        fig = res.plot_lag_distribution()
        assert fig is not None
        plt.close(fig)


class TestGuards:
    """Spec 02 §9 — ce qui doit refuser plutot que rendre un chiffre."""

    def test_r_not_smaller_than_q(self) -> None:
        y, x = _almon_dgp(n=300, seed=25)
        with pytest.raises(ValueError, match="restricts nothing"):
            AlmonModel(y, x, q=4, r=4)
        with pytest.raises(ValueError, match="restricts nothing"):
            AlmonModel(y, x, q=4, r=5)

    def test_endpoint_leaving_nothing_to_estimate(self) -> None:
        y, x = _almon_dgp(n=300, seed=26)
        with pytest.raises(ValueError, match="leaving nothing to estimate"):
            AlmonModel(y, x, q=4, r=1, endpoint="both")

    def test_unknown_endpoint_and_basis_and_cov_type(self) -> None:
        y, x = _almon_dgp(n=300, seed=27)
        with pytest.raises(ValueError, match="endpoint must be"):
            AlmonModel(y, x, q=4, r=2, endpoint="middle")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="basis must be"):
            AlmonModel(y, x, q=4, r=2, basis="legendre").fit()  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cov_type must be"):
            AlmonModel(y, x, q=4, r=2).fit(cov_type="hc3")

    def test_large_q_warns(self) -> None:
        y, x = _almon_dgp(n=60, seed=28)
        with pytest.warns(PyardlMethodologyWarning, match="third of the sample"):
            AlmonModel(y, x, q=25, r=3)

    def test_too_few_observations(self) -> None:
        y, x = _almon_dgp(n=20, seed=29)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="Not enough observations"):
                AlmonModel(y, x, q=18, r=2)

    def test_several_regressors_rejected(self) -> None:
        rng = np.random.default_rng(30)
        y = pd.Series(rng.normal(size=100), name="y")
        x = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
        with pytest.raises(ValueError, match="exactly one"):
            AlmonModel(y, x, q=4, r=2)

    def test_summary_reports_the_restriction_test(self) -> None:
        res = _fit(*_almon_dgp(n=600, seed=31), q=Q, r=R)
        text = res.summary()
        assert "Almon" in text
        assert "polynomial restriction" in text
        assert "long run" in text
