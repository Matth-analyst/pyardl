"""Spec 14 §4 — plan de tests du bootstrap ARDL.

Deux verrous, dans cet ordre :

1. Reproductibilité de bout en bout — même seed, mêmes valeurs critiques
   au bit près. Une valeur critique bootstrap non reproductible n'est pas
   vérifiable par un tiers, donc pas publiable.
2. Le DGP sous H0 régénère bien des séries NON cointégrées. C'est le
   verrou statistique : si le modèle nul était mal imposé, les valeurs
   critiques seraient fausses tout en paraissant plausibles.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.bootstrap import (
    bootstrap_bounds_test,
    estimate_null_dgp,
    simulate_path,
)
from pyardl.bootstrap.dgp import simulate_paths
from pyardl.bootstrap.resample import resample_residuals
from pyardl.bounds import bounds_test
from pyardl.exceptions import PyardlMethodologyWarning


def _cointegrated(n: int, seed: int, k: int = 1, lam: float = -0.4):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, k)), axis=0)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] + lam * (y[t - 1] - x[t - 1].sum()) + rng.standard_normal()
    return y, x


def _no_cointegration(n: int, seed: int, k: int = 1):  # type: ignore[no-untyped-def]
    """Marches aléatoires indépendantes : H0 exactement vraie."""
    rng = np.random.default_rng(seed)
    return (
        np.cumsum(rng.standard_normal(n)),
        np.cumsum(rng.standard_normal((n, k)), axis=0),
    )


class TestReproducibility:
    """§2.5 — verrou : même seed, même résultat, bit à bit."""

    def test_same_seed_identical_critical_values(self) -> None:
        y, x = _cointegrated(90, seed=1)
        a = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=7)
        b = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=7)
        for alpha in (0.10, 0.05, 0.01):
            assert a.f_critical[alpha] == b.f_critical[alpha]
            assert a.t_critical[alpha] == b.t_critical[alpha]
        assert a.f_pvalue == b.f_pvalue
        assert a.t_pvalue == b.t_pvalue

    def test_different_seed_different_values(self) -> None:
        y, x = _cointegrated(90, seed=2)
        a = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1)
        b = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=2)
        assert a.f_critical[0.05] != b.f_critical[0.05]

    def test_seed_is_recorded_even_when_not_given(self) -> None:
        """Sans seed explicite, une seed est tirée PUIS journalisée.

        Un run doit rester reproductible après coup, même lancé sans
        précaution.
        """
        y, x = _cointegrated(80, seed=3)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199)
        assert isinstance(res.seed, int)
        again = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=res.seed)
        assert again.f_critical[0.05] == res.f_critical[0.05]

    def test_all_settings_recorded(self) -> None:
        y, x = _cointegrated(80, seed=4)
        res = bootstrap_bounds_test(
            y,
            x,
            order=(1, 1),
            n_boot=199,
            seed=5,
            resample="wild",
            var_order=2,
            burn_in=30,
        )
        assert res.resample == "wild"
        assert res.var_order == 2
        assert res.burn_in == 30
        assert res.case == 3
        assert res.order[0] == 1

    def test_observed_statistics_match_the_classical_test(self) -> None:
        """Le bootstrap ne change QUE les valeurs critiques."""
        y, x = _cointegrated(100, seed=6)
        classical = bounds_test(y, x, case=3, order=(1, 1))
        boot = bootstrap_bounds_test(y, x, case=3, order=(1, 1), n_boot=199, seed=1)
        assert boot.f_stat == classical.f_stat
        assert boot.t_stat == classical.t_stat


class TestNullDGP:
    """§2.2 — le modèle sous H0 régénère bien des séries non cointégrées."""

    def test_regenerated_data_reject_at_nominal_rate(self) -> None:
        """Verrou statistique.

        On estime le DGP nul sur des données COINTÉGRÉES, puis on
        régénère : les séries produites ne doivent plus l'être. Le test
        des bornes classique doit donc rejeter à un taux proche du
        nominal, pas systématiquement.
        """
        y, x = _cointegrated(120, seed=11, k=2)
        dgp = estimate_null_dgp(y, x, p=2, q=(1, 1), case=3, var_order=1)
        rng = np.random.default_rng(21)
        rejections = 0
        reps = 60
        for _ in range(reps):
            inn = resample_residuals(dgp.residuals, 50 + 120, rng)
            y_b, x_b = simulate_path(dgp, inn, y0=y[0], x0=x[0], burn_in=50)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = bounds_test(y_b, x_b, case=3, order=(2, 1))
            rejections += res.decision_f == "cointegration"
        assert rejections / reps <= 0.20

    def test_level_terms_absent_from_the_null_model(self) -> None:
        """Sous H0 les niveaux disparaissent : seuls psi et omega restent."""
        y, x = _cointegrated(100, seed=12)
        dgp = estimate_null_dgp(y, x, p=3, q=(2,), case=3, var_order=1)
        assert dgp.psi.size == 2  # p - 1
        assert dgp.omega[0].size == 2  # q_0
        assert dgp.n_regressors == 1

    def test_residuals_are_centred(self) -> None:
        """Des résidus non centrés injecteraient une dérive dans chaque
        trajectoire régénérée."""
        y, x = _cointegrated(120, seed=13, k=2)
        dgp = estimate_null_dgp(y, x, p=2, q=(1, 1), case=3)
        assert np.abs(dgp.residuals.mean(axis=0)).max() < 1e-12

    def test_residual_blocks_are_aligned(self) -> None:
        """Une colonne par équation : conditionnelle + bloc marginal."""
        y, x = _cointegrated(120, seed=14, k=3)
        dgp = estimate_null_dgp(y, x, p=2, q=(1, 1, 1), case=3)
        assert dgp.residuals.shape[1] == 1 + 3

    def test_deterministic_case_respected(self) -> None:
        """Cases 1 et 2 : pas de constante non restreinte dans le nul.

        Sous les cas 2 et 4 le déterministe restreint appartient au
        vecteur testé ; il disparaît donc avec les niveaux.
        """
        y, x = _cointegrated(100, seed=15)
        for case in (1, 2):
            dgp = estimate_null_dgp(y, x, p=1, q=(1,), case=case)
            assert dgp.y_const == 0.0
            assert dgp.y_trend == 0.0
        assert estimate_null_dgp(y, x, p=1, q=(1,), case=3).y_const != 0.0
        assert estimate_null_dgp(y, x, p=1, q=(1,), case=5).y_trend != 0.0

    def test_burn_in_discards_initial_periods(self) -> None:
        y, x = _cointegrated(100, seed=16)
        dgp = estimate_null_dgp(y, x, p=1, q=(1,), case=3)
        inn = resample_residuals(dgp.residuals, 40 + 100, np.random.default_rng(1))
        y_b, x_b = simulate_path(dgp, inn, y0=y[0], x0=x[0], burn_in=40)
        assert y_b.shape == (100,)
        assert x_b.shape == (100, 1)

    def test_burn_in_too_large_rejected(self) -> None:
        y, x = _cointegrated(80, seed=17)
        dgp = estimate_null_dgp(y, x, p=1, q=(1,), case=3)
        inn = resample_residuals(dgp.residuals, 50, np.random.default_rng(1))
        with pytest.raises(ValueError, match="leaves no observation"):
            simulate_path(dgp, inn, y0=y[0], x0=x[0], burn_in=50)


class TestDecisions:
    """§2.4 — décision binaire, plus de zone non concluante."""

    def test_cointegrated_data_rejected(self) -> None:
        y, x = _cointegrated(120, seed=31, lam=-0.6)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=499, seed=1)
        assert res.decision_f(0.05) == "cointegration"
        assert res.classification(0.05)[0] == "cointegration"

    def test_independent_walks_not_rejected(self) -> None:
        y, x = _no_cointegration(120, seed=32)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=499, seed=1)
        assert res.decision_f(0.05) == "no_cointegration"

    def test_no_inconclusive_state(self) -> None:
        """Le verdict est binaire : aucune valeur ne vaut 'inconclusive'."""
        y, x = _cointegrated(100, seed=33)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=299, seed=1)
        for alpha in (0.10, 0.05, 0.01):
            assert res.decision_f(alpha) in ("cointegration", "no_cointegration")
            assert res.decision_t(alpha) in ("cointegration", "no_cointegration")

    def test_pvalue_never_exactly_zero(self) -> None:
        """(1 + #) / (B + 1) : jamais zéro.

        Annoncer p = 0 revendiquerait une résolution que B réplications
        ne fournissent pas.
        """
        y, x = _cointegrated(120, seed=34, lam=-0.8)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=299, seed=1)
        assert res.f_pvalue > 0.0
        assert res.f_pvalue >= 1.0 / (res.n_boot + 1)

    def test_critical_values_ordered(self) -> None:
        y, x = _cointegrated(120, seed=35)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=499, seed=1)
        assert res.f_critical[0.10] < res.f_critical[0.05] < res.f_critical[0.01]
        assert res.t_critical[0.10] > res.t_critical[0.05] > res.t_critical[0.01]

    def test_classification_is_a_named_outcome(self) -> None:
        """Spec 15 §2.4 — le verdict joint sort de la table de décision."""
        from pyardl.bounds.classification import CLASSIFICATIONS

        y, x = _cointegrated(100, seed=36)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=299, seed=1)
        label, reason = res.classification(0.01)
        assert label in CLASSIFICATIONS
        assert reason


class TestComparisonWithBounds:
    """Spec 16 §2.3 — les deux routes en regard."""

    def test_comparison_has_one_row_per_test(self) -> None:
        y, x = _cointegrated(120, seed=61)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1)
        comp = res.comparison()
        assert list(comp.index) == ["F_overall", "t_BDM", "F_indep"]
        assert list(comp.columns) == [
            "statistic",
            "boot_cv",
            "boot_p",
            "boot_decision",
            "bound_I0",
            "bound_I1",
            "bound_decision",
        ]

    def test_statistics_match_the_result_object(self) -> None:
        y, x = _cointegrated(120, seed=62)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1)
        comp = res.comparison()
        assert comp.loc["F_overall", "statistic"] == pytest.approx(res.f_stat)
        assert comp.loc["t_BDM", "statistic"] == pytest.approx(res.t_stat)
        assert comp.loc["F_indep", "statistic"] == pytest.approx(res.f_indep_stat)

    def test_classical_bounds_are_the_classical_ones(self) -> None:
        """Aucune borne n'est recalculee dans la comparaison : elle lit
        celles du test classique, sinon les deux routes pourraient
        diverger sans que personne le voie."""
        y, x = _cointegrated(120, seed=63)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1)
        comp = res.comparison(0.05)
        assert comp.loc["F_overall", "bound_I0"] == pytest.approx(
            res.classical.bounds.loc[0.05, "F_I0"]
        )
        assert comp.loc["F_indep", "bound_I1"] == pytest.approx(
            res.classical.bounds.loc[0.05, "F_indep_I1"]
        )

    def test_unavailable_classical_decision_is_named_not_hidden(self) -> None:
        """Cas II : PSS ne tabule pas le t. La cellule le DIT au lieu de
        laisser une case vide qu'on lirait comme un non-rejet."""
        y, x = _cointegrated(120, seed=64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bootstrap_bounds_test(y, x, case=2, order=(1, 1), n_boot=199, seed=1)
        assert res.comparison().loc["t_BDM", "bound_decision"] == "unavailable"

    def test_agreement_flag_matches_the_two_classifications(self) -> None:
        y, x = _cointegrated(120, seed=65)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1)
        expected = res.classification(0.05)[0] == res.classical.classification()[0]
        assert res.agrees_with_bounds(0.05) is expected

    def test_summary_shows_both_routes(self) -> None:
        y, x = _cointegrated(120, seed=66)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1)
        text = res.summary()
        assert "bootstrap against classical bounds" in text
        assert "classification: bootstrap ->" in text


class TestSizeAndPower:
    """§4.1 et §4.2 — taille et puissance."""

    @pytest.mark.fast_mc
    def test_size_under_null(self) -> None:
        """Taille empirique sous H0, grille réduite pour la CI."""
        rejections = 0
        reps = 60
        for seed in range(reps):
            y, x = _no_cointegration(80, seed=5000 + seed)
            res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=seed)
            rejections += res.decision_f(0.05) == "cointegration"
        assert rejections / reps <= 0.15

    @pytest.mark.fast_mc
    def test_power_against_cointegration(self) -> None:
        detected = 0
        reps = 40
        for seed in range(reps):
            y, x = _cointegrated(100, seed=6000 + seed, lam=-0.6)
            res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=seed)
            detected += res.decision_f(0.05) == "cointegration"
        assert detected / reps >= 0.70

    @pytest.mark.fast_mc
    def test_resolves_inconclusive_cases(self) -> None:
        """§4.2 — la zone non concluante du test classique est résorbée.

        On compte les échantillons que le test des bornes laisse sans
        verdict : le bootstrap en tranche chacun.
        """
        inconclusive = 0
        for seed in range(40):
            y, x = _cointegrated(60, seed=7000 + seed, lam=-0.25)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classical = bounds_test(y, x, case=3, order=(1, 1))
            if classical.decision_f == "inconclusive":
                inconclusive += 1
                res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=seed)
                assert res.decision_f(0.05) in (
                    "cointegration",
                    "no_cointegration",
                )
        assert inconclusive > 0, "le DGP ne produit aucun cas non concluant"

    @pytest.mark.slow
    def test_size_full(self) -> None:
        rejections = 0
        reps = 300
        for seed in range(reps):
            y, x = _no_cointegration(80, seed=8000 + seed)
            res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=299, seed=seed)
            rejections += res.decision_f(0.05) == "cointegration"
        assert 0.01 <= rejections / reps <= 0.12


class TestWildScheme:
    """§2.3 — le schéma wild sous hétéroscédasticité."""

    def _heteroskedastic(self, n: int, seed: int):  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.standard_normal((n, 1)), axis=0)
        scale = np.ones(n)
        scale[n // 2 :] = 4.0
        y = np.cumsum(rng.standard_normal(n) * scale)
        return y, x

    @pytest.mark.fast_mc
    def test_wild_size_under_heteroskedasticity(self) -> None:
        rejections = 0
        reps = 50
        for seed in range(reps):
            y, x = self._heteroskedastic(90, 9000 + seed)
            res = bootstrap_bounds_test(
                y, x, order=(1, 1), n_boot=199, seed=seed, resample="wild"
            )
            rejections += res.decision_f(0.05) == "cointegration"
        assert rejections / reps <= 0.18

    def test_wild_is_reproducible(self) -> None:
        y, x = self._heteroskedastic(90, 1)
        a = bootstrap_bounds_test(
            y, x, order=(1, 1), n_boot=199, seed=3, resample="wild"
        )
        b = bootstrap_bounds_test(
            y, x, order=(1, 1), n_boot=199, seed=3, resample="wild"
        )
        assert a.f_critical[0.05] == b.f_critical[0.05]


class TestResultsObject:
    def test_summary_shows_both_approaches(self) -> None:
        y, x = _cointegrated(100, seed=41)
        text = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1).summary()
        assert "Bootstrap bounds test" in text
        assert "bootstrap critical values" in text
        # Spec 16 §2.3 : la comparaison est devenue un tableau test par
        # test, plus une ligne resumee ; la phrase d'origine a disparu.
        assert "bootstrap against classical bounds" in text
        assert "seed=1" in text

    def test_distribution_optional(self) -> None:
        y, x = _cointegrated(90, seed=42)
        without = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1)
        assert without.distribution is None
        with_dist = bootstrap_bounds_test(
            y, x, order=(1, 1), n_boot=199, seed=1, store_distribution=True
        )
        assert isinstance(with_dist.distribution, pd.DataFrame)
        assert list(with_dist.distribution.columns) == ["F", "t", "F_indep"]
        assert len(with_dist.distribution) == with_dist.n_boot

    def test_classical_result_attached(self) -> None:
        y, x = _cointegrated(90, seed=43)
        res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1)
        assert res.classical.decision_f in (
            "cointegration",
            "no_cointegration",
            "inconclusive",
        )

    def test_order_selected_automatically(self) -> None:
        y, x = _cointegrated(100, seed=44)
        res = bootstrap_bounds_test(y, x, n_boot=199, seed=1, max_p=2, max_q=2)
        assert res.order[0] >= 1


class TestValidation:
    def test_too_few_replications_rejected(self) -> None:
        y, x = _cointegrated(80, seed=51)
        with pytest.raises(ValueError, match="too small"):
            bootstrap_bounds_test(y, x, order=(1, 1), n_boot=50)

    def test_no_regressor_rejected(self) -> None:
        """L'erreur remonte du test classique, appelé en premier.

        Une seule source de vérité pour cette validation : la dupliquer
        ici ferait diverger les deux messages au premier changement.
        """
        with pytest.raises(ValueError, match="requires x regressors"):
            bootstrap_bounds_test(np.arange(60.0), None, n_boot=199)  # type: ignore[arg-type]

    def test_var_order_too_large_rejected(self) -> None:
        """Le VAR marginal doit rester estimable.

        Le seuil dépend de T et de k : à T = 40 et k = 1 il faut
        dépasser 18 retards pour épuiser les degrés de liberté. La borne
        est calculée, pas devinée.
        """
        y, x = _cointegrated(40, seed=52)
        with pytest.raises(ValueError, match="Sample too short"):
            bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, var_order=25)

    def test_negative_var_order_rejected(self) -> None:
        y, x = _cointegrated(80, seed=53)
        with pytest.raises(ValueError, match="non-negative"):
            estimate_null_dgp(y, x, p=1, q=(1,), case=3, var_order=-1)

    def test_unestimable_replications_warn(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """Une réplication non estimable est comptée, jamais retirée en
        silence ni remplacée par un nouveau tirage — ce qui biaiserait la
        distribution vers les échantillons estimables."""
        import pyardl.bootstrap.bounds as mod

        real = mod.batch_uecm_statistics

        def flaky(*args, **kwargs):  # type: ignore[no-untyped-def]
            f_val, t_val, i_val, ssr, ok = real(*args, **kwargs)
            ok = ok.copy()
            ok[::10] = False  # une réplication sur dix devient inestimable
            return f_val, t_val, i_val, ssr, ok

        monkeypatch.setattr(mod, "batch_uecm_statistics", flaky)
        y, x = _cointegrated(90, seed=54)
        with pytest.warns(PyardlMethodologyWarning, match="discarded"):
            res = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=199, seed=1)
        assert res.n_failed > 0
        assert res.n_boot + res.n_failed == 199


class TestVectorisedRecursion:
    """§2.5 — la récursion batchée == la récursion scalaire.

    La régénération représentait jusqu'à 79 % du temps ; elle est
    désormais vectorisée sur l'axe des réplications. Le gain ne vaut que
    si l'arithmétique est rigoureusement inchangée.
    """

    def _naive_path(self, dgp, inn, y0, x0, burn_in):  # type: ignore[no-untyped-def]
        """Récursion scalaire écrite indépendamment, période par période."""
        k = dgp.n_regressors
        r = dgp.var_order
        n_total = inn.shape[0]
        lag_max = max(dgp.p, max(dgp.q, default=0), r, 1)
        dx = np.zeros((n_total, k))
        dy = np.zeros(n_total)
        for t in range(lag_max, n_total):
            acc = dgp.x_const.copy()
            for i in range(r):
                acc = acc + dgp.x_ar[i] @ dx[t - i - 1]
            dx[t] = acc + inn[t, 1:]
            val = dgp.y_const + dgp.y_trend * (t - burn_in + 1)
            for i in range(1, dgp.p):
                val += dgp.psi[i - 1] * dy[t - i]
            for j in range(k):
                for i in range(dgp.q[j]):
                    val += dgp.omega[j][i] * dx[t - i, j]
            dy[t] = val + inn[t, 0]
        return (
            np.cumsum(dy[burn_in:]) + y0,
            np.cumsum(dx[burn_in:], axis=0) + x0,
        )

    @pytest.mark.parametrize("case", [1, 3, 5])
    @pytest.mark.parametrize("k", [1, 3])
    def test_batched_matches_naive_recursion(self, case: int, k: int) -> None:
        y, x = _cointegrated(120, seed=60 + case + k, k=k)
        dgp = estimate_null_dgp(y, x, p=2, q=tuple([1] * k), case=case, var_order=1)
        rng = np.random.default_rng(71)
        block = np.stack(
            [resample_residuals(dgp.residuals, 30 + 120, rng) for _ in range(5)]
        )
        y_batch, x_batch = simulate_paths(dgp, block, y0=y[0], x0=x[0], burn_in=30)
        for b in range(5):
            y_ref, x_ref = self._naive_path(dgp, block[b], y[0], x[0], burn_in=30)
            assert np.max(np.abs(y_batch[b] - y_ref)) < 1e-10
            assert np.max(np.abs(x_batch[b] - x_ref)) < 1e-10

    def test_single_path_is_the_batched_one(self) -> None:
        """simulate_path délègue : les deux routes ne peuvent pas diverger."""
        y, x = _cointegrated(100, seed=72, k=2)
        dgp = estimate_null_dgp(y, x, p=2, q=(1, 1), case=3)
        inn = resample_residuals(dgp.residuals, 40 + 100, np.random.default_rng(73))
        y_one, x_one = simulate_path(dgp, inn, y0=y[0], x0=x[0], burn_in=40)
        y_many, x_many = simulate_paths(
            dgp, inn[None, ...], y0=y[0], x0=x[0], burn_in=40
        )
        assert np.array_equal(y_one, y_many[0])
        assert np.array_equal(x_one, x_many[0])

    def test_trend_enters_the_regenerated_data(self) -> None:
        """Sous le cas 5, la tendance du modèle nul doit être régénérée.

        Elle était estimée puis ignorée : les données bootstrap ne
        portaient pas la tendance que le modèle nul décrit, et les
        valeurs critiques du cas 5 s'en trouvaient fausses.
        """
        y, x = _cointegrated(120, seed=74)
        dgp = estimate_null_dgp(y, x, p=1, q=(1,), case=5)
        assert dgp.y_trend != 0.0
        inn = np.zeros((1, 30 + 120, 2))  # aucune innovation
        y_star, _ = simulate_paths(dgp, inn, y0=0.0, x0=np.zeros(1), burn_in=30)
        # Sans bruit, Delta y = c + delta * t : y suit une parabole, donc
        # sa différence seconde vaut exactement delta.
        second_diff = np.diff(np.diff(y_star[0]))
        assert np.allclose(second_diff, dgp.y_trend, atol=1e-9)

    def test_chunking_does_not_change_results(self) -> None:
        """Le découpage en blocs est un détail d'exécution, pas de calcul."""
        import pyardl.bootstrap.bounds as mod

        y, x = _cointegrated(100, seed=75)
        original = mod._CHUNK
        try:
            mod._CHUNK = 1000
            big = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=299, seed=9)
            mod._CHUNK = 7
            small = bootstrap_bounds_test(y, x, order=(1, 1), n_boot=299, seed=9)
        finally:
            mod._CHUNK = original
        assert big.f_critical[0.05] == small.f_critical[0.05]
        assert big.t_critical[0.05] == small.t_critical[0.05]

    def test_batch_shape_validation(self) -> None:
        y, x = _cointegrated(80, seed=76)
        dgp = estimate_null_dgp(y, x, p=1, q=(1,), case=3)
        with pytest.raises(ValueError, match="must be 3-D"):
            simulate_paths(dgp, np.zeros((10, 2)), y0=0.0, x0=np.zeros(1))
        with pytest.raises(ValueError, match="columns for"):
            simulate_paths(dgp, np.zeros((2, 80, 5)), y0=0.0, x0=np.zeros(1))
        with pytest.raises(ValueError, match="must be 2-D"):
            simulate_path(dgp, np.zeros((2, 80, 2)), y0=0.0, x0=np.zeros(1))


class TestBatchedEstimator:
    """§2.5 — la QR empilée == l'estimateur mono-échantillon.

    L'estimation concentrait 76 à 96 % du temps après vectorisation de
    la régénération. Elle est maintenant résolue par QR empilée pour tout
    le bloc. Le verrou : les statistiques doivent être celles de
    l'estimateur de la spec 10, sur les CINQ cas déterministes — sinon le
    bootstrap testerait un autre modèle que celui d'où vient la
    statistique observée.
    """

    @pytest.mark.parametrize("case", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("k,p,q", [(1, 1, 1), (2, 2, 1), (3, 3, 2), (2, 1, 0)])
    def test_matches_single_sample_estimator(
        self, case: int, k: int, p: int, q: int
    ) -> None:
        from pyardl.bootstrap.batch import batch_uecm_statistics
        from pyardl.bounds.pss import _estimate_uecm, _wald_f, _wald_f_indep

        rng = np.random.default_rng(100 + case + k + p + q)
        n_rep, n_obs = 8, 120
        y_b = np.cumsum(rng.standard_normal((n_rep, n_obs)), axis=1)
        x_b = np.cumsum(rng.standard_normal((n_rep, n_obs, k)), axis=1)
        q_tuple = tuple([q] * k)
        names = tuple(f"x{j}" for j in range(k))

        f_batch, t_batch, i_batch, _, ok = batch_uecm_statistics(
            y_b, x_b, p, q_tuple, case
        )
        assert ok.all()

        for b in range(n_rep):
            fit = _estimate_uecm(y_b[b], x_b[b], names, "y", p, q_tuple, case)
            pos = fit.names.index(fit.lam_name)
            t_ref = float(fit.params[fit.lam_name]) / float(np.sqrt(fit.cov[pos][pos]))
            f_ref = _wald_f(fit)
            i_ref = _wald_f_indep(fit)
            assert abs(f_batch[b] - f_ref) < 1e-9 * max(1.0, f_ref)
            assert abs(t_batch[b] - t_ref) < 1e-9
            # Troisieme statistique (spec 15), sur le meme ajustement.
            assert abs(i_batch[b] - i_ref) < 1e-9 * max(1.0, i_ref)

    def test_restricted_deterministic_is_tested_in_cases_2_and_4(self) -> None:
        """Sous les cas 2 et 4 le vecteur testé compte k+2 restrictions.

        Le F en dépend directement : oublier le déterministe restreint
        donnerait un test à k+1 restrictions, donc une statistique et un
        degré de liberté faux.
        """
        from pyardl.bootstrap.batch import _build_designs

        rng = np.random.default_rng(200)
        y_b = np.cumsum(rng.standard_normal((3, 100)), axis=1)
        x_b = np.cumsum(rng.standard_normal((3, 100, 2)), axis=1)
        for case, expected in ((1, 3), (2, 4), (3, 3), (4, 4), (5, 3)):
            _, tested, _ = _build_designs(y_b, x_b, 1, (1, 1), case)
            assert len(tested) == expected, f"case {case}"

    def test_singular_replication_flagged_not_raised(self) -> None:
        """Une réplication dégénérée est signalée, pas propagée.

        Avec B en milliers, interrompre tout le run parce qu'un
        échantillon régénéré a dégénéré serait pire que le compter.
        """
        from pyardl.bootstrap.batch import batch_uecm_statistics

        rng = np.random.default_rng(300)
        y_b = np.cumsum(rng.standard_normal((4, 100)), axis=1)
        x_b = np.cumsum(rng.standard_normal((4, 100, 1)), axis=1)
        # Réplication 2 : régresseur constant -> design singulier.
        x_b[2, :, 0] = 1.0
        *_, ok = batch_uecm_statistics(y_b, x_b, 1, (1,), 3)
        assert not ok[2]
        assert ok[[0, 1, 3]].all()

    def test_too_few_observations_rejected(self) -> None:
        from pyardl.bootstrap.batch import batch_uecm_statistics

        rng = np.random.default_rng(400)
        y_b = np.cumsum(rng.standard_normal((2, 12)), axis=1)
        x_b = np.cumsum(rng.standard_normal((2, 12, 3)), axis=1)
        with pytest.raises(ValueError, match="cannot be estimated"):
            batch_uecm_statistics(y_b, x_b, 4, (4, 4, 4), 5)
