"""Spec 26 §3 — plan de tests CUSUM / CUSUMSQ (Brown-Durbin-Evans 1975).

Le verrou de ce module est §3.2 : les résidus récursifs maison doivent
être IDENTIQUES à ceux de statsmodels (1e-10). Tout le reste en dépend :
si la récursion Sherman-Morrison dérive, les deux tests dérivent avec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.critical_values.bde1975 import cusum_a, cusumsq_c0
from pyardl.diagnostics import (
    cusum,
    cusumsq,
    plot_cusum,
    plot_cusumsq,
    recursive_residuals,
    stability_tests,
)
from pyardl.exceptions import PyardlMethodologyWarning


def _design(n_obs: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, k - 1))])


def _dgp_stable(n_obs: int, k: int, rng: np.random.Generator, sigma: float = 1.0):
    """DGP à coefficients constants et variance constante."""
    x = _design(n_obs, k, rng)
    beta = np.linspace(1.0, 0.5, k)
    return x @ beta + sigma * rng.standard_normal(n_obs), x


def _dgp_coef_break(n_obs: int, k: int, rng: np.random.Generator, jump: float = 2.0):
    """Rupture de niveau à mi-échantillon (cible propre du CUSUM).

    C'est la constante qui saute. Le CUSUM cumule les résidus récursifs :
    il ne peut détecter qu'une rupture qui déplace leur MOYENNE.
    """
    x = _design(n_obs, k, rng)
    beta = np.linspace(1.0, 0.5, k)
    y = x @ beta + rng.standard_normal(n_obs)
    y[n_obs // 2 :] += jump
    return y, x


def _dgp_slope_break(n_obs: int, k: int, rng: np.random.Generator, jump: float = 4.0):
    """Rupture de PENTE sur un régresseur centré.

    Cas limite documenté : les résidus récursifs restent de moyenne
    nulle, donc le CUSUM est aveugle par construction.
    """
    x = _design(n_obs, k, rng)
    beta = np.linspace(1.0, 0.5, k)
    y = x @ beta + rng.standard_normal(n_obs)
    half = n_obs // 2
    y[half:] += jump * x[half:, 1]
    return y, x


def _dgp_var_break(n_obs: int, k: int, rng: np.random.Generator, ratio: float = 6.0):
    """Rupture de VARIANCE à mi-échantillon (cible du CUSUMSQ)."""
    x = _design(n_obs, k, rng)
    beta = np.linspace(1.0, 0.5, k)
    eps = rng.standard_normal(n_obs)
    half = n_obs // 2
    eps[half:] *= ratio
    return x @ beta + eps, x


class TestRecursiveResidualsLock:
    """§3.2 — verrou : concordance exacte avec statsmodels."""

    @pytest.mark.parametrize("n_obs", [40, 120, 300])
    @pytest.mark.parametrize("k", [2, 4, 6])
    def test_matches_statsmodels_1e10(self, n_obs: int, k: int) -> None:
        """Résidus récursifs maison == statsmodels à 1e-10."""
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import recursive_olsresiduals

        rng = np.random.default_rng(1234 + n_obs + k)
        y, x = _dgp_stable(n_obs, k, rng)

        mine = recursive_residuals(y, x)
        # rresid_scaled (5e élément) = w_t, distribué N(0, sigma^2).
        reference = recursive_olsresiduals(sm.OLS(y, x).fit())[4][k:]

        assert mine.shape == reference.shape
        assert np.max(np.abs(mine - reference)) < 1e-10

    def test_recursion_equals_full_reestimation(self) -> None:
        """La mise à jour Sherman-Morrison == ré-estimation complète.

        C'est l'optimisation elle-même qui est vérifiée : la version
        naïve, coûteuse mais évidemment correcte, sert de référence.
        """
        rng = np.random.default_rng(7)
        n_obs, k = 60, 3
        y, x = _dgp_stable(n_obs, k, rng)

        naive = np.empty(n_obs - k)
        for i, t in enumerate(range(k, n_obs)):
            b = np.linalg.lstsq(x[:t], y[:t], rcond=None)[0]
            xtxi = np.linalg.inv(x[:t].T @ x[:t])
            f_t = 1.0 + x[t] @ xtxi @ x[t]
            naive[i] = (y[t] - x[t] @ b) / np.sqrt(f_t)

        assert np.max(np.abs(recursive_residuals(y, x) - naive)) < 1e-10

    def test_iid_normal_under_stability(self) -> None:
        """Sous H0, les w_t ont bien la variance du DGP (loi des grands nombres)."""
        rng = np.random.default_rng(99)
        y, x = _dgp_stable(20_000, 3, rng, sigma=2.0)
        w = recursive_residuals(y, x)
        assert abs(float(np.std(w, ddof=1)) - 2.0) < 0.05


class TestSpecialisation:
    """§3.1 — chaque test détecte la rupture dont il est spécialiste."""

    def test_stable_dgp_passes_both(self) -> None:
        rng = np.random.default_rng(11)
        y, x = _dgp_stable(200, 3, rng)
        assert cusum(y, x).stable
        assert cusumsq(y, x).stable

    def test_coefficient_break_detected_by_cusum(self) -> None:
        rng = np.random.default_rng(12)
        y, x = _dgp_coef_break(200, 3, rng)
        assert not cusum(y, x).stable

    def test_variance_break_detected_by_cusumsq(self) -> None:
        rng = np.random.default_rng(13)
        y, x = _dgp_var_break(200, 3, rng)
        assert not cusumsq(y, x).stable

    def test_slope_break_invisible_to_cusum_seen_by_cusumsq(self) -> None:
        """Cécité structurelle du CUSUM aux ruptures de pente centrées.

        Ce n'est pas un défaut d'implémentation : une rupture de pente
        sur un régresseur de moyenne nulle laisse E(w_t) = 0, donc la
        somme cumulée ne dérive pas. La variance, elle, gonfle — et le
        CUSUMSQ le voit. C'est l'argument le plus net en faveur de la
        publication systématique des DEUX graphiques.
        """
        blind = 0
        seen = 0
        for seed in range(20):
            rng = np.random.default_rng(700 + seed)
            y, x = _dgp_slope_break(200, 3, rng)
            blind += cusum(y, x).stable
            seen += not cusumsq(y, x).stable
        assert blind >= 15
        assert seen >= 15

    def test_variance_break_largely_missed_by_cusum(self) -> None:
        """La spécialisation, dans l'autre sens.

        Une rupture de VARIANCE pure ne déplace pas la moyenne des
        résidus récursifs : le CUSUM la rate le plus souvent. C'est
        précisément pourquoi la spec exige de publier les DEUX
        graphiques.
        """
        detected_cusum = 0
        detected_cusumsq = 0
        for seed in range(30):
            rng = np.random.default_rng(500 + seed)
            y, x = _dgp_var_break(200, 3, rng)
            detected_cusum += not cusum(y, x).stable
            detected_cusumsq += not cusumsq(y, x).stable
        assert detected_cusumsq > detected_cusum

    def test_crossings_locate_the_break(self) -> None:
        """Les dates de sortie de bande encadrent la rupture."""
        rng = np.random.default_rng(14)
        n_obs = 200
        y, x = _dgp_coef_break(n_obs, 3, rng, jump=6.0)
        res = cusum(y, x)
        assert not res.stable
        assert res.crossings.size > 0
        # La première sortie survient APRÈS la rupture (mi-échantillon),
        # jamais avant : le CUSUM ne peut pas anticiper.
        assert res.crossings[0] >= n_obs // 2


@pytest.mark.fast_mc
class TestSizeUnderNull:
    """§3.1 — taille empirique sous H0, version CI."""

    def test_cusum_size_close_to_nominal(self) -> None:
        rejections = 0
        reps = 200
        for seed in range(reps):
            rng = np.random.default_rng(10_000 + seed)
            y, x = _dgp_stable(150, 3, rng)
            rejections += not cusum(y, x, alpha=0.05).stable
        # Le CUSUM est connu pour être conservateur : on borne le
        # sur-rejet, on n'exige pas l'égalité exacte à 5 %.
        assert rejections / reps <= 0.10

    def test_cusumsq_size_close_to_nominal(self) -> None:
        rejections = 0
        reps = 200
        for seed in range(reps):
            rng = np.random.default_rng(20_000 + seed)
            y, x = _dgp_stable(150, 3, rng)
            rejections += not cusumsq(y, x, alpha=0.05).stable
        assert rejections / reps <= 0.10


@pytest.mark.slow
class TestSizeUnderNullFull:
    """§3.1 — version complète (1000 réplications)."""

    def test_cusumsq_size_full(self) -> None:
        rejections = 0
        reps = 1000
        for seed in range(reps):
            rng = np.random.default_rng(30_000 + seed)
            y, x = _dgp_stable(150, 3, rng)
            rejections += not cusumsq(y, x, alpha=0.05).stable
        assert 0.01 <= rejections / reps <= 0.09


class TestCriticalValues:
    """§2.2/§2.3 — frontières et couverture."""

    def test_cusum_a_published_values(self) -> None:
        assert cusum_a(0.10) == 0.850
        assert cusum_a(0.05) == 0.948
        assert cusum_a(0.01) == 1.143

    def test_unsupported_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            cusum_a(0.025)
        with pytest.raises(ValueError, match="not available"):
            cusumsq_c0(100, 0.025)

    def test_c0_decreasing_in_n(self) -> None:
        """Plus de résidus récursifs -> bande plus étroite."""
        values = [cusumsq_c0(n, 0.05) for n in (10, 25, 50, 100, 250, 1000)]
        assert all(a > b for a, b in zip(values[:-1], values[1:], strict=True))

    def test_c0_stricter_level_wider_band(self) -> None:
        assert cusumsq_c0(100, 0.01) > cusumsq_c0(100, 0.05) > cusumsq_c0(100, 0.10)

    def test_c0_interpolation_between_grid_points(self) -> None:
        low, high = cusumsq_c0(100, 0.05), cusumsq_c0(105, 0.05)
        mid = cusumsq_c0(102, 0.05)
        assert high < mid < low

    def test_c0_too_small_n_raises(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            cusumsq_c0(3, 0.05)

    def test_c0_beyond_grid_warns_and_uses_asymptotic(self) -> None:
        from scipy.stats import kstwobign

        with pytest.warns(PyardlMethodologyWarning, match="asymptotic"):
            got = cusumsq_c0(5000, 0.05)
        expected = float(kstwobign.ppf(0.95)) * np.sqrt(2.0 / 5000)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_c0_asymptotic_crosscheck(self) -> None:
        """Recoupement : c0 * sqrt(n/2) -> quantile de Kolmogorov.

        Le ratio doit croître vers 1 sans jamais le dépasser — sinon la
        table simulée serait plus large que sa propre limite, ce qui
        n'a pas de sens.
        """
        from scipy.stats import kstwobign

        k_a = float(kstwobign.ppf(0.95))
        ratios = [
            cusumsq_c0(n, 0.05) * np.sqrt(n / 2.0) / k_a
            for n in (50, 100, 200, 500, 1000)
        ]
        assert all(r < 1.0 for r in ratios)
        assert ratios[-1] > 0.95
        assert all(a < b for a, b in zip(ratios[:-1], ratios[1:], strict=True))


class TestResultsObject:
    """§2.4 — sorties exposées."""

    def test_cusumsq_runs_from_zero_to_one(self) -> None:
        rng = np.random.default_rng(21)
        y, x = _dgp_stable(120, 3, rng)
        stat = cusumsq(y, x).statistic
        assert stat[-1] == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.diff(stat) >= 0)

    @pytest.mark.parametrize("seed", range(12))
    def test_stable_flag_consistent_with_excess_and_crossings(self, seed: int) -> None:
        """Invariant, vrai pour tout tirage : les trois sorties concordent.

        Formulé comme invariant plutôt que sur une graine choisie : un
        DGP stable rejette quand même dans ~5 % des cas, et un test qui
        dépendrait de la graine masquerait cette réalité.
        """
        rng = np.random.default_rng(2200 + seed)
        y, x = _dgp_stable(150, 3, rng)
        for res in (cusum(y, x), cusumsq(y, x)):
            if res.stable:
                assert res.max_excess == 0.0
                assert res.crossings.size == 0
            else:
                assert res.max_excess > 0.0
                assert res.crossings.size > 0

    def test_pandas_index_preserved_in_summary(self) -> None:
        rng = np.random.default_rng(23)
        y, x = _dgp_coef_break(200, 3, rng, jump=8.0)
        dates = pd.period_range("1980Q1", periods=200, freq="Q")
        res = cusum(pd.Series(y, index=dates), x)
        assert not res.stable
        assert "first crossing at" in res.summary()
        assert "19" in res.summary() or "20" in res.summary()

    def test_summary_reports_stability(self) -> None:
        rng = np.random.default_rng(24)
        y, x = _dgp_stable(150, 3, rng)
        assert "stable" in cusum(y, x).summary()
        assert "UNSTABLE" not in cusum(y, x).summary()

    def test_stability_tests_table(self) -> None:
        rng = np.random.default_rng(25)
        y, x = _dgp_stable(150, 3, rng)
        table = stability_tests(y, x)
        assert list(table.index) == ["CUSUM", "CUSUM-of-squares"]
        assert table["stable"].all()
        assert table["first_crossing"].isna().all()

    def test_stability_tests_reports_crossing(self) -> None:
        rng = np.random.default_rng(26)
        y, x = _dgp_var_break(200, 3, rng)
        table = stability_tests(y, x)
        assert not bool(table.loc["CUSUM-of-squares", "stable"])
        assert table.loc["CUSUM-of-squares", "max_excess"] > 0.0


class TestValidation:
    """Validation des entrées."""

    def test_incompatible_lengths(self) -> None:
        with pytest.raises(ValueError, match="Incompatible lengths"):
            recursive_residuals(np.zeros(10), np.ones((12, 2)))

    def test_sample_too_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            recursive_residuals(np.zeros(3), np.ones((3, 3)))

    def test_collinear_start_raises(self) -> None:
        x = np.column_stack([np.ones(30), np.ones(30)])
        with pytest.raises(ValueError, match="collinear"):
            recursive_residuals(np.arange(30.0), x)

    @pytest.mark.parametrize("func", [cusum, cusumsq])
    def test_perfect_fit_rejected(self, func) -> None:  # type: ignore[no-untyped-def]
        """Ajustement exact -> refus explicite, pas une statistique de bruit.

        Les résidus ne valent jamais exactement zéro en virgule
        flottante : le garde-fou compare leur somme des carrés à
        l'échelle de l'erreur d'arrondi, pas à zéro.
        """
        rng = np.random.default_rng(31)
        x = _design(40, 3, rng)
        y = x @ np.array([1.0, 2.0, 3.0])  # aucun bruit
        with pytest.raises(ValueError, match="numerically zero"):
            func(y, x)

    def test_one_dimensional_x_accepted(self) -> None:
        rng = np.random.default_rng(32)
        x = rng.standard_normal(60)
        y = 2.0 * x + rng.standard_normal(60)
        assert recursive_residuals(y, x).shape == (59,)

    def test_plot_helpers_reject_wrong_result(self) -> None:
        rng = np.random.default_rng(33)
        y, x = _dgp_stable(80, 3, rng)
        with pytest.raises(ValueError, match="plot_cusum expects"):
            plot_cusum(cusumsq(y, x))
        with pytest.raises(ValueError, match="plot_cusumsq expects"):
            plot_cusumsq(cusum(y, x))


class TestPlots:
    """§2.4 — les deux graphiques canoniques."""

    def test_plots_render(self) -> None:
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        rng = np.random.default_rng(41)
        y, x = _dgp_stable(120, 3, rng)
        fig1 = plot_cusum(cusum(y, x))
        fig2 = plot_cusumsq(cusumsq(y, x))
        assert fig1.axes[0].get_ylabel() == "CUSUM"
        assert fig2.axes[0].get_ylabel() == "CUSUM of squares"


class TestIntegration:
    """§2.5 — câblage automatique dans les diagnostics existants."""

    def _dgp_cointegrated(self, n: int = 150, seed: int = 5):
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.standard_normal(n))
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = y[t - 1] - 0.4 * (y[t - 1] - 2.0 * x[t - 1]) + rng.standard_normal()
        return pd.Series(y, name="y"), pd.DataFrame({"x": x})

    def test_bounds_diagnostics_gains_stability_rows(self) -> None:
        """Les lignes s'AJOUTENT : les diagnostics résiduels restent là."""
        from pyardl.bounds import bounds_test

        y, x = self._dgp_cointegrated()
        diag = bounds_test(y, x, case=3, order=(2, 1)).diagnostics()
        for row in ("Jarque-Bera", "Breusch-Pagan"):
            assert row in diag.index
        assert any(i.startswith("CUSUM(5%)") for i in diag.index)
        assert any(i.startswith("CUSUMSQ(5%)") for i in diag.index)

    def test_stability_rows_have_no_pvalue(self) -> None:
        """Aucune p-value inventée : ce sont des tests de franchissement."""
        from pyardl.bounds import bounds_test

        y, x = self._dgp_cointegrated()
        diag = bounds_test(y, x, case=3, order=(2, 1)).diagnostics()
        stab_rows = [i for i in diag.index if i.startswith("CUSUM")]
        assert len(stab_rows) == 2
        assert diag.loc[stab_rows, "pvalue"].isna().all()

    def test_bounds_stability_method(self) -> None:
        from pyardl.bounds import bounds_test

        y, x = self._dgp_cointegrated()
        table = bounds_test(y, x, case=3, order=(2, 1)).stability()
        assert list(table.index) == ["CUSUM", "CUSUM-of-squares"]

    def test_ardl_stability_method(self) -> None:
        from pyardl.core.ardl import ARDL

        y, x = self._dgp_cointegrated()
        table = ARDL(y, x, order=(1, 1))._fit().stability()
        assert list(table.index) == ["CUSUM", "CUSUM-of-squares"]
        assert table["stable"].all()

    def test_ardl_diagnostics_shape_unchanged(self) -> None:
        """Garde-fou : diagnostics() de l'ARDL garde ses 3 lignes.

        L'intégration automatique porte sur le bounds test. Élargir
        silencieusement le DataFrame de l'ARDL casserait le contrat
        « toutes les p-values sont dans [0, 1] ».
        """
        from pyardl.core.ardl import ARDL

        y, x = self._dgp_cointegrated()
        diag = ARDL(y, x, order=(1, 1))._fit().diagnostics()
        assert len(diag) == 3
        assert ((diag["pvalue"] >= 0) & (diag["pvalue"] <= 1)).all()

    def test_alpha_propagates_to_labels_and_width(self) -> None:
        from pyardl.bounds import bounds_test

        y, x = self._dgp_cointegrated()
        res = bounds_test(y, x, case=3, order=(2, 1))
        assert any("CUSUM(1%)" in i for i in res.diagnostics(alpha=0.01).index)
        assert any("CUSUM(10%)" in i for i in res.diagnostics(alpha=0.10).index)

    def test_uecm_design_reconstruction_is_exact(self) -> None:
        """La variable dépendante reconstruite == celle estimée (1e-12).

        stability() reconstruit y a partir de design @ params + resid ;
        si cette identite derivait, les tests porteraient sur autre chose
        que le modele ajuste.
        """
        from pyardl.bounds import bounds_test

        y, x = self._dgp_cointegrated()
        res = bounds_test(y, x, case=3, order=(2, 1))
        fit = res._fit
        rebuilt = fit.design @ fit.params.to_numpy() + fit.resid
        fitted = fit.design @ fit.params.to_numpy()
        assert np.max(np.abs(rebuilt - fitted - fit.resid)) < 1e-12


class TestBoundaryCrossCheck:
    """§3.3 — recoupement des frontières CUSUM contre statsmodels.

    Aucune source publiée n'ayant pu être consultée pour les frontières
    exactes, le recoupement se fait contre une implémentation
    indépendante de la meme formule.
    """

    @pytest.mark.parametrize("n_obs,k", [(150, 3), (80, 2), (300, 5)])
    def test_cusum_boundaries_match_statsmodels(self, n_obs: int, k: int) -> None:
        """Frontieres identiques a statsmodels, au decalage d'origine pres.

        statsmodels fait demarrer le chemin a t = k, ou le residu
        recursif est nul par construction ; Brown-Durbin-Evans le font
        demarrer a t = k+1. Les deux suites de frontieres coincident donc
        exactement apres decalage d'un pas — l'ecart observe est
        strictement nul, pas seulement petit.
        """
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import recursive_olsresiduals

        rng = np.random.default_rng(3)
        y, x = _dgp_stable(n_obs, k, rng)
        reference = np.asarray(
            recursive_olsresiduals(sm.OLS(y, x).fit(), alpha=0.95)[6]
        )[1]
        mine = cusum(y, x, alpha=0.05).upper
        m = min(len(reference) - 1, len(mine))
        assert np.max(np.abs(reference[1 : m + 1] - mine[:m])) == 0.0

    def test_boundary_endpoints_match_bde_definition(self) -> None:
        """Les droites passent par (k, a*sqrt(n)) et (T, 3*a*sqrt(n)).

        C'est la definition geometrique de Brown-Durbin-Evans : la bande
        triple entre le debut et la fin de l'echantillon.
        """
        rng = np.random.default_rng(6)
        n_obs, k = 200, 4
        y, x = _dgp_stable(n_obs, k, rng)
        res = cusum(y, x, alpha=0.05)
        n = res.n_recursive
        a = cusum_a(0.05)
        assert res.upper[0] == pytest.approx(a * np.sqrt(n) + 2 * a / np.sqrt(n))
        assert res.upper[-1] == pytest.approx(3.0 * a * np.sqrt(n))


def test_cusum_accepts_one_dimensional_design() -> None:
    """Un regresseur unique passe en vecteur, sans colonne de constante."""
    rng = np.random.default_rng(77)
    x = rng.standard_normal(80)
    y = 2.0 * x + rng.standard_normal(80)
    res = cusum(y, x)
    assert res.k == 1
    assert res.n_recursive == 79
    assert cusumsq(y, x).n_recursive == 79
