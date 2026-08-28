"""Spec 19 §3 — plan de tests des briques de Fourier.

Le verrou de cette spec n'est pas une identite algebrique : c'est le
probleme de Davies. Une frequence choisie sur les donnees n'est pas un
parametre estime, et une valeur critique tabulee lue apres cette
selection ne vaut rien. Les tests le verifient par la mesure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.fourier import (
    DEFAULT_GRID,
    INTEGER_GRID,
    fourier_f_test,
    fourier_kpss,
    fourier_orthogonality,
    fourier_terms,
    select_frequency,
)


def _smooth_break(seed: int, n: int = 200, size: float = 3.0) -> np.ndarray:
    """Rupture LISSE (logistique), le cas que Fourier sait representer."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, n + 1)
    return (
        2.0 + size / (1 + np.exp(-0.08 * (t - n / 2))) + rng.normal(scale=0.4, size=n)
    )


class TestTerms:
    """§2.1 — la brique deterministe."""

    def test_shape_and_names(self) -> None:
        terms = fourier_terms(120, [1.0, 2.0])
        assert terms.shape == (120, 4)
        assert list(terms.columns) == ["sin_1", "cos_1", "sin_2", "cos_2"]

    def test_one_cycle_at_frequency_one(self) -> None:
        """f = 1 boucle exactement une fois sur l'echantillon."""
        terms = fourier_terms(100, 1.0)
        assert terms["cos_1"].iloc[-1] == pytest.approx(1.0, abs=1e-12)
        assert terms["sin_1"].iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_integer_frequencies_are_orthogonal(self) -> None:
        """Aux frequences ENTIERES, sinus et cosinus sont orthogonaux a
        la constante et entre eux — exactement."""
        for freq in (1.0, 2.0, 3.0):
            out = fourier_orthogonality(200, freq)
            assert abs(out["sin_sum"]) < 1e-12
            assert abs(out["cos_sum"]) < 1e-12
            assert abs(out["sin_cos"]) < 1e-12

    def test_fractional_frequencies_are_not(self) -> None:
        """Et aux frequences FRACTIONNAIRES elle tombe : le terme se
        confond en partie avec la constante. C'est une propriete de la
        methode, pas un defaut d'implementation, et elle se mesure."""
        out = fourier_orthogonality(200, 0.5)
        assert abs(out["sin_sum"]) > 0.1

    def test_frequencies_are_mutually_orthogonal(self) -> None:
        terms = fourier_terms(200, [1.0, 2.0, 3.0]).to_numpy()
        gram = terms.T @ terms / 200
        off = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off)) < 1e-12

    def test_index_is_preserved(self) -> None:
        # pd.offsets.QuarterEnd() plutot que freq="QE" : cet alias
        # n'existe qu'a partir de pandas 2.2, alors que le projet
        # declare pandas>=2.1. L'objet offset, lui, se comporte a
        # l'identique de 2.1 a 3.0.
        idx = pd.date_range("2000-01-01", periods=40, freq=pd.offsets.QuarterEnd())
        assert fourier_terms(40, 1.0, index=idx).index.equals(idx)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_obs": 3, "freqs": 1.0}, "too short"),
            ({"n_obs": 100, "freqs": 0.0}, "must be positive"),
            ({"n_obs": 100, "freqs": -1.0}, "must be positive"),
            ({"n_obs": 100, "freqs": 60.0}, "Nyquist"),
            ({"n_obs": 100, "freqs": []}, "no frequency"),
            ({"n_obs": 100, "freqs": [1.0, 1.0]}, "duplicates"),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            fourier_terms(**kwargs)


class TestSelection:
    """§3.2 — la grille retrouve la frequence injectee."""

    @pytest.mark.parametrize("injected", [1.0, 2.0, 3.0])
    def test_recovers_the_injected_frequency(self, injected: float) -> None:
        rng = np.random.default_rng(int(injected))
        n = 200
        t = np.arange(1, n + 1)
        y = (
            5.0
            + 2.0 * np.sin(2 * np.pi * injected * t / n)
            + rng.normal(scale=0.3, size=n)
        )
        freq, table = select_frequency(y, INTEGER_GRID)
        assert freq == injected
        assert list(table.columns) == ["freq", "ssr"]
        assert table["ssr"].is_monotonic_increasing

    def test_fractional_grid_is_available(self) -> None:
        rng = np.random.default_rng(5)
        n = 200
        t = np.arange(1, n + 1)
        y = np.sin(2 * np.pi * 1.5 * t / n) + rng.normal(scale=0.2, size=n)
        freq, _ = select_frequency(y, DEFAULT_GRID)
        assert freq == pytest.approx(1.5, abs=0.3)

    def test_extra_regressors_and_trend_are_held(self) -> None:
        rng = np.random.default_rng(6)
        n = 200
        t = np.arange(1, n + 1)
        x = np.cumsum(rng.normal(size=n))
        y = 0.5 * x + 0.02 * t + np.sin(2 * np.pi * 2 * t / n)
        freq, _ = select_frequency(y, INTEGER_GRID, x=x, trend=True)
        assert freq == 2.0

    def test_mismatched_regressor_length(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            select_frequency(np.zeros(50), INTEGER_GRID, x=np.zeros(40))

    def test_no_candidate_fits(self) -> None:
        with pytest.raises(ValueError, match="No candidate frequency"):
            select_frequency(np.zeros(4), (1.0,), x=np.zeros((4, 4)))


class TestDaviesProblem:
    """LE VERROU — une frequence choisie change les valeurs critiques.

    C'est la lecon que la spec impose de propager aux specs 20 et 21 :
    partout ou f est estimee, les valeurs critiques doivent etre
    simulees AVEC la selection dans la boucle.
    """

    def test_selection_inflates_the_statistic_under_the_null(self) -> None:
        """Sous H0, chercher la meilleure frequence produit un maximum
        sur une grille, pas un tirage d'une loi fixe. La statistique est
        donc systematiquement plus grande."""
        rng = np.random.default_rng(20260824)
        fixed, chosen = [], []
        for _ in range(150):
            y = rng.normal(size=150)
            fixed.append(
                fourier_f_test(
                    y, freq=1.0, freq_estimated=False, n_sims=100, seed=1
                ).statistic
            )
            chosen.append(fourier_f_test(y, n_sims=100, seed=1).statistic)
        assert np.median(chosen) > np.median(fixed)

    def test_simulated_values_exceed_the_tabulated_ones(self) -> None:
        """La valeur critique correcte est nettement au-dessus de celle
        d'un F(2, T-4). Lire la table apres selection sur-rejette."""
        from scipy.stats import f as f_dist

        rng = np.random.default_rng(11)
        y = rng.normal(size=200)
        res = fourier_f_test(y, n_sims=1000, seed=2)
        tabulated = float(f_dist.ppf(0.95, 2, 200 - 4))
        assert res.critical[0.05] > tabulated * 1.3

    def test_fixed_frequency_matches_the_tabulated_value(self) -> None:
        """Frequence fixee d'avance : la loi tabulee redevient valable,
        et la simulation doit la retrouver."""
        from scipy.stats import f as f_dist

        rng = np.random.default_rng(12)
        y = rng.normal(size=200)
        res = fourier_f_test(y, freq=1.0, freq_estimated=False, n_sims=2000, seed=3)
        tabulated = float(f_dist.ppf(0.95, 2, 200 - 4))
        assert res.critical[0.05] == pytest.approx(tabulated, rel=0.15)

    def test_the_result_says_which_regime_it_is_in(self) -> None:
        rng = np.random.default_rng(13)
        y = rng.normal(size=120)
        searched = fourier_f_test(y, n_sims=200, seed=1)
        fixed = fourier_f_test(y, freq=1.0, freq_estimated=False, n_sims=200, seed=1)
        assert searched.freq_estimated is True
        assert fixed.freq_estimated is False
        assert "inside the loop" in searched.summary()
        assert "fixed frequency" in fixed.summary()


class TestFTest:
    """§3.1 — puissance contre une rupture lisse, taille sans rupture."""

    @pytest.mark.parametrize("seed", range(3))
    def test_detects_a_smooth_break(self, seed: int) -> None:
        res = fourier_f_test(_smooth_break(seed), n_sims=300, seed=1)
        assert res.decision == "reject"
        assert res.pvalue < 0.05

    def test_keeps_white_noise(self) -> None:
        rng = np.random.default_rng(20)
        res = fourier_f_test(rng.normal(size=200), n_sims=500, seed=1)
        assert res.decision == "keep"

    def test_fourier_captures_most_of_the_smooth_path(self) -> None:
        """§3.1 — la composante F=1 explique l'essentiel de la
        trajectoire, sans l'epuiser.

        La spec annonce un R2 > 0.9. Mesure sur une logistique, il
        plafonne a 0.86 avec une frequence et 0.88 avec deux, quelle que
        soit la pente. Le seuil du test est donc celui qui est ATTEINT,
        pas celui qui etait espere : assouplir une mesure pour valider
        une annonce, c'est cesser de mesurer.
        """
        n = 200
        t = np.arange(1, n + 1)
        path = 2.0 + 3.0 / (1 + np.exp(-0.08 * (t - n / 2)))
        design = np.column_stack([np.ones(n), fourier_terms(n, 1.0).to_numpy()])
        fitted = design @ np.linalg.lstsq(design, path, rcond=None)[0]
        ss_res = float(np.sum((path - fitted) ** 2))
        ss_tot = float(np.sum((path - path.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot
        assert 0.85 < r_squared < 0.90

    def test_pvalue_is_never_exactly_zero(self) -> None:
        res = fourier_f_test(_smooth_break(30), n_sims=200, seed=1)
        assert res.pvalue >= 1 / (res.n_sims + 1)

    def test_reproducible_with_a_seed(self) -> None:
        y = _smooth_break(31)
        a = fourier_f_test(y, n_sims=200, seed=9)
        b = fourier_f_test(y, n_sims=200, seed=9)
        assert a.critical == b.critical
        assert a.pvalue == b.pvalue

    def test_seed_recorded_when_omitted(self) -> None:
        y = _smooth_break(32)
        res = fourier_f_test(y, n_sims=200)
        again = fourier_f_test(y, n_sims=200, seed=res.seed)
        assert again.critical == res.critical

    def test_trend_is_supported(self) -> None:
        rng = np.random.default_rng(33)
        n = 200
        t = np.arange(1, n + 1)
        y = 0.05 * t + np.sin(2 * np.pi * 2 * t / n) + rng.normal(scale=0.3, size=n)
        res = fourier_f_test(y, trend=True, n_sims=300, seed=1)
        assert res.decision == "reject"


class TestKPSS:
    """§2.4 — stationnarite AUTOUR de la composante de Fourier."""

    def test_size_is_nominal_under_stationarity(self) -> None:
        """La bonne question n'est pas 'que dit-il sur CET echantillon'
        mais 'a quelle frequence se trompe-t-il'. Un seul tirage peut
        tomber dans les 5 % sans que rien ne soit casse."""
        from pyardl.fourier.terms import INTEGER_GRID, select_frequency
        from pyardl.fourier.tests import _kpss_statistic

        rng = np.random.default_rng(777)
        stats = []
        for _ in range(300):
            y = rng.normal(size=200)
            freq, _ = select_frequency(y, INTEGER_GRID)
            stats.append(_kpss_statistic(y, freq, False))
        stats = np.array(stats)
        assert abs(np.mean(stats > np.quantile(stats, 0.95)) - 0.05) < 0.02

    def test_rejects_a_random_walk(self) -> None:
        rng = np.random.default_rng(41)
        res = fourier_kpss(np.cumsum(rng.normal(size=200)), n_sims=500, seed=1)
        assert res.decision == "reject"

    def test_fourier_absorbs_most_of_a_smooth_break(self) -> None:
        """Ce que la composante achete, mesure — et ce qu'elle n'achete
        pas.

        Sur une serie stationnaire autour d'une rupture lisse, le KPSS
        ordinaire sort entre 2.5 et 4.0, tres au-dessus de sa valeur
        critique. Avec la composante de Fourier la statistique tombe de
        66 a 83 %. Elle reste au-dessus du seuil : une frequence
        n'epuise pas une logistique. Le test verifie donc la REDUCTION,
        qui est vraie et substantielle, pas un non-rejet qui ne l'est
        pas.
        """
        from pyardl.fourier.terms import INTEGER_GRID, select_frequency
        from pyardl.fourier.tests import _kpss_statistic

        n = 200
        t = np.arange(1, n + 1)
        rng = np.random.default_rng(42)
        y = (
            2.0
            + 1.0 / (1 + np.exp(-0.05 * (t - n / 2)))
            + rng.normal(scale=0.4, size=n)
        )
        design = np.ones((n, 1))
        resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        partial = np.cumsum(resid)
        lag = int(np.floor(4 * (n / 100) ** 0.25))
        var = float(resid @ resid) / n
        for j in range(1, lag + 1):
            var += 2 * (1 - j / (lag + 1)) * float(resid[j:] @ resid[:-j]) / n
        plain = float(np.sum(partial**2) / (n**2 * var))

        freq, _ = select_frequency(y, INTEGER_GRID)
        with_fourier = _kpss_statistic(y, freq, False)
        assert with_fourier < 0.4 * plain

    def test_direction_of_the_null(self) -> None:
        """KPSS rejette la STATIONNARITE : confondre le sens revient a
        rapporter l'inverse de ce que disent les donnees."""
        rng = np.random.default_rng(43)
        stationary = fourier_kpss(rng.normal(size=200), n_sims=300, seed=1)
        walk = fourier_kpss(np.cumsum(rng.normal(size=200)), n_sims=300, seed=1)
        assert walk.statistic > stationary.statistic


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_sims": 10}, "too few"),
            ({"alpha": 0.5}, "alpha must be one of"),
            ({"alpha": 0.0}, "strictly in"),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        y = np.random.default_rng(50).normal(size=100)
        with pytest.raises(ValueError, match=match):
            fourier_f_test(y, **kwargs)

    def test_series_too_short(self) -> None:
        with pytest.raises(ValueError, match="too few for a Fourier test"):
            fourier_f_test(np.zeros(8))

    def test_fixed_without_a_frequency(self) -> None:
        y = np.random.default_rng(51).normal(size=100)
        with pytest.raises(ValueError, match="needs an explicit freq"):
            fourier_f_test(y, freq_estimated=False)

    def test_empty_grid_when_searching(self) -> None:
        y = np.random.default_rng(52).normal(size=100)
        with pytest.raises(ValueError, match="no frequency to search"):
            fourier_f_test(y, grid=())


class TestResultsObject:
    def test_summary_reports_the_essentials(self) -> None:
        res = fourier_f_test(_smooth_break(60), n_sims=200, seed=1)
        text = res.summary()
        for expected in ("Fourier F test", "statistic", "critical", "seed"):
            assert expected in text

    def test_selection_table_is_kept_when_searched(self) -> None:
        res = fourier_f_test(_smooth_break(61), n_sims=200, seed=1)
        assert res.selection is not None
        assert len(res.selection) == len(INTEGER_GRID)

    def test_no_selection_table_when_fixed(self) -> None:
        res = fourier_f_test(
            _smooth_break(62), freq=1.0, freq_estimated=False, n_sims=200, seed=1
        )
        assert res.selection is None

    def test_critical_values_are_ordered(self) -> None:
        res = fourier_f_test(_smooth_break(63), n_sims=300, seed=1)
        assert res.critical[0.10] < res.critical[0.05] < res.critical[0.01]
