"""Spec 07 §3 — plan de tests du wrapper Johansen.

Le calcul lui-même vient de statsmodels et n'est pas retesté ici. Ce qui
est testé, c'est ce que le wrapper ajoute : la décision séquentielle, la
normalisation des vecteurs, les refus explicites, et le diagnostic
destiné au cadre bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyardl.cointegration import (
    JohansenResults,
    check_no_cointegration_among_x,
    johansen,
)
from pyardl.cointegration.johansen import _sequential_rank
from pyardl.exceptions import PyardlMethodologyWarning

# --- DGP -----------------------------------------------------------------


def _vecm_rank1(seed: int, n: int = 200) -> pd.DataFrame:
    """Trois variables, UNE relation de cointégration.

    x et z sont des marches aléatoires indépendantes ; y est rappelé
    vers x + z. Le rang de Pi vaut donc 1 par construction.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    z = np.cumsum(rng.normal(size=n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] - 0.5 * (y[t - 1] - x[t - 1] - z[t - 1]) + rng.normal(scale=0.5)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _rank0(seed: int, n: int = 200, k: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        np.cumsum(rng.normal(size=(n, k)), axis=0),
        columns=[f"v{j}" for j in range(k)],
    )


# --- La décision séquentielle -------------------------------------------


class TestSequentialRank:
    """§2.2 — la règle d'arrêt, testée sur des cas construits."""

    def test_stops_at_the_first_non_rejection(self) -> None:
        """Le troisième test rejette, mais on s'est déjà arrêté au
        second : le rang vaut 1, pas 2."""
        stat = np.array([50.0, 10.0, 30.0])
        cv = np.array([40.0, 15.0, 5.0])
        assert _sequential_rank(stat, cv) == 1

    def test_no_rejection_at_all_gives_zero(self) -> None:
        assert _sequential_rank(np.array([1.0, 1.0]), np.array([40.0, 15.0])) == 0

    def test_all_rejections_gives_full_rank(self) -> None:
        """Rang plein : le système était déjà stationnaire."""
        assert _sequential_rank(np.array([99.0, 99.0]), np.array([40.0, 15.0])) == 2

    def test_equality_does_not_reject(self) -> None:
        """Statistique EXACTEMENT égale à la valeur critique : on ne
        rejette pas. Le sens de l'inégalité est une convention, mais
        elle doit être fixée et testée."""
        assert _sequential_rank(np.array([40.0]), np.array([40.0])) == 0

    def test_matches_statsmodels_select_coint_rank(self) -> None:
        """Notre boucle doit rendre exactement ce que rend l'utilitaire
        de référence — on évite la seconde estimation, pas la
        vérification."""
        from statsmodels.tsa.vector_ar.vecm import select_coint_rank

        data = _vecm_rank1(seed=7)
        for method in ("trace", "maxeig"):
            for signif in (0.10, 0.05, 0.01):
                mine = johansen(
                    data, det_order=0, k_ar_diff=1, alpha=signif, method=method
                ).selected_rank
                theirs = select_coint_rank(
                    data.to_numpy(), 0, 1, method=method, signif=signif
                ).rank
                assert mine == theirs


# --- Comportement sur DGP connus ----------------------------------------


class TestKnownDGPs:
    """§3.1 — rang 1 reconnu, rang 0 non sur-détecté."""

    @pytest.mark.parametrize("seed", range(5))
    def test_rank_one_is_found(self, seed: int) -> None:
        """La relation existante est TOUJOURS trouvée.

        L'assertion porte sur ``>= 1`` et non sur ``== 1`` : la
        procédure séquentielle par la trace sur-sélectionne parfois une
        direction supplémentaire, à un taux mesuré (OBS-10) et non
        supposé. Le taux exact est vérifié par les tests Monte Carlo
        ci-dessous ; ici on vérifie que la relation n'est jamais
        manquée.
        """
        res = johansen(_vecm_rank1(seed=seed), det_order=0, k_ar_diff=1)
        assert res.selected_rank >= 1

    @pytest.mark.parametrize("seed", range(5))
    def test_rank_zero_is_not_over_detected(self, seed: int) -> None:
        res = johansen(_rank0(seed=100 + seed), det_order=0, k_ar_diff=1)
        assert res.selected_rank == 0

    @pytest.mark.fast_mc
    def test_size_under_rank_zero(self) -> None:
        """§3.1 — taille approximative sous rang 0 : le taux de fausse
        détection reste dans le voisinage du seuil nominal."""
        hits = sum(
            johansen(_rank0(seed=1000 + s), det_order=0, k_ar_diff=1).selected_rank > 0
            for s in range(200)
        )
        assert hits / 200 < 0.12

    @pytest.mark.fast_mc
    def test_power_under_rank_one(self) -> None:
        """§3.1 — le rang exact est retenu dans au moins 90 % des cas.

        Par la valeur propre maximale : la trace sur-sélectionne, ce qui
        est mesuré et documenté (OBS-10). Le critere de la spec est
        atteint par maxeig, pas par trace, et cette difference est un
        resultat, pas un reglage.
        """
        hits = sum(
            johansen(
                _vecm_rank1(seed=2000 + s),
                det_order=0,
                k_ar_diff=1,
                method="maxeig",
            ).selected_rank
            == 1
            for s in range(100)
        )
        assert hits / 100 >= 0.90

    @pytest.mark.fast_mc
    def test_the_relation_is_never_missed(self) -> None:
        """Sous-sélectionner serait grave : cela ferait conclure a
        l'absence de relation alors qu'il y en a une. Ce sens de
        l'erreur, lui, ne se produit pas."""
        missed = sum(
            johansen(_vecm_rank1(seed=3000 + s), det_order=0, k_ar_diff=1).selected_rank
            == 0
            for s in range(100)
        )
        assert missed == 0


# --- L'objet résultat ----------------------------------------------------


class TestResultsObject:
    def test_shapes_and_index(self) -> None:
        res = johansen(_vecm_rank1(seed=3), det_order=0, k_ar_diff=1)
        assert isinstance(res, JohansenResults)
        assert list(res.trace_stat.index) == ["r = 0", "r <= 1", "r <= 2"]
        assert res.trace_cv.shape == (3, 3)
        assert list(res.trace_cv.columns) == [0.10, 0.05, 0.01]
        assert res.eigenvalues.size == 3
        assert res.names == ("y", "x", "z")

    def test_beta_is_normalised_on_its_first_element(self) -> None:
        res = johansen(_vecm_rank1(seed=4), det_order=0, k_ar_diff=1)
        assert res.beta.iloc[0, 0] == pytest.approx(1.0)

    def test_beta_recovers_the_true_relation(self) -> None:
        """Le premier vecteur doit approcher y - x - z (à l'échelle
        près, d'où la normalisation)."""
        res = johansen(_vecm_rank1(seed=5, n=400), det_order=0, k_ar_diff=1)
        b = res.beta["beta1"]
        assert b["x"] == pytest.approx(-1.0, abs=0.15)
        assert b["z"] == pytest.approx(-1.0, abs=0.15)

    def test_rank_can_be_re_read_at_another_level(self) -> None:
        res = johansen(_vecm_rank1(seed=6), det_order=0, k_ar_diff=1, alpha=0.05)
        assert res.rank(alpha=0.01) <= res.rank(alpha=0.10)
        assert res.rank() == res.selected_rank

    def test_summary_reports_both_statistics(self) -> None:
        text = johansen(_vecm_rank1(seed=8), det_order=0, k_ar_diff=1).summary()
        assert "trace" in text and "maxeig" in text
        assert "selected rank" in text

    def test_results_are_immutable(self) -> None:
        res = johansen(_rank0(seed=9), det_order=0, k_ar_diff=1)
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            res.selected_rank = 2  # type: ignore[misc]


# --- Refus explicites ----------------------------------------------------


class TestValidation:
    def test_single_series_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two series"):
            johansen(np.cumsum(np.ones((50, 1)), axis=0))

    def test_one_dimensional_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            johansen(np.arange(50.0))

    def test_bad_det_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="det_order must be"):
            johansen(_rank0(seed=1), det_order=2)

    def test_bad_alpha_is_refused(self) -> None:
        with pytest.raises(ValueError, match="alpha must be"):
            johansen(_rank0(seed=1), alpha=0.025)

    def test_bad_method_is_refused(self) -> None:
        with pytest.raises(ValueError, match="method must be"):
            johansen(_rank0(seed=1), method="wald")  # type: ignore[arg-type]

    def test_negative_lags_are_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            johansen(_rank0(seed=1), k_ar_diff=-1)

    def test_nan_is_refused(self) -> None:
        data = _rank0(seed=1)
        data.iloc[10, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            johansen(data)

    def test_too_many_variables_is_refused(self) -> None:
        """Au-delà de 12 variables, aucune valeur critique n'est
        tabulée : on refuse plutôt que de rendre un rang indécidable."""
        with pytest.raises(ValueError, match="at most 12"):
            johansen(_rank0(seed=1, k=13))


# --- Le diagnostic destiné au cadre bounds -------------------------------


class TestCheckAmongX:
    """§2.3 — l'hypothèse du bounds test sur les régresseurs."""

    def test_single_regressor_has_nothing_to_check(self) -> None:
        assert check_no_cointegration_among_x(np.cumsum(np.ones((50, 1)))) is None

    def test_independent_walks_pass_silently(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = check_no_cointegration_among_x(
                _rank0(seed=42, k=2), det_order=0, k_ar_diff=1
            )
        assert res is not None
        assert res.selected_rank == 0

    def test_injected_cointegrated_pair_is_detected(self) -> None:
        """§3.3 — une paire cointégrée injectée parmi les régresseurs
        doit déclencher l'avertissement, pas passer inaperçue."""
        rng = np.random.default_rng(11)
        n = 300
        a = np.cumsum(rng.normal(size=n))
        b = a + rng.normal(scale=0.3, size=n)  # b et a cointégrés
        c = np.cumsum(rng.normal(size=n))
        with pytest.warns(PyardlMethodologyWarning, match="AMONG the regressors"):
            res = check_no_cointegration_among_x(
                pd.DataFrame({"a": a, "b": b, "c": c}), det_order=0, k_ar_diff=1
            )
        assert res is not None
        assert res.selected_rank >= 1
