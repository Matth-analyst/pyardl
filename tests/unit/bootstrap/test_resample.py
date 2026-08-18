"""Spec 14 §2.3 — verrou du moteur de rééchantillonnage.

Deux exigences, écrites AVANT l'implémentation du bootstrap complet :

1. Reproductibilité stricte à seed fixée — deux exécutions identiques
   doivent produire des tirages identiques bit à bit. Sans cela, aucune
   valeur critique bootstrap n'est vérifiable par un tiers.
2. Concordance avec une implémentation naïve — la version vectorisée
   doit tirer exactement ce qu'une boucle évidente tirerait.

Le troisième point, moins évident et plus dangereux : les résidus sont
rééchantillonnés PAR LIGNE. La corrélation contemporaine entre
l'équation de y et le bloc marginal des x fait partie du DGP ; la
détruire fausserait la taille du test dans le sens optimiste, sans
qu'aucun test de forme ne s'en aperçoive.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.bootstrap.resample import draw_indices, resample_residuals


def _residuals(n: int = 40, m: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, m))


class TestReproducibility:
    """§2.3 — même seed, mêmes tirages, bit à bit."""

    @pytest.mark.parametrize("scheme", ["iid", "wild"])
    def test_same_seed_same_draws(self, scheme: str) -> None:
        res = _residuals()
        a = resample_residuals(res, 100, np.random.default_rng(42), scheme)  # type: ignore[arg-type]
        b = resample_residuals(res, 100, np.random.default_rng(42), scheme)  # type: ignore[arg-type]
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("scheme", ["iid", "wild"])
    def test_different_seed_different_draws(self, scheme: str) -> None:
        res = _residuals()
        a = resample_residuals(res, 100, np.random.default_rng(1), scheme)  # type: ignore[arg-type]
        b = resample_residuals(res, 100, np.random.default_rng(2), scheme)  # type: ignore[arg-type]
        assert not np.array_equal(a, b)

    def test_generator_state_advances(self) -> None:
        """Deux tirages successifs sur le MÊME générateur diffèrent.

        Sinon une boucle bootstrap produirait B fois le même échantillon
        tout en paraissant fonctionner.
        """
        res = _residuals()
        rng = np.random.default_rng(7)
        first = resample_residuals(res, 50, rng)
        second = resample_residuals(res, 50, rng)
        assert not np.array_equal(first, second)

    def test_no_global_state_used(self) -> None:
        """Le module ne touche jamais à l'état global de numpy.

        On perturbe l'état global entre deux appels à seed identique : le
        résultat ne doit pas bouger.
        """
        res = _residuals()
        # L'appel à l'API héritée est VOLONTAIRE : c'est l'objet même du
        # test. On perturbe l'état global pour prouver qu'il n'a aucune
        # influence sur le résultat.
        np.random.seed(123)  # noqa: NPY002
        a = resample_residuals(res, 60, np.random.default_rng(9))
        np.random.seed(456)  # noqa: NPY002
        _ = np.random.random(1000)  # noqa: NPY002
        b = resample_residuals(res, 60, np.random.default_rng(9))
        assert np.array_equal(a, b)


class TestNaiveEquivalence:
    """§2.3 — la version vectorisée == la boucle évidente."""

    def test_iid_matches_naive_loop(self) -> None:
        """Référence naïve : un index tiré à la fois, ligne copiée."""
        res = _residuals(n=25, m=4, seed=3)
        n_draw = 200

        vectorised = resample_residuals(res, n_draw, np.random.default_rng(11))

        rng = np.random.default_rng(11)
        idx = rng.integers(0, res.shape[0], size=n_draw, dtype=np.intp)
        naive = np.empty((n_draw, res.shape[1]))
        for b in range(n_draw):
            naive[b, :] = res[idx[b], :]

        assert np.array_equal(vectorised, naive)

    def test_wild_matches_naive_loop(self) -> None:
        """Idem sous wild, signe compris — un signe PAR LIGNE."""
        res = _residuals(n=25, m=4, seed=4)
        n_draw = 200

        vectorised = resample_residuals(res, n_draw, np.random.default_rng(13), "wild")

        rng = np.random.default_rng(13)
        idx = rng.integers(0, res.shape[0], size=n_draw, dtype=np.intp)
        signs = rng.integers(0, 2, size=n_draw).astype(np.float64) * 2.0 - 1.0
        naive = np.empty((n_draw, res.shape[1]))
        for b in range(n_draw):
            naive[b, :] = res[idx[b], :] * signs[b]

        assert np.array_equal(vectorised, naive)

    def test_draw_indices_in_range(self) -> None:
        idx = draw_indices(17, 5000, np.random.default_rng(5))
        assert idx.min() >= 0
        assert idx.max() < 17
        # Toutes les lignes doivent être atteignables.
        assert len(np.unique(idx)) == 17


class TestRowIntegrity:
    """Le point critique : la ligne est l'unité de tirage."""

    def test_rows_are_kept_intact_iid(self) -> None:
        """Chaque ligne tirée est une ligne d'origine, non recomposée."""
        res = _residuals(n=30, m=3, seed=6)
        drawn = resample_residuals(res, 500, np.random.default_rng(21))
        for row in drawn:
            assert np.any(np.all(np.isclose(res, row), axis=1))

    def test_rows_are_kept_intact_wild(self) -> None:
        """Sous wild, chaque ligne est une ligne d'origine ou son opposée."""
        res = _residuals(n=30, m=3, seed=7)
        drawn = resample_residuals(res, 500, np.random.default_rng(22), "wild")
        for row in drawn:
            direct = np.any(np.all(np.isclose(res, row), axis=1))
            flipped = np.any(np.all(np.isclose(-res, row), axis=1))
            assert direct or flipped

    def test_contemporaneous_covariance_preserved(self) -> None:
        """La covariance entre équations survit au rééchantillonnage.

        C'est LA raison du tirage par ligne. Sur des résidus fortement
        corrélés, un tirage indépendant par colonne ramènerait la
        corrélation vers zéro ; le tirage par ligne la conserve.
        """
        rng = np.random.default_rng(31)
        n = 400
        common = rng.standard_normal(n)
        res = np.column_stack([common + 0.1 * rng.standard_normal(n), common * 2.0])
        target = float(np.corrcoef(res.T)[0, 1])

        drawn = resample_residuals(res, 4000, np.random.default_rng(32))
        kept = float(np.corrcoef(drawn.T)[0, 1])
        assert abs(kept - target) < 0.05

        # Contre-exemple explicite : tirer chaque colonne séparément
        # détruit la corrélation. Le test échouerait si l'implémentation
        # basculait un jour sur ce schéma.
        rng2 = np.random.default_rng(33)
        broken = np.column_stack(
            [
                res[rng2.integers(0, n, 4000), 0],
                res[rng2.integers(0, n, 4000), 1],
            ]
        )
        assert abs(float(np.corrcoef(broken.T)[0, 1])) < 0.1

    def test_wild_preserves_scale_per_row(self) -> None:
        """Wild ne change que le signe : la norme de chaque ligne tirée
        est celle d'une ligne d'origine."""
        res = _residuals(n=20, m=3, seed=8)
        norms = np.sort(np.linalg.norm(res, axis=1))
        drawn = resample_residuals(res, 300, np.random.default_rng(41), "wild")
        for row in drawn:
            assert np.isclose(np.linalg.norm(row), norms, atol=1e-12).any()

    def test_wild_sign_is_row_wise_not_elementwise(self) -> None:
        """Un signe par ligne, pas par élément.

        Sur des résidus dont les colonnes sont proportionnelles, un signe
        par élément casserait la proportionnalité une fois sur deux.
        """
        base = np.arange(1.0, 21.0)
        res = np.column_stack([base, -3.0 * base])
        drawn = resample_residuals(res, 400, np.random.default_rng(51), "wild")
        assert np.allclose(drawn[:, 1], -3.0 * drawn[:, 0])


class TestValidation:
    def test_one_dimensional_residuals_accepted(self) -> None:
        out = resample_residuals(np.arange(10.0), 25, np.random.default_rng(1))
        assert out.shape == (25, 1)

    def test_three_dimensional_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-D or 2-D"):
            resample_residuals(np.zeros((2, 2, 2)), 5, np.random.default_rng(1))

    def test_unknown_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            resample_residuals(_residuals(), 10, np.random.default_rng(1), "block")  # type: ignore[arg-type]

    def test_empty_residuals_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            draw_indices(0, 5, np.random.default_rng(1))
