"""Spec 33 §5 — plan de tests Enders-Siklos.

Comme pour les specs 29/31, la calibration de taille précise à grand
nombre de réplications vit dans un futur ``validation/spec33_*.py``
(non écrit ici, cf. rapport de session). Ici : contrat d'API, non-
régression sur le bug d'alignement look-ahead corrigé (indicator devait
lire u_{t-1}/Delta u_{t-1}, pas u_t/Delta u_t), et les propriétés
qualitatives du plan de tests de la spec.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.cointegration import enders_siklos


def _asymmetric_dgp(n: int, seed: int, rho1: float = -0.6, rho2: float = -0.1):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    u = np.zeros(n)
    for t in range(1, n):
        rho = rho1 if u[t - 1] >= 0 else rho2
        u[t] = (1 + rho) * u[t - 1] + 0.3 * rng.standard_normal()
    y = 1.5 * x + u
    return y, x


def _symmetric_cointegrated_dgp(n: int, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = 1.5 * x + rng.standard_normal(n) * 0.5
    return y, x


class TestAPIAndValidation:
    """Spec 33 §3 — contrat d'API et cas limites."""

    def test_invalid_variant_raises(self) -> None:
        y, x = _symmetric_cointegrated_dgp(150, 0)
        with pytest.raises(ValueError, match="variant"):
            enders_siklos(y, x, variant="bogus")  # type: ignore[arg-type]

    def test_table_cv_source_not_implemented(self) -> None:
        y, x = _symmetric_cointegrated_dgp(150, 0)
        with pytest.raises(NotImplementedError, match="cv_source"):
            enders_siklos(y, x, cv_source="table")  # type: ignore[arg-type]

    def test_degenerate_threshold_raises(self) -> None:
        y, x = _symmetric_cointegrated_dgp(150, 0)
        with pytest.raises(ValueError, match="empty regime"):
            enders_siklos(y, x, threshold=1e9, n_boot=5)

    def test_both_variants_run(self) -> None:
        y, x = _asymmetric_dgp(200, 0)
        for variant in ("tar", "mtar"):
            res = enders_siklos(y, x, variant=variant, n_boot=49, seed=0)  # type: ignore[arg-type]
            assert res.variant == variant
            assert set(res.phi_critical_values) == {0.01, 0.05, 0.10}
            assert "Enders-Siklos" in res.summary()

    def test_estimated_threshold_skips_symmetry_test(self) -> None:
        y, x = _asymmetric_dgp(200, 0)
        res = enders_siklos(y, x, threshold="estimated", n_boot=19, seed=0)
        assert res.threshold_estimated is True
        assert res.symmetry_stat is None
        assert res.symmetry_pvalue is None

    def test_fixed_threshold_computes_symmetry_test(self) -> None:
        y, x = _asymmetric_dgp(200, 0)
        res = enders_siklos(y, x, threshold=0.0, n_boot=19, seed=0)
        assert res.threshold_estimated is False
        assert res.symmetry_stat is not None
        assert res.symmetry_pvalue is not None

    def test_decision_missing_alpha_raises(self) -> None:
        y, x = _symmetric_cointegrated_dgp(150, 0)
        res = enders_siklos(y, x, n_boot=9, seed=0)
        with pytest.raises(ValueError, match="alpha"):
            res.decision(0.2)

    def test_reproducible_with_seed(self) -> None:
        y, x = _asymmetric_dgp(200, 1)
        r1 = enders_siklos(y, x, n_boot=29, seed=42)
        r2 = enders_siklos(y, x, n_boot=29, seed=42)
        assert r1.phi_stat == r2.phi_stat
        assert r1.phi_critical_values == r2.phi_critical_values

    def test_ecm_asymmetric_shape(self) -> None:
        y, x = _asymmetric_dgp(200, 0)
        res = enders_siklos(y, x, n_boot=9, seed=0)
        ecm = res.ecm_asymmetric(y, x)
        assert list(ecm.index[:2]) == ["ecm.regime1", "ecm.regime2"]
        assert {"coef", "se", "t", "pvalue"} <= set(ecm.columns)


class TestLookAheadRegression:
    """Non-régression sur le bug d'alignement corrigé (indicator[lags:-1],
    pas indicator[lags+1:], qui lisait u_t/Delta u_t au lieu de
    u_{t-1}/Delta u_{t-1} — un biais look-ahead exact pour M-TAR).
    """

    def test_mtar_and_tar_critical_values_same_order_of_magnitude(self) -> None:
        """Avant le correctif, les CV bootstrap de M-TAR explosaient
        (~100+) alors que celles de TAR restaient raisonnables (~3-10)
        sur le meme DGP : les deux variantes doivent desormais donner
        des CV du meme ordre de grandeur sous le meme null bootstrap.
        """
        y, x = _asymmetric_dgp(200, 0)
        res_tar = enders_siklos(y, x, variant="tar", n_boot=199, seed=0)
        res_mtar = enders_siklos(y, x, variant="mtar", n_boot=199, seed=0)
        cv_tar = res_tar.phi_critical_values[0.05]
        cv_mtar = res_mtar.phi_critical_values[0.05]
        assert cv_tar < 30
        assert cv_mtar < 30

    def test_phi_stat_not_implausibly_large(self) -> None:
        """Sous un DGP cointégré ordinaire, Phi ne doit pas exploser à des
        centaines (signature du bug look-ahead corrigé)."""
        y, x = _symmetric_cointegrated_dgp(200, 2)
        for variant in ("tar", "mtar"):
            res = enders_siklos(y, x, variant=variant, n_boot=29, seed=0)  # type: ignore[arg-type]
            assert res.phi_stat < 100


class TestPowerAndSymmetry:
    """Spec 33 §5.1/§5.2 — proprietes qualitatives (pas de calibration fine)."""

    def test_asymmetric_dgp_rejects_symmetry(self) -> None:
        y, x = _asymmetric_dgp(300, 1, rho1=-0.8, rho2=-0.05)
        res = enders_siklos(y, x, variant="tar", n_boot=99, seed=1)
        assert res.decision(0.05) == "cointegration"
        assert res.symmetry_pvalue is not None
        assert res.symmetry_pvalue < 0.05

    def test_symmetric_dgp_does_not_reject_symmetry(self) -> None:
        rng = np.random.default_rng(4)
        n = 300
        x = np.cumsum(rng.standard_normal(n))
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = -0.4 * u[t - 1] + 0.3 * rng.standard_normal()
        y = 1.5 * x + u
        res = enders_siklos(y, x, variant="tar", n_boot=99, seed=4)
        assert res.decision(0.05) == "cointegration"
        assert res.symmetry_pvalue is not None
        assert res.symmetry_pvalue > 0.05
