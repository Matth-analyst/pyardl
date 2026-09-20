"""Spec 29 §5 — plan de tests Gregory-Hansen.

Les tests de calibration de taille (§5.1) exigeant beaucoup de
réplications imbriquées (chaque réplication relance elle-même un
bootstrap complet) sont coûteux : la validation de taille précise, à
grand nombre de réplications, vit dans
``validation/spec29_montecarlo.py`` (Étape 4, CLAUDE.md), dimensionnée
selon la règle n°10. Ici, on ne garde qu'une vérification de taille
grossière, avec une tolérance dérivée de l'erreur type binomiale du
nombre de réplications réellement exécutable en test unitaire.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.cointegration import engle_granger, gregory_hansen


def _cointegrated_with_break(n: int, tau: float, seed: int, slope_break: bool = False):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    phi = (np.arange(n) > int(np.floor(n * tau))).astype(float)
    if slope_break:
        beta = 1.0 + 1.5 * phi
        mu = np.zeros(n)
    else:
        beta = np.ones(n)
        mu = 3.0 * phi
    y = mu + beta * x + 0.5 * rng.standard_normal(n)
    return y, x


def _independent_walks(n: int, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.cumsum(rng.standard_normal(n))
    return y, x


class TestAPIAndValidation:
    """Spec 29 §3 — contrat d'API et cas limites."""

    def test_invalid_model_raises(self) -> None:
        y, x = _cointegrated_with_break(60, 0.5, seed=0)
        with pytest.raises(ValueError, match="model"):
            gregory_hansen(y, x, model="bogus")  # type: ignore[arg-type]

    def test_table_cv_source_not_implemented(self) -> None:
        """Spec 29 §2.2.4a : la table publiée n'est pas encodée (CLAUDE.md règle 9)."""
        y, x = _cointegrated_with_break(60, 0.5, seed=0)
        with pytest.raises(NotImplementedError, match="cv_source"):
            gregory_hansen(y, x, cv_source="table")  # type: ignore[arg-type]

    def test_invalid_trim_raises(self) -> None:
        y, x = _cointegrated_with_break(60, 0.5, seed=0)
        with pytest.raises(ValueError, match="trim"):
            gregory_hansen(y, x, trim=0.6)

    def test_four_models_all_run(self) -> None:
        y, x = _cointegrated_with_break(80, 0.5, seed=1)
        for model in ("C", "C/T", "C/S", "C/S/T"):
            res = gregory_hansen(
                y,
                x,
                model=model,
                n_boot=20,
                max_lags=2,
                seed=1,  # type: ignore[arg-type]
            )
            assert res.model == model
            assert 0.0 < res.break_fraction < 1.0
            assert set(res.critical_values) == {0.01, 0.05, 0.10}
            assert len(res.grid) > 0
            assert "Gregory-Hansen" in res.summary()

    def test_decision_missing_alpha_raises(self) -> None:
        y, x = _cointegrated_with_break(60, 0.5, seed=0)
        res = gregory_hansen(y, x, n_boot=10, max_lags=2, seed=0)
        with pytest.raises(ValueError, match="alpha"):
            res.decision(0.2)

    def test_needs_at_least_one_regressor(self) -> None:
        y = np.cumsum(np.random.default_rng(0).standard_normal(60))
        with pytest.raises(ValueError):
            gregory_hansen(y, x=None)  # type: ignore[arg-type]

    def test_reproducible_with_seed(self) -> None:
        y, x = _cointegrated_with_break(70, 0.5, seed=2)
        r1 = gregory_hansen(y, x, n_boot=30, max_lags=2, seed=42)
        r2 = gregory_hansen(y, x, n_boot=30, max_lags=2, seed=42)
        assert r1.statistic == r2.statistic
        assert r1.critical_values == r2.critical_values


class TestBreakRecovery:
    """Spec 29 §5.1 — le point de rupture est retrouvé."""

    def test_break_fraction_recovered_case_c(self) -> None:
        y, x = _cointegrated_with_break(300, 0.5, seed=3, slope_break=False)
        res = gregory_hansen(y, x, model="C", n_boot=5, max_lags=3, trim=0.15, seed=3)
        assert abs(res.break_fraction - 0.5) < 0.05

    def test_slope_break_needs_cs_specification(self) -> None:
        """Spec 29 §5.3 : seules C/S et C/S/T doivent détecter une rupture de pente.

        Sur un DGP où seule la pente change (pas le niveau), le résidu de
        la spécification C (pas de terme d'interaction) doit rester
        moins stationnaire — statistique ADF* moins négative — que celui
        de C/S, qui a le bon régresseur pour absorber la rupture.
        """
        y, x = _cointegrated_with_break(300, 0.5, seed=4, slope_break=True)
        res_c = gregory_hansen(y, x, model="C", n_boot=5, max_lags=3, seed=4)
        res_cs = gregory_hansen(y, x, model="C/S", n_boot=5, max_lags=3, seed=4)
        assert res_cs.statistic < res_c.statistic


class TestAgreementWithEngleGranger:
    """Spec 29 §5.2 : sur un DGP cointégré sans rupture, les deux tests rejettent H0."""

    def test_no_break_dgp_both_reject(self) -> None:
        rng = np.random.default_rng(5)
        n = 200
        x = np.cumsum(rng.standard_normal(n))
        y = 1.5 * x + rng.standard_normal(n)
        eg = engle_granger(y, x)
        gh = gregory_hansen(y, x, model="C", n_boot=30, max_lags=3, seed=5)
        assert eg.decision(0.05) == "cointegration"
        assert gh.decision(0.05) == "cointegration"


@pytest.mark.slow
class TestSizeUnderNull:
    """Spec 29 §5.1 — taille sous un DGP sans rupture ni cointégration.

    Chaque réplication relance un test complet (recherche sur grille +
    bootstrap), donc le nombre de réplications exécutable ici reste
    modeste (n_reps=150, n_boot=49) : erreur type binomiale attendue sous
    H0 vraie (taux nominal 5%) = sqrt(0.05*0.95/150) ~= 1.8 point de
    pourcentage. La bande de tolérance (0%-12%, soit ~+-4 erreurs types)
    ne prétend donc PAS calibrer précisément la taille — seulement
    détecter une dérive grossière. La calibration précise (règle
    CLAUDE.md n°10, réplications dimensionnées sur l'écart à détecter)
    est faite séparément dans validation/spec29_montecarlo.py.
    """

    def test_empirical_size_roughly_nominal(self) -> None:
        n_reps = 150
        n = 50
        rejections = 0
        for i in range(n_reps):
            y, x = _independent_walks(n, seed=1000 + i)
            res = gregory_hansen(
                y, x, model="C", n_boot=49, max_lags=2, trim=0.15, seed=2000 + i
            )
            if res.decision(0.05) == "cointegration":
                rejections += 1
        rate = rejections / n_reps
        assert 0.0 <= rate <= 0.12
