"""Spec 13 §3.1 (voie A2) — surfaces de réponse K&S finies-T.

STATUT (2026-07-19) : validation empirique contre la sortie Stata
publiée EN ATTENTE D'AUTORISATION (voie A3,
docs/correspondence/2026-07-10_ks_license_draft.md). Le code
(``ks2020_finite.py``) est écrit et sa forme fonctionnelle est
documentée, mais :

- aucune comparaison à un jeu de valeurs publiées n'est encodée ici —
  les valeurs précédemment utilisées provenaient d'une copie non
  vérifiée (miroir tiers) et ont été retirées (voir CHANGELOG,
  DEVIATIONS.md) ;
- ``download_surface_coefs()`` NE DOIT PAS être exécuté tant que la
  réponse des auteurs n'est pas reçue — ne pas ré-introduire de test
  qui en dépend avant cette autorisation ;
- seuls des tests de cohérence INTERNE (sans dépendance à un fichier
  externe) sont exécutés ici.

Ne pas ré-exécuter de validation 1e-3 contre une sortie Stata avant
l'accord des auteurs.
"""

from __future__ import annotations

import pytest

from ardlpy.critical_values.ks2020_finite import (
    _coefs_path,
    crit_value_bounds_finite,
    pvalue_bounds_finite,
)

# Tous les tests de ce module qui évalueraient les coefficients K&S
# nécessitent le fichier en cache local ; celui-ci n'est plus téléchargé
# (bloqué par A3). Les tests ci-dessous sont donc soit purement
# internes (aucune dépendance), soit marqués needs_review + skip.
_HAS_CACHE = _coefs_path().exists()


class TestInternalCoherence:
    """Aucune dépendance à un fichier externe : la surface EST le
    matériel testé, comparée à elle-même (limites, monotonies)."""

    @pytest.mark.skipif(not _HAS_CACHE, reason="bloqué par A3 (voir en-tête du module)")
    def test_asymptotic_limit_matches_a1(self) -> None:
        """T très grand -> CV de la voie A1 (statsmodels) à ±0.05
        (re-simulations indépendantes)."""
        from ardlpy.critical_values.ks2020 import crit_value_bounds

        for case in (1, 3, 5):
            for k in (1, 3):
                fin = crit_value_bounds_finite(
                    case=case, k=k, t_obs=10_000_000, sr=0, alpha=0.05
                )
                a1 = crit_value_bounds(case=case, k=k, alpha=0.05)
                assert fin[0] == pytest.approx(a1[0], abs=0.05)
                assert fin[1] == pytest.approx(a1[1], abs=0.05)

    @pytest.mark.skipif(not _HAS_CACHE, reason="bloqué par A3 (voir en-tête du module)")
    def test_t_asymptotic_limit_matches_pss(self) -> None:
        got = crit_value_bounds_finite(
            case=3, k=1, t_obs=10_000_000, sr=0, alpha=0.05, stat="t"
        )
        assert got[0] == pytest.approx(-2.86, abs=0.02)
        assert got[1] == pytest.approx(-3.22, abs=0.02)

    @pytest.mark.skipif(not _HAS_CACHE, reason="bloqué par A3 (voir en-tête du module)")
    def test_cv_decrease_toward_asymptotic_in_t_obs(self) -> None:
        """Bornes plus conservatrices en petit échantillon, décroissantes
        vers l'asymptotique (motivation de Narayan/K&S)."""
        values = [
            crit_value_bounds_finite(case=3, k=2, t_obs=t, sr=3, alpha=0.05)[1]
            for t in (30, 50, 80, 200, 1000)
        ]
        assert all(a >= b - 1e-9 for a, b in zip(values[:-1], values[1:], strict=True))

    @pytest.mark.skipif(not _HAS_CACHE, reason="bloqué par A3 (voir en-tête du module)")
    def test_sr_increases_cv_in_small_samples(self) -> None:
        """Plus de coefficients de court terme -> bornes plus élevées à
        T petit (consommation de degrés de liberté)."""
        low = crit_value_bounds_finite(case=3, k=2, t_obs=40, sr=0, alpha=0.05)[1]
        high = crit_value_bounds_finite(case=3, k=2, t_obs=40, sr=10, alpha=0.05)[1]
        assert high > low

    @pytest.mark.skipif(not _HAS_CACHE, reason="bloqué par A3 (voir en-tête du module)")
    def test_pvalue_roundtrip_at_cv(self) -> None:
        for stat in ("F", "t"):
            cv = crit_value_bounds_finite(3, 2, 90, 4, 0.05, stat=stat)  # type: ignore[arg-type]
            p = pvalue_bounds_finite(cv[0], 3, 2, 90, 4, df_resid=80, stat=stat)  # type: ignore[arg-type]
            assert p[0] == pytest.approx(0.05, abs=2e-3)


class TestCoverageAndErrors:
    """Validation des entrées : aucune dépendance au fichier externe."""

    def test_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="case"):
            crit_value_bounds_finite(0, 1, 100, 2, 0.05)
        with pytest.raises(ValueError, match="sr"):
            crit_value_bounds_finite(3, 1, 100, -1, 0.05)
        with pytest.raises(ValueError, match="stat"):
            crit_value_bounds_finite(3, 1, 100, 2, 0.05, stat="W")  # type: ignore[arg-type]


def test_missing_cache_raises_with_instructions(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Sans cache : erreur explicite avec la marche à suivre (pas de
    téléchargement silencieux). Ce test ne dépend PAS du cache réel et
    ne télécharge rien — il vérifie seulement le message d'erreur."""
    import ardlpy.critical_values.ks2020_finite as mod

    monkeypatch.setenv("ARDLPY_CACHE", str(tmp_path))
    monkeypatch.setattr(mod, "_TABLES", None)
    with pytest.raises(FileNotFoundError, match="download_surface_coefs"):
        mod._load_tables()


@pytest.mark.needs_review
@pytest.mark.external
class TestBoundsTestIntegrationBlockedByA3:
    """Reproduction bout-en-bout contre une sortie Stata publiée.

    BLOQUÉ PAR A3 : ne pas ré-exécuter avant accord des auteurs
    (voie A3, docs/correspondence/2026-07-10_ks_license_draft.md).
    Aucune valeur de référence n'est encodée ici tant que l'accord
    n'est pas reçu — ce test est un espace réservé documentant
    l'intention, pas une validation active.
    """

    @pytest.mark.skip(
        reason=(
            "bloqué par A3 : nécessite (a) l'autorisation des auteurs pour "
            "download_surface_coefs(), (b) une source légitime de valeurs "
            "de référence (article Stata Journal 2023 en accès direct). "
            "Ne pas réactiver avant réception de la réponse A3."
        )
    )
    def test_full_reproduction_placeholder(self) -> None:
        raise NotImplementedError("En attente d'autorisation A3.")
