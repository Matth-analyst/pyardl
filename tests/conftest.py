"""Fixtures partagées : seed explicite, jamais l'état global numpy."""

from __future__ import annotations

import numpy as np
import pytest


def pytest_configure() -> None:
    """Backend matplotlib non interactif, pour toute la suite.

    Un test qui construit une figure ne doit jamais dependre du backend
    que matplotlib choisit dans l'environnement ambiant. Sous Linux sans
    affichage il retombe sur Agg ; sous macOS il peut retenir `macosx`,
    qui suppose un contexte graphique absent d'un runner CI.

    Deux fichiers forcaient deja Agg localement, les autres non : c'est
    ce genre d'incoherence qui rend une suite verte sur une plateforme et
    rouge sur une autre. Il est fixe ici une fois, avant toute
    collecte.
    """
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - matplotlib est optionnel
        return
    matplotlib.use("Agg", force=True)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260707)
