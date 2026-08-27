"""Spec 18 §4.4 — replication R `quantreg` (Koenker).

Valeurs de reference produites par R (2026-08-23, R 4.6.1, quantreg 6.1)
via ``validation/external/spec18_quantreg.R``, sur le design UECM de
pyardl applique aux donnees danoises.

CE QUI EST VALIDE, ET POURQUOI CELA
-----------------------------------
Aucun package R n'implemente le QARDL. Ce qui peut etre valide en
externe — et c'est le coeur numerique de la spec — est la regression
quantile elle-meme. ``quantreg::rq`` la resout par le simplexe de
Barrodale-Roberts, donc EXACTEMENT : c'est la reference de ce calcul
depuis Koenker & Bassett (1978).

DEUX TOLERANCES, ET LA RAISON COMPTE
------------------------------------
Le programme lineaire et le simplexe atteignent tous deux l'optimum
exact : ils doivent coincider a la precision machine, et c'est un
controle severe — deux solveurs independants, deux algorithmes
differents.

L'estimateur iteratif, lui, ne doit egaler que la PERTE. Sur un design a
13 parametres et 52 observations, l'argmin de la perte quantile n'est
pas unique : plusieurs vecteurs l'atteignent. Exiger l'egalite des
coefficients exigerait donc quelque chose qui n'est pas vrai, et le test
tomberait sur une propriete que la theorie ne garantit pas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyardl.qardl.estimate import (
    check_loss,
    quantile_regression,
    quantile_regression_lp,
)

_HERE = Path(__file__).parent
_EXPECTED = json.loads((_HERE / "expected" / "spec18.json").read_text(encoding="utf-8"))
# Versionne AVEC le test, pas dans validation/ : ce dossier est
# gitignore, et le fichier absent faisait planter la CI sur un
# FileNotFoundError. Le design est celui de pyardl applique aux donnees
# danoises que la bibliotheque embarque deja : rien de tiers ici.
_DATA = _HERE / "data" / "spec18_design.csv"


def _design() -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(_DATA)
    names = [c for c in frame.columns if c != "dep"]
    return (
        frame["dep"].to_numpy(dtype=float),
        frame[names].to_numpy(dtype=float),
        names,
    )


def _reference(tau: float, names: list[str]) -> np.ndarray:
    table = _EXPECTED["coefficients"][str(tau)]
    return np.array([table[n] for n in names], dtype=float)


@pytest.mark.external
@pytest.mark.parametrize("tau", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_linear_program_matches_quantreg_to_machine_precision(tau: float) -> None:
    """Deux solveurs exacts, deux algorithmes, une seule reponse."""
    y, x, names = _design()
    ours = quantile_regression_lp(y, x, tau)
    assert ours == pytest.approx(
        _reference(tau, names), abs=_EXPECTED["tolerance"]["lp_vs_rq"]
    )


@pytest.mark.external
@pytest.mark.parametrize("tau", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_iterative_estimate_attains_the_same_loss(tau: float) -> None:
    """L'estimateur employe atteint la perte de la reference.

    C'est la propriete qui a un sens : l'argmin n'est pas unique, la
    valeur optimale l'est.
    """
    y, x, names = _design()
    ours, _ = quantile_regression(y, x, tau)
    excess = check_loss(y, x, ours, tau) - check_loss(y, x, _reference(tau, names), tau)
    assert abs(excess) < _EXPECTED["tolerance"]["loss_vs_rq"]


@pytest.mark.external
def test_the_default_tolerance_would_have_failed_this() -> None:
    """Le controle qui justifie le reglage retenu.

    Avec la tolerance par defaut de statsmodels, l'estimateur s'ecarte
    de l'optimum bien plus que la tolerance contractuelle de ce test.
    C'est la mesure qui a impose p_tol = 1e-10, et elle est verrouillee
    ici pour qu'un retour en arriere se voie.
    """
    y, x, names = _design()
    exact = _reference(0.5, names)
    loose, _ = quantile_regression(y, x, 0.5, p_tol=1e-6, max_iter=1000, cap_tol=1e9)
    tight, _ = quantile_regression(y, x, 0.5)
    excess_loose = check_loss(y, x, loose, 0.5) - check_loss(y, x, exact, 0.5)
    excess_tight = check_loss(y, x, tight, 0.5) - check_loss(y, x, exact, 0.5)
    assert excess_tight < excess_loose
