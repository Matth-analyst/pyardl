"""Spec 16 §3 — modèle inconditionnel, validé contre bootCT.

Valeurs de référence produites par R (2026-08-20, R 4.6.1, bootCT
2.1.0) via ``validation/external/spec16_bootct_conditional.R``.
Tolérance contractuelle : 1e-6 sur les statistiques observées.

Ce test est le VERROU de la convention inconditionnelle. bootCT
rapporte lui-même les deux versions de F_indep ; la seconde est
reproduite ici. Deux candidats avaient été calculés — le design
conditionnel privé des Δx contemporains, et l'équation VECM — et un
seul correspond. La convention n'a donc pas été choisie par lecture du
texte mais par appariement numérique.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from pyardl.bounds import bounds_test
from pyardl.datasets import load_denmark

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec16.json").read_text(encoding="utf-8")
)
_TOL = _EXPECTED["tolerance"]["stat"]
_ORDER = (3, {"LRY": 1, "IBO": 3, "IDE": 2})


def _run(conditional: bool):  # type: ignore[no-untyped-def]
    data = load_denmark()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bounds_test(
            data["LRM"],
            data[["LRY", "IBO", "IDE"]],
            case=_EXPECTED["case"],
            order=_ORDER,
            conditional=conditional,
        )


@pytest.mark.external
def test_conditional_f_indep_matches_bootct() -> None:
    assert _run(True).f_indep_stat == pytest.approx(
        _EXPECTED["f_indep_conditional"], abs=_TOL
    )


@pytest.mark.external
def test_unconditional_f_indep_matches_bootct() -> None:
    """Le verrou : la convention inconditionnelle est celle de bootCT."""
    assert _run(False).f_indep_stat == pytest.approx(
        _EXPECTED["f_indep_unconditional"], abs=_TOL
    )


@pytest.mark.external
def test_the_two_models_are_not_the_same_model() -> None:
    """Contrôle de sanité : si les deux coïncidaient, le test
    précédent passerait pour la mauvaise raison."""
    assert _run(True).f_indep_stat != pytest.approx(_run(False).f_indep_stat, abs=1e-3)
