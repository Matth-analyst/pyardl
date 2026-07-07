"""Spec 03 §6.2 — réplication externe : R::ARDL::uecm() sur le jeu de
données danois (Johansen & Juselius 1990, via le package R ``ARDL``).

Marqué ``external`` : les valeurs de référence n'existent pas encore
(elles doivent être produites en exécutant
``validation/external/spec03_ardl_uecm.R`` par un humain disposant de R,
puis stockées dans ``tests/replication/expected/spec03.json`` avec leur
provenance et leur tolérance — le projet interdit d'inventer une valeur
de référence).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_EXPECTED_PATH = Path(__file__).parent / "expected" / "spec03.json"


@pytest.mark.external
def test_ardl_uecm_matches_r_ardl_package() -> None:
    if not _EXPECTED_PATH.exists():
        pytest.skip(
            "Valeurs de référence R::ARDL::uecm() non encore produites : "
            "exécuter validation/external/spec03_ardl_uecm.R et enregistrer "
            "le résultat dans tests/replication/expected/spec03.json."
        )
    expected = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))
    raise NotImplementedError(
        "Comparaison à implémenter une fois les valeurs de référence "
        f"disponibles : {expected!r}"
    )
