"""Spec 05 §6.5 — réplication externe : R::ARDL::ardl() et auto_ardl()
sur les données danoises.

Marqué ``external`` : valeurs de référence à produire en exécutant
``validation/external/spec05_r_ardl.R`` (humain + R requis), à stocker
dans ``tests/replication/expected/spec05.json`` avec provenance et
tolérance (le projet interdit d'inventer une valeur de référence).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_EXPECTED_PATH = Path(__file__).parent / "expected" / "spec05.json"


@pytest.mark.external
def test_ardl_and_auto_ardl_match_r_package() -> None:
    if not _EXPECTED_PATH.exists():
        pytest.skip(
            "Valeurs de référence R::ARDL non encore produites : exécuter "
            "validation/external/spec05_r_ardl.R et enregistrer le résultat "
            "dans tests/replication/expected/spec05.json."
        )
    expected = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))
    raise NotImplementedError(
        "Comparaison à implémenter une fois les valeurs de référence "
        f"disponibles (et le dataset danois intégré, spec 04) : {expected!r}"
    )
