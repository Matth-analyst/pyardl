"""Spec 10 §6 — JALON DE PHASE 1 : réplication PSS 2001 (salaires UK).

Marqué ``external`` tant que les valeurs de référence n'ont pas été
produites par ``validation/external/spec10_pss2001_replication.R``
(R + package ARDL requis, exécution humaine). Les valeurs attendues ne
sont JAMAIS estimées par nous-mêmes : elles proviennent du script R
(réplication Natsiopoulos & Tzeremes 2022) et sont stockées dans
``tests/replication/expected/spec10_pss2001.json`` avec provenance et
tolérance (F et t identiques à 1e-4 — spec 10 §6).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_EXPECTED_PATH = Path(__file__).parent / "expected" / "spec10_pss2001.json"
_DATA_PATH = Path(__file__).parents[2] / "src" / "ardlpy" / "datasets"


@pytest.mark.external
def test_pss2001_uk_wages_replication() -> None:
    """F (cas IV et V) et t (cas V) identiques à 1e-4 au package R ARDL
    sur l'équation de salaires UK, ARDL(6, 0, 5, 4, 5)."""
    if not _EXPECTED_PATH.exists():
        pytest.skip(
            "Jalon de phase 1 en attente : exécuter "
            "validation/external/spec10_pss2001_replication.R, stocker les "
            "sorties dans tests/replication/expected/spec10_pss2001.json "
            "et le dataset dans src/ardlpy/datasets/."
        )
    expected = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))
    raise NotImplementedError(
        "Comparaison à implémenter à réception des valeurs de référence "
        f"(fixed_regressors D7475/D7579 requis) : {sorted(expected)}"
    )
