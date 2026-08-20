"""Spec 15 §3 — réplication R bootCT::boot_ardl (données danoises).

Valeur de référence produite par R (2026-08-20, R 4.6.1, bootCT 2.1.0)
via ``validation/external/spec15_bootct.R``. Seule la statistique
OBSERVÉE F_indep est comparée : elle est déterministe. Tolérance
contractuelle : 1e-6 (spec 15 §3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyardl.bounds import bounds_test
from pyardl.datasets import load_denmark

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec15.json").read_text(encoding="utf-8")
)

_ORDER = (3, {"LRY": 1, "IBO": 3, "IDE": 2})


@pytest.mark.external
def test_f_indep_case3_matches_bootct() -> None:
    data = load_denmark()
    res = bounds_test(
        data["LRM"],
        data[["LRY", "IBO", "IDE"]],
        case=3,
        order=_ORDER,
    )
    assert res.f_indep_stat == pytest.approx(
        _EXPECTED["f_indep_case3"], abs=_EXPECTED["tolerance"]["f_indep_stat"]
    )
    # Sur ces données, le cadre à trois tests conclut à la cointégration.
    assert res.classification()[0] == "cointegration"
