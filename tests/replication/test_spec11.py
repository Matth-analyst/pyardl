"""Spec 11 §3.3 — réplication R ARDL::bounds_t_test (données danoises).

Valeurs de référence produites par R (2026-07-07, R 4.6.1, ARDL 0.2.5)
via ``validation/external/extract_expected_json.R``. Tolérance
contractuelle : statistique t à 1e-6 (spec 11 §3.3).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from ardlpy.bounds import bounds_test
from ardlpy.datasets import load_denmark

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec11.json").read_text(encoding="utf-8")
)

_ORDER = (3, {"LRY": 1, "IBO": 3, "IDE": 2})


@pytest.mark.external
def test_t_case3_matches_r_bounds_t_test() -> None:
    data = load_denmark()
    res = bounds_test(data["LRM"], data[["LRY", "IBO", "IDE"]], case=3, order=_ORDER)
    assert res.t_stat == pytest.approx(
        _EXPECTED["t_case3"], abs=_EXPECTED["tolerance"]["t_stat"]
    )


@pytest.mark.external
def test_t_case5_matches_r_bounds_t_test() -> None:
    """Cas V : modèle avec tendance (trend(LRM) côté R)."""
    data = load_denmark()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = bounds_test(
            data["LRM"], data[["LRY", "IBO", "IDE"]], case=5, order=_ORDER
        )
    assert res.t_stat == pytest.approx(
        _EXPECTED["t_case5_with_trend"], abs=_EXPECTED["tolerance"]["t_stat"]
    )
