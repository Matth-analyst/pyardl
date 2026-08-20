"""Spec 07 §3.2 — concordance Johansen avec R urca::ca.jo.

Valeurs de référence produites par R (2026-08-20, R 4.6.1, urca) via
``validation/external/spec07_urca.R``. Tolérance contractuelle : 1e-4
sur les statistiques (spec 07 §3.2).

La correspondance des déterministes a été ÉTABLIE PAR MESURE, en
exécutant les six variantes d'urca (3 ``ecdet`` x 2 ``spec``) contre les
trois ``det_order`` de statsmodels : une seule paire coïncide. Voir
``docs/api/johansen.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyardl.cointegration import johansen
from pyardl.datasets import load_denmark

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec07.json").read_text(encoding="utf-8")
)
_TOL = _EXPECTED["tolerance"]["stat"]


def _reference(key: str) -> np.ndarray:
    """urca liste ses statistiques de r <= n-1 vers r = 0 ; on inverse."""
    return np.asarray(_EXPECTED[key], dtype=float)[::-1]


@pytest.mark.external
def test_trace_matches_urca() -> None:
    data = load_denmark()[_EXPECTED["variables"]]
    res = johansen(data, det_order=0, k_ar_diff=_EXPECTED["K_levels"] - 1)
    assert res.trace_stat.to_numpy() == pytest.approx(
        _reference("trace_urca_none_r_desc"), abs=_TOL
    )


@pytest.mark.external
def test_maxeig_matches_urca() -> None:
    data = load_denmark()[_EXPECTED["variables"]]
    res = johansen(data, det_order=0, k_ar_diff=_EXPECTED["K_levels"] - 1)
    assert res.maxeig_stat.to_numpy() == pytest.approx(
        _reference("eigen_urca_none_r_desc"), abs=_TOL
    )
