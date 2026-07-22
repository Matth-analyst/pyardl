"""Spec 03 §6.2 — réplication R ARDL::uecm/multipliers (données danoises).

Valeurs de référence produites par le package R ARDL (exécution du
2026-07-07, R 4.6.1, ARDL 0.2.5 — provenance dans le JSON), via
``validation/external/extract_expected_json.R``. Tolérance
contractuelle : theta_j et lambda à 1e-6 (spec 03 §6.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyardl.core.ardl import ARDL
from pyardl.datasets import load_denmark

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec03.json").read_text(encoding="utf-8")
)


@pytest.mark.external
def test_longrun_and_lambda_match_r_ardl() -> None:
    """theta_j et lambda identiques à 1e-6 au package R (ARDL(3,1,3,2))."""
    data = load_denmark()
    res = ARDL(
        data["LRM"],
        data[["LRY", "IBO", "IDE"]],
        order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}),
    ).fit()
    expected = _EXPECTED["model_318_2"]
    tol = _EXPECTED["tolerance"]["theta"]

    theta = res.longrun["theta"]
    for name in ("LRY", "IBO", "IDE"):
        assert theta[name] == pytest.approx(expected["theta"][name], abs=tol)

    ecm = res.to_ecm()
    assert ecm.lam == pytest.approx(expected["lambda"], abs=tol)


@pytest.mark.external
def test_q_zero_case_matches_r_ardl() -> None:
    """Cas q_IDE = 0 (docs/QUESTIONS.md) : R ARDL confirme la convention
    pyardl (niveau IDE contemporain, résidus ardl/uecm identiques) et
    les coefficients ECM concordent à 1e-6."""
    data = load_denmark()
    expected = _EXPECTED["model_q_ide_0"]
    assert expected["resid_gap_ardl_uecm"] < 1e-10  # confirmé côté R

    res = ARDL(
        data["LRM"],
        data[["LRY", "IBO", "IDE"]],
        order=(2, {"LRY": 2, "IBO": 2, "IDE": 0}),
    ).fit()
    ecm = res.to_ecm()
    r_coefs = expected["ecm_coefficients"]
    tol = _EXPECTED["tolerance"]["theta"]

    assert ecm.lam == pytest.approx(r_coefs["L(LRM, 1)"], abs=tol)
    # gamma : L(LRY,1) et L(IBO,1) retardés ; IDE contemporain (q=0)
    assert ecm.gamma[0] == pytest.approx(r_coefs["L(LRY, 1)"], abs=tol)
    assert ecm.gamma[1] == pytest.approx(r_coefs["L(IBO, 1)"], abs=tol)
    assert ecm.gamma[2] == pytest.approx(r_coefs["IDE"], abs=tol)
    # court terme
    assert ecm.psi[0] == pytest.approx(r_coefs["d(L(LRM, 1))"], abs=tol)
    np.testing.assert_allclose(
        ecm.omega[0], [r_coefs["d(LRY)"], r_coefs["d(L(LRY, 1))"]], atol=tol
    )
    np.testing.assert_allclose(
        ecm.omega[1], [r_coefs["d(IBO)"], r_coefs["d(L(IBO, 1))"]], atol=tol
    )
    assert ecm.omega[2].size == 0  # q_IDE = 0 : pas de Δ

    theta = res.longrun["theta"]
    for name in ("LRY", "IBO", "IDE"):
        assert theta[name] == pytest.approx(expected["theta"][name], abs=tol)
