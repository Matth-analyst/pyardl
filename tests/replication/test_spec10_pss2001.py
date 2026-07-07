"""Spec 10 §6 — JALON DE PHASE 1 : réplication PSS 2001 (salaires UK).

Équation de salaires réels UK de Pesaran, Shin & Smith (2001, §6),
telle que répliquée par Natsiopoulos & Tzeremes (2022, JAE) avec le
package R ARDL : ARDL(6, 0, 5, 4, 5) avec tendance et dummies D7475 /
D7579. Valeurs de référence produites par R (2026-07-07, R 4.6.1,
ARDL 0.2.5) via ``validation/external/extract_expected_json.R`` —
jamais estimées par nous. Tolérance contractuelle : F et t à 1e-4
(spec 10 §6) ; coefficients UECM à 1e-6 (bonus, même moteur OLS).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from ardlpy.bounds import bounds_test
from ardlpy.datasets import load_pss2001

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec10_pss2001.json").read_text(
        encoding="utf-8"
    )
)

_ORDER = (6, {"Prod": 0, "UR": 5, "Wedge": 4, "Union": 5})

# nom R -> nom ardlpy (q_Prod = 0 -> niveau contemporain "Prod" ✓)
_LEVEL_MAP = {
    "(Intercept)": "const",
    "trend(w)": "trend",
    "L(w, 1)": "w.L1",
    "Prod": "Prod.L0",
    "L(UR, 1)": "UR.L1",
    "L(Wedge, 1)": "Wedge.L1",
    "L(Union, 1)": "Union.L1",
    "D7475": "D7475",
    "D7579": "D7579",
}


def _run(case: int):
    data = load_pss2001()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bounds_test(
            data["w"],
            data[["Prod", "UR", "Wedge", "Union"]],
            case=case,
            order=_ORDER,
            fixed_regressors=data[["D7475", "D7579"]],
        )


@pytest.mark.external
def test_f_case4_matches_r() -> None:
    res = _run(case=4)
    assert res.f_stat == pytest.approx(
        _EXPECTED["f_case4"], abs=_EXPECTED["tolerance"]["f_stat"]
    )


@pytest.mark.external
def test_f_and_t_case5_match_r() -> None:
    res = _run(case=5)
    assert res.f_stat == pytest.approx(
        _EXPECTED["f_case5"], abs=_EXPECTED["tolerance"]["f_stat"]
    )
    assert res.t_stat == pytest.approx(
        _EXPECTED["t_case5"], abs=_EXPECTED["tolerance"]["t_stat"]
    )


@pytest.mark.external
def test_uecm_coefficients_and_ssr_match_r() -> None:
    """Mêmes coefficients UECM (niveaux, déterministes, dummies) à 1e-6
    et même SSR — le moteur OLS réplique exactement la régression R."""
    res = _run(case=5)
    coefs = res.uecm["coef"]
    for r_name, our_name in _LEVEL_MAP.items():
        expected = _EXPECTED["uecm_coefficients"][r_name]
        if our_name == "trend":
            # R ARDL::trend() sur un ts trimestriel construit t/4 (tendance
            # en années) ; ardlpy utilise t (1..T). Reparamétrisation
            # affine : coefficient R = 4 x coefficient ardlpy ; F, t, SSR
            # et tous les autres coefficients sont invariants.
            assert 4.0 * coefs[our_name] == pytest.approx(expected, abs=1e-6)
        else:
            assert coefs[our_name] == pytest.approx(expected, abs=1e-6), r_name
    assert res._fit.ssr == pytest.approx(_EXPECTED["uecm_ssr"], rel=1e-8)


@pytest.mark.external
def test_decisions_replicate_pss_conclusions() -> None:
    """PSS 2001 §6 : F au-dessus de la borne I(1) à 5 % (cas IV et V,
    k=4) -> cointégration par le F ; le t du cas V (-2.86) ne dépasse
    pas la borne I(0) (-3.41 à 5 %) -> non-rejet, conformément à la
    discussion de l'article (le F porte la conclusion)."""
    res4 = _run(case=4)
    assert res4.decision_f == "cointegration"
    res5 = _run(case=5)
    assert res5.decision_f == "cointegration"
    assert res5.decision_t == "no_cointegration"
