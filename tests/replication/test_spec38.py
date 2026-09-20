"""Spec 38 §5.4 — réplication R systemfit::systemfit(method="SUR").

Valeurs de référence produites par le package R systemfit (2026-09-20,
R 4.6.1), via ``validation/external/spec38_systemfit.R`` sur des données
générées par ``validation/external/spec38_generate_data.py`` (seed
fixée, mêmes valeurs des deux côtés). Extraction reproductible :
``validation/external/extract_spec38_json.py``.

Tolérance contractuelle (spec 38 §5.4) : coefficients à 1e-6, Sigma_hat
concordant (voir note de provenance dans le JSON sur la valeur
recalculée au point fixe, ``systemfit$residCov`` retardant d'une
itération).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyardl.system import SystemARDL

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec38.json").read_text(encoding="utf-8")
)
_DATA_PATH = Path(__file__).parent / "data" / "spec38_synthetic.csv"


def _fit_system() -> object:
    df = pd.read_csv(_DATA_PATH)
    eqs = {
        "eq1": (df["y1"].to_numpy(), pd.DataFrame({"x1": df["x1"]}), (1, 1)),
        "eq2": (df["y2"].to_numpy(), pd.DataFrame({"x2": df["x2"]}), (1, 1)),
    }
    return SystemARDL(eqs, iterate=True, max_iter=200, tol=1e-12).fit()


@pytest.mark.external
def test_sur_coefficients_match_systemfit() -> None:
    """FGLS coefficients identiques à 1e-6 à ceux de systemfit(method="SUR")."""
    res = _fit_system()
    tol = _EXPECTED["tolerance"]["coefficients"]
    expected = _EXPECTED["sur_coefficients"]

    mapping = {
        "eq1_const": ("eq1", "const"),
        "eq1_y1_L1": ("eq1", "y.L1"),
        "eq1_x1_L0": ("eq1", "x1.L0"),
        "eq1_x1_L1": ("eq1", "x1.L1"),
        "eq2_const": ("eq2", "const"),
        "eq2_y2_L1": ("eq2", "y.L1"),
        "eq2_x2_L0": ("eq2", "x2.L0"),
        "eq2_x2_L1": ("eq2", "x2.L1"),
    }
    for r_key, (eq_name, term) in mapping.items():
        value = res.fgls_params[eq_name][term]
        assert value == pytest.approx(expected[r_key], abs=tol), r_key


@pytest.mark.external
def test_sigma_hat_matches_systemfit_at_convergence() -> None:
    """Sigma_hat concordant avec systemfit recalculé au point fixe (voir JSON)."""
    res = _fit_system()
    tol = _EXPECTED["tolerance"]["sigma"]
    expected_sigma = np.array(_EXPECTED["sigma_hat"])

    np.testing.assert_allclose(res.sigma.to_numpy(), expected_sigma, atol=tol)


@pytest.mark.external
def test_ols_per_equation_matches_systemfit_ols() -> None:
    """OLS par équation identique à 1e-6 à systemfit(method="OLS") — étape 1."""
    res = _fit_system()
    tol = _EXPECTED["tolerance"]["coefficients"]
    expected = _EXPECTED["ols_coefficients"]

    mapping = {
        "eq1_const": ("eq1", "const"),
        "eq1_y1_L1": ("eq1", "y.L1"),
        "eq1_x1_L0": ("eq1", "x1.L0"),
        "eq1_x1_L1": ("eq1", "x1.L1"),
        "eq2_const": ("eq2", "const"),
        "eq2_y2_L1": ("eq2", "y.L1"),
        "eq2_x2_L0": ("eq2", "x2.L0"),
        "eq2_x2_L1": ("eq2", "x2.L1"),
    }
    for r_key, (eq_name, term) in mapping.items():
        value = res.equations[eq_name].params[term]
        assert value == pytest.approx(expected[r_key], abs=tol), r_key
