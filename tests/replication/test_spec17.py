"""Spec 17 §4.4 — replication R `nardl` (jeu inflation/alimentation).

Valeurs de reference produites par R (2026-08-22, R 4.6.1, nardl 0.1.6)
via ``validation/external/spec17_nardl.R``. Tolerance contractuelle :
1e-6 sur les coefficients et les erreurs types.

LE PIEGE, LU ET NON DEVINE
--------------------------
Le package R n'utilise pas la parametrisation UECM de PSS. Son ``lxp``
empile le NIVEAU CONTEMPORAIN de x+ puis ses retards, la ou l'UECM
utilise le niveau retarde et des differences ; et son cumul demarre sans
zero initial, d'ou un decalage d'un pas. Les noms des coefficients
(``inf_p``, ``inf_p_1``, ...) ne disent rien de tout cela : la convention
a ete etablie en lisant la construction des colonnes, puis verifiee en
reproduisant les coefficients.

Ce test verifie donc que NOTRE decomposition, placee dans LEUR
specification, rend exactement leurs chiffres. C'est ce qui isole la
decomposition — le coeur numerique de la spec — de toute difference de
parametrisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyardl.nardl import partial_sums

_HERE = Path(__file__).parent
_EXPECTED = json.loads(
    (_HERE / "expected" / "spec17.json").read_text(encoding="utf-8")
)
_DATA = _HERE.parents[1] / "validation" / "external" / "spec17_fod.csv"


def _rebuild_r_design() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Le design de R, reconstruit avec NOTRE decomposition."""
    fod = pd.read_csv(_DATA)
    y = fod["food"].to_numpy(dtype=float)
    x = pd.Series(fod["inf"].to_numpy(dtype=float), name="inf")
    pos, neg = partial_sums(x)
    x_pos = pos.to_numpy()
    x_neg = neg.to_numpy()

    n = y.size
    rows = range(4, n - 1)
    target = np.array([y[k + 1] - y[k] for k in rows])
    columns = [np.ones(target.size), np.array([y[k] for k in rows])]
    names = ["Const", "food_1"]
    for i in range(5):
        columns.append(np.array([x_pos[k + 1 - i] for k in rows]))
        names.append("inf_p" if i == 0 else f"inf_p_{i}")
    columns.append(np.array([x_neg[k + 1] for k in rows]))
    names.append("inf_n")
    return np.column_stack(columns), target, names


@pytest.mark.external
def test_coefficients_match_r_nardl() -> None:
    design, target, names = _rebuild_r_design()
    assert design.shape[0] == _EXPECTED["n_obs"]
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    tol = _EXPECTED["tolerance"]["coef"]
    for name, value in zip(names, beta, strict=True):
        assert value == pytest.approx(_EXPECTED["coefficients"][name], abs=tol)


@pytest.mark.external
def test_standard_errors_match_r_nardl() -> None:
    design, target, names = _rebuild_r_design()
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    dof = design.shape[0] - design.shape[1]
    cov = np.linalg.pinv(design.T @ design) * (resid @ resid) / dof
    se = np.sqrt(np.diag(cov))
    tol = _EXPECTED["tolerance"]["se"]
    for name, value in zip(names, se, strict=True):
        assert value == pytest.approx(_EXPECTED["standard_errors"][name], abs=tol)


@pytest.mark.external
def test_decomposition_identity_holds_on_the_real_data() -> None:
    """Le verrou de la spec, verifie sur les donnees de la reference."""
    from pyardl.nardl import decomposition_error

    fod = pd.read_csv(_DATA)
    x = fod["inf"].to_numpy(dtype=float)
    pos, neg = partial_sums(x)
    assert decomposition_error(x, pos, neg) < 1e-12
