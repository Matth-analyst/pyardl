"""Spec 32 §5.4 — concordance externe Hansen threshold regression avec R pdR.

Valeurs de référence produites par R (2026-09-20, R 4.6.1, package
`pdR`, `SMPLSplit_het()` — qui encapsule le code fourni par B. E.
Hansen lui-même pour son article de 2000) via
``validation/external/spec32_pdR.R``, exécuté sur les mêmes données que
``pyardl`` (``tests/replication/data/spec32_synthetic.csv``, versionnée).

Seul ``gamma_hat`` (l'estimation du seuil par minimisation de SSR sur
la grille, la partie de l'algorithme partagée par toute implémentation)
est comparé à une tolérance serrée. La statistique de test elle-même
n'est PAS comparée : `pdR::SMPLSplit_het` calcule un LM robuste à
l'hétéroscédasticité (White), `pyardl` un F classique homoscédastique —
deux formes légitimes mais différentes par construction, pas un écart
numérique à corriger. Voir `expected/spec32.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyardl.threshold import threshold_ardl

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec32.json").read_text(encoding="utf-8")
)
_DATA_PATH = Path(__file__).parent.parent.parent / _EXPECTED["data_file"]


@pytest.mark.external
def test_gamma_hat_matches_pdR() -> None:
    data = pd.read_csv(_DATA_PATH)
    y = data["y"].to_numpy()
    x = data["x"].to_numpy()
    q = data["q"].to_numpy()
    design = np.column_stack([np.ones(len(y)), x])
    res = threshold_ardl(
        y, design, q, delay=_EXPECTED["delay"], trim=_EXPECTED["trim"], n_boot=5, seed=0
    )
    assert res.gamma_hat == pytest.approx(
        _EXPECTED["r_gamma_hat"], abs=_EXPECTED["tolerance"]["gamma_abs"]
    )


@pytest.mark.external
def test_both_reject_linearity() -> None:
    data = pd.read_csv(_DATA_PATH)
    y = data["y"].to_numpy()
    x = data["x"].to_numpy()
    q = data["q"].to_numpy()
    design = np.column_stack([np.ones(len(y)), x])
    res = threshold_ardl(
        y,
        design,
        q,
        delay=_EXPECTED["delay"],
        trim=_EXPECTED["trim"],
        n_boot=99,
        seed=0,
    )
    assert _EXPECTED["r_pvalue"] < 0.05
    assert res.decision(0.05) == "threshold"
