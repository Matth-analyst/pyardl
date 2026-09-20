"""Spec 31 §5.4 — concordance externe Bai-Perron avec R strucchange.

Valeurs de référence produites par R (2026-09-20, R 4.6.1, package
strucchange, `breakpoints()`) via ``validation/external/spec31_strucchange.R``,
exécuté sur les mêmes données que ``pyardl``
(``tests/replication/data/spec31_synthetic.csv``, versionnées).

Contrairement à spec 29 (où la sélection de retards ADF divergeait
entre pyardl et R), ici les deux implémentations résolvent exactement
le même problème d'optimisation (minimisation de SSR par partition, sur
un modèle sans dynamique à sélectionner) : la spec 31 §5.4 demande donc
une concordance stricte (1e-6) sur les dates de rupture et SSR(m), pas
seulement une concordance d'ordre de grandeur — et c'est ce qui est
observé (SSR identique à 1e-10 près sur les données testées).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyardl.cointegration import bai_perron

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec31.json").read_text(encoding="utf-8")
)
_DATA_PATH = Path(__file__).parent.parent.parent / _EXPECTED["data_file"]


def _load() -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(_DATA_PATH)
    y = data["y"].to_numpy()
    x = np.ones(len(y)).reshape(-1, 1)
    return y, x


@pytest.mark.external
def test_break_indices_match_strucchange() -> None:
    y, x = _load()
    res = bai_perron(
        y,
        x,
        max_breaks=_EXPECTED["max_breaks"],
        trim=_EXPECTED["trim"],
        selection="ic",
        ic=_EXPECTED["ic"],
    )
    assert res.n_breaks == _EXPECTED["r_m_hat_by_bic"]
    tol = _EXPECTED["tolerance"]["break_index_abs"]
    for est, ref in zip(res.break_indices, _EXPECTED["r_break_indices"], strict=True):
        assert abs(est - ref) <= tol


@pytest.mark.external
def test_ssr_matches_strucchange() -> None:
    y, x = _load()
    res = bai_perron(
        y,
        x,
        max_breaks=_EXPECTED["max_breaks"],
        trim=_EXPECTED["trim"],
        selection="ic",
        ic=_EXPECTED["ic"],
    )
    ref_ssr = _EXPECTED["r_ssr_by_m"][res.n_breaks]
    assert res.ssr == pytest.approx(ref_ssr, abs=_EXPECTED["tolerance"]["ssr_abs"])
