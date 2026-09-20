"""Spec 29 §5.4 — concordance externe Gregory-Hansen.

Valeurs de référence produites par R (2026-09-20, R 4.6.1, package
COINT 0.0.4) via ``validation/external/spec29_ghansen.R``, exécuté sur
les mêmes données que ``pyardl``
(``tests/replication/data/spec29_synthetic.csv``, versionnées — le
script R lit ce même fichier).

Déviation par rapport à la spec 29 §5.4 : celle-ci nommait
``R funtimes::gh_coint`` comme référence externe. Cette fonction
n'existe pas dans la version courante de ``funtimes`` sur CRAN
(vérifié le 2026-09-20) — ``COINT::GHansen`` (qui cite Gregory & Hansen
1996A/1996B directement dans sa documentation) est utilisé à la place.
Voir ``docs/QUESTIONS.md``.

La sélection de retards diffère entre les deux implémentations
(``select_lags`` MAIC-style côté pyardl, ``urca::ur.df(selectlags="AIC")``
côté R) : la statistique ADF* n'est donc PAS comparée à une tolérance
serrée, seulement à son ordre de grandeur et à son signe. Le point de
rupture et la décision, en revanche, sont un verrou de concordance
strict (tolérances documentées dans ``expected/spec29.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pyardl.cointegration import gregory_hansen

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec29.json").read_text(encoding="utf-8")
)
_DATA_PATH = Path(__file__).parent.parent.parent / _EXPECTED["data_file"]


def _load() -> tuple[pd.Series, pd.Series]:
    data = pd.read_csv(_DATA_PATH)
    return data["y"], data["x"]


@pytest.mark.external
def test_break_index_matches_r_ghansen() -> None:
    y, x = _load()
    res = gregory_hansen(
        y,
        x,
        model=_EXPECTED["model"],
        trim=_EXPECTED["trim"],
        max_lags=12,
        ic="aic",
        n_boot=5,
        seed=0,
    )
    tol = _EXPECTED["tolerance"]["break_index_abs"]
    assert abs(res.break_index - _EXPECTED["r_break_index_0indexed"]) <= tol


@pytest.mark.external
def test_statistic_same_order_of_magnitude_and_sign() -> None:
    y, x = _load()
    res = gregory_hansen(
        y,
        x,
        model=_EXPECTED["model"],
        trim=_EXPECTED["trim"],
        max_lags=12,
        ic="aic",
        n_boot=5,
        seed=0,
    )
    r_stat = _EXPECTED["r_adf_stat"]
    factor = _EXPECTED["tolerance"]["stat_same_order_of_magnitude_factor"]
    assert res.statistic < 0
    assert r_stat < 0
    ratio = abs(res.statistic) / abs(r_stat)
    assert 1.0 / factor <= ratio <= factor


@pytest.mark.external
def test_both_reject_h0_at_r_critical_value() -> None:
    """R's own table decision at 1% must agree with pyardl's bootstrap decision."""
    y, x = _load()
    res = gregory_hansen(
        y,
        x,
        model=_EXPECTED["model"],
        trim=_EXPECTED["trim"],
        max_lags=12,
        ic="aic",
        n_boot=199,
        seed=0,
    )
    r_cv_1pct = _EXPECTED["r_critical_values"]["0.01"]
    assert _EXPECTED["r_adf_stat"] < r_cv_1pct
    assert res.decision(0.01) == "cointegration"
