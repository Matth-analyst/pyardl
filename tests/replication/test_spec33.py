"""Spec 33 §5.4 — concordance externe Enders-Siklos avec R apt::ciTarFit.

Valeurs de référence produites par R (2026-09-20, R 4.6.1, package
`apt`, `ciTarFit()`) via ``validation/external/spec33_apt.R``, exécuté
sur les mêmes données que ``pyardl``
(``tests/replication/data/spec29_synthetic.csv``, déjà versionnée pour
spec 29 — aucune propriété de rupture n'est requise ici, seule la
régression à seuil symétrique importe).

Contrairement à spec 29 (sélection de retards divergente entre pyardl
et R), le retard est ici FIXÉ à 1 des deux côtés — testé donc directement
sur le moteur de régression à seuil
(``pyardl.cointegration.enders_siklos._threshold_regression``), sans
passer par la sélection automatique de ``enders_siklos()`` — ce qui
autorise une tolérance stricte (1e-6), effectivement atteinte (identité
à 1e-10 près). C'est ce recoupement qui a mis au jour, puis vérifié la
correction, du biais de type « look-ahead » documenté dans
``docs/QUESTIONS.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pyardl.cointegration.enders_siklos import _threshold_regression
from pyardl.cointegration.engle_granger import engle_granger

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec33.json").read_text(encoding="utf-8")
)
_DATA_PATH = Path(__file__).parent.parent.parent / _EXPECTED["data_file"]


def _resid():  # type: ignore[no-untyped-def]
    data = pd.read_csv(_DATA_PATH)
    eg = engle_granger(
        data["y"].to_numpy(), data["x"].to_numpy(), trend=_EXPECTED["trend"]
    )
    return eg.resid.to_numpy()


@pytest.mark.external
@pytest.mark.parametrize("variant", ["tar", "mtar"])
def test_threshold_regression_matches_apt(variant: str) -> None:
    resid = _resid()
    rho1, rho2, phi_stat, _, _, _ = _threshold_regression(
        resid, variant, 0.0, _EXPECTED["lags"]
    )
    ref = _EXPECTED[variant]
    tol = _EXPECTED["tolerance"]
    assert rho1 == pytest.approx(ref["rho1"], abs=tol["rho_abs"])
    assert rho2 == pytest.approx(ref["rho2"], abs=tol["rho_abs"])
    assert phi_stat == pytest.approx(ref["phi_stat"], abs=tol["phi_abs"])
