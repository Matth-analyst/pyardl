"""Spec 22 §3.4 — concordance croisee du Mean Group avec R plm.

Ce que ce test verrouille n'est PAS l'estimation individuelle : elle est
deja validee par la spec 05 contre le package R ARDL. C'est l'agregation
— la moyenne des theta_i et surtout sa variance INTER-individus, la
formule contre-intuitive de Pesaran-Smith.

`plm::pmg(model = "mg")` calcule exactement cette agregation. Le script
R a d'ailleurs verifie separement que plm et une agregation refaite a la
main donnent le meme resultat a 0.000e+00 : la formule de reference
n'est donc pas une boite noire, elle est lisible.

Le panel est SYNTHETIQUE et genere par pyardl lui-meme
(`validation/external/spec22_make_panel.py`, graine fixee) : aucune
donnee tierce n'est redistribuee, et le fichier est identique des deux
cotes de la comparaison. Une seconde verification, sur le panel Produc
de plm (donnees publiees de Munnell 1990), a donne un accord a 4.5e-14
sans etre embarquee ici pour ne pas redistribuer ce jeu.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pyardl.panel import MeanGroup

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec22.json").read_text(encoding="utf-8")
)
# Le panel est versionne AVEC le test, pas dans validation/ : ce dossier
# est gitignore, et un fichier absent aurait transforme ce test de
# non-regression en skip silencieux sur la CI.
_PANEL = Path(__file__).parent / "data" / "spec22_panel.csv"
_TOL = _EXPECTED["_provenance"]["tolerance"]


@pytest.fixture(scope="module")
def fitted() -> object:
    df = pd.read_csv(_PANEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MeanGroup(df, y="y", X=["x"], id="id", time="t", order=(1, 1)).fit()


def test_panel_size(fitted) -> None:  # type: ignore[no-untyped-def]
    assert fitted.n_units == _EXPECTED["n_units"]


@pytest.mark.parametrize("term", ["const", "y.L1", "x.L0", "x.L1"])
def test_group_coefficients_match_plm(fitted, term: str) -> None:  # type: ignore[no-untyped-def]
    """Moyennes de groupe des coefficients bruts, contre plm::pmg."""
    ref = _EXPECTED["coefficients"][term]
    row = fitted.coefficients.loc[term]
    assert row["coef"] == pytest.approx(ref["coef"], abs=_TOL)


@pytest.mark.parametrize("term", ["const", "y.L1", "x.L0", "x.L1"])
def test_between_individual_standard_errors_match_plm(  # type: ignore[no-untyped-def]
    fitted, term: str
) -> None:
    """LE test de la spec : la variance vient de la dispersion
    inter-individus, et deux implementations independantes doivent
    tomber sur le meme nombre."""
    ref = _EXPECTED["coefficients"][term]
    row = fitted.coefficients.loc[term]
    assert row["se"] == pytest.approx(ref["se"], abs=_TOL)


def test_longrun_theta_matches(fitted) -> None:  # type: ignore[no-untyped-def]
    ref = _EXPECTED["longrun"]["x"]
    row = fitted.longrun.loc["x"]
    assert row["theta"] == pytest.approx(ref["theta"], abs=_TOL)
    assert row["se"] == pytest.approx(ref["se"], abs=_TOL)


def test_adjustment_matches(fitted) -> None:  # type: ignore[no-untyped-def]
    ref = _EXPECTED["adjustment"]
    assert fitted.adjustment["lambda"] == pytest.approx(ref["lambda"], abs=_TOL)
    assert fitted.adjustment["se"] == pytest.approx(ref["se"], abs=_TOL)


@pytest.mark.parametrize("unit", ["u00", "u01", "u02", "u03", "u04"])
def test_individual_thetas_match(fitted, unit: str) -> None:  # type: ignore[no-untyped-def]
    """Un accord sur la moyenne pourrait masquer deux erreurs qui se
    compensent. Les theta_i individuels sont donc verifies aussi."""
    assert fitted.theta_i.loc[unit, "x"] == pytest.approx(
        _EXPECTED["theta_i_head"][unit], abs=_TOL
    )
