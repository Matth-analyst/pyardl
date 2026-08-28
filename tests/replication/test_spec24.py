"""Spec 24 §4.4 — concordance croisee du CCE statique avec R plm.

CE QUI EST VALIDE ICI, ET CE QUI NE L'EST PAS
---------------------------------------------
La spec designe Stata `xtdcce2` comme reference du CS-ARDL dynamique.
Stata n'est pas disponible dans cet environnement, et aucun package R
n'implemente le CS-ARDL complet — c'est d'ailleurs l'argument de la spec
pour dire que c'est une zone vierge.

`plm::pcce(model = "mg")` implemente le CCE STATIQUE de Pesaran (2006),
qui est exactement le cas particulier de CS-DL sans differences
retardees ni retards des moyennes. Il valide donc trois choses, et le
dire precisement vaut mieux que de laisser croire que tout est couvert :

  - la construction des moyennes transversales ;
  - la regression individuelle augmentee de ces moyennes ;
  - l'agregation Mean Group et sa variance INTER-individus.

Le volet dynamique — retards des moyennes, long terme reconstruit depuis
les coefficients de court terme — n'a PAS de reference externe ici. Il
repose sur les tests internes et sur l'etude Monte Carlo, et un script
Stata est fourni pour le jour ou xtdcce2 sera accessible.

Le script R re-derive en outre l'agregation a la main depuis les OLS
individuelles et retrouve pcce aux memes chiffres : la reference est
lisible, pas une boite noire.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pyardl.panel import CSDL

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec24.json").read_text(encoding="utf-8")
)
_PANEL = Path(__file__).parent / "data" / "spec24_panel.csv"
_TOL = _EXPECTED["_provenance"]["tolerance"]


@pytest.fixture(scope="module")
def fitted() -> object:
    df = pd.read_csv(_PANEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # trunc_lags=0 et cs_lags=0 : c'est ce qui fait du CS-DL le CCE
        # statique, donc le modele que plm::pcce estime.
        return CSDL(
            df,
            y="y",
            X=["x"],
            id="id",
            time="t",
            trunc_lags=0,
            cs_lags=0,
            det="const",
        ).fit()


def test_panel_size(fitted) -> None:  # type: ignore[no-untyped-def]
    assert fitted.n_units == _EXPECTED["n_units"]


def test_group_theta_matches_pcce(fitted) -> None:  # type: ignore[no-untyped-def]
    assert fitted.longrun.loc["x", "theta"] == pytest.approx(
        _EXPECTED["cce_mg"]["theta"], abs=_TOL
    )


def test_between_individual_se_matches_pcce(fitted) -> None:  # type: ignore[no-untyped-def]
    """L'erreur type vient de la dispersion inter-individus (spec 22), et
    deux implementations independantes doivent tomber sur le meme
    nombre."""
    assert fitted.longrun.loc["x", "se"] == pytest.approx(
        _EXPECTED["cce_mg"]["se"], abs=_TOL
    )


@pytest.mark.parametrize("unit", ["u00", "u01", "u02", "u03", "u04"])
def test_individual_thetas_match(fitted, unit: str) -> None:  # type: ignore[no-untyped-def]
    """Un accord sur la moyenne pourrait masquer deux erreurs qui se
    compensent ; les theta_i sont donc verifies aussi."""
    assert fitted.theta_i.loc[unit, "x"] == pytest.approx(
        _EXPECTED["theta_i_head"][unit], abs=_TOL
    )


def test_the_augmentation_recovers_the_true_coefficient(fitted) -> None:
    """Le DGP a un facteur commun qui entre AUSSI dans x : sans les
    moyennes transversales, theta serait attire loin de 0.80. Ce test
    n'est pas une concordance, c'est la raison d'etre du module."""
    assert abs(fitted.longrun.loc["x", "theta"] - _EXPECTED["theta_true"]) < 0.02


def test_a_naive_mean_group_is_biased_on_the_same_data() -> None:
    """Verification par contraste, sur les MEMES donnees : retirer les
    moyennes transversales doit degrader l'estimation. Sans ce test, la
    concordance ci-dessus dirait seulement que deux implementations
    calculent la meme chose, pas que cette chose sert a quelque chose."""
    df = pd.read_csv(_PANEL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        augmented = CSDL(
            df, y="y", X=["x"], id="id", time="t", trunc_lags=0, cs_lags=0
        ).fit()
    naive_theta = (
        df.groupby("id")
        .apply(
            lambda block: block[["y", "x"]].cov().iloc[0, 1] / block["x"].var(),
            include_groups=False,
        )
        .mean()
    )
    true_theta = _EXPECTED["theta_true"]
    assert abs(naive_theta - true_theta) > abs(
        augmented.longrun.loc["x", "theta"] - true_theta
    )
