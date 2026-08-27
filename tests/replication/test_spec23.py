"""Spec 23 §6.3 — concordance croisee PMG / MG / DFE avec R ardlverse.

`ardlverse::panel_ardl()` annonce repliquer xtpmg (Blackburne & Frank
2007), la reference que la spec designe ; Stata n'est pas disponible
dans cet environnement.

Le panel a un theta COMMUN : c'est le monde pour lequel le PMG est
specifie, donc celui ou une divergence entre implementations ne peut
pas etre mise sur le compte d'une mauvaise specification.

Convention : ardlverse ecrit le terme de niveau en x_{it}, la spec 23 en
x_{i,t-1}. Les deux parametrisations portent le MEME theta de long terme
— les coefficients de court terme, eux, different — donc seuls theta,
lambda et la log-vraisemblance sont compares.

Un point de methode vaut d'etre retenu ici. La reference est executee a
tol = 1e-10 et NON a son defaut de 1e-6, ou elle s'arrete a 2.7e-07 de
son propre maximum. Ce n'est pas le coefficient qui l'a revele — 2.7e-07
a l'air d'un accord — mais la LOG-VRAISEMBLANCE, plus basse la qu'a
l'estimation de pyardl. Quand deux implementations divergent, la
quantite qu'elles maximisent toutes les deux les departage ; couper la
poire en deux ou relacher sa propre tolerance pour « correspondre »
aurait enterine l'erreur. Voir OBS-21.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pyardl.panel import DFE, PMG, MeanGroup, hausman

_EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "spec23.json").read_text(encoding="utf-8")
)
_PANEL = Path(__file__).parent / "data" / "spec23_panel.csv"
_TOL = _EXPECTED["_provenance"]["tolerance"]
_KW = {"y": "y", "X": ["x"], "id": "id", "time": "t", "order": (1, 1)}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(_PANEL)


@pytest.fixture(scope="module")
def fits(panel: pd.DataFrame) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "pmg": PMG(panel, **_KW).fit(),  # type: ignore[arg-type]
            "mg": MeanGroup(panel, **_KW).fit(),  # type: ignore[arg-type]
            "dfe": DFE(panel, **_KW).fit(),  # type: ignore[arg-type]
        }


def test_panel_shape(fits: dict) -> None:
    assert fits["pmg"].n_units == _EXPECTED["n_units"]
    assert fits["pmg"].nobs == _EXPECTED["nobs"]


def test_pmg_converged(fits: dict) -> None:
    assert fits["pmg"].converged


@pytest.mark.parametrize("estimator", ["pmg", "mg", "dfe"])
def test_longrun_theta_matches_ardlverse(fits: dict, estimator: str) -> None:
    ref = _EXPECTED[estimator]["theta"]
    got = fits[estimator].longrun.loc["x", "theta"]
    assert got == pytest.approx(ref, abs=_TOL)


@pytest.mark.parametrize("estimator", ["pmg", "mg", "dfe"])
def test_longrun_se_matches_ardlverse(fits: dict, estimator: str) -> None:
    """Pour le PMG c'est LE test de la variance : la premiere version de
    la formule projetait sur W_i seulement, en oubliant que lambda_i est
    lui aussi estime. Elle rendait une erreur type ~5 % trop petite et
    aurait passe n'importe quel test de coherence interne."""
    ref = _EXPECTED[estimator]["se"]
    got = fits[estimator].longrun.loc["x", "se"]
    assert got == pytest.approx(ref, abs=_TOL)


@pytest.mark.parametrize("estimator", ["pmg", "mg", "dfe"])
def test_adjustment_matches_ardlverse(fits: dict, estimator: str) -> None:
    ref = _EXPECTED[estimator]["lambda"]
    got = fits[estimator].adjustment["lambda"]
    assert got == pytest.approx(ref, abs=_TOL)


def test_concentrated_loglikelihood_matches(fits: dict) -> None:
    """La log-vraisemblance est l'objet qui departage deux estimations
    concurrentes ; qu'elle concorde a 4e-12 dit que les deux cotes
    maximisent bien la meme fonction."""
    assert fits["pmg"].loglik == pytest.approx(_EXPECTED["pmg"]["loglik"], abs=1e-8)


def test_pmg_is_more_precise_than_mg_under_homogeneity(fits: dict) -> None:
    """Le theta du panel EST commun : c'est le cas ou le PMG doit gagner
    en precision, et les deux implementations doivent etre d'accord sur
    le fait qu'il le fait."""
    assert fits["pmg"].longrun.loc["x", "se"] < fits["mg"].longrun.loc["x", "se"]
    assert _EXPECTED["pmg"]["se"] < _EXPECTED["mg"]["se"]


def test_hausman_does_not_reject_on_a_homogeneous_panel(fits: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = hausman(fits["mg"], fits["pmg"])
    assert result.pvalue > 0.05
    assert "do not reject" in result.decision


def test_backfitting_and_newton_agree(panel: pd.DataFrame) -> None:
    """§6.2 — deux chemins vers le meme maximum. La spec demande 1e-6."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = PMG(panel, **_KW).fit()  # type: ignore[arg-type]
        newton = PMG(panel, method="newton", **_KW).fit()  # type: ignore[arg-type]
    assert back.longrun.loc["x", "theta"] == pytest.approx(
        newton.longrun.loc["x", "theta"], abs=1e-6
    )
    assert back.loglik == pytest.approx(newton.loglik, abs=1e-8)
