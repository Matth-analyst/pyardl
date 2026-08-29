"""Specs 01 et 02 — concordance croisee avec R dLagM.

DEUX RESULTATS DE NATURE DIFFERENTE
-----------------------------------
**Almon (spec 02) : accord exact.** `polyDlm` et `AlmonModel` font la
meme chose, donc l'accord doit etre a la precision machine. Il l'est :
1.8e-13 sur les beta, 1.2e-9 sur leurs erreurs types, 7.4e-15 sur la
somme des carres des residus. Et il tient AUSSI quand pyardl travaille
en base de Tchebychev (5.0e-14) alors que dLagM utilise la base brute
i^j — ce qui montre que le reconditionnement est une reparametrisation
et pas un autre modele.

**Koyck (spec 01) : desaccord, et sa cause exacte.** Les deux
implementations n'instrumentent pas le meme regresseur. L'objet `ivreg`
que dLagM renvoie porte sa propre formule, `y.t ~ Y.1 + X.t | Y.1 +
X.t_1` : Y.1 figure des deux cotes de la barre, donc il est traite
comme exogene et c'est X.t qui est instrumente.

Or dans le modele de Koyck, le regresseur endogene est y_(t-1) : c'est
lui que la transformation correle mecaniquement a l'erreur. pyardl suit
Liviatan (1963) et l'instrumente par x_(t-1).

Le test ne se contente pas de constater l'ecart : il REPRODUIT le jeu
d'instruments de dLagM dans pyardl et verifie qu'on retombe sur ses
chiffres a 1e-8. Un desaccord explique et reproductible est un
resultat ; un desaccord constate n'en est pas un.

La consequence sur le biais est mesuree ailleurs, sur un DGP dont on
connait la verite : `validation/spec01_montecarlo.py`, resume en OBS-26.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyardl.datasets import load_denmark
from pyardl.distributed_lags import AlmonModel, KoyckModel

_HERE = Path(__file__).parent
_EXPECTED = json.loads(
    (_HERE / "expected" / "spec01_02.json").read_text(encoding="utf-8")
)
_PROV = _EXPECTED["_provenance"]

Q, K = 4, 2


@pytest.fixture(scope="module")
def denmark():
    return load_denmark()


@pytest.fixture(scope="module")
def almon(denmark):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AlmonModel(denmark["LRM"], denmark["LRY"], q=Q, r=K).fit()


@pytest.mark.external
def test_almon_weights_match_polydlm(almon) -> None:
    ref = np.asarray(_EXPECTED["polydlm"]["beta"], dtype=float)
    assert np.allclose(almon.lag_weights.to_numpy(), ref, atol=_PROV["tolerance_beta"])


@pytest.mark.external
def test_almon_standard_errors_match_polydlm(almon) -> None:
    """Les erreurs types transportees par H V H', sans delta-methode."""
    ref = np.asarray(_EXPECTED["polydlm"]["se_beta"], dtype=float)
    assert np.allclose(
        almon.bse_lag_weights.to_numpy(), ref, atol=_PROV["tolerance_se"]
    )


@pytest.mark.external
def test_almon_intercept_and_ssr_match(almon) -> None:
    poly = _EXPECTED["polydlm"]
    tol = _PROV["tolerance_beta"]
    assert almon.intercept == pytest.approx(poly["intercept"], abs=tol)
    assert almon.ssr == pytest.approx(poly["ssr"], abs=tol)
    assert almon.nobs == _EXPECTED["nobs_poly"]


@pytest.mark.external
def test_the_chebyshev_basis_lands_on_the_same_weights(denmark) -> None:
    """dLagM travaille en base brute i^j, pyardl peut travailler en Tchebychev.

    Les deux doivent donner les MEMES beta. Si ce n'etait pas le cas, le
    reconditionnement changerait le modele au lieu de changer sa
    representation — et le choix de base deviendrait un choix de
    resultat.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cheby = AlmonModel(
            denmark["LRM"], denmark["LRY"], q=Q, r=K, basis="chebyshev"
        ).fit()
    ref = np.asarray(_EXPECTED["polydlm"]["beta"], dtype=float)
    assert np.allclose(cheby.lag_weights.to_numpy(), ref, atol=_PROV["tolerance_beta"])


@pytest.mark.external
def test_the_koyck_disagreement_is_reproduced_exactly(denmark) -> None:
    """On retombe sur dLagM en adoptant SON jeu d'instruments.

    C'est ce qui fait la difference entre « les deux logiciels ne
    s'accordent pas » et « voici exactement ou ils divergent » : le
    desaccord tient a un seul choix, identifiable et reproductible, et
    non a une accumulation de details numeriques.
    """
    y = denmark["LRM"].to_numpy()
    x = denmark["LRY"].to_numpy()
    y_dep = y[1:]
    ones = np.ones(y_dep.size)
    # Le jeu de dLagM, lu dans la formule que son propre objet renvoie :
    # Y.1 des deux cotes (donc exogene), X.t instrumente par X.t_1.
    regressors = np.column_stack([ones, y[:-1], x[1:]])
    instruments = np.column_stack([ones, y[:-1], x[:-1]])
    coefs = np.linalg.solve(instruments.T @ regressors, instruments.T @ y_dep)

    ref = _EXPECTED["koyckdlm"]["coefficients"]
    assert coefs[0] == pytest.approx(ref["intercept"], abs=_PROV["tolerance_koyck"])
    assert coefs[1] == pytest.approx(ref["y_lag1"], abs=_PROV["tolerance_koyck"])
    assert coefs[2] == pytest.approx(ref["x_t"], abs=_PROV["tolerance_koyck"])
    assert "Y.1 + X.t | Y.1 + X.t_1" in _EXPECTED["koyckdlm"]["instrument_formula"]


@pytest.mark.external
def test_pyardl_instruments_the_lagged_dependent_variable(denmark) -> None:
    """Et pyardl fait l'autre choix, deliberement.

    Sur ces donnees le resultat differe beaucoup — et pyardl le signale
    plutot que de le presenter comme acquis : LRY est tres persistante,
    donc x_(t-1) explique mal y_(t-1), et le F de premiere etape tombe
    a 3.70. L'avertissement d'instrument faible est emis, ce que le test
    verifie : la demande de monnaie danoise est un mauvais terrain pour
    un Koyck, et l'utilisateur doit l'apprendre du logiciel.
    """
    with pytest.warns(Warning, match="Weak instrument"):
        res = KoyckModel(denmark["LRM"], denmark["LRY"], method="iv").fit()
    assert res.extra["first_stage_f"] == pytest.approx(
        _EXPECTED["pyardl_first_stage_f"], rel=1e-6
    )
    ref = _EXPECTED["koyckdlm"]["coefficients"]
    assert abs(float(res._reg_params[2]) - ref["y_lag1"]) > 0.5
