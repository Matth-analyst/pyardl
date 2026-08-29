"""Spec 08 §5.2 — concordance croisee FMOLS / DOLS / noyau avec R cointReg.

`cointReg` 0.2.0 implemente FMOLS (`cointRegFM`), DOLS (`cointRegD`) et
la covariance de long terme (`getLongRunVar`) — donc les trois briques de
cette spec ont une reference, sauf CCR que le package ne couvre pas.

DEUX CONVENTIONS ETABLIES PAR LA MESURE
---------------------------------------
Aucune des deux n'etait devinable, et chacune produisait un resultat
plausible.

1. La matrice unilaterale Delta a DEUX ecritures en circulation qui
   different par une transposee. Elles donnent le MEME Omega — donc
   comparer Omega ne les separe pas — et un lambda+ different, donc un
   theta different. La bonne est Delta = Gamma_0 + somme k_j Gamma_j',
   verifiee a 2.2e-16 contre getLongRunVar.

2. Le terme de correction de biais s'echelonne sur l'echantillon
   COMPLET (T), pas sur les T-1 lignes que la regression utilise. En
   resolvant pour le lambda+ que les coefficients publies de cointReg
   impliquent, le rapport a ma version en T-1 est ressorti exactement a
   55/54 sur les TROIS coefficients — c'est cette constance qui a
   identifie la convention plutot qu'un tatonnement.

PREBLANCHIMENT : POURQUOI CES TESTS LE DESACTIVENT
---------------------------------------------------
Les estimateurs de pyardl preblanchissent PAR DEFAUT (Andrews-Monahan),
parce que sans cela leur couverture reste sous le nominal. Les appels de
`cointReg` compares ici ne preblanchissent pas. Ces tests passent donc
`prewhiten=False` : ils verifient que les deux implementations calculent
la MEME chose quand on leur demande la meme chose, ce qui est le seul
sens qu'une concordance puisse avoir.

Un troisieme point n'etait pas une convention mais un bug : j'avais
multiplie la variance par T alors qu'Omega est deja normalise par T.
Toutes les erreurs types etaient gonflees de sqrt(T), soit un facteur
sept a T = 50 — de quoi rendre chaque coefficient insignifiant.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyardl.cointegration import dols, fmols
from pyardl.utils import longrun_covariance_kernel

_HERE = Path(__file__).parent
_EXPECTED = json.loads((_HERE / "expected" / "spec08.json").read_text(encoding="utf-8"))
_TOL = _EXPECTED["_provenance"]["tolerance"]


@pytest.fixture(scope="module")
def denmark() -> pd.DataFrame:
    return pd.read_csv(_HERE / "data" / "spec08_denmark.csv")


@pytest.fixture(scope="module")
def lrv_series() -> np.ndarray:
    return np.loadtxt(_HERE / "data" / "spec08_lrv.csv", delimiter=",", skiprows=1)


@pytest.mark.parametrize(
    "kernel", ["bartlett", "parzen", "quadratic-spectral", "truncated"]
)
def test_longrun_covariance_matches_cointreg(
    lrv_series: np.ndarray, kernel: str
) -> None:
    """La brique transversale, sur les quatre noyaux.

    Elle alimente FMOLS, CCR et toute erreur type HAC de la
    bibliotheque : si elle derive, tout ce qui la consomme derive avec
    elle sans rien signaler."""
    ref = _EXPECTED["longrun_variance"][kernel]
    out = longrun_covariance_kernel(lrv_series, kernel=kernel, bandwidth=5)
    assert out.omega[0, 0] == pytest.approx(ref["omega_00"], abs=_TOL)
    assert out.omega[0, 1] == pytest.approx(ref["omega_01"], abs=_TOL)
    assert out.delta[0, 1] == pytest.approx(ref["delta_01"], abs=_TOL)


def test_omega_is_symmetric_and_delta_is_not(lrv_series: np.ndarray) -> None:
    """Verification structurelle qui separe les deux matrices.

    Omega est symetrique par construction ; Delta ne l'est PAS, et c'est
    precisement pourquoi sa transposition est une erreur possible qui ne
    se voit pas sur Omega."""
    out = longrun_covariance_kernel(lrv_series, bandwidth=5)
    assert out.omega == pytest.approx(out.omega.T, abs=1e-14)
    assert abs(out.delta[0, 1] - out.delta[1, 0]) > 1e-6


def test_omega_equals_delta_plus_delta_t_minus_sigma(lrv_series: np.ndarray) -> None:
    """L'identite qui lie les trois matrices, verifiee plutot que supposee."""
    out = longrun_covariance_kernel(lrv_series, bandwidth=5)
    assert out.omega == pytest.approx(out.delta + out.delta.T - out.sigma, abs=1e-14)


@pytest.mark.parametrize("name", ["LRY", "IBO", "IDE"])
def test_fmols_theta_matches_cointreg(denmark: pd.DataFrame, name: str) -> None:
    ref = _EXPECTED["fmols"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fmols(
            denmark["LRM"],
            denmark[["LRY", "IBO", "IDE"]],
            bandwidth=5,
            prewhiten=False,
        )
    assert res.longrun.loc[name, "theta"] == pytest.approx(ref["theta"], abs=_TOL)


@pytest.mark.parametrize("name", ["LRY", "IBO", "IDE"])
def test_fmols_standard_errors_match_cointreg(denmark: pd.DataFrame, name: str) -> None:
    """Le verrou du facteur T. Une erreur type gonflee de sqrt(T) reste
    finie, positive et du bon signe : rien d'interne ne la signale."""
    ref = _EXPECTED["fmols"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fmols(
            denmark["LRM"],
            denmark[["LRY", "IBO", "IDE"]],
            bandwidth=5,
            prewhiten=False,
        )
    assert res.longrun.loc[name, "se"] == pytest.approx(ref["se"], abs=_TOL)


@pytest.mark.parametrize("name", ["LRY", "IBO", "IDE"])
def test_dols_matches_cointreg(denmark: pd.DataFrame, name: str) -> None:
    ref = _EXPECTED["dols"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = dols(
            denmark["LRM"],
            denmark[["LRY", "IBO", "IDE"]],
            n_leads=2,
            n_lags=2,
            bandwidth=5,
            prewhiten=False,
        )
    assert res.longrun.loc[name, "theta"] == pytest.approx(ref["theta"], abs=_TOL)
    assert res.longrun.loc[name, "se"] == pytest.approx(ref["se"], abs=_TOL)


def test_the_bias_term_uses_the_full_sample(denmark: pd.DataFrame) -> None:
    """Verification par contraste de la convention 55/54.

    La version en T-1 est recalculee ici a la main. Elle doit donner un
    theta NETTEMENT different — sans quoi la convention n'aurait pas
    d'importance et ce test ne prouverait rien."""
    y = denmark["LRM"].to_numpy()
    x = denmark[["LRY", "IBO", "IDE"]].to_numpy()
    n_full = y.size
    static = np.column_stack([np.ones(n_full), x])
    beta0, *_ = np.linalg.lstsq(static, y, rcond=None)
    u = y - static @ beta0
    v = np.diff(x, axis=0)
    lrv = longrun_covariance_kernel(np.column_stack([u[1:], v]), bandwidth=5)
    inv_vv = np.linalg.pinv(lrv.omega[1:, 1:])
    omega_vu = lrv.omega[1:, 0:1]
    lam = (lrv.delta[1:, 0:1] - lrv.delta[1:, 1:] @ inv_vv @ omega_vu).ravel()
    y_plus = y[1:] - (v @ inv_vv @ omega_vu).ravel()
    xc = x[1:] - x[1:].mean(axis=0)
    yc = y_plus - y_plus.mean()
    gram = np.linalg.pinv(xc.T @ xc)

    with_full = gram @ (xc.T @ yc - n_full * lam)
    with_short = gram @ (xc.T @ yc - (n_full - 1) * lam)
    ref = np.array([_EXPECTED["fmols"][n]["theta"] for n in ("LRY", "IBO", "IDE")])

    assert with_full == pytest.approx(ref, abs=_TOL)
    assert np.abs(with_short - ref).max() > 1e-3


def test_prewhitening_changes_the_answer(denmark: pd.DataFrame) -> None:
    """Le preblanchiment n'est pas un reglage cosmetique.

    S'il ne changeait rien, l'activer par defaut n'aurait aucun effet sur
    la couverture — or il la fait passer de 89.6 % a 94.4 % a T = 400
    (voir validation/spec08_montecarlo.py). Ce test verifie qu'il mord
    bien sur ces donnees, sans quoi les deux resultats ci-dessus
    seraient la meme mesure repetee."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = fmols(
            denmark["LRM"],
            denmark[["LRY", "IBO", "IDE"]],
            bandwidth=5,
            prewhiten=False,
        )
        white = fmols(denmark["LRM"], denmark[["LRY", "IBO", "IDE"]], bandwidth=5)
    assert white.prewhitened
    assert not plain.prewhitened
    assert abs(white.longrun.loc["LRY", "se"] - plain.longrun.loc["LRY", "se"]) > 1e-4
