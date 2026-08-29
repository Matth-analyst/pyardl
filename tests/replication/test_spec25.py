"""Spec 25 §4.4 — concordance croisee avec R dynamac (Jordan & Philips 2018).

`dynamac` 0.1.12 est l'implementation de reference de cette spec : c'est
le package qui accompagne l'article. Il tourne ici sur les donnees
danoises que pyardl embarque deja — AUCUNE donnee tierce n'entre dans la
comparaison, seulement une implementation tierce.

TROIS QUANTITES, TROIS STATUTS
------------------------------
La faute serait de leur donner la meme tolerance, ce qui reviendrait a
traiter du bruit Monte Carlo comme un desaccord numerique, ou l'inverse.

1. LES COEFFICIENTS sont exacts. `dynardl(..., ec = FALSE)` estime le
   MEME ARDL(3, {1, 3, 2}) en niveaux. Accord requis a la precision
   machine, mesure a 3.5e-14 sur les treize.

2. L'EQUILIBRE DE BASE est produit par Monte Carlo chez dynamac, pas
   chez pyardl (ou il est resolu algebriquement). Le script tourne trois
   graines pour MESURER ce bruit — 9.3e-05 — au lieu de le supposer, et
   la tolerance en decoule.

3. LE NIVEAU D'ARRIVEE porte le meme bruit, en plus gros : 5.0e-03
   d'ecart entre graines. La valeur exacte de pyardl tombe DANS
   l'intervalle des trois graines de dynamac, ce que le test verifie
   explicitement — c'est plus informatif qu'une tolerance absolue, qui
   ne dirait pas si l'ecart est du bon ordre.

CE QUI N'EST PAS COMPARE, ET POURQUOI CE N'EST PAS UN RENONCEMENT
-----------------------------------------------------------------
Les bandes. dynamac tient le niveau d'avant-choc quasiment fixe (bande a
95 % large de 0.012) ; pyardl fait partir chaque tirage de SON propre
equilibre, donc sa bande sur le NIVEAU porte aussi la dispersion
d'echantillonnage de y*. Ce sont deux questions differentes sur le
niveau, et aucune des deux reponses n'est fausse. Les comparer terme a
terme produirait un desaccord qui ne signifie rien.

La bande sur la REPONSE, elle, est une difference appariee : l'equilibre
commun s'y annule, et la divergence ci-dessus ne la touche pas.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyardl.core.ardl import ARDL
from pyardl.datasets import load_denmark

_HERE = Path(__file__).parent
_EXPECTED = json.loads((_HERE / "expected" / "spec25.json").read_text(encoding="utf-8"))
_PROV = _EXPECTED["_provenance"]

ORDER = (3, {"LRY": 1, "IBO": 3, "IDE": 2})


@pytest.fixture(scope="module")
def fit():
    d = load_denmark()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARDL(d["LRM"], d[["LRY", "IBO", "IDE"]], order=ORDER, det="const").fit()


@pytest.fixture(scope="module")
def sim(fit):
    return fit.dynardl_simulate("IBO", size=1.0, t0=10, horizon=61, r=200, seed=1)


@pytest.mark.external
def test_the_regression_is_the_same_one(fit) -> None:
    """La partie EXACTE : meme modele, donc memes coefficients.

    Si celle-ci derive, la comparaison des trajectoires ne veut plus
    rien dire — on comparerait deux modeles differents.
    """
    tol = _PROV["tolerance_coefficients"]
    name_map = _EXPECTED["name_map"]
    for r_name, value in _EXPECTED["coefficients"].items():
        assert float(fit.params[name_map[r_name]]) == pytest.approx(value, abs=tol)


@pytest.mark.external
def test_baseline_equilibrium_matches(sim) -> None:
    """pyardl resout y* algebriquement, dynamac le simule."""
    ref = np.asarray(_EXPECTED["baseline_central"], dtype=float)
    assert float(np.max(np.abs(ref - sim.equilibrium))) < _PROV["tolerance_baseline"]


@pytest.mark.external
def test_final_level_falls_inside_dynamac_seed_spread(sim) -> None:
    """La limite exacte de pyardl tombe dans le bruit de dynamac.

    Trois graines encadrent la valeur ; c'est un enonce plus fort qu'un
    ecart sous une tolerance, parce qu'il dit que la difference est du
    MEME ORDRE que la simulation de dynamac, pas seulement petite.
    """
    ref = np.asarray(_EXPECTED["final_central"], dtype=float)
    final = float(sim.equilibrium + sim.summary_df[("response", "point")].iloc[61])
    assert ref.min() <= final <= ref.max()
    assert abs(final - float(ref.mean())) < _PROV["tolerance_final"]


@pytest.mark.external
def test_the_shock_moves_y_by_the_longrun_coefficient(fit, sim) -> None:
    """Le pont interne : la trajectoire arrive ou l'algebre l'annonce.

    Ce test n'a pas besoin de dynamac. Il est ici parce que c'est lui
    qui rend la comparaison externe interpretable : les deux
    implementations convergent vers la meme chose parce que cette chose
    est le theta de long terme, pas par coincidence numerique.
    """
    theta = float(fit.longrun.loc["IBO", "theta"])
    # A h = 61 il reste 1.3e-06 de decroissance geometrique a courir sur
    # ce modele persistant ; a h = 200 la recursion a fini d'arriver.
    # Les deux chiffres sont dans le test parce que le premier dit ou en
    # est la convergence et le second qu'elle a bien lieu.
    assert float(sim.summary_df[("response", "point")].iloc[61]) == pytest.approx(
        theta, abs=1e-5
    )
    far = fit.dynardl_simulate("IBO", size=1.0, t0=10, horizon=200, r=10, seed=1)
    assert float(far.summary_df[("response", "point")].iloc[200]) == pytest.approx(
        theta, abs=1e-10
    )
    ref = np.asarray(_EXPECTED["final_central"], dtype=float)
    assert float(ref.mean() - np.mean(_EXPECTED["baseline_central"])) == pytest.approx(
        theta, abs=_PROV["tolerance_final"]
    )
