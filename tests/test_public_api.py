"""La surface publique, verrouillee — relecture transversale avant v0.5.

Trois invariants que rien d'autre ne verifie, et dont la violation est
silencieuse dans les trois cas.

1. `import pyardl` doit rester BON MARCHE. Les re-exports du premier
   niveau sont paresseux (PEP 562) parce qu'importer la machinerie ARDL
   tire statsmodels, soit environ 3,5 secondes. Lier les noms
   eagerly ferait payer ce cout a tout le monde, y compris a qui ne veut
   que `__version__`. Une regression ici ne casse aucun test
   fonctionnel : elle rend juste la bibliotheque lente a charger, et
   personne ne s'en apercoit.

2. Chaque nom annonce doit se resoudre. Une table nom -> module se
   desynchronise des qu'on renomme quelque chose, et `__getattr__` ne
   le dira qu'au premier acces d'un utilisateur.

3. Les objets Resultats sont immuables. C'est le contrat annonce par
   l'architecture ; `ARDLResults` et `BoundsTestResults` y avaient
   echappe jusqu'a cette relecture, seuls parmi vingt-sept.
"""

from __future__ import annotations

import dataclasses
import importlib
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import pyardl


class TestLazySurface:
    def test_importing_pyardl_does_not_pull_statsmodels(self) -> None:
        """Le test qui donne son sens a la paresse.

        Il tourne dans un interpreteur NEUF : dans la session pytest,
        statsmodels est deja importe par les autres tests et la question
        ne se poserait plus.
        """
        code = (
            "import sys; import pyardl; "
            "print('statsmodels' in sys.modules, 'scipy' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False False"

    def test_every_advertised_name_resolves(self) -> None:
        for name in pyardl.__all__:
            assert hasattr(pyardl, name), name

    def test_the_export_table_points_at_real_modules(self) -> None:
        """Et le nom doit exister DANS le module annonce, pas ailleurs."""
        for name, module in pyardl._EXPORTS.items():
            assert hasattr(importlib.import_module(module), name), (name, module)

    def test_the_lazy_object_is_the_same_object(self) -> None:
        from pyardl.core import ARDL

        assert pyardl.ARDL is ARDL

    def test_an_unknown_name_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute"):
            pyardl.definitely_not_exported  # noqa: B018

    def test_dir_matches_all(self) -> None:
        assert dir(pyardl) == sorted(pyardl.__all__)

    def test_version_is_a_release_string(self) -> None:
        parts = pyardl.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestResultsAreImmutable:
    """Un objet Resultats se lit, ne se retouche pas.

    Une valeur reecrite apres coup ne serait signalee nulle part : le
    `summary()` afficherait le nouveau chiffre avec le meme aplomb que
    l'ancien.
    """

    @staticmethod
    def _fitted():
        rng = np.random.default_rng(0)
        x = pd.DataFrame({"x": rng.normal(size=120).cumsum()})
        y = pd.Series(rng.normal(size=120) + 0.5 * x["x"], name="y")
        return y, x

    def test_ardl_results(self) -> None:
        from pyardl.core import ARDL

        y, x = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARDL(y, x, order=(1, 1)).fit()
        assert dataclasses.fields(res)  # c'est bien une dataclass
        with pytest.raises(dataclasses.FrozenInstanceError):
            res._params = np.zeros(4)  # type: ignore[misc]

    def test_bounds_test_results(self) -> None:
        from pyardl.bounds import bounds_test

        y, x = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(y, x, case=3, order=(1, 1))
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.f_stat = 0.0  # type: ignore[misc]

    def test_replace_still_works_on_frozen_results(self) -> None:
        """Geler n'interdit pas de DERIVER un resultat modifie.

        `dataclasses.replace` reste disponible, ce dont les tests qui
        fabriquent un cas limite ont besoin. Ce qui devient impossible,
        c'est la mutation en place d'un objet qu'on croit inchange.
        """
        from pyardl.core import ARDL

        y, x = self._fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARDL(y, x, order=(1, 1)).fit()
        other = dataclasses.replace(res, _params=np.zeros_like(res._params))
        assert other is not res
        assert float(np.max(np.abs(other._params))) == 0.0
        assert float(np.max(np.abs(res._params))) > 0.0


def test_every_subpackage_exports_something() -> None:
    """`pyardl.core` etait le seul `__init__` vide de la bibliotheque.

    Consequence : `ARDL`, la classe centrale, etait la SEULE a exiger un
    chemin profond (`from pyardl.core.ardl import ARDL`) quand tout le
    reste s'importait depuis son paquet. Une incoherence qu'aucun test
    ne pouvait signaler, puisque le chemin profond fonctionne.
    """
    subpackages = [
        "bootstrap",
        "bounds",
        "cointegration",
        "core",
        "critical_values",
        "datasets",
        "diagnostics",
        "distributed_lags",
        "fourier",
        "nardl",
        "panel",
        "qardl",
        "simulate",
        "unified",
        "unitroot",
    ]
    for name in subpackages:
        module = importlib.import_module(f"pyardl.{name}")
        exported = getattr(module, "__all__", [])
        assert exported, f"pyardl.{name} exports nothing"
        for symbol in exported:
            assert hasattr(module, symbol), f"pyardl.{name}.{symbol}"
