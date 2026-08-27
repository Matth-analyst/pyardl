"""Garde-fou : un test ne doit jamais dependre d'un fichier non versionne.

POURQUOI CE FICHIER EXISTE
--------------------------
Le dossier `validation/` est gitignore : il contient des sorties de
Monte Carlo, des logs et des jeux regeneres localement. Un test de
non-regression qui lit un fichier depuis ce dossier passe donc en local
— ou le fichier existe — et echoue en CI, ou il n'existe pas.

Ce n'est pas hypothetique : c'est arrive deux fois. Une fois en silence
(spec 22, ou le test se serait transforme en `skip` sans que personne ne
le remarque), une fois bruyamment (spec 17 et 18, `FileNotFoundError` en
CI). Les deux ont ete corrigees en versionnant la donnee AVEC le test,
dans `tests/replication/data/`.

La regle qui en decoule, et que ce module fait respecter :

  toute donnee dont un test a besoin est versionnee, ou bien son absence
  fait SAUTER le test avec un motif nomme — jamais planter, jamais
  passer inapercu.

Le cas restant legitime est celui d'une donnee tierce qu'on n'a pas le
droit de redistribuer (le jeu `fod` du package R nardl, GPL-3). Il est
alors garde derriere un `skipif` explicite, et ce module verifie que
c'est bien le cas plutot que de l'interdire.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TESTS = _ROOT / "tests"


def _test_modules() -> list[Path]:
    return sorted(p for p in _TESTS.rglob("test_*.py") if "__pycache__" not in p.parts)


def _is_tracked(path: Path) -> bool:
    """Whether git has this file under version control."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _module_paths_referencing(module: Path, needle: str) -> list[str]:
    """Module-level string literals that look like a path into ``needle``."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == needle
        ):
            found.append(node.value)
    return found


class TestNoTestDependsOnAnUntrackedFile:
    def test_every_shipped_test_datafile_is_tracked(self) -> None:
        """Les jeux poses dans tests/**/data/ doivent etre versionnes.

        Un fichier depose la mais oublie par git redonnerait exactement
        le bug d'origine, en plus discret : il serait a cote du test, ce
        qui inspire confiance, et absent du depot.
        """
        data_files = [
            p
            for p in _TESTS.rglob("*")
            if p.is_file() and p.parent.name == "data" and p.suffix in {".csv", ".json"}
        ]
        assert data_files, "aucun fichier de donnees de test trouve"
        untracked = [
            str(p.relative_to(_ROOT)) for p in data_files if not _is_tracked(p)
        ]
        assert not untracked, (
            f"Fichiers de donnees de test non versionnes : {untracked}. "
            "Ils existent en local et manqueront en CI."
        )

    def test_every_expected_json_is_tracked(self) -> None:
        expected = sorted((_TESTS / "replication" / "expected").glob("*.json"))
        assert expected, "aucun fichier de valeurs attendues trouve"
        untracked = [str(p.relative_to(_ROOT)) for p in expected if not _is_tracked(p)]
        assert not untracked, f"Valeurs attendues non versionnees : {untracked}."

    def test_modules_reading_from_validation_guard_the_absence(self) -> None:
        """Un test qui lit dans validation/ doit porter un skipif.

        `validation/` est gitignore. Lire dedans est tolere UNIQUEMENT
        pour une donnee tierce non redistribuable, et alors le test doit
        sauter proprement quand elle manque au lieu de planter.
        """
        offenders: list[str] = []
        for module in _test_modules():
            text = module.read_text(encoding="utf-8")
            if not _module_paths_referencing(module, "validation"):
                continue
            # Le module construit un chemin vers validation/ : il doit
            # aussi contenir un garde d'absence.
            if "skipif" not in text:
                offenders.append(str(module.relative_to(_ROOT)))
        assert not offenders, (
            f"Ces modules lisent dans validation/ (gitignore) sans garde "
            f"d'absence : {offenders}. Soit la donnee est la notre et doit "
            "etre versionnee dans tests/**/data/, soit elle est tierce et "
            "son absence doit faire SAUTER le test avec un motif nomme."
        )


class TestTheGuardItselfWorks:
    """Un garde-fou qu'on ne teste pas est une decoration."""

    def test_tracked_detection_is_not_vacuous(self) -> None:
        """`_is_tracked` doit savoir dire OUI et NON.

        Un detecteur qui repond toujours vrai ferait passer les tests
        ci-dessus quoi qu'il arrive ; un detecteur qui repond toujours
        faux les ferait tous echouer. Les deux sens sont verifies, sur
        un fichier dont le suivi ne fait aucun doute (pyproject.toml).
        """
        assert _is_tracked(_ROOT / "pyproject.toml")
        assert not _is_tracked(_ROOT / "un-fichier-qui-nexiste-pas.csv")

    def test_a_gitignored_path_is_reported_untracked(self) -> None:
        """Verification sur le dossier reellement gitignore du projet."""
        candidate = _ROOT / "validation" / "external" / "spec17_fod.csv"
        if not candidate.exists():
            pytest.skip("fichier regenere localement, absent ici")
        assert not _is_tracked(candidate)
