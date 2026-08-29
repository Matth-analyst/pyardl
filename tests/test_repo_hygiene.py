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
import re
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


def _code_tokens(path: Path) -> str:
    """Le source d'un fichier, commentaires et chaines RETIRES.

    Un garde-fou qui cherche une chaine brute ne distingue pas un appel
    d'un commentaire qui explique pourquoi on ne fait pas cet appel. Les
    deux fichiers corriges le 2026-08-29 contiennent justement une telle
    explication : une recherche naive les aurait signales, et le
    garde-fou aurait ete desactive dans la semaine.
    """
    import tokenize

    kept: list[str] = []
    with path.open("rb") as handle:
        try:
            for token in tokenize.tokenize(handle.readline):
                if token.type not in (tokenize.COMMENT, tokenize.STRING):
                    kept.append(token.string)
        except (tokenize.TokenError, SyntaxError):  # pragma: no cover
            return path.read_text(encoding="utf-8")
    return " ".join(kept)


class TestTheDeclaredFloorsAreRespected:
    """Le code ne doit pas utiliser d'API plus recente que ses planchers.

    `pyproject.toml` annonce `pandas>=2.1`. Utiliser une API apparue en
    2.2 rend cette borne mensongere : le paquet s'installe sur 2.1, et
    casse a l'execution.

    Le job `floors` de la CI est le garde-fou reel — il installe
    vraiment les versions plancher et lance toute la suite. Ce test-ci
    est son avant-poste local : il coute une milliseconde et signale la
    faute au moment ou on l'ecrit, pas vingt minutes plus tard dans un
    log de CI.

    Il ne remplace pas le job : une API peut changer de COMPORTEMENT
    sans changer de nom, et seul un vrai pandas 2.1 le dirait.
    """

    #: nom d'API -> version qui l'a introduite. La liste n'a pas
    #: vocation a etre exhaustive : elle retient ce qui a deja casse une
    #: fois, ce qui est le seul critere qui garde une denylist honnete.
    _TOO_RECENT = {
        # `DataFrameGroupBy.apply(..., include_groups=...)`, pandas 2.2.
        # Sur 2.1 l'argument n'est pas ignore : il est transmis a la
        # fonction appliquee, qui leve un TypeError. Attrape par le job
        # `floors` le 2026-08-29 sur `cross_section_averages` et sur la
        # replication de la spec 24. Le contournement est de selectionner
        # les colonnes AVANT `apply`, ce qui exclut deja la cle de groupe.
        "include_groups": ("pandas", "2.2"),
    }

    def _floor(self, package: str) -> str:
        import tomllib

        data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        for spec in data["project"]["dependencies"]:
            name = re.split(r"[<>=!~ ]", spec, maxsplit=1)[0]
            if name == package:
                match = re.search(r">=\s*([0-9.]+)", spec)
                assert match, f"pas de plancher declare pour {package}"
                return match.group(1)
        raise AssertionError(f"{package} absent des dependances")

    @pytest.mark.parametrize("api", sorted(_TOO_RECENT))
    def test_no_api_newer_than_its_floor(self, api: str) -> None:
        package, introduced = self._TOO_RECENT[api]
        floor = self._floor(package)
        if tuple(int(p) for p in floor.split(".")[:2]) >= tuple(
            int(p) for p in introduced.split(".")[:2]
        ):
            pytest.skip(
                f"le plancher {package}>={floor} couvre desormais {api} "
                f"(introduit en {introduced}) : la garde ne s'applique plus"
            )
        offenders = [
            str(path.relative_to(_ROOT))
            for path in list((_ROOT / "src").rglob("*.py"))
            + list((_ROOT / "tests").rglob("*.py"))
            if api in _code_tokens(path)
        ]
        assert not offenders, (
            f"{api} a ete introduit dans {package} {introduced}, mais le "
            f"plancher declare est {floor}. Fichiers concernes : {offenders}"
        )

    def test_the_guard_reads_code_and_not_prose(self, tmp_path) -> None:
        """Il doit dire OUI sur un appel et NON sur un commentaire.

        Une recherche de chaine naive echoue ici : les deux fichiers
        corriges PARLENT de `include_groups` dans un commentaire pour
        expliquer pourquoi ils ne l'utilisent pas. Un garde-fou qui les
        signalerait serait vite desactive, et c'est ainsi que meurent
        les garde-fous.

        Sans ce test, une faute de frappe dans le nom de l'API rendrait
        le controle vert pour toujours.
        """
        offender = tmp_path / "offender.py"
        offender.write_text(
            "res = grouped.apply(f, include_groups=False)" + chr(10),
            encoding="utf-8",
        )
        assert "include_groups" in _code_tokens(offender)

        innocent = tmp_path / "innocent.py"
        innocent.write_text(
            chr(10).join(
                [
                    '"""On evite include_groups : pandas 2.2 seulement."""',
                    "# include_groups casserait le plancher",
                    "res = grouped[cols].apply(f)",
                ]
            ),
            encoding="utf-8",
        )
        assert "include_groups" not in _code_tokens(innocent)
