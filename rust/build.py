"""Construit le noyau natif et le depose dans le paquet.

POURQUOI PAS MATURIN
--------------------
maturin veut etre le systeme de construction du paquet Python. Ici il ne
peut pas l'etre : `pyardl` est un paquet PUR PYTHON, construit par
hatchling, installable sans chaine Rust — et il doit le rester, parce
que le chemin NumPy est le chemin de reference et non un repli.

Ce script fait donc la seule chose necessaire : `cargo build --release`,
puis copie la bibliotheque partagee sous le nom que Python attend.
`pip install pyardl` continue de ne rien savoir de Rust ; qui veut le
noyau natif lance ce script.

    python rust/build.py            # construit et installe
    python rust/build.py --check    # verifie seulement l'etat actuel

Le nom du fichier produit n'est pas libre : Python cherche le symbole
`PyInit__rust` dans `pyardl/_rust.pyd`, et ce symbole vient du nom de la
fonction `#[pymodule]` du crate. Les deux doivent dire `_rust`.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "src" / "pyardl"

# Nom de la bibliotheque produite par cargo, et nom attendu par Python.
_ARTEFACTS = {
    "Windows": ("pyardl_rust.dll", "_rust.pyd"),
    "Darwin": ("libpyardl_rust.dylib", "_rust.so"),
    "Linux": ("libpyardl_rust.so", "_rust.so"),
}


def _names() -> tuple[str, str]:
    system = platform.system()
    if system not in _ARTEFACTS:
        raise SystemExit(f"Plateforme non prevue : {system!r}.")
    return _ARTEFACTS[system]


def check() -> int:
    """Le noyau est-il installe, et repond-il ?"""
    sys.path.insert(0, str(HERE.parent / "src"))
    try:
        from pyardl import backend
    except ImportError as exc:  # pragma: no cover - arbre casse
        print(f"pyardl introuvable : {exc}")
        return 1
    if backend.rust_available():
        print(f"noyau natif present, {backend.thread_count()} thread(s) rayon")
        return 0
    print("noyau natif absent — pyardl utilise le chemin NumPy")
    return 1


def build() -> int:
    source_name, target_name = _names()
    print(f"cargo build --release  ({platform.system()})")
    result = subprocess.run(
        ["cargo", "build", "--release"], cwd=HERE, check=False
    )
    if result.returncode != 0:
        return result.returncode

    built = HERE / "target" / "release" / source_name
    if not built.exists():  # pragma: no cover - cargo aurait echoue avant
        raise SystemExit(f"cargo a reussi mais {built} est absent.")
    destination = PACKAGE / target_name
    shutil.copy2(built, destination)
    print(f"copie -> {destination.relative_to(HERE.parent)}")
    return check()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="ne construit rien, rapporte l'etat du noyau natif",
    )
    args = parser.parse_args()
    return check() if args.check else build()


if __name__ == "__main__":
    raise SystemExit(main())
