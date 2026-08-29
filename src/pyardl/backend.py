r"""Choix du moteur de calcul : NumPy, ou le noyau natif optionnel.

**Le chemin NumPy est la reference, pas un repli.** Il est le defaut,
il est teste partout, et le noyau Rust est verifie CONTRE lui — jamais
l'inverse. Un backend natif qui deviendrait la reference ferait de
l'accord entre les deux une tautologie.

CE QUE LE NOYAU NATIF COUVRE, ET CE QU'IL NE COUVRE PAS
-------------------------------------------------------
Une seule fonction : :func:`pyardl.bootstrap.dgp.simulate_paths`, la
recursion qui regenere les trajectoires sous l'hypothese nulle. Le choix
vient d'un profilage, pas d'une intuition — a T = 1000, B = 9999, k = 3,
elle pese 27 % du temps, contre 36 % pour ``numpy.linalg.qr`` qui est
deja du LAPACK et n'a rien a gagner a etre reecrit.

**Le gain de bout en bout est donc borne par Amdahl a environ 1,4x.**
C'est peu, et le dire est plus utile que de l'esperer : le vrai gout du
bootstrap est la decomposition QR, et l'optimisation suivante sera
algorithmique.

La decomposition NARDL passe un rappel Python (``expand``) appele a
chaque periode. Le faire traverser la frontiere mille fois couterait
plus que la boucle n'economise, donc ce cas retombe sur NumPy —
silencieusement, parce que c'est une decision de performance et pas un
avertissement methodologique.

L'EQUIVALENCE EST EXACTE, PAS DISTRIBUTIONNELLE
-----------------------------------------------
Les innovations sont tirees cote Python, par un
``numpy.random.Generator`` a graine explicite, puis PASSEES au noyau.
Celui-ci ne tire rien. Les deux backends voient donc exactement les
memes nombres et doivent produire exactement les memes trajectoires : le
test de conformite est une egalite a 1e-12, pas un Kolmogorov-Smirnov.

Le KS reste dans la suite de tests, applique aux statistiques de bout en
bout, parce qu'il repond a une autre question : que la substitution ne
deplace pas la distribution des decisions. Mais c'est l'egalite exacte
qui est le verrou. Un KS a 2000 points ne distingue pas deux lois qui
different de 1e-9 ; il aurait laisse passer une erreur de signe sur un
coefficient rarement actif.

Examples
--------
>>> from pyardl import backend
>>> backend.resolve("numpy")
'numpy'
>>> backend.resolve("auto") in ("numpy", "rust")
True
"""

from __future__ import annotations

from typing import Any, Literal

__all__ = [
    "BACKENDS",
    "rust_available",
    "resolve",
    "thread_count",
    "why_unavailable",
]

Backend = Literal["numpy", "rust", "auto"]
BACKENDS: tuple[str, ...] = ("numpy", "rust", "auto")

_MODULE: Any | None = None
_REASON: str | None = None
_PROBED = False


def _probe() -> None:
    """Tente l'import une seule fois, et retient POURQUOI il a echoue."""
    global _MODULE, _REASON, _PROBED
    if _PROBED:
        return
    _PROBED = True
    # Import par nom plutot que `from pyardl import _rust` : le verdict
    # de mypy ne doit pas dependre de la presence d'un binaire. Avec
    # l'import statique, une machine ou le noyau est compile rapporte
    # « type: ignore inutile » et une machine sans lui rapporte l'inverse
    # — le job `types` de la CI et le poste de travail se contrediraient.
    import importlib

    try:
        _MODULE = importlib.import_module("pyardl._rust")
    except ImportError as exc:
        _MODULE = None
        _REASON = str(exc)
    else:
        _REASON = None


def rust_available() -> bool:
    """Le noyau natif est-il compile et importable ?"""
    _probe()
    return _MODULE is not None


def why_unavailable() -> str | None:
    """Le message d'import, ou ``None`` si le noyau est la.

    Rapporter la raison plutot qu'un simple ``False`` evite l'heure
    perdue a chercher pourquoi ``--backend rust`` ne change rien : une
    bibliotheque construite pour une autre version de Python echoue avec
    un message tres different d'un fichier absent.
    """
    _probe()
    return _REASON


def module() -> Any:
    """Le module natif, ou une erreur qui dit comment l'obtenir."""
    _probe()
    if _MODULE is None:
        raise ImportError(
            f"The native kernel is not available ({_REASON}). Build it with "
            "`python rust/build.py`, or use backend='numpy' (the default, "
            "and the reference implementation)."
        )
    return _MODULE


def thread_count() -> int:
    """Nombre de threads que le noyau natif utilisera."""
    return int(module().thread_count())


def resolve(backend: str) -> str:
    """Traduit le choix de l'appelant en backend effectivement utilise.

    ``"auto"`` prend le noyau natif s'il est la. ``"rust"`` demande
    explicitement le noyau et **echoue** s'il manque, au lieu de retomber
    en silence sur NumPy : quelqu'un qui mesure une acceleration doit
    savoir laquelle des deux implementations il vient de chronometrer.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}.")
    if backend == "numpy":
        return "numpy"
    if backend == "rust":
        module()  # leve avec le mode d'emploi si absent
        return "rust"
    return "rust" if rust_available() else "numpy"
