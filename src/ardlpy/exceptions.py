"""Classes d'avertissement méthodologique dédiées.

Toute limitation méthodologique (cas dégénéré, échantillon trop petit,
instrument faible, non-convergence...) doit être signalée via une sous-classe
de :class:`ArdlpyMethodologyWarning`, jamais via ``UserWarning`` nu, afin que
l'utilisateur puisse filtrer ces avertissements spécifiquement
(``warnings.filterwarnings``) et que les tests puissent les cibler
(``pytest.warns``).
"""

from __future__ import annotations


class ArdlpyMethodologyWarning(UserWarning):
    """Classe de base pour tout avertissement méthodologique émis par ardlpy."""


class DegenerateCaseWarning(ArdlpyMethodologyWarning):
    """Cas dégénéré : absence de force de rappel (lambda proche de 0) ou de
    convergence vers l'équilibre de long terme (lambda hors de ]-1, 0[).

    Les quantités de long terme (theta, half-life) ne sont alors pas
    définies statistiquement ; cf. specs 14-15 (dégénérescences) pour la
    classification complète.
    """
