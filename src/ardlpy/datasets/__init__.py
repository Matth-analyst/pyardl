"""Jeux de données d'exemple et de réplication (specs 04, 06, 10).

Provenance
----------
- ``denmark`` : données trimestrielles danoises de Johansen & Juselius
  (1990), redistribuées par le package R ARDL (Natsiopoulos & Tzeremes),
  exportées le 2026-07-07 (ARDL 0.2.5). Colonnes : LRM (log monnaie
  réelle M2), LRY (log revenu réel), LPY (log déflateur), IBO (taux
  obligataire), IDE (taux de dépôt). 1974Q1-1987Q3.
- ``pss2001`` : données trimestrielles UK de l'application salaires
  réels de Pesaran, Shin & Smith (2001, §6), redistribuées par le
  package R ARDL, exportées le 2026-07-07. Colonnes : w (log salaire
  réel), Prod (log productivité), UR (log taux de chômage), Wedge (coin
  fiscal), Union (taux de syndicalisation), D7475/D7579 (dummies).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"

__all__ = ["load_denmark", "load_pss2001"]


def load_denmark() -> pd.DataFrame:
    """Données danoises de Johansen & Juselius (1990) — voir module."""
    return pd.read_csv(_DATA_DIR / "denmark.csv")


def load_pss2001() -> pd.DataFrame:
    """Données salaires UK de PSS (2001, §6) — voir module."""
    return pd.read_csv(_DATA_DIR / "pss2001.csv")
