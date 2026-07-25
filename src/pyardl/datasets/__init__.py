"""Example datasets used in the documentation and in the test suite.

Both are classic ARDL applications, redistributed by the R ``ARDL``
package (Natsiopoulos & Tzeremes) and exported from it.

``denmark``
    Quarterly Danish data from Johansen & Juselius (1990), 1974Q1 to
    1987Q3. Columns: ``LRM`` (log real money M2), ``LRY`` (log real
    income), ``LPY`` (log deflator), ``IBO`` (bond rate), ``IDE``
    (deposit rate).

``pss2001``
    Quarterly UK data from the real-wage application of Pesaran, Shin &
    Smith (2001). Columns: ``w`` (log real wage), ``Prod`` (log
    productivity), ``UR`` (log unemployment rate), ``Wedge`` (tax
    wedge), ``Union`` (union density), ``D7475`` and ``D7579``
    (dummies).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"

__all__ = ["load_denmark", "load_pss2001"]


def load_denmark() -> pd.DataFrame:
    """Load the Danish money-demand dataset (Johansen & Juselius 1990).

    Returns
    -------
    pandas.DataFrame
        55 quarterly observations; see the module documentation for the
        column definitions.
    """
    return pd.read_csv(_DATA_DIR / "denmark.csv")


def load_pss2001() -> pd.DataFrame:
    """Load the UK real-wage dataset used by Pesaran, Shin & Smith (2001).

    Returns
    -------
    pandas.DataFrame
        112 quarterly observations; see the module documentation for the
        column definitions.
    """
    return pd.read_csv(_DATA_DIR / "pss2001.csv")
