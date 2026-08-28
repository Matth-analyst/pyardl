"""Heterogeneous dynamic panels (specs 22-24).

The organising idea, from Pesaran & Smith (1995): a heterogeneous
dynamic panel is **N time-series problems plus an aggregation step**.
Pooling the dynamics when they differ is not a loss of efficiency but a
loss of consistency, and no amount of data repairs it. So the panel
estimators here own no regression of their own — they orchestrate
:class:`pyardl.ARDL` over individuals and aggregate what comes back.
"""

from pyardl.panel.container import PanelData, PanelUnit, panel_from_frame
from pyardl.panel.crosssection import (
    CDResult,
    cd_test,
    cross_section_averages,
    default_cs_lags,
)
from pyardl.panel.csardl import CSARDL, CSDL, CSARDLResults, CSDLResults
from pyardl.panel.mg import MeanGroup, MeanGroupResults
from pyardl.panel.pmg import (
    DFE,
    PMG,
    DFEResults,
    HausmanResult,
    PMGResults,
    compare,
    hausman,
)

__all__ = [
    "CSDL",
    "CSARDL",
    "CDResult",
    "CSDLResults",
    "CSARDLResults",
    "DFE",
    "PMG",
    "DFEResults",
    "HausmanResult",
    "MeanGroup",
    "MeanGroupResults",
    "PMGResults",
    "PanelData",
    "PanelUnit",
    "cd_test",
    "compare",
    "cross_section_averages",
    "default_cs_lags",
    "hausman",
    "panel_from_frame",
]
