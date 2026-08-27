"""Heterogeneous dynamic panels (specs 22-24).

The organising idea, from Pesaran & Smith (1995): a heterogeneous
dynamic panel is **N time-series problems plus an aggregation step**.
Pooling the dynamics when they differ is not a loss of efficiency but a
loss of consistency, and no amount of data repairs it. So the panel
estimators here own no regression of their own — they orchestrate
:class:`pyardl.ARDL` over individuals and aggregate what comes back.
"""

from pyardl.panel.container import PanelData, PanelUnit, panel_from_frame
from pyardl.panel.mg import MeanGroup, MeanGroupResults

__all__ = [
    "MeanGroup",
    "MeanGroupResults",
    "PanelData",
    "PanelUnit",
    "panel_from_frame",
]
