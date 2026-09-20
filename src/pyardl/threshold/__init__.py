"""Threshold regression: an unknown regime split on a transition variable.

Distinct from NARDL (:mod:`pyardl.nardl`), which splits a regressor by
the sign of its own change with a threshold fixed at zero. Here the
regime is a function of a (possibly different) transition variable
against a threshold estimated from the data — Hansen (1999, 2000).
"""

from pyardl.threshold.hansen import ThresholdARDLResults, threshold_ardl

__all__ = ["ThresholdARDLResults", "threshold_ardl"]
