"""MIDAS-ARDL: mixed sampling frequencies (Ghysels, Santa-Clara & Valkanov 2004).

Keeps a high-frequency regressor at its native frequency instead of
aggregating it down to the dependent variable's frequency, modelling the
aggregation weights as a low-dimensional parametric function reused from
the Almon-exponential idea (:mod:`pyardl.distributed_lags.almon`).
"""

from pyardl.midas.model import MIDASARDL, MIDASARDLResults, midas_weights

__all__ = ["MIDASARDL", "MIDASARDLResults", "midas_weights"]
