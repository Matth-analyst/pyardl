"""Data generation for the library's Monte Carlo studies.

One generator, used everywhere. A disagreement between two validation
studies should be a disagreement about estimators, never about the data.

The module also holds the *interpretation* layer of Jordan and Philips
(2018): :func:`~pyardl.simulate.dynardl.dynardl_simulate` runs a fitted
ARDL forward under a counterfactual shock, which is what turns a
coefficient table into a statement about the world.
"""

from pyardl.simulate.dynardl import DynardlSimulation, dynardl_simulate
from pyardl.simulate.vecm import VECMSimulation, degenerate_system, vecm_ardl

__all__ = [
    "DynardlSimulation",
    "VECMSimulation",
    "degenerate_system",
    "dynardl_simulate",
    "vecm_ardl",
]
