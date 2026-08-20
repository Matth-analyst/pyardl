"""Data generation for the library's Monte Carlo studies.

One generator, used everywhere. A disagreement between two validation
studies should be a disagreement about estimators, never about the data.
"""

from pyardl.simulate.vecm import VECMSimulation, degenerate_system, vecm_ardl

__all__ = ["VECMSimulation", "degenerate_system", "vecm_ardl"]
