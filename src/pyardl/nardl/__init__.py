"""Nonlinear ARDL: asymmetric long-run and short-run responses.

The framework of Shin, Yu & Greenwood-Nimmo (2014). A regressor is split
into the part built by its rises and the part built by its falls, and the
model is then estimated exactly as a linear ARDL — which is what makes
the whole apparatus of the library reusable here, rather than
reimplemented.
"""

from pyardl.nardl.decompose import decomposition_error, partial_sums
from pyardl.nardl.model import NARDL, NARDLBoundsResults, NARDLResults

__all__ = [
    "NARDL",
    "NARDLBoundsResults",
    "NARDLResults",
    "decomposition_error",
    "partial_sums",
]
