"""Distributed lag models: the root of the ARDL genealogy.

Before the ARDL, the question was how to fit an effect spread over many
periods without estimating one free coefficient per period. Two answers
survived, and both are here:

- Koyck (1954) assumes the weights decline **geometrically**, which
  collapses an infinite lag into one parameter — at the cost of an
  endogenous regressor that makes least squares inconsistent.
- Almon (1965) assumes the weights lie on a **polynomial** in the lag
  index, which keeps the lag finite and the regression exogenous.

Both are special cases of the ARDL the rest of the library estimates,
and both results objects carry a bridge to it (``to_ardl``, ``to_fdl``)
so the restriction can be seen for what it is: a restriction, testable
against the unrestricted model rather than assumed.
"""

from pyardl.distributed_lags.almon import AlmonModel, AlmonResults
from pyardl.distributed_lags.koyck import KoyckModel, KoyckResults

PDL = AlmonModel

__all__ = [
    "PDL",
    "AlmonModel",
    "AlmonResults",
    "KoyckModel",
    "KoyckResults",
]
