"""Bootstrap inference for the bounds test.

Rather than comparing a statistic against tabulated bounds — which
leaves an inconclusive zone whenever it falls between them — the
bootstrap builds the distribution of the statistic under the null
*conditionally on the data at hand*. The critical values are then
specific to the sample, and the verdict is binary.
"""

from pyardl.bootstrap.bounds import BootstrapBoundsResults, bootstrap_bounds_test
from pyardl.bootstrap.dgp import NullDGP, estimate_null_dgp, simulate_path
from pyardl.bootstrap.resample import resample_residuals

__all__ = [
    "BootstrapBoundsResults",
    "NullDGP",
    "bootstrap_bounds_test",
    "estimate_null_dgp",
    "resample_residuals",
    "simulate_path",
]
