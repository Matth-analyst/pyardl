"""The ARDL model itself, and the algebra that connects it to the ECM.

This is the one subpackage everything else is built on: the bounds
tests, the bootstrap, NARDL, QARDL, the Fourier variants and the panel
estimators all consume `ARDL` or the transforms below.

The exact reparameterisation `ardl_to_ecm` / `ecm_to_ardl` is the piece
worth knowing about. It is not an approximation: the same data, the same
residuals and the same fit, written in terms of the adjustment speed and
the level coefficients instead of the raw lag polynomials. The mapping
of the short-run coefficients is the library's most error-prone corner,
which is why the equivalence of the two regressions is checked to 1e-10
in the test suite rather than assumed.
"""

from pyardl.core.ardl import ARDL, ARDLOrderSelection, ARDLResults, GETSResults
from pyardl.core.restrictions import LongRunRestrictionResults, longrun_restriction
from pyardl.core.transforms import (
    ARDLParams,
    ECMParams,
    ardl_to_ecm,
    ecm_to_ardl,
    half_life,
    longrun_coefs,
    longrun_covariance,
    speed_of_adjustment,
)

__all__ = [
    "ARDL",
    "ARDLOrderSelection",
    "ARDLParams",
    "ARDLResults",
    "ECMParams",
    "GETSResults",
    "LongRunRestrictionResults",
    "ardl_to_ecm",
    "ecm_to_ardl",
    "half_life",
    "longrun_coefs",
    "longrun_covariance",
    "longrun_restriction",
    "speed_of_adjustment",
]
