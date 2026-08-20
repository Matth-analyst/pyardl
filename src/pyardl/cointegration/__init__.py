"""Classical cointegration tests.

Provided for comparison with the bounds test, and because much of the
applied literature reports them. The bounds test remains the recommended
route: it does not require every series to be I(1).
"""

from pyardl.cointegration.engle_granger import EGResults, engle_granger
from pyardl.cointegration.johansen import (
    JohansenResults,
    check_no_cointegration_among_x,
    johansen,
)

__all__ = [
    "EGResults",
    "JohansenResults",
    "check_no_cointegration_among_x",
    "engle_granger",
    "johansen",
]
