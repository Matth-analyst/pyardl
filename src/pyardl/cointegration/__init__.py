"""Classical cointegration tests and efficient long-run estimators.

Provided for comparison with the bounds test, and because much of the
applied literature reports them. The bounds test remains the recommended
route: it does not require every series to be I(1).

The efficient estimators (DOLS, FMOLS, CCR) are the robustness block
that referees ask for next to an ARDL long run: static OLS on
cointegrated data is consistent but its inference is not, and these
three repair that by three different routes.
"""

from pyardl.cointegration.efficient import (
    EfficientLongRunResults,
    ccr,
    compare_longrun,
    default_dols_lags,
    dols,
    fmols,
)
from pyardl.cointegration.engle_granger import EGResults, engle_granger
from pyardl.cointegration.johansen import (
    JohansenResults,
    check_no_cointegration_among_x,
    johansen,
)

__all__ = [
    "EGResults",
    "EfficientLongRunResults",
    "JohansenResults",
    "ccr",
    "check_no_cointegration_among_x",
    "compare_longrun",
    "default_dols_lags",
    "dols",
    "engle_granger",
    "fmols",
    "johansen",
]
