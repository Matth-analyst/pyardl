"""Modern unit-root pre-tests.

Step zero of any bounds-test workflow: the bounds test tolerates a
mixture of I(0) and I(1) regressors but is invalid if any series is
I(2), and it cannot detect that on its own.

The tests exposed here — DF-GLS and the four M statistics — dominate the
classical ADF on both power and size, and are what current applied work
is expected to report.
"""

from pyardl.unitroot.ers import DFGLSResults, dfgls
from pyardl.unitroot.gls import (
    CBAR,
    ADFRegression,
    adf_regression,
    gls_detrend,
    select_lags,
)
from pyardl.unitroot.ngperron import M_STATISTICS, NgPerronResults, ng_perron
from pyardl.unitroot.report import integration_order, report

__all__ = [
    "CBAR",
    "ADFRegression",
    "DFGLSResults",
    "M_STATISTICS",
    "NgPerronResults",
    "adf_regression",
    "dfgls",
    "gls_detrend",
    "integration_order",
    "ng_perron",
    "report",
    "select_lags",
]
