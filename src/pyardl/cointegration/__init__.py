"""Classical cointegration tests.

Provided for comparison with the bounds test, and because much of the
applied literature reports them. The bounds test remains the recommended
route: it does not require every series to be I(1).
"""

from pyardl.cointegration.engle_granger import EGResults, engle_granger

__all__ = ["EGResults", "engle_granger"]
