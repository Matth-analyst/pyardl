"""Fourier terms for smooth structural change.

Rather than dating and counting breaks, approximate a time-varying
deterministic component by a few low-frequency sinusoids. Two parameters
per frequency capture several breaks of unknown shape and date —
provided the change is smooth, which is the method's whole proviso.

Every test here simulates its own critical values with the frequency
search inside the loop. Skipping that turns a 5% test into a 24.6% one,
which is measured rather than feared: see :mod:`pyardl.fourier.tests`.
"""

from pyardl.fourier.bounds import FourierBoundsResults, fourier_bounds_test
from pyardl.fourier.terms import (
    DEFAULT_GRID,
    INTEGER_GRID,
    fourier_orthogonality,
    fourier_terms,
    select_frequency,
)
from pyardl.fourier.tests import FourierTestResults, fourier_f_test, fourier_kpss

__all__ = [
    "DEFAULT_GRID",
    "INTEGER_GRID",
    "FourierBoundsResults",
    "FourierTestResults",
    "fourier_bounds_test",
    "fourier_f_test",
    "fourier_kpss",
    "fourier_orthogonality",
    "fourier_terms",
    "select_frequency",
]
