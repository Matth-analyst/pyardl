"""Diagnostic tests on a fitted model.

Currently the parameter-constancy tests of Brown, Durbin & Evans (1975):
CUSUM and CUSUM of squares, both built on recursive residuals.
"""

from pyardl.diagnostics.stability import (
    CUSUMResults,
    cusum,
    cusumsq,
    plot_cusum,
    plot_cusumsq,
    recursive_residuals,
    stability_tests,
)

__all__ = [
    "CUSUMResults",
    "cusum",
    "cusumsq",
    "plot_cusum",
    "plot_cusumsq",
    "recursive_residuals",
    "stability_tests",
]
