"""Unified cointegration analysis (spec 21).

The orchestration layer over the ARDL genealogy: linear or asymmetric,
with or without Fourier terms, tabulated or bootstrapped inference. It
owns no estimator of its own — its one responsibility is routing each
combination to the brick validated for it, and giving each the critical
values that combination requires.
"""

from pyardl.unified.analysis import (
    UnifiedResults,
    cointegration_analysis,
    resolve_critical_values,
)

__all__ = ["UnifiedResults", "cointegration_analysis", "resolve_critical_values"]
