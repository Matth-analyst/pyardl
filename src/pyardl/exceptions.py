"""Dedicated warning classes for methodological limitations.

Any methodological caveat (degenerate case, very small sample, failure to
converge, ...) is signalled through a subclass of
:class:`PyardlMethodologyWarning` rather than a bare ``UserWarning``, so
that these warnings can be filtered specifically::

    import warnings
    from pyardl.exceptions import PyardlMethodologyWarning

    warnings.filterwarnings("error", category=PyardlMethodologyWarning)

Raising them to errors is a good habit in scripted analyses: it prevents a
silently unreliable result from going unnoticed.
"""

from __future__ import annotations


class PyardlMethodologyWarning(UserWarning):
    """Base class for every methodological warning issued by pyardl."""


class DegenerateCaseWarning(PyardlMethodologyWarning):
    """No error-correction force, or no convergence to a long-run equilibrium.

    Emitted when the adjustment speed ``lambda`` is close to zero (no pull
    back towards equilibrium) or falls outside ``]-1, 0[`` (no convergence).
    Long-run quantities such as ``theta`` and the half-life are then not
    statistically defined and are reported as ``NaN``.
    """
