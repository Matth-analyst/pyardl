r"""Critical values for the bounds test on a NARDL.

The PSS tables assume ``k`` regressors that behave like independent
random walks. The partial sums of a decomposition do not: ``x⁺`` and
``x⁻`` are two pieces of one series, correlated at −0.99 in levels, and
their increments are never both non-zero. Reading a NARDL statistic
against those tables therefore reads it against the wrong null, and it
shows up as a size distortion in both directions:

=======================================  ==========================
Reading, at a nominal 5%                 Measured rejection rate
=======================================  ==========================
``k = 2`` per decomposed variable        7.3%
``k = 1`` per decomposed variable        2.6%
*control:* a linear ARDL, 2 regressors   4.8%
=======================================  ==========================

The control is what rules out the lazy explanation: a genuine two-
regressor model at the same sample size is correctly sized, so this is
not the familiar small-sample behaviour of asymptotic critical values.

There is also **no meaningful lower bound** here. The I(0) bound of the
PSS framework covers stationary regressors; decomposing a stationary
series produces two *trending* ones (measured slope +0.56 over 400
observations), so that world is unreachable through the decomposition. A
single critical value is the honest object, and that is what this table
holds.

These values are simulated, under the same protocol as the rest of the
library: ``y`` a random walk, each asymmetric regressor drawn as a random
walk and *then* decomposed — the transformation real data undergoes.
Provenance, parameters and structural cross-checks: ``PROVENANCE.md``.

This file is generated; do not edit it by hand.

References
----------
.. [1] Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). Modelling
       asymmetric cointegration and dynamic multipliers in a nonlinear
       ARDL framework. In *Festschrift in Honor of Peter Schmidt*
       (pp. 281-314). Springer.
"""

from __future__ import annotations

__all__ = ["MAX_K_ASYM", "nardl_critical_value"]

#: Largest number of decomposed variables the table covers.
MAX_K_ASYM = 3

# --- GENERATED BLOCK: injected from validation/results/spec17_nardl_cv_table.py
_NARDL_CV: dict[tuple[int, int], dict[float, float]] = {
    (1, 1): {0.1: 3.733867, 0.05: 4.409536, 0.01: 5.874555},
    (1, 2): {0.1: 3.337796, 0.05: 3.810113, 0.01: 4.807608},
    (1, 3): {0.1: 3.112311, 0.05: 3.500351, 0.01: 4.296998},
    (2, 1): {0.1: 3.685262, 0.05: 4.222035, 0.01: 5.394711},
    (2, 2): {0.1: 3.313973, 0.05: 3.738053, 0.01: 4.646313},
    (2, 3): {0.1: 3.114431, 0.05: 3.467073, 0.01: 4.210846},
    (3, 1): {0.1: 4.468317, 0.05: 5.162020, 0.01: 6.657426},
    (3, 2): {0.1: 3.721536, 0.05: 4.234464, 0.01: 5.266955},
    (3, 3): {0.1: 3.366513, 0.05: 3.760700, 0.01: 4.563442},
    (4, 1): {0.1: 4.001784, 0.05: 4.557102, 0.01: 5.799731},
    (4, 2): {0.1: 3.524718, 0.05: 3.968151, 0.01: 4.890523},
    (4, 3): {0.1: 3.259058, 0.05: 3.626284, 0.01: 4.368016},
    (5, 1): {0.1: 5.067259, 0.05: 5.854285, 0.01: 7.513768},
    (5, 2): {0.1: 4.048606, 0.05: 4.566200, 0.01: 5.657178},
    (5, 3): {0.1: 3.604300, 0.05: 4.024379, 0.01: 4.892368},
}
# --- END GENERATED BLOCK


def nardl_critical_value(case: int, k_asym: int, alpha: float) -> float:
    r"""Critical value of the NARDL bounds F test.

    Parameters
    ----------
    case : int
        Deterministic case, 1 to 5, in the PSS numbering.
    k_asym : int
        Number of **decomposed** variables — not the number of level
        terms, which is twice that.
    alpha : float
        Significance level; one of 0.10, 0.05, 0.01.

    Returns
    -------
    float
        The value the statistic must exceed to reject. A single value,
        not a pair: see the module docstring for why no lower bound
        applies.

    Raises
    ------
    ValueError
        If the configuration falls outside the simulated table. Nothing
        is interpolated or substituted from a neighbouring cell.
    """
    if not _NARDL_CV:  # pragma: no cover - guards a partial install
        raise ValueError(
            "The NARDL critical-value table is empty: this build was "
            "generated without running validation/spec17_nardl_cv.py."
        )
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be 1..5, got {case}.")
    if not 1 <= k_asym <= MAX_K_ASYM:
        raise ValueError(
            f"The table covers 1 to {MAX_K_ASYM} decomposed variables, got "
            f"{k_asym}. Extend validation/spec17_nardl_cv.py to cover it."
        )
    try:
        return _NARDL_CV[(case, k_asym)][alpha]
    except KeyError as exc:
        raise ValueError(
            f"No simulated value at alpha={alpha}; available levels are "
            f"{sorted(next(iter(_NARDL_CV.values())))}."
        ) from exc
