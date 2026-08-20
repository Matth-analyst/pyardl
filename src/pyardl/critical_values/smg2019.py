r"""Bounds for :math:`F_{indep}`, the third test of Sam, McNown & Goh.

:math:`F_{indep}` tests :math:`\gamma = 0`: the levels of the regressors
alone, leaving :math:`\lambda` unrestricted. Like :math:`F_{overall}` it
has no standard distribution — it depends on whether the regressors are
I(0) or I(1) — so it is read against a pair of bounds, not a single
critical value.

Sam, McNown & Goh (2019) publish their bounds in *Economic Modelling*,
behind an access barrier, and the project rule forbids encoding a
critical value it has not computed. The spec anticipates that and
provides for the fallback: simulate them with the same engine that
produces the 2.5% PSS bounds. That is what
``validation/spec15_findep_cv.py`` does, under exactly the null of PSS —
``y`` a random walk, regressors i.i.d. for the I(0) bound and
independent random walks for the I(1) bound — and it computes
:math:`F_{indep}` on the *same* replications as :math:`F_{overall}` and
:math:`t_{BDM}`, so the three sets of bounds describe one single world.

The table is keyed by ``(case, k, i1)`` and holds the 10%, 5% and 1%
quantiles. Provenance, parameters and structural cross-checks:
``PROVENANCE.md``.

This file is generated; do not edit it by hand.

References
----------
.. [1] Sam, C. Y., McNown, R. & Goh, S. K. (2019). An augmented
       autoregressive distributed lag bounds test for cointegration.
       *Economic Modelling*, 80, 130-141.
"""

from __future__ import annotations

__all__ = ["MAX_K_FINDEP", "findep_bounds"]

#: Largest ``k`` covered by the simulated table.
MAX_K_FINDEP = 10

# --- GENERATED BLOCK: injected from validation/results/spec15_findep_table.py
_FINDEP_TABLE: dict[tuple[int, int, bool], dict[float, float]] = {
    (1, 1, False): {0.1: 2.735022, 0.05: 3.891120, 0.01: 6.730497},
    (1, 1, True): {0.1: 4.951525, 0.05: 6.540534, 0.01: 10.096828},
    (1, 2, False): {0.1: 2.309105, 0.05: 2.996693, 0.01: 4.635643},
    (1, 2, True): {0.1: 4.113364, 0.05: 5.067728, 0.01: 7.135546},
    (1, 3, False): {0.1: 2.091255, 0.05: 2.617317, 0.01: 3.801713},
    (1, 3, True): {0.1: 3.689084, 0.05: 4.384896, 0.01: 5.928410},
    (1, 4, False): {0.1: 1.956756, 0.05: 2.397247, 0.01: 3.347750},
    (1, 4, True): {0.1: 3.458771, 0.05: 4.035595, 0.01: 5.278902},
    (1, 5, False): {0.1: 1.848784, 0.05: 2.216625, 0.01: 3.021915},
    (1, 5, True): {0.1: 3.280149, 0.05: 3.772825, 0.01: 4.824196},
    (1, 6, False): {0.1: 1.780037, 0.05: 2.110051, 0.01: 2.807943},
    (1, 6, True): {0.1: 3.172250, 0.05: 3.618597, 0.01: 4.556185},
    (1, 7, False): {0.1: 1.726286, 0.05: 2.018696, 0.01: 2.660640},
    (1, 7, True): {0.1: 3.070066, 0.05: 3.457478, 0.01: 4.311428},
    (1, 8, False): {0.1: 1.682194, 0.05: 1.955402, 0.01: 2.537986},
    (1, 8, True): {0.1: 2.987807, 0.05: 3.353650, 0.01: 4.123113},
    (1, 9, False): {0.1: 1.639915, 0.05: 1.896974, 0.01: 2.424308},
    (1, 9, True): {0.1: 2.946426, 0.05: 3.285645, 0.01: 3.980695},
    (1, 10, False): {0.1: 1.606702, 0.05: 1.840457, 0.01: 2.338236},
    (1, 10, True): {0.1: 2.894324, 0.05: 3.218311, 0.01: 3.873806},
    (2, 1, False): {0.1: 3.949735, 0.05: 4.819101, 0.01: 6.755646},
    (2, 1, True): {0.1: 4.650596, 0.05: 5.604956, 0.01: 7.776196},
    (2, 2, False): {0.1: 3.117076, 0.05: 3.752437, 0.01: 5.123374},
    (2, 2, True): {0.1: 4.054639, 0.05: 4.767646, 0.01: 6.276069},
    (2, 3, False): {0.1: 2.678950, 0.05: 3.179199, 0.01: 4.279544},
    (2, 3, True): {0.1: 3.713884, 0.05: 4.287767, 0.01: 5.527256},
    (2, 4, False): {0.1: 2.410072, 0.05: 2.835094, 0.01: 3.760922},
    (2, 4, True): {0.1: 3.482760, 0.05: 3.981685, 0.01: 5.022731},
    (2, 5, False): {0.1: 2.235689, 0.05: 2.598171, 0.01: 3.356003},
    (2, 5, True): {0.1: 3.324752, 0.05: 3.762597, 0.01: 4.667585},
    (2, 6, False): {0.1: 2.105515, 0.05: 2.427082, 0.01: 3.132969},
    (2, 6, True): {0.1: 3.215052, 0.05: 3.607830, 0.01: 4.410061},
    (2, 7, False): {0.1: 2.014807, 0.05: 2.310261, 0.01: 2.934063},
    (2, 7, True): {0.1: 3.114247, 0.05: 3.474024, 0.01: 4.233790},
    (2, 8, False): {0.1: 1.929478, 0.05: 2.201704, 0.01: 2.790412},
    (2, 8, True): {0.1: 3.043923, 0.05: 3.383327, 0.01: 4.093069},
    (2, 9, False): {0.1: 1.864318, 0.05: 2.119731, 0.01: 2.664446},
    (2, 9, True): {0.1: 2.984038, 0.05: 3.303515, 0.01: 3.942003},
    (2, 10, False): {0.1: 1.813083, 0.05: 2.051299, 0.01: 2.546371},
    (2, 10, True): {0.1: 2.924922, 0.05: 3.225361, 0.01: 3.831872},
    (3, 1, False): {0.1: 2.697153, 0.05: 3.837109, 0.01: 6.596474},
    (3, 1, True): {0.1: 5.256132, 0.05: 7.103762, 0.01: 11.197989},
    (3, 2, False): {0.1: 2.287807, 0.05: 2.993737, 0.01: 4.567842},
    (3, 2, True): {0.1: 4.367216, 0.05: 5.463659, 0.01: 7.765496},
    (3, 3, False): {0.1: 2.083584, 0.05: 2.619099, 0.01: 3.814435},
    (3, 3, True): {0.1: 3.864451, 0.05: 4.645723, 0.01: 6.311204},
    (3, 4, False): {0.1: 1.944662, 0.05: 2.381098, 0.01: 3.339157},
    (3, 4, True): {0.1: 3.582863, 0.05: 4.211528, 0.01: 5.502983},
    (3, 5, False): {0.1: 1.847475, 0.05: 2.217798, 0.01: 3.028497},
    (3, 5, True): {0.1: 3.401746, 0.05: 3.926128, 0.01: 5.010260},
    (3, 6, False): {0.1: 1.782144, 0.05: 2.109008, 0.01: 2.802702},
    (3, 6, True): {0.1: 3.245878, 0.05: 3.716914, 0.01: 4.691881},
    (3, 7, False): {0.1: 1.729625, 0.05: 2.028644, 0.01: 2.664811},
    (3, 7, True): {0.1: 3.145480, 0.05: 3.563380, 0.01: 4.423930},
    (3, 8, False): {0.1: 1.675043, 0.05: 1.946741, 0.01: 2.529446},
    (3, 8, True): {0.1: 3.063957, 0.05: 3.446798, 0.01: 4.233493},
    (3, 9, False): {0.1: 1.638937, 0.05: 1.894539, 0.01: 2.420037},
    (3, 9, True): {0.1: 2.990598, 0.05: 3.354638, 0.01: 4.080371},
    (3, 10, False): {0.1: 1.601672, 0.05: 1.840718, 0.01: 2.337560},
    (3, 10, True): {0.1: 2.935658, 0.05: 3.269031, 0.01: 3.941097},
    (4, 1, False): {0.1: 4.566680, 0.05: 5.589987, 0.01: 7.743583},
    (4, 1, True): {0.1: 5.271637, 0.05: 6.362721, 0.01: 8.659515},
    (4, 2, False): {0.1: 3.483828, 0.05: 4.211367, 0.01: 5.768465},
    (4, 2, True): {0.1: 4.427968, 0.05: 5.208469, 0.01: 6.870100},
    (4, 3, False): {0.1: 2.959942, 0.05: 3.517398, 0.01: 4.680710},
    (4, 3, True): {0.1: 3.969112, 0.05: 4.584025, 0.01: 5.899057},
    (4, 4, False): {0.1: 2.628844, 0.05: 3.097177, 0.01: 4.043361},
    (4, 4, True): {0.1: 3.677254, 0.05: 4.219689, 0.01: 5.361099},
    (4, 5, False): {0.1: 2.397808, 0.05: 2.806205, 0.01: 3.623934},
    (4, 5, True): {0.1: 3.484513, 0.05: 3.953049, 0.01: 4.929605},
    (4, 6, False): {0.1: 2.246494, 0.05: 2.607401, 0.01: 3.335911},
    (4, 6, True): {0.1: 3.331175, 0.05: 3.756208, 0.01: 4.605512},
    (4, 7, False): {0.1: 2.126603, 0.05: 2.449589, 0.01: 3.121579},
    (4, 7, True): {0.1: 3.212412, 0.05: 3.602579, 0.01: 4.377990},
    (4, 8, False): {0.1: 2.024227, 0.05: 2.318754, 0.01: 2.937073},
    (4, 8, True): {0.1: 3.121030, 0.05: 3.482224, 0.01: 4.205672},
    (4, 9, False): {0.1: 1.946042, 0.05: 2.212493, 0.01: 2.765997},
    (4, 9, True): {0.1: 3.063029, 0.05: 3.398088, 0.01: 4.090541},
    (4, 10, False): {0.1: 1.886950, 0.05: 2.132071, 0.01: 2.670157},
    (4, 10, True): {0.1: 3.004470, 0.05: 3.310906, 0.01: 3.919283},
    (5, 1, False): {0.1: 2.713644, 0.05: 3.868895, 0.01: 6.616655},
    (5, 1, True): {0.1: 5.251554, 0.05: 7.171061, 0.01: 11.580091},
    (5, 2, False): {0.1: 2.312581, 0.05: 2.995713, 0.01: 4.675829},
    (5, 2, True): {0.1: 4.375682, 0.05: 5.493471, 0.01: 7.840455},
    (5, 3, False): {0.1: 2.086407, 0.05: 2.611470, 0.01: 3.802913},
    (5, 3, True): {0.1: 3.917383, 0.05: 4.729599, 0.01: 6.438412},
    (5, 4, False): {0.1: 1.943463, 0.05: 2.372020, 0.01: 3.345275},
    (5, 4, True): {0.1: 3.602319, 0.05: 4.241978, 0.01: 5.610674},
    (5, 5, False): {0.1: 1.859030, 0.05: 2.228667, 0.01: 3.048608},
    (5, 5, True): {0.1: 3.417710, 0.05: 3.964702, 0.01: 5.121782},
    (5, 6, False): {0.1: 1.779045, 0.05: 2.102856, 0.01: 2.808950},
    (5, 6, True): {0.1: 3.263171, 0.05: 3.756575, 0.01: 4.748531},
    (5, 7, False): {0.1: 1.725576, 0.05: 2.024900, 0.01: 2.687407},
    (5, 7, True): {0.1: 3.166784, 0.05: 3.595322, 0.01: 4.478467},
    (5, 8, False): {0.1: 1.675642, 0.05: 1.950355, 0.01: 2.524270},
    (5, 8, True): {0.1: 3.077514, 0.05: 3.480770, 0.01: 4.282702},
    (5, 9, False): {0.1: 1.643095, 0.05: 1.901584, 0.01: 2.451376},
    (5, 9, True): {0.1: 3.012082, 0.05: 3.380618, 0.01: 4.149160},
    (5, 10, False): {0.1: 1.605332, 0.05: 1.844290, 0.01: 2.330192},
    (5, 10, True): {0.1: 2.952580, 0.05: 3.294543, 0.01: 3.996083},
}
# --- END GENERATED BLOCK


def findep_bounds(case: int, k: int, alpha: float) -> tuple[float, float]:
    r"""Lower and upper bound of :math:`F_{indep}` at level ``alpha``.

    Parameters
    ----------
    case : int
        Deterministic case, 1 to 5, in the numbering of PSS (2001).
    k : int
        Number of regressors, 1 to :data:`MAX_K_FINDEP`.
    alpha : float
        Significance level; one of 0.10, 0.05, 0.01.

    Returns
    -------
    lower, upper : float
        The I(0) and I(1) bounds. The statistic is read exactly as
        :math:`F_{overall}`: below the lower bound is a failure to
        reject, above the upper bound a rejection, and between them the
        test is inconclusive.

    Raises
    ------
    ValueError
        If the configuration falls outside the simulated table. No
        neighbouring value is substituted: an unavailable bound is
        reported, never approximated.
    """
    if not _FINDEP_TABLE:  # pragma: no cover - guards a partial install
        raise ValueError(
            "The F_indep table is empty: this build was generated without "
            "running validation/spec15_findep_cv.py."
        )
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be 1..5, got {case}.")
    if not 1 <= k <= MAX_K_FINDEP:
        raise ValueError(
            f"F_indep bounds are simulated for k = 1..{MAX_K_FINDEP}, got {k}. "
            "Extend validation/spec15_findep_cv.py to cover it."
        )
    try:
        lower = _FINDEP_TABLE[(case, k, False)][alpha]
        upper = _FINDEP_TABLE[(case, k, True)][alpha]
    except KeyError as exc:  # pragma: no cover - alpha guarded upstream
        raise ValueError(
            f"No simulated F_indep bound at alpha={alpha}; available levels "
            f"are {sorted(next(iter(_FINDEP_TABLE.values())))}."
        ) from exc
    return lower, upper
