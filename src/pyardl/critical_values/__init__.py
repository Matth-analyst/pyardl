"""Critical values for the bounds test.

Three sources are available, selected with ``cv_source``:

- ``"kripfganz"`` (recommended, and the default used by
  :func:`~pyardl.bounds.bounds_test`): response-surface based. More
  precise than the published tables, available at any significance
  level, and it also yields approximate p-values at both bounds through
  :func:`pvalue_bounds`. Currently asymptotic, covers the F statistic
  and ``k = 1..10``.
- ``"pss"``: the bounds published in Pesaran, Shin & Smith (2001),
  served exactly as printed. Use this to reproduce published results.
  They carry the Monte Carlo error of the original article (40 000
  replications), roughly +/-0.05 at the usual levels and up to +/-0.15
  in the 1% tail.
- ``"narayan"``: small-sample bounds from Narayan (2005), for
  ``30 <= T <= 80``. Asymptotic bounds over-reject at these sample
  sizes, which is exactly the situation annual data often falls into.
  Covers cases 2, 3 and 5, the F statistic only.

Every encoded table documents its exact source and how it was
cross-checked in ``PROVENANCE.md``, alongside this module.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from pyardl.critical_values.ks2020 import crit_value_bounds as _ks_bounds
from pyardl.critical_values.ks2020 import pvalue_bounds
from pyardl.critical_values.narayan2005 import (
    F_NARAYAN,
    MAX_K_NARAYAN,
    T_GRID,
)
from pyardl.critical_values.pss2001 import get_bounds as _pss_bounds
from pyardl.critical_values.simulate import SimulatedBounds, simulate_bounds
from pyardl.exceptions import PyardlMethodologyWarning

__all__ = ["get_bounds", "pvalue_bounds", "simulate_bounds", "SimulatedBounds"]

_NARAYAN_LEVELS = {0.10: 0, 0.05: 1, 0.01: 2}


def _narayan_bounds(
    stat: str, case: int, k: int, alpha: float, t_obs: int
) -> tuple[float, float]:
    """Narayan (2005) bounds, linearly interpolated in the sample size."""
    if stat != "F":
        raise ValueError(
            "Narayan (2005) only publishes F bounds; use cv_source='pss' or "
            "cv_source='kripfganz' for the t statistic."
        )
    if case in (1, 4):
        raise ValueError(
            f"Narayan (2005) does not cover case {case} (only cases 2, 3 "
            "and 5); use cv_source='kripfganz' or cv_source='pss'."
        )
    if case not in (2, 3, 5):
        raise ValueError(f"case must be between 1 and 5, got {case}.")
    if k > MAX_K_NARAYAN:
        raise ValueError(
            f"k={k} exceeds the Narayan tables (k <= {MAX_K_NARAYAN}); use "
            "cv_source='kripfganz' or cv_source='pss'."
        )
    if alpha not in _NARAYAN_LEVELS:
        raise ValueError(
            f"alpha={alpha} is not covered by Narayan (2005); available "
            f"levels are {tuple(_NARAYAN_LEVELS)}."
        )

    if t_obs < T_GRID[0] or t_obs > T_GRID[-1]:
        warnings.warn(
            f"T={t_obs} lies outside the range of the Narayan tables "
            f"({T_GRID[0]}-{T_GRID[-1]}); falling back to the asymptotic PSS "
            "bounds.",
            PyardlMethodologyWarning,
            stacklevel=3,
        )
        return _pss_bounds("F", case, k, alpha)

    level = _NARAYAN_LEVELS[alpha]
    grid = np.asarray(T_GRID)
    hi = int(np.searchsorted(grid, t_obs))
    if grid[hi] == t_obs:
        cell = F_NARAYAN[t_obs][case][k, level]
        return float(cell[0]), float(cell[1])
    lo = hi - 1
    w = (t_obs - grid[lo]) / (grid[hi] - grid[lo])
    cell_lo = F_NARAYAN[int(grid[lo])][case][k, level]
    cell_hi = F_NARAYAN[int(grid[hi])][case][k, level]
    return (
        float((1 - w) * cell_lo[0] + w * cell_hi[0]),
        float((1 - w) * cell_lo[1] + w * cell_hi[1]),
    )


def get_bounds(
    stat: Literal["F", "t"],
    case: int,
    k: int,
    alpha: float,
    cv_source: Literal["pss", "narayan", "kripfganz"] = "pss",
    t_obs: int | None = None,
) -> tuple[float, float]:
    """Return the (lower, upper) critical value bounds.

    Parameters
    ----------
    stat : {"F", "t"}
        Which statistic the bounds are for.
    case : int
        Deterministic case, 1 to 5.
    k : int
        Number of level regressors.
    alpha : float
        Significance level.
    cv_source : {"pss", "narayan", "kripfganz"}
        Source of the critical values; see the module documentation.
    t_obs : int, optional
        Sample size. Required for ``"narayan"``, which interpolates
        between the tabulated sizes 30 to 80 and falls back to the
        asymptotic bounds (with a warning) outside that range.

    Returns
    -------
    tuple of float
        Lower bound (all regressors I(0)) and upper bound (all I(1)).

    Raises
    ------
    ValueError
        If the requested combination is not covered by the chosen source.
        The message then points to a source that does cover it.

    Examples
    --------
    >>> get_bounds("F", case=3, k=1, alpha=0.05)
    (4.94, 5.73)
    >>> get_bounds("F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=40)
    (5.26, 6.16)
    """
    if cv_source == "pss":
        return _pss_bounds(stat, case, k, alpha)
    if cv_source == "narayan":
        if t_obs is None:
            raise ValueError("cv_source='narayan' requires t_obs.")
        return _narayan_bounds(stat, case, k, alpha, t_obs)
    if cv_source == "kripfganz":
        # Asymptotic, F only; t_obs is not used here.
        return _ks_bounds(case, k, alpha) if stat == "F" else _ks_t_error()
    raise ValueError(f"Unknown cv_source: {cv_source!r}.")


def _ks_t_error() -> tuple[float, float]:
    raise ValueError(
        "cv_source='kripfganz' does not cover the t statistic; use "
        "cv_source='pss' for t bounds."
    )
