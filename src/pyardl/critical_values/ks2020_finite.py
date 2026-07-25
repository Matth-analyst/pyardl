r"""Finite-sample response surfaces (experimental, not validated).

.. warning::

   This module is **experimental and unvalidated**. Permission to use
   the underlying coefficient file is still pending with its authors,
   and no admissible comparison against a reference implementation has
   been carried out. Do not rely on it in production.

Would provide critical values and p-values for both the F and t
statistics, adjusted for the sample size and the number of short-run
coefficients, by evaluating the coefficients published by Kripfganz &
Schneider alongside their Stata package.

Those coefficients carry no explicit licence, so pyardl does not
redistribute them: they would be downloaded from the authors' site on
first use, cached locally with a checksum. That download must not be
performed until permission has been granted.

References
----------
Kripfganz, S. & Schneider, D. C. (2020). "Response Surface Regressions
for Critical Value Bounds and Approximate p-values in Equilibrium
Correction Models", *Oxford Bulletin of Economics and Statistics*,
82(6), 1456-1481.
MacKinnon, J. G. (1996). "Numerical Distribution Functions for Unit
Root and Cointegration Tests", *Journal of Applied Econometrics*,
11(6), 601-618.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as sps

__all__ = [
    "download_surface_coefs",
    "crit_value_bounds_finite",
    "pvalue_bounds_finite",
    "cache_dir",
]

_URL = "http://www.kripfganz.de/stata/ardl_surfreg_coefs.dta"
_T_CASES = (1, 3, 5)


def cache_dir() -> Path:
    """Dossier de cache local (``PYARDL_CACHE`` ou ``~/.pyardl``)."""
    base = os.environ.get("PYARDL_CACHE")
    return (Path(base) if base else Path.home() / ".pyardl") / "ks2020"


def _coefs_path() -> Path:
    return cache_dir() / "ardl_surfreg_coefs.dta"


def download_surface_coefs(force: bool = False, url: str = _URL) -> Path:
    """Download the coefficient file from the authors' site.

    Writes it to :func:`cache_dir` together with a provenance record
    (URL, date, SHA-256). Later calls read the cache; ``force=True``
    downloads again.

    .. warning::

       Do not call this until permission to use the file has been
       granted; see the module warning.
    """
    path = _coefs_path()
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        payload = resp.read()
    if not payload[:20].count(b"<") == 0 or len(payload) < 10_000:
        raise RuntimeError(
            f"Invalid download from {url} ({len(payload)} bytes); check the "
            f"URL or place the file manually at {path}."
        )
    path.write_bytes(payload)
    meta = {
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "downloaded": datetime.date.today().isoformat(),
        "source": "Kripfganz & Schneider, package Stata ardl "
        "(kripfganz.de); not redistributed by pyardl.",
    }
    (path.parent / "provenance.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return path


_TABLES: dict[tuple[str, int, int], pd.DataFrame] | None = None


def _load_tables() -> dict[tuple[str, int, int], pd.DataFrame]:
    """Load the coefficient tables, caching them in memory."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    path = _coefs_path()
    if not path.exists():
        raise FileNotFoundError(
            "The finite-sample coefficient tables are not in the local "
            f"cache ({path}). They are not distributed with pyardl for "
            "licensing reasons; run once:\n"
            "    from pyardl.critical_values.ks2020_finite import "
            "download_surface_coefs\n"
            "    download_surface_coefs()\n"
            "to fetch them from the authors' site."
        )
    df = pd.read_stata(path)
    _TABLES = {
        (str(stat), int(case), int(i1)): grp.sort_values("p").reset_index(drop=True)
        for (stat, case, i1), grp in df.groupby(["stat", "c", "I"])
    }
    return _TABLES


def _check(stat: str, case: int, k: int, t_obs: int, sr: int) -> int:
    """Validate the inputs and return the effective case for the tables.

    For the t statistic, the restricted-deterministic cases are served by
    the surfaces of the corresponding unrestricted case (2 -> 3, 4 -> 5),
    since restricting a deterministic term changes the vector tested by
    F but not the distribution of t. Not independently revalidated.
    """
    if stat not in ("F", "t"):
        raise ValueError(f"stat must be 'F' or 't', got {stat!r}.")
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be between 1 and 5, got {case}.")
    if k < 0:
        raise ValueError("k must be >= 0.")
    if t_obs < 5:
        raise ValueError("t_obs must be >= 5.")
    if sr < 0:
        raise ValueError("sr must be >= 0 (number of short-run coefficients).")
    if stat == "t" and case in (2, 4):
        return case + 1
    return case


def _cv_grid(
    stat: str, case: int, i1: bool, k: int, t_obs: int | None, sr: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the tabulated levels and predicted bounds for one side."""
    tab = _load_tables()[(stat, case, int(i1))]
    kp1 = float(k + 1)
    cv = np.zeros(len(tab))
    for j in range(5):
        cv += tab[f"theta_{j}_0_0"].to_numpy() / kp1**j
        if t_obs is not None:
            cv += (
                tab[f"theta_{j}_1_0"].to_numpy() + tab[f"theta_{j}_1_1"].to_numpy() * sr
            ) / (kp1**j * t_obs)
    if t_obs is not None:
        cv += (
            tab["theta_0_2_0"].to_numpy() + tab["theta_0_2_1"].to_numpy() * sr
        ) / t_obs**2
        cv += (
            tab["theta_0_3_0"].to_numpy() + tab["theta_0_3_1"].to_numpy() * sr
        ) / t_obs**3
    return tab["p"].to_numpy(dtype=np.float64) / 10_000.0, cv


def crit_value_bounds_finite(
    case: int,
    k: int,
    t_obs: int,
    sr: int,
    alpha: float,
    stat: Literal["F", "t"] = "F",
) -> tuple[float, float]:
    """Finite-sample bounds from the published response surfaces.

    Parameters
    ----------
    case, k : int
        Deterministic case and number of level regressors.
    t_obs : int
        Estimation sample size of the error-correction model.
    sr : int
        Number of short-run coefficients, ``(p - 1) + sum(q_j)``,
        excluding deterministics and level terms.
    alpha : float
        Significance level; interpolated between adjacent tabulated
        levels when needed.
    stat : {"F", "t"}
        Which statistic the bounds are for.
    """
    case_eff = _check(stat, case, k, t_obs, sr)
    out = []
    for i1 in (False, True):
        p_grid, cv = _cv_grid(stat, case_eff, i1, k, t_obs, sr)
        # The p column holds the significance level (right tail for F,
        # left tail for t), so both are interpolated at p = alpha.
        if not p_grid[0] <= alpha <= p_grid[-1]:
            raise ValueError(f"alpha={alpha} is outside the tabulated grid.")
        out.append(float(np.interp(alpha, p_grid, cv)))
    return out[0], out[1]


def pvalue_bounds_finite(
    stat_value: float,
    case: int,
    k: int,
    t_obs: int,
    sr: int,
    df_resid: int,
    stat: Literal["F", "t"] = "F",
) -> tuple[float, float]:
    """Finite-sample p-values at both bounds (MacKinnon 1996, eq. 12).

    The nine tabulated levels whose predicted bound is closest to the
    observed statistic are mapped onto the scale of a reference
    distribution, a local quadratic is fitted, and the p-value is the
    tail probability of the fitted value. Outside the grid the p-value
    is clipped to 0 or 1 on the appropriate side.

    ``df_resid`` is the residual degrees of freedom of the fitted model.
    """
    case_eff = _check(stat, case, k, t_obs, sr)
    df1 = k + 1 + (1 if case in (2, 4) else 0)
    out = []
    for i1 in (False, True):
        p_grid, cv = _cv_grid(stat, case_eff, i1, k, t_obs, sr)
        # p is a tail probability: for F the bound decreases with p
        # (cv[0] is the far right quantile); for t it increases.
        if stat == "F":
            if stat_value >= cv[0]:
                out.append(0.0)
                continue
            if stat_value <= cv[-1]:
                out.append(1.0)
                continue
            invtail = sps.f.isf(p_grid, df1, df_resid)
        else:
            if stat_value <= cv[0]:
                out.append(0.0)
                continue
            if stat_value >= cv[-1]:
                out.append(1.0)
                continue
            invtail = sps.t.ppf(p_grid, df_resid)

        center = int(np.argmin(np.abs(cv - stat_value)))
        lo = max(0, center - 4)
        hi = min(len(cv), center + 5)
        x = cv[lo:hi]
        y = invtail[lo:hi]
        design = np.column_stack([np.ones_like(x), x, x**2])
        coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = float(coefs[0] + coefs[1] * stat_value + coefs[2] * stat_value**2)
        if stat == "F":
            out.append(float(sps.f.sf(fitted, df1, df_resid)))
        else:
            out.append(float(sps.t.cdf(fitted, df_resid)))
    return out[0], out[1]
