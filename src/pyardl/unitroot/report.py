r"""Sequential pre-test report: is any series I(2)?

The bounds test tolerates a mixture of I(0) and I(1) regressors — that
is its main advantage — but it is **not valid if any series is I(2)**.
Its distribution theory simply does not cover that case, and a bounds
test run on I(2) data produces a number with no interpretation rather
than an error. Screening for it is step zero of the workflow.

The protocol
------------
Testing the level alone cannot answer the question. Failing to reject a
unit root in the level is compatible with I(1) *and* with I(2). The
series must therefore be tested twice:

============================  ===============  ==================
Level                         First difference  Verdict
============================  ===============  ==================
rejects                       (not needed)      I(0)
does not reject               rejects           I(1)
does not reject               does not reject   **I(2) suspect**
============================  ===============  ==================

The last row is a suspicion, not a verdict. Failing to reject twice is
also what a short, noisy sample looks like — the tests have little power
against near-unit roots. The word "suspect" is therefore literal, and
the report says so instead of asserting an order of integration it
cannot establish.

Crossing the null hypotheses
----------------------------
DF-GLS and the M tests share the null of a unit root; KPSS reverses it,
testing stationarity. Running both gives four outcomes, two of which are
informative and two of which are honest admissions of ignorance:

- unit root rejected, stationarity not rejected → I(0), agreed
- unit root not rejected, stationarity rejected → I(1), agreed
- neither rejected → the data cannot separate the hypotheses
- both rejected → the specification is wrong (a break, or a trend that
  was not modelled)

References
----------
.. [1] Elliott, G., Rothenberg, T. J. & Stock, J. H. (1996).
       *Econometrica*, 64(4), 813-836.
.. [2] Ng, S. & Perron, P. (2001). *Econometrica*, 69(6), 1519-1554.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.unitroot.ers import dfgls
from pyardl.unitroot.gls import LagMethod, Trend
from pyardl.unitroot.ngperron import ng_perron

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike

__all__ = ["integration_order", "report"]


def integration_order(
    y: ArrayLike,
    trend: Trend = "c",
    alpha: float = 0.05,
    method: LagMethod = "bic",
) -> dict[str, object]:
    """Classify one series as I(0), I(1) or I(2)-suspect.

    Parameters
    ----------
    y : array_like
        Series to classify.
    trend : {'c', 'ct'}, default 'c'
        Deterministic terms used on the **level**. The first difference
        is always tested with a constant only: a trend in the difference
        would mean a quadratic trend in the level, which is not what is
        being asked here.
    alpha : float, default 0.05
    method : {'bic', 'maic', 'mbic', 'aic', 't-stat'}, default 'bic'
        Lag-selection criterion. BIC is the default for the same reason
        as in :func:`report`, of which this is the single-series
        version: on screening work it classifies clean data markedly
        better (40/40 against 29/40 on I(0), 40/40 against 32/40 on
        I(1), over 40 replications of length 250). The figures and the
        reasoning are in :func:`report`.

        :func:`~pyardl.unitroot.dfgls` and
        :func:`~pyardl.unitroot.ng_perron` keep ``'maic'``, which is
        what the literature recommends once you are testing one series
        deliberately.

    Returns
    -------
    dict
        ``order``, the two DF-GLS statistics and decisions, and the
        MZt statistic on the level.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.unitroot import integration_order
    >>> rng = np.random.default_rng(0)
    >>> integration_order(np.cumsum(rng.standard_normal(200)))["order"]
    'I(1)'
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    level = dfgls(y_arr, trend=trend, method=method)
    m_level = ng_perron(y_arr, trend=trend, method=method)

    if level.decision(alpha) == "stationary":
        order = "I(0)"
        diff = None
    else:
        diff = dfgls(np.diff(y_arr), trend="c", method=method)
        order = "I(1)" if diff.decision(alpha) == "stationary" else "I(2)-suspect"

    return {
        "order": order,
        "dfgls_level": level.stat,
        "decision_level": level.decision(alpha),
        "dfgls_diff": diff.stat if diff is not None else np.nan,
        "decision_diff": diff.decision(alpha) if diff is not None else "",
        "mzt_level": m_level.stats["MZt"],
        "lags_level": level.lags,
    }


def report(
    data: pd.DataFrame | pd.Series,
    trend: Trend = "c",
    alpha: float = 0.05,
    method: LagMethod = "bic",
) -> pd.DataFrame:
    """Pre-test every series and tabulate the verdicts.

    Parameters
    ----------
    data : pandas.DataFrame or Series
        One column per variable.
    trend : {'c', 'ct'}, default 'c'
    alpha : float, default 0.05
    method : {'bic', 'maic', 'mbic', 'aic', 't-stat'}, default 'bic'
        Lag-selection criterion.

        **Why the default here is BIC, and not MAIC.** This function is
        a first pass, run before you know what the data look like, and
        on that job BIC classifies clean series markedly better. MAIC's
        penalty term is large precisely when a series looks stationary,
        so it over-selects on I(0) data — 6.1 lags on white noise
        against 0.0 for BIC — which costs power at the differencing
        stage and turns genuine I(1) series into false I(2) suspicions.

        Measured over 40 replications of length 250:

        ==========  ==============  ==============  ==============
        criterion   I(0) correct    I(1) correct    I(2) flagged
        ==========  ==============  ==============  ==============
        BIC         40/40           40/40           35/40
        MAIC        29/40           32/40           37/40
        ==========  ==============  ==============  ==============

        :func:`~pyardl.unitroot.dfgls` and
        :func:`~pyardl.unitroot.ng_perron` keep ``'maic'`` as their own
        default: once you are past screening and testing one series
        deliberately, MAIC is what the literature recommends, because it
        is what protects against a negative moving-average component —
        the case these clean simulations do not exercise. Switch back to
        ``method='maic'`` here whenever you suspect one.

    Returns
    -------
    pandas.DataFrame
        One row per variable: ``order``, the DF-GLS statistic on the
        level and on the difference, their decisions, ``MZt`` on the
        level, and the lag order chosen.

    Warns
    -----
    PyardlMethodologyWarning
        When at least one series is I(2)-suspect. Running a bounds test
        on such data yields a statistic with no valid distribution, so
        this is flagged loudly rather than left in a table the user may
        not read.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.unitroot import report
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"a": np.cumsum(rng.standard_normal(200))})
    >>> report(df)["order"].tolist()
    ['I(1)']
    """
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    rows = {
        str(name): integration_order(
            frame[name].to_numpy(), trend=trend, alpha=alpha, method=method
        )
        for name in frame.columns
    }
    table = pd.DataFrame(rows).T
    table.index.name = "variable"

    suspects = [n for n, r in rows.items() if r["order"] == "I(2)-suspect"]
    if suspects:
        warnings.warn(
            f"I(2) suspected for: {', '.join(suspects)}. The bounds test is "
            "not valid with I(2) variables — its limiting distribution does "
            "not cover them, so it would return a number rather than an "
            "error. Difference the series, or establish that the double "
            "failure to reject reflects low power rather than a second unit "
            "root.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )
    return table
