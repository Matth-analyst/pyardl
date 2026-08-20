r"""Critical values and p-values for the Engle-Granger test.

The two-step test compares a t-ratio computed on *estimated* residuals,
not on observed data. Estimating the cointegrating vector in step one
uses up information, and the null distribution of the step-two statistic
shifts accordingly — further left as the number of regressors grows.
Using ordinary Dickey-Fuller critical values here is a real error, not a
refinement: it over-rejects, and increasingly so with ``k``.

MacKinnon (1994, 2010) tabulated the right distribution as a response
surface in the sample size,

.. math::

    C(\alpha, k, T) = \tau_\infty + \frac{b_1}{T} + \frac{b_2}{T^2}
                      + \frac{b_3}{T^3}

with a separate set of coefficients for every combination of
significance level, number of variables, and deterministic case.

Provenance
----------
The surface coefficients are those published by MacKinnon and are
evaluated through ``statsmodels.tsa.adfvalues``, which transcribes them.
``statsmodels`` is already a required runtime dependency of this
library, so no material is duplicated and no transcription of our own
can drift: there is exactly one copy of those numbers in the
environment, and it is not ours.

That choice is a documented departure from the specification, which
called for our own table of surface coefficients — see
``PROVENANCE.md``. The reasoning is that a second transcription of
the same published numbers adds a failure mode (copy errors) without
adding a source.

What *is* ours is the verification. The surface is cross-checked against
an independent in-house simulation of the null in
``validation/spec06_eg_cv.py``; results and tolerance in
``PROVENANCE.md``.

References
----------
.. [1] MacKinnon, J. G. (1994). Approximate asymptotic distribution
       functions for unit-root and cointegration tests. *Journal of
       Business & Economic Statistics*, 12(2), 167-176.
.. [2] MacKinnon, J. G. (2010). Critical values for cointegration tests.
       Queen's University, Department of Economics, Working Paper 1227.
"""

from __future__ import annotations

import warnings

from pyardl.exceptions import PyardlMethodologyWarning

__all__ = ["eg_critical_values", "eg_pvalue", "EG_MAX_VARS"]

#: Largest number of variables (dependent plus regressors) the published
#: surfaces cover.
EG_MAX_VARS = 12

_ALPHAS = (0.10, 0.05, 0.01)
_SUPPORTED_TRENDS = ("n", "c", "ct", "ctt")


def _check(n_vars: int, trend: str) -> None:
    if trend not in _SUPPORTED_TRENDS:
        raise ValueError(
            f"trend={trend!r} is not available. Use one of {_SUPPORTED_TRENDS}."
        )
    if not 1 <= n_vars <= EG_MAX_VARS:
        raise ValueError(
            f"n_vars={n_vars} is outside the published surfaces "
            f"(1 to {EG_MAX_VARS}). With more variables, use the bounds "
            "test, which does not require a first-step regression."
        )


def eg_critical_values(n_vars: int, nobs: int, trend: str = "c") -> dict[float, float]:
    """Left-tail critical values of the Engle-Granger statistic.

    Parameters
    ----------
    n_vars : int
        Total number of variables, dependent included: ``k + 1`` for
        ``k`` regressors. This is the axis the surfaces are indexed on.
    nobs : int
        Sample size used in the first-step regression.
    trend : {'n', 'c', 'ct', 'ctt'}, default 'c'
        Deterministic terms of the **first step**. The distribution
        depends on them, so a mismatch here silently invalidates the
        test.

    Returns
    -------
    dict
        Level to critical value, at 10%, 5% and 1%.

    Raises
    ------
    ValueError
        For an unsupported trend or a number of variables outside the
        published range.

    Warns
    -----
    PyardlMethodologyWarning
        When ``trend='n'``, for which MacKinnon (2010) published no
        cointegration surfaces; the values are then unavailable rather
        than approximated by a neighbouring case.

    Examples
    --------
    >>> from pyardl.critical_values.mackinnon import eg_critical_values
    >>> cv = eg_critical_values(n_vars=2, nobs=200, trend="c")
    >>> round(cv[0.05], 3)
    -3.367
    """
    _check(n_vars, trend)
    if trend == "n":
        warnings.warn(
            "MacKinnon (2010) published no cointegration critical values "
            "for trend='n'. They are reported as unavailable rather than "
            "borrowed from another deterministic case.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )
        return dict.fromkeys(_ALPHAS, float("nan"))

    from statsmodels.tsa.adfvalues import mackinnoncrit

    values = mackinnoncrit(N=n_vars, regression=trend, nobs=nobs)
    return {0.01: float(values[0]), 0.05: float(values[1]), 0.10: float(values[2])}


def eg_pvalue(stat: float, n_vars: int, trend: str = "c") -> float:
    """Approximate asymptotic p-value of the Engle-Granger statistic.

    Parameters
    ----------
    stat : float
        The step-two t-ratio.
    n_vars : int
        Total number of variables, dependent included.
    trend : {'n', 'c', 'ct', 'ctt'}, default 'c'
        Deterministic terms of the first step.

    Returns
    -------
    float
        Left-tail probability. Small values are evidence of
        cointegration.

    Examples
    --------
    >>> from pyardl.critical_values.mackinnon import eg_pvalue
    >>> round(eg_pvalue(-4.5, n_vars=2, trend="c"), 4)
    0.0012
    """
    _check(n_vars, trend)
    from statsmodels.tsa.adfvalues import mackinnonp

    return float(mackinnonp(stat, regression=trend, N=n_vars))
