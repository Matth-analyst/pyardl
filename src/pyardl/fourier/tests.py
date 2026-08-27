r"""Tests built on Fourier terms, and the trap they all share.

Two tests live here: one asks whether a smooth deterministic component
is present at all, the other whether a series is stationary around one.
They share a difficulty that governs both.

**The Davies problem.** Under the null that the Fourier terms are
absent, the frequency :math:`f` is not identified — there is no true
value for it to converge to. Choosing it by fitting every candidate and
keeping the best is therefore not estimation but *search*, and a
statistic computed at the winning frequency is the maximum over a grid,
not a draw from a fixed distribution. Tabulated critical values do not
apply to it.

The size of the error is not subtle. Measured on 2000 replications with
``T = 200``, a white-noise null and the integer grid 1 to 5:

===============================  ===================
Frequency                        Rejection at 5%
===============================  ===================
fixed in advance                 4.8%
selected on the data             **24.6%**
===============================  ===================

One rejection in four where five in a hundred were promised. The
correct 95% quantile is 5.05 against the tabulated 3.04.

So every test here simulates its own critical values **with the
selection inside the loop** — each replication re-runs the frequency
search on its own null sample, exactly as the real call does. That is
the only construction under which the reported level means what it says.
It is also why these functions take a seed and record it: a critical
value nobody can reproduce is not a result anybody can check.

References
----------
.. [1] Becker, R., Enders, W. & Lee, J. (2006). A stationarity test in
       the presence of an unknown number of smooth breaks. *Journal of
       Time Series Analysis*, 27(3), 381-409.
.. [2] Davies, R. B. (1987). Hypothesis testing when a nuisance
       parameter is present only under the alternative. *Biometrika*,
       74(1), 33-43.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.fourier.terms import INTEGER_GRID, fourier_terms, select_frequency

__all__ = ["FourierTestResults", "fourier_f_test", "fourier_kpss"]

FloatArray = npt.NDArray[np.float64]

_ALPHAS: tuple[float, ...] = (0.10, 0.05, 0.01)


@dataclass(frozen=True)
class FourierTestResults:
    """Outcome of a test built on Fourier terms.

    Attributes
    ----------
    statistic : float
        The test statistic at the frequency actually used.
    frequency : float
        The frequency used — selected from the grid when
        ``freq_estimated`` is true, otherwise the one supplied.
    critical : dict
        Simulated critical values at the 10%, 5% and 1% levels.
    pvalue : float
        Simulated p-value, never exactly zero: ``n_sims`` replications
        cannot resolve more than ``1/(n_sims + 1)``.
    freq_estimated : bool
        Whether the frequency was searched for. When it was, the
        critical values were simulated **with that search inside the
        loop**; when it was not, they were simulated at the fixed
        frequency.
    """

    name: str
    statistic: float
    frequency: float
    critical: dict[float, float]
    pvalue: float
    alpha: float
    freq_estimated: bool
    grid: tuple[float, ...]
    n_sims: int
    seed: int
    nobs: int
    right_tailed: bool
    selection: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def decision(self) -> str:
        """Verdict at ``alpha``, in the direction the test rejects."""
        threshold = self.critical[self.alpha]
        rejected = (
            self.statistic > threshold
            if self.right_tailed
            else self.statistic < threshold
        )
        return "reject" if rejected else "keep"

    def summary(self) -> str:
        """Readable report, naming how the critical values were built."""
        how = (
            "simulated WITH the frequency search inside the loop"
            if self.freq_estimated
            else "simulated at the fixed frequency"
        )
        lines = [
            f"{self.name} - {self.nobs} observations, frequency "
            f"{self.frequency:g}"
            + (" (selected)" if self.freq_estimated else " (fixed)"),
            f"  critical values: {how}, n_sims={self.n_sims}, seed={self.seed}",
            "",
            f"  statistic = {self.statistic:.4f}   simulated p = {self.pvalue:.4f}"
            f"   decision ({self.alpha:.0%}): {self.decision}",
            "",
            f"  {'alpha':>7}{'critical':>12}",
        ]
        for a in _ALPHAS:
            lines.append(f"  {a:>7}{self.critical[a]:>12.4f}")
        return "\n".join(lines)


def _deterministic(n_obs: int, trend: bool) -> FloatArray:
    """Constant, and a linear trend when the model carries one."""
    columns: list[FloatArray] = [np.ones(n_obs)]
    if trend:
        columns.append(np.arange(1.0, n_obs + 1.0, dtype=np.float64))
    return np.column_stack(columns)


def _f_statistic(y: FloatArray, freq: float, trend: bool) -> float:
    r"""F for :math:`H_0: a_f = b_f = 0` at a given frequency."""
    n_obs = y.size
    base = _deterministic(n_obs, trend)
    full = np.column_stack([base, fourier_terms(n_obs, freq).to_numpy()])
    resid0 = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
    resid1 = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
    ssr0 = float(resid0 @ resid0)
    ssr1 = float(resid1 @ resid1)
    # int() sur l'element de shape : sous les stubs numpy >= 2.5,
    # indexer `.shape` rend Any, ce qui contamine `dof` puis toute
    # l'expression de retour et fait echouer mypy --strict sur un
    # "Returning Any". Le cast fixe le type a la source plutot que
    # d'envelopper le retour.
    dof = n_obs - int(full.shape[1])
    if ssr1 <= 0 or dof <= 0:  # pragma: no cover - degenerate design
        return float("nan")
    return ((ssr0 - ssr1) / 2.0) / (ssr1 / dof)


def _kpss_statistic(y: FloatArray, freq: float, trend: bool) -> float:
    r"""KPSS statistic around a Fourier deterministic component.

    The residual of ``y`` on the deterministic terms *and* the Fourier
    pair, then the usual ratio of the cumulated sum of squares to the
    long-run variance. Large values reject stationarity.
    """
    n_obs = y.size
    design = np.column_stack(
        [_deterministic(n_obs, trend), fourier_terms(n_obs, freq).to_numpy()]
    )
    resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    partial = np.cumsum(resid)
    # Bartlett kernel with the usual rule-of-thumb bandwidth.
    lag = int(np.floor(4.0 * (n_obs / 100.0) ** 0.25))
    variance = float(resid @ resid) / n_obs
    for j in range(1, lag + 1):
        weight = 1.0 - j / (lag + 1.0)
        variance += 2.0 * weight * float(resid[j:] @ resid[:-j]) / n_obs
    if variance <= 0:  # pragma: no cover - degenerate long-run variance
        return float("nan")
    return float(np.sum(partial**2) / (n_obs**2 * variance))


def _simulate(
    kind: str,
    n_obs: int,
    trend: bool,
    grid: tuple[float, ...],
    freq: float | None,
    n_sims: int,
    rng: np.random.Generator,
) -> FloatArray:
    r"""Null distribution, with the frequency search inside the loop.

    Under the null both statistics are free of nuisance parameters once
    the deterministic terms are removed, so white noise generates them.
    What is *not* free is the frequency: when the real call searches for
    it, every replication must search too, or the simulated distribution
    describes a procedure nobody ran.
    """
    stat = _f_statistic if kind == "f" else _kpss_statistic
    out = np.empty(n_sims)
    out.fill(np.nan)
    for i in range(n_sims):
        sample = rng.standard_normal(n_obs)
        if freq is None:
            chosen, _ = select_frequency(sample, grid, trend=trend)
        else:
            chosen = freq
        out[i] = stat(sample, chosen, trend)
    return np.asarray(out[np.isfinite(out)], dtype=np.float64)


def _run(
    kind: str,
    name: str,
    right_tailed: bool,
    y: npt.ArrayLike,
    grid: Sequence[float],
    freq: float | None,
    trend: bool,
    freq_estimated: bool,
    alpha: float,
    n_sims: int,
    seed: int | None,
) -> FourierTestResults:
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if y_arr.size < 12:
        raise ValueError(
            f"{y_arr.size} observations is too few for a Fourier test: the "
            "deterministic component alone takes three or four parameters."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}.")
    if alpha not in _ALPHAS:
        raise ValueError(f"alpha must be one of {list(_ALPHAS)}, got {alpha}.")
    if n_sims < 100:
        raise ValueError(
            f"n_sims={n_sims} is too few to place a 1% quantile; use at "
            "least 100, and far more when reporting one."
        )
    tuple_grid = tuple(float(f) for f in grid)
    if freq_estimated and not tuple_grid:
        raise ValueError("grid is empty: there is no frequency to search over.")
    if not freq_estimated and freq is None:
        raise ValueError(
            "freq_estimated=False needs an explicit freq: with no search and "
            "no value there is no frequency to test at."
        )

    if seed is None:
        entropy = np.random.SeedSequence().entropy
        seed = int(entropy) % (2**63) if isinstance(entropy, int) else 0
    rng = np.random.default_rng(seed)

    selection: pd.DataFrame | None = None
    if freq_estimated:
        used, selection = select_frequency(y_arr, tuple_grid, trend=trend)
    else:
        used = float(freq)  # type: ignore[arg-type]

    statistic = (_f_statistic if kind == "f" else _kpss_statistic)(y_arr, used, trend)
    drawn = _simulate(
        kind,
        y_arr.size,
        trend,
        tuple_grid,
        None if freq_estimated else used,
        n_sims,
        rng,
    )
    if right_tailed:
        critical = {a: float(np.quantile(drawn, 1.0 - a)) for a in _ALPHAS}
        pvalue = float((1 + np.sum(drawn >= statistic)) / (drawn.size + 1))
    else:  # pragma: no cover - both shipped tests reject on the right
        critical = {a: float(np.quantile(drawn, a)) for a in _ALPHAS}
        pvalue = float((1 + np.sum(drawn <= statistic)) / (drawn.size + 1))

    return FourierTestResults(
        name=name,
        statistic=float(statistic),
        frequency=float(used),
        critical=critical,
        pvalue=pvalue,
        alpha=alpha,
        freq_estimated=freq_estimated,
        grid=tuple_grid,
        n_sims=int(drawn.size),
        seed=int(seed),
        nobs=int(y_arr.size),
        right_tailed=right_tailed,
        selection=selection,
    )


def fourier_f_test(
    y: npt.ArrayLike,
    grid: Sequence[float] = INTEGER_GRID,
    freq: float | None = None,
    trend: bool = False,
    freq_estimated: bool = True,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int | None = None,
) -> FourierTestResults:
    r"""Test whether a smooth deterministic component is present.

    :math:`H_0: a_f = b_f = 0` — no Fourier component, so the
    deterministic part is the constant (and trend) alone.

    Parameters
    ----------
    y : array_like
        The series.
    grid : sequence of float
        Candidate frequencies when the frequency is searched for.
    freq : float, optional
        A fixed frequency. Required when ``freq_estimated`` is false.
    trend : bool, default False
        Whether a linear trend belongs to the deterministic part.
    freq_estimated : bool, default True
        Whether to search the grid. **This changes the critical
        values**, and by a lot: searching turns a 5% test into a 24.6%
        one if the tabulated values are used. Simulated values are
        produced either way, with the search inside the loop when there
        is one.
    alpha : float
        Level of the reported decision; one of 0.10, 0.05, 0.01.
    n_sims : int, default 2000
        Replications behind the critical values.
    seed : int, optional
        Drawn from entropy and **recorded** when omitted.

    Returns
    -------
    FourierTestResults

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(1, 201)
    >>> y = np.sin(2 * np.pi * 2 * t / 200) + 0.3 * np.random.default_rng(
    ...     0).normal(size=200)
    >>> res = fourier_f_test(y, n_sims=200, seed=1)
    >>> res.decision
    'reject'
    """
    return _run(
        "f",
        "Fourier F test (Becker, Enders & Lee 2006)",
        True,
        y,
        grid,
        freq,
        trend,
        freq_estimated,
        alpha,
        n_sims,
        seed,
    )


def fourier_kpss(
    y: npt.ArrayLike,
    grid: Sequence[float] = INTEGER_GRID,
    freq: float | None = None,
    trend: bool = False,
    freq_estimated: bool = True,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int | None = None,
) -> FourierTestResults:
    r"""Stationarity around a smooth deterministic component.

    KPSS in spirit: the null is **stationarity** around the Fourier
    component, and large values reject it. That direction is the
    opposite of a unit-root test's, and mixing the two up is the classic
    way to report the reverse of what the data say.

    Its usefulness is as a pre-test: a series that looks non-stationary
    to an ADF may simply be stationary around a shifting mean, and this
    tells the two apart without dating the shifts.

    Parameters and returns are those of :func:`fourier_f_test`; the
    critical values are simulated under the same discipline, with the
    frequency search inside the loop when the frequency is searched for.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=200)
    >>> res = fourier_kpss(y, n_sims=200, seed=1)
    >>> res.decision
    'keep'
    """
    return _run(
        "kpss",
        "Fourier KPSS (Becker, Enders & Lee 2006)",
        True,
        y,
        grid,
        freq,
        trend,
        freq_estimated,
        alpha,
        n_sims,
        seed,
    )
