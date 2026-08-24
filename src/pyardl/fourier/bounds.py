r"""Cointegration testing when the deterministic part moves smoothly.

Banerjee, Arčabić & Lee (2017) put the Fourier terms of
:mod:`pyardl.fourier.terms` inside a single-equation error-correction
test. The reasoning is simple and the consequence is not: if the
intercept drifts over the sample and the model holds it fixed, the drift
lands in the residual, the residual looks persistent, and the test
concludes there is no error correction when there is one. Ignoring a
smooth break costs power.

The model is the usual UECM with the sinusoids added to the
deterministic block:

.. math::

    \Delta y_t = d_t + \sum_f \left[a_f \sin\!\tfrac{2\pi f t}{T}
                 + b_f \cos\!\tfrac{2\pi f t}{T}\right]
                 + \lambda y_{t-1} + \gamma' x_{t-1} + \text{short run},

and the statistic is the one of Banerjee, Dolado & Mestre: the
left-tailed ``t`` on :math:`\lambda`.

**Two things make the critical values non-standard, not one.** The
regressors are integrated, as in the ordinary bounds test — and the
frequency is chosen on the data, which is the Davies problem measured in
:mod:`pyardl.fourier.tests`: selection alone turned a 5% test into a
24.6% one there. So the null distribution is simulated here with the
frequency search **inside** every replication, exactly as the real call
does it.

A pre-test comes with the result rather than beside it. If the Fourier
terms are not significant, there is no smooth break to accommodate, and
this test is the wrong tool: it spends degrees of freedom on a component
that is not there and loses power for nothing. The result says so.

References
----------
.. [1] Banerjee, P., Arčabić, V. & Lee, H. (2017). Fourier ADL
       cointegration test to approximate smooth breaks with new
       evidence from crude oil market. *Economic Modelling*, 67,
       114-124.
.. [2] Banerjee, A., Dolado, J. & Mestre, R. (1998). Error-correction
       mechanism tests for cointegration in a single-equation framework.
       *Journal of Time Series Analysis*, 19(3), 267-283.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.fourier.terms import INTEGER_GRID, fourier_terms

__all__ = ["FourierBoundsResults", "fourier_bounds_test"]

FloatArray = npt.NDArray[np.float64]

_ALPHAS: tuple[float, ...] = (0.10, 0.05, 0.01)

CVSource = Literal["sim"]


@dataclass(frozen=True)
class FourierBoundsResults:
    """Outcome of a Fourier-ADL cointegration test.

    Attributes
    ----------
    t_stat : float
        Left-tailed ``t`` on the adjustment coefficient.
    critical : dict
        Simulated critical values at the 10%, 5% and 1% levels, built
        with the frequency search inside the loop when the frequency was
        searched for.
    frequency : float
        The frequency used.
    fourier_pvalue : float
        Pre-test: significance of the Fourier terms themselves. When
        they are not significant there is no smooth break to
        accommodate, and the plain bounds test is the better tool.
    fourier_critical : dict
        Critical values of that pre-test, simulated on the same
        regenerated samples. They sit well above a tabulated ``F(2, ·)``,
        for the same two reasons the ``t`` ones do: the data are
        integrated and the frequency was searched for.
    """

    t_stat: float
    critical: dict[float, float]
    pvalue: float
    frequency: float
    freq_estimated: bool
    fourier_stat: float
    fourier_pvalue: float
    fourier_critical: dict[float, float]
    alpha: float
    case: int
    k: int
    order: tuple[int, dict[str, int]]
    n_sims: int
    seed: int
    nobs: int
    grid: tuple[float, ...]
    selection: pd.DataFrame | None = field(default=None, repr=False)
    _fit: Any = field(default=None, repr=False)

    @property
    def decision(self) -> str:
        """Verdict at ``alpha``. Left-tailed, as error correction must be."""
        return (
            "cointegration"
            if self.t_stat < self.critical[self.alpha]
            else "no_cointegration"
        )

    @property
    def fourier_is_warranted(self) -> bool:
        """Whether the smooth component the test pays for actually exists."""
        return self.fourier_pvalue < 0.05

    @property
    def recommendation(self) -> str:
        """What to run, given the pre-test.

        The Fourier test buys robustness to a moving deterministic part
        and pays for it in degrees of freedom. When the pre-test finds no
        such movement, the payment buys nothing.
        """
        if self.fourier_is_warranted:
            return (
                "The Fourier terms are significant: a smooth break is present "
                "and this test is the right one for it."
            )
        return (
            "The Fourier terms are NOT significant (p = "
            f"{self.fourier_pvalue:.4f}): there is no smooth break to "
            "accommodate. The plain bounds test spends fewer parameters and "
            "has more power here — prefer pyardl.bounds.bounds_test."
        )

    def summary(self) -> str:
        """Readable report, pre-test and recommendation included."""
        p, q = self.order
        q_desc = ", ".join(f"{n}:{v}" for n, v in q.items())
        lines = [
            f"Fourier-ADL cointegration test (Banerjee, Arcabic & Lee 2017) - "
            f"case {self.case}, k={self.k}, ECM({p}; {q_desc})",
            f"  frequency {self.frequency:g}"
            + (" (selected)" if self.freq_estimated else " (fixed)")
            + f", {self.nobs} observations",
            "  critical values: simulated"
            + (
                " WITH the frequency search inside the loop"
                if self.freq_estimated
                else " at the fixed frequency"
            )
            + f", n_sims={self.n_sims}, seed={self.seed}",
            "",
            f"  t_BDM = {self.t_stat:.4f}   simulated p = {self.pvalue:.4f}"
            f"   decision ({self.alpha:.0%}): {self.decision}",
            "",
            f"  {'alpha':>7}{'critical':>12}",
        ]
        for a in _ALPHAS:
            lines.append(f"  {a:>7}{self.critical[a]:>12.4f}")
        lines += [
            "",
            f"  pre-test on the Fourier terms: F = {self.fourier_stat:.4f}, "
            f"p = {self.fourier_pvalue:.4f}, "
            f"critical (5%) = {self.fourier_critical[0.05]:.4f}",
            f"  {self.recommendation}",
        ]
        return "\n".join(lines)


def _fourier_columns(n_obs: int, freq: float, fourier_k: int) -> FloatArray:
    """Sine and cosine columns for the first ``fourier_k`` harmonics."""
    freqs = [freq * (i + 1) for i in range(fourier_k)]
    return np.asarray(fourier_terms(n_obs, freqs).to_numpy(), dtype=np.float64)


def _n_parameters(p: int, q: tuple[int, ...], case: int, k: int) -> int:
    """Parameters of the plain UECM, Fourier terms excluded."""
    det = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}[case]
    return det + 1 + k + (p - 1) + sum(q)


def _fit_at(
    y: FloatArray,
    x: FloatArray,
    x_names: tuple[str, ...],
    y_name: str,
    p: int,
    q: tuple[int, ...],
    case: int,
    freq: float,
    fourier_k: int,
) -> Any:
    """The UECM augmented with a Fourier component at one frequency."""
    import warnings

    from pyardl.bounds.pss import _estimate_uecm

    columns: FloatArray | None = None
    names: tuple[str, ...] = ()
    if fourier_k > 0:
        columns = _fourier_columns(y.shape[0], freq, fourier_k)
        names = tuple(f"fourier{j}" for j in range(columns.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _estimate_uecm(y, x, x_names, y_name, p, q, case, columns, names)


def _simulate_null(
    n_obs: int,
    k: int,
    p: int,
    q: tuple[int, ...],
    case: int,
    grid: tuple[float, ...],
    freq: float | None,
    fourier_k: int,
    n_sims: int,
    rng: np.random.Generator,
    chunk: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Null distribution of the ``t``, frequency search included.

    Under the null there is no level relationship: ``y`` and the
    regressors are independent random walks. Every replication then runs
    the *same* frequency search the real call runs — which is the whole
    point, and the reason a tabulated value cannot serve.

    The search is vectorised rather than looped: for each candidate
    frequency the whole batch is fitted at once, and the winner is picked
    per replication on the residual sum of squares. Same arithmetic as a
    loop over replications, without the interpreter cost.
    """
    from pyardl.bootstrap.batch import batch_uecm_statistics

    candidates = grid if freq is None else (freq,)
    n_terms = 2 * fourier_k
    t_out = np.full(n_sims, np.nan)
    f_out = np.full(n_sims, np.nan)
    done = 0
    while done < n_sims:
        size = min(chunk, n_sims - done)
        y_star = np.cumsum(rng.standard_normal((size, n_obs)), axis=1)
        x_star = np.cumsum(rng.standard_normal((size, n_obs, k)), axis=1)

        # The restricted fit: the same UECM WITHOUT the Fourier columns.
        # Its residual sum of squares is what the pre-test compares
        # against, so it is computed on the same regenerated samples
        # rather than on a separate simulation.
        _, _, _, ssr_plain, ok_plain = batch_uecm_statistics(
            y_star, x_star, p, q, case, True, None
        )

        best_ssr: FloatArray = np.full(size, np.inf)
        best_t: FloatArray = np.full(size, np.nan)
        for candidate in candidates:
            columns = _fourier_columns(n_obs, candidate, fourier_k)
            extra = np.broadcast_to(columns, (size, n_obs, columns.shape[1])).copy()
            _, t_stat, _, ssr, ok = batch_uecm_statistics(
                y_star, x_star, p, q, case, True, extra
            )
            better = np.logical_and(ok, ssr < best_ssr)
            best_ssr = np.asarray(np.where(better, ssr, best_ssr), dtype=np.float64)
            best_t = np.asarray(np.where(better, t_stat, best_t), dtype=np.float64)

        n_par = _n_parameters(p, q, case, k) + n_terms
        dof = n_obs - max(p, max(q, default=0)) - n_par
        with np.errstate(invalid="ignore", divide="ignore"):
            f_stat = ((ssr_plain - best_ssr) / n_terms) / (best_ssr / dof)
        f_out[done : done + size] = np.where(ok_plain, f_stat, np.nan)
        t_out[done : done + size] = best_t
        done += size
    keep = np.isfinite(t_out) & np.isfinite(f_out)
    return (
        np.asarray(t_out[keep], dtype=np.float64),
        np.asarray(f_out[keep], dtype=np.float64),
    )


def fourier_bounds_test(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    case: int = 3,
    order: tuple[int, int | dict[str, int]] = (1, 1),
    fourier_k: int = 1,
    freq: float | str = "auto",
    grid: Sequence[float] = INTEGER_GRID,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int | None = None,
) -> FourierBoundsResults:
    r"""Test for cointegration allowing a smooth break in the deterministic part.

    Parameters
    ----------
    y, x : array_like
        Dependent variable and regressors.
    case : int, default 3
        Deterministic case, PSS numbering.
    order : tuple
        ``(p, q)`` of the error-correction model.
    fourier_k : int, default 1
        Number of harmonics. One frequency and two parameters usually
        suffice; each extra harmonic costs two more.
    freq : float or ``'auto'``
        ``'auto'`` searches ``grid`` on the residual sum of squares of
        the model itself. A number fixes it — which changes the critical
        values, because a fixed frequency is not a searched one.
    grid : sequence of float
        Candidate frequencies for the search.
    alpha : float
        Level of the reported decision; one of 0.10, 0.05, 0.01.
    n_sims : int, default 2000
        Replications behind the simulated critical values.
    seed : int, optional
        Drawn from entropy and **recorded** when omitted.

    Returns
    -------
    FourierBoundsResults

    Notes
    -----
    The critical values are simulated on every call. No table could
    cover the sample size, the number of regressors, the deterministic
    case, the number of harmonics *and* whether the frequency was
    searched for — and quietly reading a table that covers only some of
    those is how a 5% test becomes something else.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 150
    >>> t = np.arange(1, n + 1)
    >>> x = np.cumsum(rng.normal(size=n))
    >>> shift = 3.0 / (1 + np.exp(-0.1 * (t - n / 2)))
    >>> y = np.zeros(n)
    >>> for i in range(1, n):
    ...     y[i] = y[i - 1] - 0.4 * (y[i - 1] - x[i - 1] - shift[i - 1]) + rng.normal(
    ...         scale=0.3)
    >>> res = fourier_bounds_test(pd.Series(y, name="y"),
    ...                           pd.DataFrame({"x": x}), n_sims=200, seed=1)
    >>> res.decision
    'cointegration'
    """
    from pyardl.core.ardl import _parse_order
    from pyardl.utils import check_series

    y_arr, x_arr, _, y_name, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("The Fourier-ADL test needs at least one regressor.")
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be 1..5 (PSS numbering), got {case}.")
    if fourier_k < 1:
        raise ValueError(f"fourier_k must be at least 1, got {fourier_k}.")
    if alpha not in _ALPHAS:
        raise ValueError(f"alpha must be one of {list(_ALPHAS)}, got {alpha}.")
    if n_sims < 100:
        raise ValueError(
            f"n_sims={n_sims} is too few to place a 1% quantile; use at "
            "least 100, and far more when reporting one."
        )

    p_order, q_map = _parse_order(order, x_names)
    q_tuple = tuple(int(q_map[name]) for name in x_names)
    n_obs = y_arr.shape[0]
    tuple_grid = tuple(float(f) for f in grid)

    freq_estimated = isinstance(freq, str) and freq == "auto"
    if freq_estimated and not tuple_grid:
        raise ValueError("grid is empty: there is no frequency to search over.")
    if not freq_estimated and not isinstance(freq, (int, float)):
        raise ValueError(f"freq must be a number or 'auto', got {freq!r}.")

    if seed is None:
        entropy = np.random.SeedSequence().entropy
        seed = int(entropy) % (2**63) if isinstance(entropy, int) else 0
    rng = np.random.default_rng(seed)

    selection: pd.DataFrame | None = None
    if freq_estimated:
        rows = []
        for candidate in tuple_grid:
            fit = _fit_at(
                y_arr,
                x_arr,
                x_names,
                y_name,
                p_order,
                q_tuple,
                case,
                candidate,
                fourier_k,
            )
            rows.append({"freq": float(candidate), "ssr": float(fit.ssr)})
        selection = pd.DataFrame(rows).sort_values("ssr").reset_index(drop=True)
        used = float(selection.loc[0, "freq"])
    else:
        used = float(freq)

    fit = _fit_at(
        y_arr, x_arr, x_names, y_name, p_order, q_tuple, case, used, fourier_k
    )
    lam_pos = fit.names.index(fit.lam_name)
    t_stat = float(fit.params[fit.lam_name]) / float(np.sqrt(fit.cov[lam_pos, lam_pos]))

    drawn, drawn_f = _simulate_null(
        n_obs,
        x_arr.shape[1],
        p_order,
        q_tuple,
        case,
        tuple_grid,
        None if freq_estimated else used,
        fourier_k,
        n_sims,
        rng,
    )
    critical = {a: float(np.quantile(drawn, a)) for a in _ALPHAS}
    pvalue = float((1 + np.sum(drawn <= t_stat)) / (drawn.size + 1))

    # Pre-test: are the Fourier terms worth their two parameters? The
    # comparison is made INSIDE the model — the same UECM without them —
    # and read against the null distribution simulated just above, on the
    # same regenerated samples. Running the standalone Fourier F test on
    # y would be wrong here: its null is white noise, and y is
    # integrated, so it would call the terms significant every time.
    plain = _fit_at(y_arr, x_arr, x_names, y_name, p_order, q_tuple, case, used, 0)
    n_terms = 2 * fourier_k
    dof = fit.nobs - len(fit.names)
    fourier_stat = ((float(plain.ssr) - float(fit.ssr)) / n_terms) / (
        float(fit.ssr) / dof
    )
    fourier_pvalue = float((1 + np.sum(drawn_f >= fourier_stat)) / (drawn_f.size + 1))
    fourier_critical = {a: float(np.quantile(drawn_f, 1.0 - a)) for a in _ALPHAS}

    return FourierBoundsResults(
        t_stat=t_stat,
        critical=critical,
        pvalue=pvalue,
        frequency=used,
        freq_estimated=freq_estimated,
        fourier_stat=fourier_stat,
        fourier_pvalue=fourier_pvalue,
        fourier_critical=fourier_critical,
        alpha=alpha,
        case=case,
        k=x_arr.shape[1],
        order=(p_order, q_map),
        n_sims=int(drawn.size),
        seed=int(seed),
        nobs=int(fit.nobs),
        grid=tuple_grid,
        selection=selection,
        _fit=fit,
    )
