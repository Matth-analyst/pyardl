r"""The system approach: Johansen (1988, 1991).

Where the bounds test asks whether *one* long-run relationship holds
between a dependent variable and its regressors, Johansen's procedure
asks how many hold among a set of variables treated symmetrically. It
works on the VECM

.. math::

    \Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i}
                 + d_t + \varepsilon_t,

where :math:`y_t` is an ``n``-vector. The rank of :math:`\Pi` **is** the
number of cointegrating relations: rank 0 means none, full rank means
the system was stationary to begin with, and anything in between counts
the relations. The rank is read off the eigenvalues of a generalised
eigenvalue problem, through two statistics — the trace and the maximum
eigenvalue.

This module is a **thin wrapper** over
:func:`statsmodels.tsa.vector_ar.vecm.coint_johansen`, which is mature
and validated. Reimplementing it would add risk and buy nothing. What is
added here is what the library needs and statsmodels leaves to the
caller:

1. the pyardl result object, with the usual ``.summary()``;
2. the **sequential decision** — statsmodels returns statistics, not a
   rank, and the loop that turns one into the other has a rule that is
   easy to get wrong (stop at the *first* non-rejection, never continue
   past it);
3. :func:`check_no_cointegration_among_x`, the diagnostic the bounds
   framework needs: the ARDL bounds test assumes there is no
   cointegration *among the regressors*, and nothing in the test itself
   would reveal a violation.

Notes
-----
Deterministic conventions differ between implementations, and the
difference is not cosmetic — it changes the statistics and the critical
values. The correspondence with ``urca::ca.jo`` was established by
running both sides across six variants, not by reading either manual:
``ecdet="none"`` matches ``det_order=0`` despite its name, because it
keeps an unrestricted constant. Full table in ``docs/api/johansen.md``.

The trace statistic over-selects the rank — measured, not assumed — and
never under-selects it. It remains the default because it is what the
applied literature reports; see OBS-10 in
``docs/VALIDATION_OBSERVATIONS.md`` before trusting a borderline rank.

References
----------
.. [1] Johansen, S. (1988). Statistical analysis of cointegration
       vectors. *Journal of Economic Dynamics and Control*, 12, 231-254.
.. [2] Johansen, S. (1991). Estimation and hypothesis testing of
       cointegration vectors in Gaussian vector autoregressive models.
       *Econometrica*, 59(6), 1551-1580.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning

__all__ = ["JohansenResults", "check_no_cointegration_among_x", "johansen"]

_ALPHAS: tuple[float, ...] = (0.10, 0.05, 0.01)
#: Column order of the critical values returned by statsmodels.
_CV_COLUMNS = {0.10: 0, 0.05: 1, 0.01: 2}

Method = Literal["trace", "maxeig"]


@dataclass(frozen=True)
class JohansenResults:
    """Outcome of a Johansen test.

    Attributes
    ----------
    trace_stat, maxeig_stat : pandas.Series
        The statistics, indexed by the null hypothesis they test
        (``r = 0``, ``r <= 1``, ...).
    trace_cv, maxeig_cv : pandas.DataFrame
        Critical values at the 10%, 5% and 1% levels, same index.
    eigenvalues : numpy.ndarray
        The eigenvalues of the generalised problem, in decreasing order.
    beta : pandas.DataFrame
        Cointegrating vectors, one per column, each normalised so that
        its first element is 1. The normalisation is arbitrary — only
        the *space* they span is identified — and it is applied so that
        two runs can be compared at all.
    selected_rank : int
        Rank retained by the sequential procedure at ``alpha``, using
        ``method``.
    n_vars, det_order, k_ar_diff, alpha, method
        Settings the test was run with.
    """

    names: tuple[str, ...]
    trace_stat: pd.Series
    maxeig_stat: pd.Series
    trace_cv: pd.DataFrame
    maxeig_cv: pd.DataFrame
    eigenvalues: npt.NDArray[np.float64]
    beta: pd.DataFrame
    selected_rank: int
    n_vars: int
    det_order: int
    k_ar_diff: int
    alpha: float
    method: Method
    _raw: object = field(repr=False, default=None)

    def rank(self, method: Method | None = None, alpha: float | None = None) -> int:
        """Sequential rank at another level or by the other statistic.

        Parameters
        ----------
        method : {'trace', 'maxeig'}, optional
            Defaults to the one the test was run with.
        alpha : float, optional
            One of 0.10, 0.05, 0.01. Defaults to the one the test was
            run with.
        """
        m: Method = method if method is not None else self.method
        a = alpha if alpha is not None else self.alpha
        stat = self.trace_stat if m == "trace" else self.maxeig_stat
        cv = self.trace_cv if m == "trace" else self.maxeig_cv
        return _sequential_rank(stat.to_numpy(), cv[a].to_numpy())

    def summary(self) -> str:
        """Return a readable report of the test as a string."""
        lines = [
            f"Johansen test (1988, 1991) - {self.n_vars} variables "
            f"({', '.join(self.names)}), det_order={self.det_order}, "
            f"k_ar_diff={self.k_ar_diff}",
            "",
            f"  {'H0':>8}{'trace':>12}{'cv 5%':>10}{'maxeig':>12}{'cv 5%':>10}",
        ]
        for h0 in self.trace_stat.index:
            lines.append(
                f"  {h0:>8}{self.trace_stat[h0]:>12.4f}"
                f"{self.trace_cv.loc[h0, 0.05]:>10.4f}"
                f"{self.maxeig_stat[h0]:>12.4f}"
                f"{self.maxeig_cv.loc[h0, 0.05]:>10.4f}"
            )
        lines += [
            "",
            f"selected rank ({self.method}, {self.alpha:.0%}): "
            f"{self.selected_rank}"
            + (
                "  -> no cointegration"
                if self.selected_rank == 0
                else (
                    "  -> the system was already stationary"
                    if self.selected_rank == self.n_vars
                    else ""
                )
            ),
        ]
        return "\n".join(lines)


def _sequential_rank(stat: npt.NDArray[np.float64], cv: npt.NDArray[np.float64]) -> int:
    """Rank retained by the sequential procedure.

    Test ``r = 0``, then ``r <= 1``, and so on, and **stop at the first
    non-rejection**. Continuing past it — for instance taking the last
    rejection, or the largest rank that rejects — would be a different
    procedure with a different size, and it is the classic way to read
    the output of an implementation that returns statistics only.
    """
    for r, (s, c) in enumerate(zip(stat, cv, strict=True)):
        if s <= c:
            return r
    return int(stat.size)


def johansen(
    y: npt.ArrayLike,
    det_order: int = 0,
    k_ar_diff: int = 1,
    alpha: float = 0.05,
    method: Method = "trace",
) -> JohansenResults:
    r"""Test the cointegration rank of a system of time series.

    Parameters
    ----------
    y : array_like
        Observations, ``(T, n)``. A :class:`pandas.DataFrame` keeps its
        column names in the output.
    det_order : int
        Deterministic terms, in the statsmodels numbering: ``-1`` none,
        ``0`` constant, ``1`` linear trend. See the correspondence table
        in the documentation before comparing with R or Stata.
    k_ar_diff : int
        Number of lagged differences in the VECM — that is, ``p - 1``
        where ``p`` is the lag order of the VAR in levels.
    alpha : float
        Significance level of the sequential procedure; one of 0.10,
        0.05, 0.01.
    method : {'trace', 'maxeig'}
        Statistic driving the sequential procedure. Both are computed
        and reported either way; this only selects which one sets
        ``selected_rank``.

    Returns
    -------
    JohansenResults

    Raises
    ------
    ValueError
        If the inputs or the settings fall outside what the tabulated
        critical values cover. Nothing is substituted silently.

    Notes
    -----
    The critical values come from statsmodels, which tabulates them for
    up to 12 variables. Beyond that the test cannot be decided here and
    an exception is raised rather than a rank returned.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = np.cumsum(rng.normal(size=n))
    >>> z = np.cumsum(rng.normal(size=n))
    >>> data = pd.DataFrame({"x": x, "z": z})
    >>> res = johansen(data, det_order=0, k_ar_diff=1)
    >>> res.selected_rank
    0
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    if det_order not in (-1, 0, 1):
        raise ValueError(
            f"det_order must be -1 (none), 0 (constant) or 1 (trend), got {det_order}."
        )
    if alpha not in _CV_COLUMNS:
        raise ValueError(f"alpha must be one of {sorted(_CV_COLUMNS)}, got {alpha}.")
    if method not in ("trace", "maxeig"):
        raise ValueError(f"method must be 'trace' or 'maxeig', got {method!r}.")
    if k_ar_diff < 0:
        raise ValueError(f"k_ar_diff must be non-negative, got {k_ar_diff}.")

    if isinstance(y, pd.DataFrame):
        names = tuple(str(c) for c in y.columns)
        arr = y.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"y must be two-dimensional (T, n), got shape {arr.shape}."
            )
        names = tuple(f"y{j}" for j in range(arr.shape[1]))

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            "The Johansen test needs at least two series; with one series "
            "there is no rank to test. Use a unit-root test instead "
            "(pyardl.unitroot)."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("y contains NaN or infinite values.")
    n_vars = arr.shape[1]
    if n_vars > 12:
        raise ValueError(
            f"Critical values are tabulated for at most 12 variables, got "
            f"{n_vars}. The statistics could be computed, but no decision "
            "could be taken from them."
        )

    raw = coint_johansen(arr, det_order, k_ar_diff)

    index = ["r = 0"] + [f"r <= {r}" for r in range(1, n_vars)]
    trace_stat = pd.Series(np.asarray(raw.lr1, dtype=float), index=index, name="trace")
    maxeig_stat = pd.Series(
        np.asarray(raw.lr2, dtype=float), index=index, name="maxeig"
    )
    cols = [_CV_COLUMNS[a] for a in _ALPHAS]
    trace_cv = pd.DataFrame(
        np.asarray(raw.cvt, dtype=float)[:, cols], index=index, columns=list(_ALPHAS)
    )
    maxeig_cv = pd.DataFrame(
        np.asarray(raw.cvm, dtype=float)[:, cols], index=index, columns=list(_ALPHAS)
    )

    # Each cointegrating vector is identified only up to scale: normalise
    # by its first element so two runs can be compared. A first element
    # numerically indistinguishable from zero is left unscaled rather
    # than divided by, which would manufacture huge coefficients.
    evec = np.asarray(raw.evec, dtype=float)
    beta = evec.copy()
    for j in range(beta.shape[1]):
        head = beta[0, j]
        if abs(head) > 1e-12:
            beta[:, j] = beta[:, j] / head
    beta_df = pd.DataFrame(
        beta[: len(names), :],
        index=list(names),
        columns=[f"beta{j + 1}" for j in range(beta.shape[1])],
    )

    stat = trace_stat if method == "trace" else maxeig_stat
    cv = trace_cv if method == "trace" else maxeig_cv
    selected = _sequential_rank(stat.to_numpy(), cv[alpha].to_numpy())

    return JohansenResults(
        names=names,
        trace_stat=trace_stat,
        maxeig_stat=maxeig_stat,
        trace_cv=trace_cv,
        maxeig_cv=maxeig_cv,
        eigenvalues=np.asarray(raw.eig, dtype=float),
        beta=beta_df,
        selected_rank=selected,
        n_vars=n_vars,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        alpha=alpha,
        method=method,
        _raw=raw,
    )


def check_no_cointegration_among_x(
    x: npt.ArrayLike,
    det_order: int = 0,
    k_ar_diff: int = 1,
    alpha: float = 0.05,
    method: Method = "trace",
) -> JohansenResults | None:
    r"""Check the bounds test's assumption that the regressors are not
    cointegrated among themselves.

    The ARDL bounds framework conditions on the regressors and tests for
    **one** long-run relationship, the one involving ``y``. If the
    regressors are cointegrated among themselves, that premise fails:
    the system has more relations than the single-equation setup can
    represent, and the test's distribution is not the tabulated one. The
    bounds test itself gives no sign of this — which is why it takes a
    separate check.

    Parameters
    ----------
    x : array_like
        The regressors, ``(T, k)``. With a single regressor there is
        nothing to check and ``None`` is returned.
    det_order, k_ar_diff, alpha, method
        Passed through to :func:`johansen`.

    Returns
    -------
    JohansenResults or None
        The full test result, so the caller can inspect it, or ``None``
        when there is nothing to test (``k < 2``).

    Warns
    -----
    PyardlMethodologyWarning
        If the retained rank is greater than zero.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> n = 200
    >>> a = np.cumsum(rng.normal(size=n))
    >>> b = np.cumsum(rng.normal(size=n))
    >>> res = check_no_cointegration_among_x(pd.DataFrame({"a": a, "b": b}))
    >>> res.selected_rank
    0
    """
    arr = x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)
    if arr.ndim == 1 or arr.shape[1] < 2:
        return None

    res = johansen(
        x, det_order=det_order, k_ar_diff=k_ar_diff, alpha=alpha, method=method
    )
    if res.selected_rank > 0:
        warnings.warn(
            f"The Johansen test finds {res.selected_rank} cointegrating "
            f"relation(s) AMONG the regressors at {alpha:.0%}. The bounds "
            "test assumes there are none: it conditions on the regressors "
            "and tests for a single relation, so its critical values do "
            "not apply here. Consider a system approach (VECM), or drop "
            "the redundant regressor.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )
    return res
