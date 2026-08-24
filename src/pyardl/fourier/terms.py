r"""Fourier terms for smooth structural change.

The founding idea of this branch. Rather than dating and counting
breaks, approximate a time-varying deterministic component by a few
low-frequency sinusoids:

.. math::

    d(t) \approx a_0 + \sum_{f} \left[
        a_f \sin\!\left(\frac{2\pi f t}{T}\right)
      + b_f \cos\!\left(\frac{2\pi f t}{T}\right) \right].

A single frequency, two parameters, captures several breaks of unknown
shape and unknown date — provided the change is *smooth*. That proviso
is the whole method: a Fourier component cannot represent a jump, and
fitting one to a sharp break produces a wave that overshoots on both
sides of it.

**Integer and fractional frequencies do not behave alike**, and the
difference is not cosmetic. At an integer frequency the sine and cosine
are exactly orthogonal — to each other, to the constant, and across
frequencies. At a fractional one none of that holds: at ``f = 0.5`` on
200 observations the sine sums to 127 rather than to zero, so the
component is entangled with the intercept and the two split the same
variation between them. The library supports fractional frequencies
because the literature uses them, computes
:func:`fourier_orthogonality` so the entanglement can be seen rather
than guessed at, and says which regime a call is in.

References
----------
.. [1] Becker, R., Enders, W. & Lee, J. (2006). A stationarity test in
       the presence of an unknown number of smooth breaks. *Journal of
       Time Series Analysis*, 27(3), 381-409.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "DEFAULT_GRID",
    "INTEGER_GRID",
    "fourier_orthogonality",
    "fourier_terms",
    "select_frequency",
]

FloatArray = npt.NDArray[np.float64]

#: Fractional grid of the specification: 0.1 to 5.0 in steps of 0.1.
DEFAULT_GRID: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(1, 51))

#: Integer frequencies only, where the terms stay orthogonal.
INTEGER_GRID: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)


def fourier_terms(
    n_obs: int,
    freqs: float | Sequence[float] = 1.0,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    r"""Sine and cosine columns for one or several frequencies.

    Parameters
    ----------
    n_obs : int
        Length of the series, ``T``. The period of frequency ``f`` is
        ``T / f``, so ``f = 1`` completes exactly one cycle over the
        sample.
    freqs : float or sequence of float
        Frequencies, integer or fractional. Must be strictly positive
        and at most ``T / 2`` — beyond the Nyquist limit a sinusoid is
        indistinguishable from a slower one sampled at these dates, so
        asking for it is asking for a column that means something other
        than what it says.
    index : pandas.Index, optional
        Index for the result; defaults to a range starting at 1.

    Returns
    -------
    pandas.DataFrame
        Columns ``sin_f`` and ``cos_f`` per frequency, in the order
        given.

    Raises
    ------
    ValueError
        On a non-positive frequency, one beyond Nyquist, a duplicate, or
        a sample too short to carry the terms.

    Examples
    --------
    >>> terms = fourier_terms(100, 1.0)
    >>> list(terms.columns)
    ['sin_1', 'cos_1']
    >>> bool(abs(terms["sin_1"].sum()) < 1e-10)
    True
    """
    if n_obs < 4:
        raise ValueError(
            f"n_obs={n_obs} is too short for a Fourier component: two "
            "parameters per frequency need observations to be identified."
        )
    if isinstance(freqs, (int, float)):
        grid: tuple[float, ...] = (float(freqs),)
    else:
        grid = tuple(float(f) for f in freqs)
    if not grid:
        raise ValueError("freqs is empty: there is no frequency to build.")
    if len(set(grid)) != len(grid):
        raise ValueError(f"freqs holds duplicates: {grid}.")
    nyquist = n_obs / 2.0
    for f in grid:
        if f <= 0:
            raise ValueError(f"every frequency must be positive, got {f}.")
        if f > nyquist:
            raise ValueError(
                f"frequency {f} exceeds the Nyquist limit {nyquist} for "
                f"{n_obs} observations: sampled at these dates it is "
                "indistinguishable from a slower one."
            )

    t = np.arange(1, n_obs + 1, dtype=np.float64)
    columns: dict[str, FloatArray] = {}
    for f in grid:
        label = f"{f:g}"
        columns[f"sin_{label}"] = np.sin(2.0 * np.pi * f * t / n_obs)
        columns[f"cos_{label}"] = np.cos(2.0 * np.pi * f * t / n_obs)
    return pd.DataFrame(
        columns, index=index if index is not None else pd.RangeIndex(1, n_obs + 1)
    )


def fourier_orthogonality(n_obs: int, freq: float) -> dict[str, float]:
    r"""How far a frequency's terms are from orthogonal.

    Integer frequencies give zeros to rounding level. Fractional ones do
    not, and the size of the departure is what decides whether the
    intercept and the Fourier component are separately interpretable.

    Returns
    -------
    dict
        ``sin_sum`` and ``cos_sum`` (inner products with the constant)
        and ``sin_cos`` (with each other), each scaled by ``n_obs`` so
        the numbers compare across sample sizes.

    Examples
    --------
    >>> out = fourier_orthogonality(200, 1.0)
    >>> bool(abs(out["sin_sum"]) < 1e-12)
    True
    >>> out = fourier_orthogonality(200, 0.5)
    >>> bool(abs(out["sin_sum"]) > 0.1)
    True
    """
    terms = fourier_terms(n_obs, freq)
    sin = terms.iloc[:, 0].to_numpy()
    cos = terms.iloc[:, 1].to_numpy()
    return {
        "sin_sum": float(sin.sum() / n_obs),
        "cos_sum": float(cos.sum() / n_obs),
        "sin_cos": float(sin @ cos / n_obs),
    }


def select_frequency(
    y: npt.ArrayLike,
    grid: Sequence[float] = INTEGER_GRID,
    x: npt.ArrayLike | None = None,
    trend: bool = False,
) -> tuple[float, pd.DataFrame]:
    r"""Pick the frequency that fits the deterministic component best.

    Every candidate is fitted by least squares on a constant, the
    optional trend and extra regressors, and its own pair of Fourier
    terms; the one with the smallest sum of squared residuals wins.

    **The frequency chosen this way is not an ordinary estimate.** Under
    the null that the Fourier terms are absent, ``f`` is not identified —
    there is no true value for it to converge to. That is the Davies
    problem, and it means the usual critical values do not apply to any
    test that follows: they must be simulated with this selection
    *inside* the loop. :func:`pyardl.fourier.tests.fourier_f_test`
    does that; anything else that conditions on the selected frequency
    must too.

    Parameters
    ----------
    y : array_like
        Series whose deterministic component is being approximated.
    grid : sequence of float
        Candidate frequencies. Defaults to the integers 1 to 5, where
        the terms stay orthogonal; :data:`DEFAULT_GRID` offers the
        fractional grid of the specification.
    x : array_like, optional
        Extra regressors held in every candidate.
    trend : bool, default False
        Whether to include a linear trend.

    Returns
    -------
    frequency : float
        The selected frequency.
    table : pandas.DataFrame
        Every candidate with its residual sum of squares, sorted best
        first, so a near-tie is visible instead of hidden behind a single
        number.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(1, 201)
    >>> y = 2.0 + np.sin(2 * np.pi * 2 * t / 200) + 0.01 * np.arange(200)
    >>> freq, table = select_frequency(y)
    >>> float(freq)
    2.0
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_obs = y_arr.size
    base: list[FloatArray] = [np.ones(n_obs)]
    if trend:
        base.append(np.arange(1, n_obs + 1, dtype=np.float64))
    if x is not None:
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if x_arr.shape[0] != n_obs:
            raise ValueError(f"x has {x_arr.shape[0]} rows and y has {n_obs}.")
        base += [np.ascontiguousarray(x_arr[:, j]) for j in range(x_arr.shape[1])]

    rows = []
    for f in grid:
        terms = fourier_terms(n_obs, f).to_numpy()
        design = np.column_stack([*base, terms])
        if design.shape[0] <= design.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
        resid = y_arr - design @ beta
        rows.append({"freq": float(f), "ssr": float(resid @ resid)})
    if not rows:
        raise ValueError(
            f"No candidate frequency could be fitted on {n_obs} observations."
        )
    table = pd.DataFrame(rows).sort_values("ssr").reset_index(drop=True)
    return float(table.loc[0, "freq"]), table
