r"""Partial-sum decomposition — the numerical core of the NARDL.

Shin, Yu & Greenwood-Nimmo (2014) split a regressor into the part built
by its rises and the part built by its falls:

.. math::

    x_t^{+} = \sum_{s \le t} \max(\Delta x_s - c,\, 0), \qquad
    x_t^{-} = \sum_{s \le t} \min(\Delta x_s - c,\, 0),

with :math:`x_0^{+} = x_0^{-} = 0`. Everything else in the framework —
estimation, Wald tests, bounds test — is standard OLS on the model built
from these two series. So this decomposition is the one place where a
silent error would propagate into every NARDL result, and it is checked
by an exact identity rather than by inspection:

.. math::

    x_t = x_0 + x_t^{+} + x_t^{-} + c\,t.

**The threshold is not free.** At the default ``c = 0`` the identity
reduces to :math:`x = x_0 + x^{+} + x^{-}`: the decomposition is a
regrouping of the same information, nothing is created or lost. At any
other threshold the two partial sums no longer add back to the series —
they add back to the series *minus a linear drift* :math:`c\,t`. That
drift does not disappear; it moves into the deterministic part of the
model, and reading :math:`\theta^{+}` as a long-run response then means
reading it net of a trend nobody declared. The library therefore
computes any threshold you ask for, reports it on the result, and warns
when a non-zero one is used.

Greenwood-Nimmo, Shin, van Treeck & Yu (2013) generalise the same idea
to more than two regimes: small rises, large rises, small falls, large
falls, split by one or more magnitude thresholds
:math:`0 < c_1 < \dots < c_K`. :func:`partial_sums_multi` implements
that decomposition directly, rather than through the differencing
recipe (``partial_sums`` at two thresholds, subtracted) that a caller
would otherwise have to build by hand — see the module docstring of
that function for the per-period band formula and its identity.

References
----------
.. [1] Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). Modelling
       asymmetric cointegration and dynamic multipliers in a nonlinear
       ARDL framework. In *Festschrift in Honor of Peter Schmidt*
       (pp. 281-314). Springer.
.. [2] Greenwood-Nimmo, M., Shin, Y., van Treeck, T. & Yu, B. (2013).
       The decoupling of monetary policy from long-term rates in the
       U.S. and Germany. Working paper.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning

__all__ = [
    "decomposition_error",
    "multi_decomposition_error",
    "partial_sums",
    "partial_sums_multi",
]

Threshold = float | Literal["mean"]


def partial_sums(
    x: npt.ArrayLike,
    threshold: Threshold = 0.0,
    name: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    r"""Split a series into its cumulated rises and cumulated falls.

    Parameters
    ----------
    x : array_like
        The series to decompose. A :class:`pandas.Series` keeps its index
        and lends its name to the outputs.
    threshold : float or {'mean'}, default 0.0
        The threshold ``c`` applied to the first differences. ``'mean'``
        uses the sample mean of :math:`\Delta x`. Any non-zero threshold
        introduces a deterministic drift — see the module docstring — and
        raises a :class:`~pyardl.exceptions.PyardlMethodologyWarning`.
    name : str, optional
        Base name for the outputs; defaults to the series name, or
        ``'x'``.

    Returns
    -------
    x_pos, x_neg : pandas.Series
        The two partial sums, named ``<name>_pos`` and ``<name>_neg``,
        both starting at zero and aligned on the input index.

    Raises
    ------
    ValueError
        If ``x`` is not one-dimensional, holds fewer than two
        observations, or contains non-finite values. A NaN would
        propagate through the cumulative sum and silently poison every
        observation after it.

    Examples
    --------
    >>> import pandas as pd
    >>> x = pd.Series([1.0, 3.0, 2.0, 5.0], name="oil")
    >>> pos, neg = partial_sums(x)
    >>> pos.tolist()
    [0.0, 2.0, 2.0, 5.0]
    >>> neg.tolist()
    [0.0, 0.0, -1.0, -1.0]
    >>> (x.iloc[0] + pos + neg).tolist() == x.tolist()
    True
    """
    if isinstance(x, pd.Series):
        index: pd.Index | None = x.index
        base = str(x.name) if x.name is not None else "x"
        arr = x.to_numpy(dtype=np.float64)
    else:
        index = None
        base = "x"
        arr = np.asarray(x, dtype=np.float64)

    if name is not None:
        base = str(name)
    if arr.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {arr.shape}.")
    if arr.size < 2:
        raise ValueError(
            f"The decomposition needs at least two observations, got {arr.size}: "
            "with one there is no change to classify as a rise or a fall."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "x contains NaN or infinite values. A cumulative sum would carry "
            "them into every later observation, so they are refused here "
            "rather than propagated."
        )

    delta = np.diff(arr)
    c = float(delta.mean()) if threshold == "mean" else float(threshold)
    if c != 0.0:
        warnings.warn(
            f"threshold={c:.6g} is not zero: the partial sums then add back "
            "to x minus a linear drift c*t, not to x. That drift belongs to "
            "the deterministic part of the model, so the long-run "
            "coefficients are read net of a trend. Use threshold=0.0 unless "
            "you mean this.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    centred = delta - c
    pos = np.concatenate([[0.0], np.cumsum(np.maximum(centred, 0.0))])
    neg = np.concatenate([[0.0], np.cumsum(np.minimum(centred, 0.0))])

    x_pos = pd.Series(pos, index=index, name=f"{base}_pos")
    x_neg = pd.Series(neg, index=index, name=f"{base}_neg")
    return x_pos, x_neg


def decomposition_error(
    x: npt.ArrayLike,
    x_pos: npt.ArrayLike,
    x_neg: npt.ArrayLike,
    threshold: float = 0.0,
) -> float:
    r"""Largest violation of :math:`x_t = x_0 + x^{+}_t + x^{-}_t + c\,t`.

    The identity the whole framework rests on, returned as a number so it
    can be asserted rather than eyeballed.

    Returns
    -------
    float
        The maximum absolute deviation, which should sit at rounding
        level (below 1e-12 on any realistic series).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.cumsum(rng.normal(size=100))
    >>> pos, neg = partial_sums(x)
    >>> bool(decomposition_error(x, pos, neg) < 1e-12)
    True
    """
    arr = np.asarray(x, dtype=np.float64)
    pos = np.asarray(x_pos, dtype=np.float64)
    neg = np.asarray(x_neg, dtype=np.float64)
    if not (arr.shape == pos.shape == neg.shape):
        raise ValueError(
            f"Shapes differ: x {arr.shape}, x_pos {pos.shape}, x_neg {neg.shape}."
        )
    drift = float(threshold) * np.arange(arr.size, dtype=np.float64)
    return float(np.max(np.abs(arr - (arr[0] + pos + neg + drift))))


def partial_sums_multi(
    x: npt.ArrayLike,
    thresholds: Sequence[float],
    name: str | None = None,
) -> pd.DataFrame:
    r"""Split a series into more than two cumulated regimes.

    The two-regime decomposition of :func:`partial_sums` assigns every
    rise to :math:`x^{+}` and every fall to :math:`x^{-}`, whatever their
    size. Greenwood-Nimmo, Shin, van Treeck & Yu (2013) split each side
    further, by one or more magnitude thresholds
    :math:`0 = c_0 < c_1 < \dots < c_K`, into bands: small rises, large
    rises (and the mirror image on the fall side). Band :math:`j` on the
    rise side absorbs the part of a period's change that falls between
    :math:`c_{j-1}` and :math:`c_j`, per period:

    .. math::

        b^{+}_{j,t} = \max\bigl(\min(\Delta x_t, c_j) - c_{j-1},\; 0\bigr),
        \qquad j = 1, \dots, K,

        b^{+}_{K+1,t} = \max(\Delta x_t - c_K,\, 0),

    and symmetrically on the fall side with :math:`-c_j` in place of
    :math:`c_j`. A rise of, say, 0.02 against a single threshold
    :math:`c_1 = 0.01` splits into 0.01 in the first band (its share of
    the rise up to the threshold) and 0.01 in the second (the excess
    beyond it) — the two add back to the full 0.02, band by band, which
    is what :func:`multi_decomposition_error` checks. Cumulating each
    band over time gives :math:`K+1` rising series and :math:`K+1`
    falling series; passing all :math:`2(K+1)` as ordinary regressors is
    the whole of the extension, since nothing about the ARDL that
    consumes them needs to change (see the module docstring of
    :func:`partial_sums`, and the multiple-asymmetries example on the
    project's documentation site).

    A single threshold reproduces :func:`partial_sums` split by
    magnitude around ``thresholds[0]`` rather than by sign alone; the
    band boundaries at 0 are exactly the two series that function
    returns when its own ``threshold`` stays at the default of 0.0.

    Parameters
    ----------
    x : array_like
        The series to decompose. A :class:`pandas.Series` keeps its
        index and lends its name to the outputs.
    thresholds : sequence of float
        The band boundaries :math:`c_1 < \dots < c_K`, strictly
        increasing and strictly positive. ``K`` thresholds produce
        ``K + 1`` bands per side.
    name : str, optional
        Base name for the outputs; defaults to the series name, or
        ``'x'``.

    Returns
    -------
    pandas.DataFrame
        One column per band, named ``<name>_pos_1`` .. ``<name>_pos_{K+1}``
        and ``<name>_neg_1`` .. ``<name>_neg_{K+1}``, ordered from the
        smallest band to the unbounded one on each side, aligned on the
        input index.

    Raises
    ------
    ValueError
        If ``thresholds`` is empty, not strictly increasing, or contains
        a non-positive value — or if ``x`` fails the same checks as in
        :func:`partial_sums`.

    Examples
    --------
    >>> import pandas as pd
    >>> x = pd.Series([1.0, 1.005, 1.03, 1.02, 0.98], name="oil")
    >>> bands = partial_sums_multi(x, thresholds=[0.01])
    >>> list(bands.columns)
    ['oil_pos_1', 'oil_pos_2', 'oil_neg_1', 'oil_neg_2']
    >>> bands["oil_pos_1"].round(4).tolist()
    [0.0, 0.005, 0.015, 0.015, 0.015]
    >>> bands["oil_pos_2"].round(4).tolist()
    [0.0, 0.0, 0.015, 0.015, 0.015]
    >>> reconstructed = x.iloc[0] + np.sum(bands.to_numpy(), axis=1)
    >>> bool(np.max(np.abs(reconstructed - x.to_numpy())) < 1e-12)
    True
    """
    if isinstance(x, pd.Series):
        index: pd.Index | None = x.index
        base = str(x.name) if x.name is not None else "x"
        arr = x.to_numpy(dtype=np.float64)
    else:
        index = None
        base = "x"
        arr = np.asarray(x, dtype=np.float64)

    if name is not None:
        base = str(name)
    if arr.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {arr.shape}.")
    if arr.size < 2:
        raise ValueError(
            f"The decomposition needs at least two observations, got {arr.size}: "
            "with one there is no change to classify into a band."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "x contains NaN or infinite values. A cumulative sum would carry "
            "them into every later observation, so they are refused here "
            "rather than propagated."
        )

    c = [float(v) for v in thresholds]
    if len(c) == 0:
        raise ValueError("thresholds must contain at least one value.")
    if any(v <= 0.0 for v in c):
        raise ValueError(
            f"thresholds must be strictly positive, got {c}. The bottom band "
            "always starts at 0 implicitly; a non-positive entry would either "
            "duplicate that boundary or invert the ordering."
        )
    if any(later <= earlier for earlier, later in zip(c, c[1:], strict=False)):
        raise ValueError(f"thresholds must be strictly increasing, got {c}.")

    delta = np.diff(arr)
    edges = [0.0, *c]
    k = len(c)

    pos_cols: list[pd.Series] = []
    neg_cols: list[pd.Series] = []
    for j in range(1, k + 1):
        lo, hi = edges[j - 1], edges[j]
        band_pos = np.maximum(np.minimum(delta, hi) - lo, 0.0)
        band_neg = np.minimum(np.maximum(delta, -hi) + lo, 0.0)
        pos_cols.append(
            pd.Series(
                np.concatenate([[0.0], np.cumsum(band_pos)]),
                index=index,
                name=f"{base}_pos_{j}",
            )
        )
        neg_cols.append(
            pd.Series(
                np.concatenate([[0.0], np.cumsum(band_neg)]),
                index=index,
                name=f"{base}_neg_{j}",
            )
        )

    top_pos = np.maximum(delta - edges[k], 0.0)
    top_neg = np.minimum(delta + edges[k], 0.0)
    pos_cols.append(
        pd.Series(
            np.concatenate([[0.0], np.cumsum(top_pos)]),
            index=index,
            name=f"{base}_pos_{k + 1}",
        )
    )
    neg_cols.append(
        pd.Series(
            np.concatenate([[0.0], np.cumsum(top_neg)]),
            index=index,
            name=f"{base}_neg_{k + 1}",
        )
    )

    return pd.concat(pos_cols + neg_cols, axis=1)


def multi_decomposition_error(x: npt.ArrayLike, bands: pd.DataFrame) -> float:
    r"""Largest violation of :math:`x_t = x_0 + \sum_j b_{j,t}`.

    The many-regime analogue of :func:`decomposition_error`: every band
    returned by :func:`partial_sums_multi` should sum back to the
    original series exactly, with no threshold-driven drift term, since
    the bottom edge of the bottom band is always 0.

    Returns
    -------
    float
        The maximum absolute deviation, which should sit at rounding
        level (below 1e-12 on any realistic series).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = np.cumsum(rng.normal(scale=0.02, size=200))
    >>> bands = partial_sums_multi(x, thresholds=[0.01, 0.02])
    >>> bool(multi_decomposition_error(x, bands) < 1e-12)
    True
    """
    arr = np.asarray(x, dtype=np.float64)
    total = np.sum(bands.to_numpy(dtype=np.float64), axis=1)
    if total.shape != arr.shape:
        raise ValueError(f"Shapes differ: x {arr.shape}, bands sum {total.shape}.")
    return float(np.max(np.abs(arr - (arr[0] + total))))
