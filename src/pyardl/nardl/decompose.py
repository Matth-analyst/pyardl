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

References
----------
.. [1] Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). Modelling
       asymmetric cointegration and dynamic multipliers in a nonlinear
       ARDL framework. In *Festschrift in Honor of Peter Schmidt*
       (pp. 281-314). Springer.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning

__all__ = ["decomposition_error", "partial_sums"]

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
