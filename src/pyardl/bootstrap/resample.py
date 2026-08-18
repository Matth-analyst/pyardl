r"""Residual resampling schemes for the bootstrap.

Two schemes, and one rule that governs both.

**The rule: resample by date, not by equation.** The bootstrap
regenerates a whole system — the regressors from their own marginal
model, then the dependent variable conditionally on them. The residuals
of those equations are contemporaneously correlated, and that
correlation is part of the data-generating process. Drawing each
equation's residuals independently would destroy it and produce
bootstrap samples that are *more* favourable to the null than the data
warrant, inflating the critical values in the optimistic direction.

So a draw picks a **row index** :math:`t`, and the whole residual vector
:math:`(\tilde\varepsilon_t, \tilde\eta_t')` travels together.

**iid** — the ordinary residual bootstrap: sample rows with replacement.
Valid when the errors are homoskedastic.

**wild** — multiply each residual row by a Rademacher draw
:math:`\pm 1`, the same sign for the whole row. This preserves whatever
heteroskedasticity is attached to each date, because a row keeps its own
scale, and only its sign changes. Use it when the residuals show
volatility clustering or a variance break.

References
----------
.. [1] McNown, R., Sam, C. Y. & Goh, S. K. (2018). Bootstrapping the
       autoregressive distributed lag test for cointegration.
       *Applied Economics*, 50(13), 1509-1521.
.. [2] Liu, R. Y. (1988). Bootstrap procedures under some non-i.i.d.
       models. *The Annals of Statistics*, 16(4), 1696-1708.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
    IntArray = NDArray[np.intp]

ResampleScheme = Literal["iid", "wild"]

__all__ = ["draw_indices", "resample_residuals", "ResampleScheme"]


def draw_indices(n_obs: int, size: int, rng: np.random.Generator) -> IntArray:
    """Row indices drawn with replacement.

    Parameters
    ----------
    n_obs : int
        Number of residual rows available.
    size : int
        Number of draws.
    rng : numpy.random.Generator
        Explicit generator. There is no module-level random state
        anywhere in this library: reproducibility is a property of the
        call, not of the process.

    Returns
    -------
    numpy.ndarray
        Integer indices in ``[0, n_obs)``.
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs={n_obs} must be positive.")
    return rng.integers(0, n_obs, size=size, dtype=np.intp)


def resample_residuals(
    residuals: FloatArray,
    n_draw: int,
    rng: np.random.Generator,
    scheme: ResampleScheme = "iid",
) -> FloatArray:
    r"""Draw a bootstrap sample of residual rows.

    Parameters
    ----------
    residuals : numpy.ndarray, shape (n, m)
        Centred residuals, one column per equation. Rows are dates.
    n_draw : int
        Number of rows to draw.
    rng : numpy.random.Generator
    scheme : {'iid', 'wild'}, default 'iid'
        ``'iid'`` samples rows with replacement. ``'wild'`` samples rows
        *and* flips the sign of each drawn row with probability one half.

    Returns
    -------
    numpy.ndarray, shape (n_draw, m)

    Notes
    -----
    Whole rows move together, so the contemporaneous covariance across
    equations is preserved by construction — see the module docstring
    for why that matters.

    Under ``'wild'`` the sign is drawn once per row, not per element.
    Flipping elements independently would break the same covariance the
    row-wise draw is there to protect.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.bootstrap.resample import resample_residuals
    >>> res = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    >>> rng = np.random.default_rng(0)
    >>> sample = resample_residuals(res, 4, rng)
    >>> sample.shape
    (4, 2)
    >>> # Every drawn row is one of the original rows, intact.
    >>> bool(np.all(sample[:, 1] == 10.0 * sample[:, 0]))
    True
    """
    arr = np.asarray(residuals, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("residuals must be 1-D or 2-D.")
    if scheme not in ("iid", "wild"):
        raise ValueError(f"scheme={scheme!r} is not available. Use 'iid' or 'wild'.")

    idx = draw_indices(arr.shape[0], n_draw, rng)
    drawn = arr[idx]
    if scheme == "wild":
        # One sign per ROW: the whole residual vector of a date keeps its
        # relative structure, only its direction changes.
        signs = rng.integers(0, 2, size=n_draw).astype(np.float64) * 2.0 - 1.0
        drawn = drawn * signs[:, None]
    return np.asarray(drawn, dtype=np.float64)
