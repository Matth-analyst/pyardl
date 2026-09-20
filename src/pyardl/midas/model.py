r"""MIDAS-ARDL: mixed sampling frequencies (Ghysels, Santa-Clara & Valkanov 2004).

The rest of the library assumes :math:`y` and :math:`x` are observed at
the **same** frequency. In practice a quarterly :math:`y` (GDP) often
comes with a monthly or daily :math:`x` (equity indices, confidence
surveys) — aggregating :math:`x` down to :math:`y`'s frequency (a
monthly-to-quarterly average) throws away information and imposes equal
weights on every observation within the period, an assumption never
tested. MIDAS answers by keeping :math:`x` at its native frequency and
modelling the weights linking its fine observations to each aggregated
:math:`y` observation as a low-dimensional parametric function — the
same idea as the Almon weights (:mod:`pyardl.distributed_lags.almon`)
already in the library, applied here to a frequency problem rather than
a classical finite lag.

Estimation is concentrated, the same trick as STAR
(:mod:`pyardl.threshold.star`): for *fixed* weight parameters
:math:`\theta`, the aggregated regressor :math:`z_t = \sum_k b(k;
\theta) x_{t - k/m}` is just a number for every :math:`t`, and the
outer UECM in :math:`(y_{t-1}, z_{t-1}, \dots)` is then ordinary linear
least squares — so only :math:`\theta` (two parameters, for either
weight form) needs a nonlinear search, seeded from a coarse grid to
avoid local optima.

References
----------
.. [1] Ghysels, E., Santa-Clara, P. & Valkanov, R. (2004). The MIDAS
       touch: mixed data sampling regression models. Working paper,
       UNC/UCLA.
.. [2] Ghysels, E., Sinko, A. & Valkanov, R. (2007). MIDAS regressions:
       further results and new directions. *Econometric Reviews*,
       26(1), 53-90.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln

from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

WeightForm = Literal["almon_exp", "beta"]
Det = Literal["none", "const", "trend"]

__all__ = ["midas_weights", "MIDASARDL", "MIDASARDLResults"]


def midas_weights(
    theta: ArrayLike, k_max: int, form: WeightForm = "almon_exp"
) -> FloatArray:
    r"""MIDAS weight function :math:`b(k; \theta)`, :math:`k = 0, \dots, k_{\max}`.

    Parameters
    ----------
    theta : array_like, length 2
        Shape parameters.
    k_max : int
        Highest lag included.
    form : {'almon_exp', 'beta'}, default 'almon_exp'
        ``'almon_exp'``: :math:`b(k) \propto \exp(\theta_1 k + \theta_2
        k^2)` — always positive, the form most used in applied
        macro-finance. A rigid special case of the polynomial Almon
        weights (:func:`pyardl.distributed_lags.almon`, spec 02 §1):
        exponential rather than polynomial in :math:`k`, chosen because
        the exponential form is what the MIDAS literature standardised
        on, not a re-derivation of the same idea under a new name.
        ``'beta'`` (Ghysels et al. 2004): weights built from a Beta
        density on ``[0, 1]``, two shape parameters.

    Returns
    -------
    ndarray, shape (k_max + 1,)
        Normalised to sum to 1.

    Examples
    --------
    >>> import numpy as np
    >>> w = midas_weights([0.0, -0.02], k_max=5, form="almon_exp")
    >>> bool(np.isclose(w.sum(), 1.0))
    True
    >>> bool(w[0] > w[-1])
    True
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.shape != (2,):
        raise ValueError(f"theta must have length 2, got shape {theta_arr.shape}.")
    if k_max < 0:
        raise ValueError(f"k_max={k_max} must be non-negative.")
    k = np.arange(k_max + 1, dtype=np.float64)

    if form == "almon_exp":
        arg = theta_arr[0] * k + theta_arr[1] * k**2
        arg = arg - arg.max()
        raw = np.exp(arg)
    elif form == "beta":
        a, b = max(theta_arr[0], 1e-3), max(theta_arr[1], 1e-3)
        x = (k + 0.5) / (k_max + 1)
        x = np.clip(x, 1e-6, 1 - 1e-6)
        log_raw = (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - betaln(a, b)
        log_raw = log_raw - log_raw.max()
        raw = np.exp(log_raw)
    else:
        raise ValueError(f"form={form!r} must be 'almon_exp' or 'beta'.")

    total = raw.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(k_max + 1, 1.0 / (k_max + 1))
    return np.asarray(raw / total, dtype=np.float64)


def _aggregate(
    x_hf: FloatArray, freq_ratio: int, k_max: int, theta: FloatArray, form: WeightForm
) -> FloatArray:
    """z_t = sum_k b(k;theta) x_{t*m - k}, for every low-frequency t with a
    full history window; NaN when the window would run before index 0.
    """
    w = midas_weights(theta, k_max, form)
    n_hf = x_hf.shape[0]
    n_lf = n_hf // freq_ratio
    z = np.full(n_lf, np.nan, dtype=np.float64)
    for t in range(n_lf):
        hf_idx = (t + 1) * freq_ratio - 1
        if hf_idx - k_max < 0:
            continue
        window = x_hf[hf_idx - k_max : hf_idx + 1][::-1]
        z[t] = float(w @ window)
    return z


@dataclass(frozen=True)
class MIDASARDLResults:
    """Outcome of a MIDAS-ARDL fit.

    Attributes
    ----------
    theta_hat : ndarray, shape (2,)
        Fitted weight-function parameters.
    weights : ndarray, shape (k_max + 1,)
        :math:`b(k; \\hat\\theta)`.
    params : pandas.Series
        Coefficients of the outer UECM (linear given ``theta_hat``).
    longrun : float
        :math:`\\theta = -\\gamma/\\lambda`, the long-run coefficient on
        the aggregated regressor.
    ssr : float
    form : str
    k_max : int
    freq_ratio : int
    n_obs : int
    grid_start : tuple of float
    """

    theta_hat: FloatArray
    weights: FloatArray
    params: pd.Series
    longrun: float
    ssr: float
    form: WeightForm
    k_max: int
    freq_ratio: int
    n_obs: int
    grid_start: tuple[float, float]

    def summary(self) -> str:
        """Readable report of the fit."""
        lines = [
            f"MIDAS-ARDL ({self.form}, Ghysels et al.) - freq_ratio={self.freq_ratio}, "
            f"k_max={self.k_max}, nobs={self.n_obs}",
            f"  theta_hat = [{self.theta_hat[0]:.4f}, {self.theta_hat[1]:.4f}]",
            f"  longrun theta (on the aggregated regressor) = {self.longrun:.4f}",
            f"  SSR = {self.ssr:.4f}",
        ]
        return "\n".join(lines)


def _uecm_ssr_and_params(
    y: FloatArray, z: FloatArray, order: tuple[int, int], det: Det
) -> tuple[float, FloatArray, list[str]]:
    """UECM(y, z): lambda*y_{t-1} + gamma*z_{t-1} + short-run + det."""
    p, q = order
    start = max(p, q, 1)
    n = y.shape[0]
    dy = np.diff(y)
    dz = np.diff(z)
    target = dy[start - 1 :]

    cols = [y[start - 1 : n - 1], z[start - 1 : n - 1]]
    names = ["y.L1", "z.L1"]
    if det in ("const", "trend"):
        cols.append(np.ones_like(target))
        names.append("const")
    if det == "trend":
        cols.append(np.arange(1, target.shape[0] + 1, dtype=np.float64))
        names.append("trend")
    for i in range(1, p):
        cols.append(dy[start - i - 1 : n - i - 1])
        names.append(f"D.y.L{i}")
    for i in range(q):
        cols.append(dz[start - i - 1 : n - i - 1])
        names.append(f"D.z.L{i + 1}")

    design = np.column_stack(cols)
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    return float(resid @ resid), beta, names


def _ssr_given_theta(
    theta: FloatArray,
    y: FloatArray,
    x_hf: FloatArray,
    freq_ratio: int,
    k_max: int,
    order: tuple[int, int],
    det: Det,
    form: WeightForm,
) -> float:
    z = _aggregate(x_hf, freq_ratio, k_max, theta, form)
    valid = ~np.isnan(z)
    if valid.sum() < 20:
        return np.inf
    y_use = y[: z.shape[0]][valid]
    z_use = z[valid]
    if y_use.shape[0] < 10:
        return np.inf
    ssr, _, _ = _uecm_ssr_and_params(y_use, z_use, order, det)
    return ssr


class MIDASARDL:
    r"""MIDAS-ARDL: an error-correction model with a mixed-frequency regressor.

    Parameters
    ----------
    y : array_like, shape (T_low,)
        Low-frequency dependent variable.
    x_high_freq : array_like, shape (T_low * freq_ratio,)
        High-frequency regressor, ordered so that
        ``x_high_freq[(t+1)*freq_ratio - 1]`` is the last high-frequency
        observation within low-frequency period ``t``.
    freq_ratio : int
        Ratio of frequencies (``m`` in the spec — e.g. 3 for
        quarterly/monthly).
    k_max : int, default 12
        Highest high-frequency lag included in the aggregation window.
        A modelling choice, not selected automatically — see the module
        notes.
    form : {'almon_exp', 'beta'}, default 'almon_exp'
    order : tuple of int, default (1, 1)
        ``(p, q)``: lags of :math:`\Delta y` and of the aggregated
        regressor's own difference in the outer UECM.
    det : {'none', 'const', 'trend'}, default 'const'
    start_grid : int, default 10
        Grid resolution per dimension for the coarse :math:`\theta`
        search seeding the local optimiser.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.midas import MIDASARDL
    >>> rng = np.random.default_rng(0)
    >>> n_lf, m = 150, 3
    >>> x_hf = np.cumsum(rng.standard_normal(n_lf * m))
    >>> w_true = midas_weights([0.0, -0.15], k_max=8, form="almon_exp")
    >>> z_true = np.full(n_lf, np.nan)
    >>> for t in range(n_lf):
    ...     idx = (t + 1) * m - 1
    ...     if idx - 8 >= 0:
    ...         z_true[t] = w_true @ x_hf[idx - 8 : idx + 1][::-1]
    >>> y = np.zeros(n_lf)
    >>> for t in range(1, n_lf):
    ...     if not np.isnan(z_true[t - 1]):
    ...         y[t] = y[t - 1] - 0.5 * (y[t - 1] - 1.5 * z_true[t - 1])
    ...         y[t] += rng.standard_normal() * 0.3
    ...     else:
    ...         y[t] = y[t - 1]
    >>> res = MIDASARDL(y, x_hf, freq_ratio=m, k_max=8, order=(1, 1)).fit()
    >>> res.n_obs > 100
    True

    Notes
    -----
    ``midas_weights`` (:func:`pyardl.midas.model.midas_weights`) is
    reused directly from the Almon-exponential idea (spec 02 §1) rather
    than reimplemented; only the aggregation over a high-frequency
    window is new.
    """

    def __init__(
        self,
        y: ArrayLike,
        x_high_freq: ArrayLike,
        freq_ratio: int,
        k_max: int = 12,
        form: WeightForm = "almon_exp",
        order: tuple[int, int] = (1, 1),
        det: Det = "const",
        start_grid: int = 10,
    ) -> None:
        if freq_ratio < 1:
            raise ValueError(f"freq_ratio={freq_ratio} must be >= 1.")
        if k_max < 0:
            raise ValueError(f"k_max={k_max} must be non-negative.")
        if form not in ("almon_exp", "beta"):
            raise ValueError(f"form={form!r} must be 'almon_exp' or 'beta'.")
        if det not in ("none", "const", "trend"):
            raise ValueError(f"det={det!r} must be 'none', 'const' or 'trend'.")

        y_arr, _, _, _, _ = check_series(y, None)
        x_hf = np.asarray(x_high_freq, dtype=np.float64).ravel()
        if x_hf.shape[0] != y_arr.shape[0] * freq_ratio:
            raise ValueError(
                f"x_high_freq must have length y * freq_ratio: expected "
                f"{y_arr.shape[0] * freq_ratio}, got {x_hf.shape[0]}."
            )

        self.y = y_arr
        self.x_hf = x_hf
        self.freq_ratio = int(freq_ratio)
        self.k_max = int(k_max)
        self.form: WeightForm = form
        self.order = order
        self.det: Det = det
        self.start_grid = int(start_grid)

    def fit(self) -> MIDASARDLResults:
        """Concentrated NLS fit: grid-seeded Nelder-Mead over theta.

        Returns
        -------
        MIDASARDLResults
        """
        if self.form == "almon_exp":
            grid1 = np.linspace(-0.5, 0.5, self.start_grid)
            grid2 = np.linspace(-0.5, 0.0, self.start_grid)
        else:
            grid1 = np.linspace(0.5, 5.0, self.start_grid)
            grid2 = np.linspace(0.5, 5.0, self.start_grid)

        best_ssr = np.inf
        best_start = (float(grid1[0]), float(grid2[0]))
        for t1 in grid1:
            for t2 in grid2:
                ssr = _ssr_given_theta(
                    np.array([t1, t2]),
                    self.y,
                    self.x_hf,
                    self.freq_ratio,
                    self.k_max,
                    self.order,
                    self.det,
                    self.form,
                )
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_start = (float(t1), float(t2))

        result = minimize(
            _ssr_given_theta,
            x0=np.array(best_start),
            args=(
                self.y,
                self.x_hf,
                self.freq_ratio,
                self.k_max,
                self.order,
                self.det,
                self.form,
            ),
            method="Nelder-Mead",
        )
        theta_hat = np.asarray(result.x, dtype=np.float64)

        z = _aggregate(self.x_hf, self.freq_ratio, self.k_max, theta_hat, self.form)
        valid = ~np.isnan(z)
        y_use = self.y[: z.shape[0]][valid]
        z_use = z[valid]
        ssr, beta, names = _uecm_ssr_and_params(y_use, z_use, self.order, self.det)

        lam = beta[names.index("y.L1")]
        gamma = beta[names.index("z.L1")]
        longrun = -gamma / lam if lam != 0 else float("nan")

        return MIDASARDLResults(
            theta_hat=theta_hat,
            weights=midas_weights(theta_hat, self.k_max, self.form),
            params=pd.Series(beta, index=names, name="coef"),
            longrun=float(longrun),
            ssr=ssr,
            form=self.form,
            k_max=self.k_max,
            freq_ratio=self.freq_ratio,
            n_obs=int(y_use.shape[0]),
            grid_start=best_start,
        )
