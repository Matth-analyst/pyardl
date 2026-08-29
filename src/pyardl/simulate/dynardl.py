r"""Stochastic dynamic simulation of an ARDL — Jordan & Philips (2018).

An ARDL coefficient table is close to unreadable as a statement about
the world. The effect of a regressor is spread over its own lags, over
the lagged dependent variable, and over the error-correction term, and
the quantity a reader actually wants — *what happens to y, and when, if
this variable rises* — is nowhere in the table.

This module answers that question directly: hold every regressor at a
baseline, apply a counterfactual shock to one of them, and run the ARDL
recursion forward. Confidence bands come from drawing parameter vectors
from :math:`N(\hat\theta, \hat V)` and repeating the whole trajectory
for each draw.

Two things about the construction are worth stating up front, because
both are choices and both are visible in the output.

**The response is a paired difference.** Each draw produces two
trajectories — one with the shock, one without — and the reported
response is their difference, draw by draw. Anything common to the two
branches cancels exactly rather than approximately, which is why the
bands are as tight as they are.

**Forecast uncertainty cancels out of the response, exactly.** With
``stochastic=True`` the recursion adds innovations
:math:`\varepsilon_t \sim N(0, \hat\sigma^2)`. They are drawn once per
draw and used in *both* branches: an innovation is a property of the
world, not of the intervention. Because the model is linear in ``y``,
the difference between the two branches does not depend on them at all
— the response columns come out identical to ``stochastic=False``, to
machine precision, and the test suite checks it. What widens is the band
on the *level*, which is the honest place for forecast uncertainty to
show up.

References
----------
.. [1] Jordan, S. & Philips, A. Q. (2018). Cointegration testing and
       dynamic simulations of autoregressive distributed lag models.
       *The Stata Journal*, 18(4), 902-923.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning

__all__ = ["DynardlSimulation", "dynardl_simulate"]

FloatArray = npt.NDArray[np.float64]
ShockType = Literal["step", "impulse"]
StartRule = Literal["mean", "last"]


def _baseline_values(
    data: FloatArray,
    names: tuple[str, ...],
    start: StartRule,
    scenario: dict[str, float] | None,
) -> FloatArray:
    """Level each regressor is held at when nothing shocks it."""
    if start == "mean":
        base = data.mean(axis=0)
    elif start == "last":
        base = data[-1].copy()
    else:
        raise ValueError(f'start must be "mean" or "last", got {start!r}.')
    base = np.asarray(base, dtype=np.float64)
    if scenario:
        for name, value in scenario.items():
            if name not in names:
                raise KeyError(
                    f"{name!r} is not a regressor of the model; the "
                    f"scenario can only set {list(names)}."
                )
            base[names.index(name)] = float(value)
    return base


@dataclass(frozen=True)
class DynardlSimulation:
    """Result of :func:`dynardl_simulate`.

    Attributes
    ----------
    summary_df : pandas.DataFrame
        Indexed by horizon, with two column blocks. ``response`` is the
        paired difference from the no-shock counterfactual — the causal
        reading — and ``level`` is the simulated path of ``y`` itself.
        Each block carries ``point`` (the trajectory at
        :math:`\\hat\\theta`), ``mean`` (across draws) and the band
        bounds.
    equilibrium : float
        The value ``y`` is started from and, with ``det="const"``, stays
        at when nothing shocks it: :math:`\\hat y^{*}`.
    longrun_target : float
        Where a permanent step sends the response:
        :math:`\\hat\\theta_j \\cdot \\Delta x`. ``nan`` for an impulse,
        which returns to zero.
    """

    summary_df: pd.DataFrame
    shock: str
    shock_type: str
    shock_size: float
    t0: int
    horizon: int
    n_draws: int
    seed: int
    stochastic: bool
    equilibrium: float
    longrun_target: float
    bands: tuple[int, ...]
    _draws: FloatArray = field(repr=False, default_factory=lambda: np.empty(0))

    def summary(self) -> str:
        """Text report: the setup, and where the response ends up."""
        resp = self.summary_df["response"]
        final = float(resp["point"].iloc[-1])
        widest = max(self.bands)
        lines = [
            f"Dynamic simulation - {self.shock_type} shock of "
            f"{self.shock_size:.6g} on {self.shock}",
            f"  t0 = {self.t0}, horizon = {self.horizon}, "
            f"{self.n_draws} parameter draws, seed = {self.seed}",
            f"  innovations: {'on' if self.stochastic else 'off'}"
            " (they cancel out of the response either way)",
            f"  baseline equilibrium y* = {self.equilibrium:.6f}",
            "",
            f"  response at h = {self.horizon}: {final:.6f}",
            f"  {widest}% band: [{float(resp[f'lo_{widest}'].iloc[-1]):.6f}, "
            f"{float(resp[f'hi_{widest}'].iloc[-1]):.6f}]",
        ]
        if np.isfinite(self.longrun_target):
            lines.append(f"  long-run target theta * dx = {self.longrun_target:.6f}")
        return "\n".join(lines)

    def plot(
        self, block: str = "response", bands: tuple[int, ...] | None = None
    ) -> Any:
        """Fan chart of the simulated trajectory.

        Parameters
        ----------
        block : {"response", "level"}
            Which of the two column blocks to draw.
        bands : tuple of int, optional
            Subset of the bands computed at simulation time. Defaults to
            all of them, drawn from widest to narrowest.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ImportError
            If matplotlib, an optional dependency, is missing.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Plotting requires matplotlib, an optional dependency. "
                "Install it with: pip install pyardl[plot]"
            ) from exc

        if block not in ("response", "level"):
            raise ValueError(f'block must be "response" or "level", got {block!r}.')
        chosen = tuple(self.bands) if bands is None else tuple(int(b) for b in bands)
        unknown = [b for b in chosen if b not in self.bands]
        if unknown:
            raise ValueError(
                f"Bands {unknown} were not computed; available: {list(self.bands)}."
            )

        table = self.summary_df[block]
        h = table.index.to_numpy()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for depth, level in enumerate(sorted(chosen, reverse=True)):
            ax.fill_between(
                h,
                table[f"lo_{level}"],
                table[f"hi_{level}"],
                color="C0",
                alpha=0.15 + 0.12 * depth,
                linewidth=0,
                label=f"{level}%",
            )
        ax.plot(h, table["point"], color="C0", linewidth=1.8, label="point estimate")
        if block == "response":
            ax.axhline(0.0, color="0.3", linewidth=0.8)
            if np.isfinite(self.longrun_target):
                ax.axhline(
                    self.longrun_target,
                    color="C3",
                    linestyle="--",
                    linewidth=1.0,
                    label="long run",
                )
        else:
            ax.axhline(self.equilibrium, color="0.3", linewidth=0.8, label="y*")
        ax.axvline(self.t0, color="0.5", linestyle=":", linewidth=1.0)
        ax.set_xlabel("horizon")
        ax.set_ylabel("response of y" if block == "response" else "y")
        ax.set_title(
            f"{self.shock_type.capitalize()} shock of "
            f"{self.shock_size:.4g} on {self.shock}"
        )
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        return fig


def _quantile_block(
    paths: FloatArray, point: FloatArray, bands: tuple[int, ...]
) -> dict[str, FloatArray]:
    """Point trajectory, mean across draws, and the pointwise bands."""
    out = {"point": point, "mean": paths.mean(axis=0)}
    for level in bands:
        alpha = (100 - level) / 200.0
        out[f"lo_{level}"] = np.quantile(paths, alpha, axis=0)
        out[f"hi_{level}"] = np.quantile(paths, 1 - alpha, axis=0)
    return out


def dynardl_simulate(
    results: Any,
    shock: str,
    shock_type: ShockType = "step",
    size: float | str = "1sd",
    t0: int = 10,
    horizon: int = 50,
    r: int = 1000,
    stochastic: bool = False,
    seed: int | None = None,
    start: StartRule = "mean",
    scenario: dict[str, float] | None = None,
    param_draws: npt.ArrayLike | None = None,
    bands: tuple[int, ...] = (75, 90, 95),
) -> DynardlSimulation:
    r"""Simulate the response of ``y`` to a counterfactual shock.

    Parameters
    ----------
    results : ARDLResults
        A fitted ARDL. A NARDL is simulated through its underlying
        linear ARDL, where the decomposed columns ``x_pos`` and
        ``x_neg`` are ordinary regressors and can be shocked separately.
    shock : str
        Name of the regressor to shock.
    shock_type : {"step", "impulse"}
        A ``step`` moves the regressor permanently from ``t0`` onwards;
        an ``impulse`` moves it for one period only.
    size : float or "1sd"
        Amplitude. ``"1sd"`` uses the sample standard deviation of the
        regressor's own level.
    t0 : int, default 10
        Horizon at which the shock lands. The periods before it show the
        baseline, which is what makes the figure readable.
    horizon : int, default 50
        Last horizon simulated.
    r : int, default 1000
        Number of parameter draws.
    stochastic : bool, default False
        Add innovations :math:`N(0, \hat\sigma^2)` to the recursion.
        They enter both branches of the paired difference, so they widen
        the band on the level and leave the response untouched.
    seed : int, optional
        Drawn from entropy and recorded on the result when omitted.
    start : {"mean", "last"}
        Baseline for every regressor: its sample mean, or its final
        observed value.
    scenario : dict, optional
        Override the baseline of individual regressors.
    param_draws : array-like, shape (r, n_params), optional
        Use these parameter vectors instead of drawing from
        :math:`N(\hat\theta, \hat V)` — the hook for feeding in bootstrap
        replications, so that a figure and a bounds test can rest on the
        same notion of uncertainty.
    bands : tuple of int
        Band levels, in percent.

    Returns
    -------
    DynardlSimulation

    Raises
    ------
    ValueError
        If an argument is out of range, or the autoregressive polynomial
        has a unit root, in which case there is no equilibrium to
        simulate around.

    Notes
    -----
    The recursion is the ARDL form itself,

    .. math::

        y_t = d_t + \sum_i \phi_i y_{t-i}
              + \sum_j \sum_l \beta_{j,l} x_{j,t-l} + \varepsilon_t,

    started from the equilibrium implied by *each draw* rather than by
    :math:`\hat\theta`. That is what makes the no-shock branch exactly
    flat for every draw under ``det="const"``, and it is why the
    response isolates the shock instead of mixing it with a transient
    from a mismatched starting point.

    With ``det="trend"`` the deterministic part keeps moving, so the
    ``level`` block drifts; the ``response`` block is unaffected, since
    the trend is common to both branches and cancels.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.core.ardl import ARDL
    >>> rng = np.random.default_rng(0)
    >>> x = pd.Series(rng.normal(size=200).cumsum(), name="x")
    >>> y = pd.Series(2.0 + 1.5 * x.to_numpy() + rng.normal(size=200), name="y")
    >>> res = ARDL(y, pd.DataFrame({"x": x}), order=(1, 1)).fit()
    >>> sim = dynardl_simulate(res, "x", size=1.0, horizon=60, r=200, seed=7)
    >>> round(float(sim.summary_df[("response", "point")].iloc[-1]), 6)
    1.502278
    """
    model = results.model
    if shock_type not in ("step", "impulse"):
        raise ValueError(f'shock_type must be "step" or "impulse", got {shock_type!r}.')
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}.")
    if not 0 <= t0 <= horizon:
        raise ValueError(f"t0={t0} must lie in [0, horizon={horizon}].")
    if r < 2:
        raise ValueError(f"r must be at least 2 to form a band, got {r}.")
    if not bands or any(not 0 < int(b) < 100 for b in bands):
        raise ValueError(
            f"bands must be percentages strictly in (0, 100), got {bands}."
        )
    bands = tuple(sorted({int(b) for b in bands}))

    names = [str(n) for n in results._param_names]
    point_params = np.asarray(results._params, dtype=np.float64)
    x_names = tuple(model._x_names)
    if model._x is None or shock not in x_names:
        raise KeyError(
            f"{shock!r} is not a distributed-lag regressor of the model; "
            f"available: {list(x_names)}."
        )

    p = int(model.p)
    q = {name: int(model._q[name]) for name in x_names}
    max_q = max(q.values()) if q else 0
    lead = max(p, max_q, 1)
    n_time = lead + horizon + 1

    phi_idx = [names.index(f"{model._y_name}.L{i}") for i in range(1, p + 1)]

    # --- baseline paths -------------------------------------------------
    x_data = np.asarray(model._x, dtype=np.float64)
    base_x = _baseline_values(x_data, x_names, start, scenario)
    if isinstance(size, str):
        if size != "1sd":
            raise ValueError(f'size must be a number or "1sd", got {size!r}.')
        shock_size = float(np.std(x_data[:, x_names.index(shock)], ddof=1))
    else:
        shock_size = float(size)

    x_path = np.tile(base_x[:, None], (1, n_time))
    x_shocked = x_path.copy()
    j_shock = x_names.index(shock)
    if shock_type == "step":
        x_shocked[j_shock, lead + t0 :] += shock_size
    else:
        x_shocked[j_shock, lead + t0] += shock_size

    # --- regressor design over the timeline -----------------------------
    def _exog_matrix(path: FloatArray) -> tuple[FloatArray, list[int]]:
        cols: list[FloatArray] = []
        idx: list[int] = []
        for j, name in enumerate(x_names):
            for lag in range(q[name] + 1):
                shifted = np.empty(n_time)
                if lag:
                    shifted[lag:] = path[j, : n_time - lag]
                    shifted[:lag] = path[j, 0]
                else:
                    shifted[:] = path[j]
                cols.append(shifted)
                idx.append(names.index(f"{name}.L{lag}"))
        return np.column_stack(cols), idx

    z_base, exog_idx = _exog_matrix(x_path)
    z_shock, _ = _exog_matrix(x_shocked)

    # --- deterministic terms, continuing the sample ---------------------
    n_sample = int(model._y.shape[0])
    abs_t: FloatArray = np.asarray(
        n_sample + np.arange(-lead, horizon + 1), dtype=np.float64
    )
    det_cols: list[FloatArray] = []
    det_idx: list[int] = []
    if model.det in ("const", "trend"):
        det_cols.append(np.ones(n_time, dtype=np.float64))
        det_idx.append(names.index("const"))
    if model.det == "trend":
        det_cols.append(abs_t)
        det_idx.append(names.index("trend"))
    if model.seasonal:
        s = int(model.seasonal_periods)
        phase = (n_sample - 1 + np.arange(-lead, horizon + 1)) % s
        drop = 1 if model.det in ("const", "trend") else 0
        for k in range(drop, s):
            det_cols.append((phase == k).astype(np.float64))
            det_idx.append(names.index(f"season.{k + 1}"))
    if model._fixed is not None:
        base_fixed = _baseline_values(
            np.asarray(model._fixed, dtype=np.float64), model._fixed_names, start, None
        )
        for value, name in zip(base_fixed, model._fixed_names, strict=True):
            det_cols.append(np.full(n_time, float(value)))
            det_idx.append(names.index(name))
    det_matrix = (
        np.column_stack(det_cols)
        if det_cols
        else np.zeros((n_time, 0), dtype=np.float64)
    )

    # --- parameter draws -------------------------------------------------
    if seed is None:
        entropy = np.random.SeedSequence().entropy
        seed = int(entropy) % (2**63) if isinstance(entropy, int) else 0
    rng = np.random.default_rng(seed)
    if param_draws is None:
        draws = rng.multivariate_normal(
            point_params, np.asarray(results._cov_params, dtype=np.float64), size=r
        )
    else:
        draws = np.atleast_2d(np.asarray(param_draws, dtype=np.float64))
        if draws.shape[1] != point_params.size:
            raise ValueError(
                f"param_draws has {draws.shape[1]} columns for "
                f"{point_params.size} parameters."
            )
        r = draws.shape[0]
    # Row 0 is the point estimate: it travels through the same recursion
    # as the draws, so the reported trajectory and its band can never
    # come from two different code paths.
    stacked = np.vstack([point_params[None, :], draws])

    phi = stacked[:, phi_idx] if phi_idx else np.zeros((stacked.shape[0], 0))
    root_sum = phi.sum(axis=1)
    if abs(1.0 - float(root_sum[0])) < 1e-10:
        raise ValueError(
            "The estimated autoregressive polynomial has a unit root "
            "(the phi sum to one): there is no equilibrium to simulate "
            "around, and the recursion would not converge."
        )
    if not bool(results.is_stable):
        warnings.warn(
            "The fitted ARDL is not stable, so the simulated paths "
            "diverge rather than settle. The bands below are still what "
            "the estimates imply, but they do not describe an "
            "equilibrium response.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    drive_det = stacked[:, det_idx] @ det_matrix.T if det_idx else 0.0
    beta = stacked[:, exog_idx]
    drive_base: FloatArray = np.asarray(drive_det + beta @ z_base.T, dtype=np.float64)
    drive_shock: FloatArray = np.asarray(drive_det + beta @ z_shock.T, dtype=np.float64)

    # Each draw starts at ITS OWN implied equilibrium, so the branch with
    # no shock stays exactly flat instead of drifting in from a starting
    # value borrowed from another parameter vector.
    y_star = drive_base[:, lead] / (1.0 - root_sum)

    eps = (
        rng.normal(
            scale=np.sqrt(float(results.sigma2)), size=(stacked.shape[0], n_time)
        )
        if stochastic
        else None
    )

    def _run(drive: FloatArray) -> FloatArray:
        y = np.empty((stacked.shape[0], n_time))
        y[:, :lead] = y_star[:, None]
        for t in range(lead, n_time):
            value = drive[:, t].copy()
            for i in range(p):
                value += phi[:, i] * y[:, t - 1 - i]
            if eps is not None:
                value += eps[:, t]
            y[:, t] = value
        return y[:, lead:]

    level = _run(drive_shock)
    counterfactual = _run(drive_base)
    response: FloatArray = np.asarray(level - counterfactual, dtype=np.float64)

    blocks = {
        "response": _quantile_block(response[1:], response[0], bands),
        "level": _quantile_block(level[1:], level[0], bands),
    }
    frame = pd.concat(
        {name: pd.DataFrame(block) for name, block in blocks.items()}, axis=1
    )
    frame.index = pd.RangeIndex(horizon + 1, name="horizon")

    target = float("nan")
    if shock_type == "step":
        try:
            target = float(results.longrun.loc[shock, "theta"]) * shock_size
        except (NotImplementedError, ValueError):
            # A pure distributed-lag model, or one with fixed regressors,
            # has no error-correction view to read theta from. The
            # simulation is still well defined; only the reference line
            # on the figure is missing, and saying so beats guessing.
            target = float("nan")

    return DynardlSimulation(
        summary_df=frame,
        shock=str(shock),
        shock_type=str(shock_type),
        shock_size=shock_size,
        t0=int(t0),
        horizon=int(horizon),
        n_draws=int(r),
        seed=int(seed),
        stochastic=bool(stochastic),
        equilibrium=float(y_star[0]),
        longrun_target=target,
        bands=bands,
        _draws=response[1:],
    )
