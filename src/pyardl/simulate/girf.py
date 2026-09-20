r"""Generalized impulse responses and forecast-error variance decomposition.

``dynardl_simulate`` (:mod:`pyardl.simulate.dynardl`, spec 25) answers
"what is the effect of *one* shock to x, at a given horizon?" — one
shock, one response trajectory. Pesaran & Shin (1998) name two things
that stays out of reach with that alone:

- **generalized impulse responses (GIRF)**: in a nonlinear model
  (NARDL, Threshold ARDL, STAR) the effect of a shock does not depend
  only on its size but on the **state the system was in** when the
  shock hit — a positive shock does not have the same cumulated effect
  arriving in a high regime as in a low one.
- **forecast-error variance decomposition (FEVD)**: what share of the
  forecast variance of :math:`y` at horizon :math:`h` is attributable
  to shocks on :math:`x` rather than :math:`y`'s own innovations.

Scope of this version
----------------------
``generalized_irf`` reuses ``dynardl_simulate`` directly, orchestrating
it over a set of caller-supplied "histories" (each a ``scenario`` dict
of regressor baselines, exactly ``dynardl_simulate``'s own ``scenario``
parameter). For a **linear** ARDL (or a NARDL/STAR/Threshold model
simulated through its underlying linear ARDL on already-decomposed
columns — the same limitation ``dynardl_simulate`` already documents),
the paired-difference response does not depend on the baseline at all,
so the GIRF is history-invariant by construction and coincides with
``dynardl_simulate`` exactly — which is precisely what spec 39 §5 tests
1-2 check. Genuine history-*dependent* GIRF for a model whose nonlinear
decomposition is recomputed at each simulated step (spec 39 §5.3, a
NARDL/STAR path where the regime itself evolves during the simulation)
would need the nonlinear recursion rebuilt from scratch and is **not
implemented** in this version — see ``docs/DEVIATIONS.md``.

FEVD is a genuine single-equation construction (§2.3): the MA weights
of a unit shock to :math:`x` come from ``dynardl_simulate`` itself
(``shock_type='impulse'``, unit size); the MA weights of a unit shock to
:math:`y`'s own innovation come from the AR(:math:`p`) polynomial
directly. Both squared and cumulated give the two variance sources.

References
----------
.. [1] Pesaran, M. H. & Shin, Y. (1998). Generalized impulse response
       analysis in linear multivariate models. *Economics Letters*,
       58(1), 17-29.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pyardl.simulate.dynardl import dynardl_simulate

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]

__all__ = ["generalized_irf", "fevd", "GIRFResults", "FEVDResults"]

HistorySpec = Literal["sample"]


@dataclass(frozen=True)
class GIRFResults:
    """Outcome of :func:`generalized_irf`.

    Attributes
    ----------
    girf_by_history : dict
        ``{history_label: pandas.Series}``, one GIRF trajectory
        (indexed by horizon) per history used.
    girf_mean : pandas.Series
        Average over histories.
    shock : str
    shock_size : float
    horizon : int
    r : int
    seed : int
    n_histories : int
    """

    girf_by_history: dict[Any, pd.Series] = field(repr=False)
    girf_mean: pd.Series
    shock: str
    shock_size: float
    horizon: int
    r: int
    seed: int
    n_histories: int

    def summary(self) -> str:
        """Readable report."""
        lines = [
            f"Generalized IRF (Pesaran & Shin 1998) - shock={self.shock}, "
            f"size={self.shock_size:.4f}, {self.n_histories} histories, "
            f"horizon={self.horizon}",
            f"  girf_mean[0] = {self.girf_mean.iloc[0]:.4f}   "
            f"girf_mean[{self.horizon}] = {self.girf_mean.iloc[-1]:.4f}",
        ]
        return "\n".join(lines)

    def plot_girf(self, ax: object = None):  # type: ignore[no-untyped-def]
        """Plot every history's GIRF plus the mean. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "plot_girf requires matplotlib, an optional dependency. "
                "Install it with: pip install matplotlib"
            ) from exc
        if ax is None:  # pragma: no cover - trivial plumbing
            _, ax = plt.subplots(figsize=(7, 4))
        axes: Any = ax
        for series in self.girf_by_history.values():
            axes.plot(series.index, series.to_numpy(), alpha=0.3, color="#2A2F86")
        axes.plot(
            self.girf_mean.index,
            self.girf_mean.to_numpy(),
            color="black",
            linewidth=2,
            label="mean",
        )
        axes.axhline(0.0, color="grey", linewidth=0.5)
        axes.set_xlabel("horizon")
        axes.set_ylabel(f"GIRF({self.shock})")
        axes.legend()
        return axes


def generalized_irf(
    results: Any,
    shock: str,
    shock_size: float | str = "1sd",
    histories: HistorySpec | list[dict[str, float]] = "sample",
    n_histories: int = 10,
    h: int = 40,
    r: int = 200,
    seed: int | None = None,
) -> GIRFResults:
    r"""Generalized impulse response: :func:`~pyardl.simulate.dynardl.dynardl_simulate`,
    orchestrated over a set of conditioning histories.

    Parameters
    ----------
    results : ARDLResults
        A fitted ARDL (or NARDL, simulated through its underlying linear
        ARDL on decomposed columns).
    shock : str
        Name of the regressor to shock.
    shock_size : float or "1sd", default "1sd"
    histories : "sample" or list of dict, default "sample"
        ``"sample"`` draws ``n_histories`` rows from the observed sample
        and uses each regressor's value there as a
        :func:`~pyardl.simulate.dynardl.dynardl_simulate` ``scenario``
        (the conditioning history). A list of explicit ``scenario``
        dicts overrides the regressors named in each.
    n_histories : int, default 10
        Number of histories drawn when ``histories="sample"``.
    h : int, default 40
        Horizon.
    r : int, default 200
        Parameter draws per history (passed to ``dynardl_simulate``).
    seed : int, optional
        Seed for both history sampling and every simulation.

    Returns
    -------
    GIRFResults

    Notes
    -----
    For a linear model, the response is history-invariant by
    construction (spec 39 §5 test 1) and coincides with
    ``dynardl_simulate`` exactly (test 2) — both are properties of the
    underlying paired-difference mechanism, not asserted here but a
    direct consequence of reusing it unchanged.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.core.ardl import ARDL
    >>> from pyardl.simulate import generalized_irf, dynardl_simulate
    >>> rng = np.random.default_rng(0)
    >>> x = pd.Series(rng.normal(size=200).cumsum(), name="x")
    >>> y = pd.Series(2.0 + 1.5 * x.to_numpy() + rng.normal(size=200), name="y")
    >>> res = ARDL(y, pd.DataFrame({"x": x}), order=(1, 1)).fit()
    >>> girf = generalized_irf(res, "x", shock_size=1.0, h=20, r=50,
    ...                        n_histories=5, seed=7)
    >>> sim = dynardl_simulate(res, "x", shock_type="impulse", size=1.0,
    ...                        t0=0, horizon=20, r=50, seed=7)
    >>> diff = girf.girf_mean.iloc[-1] - sim.summary_df[("response", "point")].iloc[-1]
    >>> bool(abs(diff) < 1e-6)
    True
    """
    rng = np.random.default_rng(seed)
    model = results.model
    x_names = tuple(model._x_names)

    scenarios: list[dict[str, float]]
    labels: list[Any]
    if histories == "sample":
        x_data = np.asarray(model._x, dtype=np.float64)
        n_obs = x_data.shape[0]
        idx = rng.choice(n_obs, size=min(n_histories, n_obs), replace=False)
        scenarios = [
            {name: float(x_data[i, j]) for j, name in enumerate(x_names)} for i in idx
        ]
        labels = [int(i) for i in idx]
    else:
        scenarios = list(histories)
        labels = list(range(len(scenarios)))

    girf_by_history: dict[Any, pd.Series] = {}
    shock_size_resolved: float | None = None
    for label, scenario in zip(labels, scenarios, strict=True):
        sim = dynardl_simulate(
            results,
            shock,
            shock_type="impulse",
            size=shock_size,
            t0=0,
            horizon=h,
            r=r,
            seed=int(rng.integers(0, 2**31 - 1)),
            start="last",
            scenario=scenario,
        )
        girf_by_history[label] = sim.summary_df[("response", "point")]
        shock_size_resolved = sim.shock_size

    girf_mean = pd.concat(girf_by_history.values(), axis=1).mean(axis=1)
    girf_mean.name = "girf_mean"

    return GIRFResults(
        girf_by_history=girf_by_history,
        girf_mean=girf_mean,
        shock=shock,
        shock_size=(
            float(shock_size_resolved) if shock_size_resolved is not None else 0.0
        ),
        horizon=h,
        r=r,
        seed=int(seed) if seed is not None else -1,
        n_histories=len(scenarios),
    )


@dataclass(frozen=True)
class FEVDResults:
    """Outcome of :func:`fevd`.

    Attributes
    ----------
    shares : pandas.Series
        Indexed by horizon: cumulative share of the forecast-error
        variance of ``y`` attributable to shocks on ``shock``.
    shock : str
    x_shock_variance : float
        Variance assumed for a single shock on ``shock`` — supplied
        externally to the ARDL, which treats ``x`` as given (never
        presented as endogenous to the estimated model, spec 39 §4).
    horizon : int
    """

    shares: pd.Series
    shock: str
    x_shock_variance: float
    horizon: int

    def summary(self) -> str:
        """Readable report."""
        return (
            f"FEVD - shock={self.shock}, x_shock_variance={self.x_shock_variance:.4f}\n"
            f"  share[0] = {self.shares.iloc[0]:.4f}   "
            f"share[{self.horizon}] = {self.shares.iloc[-1]:.4f}"
        )

    def plot_fevd(self, ax: object = None):  # type: ignore[no-untyped-def]
        """Plot the variance share over horizon. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "plot_fevd requires matplotlib, an optional dependency. "
                "Install it with: pip install matplotlib"
            ) from exc
        if ax is None:  # pragma: no cover - trivial plumbing
            _, ax = plt.subplots(figsize=(7, 4))
        axes: Any = ax
        axes.plot(self.shares.index, self.shares.to_numpy(), color="#2A2F86")
        axes.set_ylim(0, 1)
        axes.set_xlabel("horizon")
        axes.set_ylabel(f"share attributable to {self.shock}")
        return axes


def fevd(
    results: Any,
    shock: str,
    h: int = 40,
    x_shock_variance: float | Literal["auto"] = "auto",
) -> FEVDResults:
    r"""Single-equation forecast-error variance decomposition.

    Parameters
    ----------
    results : ARDLResults
    shock : str
        Name of the regressor whose contribution is decomposed.
    h : int, default 40
        Horizon.
    x_shock_variance : float or "auto", default "auto"
        Variance of a single shock to ``shock``. ``"auto"`` uses the
        sample variance of the regressor's own first difference — a
        stated modelling choice, not estimated by the ARDL itself
        (which treats ``x`` as given; see the module notes).

    Returns
    -------
    FEVDResults

    Notes
    -----
    :math:`\text{FEVD}_x(h) = \dfrac{\sum_{k=0}^{h} \psi_x(k)^2
    \sigma_x^2}{\sum_{k=0}^{h} \psi_x(k)^2 \sigma_x^2 + \sum_{k=0}^{h}
    \psi_y(k)^2 \hat\sigma^2}`, where :math:`\psi_x` is the MA response
    of :math:`y` to a unit shock on :math:`x` (from
    :func:`~pyardl.simulate.dynardl.dynardl_simulate`) and
    :math:`\psi_y` is the MA response of :math:`y` to a unit shock on
    its own innovation (from the AR(:math:`p`) polynomial directly —
    no simulation needed, it is a deterministic recursion).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.core.ardl import ARDL
    >>> from pyardl.simulate import fevd
    >>> rng = np.random.default_rng(0)
    >>> x = pd.Series(rng.normal(size=200).cumsum(), name="x")
    >>> y = pd.Series(2.0 + 1.5 * x.to_numpy() + rng.normal(size=200), name="y")
    >>> res = ARDL(y, pd.DataFrame({"x": x}), order=(1, 1)).fit()
    >>> out = fevd(res, "x", h=20)
    >>> bool(0.0 <= out.shares.iloc[0] <= 1.0)
    True
    """
    if h < 1:
        raise ValueError(f"h={h} must be at least 1.")

    model = results.model
    x_names = tuple(model._x_names)
    if shock not in x_names:
        raise KeyError(f"{shock!r} is not a regressor; available: {list(x_names)}.")

    sim = dynardl_simulate(
        results, shock, shock_type="impulse", size=1.0, t0=0, horizon=h, r=2, seed=0
    )
    psi_x = sim.summary_df[("response", "point")].to_numpy()

    names = [str(n) for n in results._param_names]
    point_params = np.asarray(results._params, dtype=np.float64)
    p = int(model.p)
    phi = np.array(
        [point_params[names.index(f"{model._y_name}.L{i}")] for i in range(1, p + 1)]
    )
    psi_y = np.zeros(h + 1)
    psi_y[0] = 1.0
    for k in range(1, h + 1):
        psi_y[k] = sum(phi[i - 1] * psi_y[k - i] for i in range(1, min(p, k) + 1))

    if x_shock_variance == "auto":
        x_data = np.asarray(model._x, dtype=np.float64)
        j = x_names.index(shock)
        sigma_x2 = float(np.var(np.diff(x_data[:, j]), ddof=1))
    else:
        sigma_x2 = float(x_shock_variance)

    sigma_y2 = float(results.sigma2)

    var_x_cum = np.cumsum(psi_x**2) * sigma_x2
    var_y_cum = np.cumsum(psi_y**2) * sigma_y2
    shares = var_x_cum / (var_x_cum + var_y_cum)

    return FEVDResults(
        shares=pd.Series(shares, index=pd.RangeIndex(h + 1, name="horizon")),
        shock=shock,
        x_shock_variance=sigma_x2,
        horizon=h,
    )
