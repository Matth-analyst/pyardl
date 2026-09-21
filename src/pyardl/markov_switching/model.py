r"""Markov-switching ARDL: coefficients driven by an unobserved regime (Hamilton 1989).

STAR (:mod:`pyardl.star`, spec 34) makes the regime depend on an
**observed** transition variable. This module makes it depend on an
unobserved state :math:`S_t \in \{1, ..., K\}` following a first-order
Markov chain, relevant when the regime change is driven by a latent
economic state (recession/expansion, financial stress) with no single
reliable observed proxy — at the cost of a much heavier estimation: the
state itself must be inferred, not just the coefficients.

Like specs 30/34/36/41/42, "Markov-switching ARDL" has no single
founding paper — this module adapts Hamilton's (1989) framework
(designed for a plain AR process) to the library's UECM structure:

.. math::
    \Delta y_t = \det(S_t) + \lambda(S_t) y_{t-1} + \gamma(S_t)
    x_{t-1} + \sum(\ldots) + \varepsilon_t, \quad
    \varepsilon_t \mid S_t \sim N(0, \sigma^2(S_t))

**Implementation strategy, documented explicitly (spec 35 Section
2.3).** Every other spec in this library reuses an OLS regression
inside a grid search or a light nonlinear optimisation on top of
bricks already validated elsewhere. This one does not: the Hamilton
filter, Kim (1994) smoother, and EM estimation it requires are a
genuinely new estimation engine, absent from the rest of the library —
the spec itself flags this as the costliest of the proposed specs to
build from scratch. Rather than reimplement the Hamilton filter and EM
loop, this module wraps
:class:`statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`
— an already-validated implementation of exactly this engine, and
`statsmodels` is already an approved runtime dependency of this
project (used elsewhere for diagnostics, quantile regression, and
critical values). The UECM design matrix (deterministic terms, the
error-correction term ``y.L1``, the level regressors ``x{j}.L1``, and
the short-run difference terms) is built with this library's own
naming conventions and handed to `statsmodels` as `exog` with every
coefficient switching by regime — the spec's own Section 6 already
names `statsmodels` `MarkovRegression`/`MarkovAutoregression` as the
preferred route for external validation "when functional coverage
matches"; using it as the estimation engine itself is the same
judgement applied one step earlier. See ``docs/DEVIATIONS.md``.

**Out of scope** (spec 35 Section 6): Markov chains of order > 1,
time-varying transition probabilities (TVTP), and the regime-specific
long-run ratio :math:`\theta(S_t) = -\gamma(S_t)/\lambda(S_t)` (its
inference would need the delta method combined with the latent-regime
uncertainty, not addressed here).

References
----------
.. [1] Hamilton, J. D. (1989). A new approach to the economic analysis
       of nonstationary time series and the business cycle.
       *Econometrica*, 57(2), 357-384.
.. [2] Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
       *Journal of Econometrics*, 60(1-2), 1-22.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

Det = Literal["none", "const", "trend"]
Method = Literal["em", "direct"]

__all__ = ["ms_ardl", "MSARDLResults"]


def _build_design(
    y: FloatArray, x: FloatArray, p: int, q: int, det: Det
) -> tuple[FloatArray, FloatArray, list[str]]:
    """UECM design at order (p, q); same layout as pyardl.bayesian.model."""
    k = x.shape[1]
    start = max(p, q, 1)
    n = y.shape[0]
    dy = np.diff(y)
    dx = np.diff(x, axis=0)
    target = dy[start - 1 :]

    cols: list[FloatArray] = []
    names: list[str] = []

    if det in ("const", "trend"):
        cols.append(np.ones_like(target))
        names.append("const")
    if det == "trend":
        cols.append(np.arange(1, target.shape[0] + 1, dtype=np.float64))
        names.append("trend")

    cols.append(y[start - 1 : n - 1])
    names.append("y.L1")

    for j in range(k):
        cols.append(x[start - 1 : n - 1, j])
        names.append(f"x{j}.L1")

    for i in range(1, p):
        cols.append(dy[start - i - 1 : n - i - 1])
        names.append(f"D.y.L{i}")
    for j in range(k):
        for i in range(q):
            cols.append(dx[start - i - 1 : n - i - 1, j])
            names.append(f"D.x{j}.L{i + 1}")

    design = np.column_stack(cols)
    return design, target, names


@dataclass(frozen=True)
class MSARDLResults:
    """Outcome of :func:`ms_ardl`.

    Attributes
    ----------
    transition_matrix : pandas.DataFrame
        ``P[i, j] = P(S_t = j | S_{t-1} = i)``, rows sum to 1.
    regime_params : dict
        ``{state: pandas.Series}`` — one series per regime, indexed by
        design term name, plus ``sigma2``.
    filtered_probs, smoothed_probs : pandas.DataFrame
        ``P(S_t = j | information up to t)`` and, respectively,
        ``P(S_t = j | all the information, Kim 1994)`` — never the
        same quantity, a classic mix-up in the applied literature.
    expected_duration : pandas.Series
        ``1 / (1 - P[i, i])``, indexed by state.
    llf : float
        Maximised log-likelihood.
    n_states : int
    names : list of str
        Design term names (shared across regimes).
    """

    transition_matrix: pd.DataFrame
    regime_params: dict[int, pd.Series]
    filtered_probs: pd.DataFrame
    smoothed_probs: pd.DataFrame
    expected_duration: pd.Series
    llf: float
    n_states: int
    names: list[str] = field(repr=False)
    _em_llf_history: FloatArray | None = field(default=None, repr=False)

    def summary(self) -> str:
        """Readable report: per-regime coefficients, transition matrix, durations."""
        lines = [
            f"Markov-Switching ARDL (Hamilton 1989) - {self.n_states} states, "
            f"llf={self.llf:.4f}",
            "",
            "  Transition matrix (rows: from, cols: to):",
            "    "
            + self.transition_matrix.round(4).to_string().replace("\n", "\n    "),
            "",
            "  Expected duration (periods):",
        ]
        for state, duration in self.expected_duration.items():
            lines.append(f"    state {state}: {duration:.2f}")
        for state, params in self.regime_params.items():
            lines.append(f"\n  Regime {state}:")
            for name, value in params.items():
                lines.append(f"    {name:<14}{value: .4f}")
        return "\n".join(lines)

    def plot_regimes(self, ax: object = None):  # type: ignore[no-untyped-def]
        """Smoothed regime probabilities overlaid on the dependent variable's changes.

        Requires matplotlib (optional dependency, lazy import).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "plot_regimes requires matplotlib, an optional dependency. "
                "Install it with: pip install matplotlib"
            ) from exc
        if ax is None:  # pragma: no cover - trivial plumbing
            _, ax = plt.subplots(figsize=(9, 4))
        axes: object = ax
        for state in self.smoothed_probs.columns:
            axes.plot(  # type: ignore[attr-defined]
                self.smoothed_probs.index, self.smoothed_probs[state], label=str(state)
            )
        axes.set_ylabel("P(S_t = state | all data)")  # type: ignore[attr-defined]
        axes.set_xlabel("t")  # type: ignore[attr-defined]
        axes.legend()  # type: ignore[attr-defined]
        return axes


def _restart_fit(
    mod: MarkovRegression,
    use_em: bool,
    maxiter: int,
    gtol: float,
    n_starts: int,
    seed: int | np.random.Generator | None,
) -> Any:
    """Multiple restarts perturbing the coefficient block, via an explicit Generator.

    ``method="em"`` fits by **pure** EM to convergence
    (``MarkovSwitching._fit_em``), not a short EM warm-up followed by
    direct BFGS maximisation of the (highly non-concave, badly scaled
    on typical ARDL-sized regressors) likelihood — the latter was found
    to diverge to nonsensical parameter values on ordinary synthetic
    data during this spec's own test development, while plain EM to
    convergence recovered the true regime parameters reliably.
    ``method="direct"`` fits by BFGS from the model's default starting
    values, with no EM step at all, exactly as the spec's ``"direct"``
    option describes.

    ``statsmodels``' own ``search_reps`` draws restarts from the global
    NumPy random state (CLAUDE.md rule 2 forbids relying on it), so
    restarts are instead drawn here with an explicit
    ``numpy.random.Generator``. Only the coefficient/variance block is
    perturbed — the transition-probability starting values are kept
    fixed across restarts, a documented scope simplification (the
    dominant local-optima risk is in the regime-specific coefficients,
    not the transition probabilities).
    """
    base_start = np.asarray(mod.start_params, dtype=np.float64)
    param_names = mod.param_names
    is_transition = np.array([name.startswith("p[") for name in param_names])

    rng = np.random.default_rng(seed)
    best_res = None
    best_llf = -np.inf
    for i in range(max(n_starts, 1)):
        start = base_start.copy()
        if i > 0:
            noise = rng.normal(scale=0.3, size=base_start.shape)
            start = np.where(
                is_transition, start, start + noise * (np.abs(start) + 0.1)
            )
        try:
            if use_em:
                em_res = mod._fit_em(
                    start_params=start, maxiter=maxiter, tolerance=gtol
                )
                # _fit_em does not run the Kim (1994) smoother -- only the
                # forward filter -- so smoothed_marginal_probabilities is
                # None on its own result; mod.smooth() runs the smoother
                # at the converged params and reproduces the same llf.
                res_i = mod.smooth(em_res.params)
            else:
                res_i = mod.fit(
                    start_params=start, em_iter=0, maxiter=maxiter, gtol=gtol, disp=0
                )
        except Exception:  # noqa: BLE001 - a bad restart is skipped, not fatal
            continue
        if not np.isfinite(res_i.llf):
            continue
        if res_i.llf > best_llf:
            best_llf = res_i.llf
            best_res = res_i
    if best_res is None:
        raise RuntimeError("ms_ardl: every restart failed to converge.")
    return best_res


def ms_ardl(
    y: ArrayLike,
    x: ArrayLike,
    order: tuple[int, int] = (1, 1),
    n_states: int = 2,
    det: Det = "const",
    method: Method = "em",
    n_starts: int = 3,
    max_iter: int = 1000,
    tol: float = 1e-6,
    seed: int | np.random.Generator | None = None,
) -> MSARDLResults:
    r"""Fit a Markov-switching UECM by maximum likelihood (Hamilton filter).

    Parameters
    ----------
    y, x : array_like
        As in :class:`pyardl.core.ardl.ARDL`.
    order : tuple of (int, int), default (1, 1)
        ``(p, q)``, the same uniform-lag convention as
        :func:`pyardl.regularized.select_order_regularized` (``q``
        applied to every regressor). ``p >= 1`` is required (the
        error-correction term ``y.L1`` needs at least one lag).
    n_states : int, default 2
        Number of regimes ``K``. A
        :class:`pyardl.exceptions.PyardlMethodologyWarning` is raised
        for ``K > 3`` (spec 35 Section 2.4: identification degrades
        fast as the number of transition probabilities, :math:`K(K-1)`,
        grows).
    det : {'none', 'const', 'trend'}, default 'const'
    method : {'em', 'direct'}, default 'em'
        ``'em'`` fits by pure EM to convergence (the practice the spec
        documents as converging most reliably, and the only one found
        numerically stable here — a short EM warm-up followed by direct
        BFGS maximisation was found to diverge on ordinary synthetic
        ARDL-scaled data, see ``docs/DEVIATIONS.md``); ``'direct'``
        maximises the likelihood by BFGS with no EM step at all.
    n_starts : int, default 3
        Restarts from perturbed starting values (spec 35 Section 4: EM
        convergence is sensitive to the starting point on short
        series); the restart with the highest likelihood is kept.
        ``1`` disables restarting.
    max_iter : int, default 1000
    tol : float, default 1e-6
        The EM convergence tolerance (``method='em'``) or the BFGS
        optimiser's gradient tolerance ``gtol`` (``method='direct'``).
    seed : int or numpy.random.Generator, optional
        Seeds the restart perturbations (CLAUDE.md rule 2).

    Returns
    -------
    MSARDLResults

    Notes
    -----
    The long-run ratio :math:`\theta(S_t) = -\gamma(S_t)/\lambda(S_t)`
    is out of scope (spec 35 Section 6) — inference on a ratio of
    coefficients that themselves depend on an inferred latent state is
    not addressed by this module.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.markov_switching import ms_ardl
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     lam, theta = (-0.6, 1.0) if t < 150 else (-0.2, 2.0)
    ...     y[t] = y[t - 1] + lam * (y[t - 1] - theta * x.iloc[t - 1])
    ...     y[t] += rng.standard_normal() * 0.3
    >>> res = ms_ardl(y, pd.DataFrame({"x": x}), order=(1, 1), n_states=2, seed=0)
    >>> res.n_states
    2
    """
    p, q = order
    if p < 1:
        raise ValueError(
            "order[0] (p) must be >= 1: no error-correction term otherwise."
        )
    if n_states < 2:
        raise ValueError("n_states must be >= 2.")
    if n_states > 3:
        import warnings

        warnings.warn(
            f"n_states={n_states}: the number of transition probabilities grows "
            "as K*(K-1) and identification degrades fast beyond K=3 "
            "(spec 35 Section 2.4).",
            PyardlMethodologyWarning,
            stacklevel=2,
        )
    if method not in ("em", "direct"):
        raise ValueError(f"method={method!r} must be 'em' or 'direct'.")

    y_arr, x_arr, index, _, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("ms_ardl needs at least one regressor.")

    design, target, names = _build_design(y_arr, x_arr, p, q, det)
    exog_df = pd.DataFrame(design, columns=names)

    mod = MarkovRegression(
        target,
        k_regimes=n_states,
        trend="n",
        exog=exog_df,
        switching_exog=True,
        switching_variance=True,
    )

    res = _restart_fit(mod, method == "em", max_iter, tol, n_starts, seed)

    transition = np.asarray(res.regime_transition[:, :, 0], dtype=np.float64).T
    transition_df = pd.DataFrame(
        transition,
        index=pd.Index(range(n_states), name="from"),
        columns=pd.Index(range(n_states), name="to"),
    )

    regime_params: dict[int, pd.Series] = {}
    for state in range(n_states):
        values = {name: res.params[f"{name}[{state}]"] for name in names}
        values["sigma2"] = res.params[f"sigma2[{state}]"]
        regime_params[state] = pd.Series(values, name=f"regime_{state}")

    row_index = index[-target.shape[0] :] if index is not None else None
    filtered_probs = pd.DataFrame(
        np.asarray(res.filtered_marginal_probabilities),
        columns=list(range(n_states)),
        index=row_index,
    )
    smoothed_probs = pd.DataFrame(
        np.asarray(res.smoothed_marginal_probabilities),
        columns=list(range(n_states)),
        index=row_index,
    )
    expected_duration = pd.Series(
        np.asarray(res.expected_durations, dtype=np.float64),
        index=pd.Index(range(n_states), name="state"),
        name="expected_duration",
    )

    return MSARDLResults(
        transition_matrix=transition_df,
        regime_params=regime_params,
        filtered_probs=filtered_probs,
        smoothed_probs=smoothed_probs,
        expected_duration=expected_duration,
        llf=float(res.llf),
        n_states=n_states,
        names=names,
    )
