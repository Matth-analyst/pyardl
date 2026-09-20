r"""STAR-ARDL: smooth transition between regimes (Teräsvirta 1994).

Threshold ARDL (:mod:`pyardl.threshold.hansen`) and Enders-Siklos
(:mod:`pyardl.cointegration.enders_siklos`) assume a **sharp** regime
change: at the threshold exactly, the regime switches all at once. STAR
replaces the indicator :math:`1\{q_t > \gamma\}` with a **continuous**
transition function :math:`G(q_t; \gamma, c) \in (0, 1)` that moves
gradually between regimes — economically more plausible for aggregate
phenomena (an economy does not switch all at once), at the cost of an
extra slope parameter :math:`\gamma` (the speed of transition) that is
harder to identify.

Not the same idea as Fourier-ADL (:mod:`pyardl.fourier`): Fourier models
a **deterministic** slow drift in the constant terms; STAR models a
smooth transition in the **coefficients** themselves.

Estimation is concentrated: for a *fixed* :math:`(\gamma, c)` the model
is linear in every other parameter (an ordinary least-squares fit on
:math:`[y_{t-1}, G \cdot y_{t-1}, x_{t-1}, G \cdot x_{t-1}, \dots]`), so
the search only needs to be over the two nonlinear parameters —
:func:`scipy.optimize.minimize` on the profiled SSR, started from the
best of a coarse grid over :math:`(\gamma, c)` to avoid local optima, as
the spec requires.

References
----------
.. [1] Teräsvirta, T. (1994). Specification, estimation, and evaluation
       of smooth transition autoregressive models. *Journal of the
       American Statistical Association*, 89(425), 208-218.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from pyardl.utils import check_regressors_allow_constant, check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

Form = Literal["lstar", "estar"]

__all__ = ["star_ardl", "STARARDLResults"]

_EXP_CLIP = 500.0


def _transition(q: FloatArray, gamma: float, c: float, form: Form) -> FloatArray:
    """G(q; gamma, c), clipped so exp() never overflows."""
    if form == "lstar":
        arg = np.clip(-gamma * (q - c), -_EXP_CLIP, _EXP_CLIP)
        return np.asarray(1.0 / (1.0 + np.exp(arg)), dtype=np.float64)
    arg = np.clip(-gamma * (q - c) ** 2, -_EXP_CLIP, _EXP_CLIP)
    return np.asarray(1.0 - np.exp(arg), dtype=np.float64)


def _design(
    y: FloatArray,
    x: FloatArray,
    q: FloatArray,
    order: tuple[int, int],
    gamma: float,
    c: float,
    form: Form,
) -> tuple[FloatArray, FloatArray, list[str]]:
    """Build the (linear-given-gamma,c) design matrix and target."""
    p, k_lag = order
    k = x.shape[1]
    start = max(p, k_lag, 1)
    n = y.shape[0]

    dy = np.diff(y)
    dx = np.diff(x, axis=0)
    target = dy[start - 1 :]

    g = _transition(q[start - 1 : n - 1], gamma, c, form)

    y_lag = y[start - 1 : n - 1]
    x_lag = x[start - 1 : n - 1]

    cols = [np.ones_like(target), y_lag, g * y_lag]
    names = ["const", "y.L1", "y.L1:G"]
    for j in range(k):
        cols.append(x_lag[:, j])
        cols.append(g * x_lag[:, j])
        names.append(f"x{j}.L1")
        names.append(f"x{j}.L1:G")
    for i in range(1, p):
        cols.append(dy[start - i - 1 : n - i - 1])
        names.append(f"D.y.L{i}")
    for j in range(k):
        for i in range(k_lag):
            cols.append(dx[start - i - 1 : n - i - 1, j])
            names.append(f"D.x{j}.L{i + 1}")

    design = np.column_stack(cols)
    return design, target, names


def _ssr(
    params: FloatArray,
    y: FloatArray,
    x: FloatArray,
    q: FloatArray,
    order: tuple[int, int],
    form: Form,
) -> float:
    gamma, c = params
    if gamma <= 0:
        return np.inf
    design, target, _ = _design(y, x, q, order, gamma, c, form)
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    return float(resid @ resid)


def _teraesvirta_lm_test(
    y: FloatArray, x: FloatArray, q: FloatArray, order: tuple[int, int]
) -> tuple[float, float, float, float, float]:
    """Third-order Taylor LM linearity test, and the LSTAR/ESTAR decision rule.

    Fits the *linear* ARDL (q does not appear), then regresses its
    residual on the original regressors plus q, q^2, q^3 interactions
    with (y_{t-1}, x_{t-1}) (Teräsvirta 1994's auxiliary regression).
    Returns the joint F-stat/p-value for linearity, and the p-values of
    the b1 (linear) and b2 (quadratic) blocks used for the form
    decision: reject whichever of b1/b3-combined vs b2 is smaller,
    choosing LSTAR when the linear/cubic terms dominate and ESTAR when
    the quadratic term does. This mirrors the spirit of Teräsvirta's
    nested-test sequence rather than reproducing its exact statistic.
    """
    p, k_lag = order
    k = x.shape[1]
    start = max(p, k_lag, 1)
    n = y.shape[0]

    dy = np.diff(y)
    dx = np.diff(x, axis=0)
    target = dy[start - 1 :]
    y_lag = y[start - 1 : n - 1]
    x_lag = x[start - 1 : n - 1]
    q_lag = q[start - 1 : n - 1]

    base_cols = [np.ones_like(target), y_lag]
    for j in range(k):
        base_cols.append(x_lag[:, j])
    for i in range(1, p):
        base_cols.append(dy[start - i - 1 : n - i - 1])
    for j in range(k):
        for i in range(k_lag):
            base_cols.append(dx[start - i - 1 : n - i - 1, j])
    base_design = np.column_stack(base_cols)

    beta0, _, _, _ = np.linalg.lstsq(base_design, target, rcond=None)
    resid0 = target - base_design @ beta0
    ssr0 = float(resid0 @ resid0)

    levels = np.column_stack([y_lag, x_lag])
    aux_blocks = [
        q_lag[:, None] * levels,
        (q_lag**2)[:, None] * levels,
        (q_lag**3)[:, None] * levels,
    ]
    aux_design = np.column_stack([base_design, *aux_blocks])
    beta1, _, _, _ = np.linalg.lstsq(aux_design, target, rcond=None)
    resid1 = target - aux_design @ beta1
    ssr1 = float(resid1 @ resid1)

    n_obs = target.shape[0]
    k_full = aux_design.shape[1]
    k_extra = aux_design.shape[1] - base_design.shape[1]
    dof = n_obs - k_full
    lm_f = ((ssr0 - ssr1) / k_extra) / (ssr1 / dof) if dof > 0 and ssr1 > 0 else 0.0
    lm_p = float(stats.f.sf(max(lm_f, 0.0), k_extra, dof))

    # b1 (linear, q^1) vs b2 (quadratic, q^2), each tested against the
    # design with only the other two Taylor blocks included.
    def _partial_f(exclude_block: int) -> float:
        keep = [aux_blocks[i] for i in range(3) if i != exclude_block]
        design_r = np.column_stack([base_design, *keep])
        beta_r, _, _, _ = np.linalg.lstsq(design_r, target, rcond=None)
        resid_r = target - design_r @ beta_r
        ssr_r = float(resid_r @ resid_r)
        n_block = aux_blocks[exclude_block].shape[1]
        dof_r = n_obs - design_r.shape[1] - n_block
        if dof_r <= 0 or ssr1 <= 0:
            return 1.0
        f_r = ((ssr_r - ssr1) / n_block) / (ssr1 / dof_r)
        return float(stats.f.sf(max(f_r, 0.0), n_block, dof_r))

    p_linear = _partial_f(0)
    p_quadratic = _partial_f(1)
    return lm_f, lm_p, p_linear, p_quadratic, ssr0


@dataclass(frozen=True)
class STARARDLResults:
    """Outcome of a STAR-ARDL fit.

    Attributes
    ----------
    form : {'lstar', 'estar'}
        Transition function used.
    gamma_hat, c_hat : float
        Transition speed and centre.
    params : pandas.Series
        Coefficients of the concentrated linear fit at
        ``(gamma_hat, c_hat)``.
    ssr : float
    linearity_stat, linearity_pvalue : float
        Teräsvirta (1994) third-order Taylor LM test of linearity
        (:math:`H_0`: no transition — a standard asymptotic F-test, not
        a bootstrap: linearising :math:`G` removes the "problem of
        Davies" that :class:`pyardl.threshold.hansen.threshold_ardl`
        needs a bootstrap for).
    form_selected : str or None
        When ``form='auto'`` was requested: which form the decision
        rule chose. ``None`` otherwise.
    n_obs : int
    order : tuple of int
    grid_start : tuple of float
        The ``(gamma, c)`` grid point the optimiser was started from.
    """

    form: Form
    gamma_hat: float
    c_hat: float
    params: pd.Series
    ssr: float
    linearity_stat: float
    linearity_pvalue: float
    form_selected: Form | None
    n_obs: int
    order: tuple[int, int]
    grid_start: tuple[float, float]

    def longrun_at(self, q_values: ArrayLike) -> pd.DataFrame:
        r"""Evaluate the long-run coefficients :math:`\theta(q)` on a grid.

        :math:`\theta_j(q) = -(\gamma^+_{1,j} + \gamma^+_{2,j} G(q)) /
        (\lambda_1 + \lambda_2 G(q))`, the signature output of STAR: a
        continuous curve rather than two discrete regimes.

        Parameters
        ----------
        q_values : array_like
            Transition-variable values to evaluate the curve at.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``q``, columns ``G`` and one per regressor.
        """
        q_arr = np.asarray(q_values, dtype=np.float64)
        g = _transition(q_arr, self.gamma_hat, self.c_hat, self.form)
        lam1 = self.params["y.L1"]
        lam2 = self.params["y.L1:G"]
        denom = lam1 + lam2 * g
        out: dict[str, FloatArray] = {"G": g}
        j = 0
        while f"x{j}.L1" in self.params.index:
            num = self.params[f"x{j}.L1"] + self.params[f"x{j}.L1:G"] * g
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"x{j}"] = np.where(np.abs(denom) > 1e-12, -num / denom, np.nan)
            j += 1
        return pd.DataFrame(out, index=pd.Index(q_arr, name="q"))

    def summary(self) -> str:
        """Readable report of the fit."""
        lines = [
            f"STAR-ARDL ({self.form.upper()}, Terasvirta 1994) - "
            f"order={self.order}, nobs={self.n_obs}",
            f"  gamma_hat = {self.gamma_hat:.4f}   c_hat = {self.c_hat:.4f}",
            f"  grid start: gamma={self.grid_start[0]:.4f}, c={self.grid_start[1]:.4f}",
            f"  SSR = {self.ssr:.4f}",
            f"  Linearity (Terasvirta LM): stat={self.linearity_stat:.4f}  "
            f"p-value={self.linearity_pvalue:.4f}",
        ]
        if self.form_selected is not None:
            lines.append(f"  form selected (auto): {self.form_selected}")
        return "\n".join(lines)


def star_ardl(
    y: ArrayLike,
    x: ArrayLike,
    transition_var: ArrayLike,
    form: Form | Literal["auto"] = "lstar",
    order: tuple[int, int] = (1, 1),
    start_grid: int = 15,
    trim: float = 0.15,
) -> STARARDLResults:
    r"""Fit a STAR-ARDL: a smooth-transition UECM with a concentrated NLS.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like
        Regressors (no constant needed — one is added automatically for
        the level terms, matching :func:`pyardl.threshold.hansen.threshold_ardl`'s
        convention of an explicit design but here built in, since the
        UECM in spec 34 §2.2 always carries a deterministic term).
    transition_var : array_like
        Transition variable :math:`q_t`; entered at lag 1,
        :math:`q_{t-1}`, as in spec 34 §2.2.
    form : {'lstar', 'estar', 'auto'}, default 'lstar'
        Transition function (see the module notes). ``'auto'`` runs
        Teräsvirta's (1994) decision rule (approximated — see
        :func:`_teraesvirta_lm_test`) on the linearity test's auxiliary
        regression and picks the form it points to.
    order : tuple of int, default (1, 1)
        ``(p, q)``: lags of :math:`\Delta y` and of each
        :math:`\Delta x_j`. The level terms :math:`y_{t-1}`,
        :math:`x_{t-1}` are always regime-varying; the short-run
        difference terms are not (spec 34 §2.2's model).
    start_grid : int, default 15
        Grid resolution per dimension for the coarse
        :math:`(\gamma, c)` search that seeds the local optimiser.
    trim : float, default 0.15
        Fraction of :math:`q`'s empirical distribution excluded at each
        end when building the candidate grid for :math:`c`.

    Returns
    -------
    STARARDLResults

    Notes
    -----
    Weak identification of :math:`\gamma` and :math:`c` separately is
    common in small samples, when the transition is nearly sharp or
    nearly absent — a property of the STAR literature in general, not
    specific to this implementation. Report :attr:`gamma_hat` and
    :attr:`c_hat` together with the linearity test rather than trusting
    the point estimates alone in that regime.

    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.threshold import star_ardl
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> q = np.cumsum(rng.standard_normal(n)) * 0.1
    >>> x = np.cumsum(rng.standard_normal(n))
    >>> g_true = 1.0 / (1.0 + np.exp(-3.0 * (q - 0.0)))
    >>> beta = 1.0 + 2.0 * g_true
    >>> y = beta * x + rng.standard_normal(n) * 0.3
    >>> res = star_ardl(y, x, transition_var=q, form="lstar", order=(1, 1))
    >>> res.linearity_pvalue < 0.05
    True
    """
    if form not in ("lstar", "estar", "auto"):
        raise ValueError(f"form={form!r} must be 'lstar', 'estar' or 'auto'.")
    if not 0.0 < trim < 0.5:
        raise ValueError(f"trim={trim} must be in (0, 0.5).")

    y_arr, _, _, _, _ = check_series(y, None)
    x_arr, _ = check_regressors_allow_constant(x, y_arr.shape[0])
    q_arr = np.asarray(transition_var, dtype=np.float64).ravel()
    if q_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"transition_var must have the same length as y: got "
            f"{q_arr.shape[0]} against {y_arr.shape[0]}."
        )

    lm_f, lm_p, p_linear, p_quadratic, _ = _teraesvirta_lm_test(
        y_arr, x_arr, q_arr, order
    )

    form_selected: Form | None = None
    chosen_form: Form
    if form == "auto":
        chosen_form = "estar" if p_quadratic < p_linear else "lstar"
        form_selected = chosen_form
    else:
        chosen_form = form

    q_sorted = np.sort(q_arr)
    lo = int(np.floor(trim * q_sorted.size))
    hi = int(np.ceil((1.0 - trim) * q_sorted.size))
    c_candidates = np.linspace(q_sorted[lo], q_sorted[hi - 1], start_grid)
    sigma_q = float(np.std(q_arr))
    gamma_candidates = np.geomspace(
        0.1 / max(sigma_q, 1e-6), 20.0 / max(sigma_q, 1e-6), start_grid
    )

    best_ssr = np.inf
    best_start = (float(gamma_candidates[0]), float(c_candidates[0]))
    for gamma0 in gamma_candidates:
        for c0 in c_candidates:
            ssr = _ssr(np.array([gamma0, c0]), y_arr, x_arr, q_arr, order, chosen_form)
            if ssr < best_ssr:
                best_ssr = ssr
                best_start = (float(gamma0), float(c0))

    result = minimize(
        _ssr,
        x0=np.array(best_start),
        args=(y_arr, x_arr, q_arr, order, chosen_form),
        method="Nelder-Mead",
    )
    gamma_hat, c_hat = float(result.x[0]), float(result.x[1])

    design, target, names = _design(
        y_arr, x_arr, q_arr, order, gamma_hat, c_hat, chosen_form
    )
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    ssr_final = float(resid @ resid)

    return STARARDLResults(
        form=chosen_form,
        gamma_hat=gamma_hat,
        c_hat=c_hat,
        params=pd.Series(beta, index=names, name="coef"),
        ssr=ssr_final,
        linearity_stat=lm_f,
        linearity_pvalue=lm_p,
        form_selected=form_selected,
        n_obs=target.shape[0],
        order=order,
        grid_start=best_start,
    )
