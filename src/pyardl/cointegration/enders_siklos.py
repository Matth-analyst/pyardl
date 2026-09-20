r"""Enders & Siklos (2001) — asymmetric (TAR / Momentum-TAR) cointegration.

Engle-Granger tests a unit root on the step-one residual through a
**symmetric** ADF regression: :math:`\Delta \hat u_t = \rho \hat
u_{t-1} + \dots`. Enders-Siklos asks whether the residual reverts to
zero faster after a positive deviation than after a negative one (or
the reverse) — the "asymmetric adjustment of the cointegrating
residual" analogue of NARDL's "asymmetric response to a regressor"
(:mod:`pyardl.nardl`). The two notions of asymmetry are different and
must never be presented as interchangeable.

Two variants, both implemented (same discipline as the five PSS
deterministic cases):

- **TAR**: the regime depends on the *sign of the residual itself*,
  :math:`\hat u_{t-1}`.
- **Momentum-TAR (M-TAR)**: the regime depends on the *sign of the
  residual's change*, :math:`\Delta \hat u_{t-1}` — a speed asymmetry
  (the residual falls fast but climbs back slowly, or the reverse),
  distinct from the level asymmetry TAR captures.

Step one is Engle-Granger's, reused unchanged
(:func:`pyardl.cointegration.engle_granger.engle_granger`); only step
two's regression changes, splitting the adjustment coefficient by
regime instead of pooling it.

References
----------
.. [1] Enders, W. & Siklos, P. L. (2001). Cointegration and threshold
       adjustment. *Journal of Business & Economic Statistics*, 19(2),
       166-176.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from pyardl.bootstrap.dgp import _fit_marginal_var
from pyardl.cointegration.engle_granger import EGTrend, engle_granger
from pyardl.unitroot.gls import select_lags
from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

Variant = Literal["tar", "mtar"]
CVSource = Literal["bootstrap"]

__all__ = ["enders_siklos", "EndersSiklosResults"]


@dataclass(frozen=True)
class EndersSiklosResults:
    """Outcome of an Enders-Siklos asymmetric cointegration test.

    Attributes
    ----------
    variant : {'tar', 'mtar'}
        Regime-switching rule used.
    threshold : float
        Threshold value (0.0 unless estimated).
    threshold_estimated : bool
        Whether the threshold was searched for (Chan 1993) rather than
        fixed at the caller's value.
    rho1, rho2 : float
        Adjustment coefficients in each regime.
    phi_stat : float
        Joint F-test statistic of :math:`H_0: \\rho_1 = \\rho_2 = 0`
        (no cointegration, either regime).
    phi_critical_values : dict
        Level to critical value of ``phi_stat``, from ``cv_source``.
    lags : int
        Lag order of the step-two regression.
    nobs : int
        Observations used in the step-two regression.
    resid : pandas.Series
        Step-one (Engle-Granger) residuals.
    symmetry_stat : float or None
        F-test statistic of :math:`H_0: \\rho_1 = \\rho_2`, conditional
        on cointegration being established by ``phi_stat``. ``None``
        when the threshold was estimated (searching it invalidates the
        asymptotic F distribution this test relies on) or when
        cointegration is not established.
    symmetry_pvalue : float or None
        Asymptotic p-value of ``symmetry_stat``.
    trend : str
        Deterministic terms of the (reused) step-one regression.
    cv_source : str
        How ``phi_critical_values`` were obtained. Only ``'bootstrap'``
        is implemented — see the module notes.
    n_boot : int
        Bootstrap replications for ``phi_critical_values``.
    seed : int or None
        Seed of the bootstrap generator.
    """

    variant: Variant
    threshold: float
    threshold_estimated: bool
    rho1: float
    rho2: float
    phi_stat: float
    phi_critical_values: dict[float, float]
    lags: int
    nobs: int
    resid: pd.Series
    symmetry_stat: float | None
    symmetry_pvalue: float | None
    trend: EGTrend
    cv_source: CVSource
    n_boot: int
    seed: int | None

    def decision(self, alpha: float = 0.05) -> str:
        """``'cointegration'`` or ``'no_cointegration'`` at ``alpha``."""
        if alpha not in self.phi_critical_values:
            raise ValueError(f"No bootstrap critical value at alpha={alpha}.")
        return (
            "cointegration"
            if self.phi_stat > self.phi_critical_values[alpha]
            else "no_cointegration"
        )

    def ecm_asymmetric(self, y: ArrayLike, x: ArrayLike) -> pd.DataFrame:
        """Step-two error-correction model with a regime-split adjustment term.

        Parameters
        ----------
        y, x : array_like
            The same series passed to :func:`enders_siklos` (not stored
            on the result, to avoid holding large arrays in every
            instance).

        Returns
        -------
        pandas.DataFrame
            Coefficients ``ecm.regime1``, ``ecm.regime2`` (adjustment
            speeds by regime) and ``D.<name>`` for each regressor, with
            ``se``, ``t`` and ``pvalue`` columns.
        """
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        resid = self.resid.to_numpy()
        indicator = _regime_indicator(resid, self.variant, self.threshold)[:-1]

        dy = np.diff(y_arr)
        dx = np.diff(x_arr, axis=0)
        lagged = resid[:-1]
        regime1 = indicator * lagged
        regime2 = (1.0 - indicator) * lagged

        design = np.column_stack([regime1, regime2, dx])
        names = ["ecm.regime1", "ecm.regime2"] + [f"D.x{j}" for j in range(dx.shape[1])]

        beta, _, _, _ = np.linalg.lstsq(design, dy, rcond=None)
        err = dy - design @ beta
        n, m = design.shape
        sigma2 = float(err @ err) / (n - m)
        xtx_inv = np.linalg.inv(design.T @ design)
        se = np.sqrt(sigma2 * np.diag(xtx_inv))
        tvalues = beta / se
        return pd.DataFrame(
            {
                "coef": beta,
                "se": se,
                "t": tvalues,
                "pvalue": 2 * t_dist.sf(np.abs(tvalues), n - m),
            },
            index=names,
        )

    def summary(self) -> str:
        """Readable report of the test."""
        cv = "  ".join(
            f"{int(a * 100)}%: {v:.4f}"
            for a, v in sorted(self.phi_critical_values.items())
        )
        thr = "estimated" if self.threshold_estimated else self.threshold
        lines = [
            f"Enders-Siklos test (2001) - variant '{self.variant}', "
            f"threshold={thr}, lags={self.lags}, nobs={self.nobs}",
            f"  rho1={self.rho1:.4f}  rho2={self.rho2:.4f}",
            f"  Phi statistic = {self.phi_stat:.4f}   "
            f"decision (5%): {self.decision(0.05)}",
            f"  Phi critical values ({self.n_boot} bootstrap reps, "
            f"left is 'no coint.')   {cv}",
            "  H0 (Phi): rho1 = rho2 = 0 (no cointegration)",
        ]
        if self.symmetry_stat is not None:
            lines.append(
                f"  Symmetry test (H0: rho1=rho2): stat={self.symmetry_stat:.4f}  "
                f"p-value={self.symmetry_pvalue:.4f}"
            )
        else:
            lines.append(
                "  Symmetry test: not computed (threshold estimated, or "
                "cointegration not established at 5%)"
            )
        return "\n".join(lines)


def _regime_indicator(
    resid: FloatArray, variant: Variant, threshold: float
) -> FloatArray:
    """``I_t``: 1 in the "above threshold" regime, 0 otherwise.

    Has the same length as ``resid``; entry ``t`` reads ``resid[t-1]``
    (TAR) or ``resid[t] - resid[t-1]`` (M-TAR) — computed on the full
    series so callers can slice consistently with the regression rows
    they build (row ``t`` of the step-two regression uses
    ``indicator[t]`` against ``resid[t-1]``, i.e. one lag behind).
    """
    if variant == "tar":
        level = resid
    elif variant == "mtar":
        level = np.empty_like(resid)
        level[0] = np.nan
        level[1:] = np.diff(resid)
    else:
        raise ValueError(f"variant={variant!r} must be 'tar' or 'mtar'.")
    return (level >= threshold).astype(np.float64)


def _threshold_regression(
    resid: FloatArray, variant: Variant, threshold: float, lags: int
) -> tuple[float, float, float, int, int, float]:
    """Fit the two-regime step-two regression.

    Returns rho1, rho2, phi_stat, nobs, dof, ssr_u.

    ``phi_stat`` is the joint F-test of rho1 = rho2 = 0 against the
    regression that drops the level terms entirely (pure lagged
    differences), the natural restricted alternative for this null.
    """
    indicator_full = _regime_indicator(resid, variant, threshold)
    du = np.diff(resid)
    n = du.size
    if n - lags < lags + 3:
        raise ValueError(
            f"Sample too short for {lags} lags: {n + 1} residual observations "
            f"leave {n - lags} usable rows."
        )
    target = du[lags:]
    lagged_level = resid[lags:-1]
    # Row i predicts Delta u at t = lags+i+1 from u_{t-1} = lagged_level[i]
    # = resid[lags+i]; the regime dummy for that row must read the SAME
    # lag, resid[lags+i] (TAR) or its own lagged difference (M-TAR) — i.e.
    # indicator_full[lags+i], the same slice as lagged_level, not one step
    # ahead. Reading indicator_full[lags+i+1] instead would let the dummy
    # see the change it is trying to predict (M-TAR) or next period's level
    # (TAR): a look-ahead bias that inflates phi_stat sharply, worst for
    # M-TAR where it is exactly the target's own sign.
    indicator = indicator_full[lags:-1]

    if indicator.max() == indicator.min():
        raise ValueError(
            "The threshold splits the sample into an empty regime: no "
            "observation falls on the other side. Choose a different "
            "threshold, or use threshold='estimated'."
        )

    regime1 = indicator * lagged_level
    regime2 = (1.0 - indicator) * lagged_level
    cols = [regime1, regime2]
    for j in range(1, lags + 1):
        cols.append(du[lags - j : n - j])
    design_u = np.column_stack(cols)

    beta_u, _, _, _ = np.linalg.lstsq(design_u, target, rcond=None)
    resid_u = target - design_u @ beta_u
    ssr_u = float(resid_u @ resid_u)
    n_obs, k_u = design_u.shape
    dof = n_obs - k_u

    design_r = design_u[:, 2:]
    if design_r.shape[1] == 0:
        ssr_r = float(target @ target)
    else:
        beta_r, _, _, _ = np.linalg.lstsq(design_r, target, rcond=None)
        resid_r = target - design_r @ beta_r
        ssr_r = float(resid_r @ resid_r)

    phi = ((ssr_r - ssr_u) / 2.0) / (ssr_u / dof) if ssr_u > 0 and dof > 0 else 0.0
    return float(beta_u[0]), float(beta_u[1]), max(phi, 0.0), n_obs, dof, ssr_u


def _estimate_threshold(
    resid: FloatArray, variant: Variant, lags: int, trim: float = 0.15
) -> float:
    """Chan (1993): grid search over candidate thresholds, minimise SSR."""
    level = resid if variant == "tar" else np.diff(resid, prepend=resid[0])
    candidates = np.sort(np.unique(level))
    lo = int(np.floor(trim * candidates.size))
    hi = int(np.ceil((1.0 - trim) * candidates.size))
    candidates = candidates[lo:hi]
    if candidates.size == 0:
        raise ValueError(
            "trim leaves no candidate threshold; lower trim or use more data."
        )

    best_ssr = np.inf
    best_tau = float(candidates[0])
    for tau in candidates:
        try:
            _, _, _, _, _, ssr = _threshold_regression(resid, variant, float(tau), lags)
        except ValueError:
            continue
        if ssr < best_ssr:
            best_ssr = ssr
            best_tau = float(tau)
    return best_tau


def _bootstrap_phi_cv(
    rng: np.random.Generator,
    y: FloatArray,
    x: FloatArray,
    trend: EGTrend,
    variant: Variant,
    threshold: float,
    threshold_estimated: bool,
    ic: str,
    max_lags: int | None,
    var_order: int,
    n_boot: int,
) -> dict[float, float]:
    """Bootstrap null distribution of ``phi_stat`` under H0: no cointegration.

    The null of Enders-Siklos, like Gregory-Hansen's, is "no
    cointegration": :math:`y` and :math:`x` are independent I(1)
    processes. The step-one Engle-Granger *residual* is therefore not
    itself a primitive object with a simple, self-contained short-run
    dynamic — its distribution under the null comes from the spurious
    regression of one independent random walk on another (Phillips
    1986), not from resampling its own differences in isolation. An
    earlier version of this function did exactly that (fit an AR model
    directly to :math:`\\Delta \\hat u_t` and resample), which measurably
    under-sized the bootstrap critical values (empirical rejection rate
    ~24% at a nominal 5%, see ``docs/QUESTIONS.md``, spec 33 §5.1).

    The fix, and the only correct null DGP for this test, mirrors
    :mod:`pyardl.cointegration.gregory_hansen`: regenerate :math:`y^*,
    x^*` **jointly** as independent I(1) series (a VAR-in-differences
    fitted to the stacked system, :func:`pyardl.bootstrap.dgp._fit_marginal_var`,
    reused unchanged), then rerun the **entire** step-one Engle-Granger
    regression and step-two threshold test on the regenerated data —
    not just step two on a resampled residual.
    """
    n = y.shape[0]
    stacked = np.column_stack([y, x])
    d_stacked = np.diff(stacked, axis=0)
    const, ar, boot_resid = _fit_marginal_var(d_stacked, var_order)
    boot_resid = boot_resid - boot_resid.mean(axis=0)
    k = stacked.shape[1]

    phi_boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draws = rng.integers(0, boot_resid.shape[0], size=n - 1)
        innovations = boot_resid[draws]
        dz = np.empty((n - 1, k), dtype=np.float64)
        history: FloatArray = (
            np.tile(stacked[0], (var_order, 1)) if var_order > 0 else np.empty((0, k))
        )
        for t in range(n - 1):
            val = const.copy()
            for i in range(var_order):
                val = val + ar[i] @ history[-(i + 1)]
            val = val + innovations[t]
            dz[t] = val
            if var_order > 0:
                history = np.asarray(np.vstack([history, val])[1:], dtype=np.float64)
        z0 = stacked[0]
        z_path = np.vstack([z0, z0 + np.cumsum(dz, axis=0)])
        y_b, x_b = z_path[:, 0], z_path[:, 1:]
        try:
            eg_b = engle_granger(y_b, x_b, trend=trend, max_lags=max_lags, ic=ic)
            resid_b = eg_b.resid.to_numpy()
            lags_b, _ = select_lags(resid_b, method=ic, max_lags=max_lags)  # type: ignore[arg-type]
            tau_b = (
                _estimate_threshold(resid_b, variant, lags_b)
                if threshold_estimated
                else threshold
            )
            _, _, phi_b, _, _, _ = _threshold_regression(
                resid_b, variant, tau_b, lags_b
            )
        except ValueError:
            phi_b = 0.0
        phi_boot[b] = phi_b

    return {
        alpha: float(np.quantile(phi_boot, 1.0 - alpha)) for alpha in (0.01, 0.05, 0.10)
    }


def enders_siklos(
    y: ArrayLike,
    x: ArrayLike,
    variant: Variant = "tar",
    threshold: float | Literal["estimated"] = 0.0,
    trend: EGTrend = "c",
    max_lags: int | None = None,
    ic: str = "aic",
    cv_source: CVSource = "bootstrap",
    var_order: int = 1,
    n_boot: int = 999,
    seed: int | None = None,
) -> EndersSiklosResults:
    r"""Test for cointegration with asymmetric (TAR/M-TAR) adjustment.

    Parameters
    ----------
    y, x : array_like
        As in :func:`pyardl.cointegration.engle_granger.engle_granger`
        (step one is reused unchanged).
    variant : {'tar', 'mtar'}, default 'tar'
        ``'tar'`` splits the regime on the sign of the residual level
        :math:`\hat u_{t-1}`; ``'mtar'`` on the sign of its change
        :math:`\Delta \hat u_{t-1}` — a different notion of asymmetry
        (speed rather than level), see the module notes.
    threshold : float or 'estimated', default 0.0
        Fixed threshold (0.0 is the article's usual default), or
        ``'estimated'`` to search for it by grid search minimising SSR
        (Chan 1993). Searching the threshold invalidates the asymptotic
        distribution of :attr:`EndersSiklosResults.symmetry_stat`,
        which is then left ``None`` (see the module notes).
    trend : {'n', 'c', 'ct', 'ctt'}, default 'c'
        Deterministic terms of the (reused) step-one regression.
    max_lags, ic :
        Passed to :func:`~pyardl.unitroot.gls.select_lags` for the
        step-two lag order.
    cv_source : {'bootstrap'}, default 'bootstrap'
        How ``phi_critical_values`` are obtained. The published
        Enders-Siklos critical values are simulation-tabulated and
        **not encoded here**: pyardl does not hold that table with a
        verified provenance (CLAUDE.md rule 9). ``cv_source='table'``
        is not implemented; see ``docs/QUESTIONS.md``.
    var_order : int, default 1
        Lag order of the bootstrap null's AR model on
        :math:`\Delta \hat u_t`.
    n_boot : int, default 999
        Bootstrap replications for ``phi_critical_values``.
    seed : int, optional
        Seed for the bootstrap's :class:`numpy.random.Generator`.

    Returns
    -------
    EndersSiklosResults

    Notes
    -----
    Report both TAR and M-TAR rather than only the one that "worked" —
    the choice changes conclusions on real data (Enders & Siklos'
    original exchange-rate application), and presenting a single
    variant as sufficient without justifying the choice a priori is a
    form of specification search the spec explicitly warns against.


    Examples
    --------
    >>> import numpy as np
    >>> from pyardl.cointegration import enders_siklos
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = np.cumsum(rng.standard_normal(n))
    >>> u = np.zeros(n)
    >>> for t in range(1, n):
    ...     rho = -0.6 if u[t - 1] >= 0 else -0.1
    ...     u[t] = (1 + rho) * u[t - 1] + 0.3 * rng.standard_normal()
    >>> y = 1.5 * x + u
    >>> res = enders_siklos(y, x, variant="tar", n_boot=99, seed=0)
    >>> res.decision(0.05)
    'cointegration'
    """
    if cv_source != "bootstrap":
        raise NotImplementedError(
            "cv_source='table' is not implemented: pyardl does not hold a "
            "verified copy of the Enders-Siklos (2001) published critical "
            "value table (CLAUDE.md rule 9). Use cv_source='bootstrap' "
            "(the default). See docs/QUESTIONS.md."
        )
    if variant not in ("tar", "mtar"):
        raise ValueError(f"variant={variant!r} must be 'tar' or 'mtar'.")

    y_arr, x_arr, _, _, _ = check_series(y, x)
    if x_arr is None:
        raise ValueError("Enders-Siklos needs at least one regressor.")

    eg = engle_granger(y_arr, x_arr, trend=trend, max_lags=max_lags, ic=ic)
    resid = eg.resid.to_numpy()

    chosen_lags, _ = select_lags(resid, method=ic, max_lags=max_lags)  # type: ignore[arg-type]

    threshold_estimated = threshold == "estimated"
    tau = (
        _estimate_threshold(resid, variant, chosen_lags)
        if threshold_estimated
        else float(threshold)
    )

    rho1, rho2, phi_stat, n_obs, dof, _ = _threshold_regression(
        resid, variant, tau, chosen_lags
    )

    rng = np.random.default_rng(seed)
    phi_cv = _bootstrap_phi_cv(
        rng,
        y_arr,
        x_arr,
        trend,
        variant,
        tau,
        threshold_estimated,
        ic,
        max_lags,
        var_order,
        n_boot,
    )

    symmetry_stat: float | None = None
    symmetry_pvalue: float | None = None
    if not threshold_estimated:
        du = np.diff(resid)
        indicator = _regime_indicator(resid, variant, tau)[chosen_lags:-1]
        lagged_level = resid[chosen_lags:-1]
        regime1 = indicator * lagged_level
        regime2 = (1.0 - indicator) * lagged_level
        cols = [regime1, regime2]
        for j in range(1, chosen_lags + 1):
            cols.append(du[chosen_lags - j : du.size - j])
        design_u = np.column_stack(cols)
        target = du[chosen_lags:]
        beta_u, _, _, _ = np.linalg.lstsq(design_u, target, rcond=None)
        resid_u = target - design_u @ beta_u
        ssr_u = float(resid_u @ resid_u)

        r_vec = np.zeros(design_u.shape[1])
        r_vec[0], r_vec[1] = 1.0, -1.0
        xtx_inv = np.linalg.inv(design_u.T @ design_u)
        sigma2 = ssr_u / dof
        var_diff = float(r_vec @ xtx_inv @ r_vec) * sigma2
        diff = float(beta_u[0] - beta_u[1])
        if var_diff > 0:
            sym_f = (diff**2) / var_diff
            symmetry_stat = sym_f
            symmetry_pvalue = float(f_dist.sf(sym_f, 1, dof))

    return EndersSiklosResults(
        variant=variant,
        threshold=tau,
        threshold_estimated=threshold_estimated,
        rho1=rho1,
        rho2=rho2,
        phi_stat=phi_stat,
        phi_critical_values=phi_cv,
        lags=chosen_lags,
        nobs=n_obs,
        resid=eg.resid,
        symmetry_stat=symmetry_stat,
        symmetry_pvalue=symmetry_pvalue,
        trend=trend,
        cv_source=cv_source,
        n_boot=n_boot,
        seed=seed,
    )
