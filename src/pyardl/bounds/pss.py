r"""Bounds test for the existence of a level relationship (PSS 2001).

The test is run on the unrestricted error-correction form

    Δy_t = det_t + lam*y_{t-1} + sum_j gamma_j x_{j,t-1}
           + sum_i psi_i Δy_{t-i} + sum_j sum_i omega_{j,i} Δx_{j,t-i} + eps_t

and answers a question that ordinary cointegration tests cannot: is there
a long-run relationship between y and the x's, *without* having to know
first whether the regressors are I(0) or I(1)?

Two statistics are computed:

1. ``F_overall`` tests ``lam = gamma_1 = ... = gamma_k = 0``, i.e. no
   level relationship at all. Under cases II and IV the restricted
   deterministic term is part of the tested vector, which gives ``k+2``
   restrictions instead of ``k+1``.
2. ``t_BDM`` tests ``lam = 0`` alone. It is a **left-tailed** test:
   rejection requires an estimated ``lam`` that is negative, i.e. an
   actual pull back towards equilibrium.

Because the limiting distributions depend on the unknown integration
order of the regressors, critical values come in pairs: a lower bound
(all regressors I(0)) and an upper bound (all regressors I(1)). The
outcome therefore has **three** states rather than two —
``"cointegration"``, ``"no_cointegration"`` and ``"inconclusive"`` when
the statistic falls between the bounds.

Note on ``q_j = 0``: a regressor with no lags of its own enters the
tested vector through its contemporaneous level ``x_{j,t}``. This does
not change the asymptotics, since ``x_{j,t} = x_{j,t-1} + Δx_{j,t}`` and
the difference is stationary; only the dating shifts by one period.

References
----------
Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). "Bounds Testing
Approaches to the Analysis of Level Relationships", *Journal of Applied
Econometrics*, 16(3), 289-326.
Banerjee, A., Dolado, J. & Mestre, R. (1998). "Error-correction
Mechanism Tests for Cointegration in a Single-equation Framework",
*Journal of Time Series Analysis*, 19(3), 267-283.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

from pyardl.bounds.classification import Classification, classify
from pyardl.core.ardl import ARDL
from pyardl.critical_values import get_bounds
from pyardl.critical_values.smg2019 import findep_bounds
from pyardl.exceptions import DegenerateCaseWarning, PyardlMethodologyWarning
from pyardl.utils import check_series

FloatArray = npt.NDArray[np.float64]

Decision = Literal["cointegration", "no_cointegration", "inconclusive"]

_CASE_DET = {1: "none", 2: "const", 3: "const", 4: "trend", 5: "trend"}
_CASE_RESTRICTED_DET = {2: "const", 4: "trend"}


@dataclass
class _UECMFit:
    """OLS fit of the unrestricted error-correction model (internal)."""

    params: pd.Series
    cov: FloatArray
    resid: FloatArray
    ssr: float
    design: FloatArray
    names: list[str]
    tested: list[str]  # columns entering the F test
    lam_name: str

    @property
    def nobs(self) -> int:
        return len(self.resid)

    @property
    def df_resid(self) -> int:
        return self.nobs - len(self.names)


def _estimate_uecm(
    y: FloatArray,
    x: FloatArray,
    x_names: tuple[str, ...],
    y_name: str,
    p: int,
    q: tuple[int, ...],
    case: int,
    fixed: FloatArray | None = None,
    fixed_names: tuple[str, ...] = (),
) -> _UECMFit:
    """Build and fit the error-correction model for the requested case."""
    n, k = x.shape
    start = max([p, *q]) if q else p
    if start < 1:
        raise ValueError(
            "p >= 1 is required: the error-correction model needs y_{t-1}."
        )
    dy = np.diff(y)
    dx = np.diff(x, axis=0)

    cols: list[FloatArray] = []
    names: list[str] = []
    tested: list[str] = []

    det = _CASE_DET[case]
    if det in ("const", "trend"):
        cols.append(np.ones(n - start))
        names.append("const")
    if det == "trend":
        cols.append(np.arange(start + 1, n + 1, dtype=np.float64))
        names.append("trend")
    if case in _CASE_RESTRICTED_DET:
        tested.append(_CASE_RESTRICTED_DET[case])

    lam_name = f"{y_name}.L1"
    cols.append(y[start - 1 : n - 1])
    names.append(lam_name)
    tested.append(lam_name)

    for j, name in enumerate(x_names):
        if q[j] == 0:
            # Contemporaneous level (q_j = 0 convention, see the module
            # docstring): still I(1) under the null.
            cols.append(x[start:n, j])
            names.append(f"{name}.L0")
            tested.append(f"{name}.L0")
        else:
            cols.append(x[start - 1 : n - 1, j])
            names.append(f"{name}.L1")
            tested.append(f"{name}.L1")

    for i in range(1, p):
        cols.append(dy[start - i - 1 : n - i - 1])
        names.append(f"D.{y_name}.L{i}")
    for j, name in enumerate(x_names):
        for i in range(q[j]):
            cols.append(dx[start - i - 1 : n - i - 1, j])
            names.append(f"D.{name}.L{i}")

    if fixed is not None:
        # Unlagged z_t (dummies and the like): never part of the tested vector.
        cols.extend(fixed[start:].T)
        names.extend(fixed_names)

    design = np.column_stack(cols)
    y_dep = dy[start - 1 :]

    coefs, _, rank, _ = np.linalg.lstsq(design, y_dep, rcond=None)
    if rank < design.shape[1]:
        warnings.warn(
            "Singular design matrix: the covariance estimates are unreliable.",
            PyardlMethodologyWarning,
            stacklevel=3,
        )
    resid = y_dep - design @ coefs
    ssr = float(resid @ resid)

    n_est, k_par = design.shape
    q_mat, r_mat = np.linalg.qr(design)
    r_inv = scipy.linalg.solve_triangular(r_mat, np.eye(k_par))
    xtx_inv = r_inv @ r_inv.T
    cov = (ssr / (n_est - k_par)) * xtx_inv

    return _UECMFit(
        params=pd.Series(coefs, index=names, name="coef"),
        cov=cov.astype(np.float64),
        resid=resid.astype(np.float64),
        ssr=ssr,
        design=design,
        names=names,
        tested=tested,
        lam_name=lam_name,
    )


def _wald_subset(fit: _UECMFit, names: list[str]) -> float:
    """Wald F statistic on an arbitrary subset of the fitted columns.

    Algebraically identical to the F computed from restricted and
    unrestricted sums of squared residuals.
    """
    if not names:
        raise ValueError("The tested vector is empty: there is nothing to test.")
    idx = [fit.names.index(name) for name in names]
    r_vec = fit.params.to_numpy()[idx]
    v_sub = fit.cov[np.ix_(idx, idx)]
    return float(r_vec @ np.linalg.solve(v_sub, r_vec)) / len(idx)


def _wald_f(fit: _UECMFit) -> float:
    """Overall F statistic: all level terms jointly zero."""
    return _wald_subset(fit, fit.tested)


def _indep_names(fit: _UECMFit) -> list[str]:
    """Columns entering ``F_indep``: the tested vector minus lambda.

    The null of the third test is that the levels of the *independent*
    variables carry no long-run relationship, ``gamma = 0``, leaving the
    adjustment coefficient free.

    Under cases 2 and 4 the restricted deterministic term stays in. That
    looks odd — a constant is not an independent variable — but it is
    where Pesaran, Shin & Smith put it: in those cases the deterministic
    belongs to the cointegrating vector itself, so testing that vector
    for the absence of a long-run relationship has to include it. Drop it
    and the restriction count no longer matches the critical values.
    """
    return [name for name in fit.tested if name != fit.lam_name]


def _wald_f_indep(fit: _UECMFit) -> float:
    """F statistic of Sam, McNown & Goh (2019): ``gamma = 0``."""
    return _wald_subset(fit, _indep_names(fit))


JointDecision = Literal[
    "cointegration", "no_cointegration", "inconclusive", "degenerate_suspicion"
]


def _joint_decision(
    decision_f: Decision, decision_t: Decision | None
) -> JointDecision | None:
    """Combine the F and t decisions into a single verdict.

    Establishing a level relationship requires **both** tests to agree:

    - both reject -> ``"cointegration"``;
    - F rejects but t does not -> ``"degenerate_suspicion"``. The level
      terms are jointly significant, yet y shows no pull back towards
      equilibrium; the apparent relationship is likely carried by the
      regressors alone;
    - neither rejects -> ``"no_cointegration"``;
    - any other disagreement -> ``"inconclusive"``;
    - t unavailable (cases II and IV) -> ``None``.
    """
    if decision_t is None:
        return None
    if decision_f == "cointegration":
        if decision_t == "cointegration":
            return "cointegration"
        return "degenerate_suspicion"
    if decision_f == "no_cointegration" and decision_t == "no_cointegration":
        return "no_cointegration"
    return "inconclusive"


def _findep_decision(stat: float, case: int, k: int, alpha: float) -> Decision | None:
    """Verdict of F_indep, or ``None`` when no bounds cover the setting.

    The bounds are simulated rather than transcribed (see
    :mod:`pyardl.critical_values.smg2019`); outside the simulated grid no
    neighbouring value is substituted, the test is simply reported as
    unavailable.
    """
    try:
        lo, up = findep_bounds(case, k, alpha)
    except ValueError:
        return None
    return _classify(stat, lo, up, left_tail=False)


def _classify(stat: float, lower: float, upper: float, *, left_tail: bool) -> Decision:
    """Classify a statistic against its bounds, in three states."""
    if left_tail:  # t_BDM: reject when t is below the (more negative) I(1) bound
        if stat < upper:
            return "cointegration"
        if stat > lower:
            return "no_cointegration"
    else:
        if stat > upper:
            return "cointegration"
        if stat < lower:
            return "no_cointegration"
    return "inconclusive"


@dataclass
class BoundsTestResults:
    """Outcome of a bounds test.

    Attributes
    ----------
    f_stat, t_stat, f_indep_stat : float
        The three test statistics of the Sam-McNown-Goh framework.
    decision_f, decision_t, decision_indep : str or None
        Three-state verdicts, ``"cointegration"`` / ``"no_cointegration"``
        / ``"inconclusive"``. ``decision_t`` is ``None`` when no t bounds
        are available for the chosen case and critical-value source.
    decision_joint : str or None
        Combined verdict of the two original tests, kept for continuity.
        May also be ``"degenerate_suspicion"``. The three-test verdict,
        which separates the two degeneracies instead of merely suspecting
        one, is :meth:`classification`.
    bounds : pandas.DataFrame
        Lower and upper bounds for F and t at the 10%, 5% and 1% levels.
    p_values : pandas.Series or None
        Approximate p-values at both bounds, when the critical-value
        source provides them.
    uecm : pandas.DataFrame
        Coefficients, standard errors and t-ratios of the fitted
        error-correction model.
    case, k, order, alpha, cv_source
        Settings the test was run with.
    """

    case: int
    k: int
    order: tuple[int, dict[str, int]]
    f_stat: float
    t_stat: float
    f_indep_stat: float
    alpha: float
    bounds: pd.DataFrame
    decision_f: Decision
    decision_t: Decision | None
    decision_indep: Decision | None
    decision_joint: JointDecision | None
    uecm: pd.DataFrame
    cv_source: str
    p_values: pd.Series | None  # (p_I0, p_I1) du F — None si indisponible
    _fit: _UECMFit = field(repr=False)

    def adjustment(self, alpha: float = 0.05) -> pd.Series:
        """Adjustment speed with a confidence interval, when it is valid.

        The usual normal confidence interval for ``lambda`` is only valid
        **once cointegration has been established**: under the null the
        distribution is non-standard, so an interval computed anyway would
        be misleading. If the joint decision is not ``"cointegration"``,
        the bounds are returned as NaN together with a
        :class:`~pyardl.exceptions.PyardlMethodologyWarning`; the point
        estimate and its standard error remain available.

        Parameters
        ----------
        alpha : float
            Significance level of the interval.

        Returns
        -------
        pandas.Series
            ``lambda``, ``se``, ``ci_lower`` and ``ci_upper``.
        """
        from scipy.stats import norm

        lam = float(self._fit.params[self._fit.lam_name])
        pos = self._fit.names.index(self._fit.lam_name)
        se = float(np.sqrt(self._fit.cov[pos, pos]))
        if self.decision_joint == "cointegration":
            z = float(norm.ppf(1 - alpha / 2))
            ci_lower, ci_upper = lam - z * se, lam + z * se
        else:
            warnings.warn(
                "Confidence interval withheld: cointegration is not "
                f"established (joint decision: {self.decision_joint}). The "
                "standard interval for the adjustment speed is only valid "
                "under cointegration.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )
            ci_lower = ci_upper = np.nan
        return pd.Series(
            {"lambda": lam, "se": se, "ci_lower": ci_lower, "ci_upper": ci_upper},
            name="adjustment",
        )

    def stability(self, alpha: float = 0.05) -> pd.DataFrame:
        """CUSUM and CUSUM-of-squares tests on the error-correction model.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level of the boundaries. One of 0.10, 0.05, 0.01.

        Returns
        -------
        pandas.DataFrame
            One row per test, with ``stable``, ``max_excess`` and
            ``first_crossing``.

        Notes
        -----
        A bounds test assumes the relationship it is testing held
        unchanged over the whole sample. A break makes the long-run
        coefficients an average of two different regimes, which is not a
        long-run relationship at all. These tests check that assumption,
        and applied work is expected to report both.
        """
        from pyardl.diagnostics import stability_tests

        y_dep = self._fit.design @ self._fit.params.to_numpy() + self._fit.resid
        return stability_tests(y_dep, self._fit.design, alpha=alpha)

    def diagnostics(self, alpha: float = 0.05) -> pd.DataFrame:
        """Residual and stability diagnostics of the error-correction model.

        Parameters
        ----------
        alpha : float, default 0.05
            Level of the stability boundaries.

        Returns
        -------
        pandas.DataFrame
            Ljung-Box (autocorrelation), Jarque-Bera (normality),
            Breusch-Pagan (heteroskedasticity), and the two
            parameter-constancy tests of Brown, Durbin & Evans (1975).

        Notes
        -----
        The two stability rows carry no p-value: they are
        boundary-crossing procedures, not statistics with a null
        distribution to integrate. Their ``statistic`` column reports the
        largest excursion beyond the band — zero when the model is
        stable. Use :meth:`stability` for the full verdict, including
        where the crossing occurs.
        """
        resid = self._fit.resid
        lb_lags = max(1, min(10, len(resid) // 5))
        lb = acorr_ljungbox(resid, lags=[lb_lags])
        jb_stat, jb_p, _, _ = jarque_bera(resid)
        bp_design = self._fit.design
        if not (bp_design[:, 0] == 1.0).all():
            bp_design = np.column_stack([np.ones(bp_design.shape[0]), bp_design])
        bp_p = float(het_breuschpagan(resid, bp_design)[1])
        stab = self.stability(alpha=alpha)
        pct = int(alpha * 100)
        return pd.DataFrame(
            {
                "statistic": [
                    float(lb["lb_stat"].iloc[0]),
                    float(jb_stat),
                    np.nan,
                    float(stab.loc["CUSUM", "max_excess"]),
                    float(stab.loc["CUSUM-of-squares", "max_excess"]),
                ],
                "pvalue": [
                    float(lb["lb_pvalue"].iloc[0]),
                    float(jb_p),
                    bp_p,
                    np.nan,
                    np.nan,
                ],
            },
            index=[
                f"Ljung-Box({lb_lags})",
                "Jarque-Bera",
                "Breusch-Pagan",
                f"CUSUM({pct}%) excess",
                f"CUSUMSQ({pct}%) excess",
            ],
        )

    def classification(self) -> tuple[Classification, str]:
        """Three-test verdict of Sam, McNown & Goh (2019), and its reason.

        Unlike :attr:`decision_joint`, which can only *suspect* a
        degeneracy, this tells the two apart: ``degenerate_1`` when y
        adjusts towards its own past while the regressors carry nothing,
        ``degenerate_2`` when the regressors' levels matter but nothing
        pulls y back. See :mod:`pyardl.bounds.classification`.

        Returns
        -------
        classification : str
            One of the keys of
            :data:`~pyardl.bounds.classification.CLASSIFICATIONS`.
        reason : str
            Which test decided, in one sentence.
        """
        return classify(self.decision_f, self.decision_t, self.decision_indep)

    def summary(self) -> str:
        """Return a readable report of the test as a string.

        Shows both statistics, their p-values at each bound when
        available, the decisions, and the critical value bounds at the
        10%, 5% and 1% levels. When the F decision is inconclusive, the
        p-value interval is displayed so that the result can be read on a
        continuous scale rather than as a bare verdict.
        """
        p, q = self.order
        q_desc = ", ".join(f"{n}:{v}" for n, v in q.items())

        decision_f_txt: str = self.decision_f
        if self.decision_f == "inconclusive" and self.p_values is not None:
            decision_f_txt = (
                f"inconclusive, p in [{self.p_values['p_I1']:.4f}, "
                f"{self.p_values['p_I0']:.4f}]"
            )
        p_line = (
            f"F p-values: p_I0 = {self.p_values['p_I0']:.4f}, "
            f"p_I1 = {self.p_values['p_I1']:.4f}"
            if self.p_values is not None
            else "F p-values: unavailable for this number of regressors"
        )
        if self.p_values is not None and "t_p_I0" in self.p_values.index:
            p_line += (
                f"\nt p-values: p_I0 = {self.p_values['t_p_I0']:.4f}, "
                f"p_I1 = {self.p_values['t_p_I1']:.4f}"
            )

        label, reason = self.classification()
        lines = [
            f"Bounds test (Pesaran, Shin & Smith 2001) - case {self.case}, "
            f"k={self.k}, ECM({p}; {q_desc}), critical values: {self.cv_source}",
            "",
            f"F_overall = {self.f_stat:.4f}   decision ({self.alpha:.0%}): "
            f"{decision_f_txt}",
            p_line,
            f"t_BDM     = {self.t_stat:.4f}   decision ({self.alpha:.0%}): "
            + (
                self.decision_t
                if self.decision_t is not None
                else f"not tabulated for case {self.case}"
            ),
            f"F_indep   = {self.f_indep_stat:.4f}   decision "
            f"({self.alpha:.0%}): "
            + (
                self.decision_indep
                if self.decision_indep is not None
                else "bounds unavailable for this configuration"
            ),
            "",
            f"CLASSIFICATION ({self.alpha:.0%}): {label}",
            f"  {reason}",
            "",
            self.bounds.to_string(float_format=lambda v: f"{v: .3f}"),
        ]
        return "\n".join(lines)


def _finalize_results(
    case: int,
    k: int,
    p: int,
    q_dict: dict[str, int],
    f_stat: float,
    t_stat: float,
    f_indep_stat: float,
    alpha: float,
    bounds_df: pd.DataFrame,
    decision_f: Decision,
    decision_t: Decision | None,
    decision_indep: Decision | None,
    decision_joint: JointDecision | None,
    cv_source: str,
    p_values: pd.Series | None,
    fit: _UECMFit,
) -> BoundsTestResults:
    """Run the autocorrelation check and assemble the result object."""
    lb_lags = max(1, min(10, fit.nobs // 5))
    lb_p = float(acorr_ljungbox(fit.resid, lags=[lb_lags])["lb_pvalue"].iloc[0])
    if lb_p < 0.05:
        warnings.warn(
            f"Autocorrelated residuals (Ljung-Box p={lb_p:.4f} < 0.05): the "
            "bounds test is not reliable. Increase p/q.",
            PyardlMethodologyWarning,
            stacklevel=3,
        )
    se = np.sqrt(np.diag(fit.cov))
    uecm_table = pd.DataFrame({"coef": fit.params, "se": se, "t": fit.params / se})
    return BoundsTestResults(
        case=case,
        k=k,
        order=(p, q_dict),
        f_stat=f_stat,
        t_stat=t_stat,
        f_indep_stat=f_indep_stat,
        alpha=alpha,
        bounds=bounds_df,
        decision_f=decision_f,
        decision_t=decision_t,
        decision_indep=decision_indep,
        decision_joint=decision_joint,
        uecm=uecm_table,
        cv_source=cv_source,
        p_values=p_values,
        _fit=fit,
    )


def bounds_test(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    case: int = 3,
    order: tuple[int, int | dict[str, int]] | None = None,
    ic: Literal["aic", "bic", "hq"] = "aic",
    max_p: int = 4,
    max_q: int = 4,
    alpha: float = 0.05,
    cv_source: Literal["kripfganz", "pss", "narayan"] = "kripfganz",
    finite_t: bool = False,
    fixed_regressors: npt.ArrayLike | None = None,
) -> BoundsTestResults:
    """Test whether a long-run level relationship exists between y and x.

    Fits the unrestricted error-correction model and confronts the F and
    t statistics with the appropriate pair of critical value bounds. See
    the module documentation for the interpretation of the three-state
    outcome.

    Parameters
    ----------
    y, x : array-like
        Dependent variable and level regressors.
    case : int
        Deterministic case, 1 to 5, following Pesaran, Shin & Smith:

        =====  ===============  ===============
        case   intercept        trend
        =====  ===============  ===============
        1      none             none
        2      restricted       none
        3      unrestricted     none
        4      unrestricted     restricted
        5      unrestricted     unrestricted
        =====  ===============  ===============

        Case 3 is the most common choice in applied work.
    order : tuple (p, q), optional
        Lag orders. If omitted, they are selected automatically with
        :meth:`~pyardl.core.ardl.ARDL.select_order` using ``ic``,
        ``max_p`` and ``max_q``.
    ic : {"aic", "bic", "hq"}
        Criterion used when ``order`` is not given.
    max_p, max_q : int
        Search bounds used when ``order`` is not given.
    alpha : float
        Significance level driving the reported decisions. The ``bounds``
        table always reports the 10%, 5% and 1% levels.
    cv_source : {"kripfganz", "pss", "narayan"}
        Where the critical values come from.

        - ``"kripfganz"`` (default): response surfaces, giving precise
          asymptotic F bounds at any level plus p-values at both bounds.
          The t bounds fall back to the published PSS tables.
        - ``"pss"``: the values published in PSS (2001), served
          unchanged. Use this to reproduce published results exactly.
        - ``"narayan"``: small-sample bounds (Narayan 2005), recommended
          when ``30 <= T <= 80``, where asymptotic bounds over-reject.
          Covers cases 2, 3 and 5, and the F statistic only.
    finite_t : bool
        Experimental and not validated; see
        :mod:`pyardl.critical_values.ks2020_finite`. Requires
        ``cv_source="kripfganz"``.
    fixed_regressors : array-like, shape (T, m), optional
        Variables entered without lags, such as dummies. They are never
        part of the tested vector, and are ignored by automatic order
        selection.

    Returns
    -------
    BoundsTestResults

    Notes
    -----
    The test is valid under assumptions you should check separately: the
    regressors are weakly exogenous, they are not themselves
    cointegrated, no variable is I(2), and the residuals are not
    autocorrelated. The last one is checked automatically and a warning
    is issued if the Ljung-Box test rejects.

    Examples
    --------
    >>> from pyardl.bounds import bounds_test
    >>> from pyardl.datasets import load_denmark
    >>> data = load_denmark()
    >>> res = bounds_test(
    ...     data["LRM"], data[["LRY", "IBO", "IDE"]],
    ...     case=3, order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}),
    ... )
    >>> res.decision_f
    'cointegration'
    """
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be between 1 and 5, got {case}.")
    if cv_source not in ("kripfganz", "pss", "narayan"):
        raise ValueError(f"Unknown cv_source: {cv_source!r}.")
    if finite_t and cv_source != "kripfganz":
        raise ValueError('finite_t=True requires cv_source="kripfganz".')
    if finite_t:
        warnings.warn(
            "finite_t=True is experimental and not validated: permission to "
            "use the underlying coefficient file is still pending with its "
            "authors, and no admissible comparison against a reference "
            "implementation has been carried out. Do not use in production.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    y_arr, x_arr, _, y_name, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("bounds_test requires x regressors.")
    k = x_arr.shape[1]

    if order is None:
        det = _CASE_DET[case]
        sel_det: Literal["const", "trend"] = "trend" if det == "trend" else "const"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyardlMethodologyWarning)
            sel = ARDL.select_order(y, x, max_p=max_p, max_q=max_q, ic=ic, det=sel_det)
        p, q_dict = sel.best_order
    else:
        from pyardl.core.ardl import _parse_order

        p, q_dict = _parse_order(order, x_names)
    q = tuple(q_dict[name] for name in x_names)

    fixed_arr: FloatArray | None = None
    fixed_names: tuple[str, ...] = ()
    if fixed_regressors is not None:
        fixed_arr = np.asarray(fixed_regressors, dtype=np.float64)
        if fixed_arr.ndim == 1:
            fixed_arr = fixed_arr[:, None]
        if fixed_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("fixed_regressors : longueur incompatible avec y.")
        if isinstance(fixed_regressors, pd.DataFrame):
            fixed_names = tuple(str(c) for c in fixed_regressors.columns)
        else:
            fixed_names = tuple(f"z.{j}" for j in range(fixed_arr.shape[1]))

    fit = _estimate_uecm(
        y_arr, x_arr, x_names, y_name, p, q, case, fixed_arr, fixed_names
    )

    f_stat = _wald_f(fit)
    lam_hat = float(fit.params[fit.lam_name])
    se_lam = float(
        np.sqrt(fit.cov[fit.names.index(fit.lam_name), fit.names.index(fit.lam_name)])
    )
    t_stat = lam_hat / se_lam
    f_indep_stat = _wald_f_indep(fit)
    decision_indep = _findep_decision(f_indep_stat, case, k, alpha)

    decision_t: Decision | None

    # Bounds at every available level. The F bounds come from the chosen
    # source; the t bounds come from the PSS tables unless finite_t is set.
    if finite_t:
        from pyardl.critical_values.ks2020_finite import (
            crit_value_bounds_finite,
            pvalue_bounds_finite,
        )

        # sr = number of short-run coefficients, fixed regressors included
        sr = (p - 1) + sum(q) + len(fixed_names)
        rows = []
        for a in (0.10, 0.05, 0.01):
            f_lo, f_up = crit_value_bounds_finite(case, k, fit.nobs, sr, a)
            t_lo, t_up = crit_value_bounds_finite(case, k, fit.nobs, sr, a, stat="t")
            rows.append(
                {"alpha": a, "F_I0": f_lo, "F_I1": f_up, "t_I0": t_lo, "t_I1": t_up}
            )
        bounds_df = pd.DataFrame(rows).set_index("alpha")

        f_lo, f_up = crit_value_bounds_finite(case, k, fit.nobs, sr, alpha)
        decision_f = _classify(f_stat, f_lo, f_up, left_tail=False)
        t_lo, t_up = crit_value_bounds_finite(case, k, fit.nobs, sr, alpha, stat="t")
        decision_t = _classify(t_stat, t_lo, t_up, left_tail=True)
        if lam_hat >= 0:
            warnings.warn(
                f"Estimated lambda = {lam_hat:.4f} >= 0: there is no pull "
                "back towards equilibrium, so the left-tailed t test has no "
                "cointegration interpretation here.",
                DegenerateCaseWarning,
                stacklevel=2,
            )
            decision_t = "no_cointegration"

        decision_joint = _joint_decision(decision_f, decision_t)
        if decision_joint == "degenerate_suspicion":
            warnings.warn(
                "F rejects but t does not: the level relationship may be "
                "degenerate (see the joint decision).",
                PyardlMethodologyWarning,
                stacklevel=2,
            )
        p_f = pvalue_bounds_finite(f_stat, case, k, fit.nobs, sr, fit.df_resid)
        p_t = pvalue_bounds_finite(
            t_stat, case, k, fit.nobs, sr, fit.df_resid, stat="t"
        )
        p_values_fin = pd.Series(
            {"p_I0": p_f[0], "p_I1": p_f[1], "t_p_I0": p_t[0], "t_p_I1": p_t[1]},
            name="pvalues_finite_t",
        )
        return _finalize_results(
            case,
            k,
            p,
            q_dict,
            f_stat,
            t_stat,
            f_indep_stat,
            alpha,
            bounds_df,
            decision_f,
            decision_t,
            decision_indep,
            decision_joint,
            cv_source,
            p_values_fin,
            fit,
        )

    rows = []
    for a in (0.10, 0.05, 0.01):
        f_lo, f_up = get_bounds(
            "F", case=case, k=k, alpha=a, cv_source=cv_source, t_obs=fit.nobs
        )
        try:
            t_lo, t_up = get_bounds("t", case=case, k=k, alpha=a)
        except ValueError:
            t_lo = t_up = np.nan
        rows.append(
            {"alpha": a, "F_I0": f_lo, "F_I1": f_up, "t_I0": t_lo, "t_I1": t_up}
        )
    bounds_df = pd.DataFrame(rows).set_index("alpha")

    f_lo, f_up = get_bounds(
        "F", case=case, k=k, alpha=alpha, cv_source=cv_source, t_obs=fit.nobs
    )
    decision_f = _classify(f_stat, f_lo, f_up, left_tail=False)

    if cv_source == "narayan" and case in (3, 5):
        decision_t = None
        warnings.warn(
            "Narayan (2005) publishes no t bounds, so no t decision is "
            'available with cv_source="narayan". The statistic is still '
            'reported; use cv_source="pss" for an asymptotic t decision, '
            "bearing in mind it over-rejects in small samples.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )
    elif case in (1, 3, 5):
        t_lo, t_up = get_bounds("t", case=case, k=k, alpha=alpha)
        decision_t = _classify(t_stat, t_lo, t_up, left_tail=True)
        if lam_hat >= 0:
            warnings.warn(
                f"Estimated lambda = {lam_hat:.4f} >= 0: there is no pull "
                "back towards equilibrium, so the left-tailed t test has no "
                "cointegration interpretation here.",
                DegenerateCaseWarning,
                stacklevel=2,
            )
            decision_t = "no_cointegration"
    else:
        decision_t = None
        warnings.warn(
            f"Case {case}: PSS (2001) does not tabulate the t statistic for "
            "cases with restricted deterministics, so no t decision is "
            "available. Use the F statistic.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    decision_joint = _joint_decision(decision_f, decision_t)
    if decision_joint == "degenerate_suspicion":
        warnings.warn(
            "F rejects but t does not: the level terms are jointly "
            "significant, yet y shows no pull back towards equilibrium. The "
            "relationship may be degenerate; cointegration is NOT "
            "established.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    # Approximate p-values at both bounds, when the source provides them
    p_values: pd.Series | None
    if 1 <= k <= 10:
        from pyardl.critical_values import pvalue_bounds

        p_i0, p_i1 = pvalue_bounds(f_stat, case=case, k=k)
        p_values = pd.Series({"p_I0": p_i0, "p_I1": p_i1}, name="F_pvalues")
    else:
        p_values = None  # k hors couverture des surfaces (k = 0)

    return _finalize_results(
        case,
        k,
        p,
        q_dict,
        f_stat,
        t_stat,
        f_indep_stat,
        alpha,
        bounds_df,
        decision_f,
        decision_t,
        decision_indep,
        decision_joint,
        cv_source,
        p_values,
        fit,
    )
