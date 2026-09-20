r"""System ARDL: multiple ECM equations linked by contemporaneous correlation (SUR).

Every other module in this library is **single-equation**: one
:math:`y`, possibly several individuals in a panel
(:mod:`pyardl.panel`), but never several dependent variables
**simultaneously linked** through their errors. This module covers two
uses distinct from what Johansen (:mod:`pyardl.cointegration.johansen`)
already gives:

- estimating two or more ARDL/ECMs jointly whose errors are correlated
  contemporaneously (money demand and credit demand in the same
  economy, say), for an efficiency gain over separate OLS regressions;
- testing **cross-equation** restrictions (a long-run coefficient
  identical across two equations), which no single-equation module can
  express since each ``ARDL`` is fit in isolation.

**Distinct from Johansen, explicitly.** Johansen estimates a **full VAR
system** where every variable is endogenous and the number of
cointegrating relations is itself an open question. This module
estimates **several already-specified single-equation ECMs** (each with
its own cointegrating relation, assumed known and tested separately by
the bounds test), linked only through the contemporaneous correlation
of their residuals — a much more restrictive system, but one that never
needs a system-wide cointegration rank chosen.

Estimation is Feasible Generalized Least Squares (Zellner 1962,
Seemingly Unrelated Regressions): fit every equation separately by
plain OLS (reusing :class:`pyardl.core.ardl.ARDL` unchanged), estimate
the residual covariance :math:`\hat\Sigma`, then refit the whole system
by GLS with :math:`\hat\Sigma \otimes I_T` as the error covariance —
iterated to convergence when ``iterate=True`` (equivalent to maximum
likelihood under normality).

References
----------
.. [1] Zellner, A. (1962). An efficient method of estimating seemingly
       unrelated regressions and tests for aggregation bias. *Journal
       of the American Statistical Association*, 57(298), 348-368.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from pyardl.core.ardl import ARDL

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    from pyardl.core.ardl import ARDLResults

    FloatArray = NDArray[np.float64]

__all__ = ["SystemARDL", "SystemARDLResults", "WaldResult"]


@dataclass(frozen=True)
class WaldResult:
    """Outcome of :meth:`SystemARDLResults.test_cross_equation_restriction`.

    Attributes
    ----------
    stat : float
        Wald chi-squared statistic.
    df : int
        Degrees of freedom, the number of restrictions (rows of ``R``).
    pvalue : float
    """

    stat: float
    df: int
    pvalue: float


@dataclass(frozen=True)
class SystemARDLResults:
    """Outcome of :class:`SystemARDL`.

    Attributes
    ----------
    equations : dict
        ``{name: ARDLResults}`` — the single-equation OLS fit for each
        equation, kept for comparison (:meth:`efficiency_gain`).
    sigma : pandas.DataFrame
        :math:`\\hat\\Sigma`, the ``M x M`` residual covariance, from
        the final FGLS iteration.
    fgls_params : dict
        ``{name: pandas.Series}`` — FGLS coefficients per equation.
    fgls_se : dict
        ``{name: pandas.Series}`` — FGLS standard errors per equation,
        from the system covariance (not equation-by-equation).
    n_iter : int
        Iterations run (1 when ``iterate=False``).
    names : dict
        ``{name: [term, ...]}`` — parameter names per equation, in the
        order they appear in the stacked system vector.
    """

    equations: dict[str, ARDLResults] = field(repr=False)
    sigma: pd.DataFrame
    fgls_params: dict[str, pd.Series]
    fgls_se: dict[str, pd.Series]
    n_iter: int
    names: dict[str, list[str]] = field(repr=False)
    _theta_stacked: FloatArray = field(repr=False)
    _cov_stacked: FloatArray = field(repr=False)
    _slices: dict[str, slice] = field(repr=False)

    def efficiency_gain(self, name: str) -> pd.Series:
        r"""Ratio of single-equation OLS standard errors to FGLS ones.

        A value above 1 means FGLS is more precise than OLS alone for
        that coefficient — the gain Zellner's method promises, larger
        when the residual correlation across equations is stronger and
        the regressors differ more between equations. Not guaranteed to
        exceed 1 in every finite sample (see the module's own limits
        note on a small ``T`` relative to ``M``); this reports what was
        measured, not what theory promises asymptotically.
        """
        if name not in self.equations:
            raise KeyError(
                f"{name!r} is not an equation; available: {list(self.equations)}."
            )
        ols_se = np.sqrt(np.diag(self.equations[name].cov_params_matrix.to_numpy()))
        ols_se_series = pd.Series(ols_se, index=self.equations[name].params.index)
        fgls_se = self.fgls_se[name]
        common = [n for n in fgls_se.index if n in ols_se_series.index]
        return (ols_se_series[common] / fgls_se[common]).rename("efficiency_gain")

    def test_cross_equation_restriction(
        self, r_matrix: ArrayLike, r: ArrayLike
    ) -> WaldResult:
        r"""Wald test of :math:`H_0: R\theta = r` on the stacked system vector.

        Parameters
        ----------
        r_matrix : array_like, shape (n_restrictions, n_total_params)
            Rows can mix coefficients from different equations — build
            them against the order in :attr:`names` (concatenated
            equation by equation, in insertion order).
        r : array_like, shape (n_restrictions,)

        Returns
        -------
        WaldResult

        Notes
        -----
        Operates on the raw stacked UECM coefficients (adjustment
        speeds, level coefficients, short-run terms) — a restriction
        like :math:`\lambda_1 = \lambda_2` or
        :math:`\gamma_{1,x} = \gamma_{2,x}` is linear in these and
        directly testable. A restriction on the **long-run** ratio
        :math:`\theta = -\gamma/\lambda` across equations would need
        the delta method extended to the stacked system, which is
        **not implemented** here — see ``docs/DEVIATIONS.md``.
        """
        R = np.asarray(r_matrix, dtype=np.float64)
        r_vec = np.asarray(r, dtype=np.float64).ravel()
        if R.ndim != 2 or R.shape[1] != self._theta_stacked.shape[0]:
            raise ValueError(
                f"r_matrix must have {self._theta_stacked.shape[0]} columns "
                f"(the stacked parameter count), got shape {R.shape}."
            )
        if R.shape[0] != r_vec.shape[0]:
            raise ValueError("r_matrix and r must have the same number of rows.")

        diff = R @ self._theta_stacked - r_vec
        cov_diff = R @ self._cov_stacked @ R.T
        stat = float(diff @ np.linalg.solve(cov_diff, diff))
        df = R.shape[0]
        pvalue = float(stats.chi2.sf(stat, df))
        return WaldResult(stat=stat, df=df, pvalue=pvalue)

    def summary(self) -> str:
        """Readable report of the system."""
        lines = [
            f"System ARDL (Zellner 1962, SUR-ECM) - {len(self.equations)} equations, "
            f"{self.n_iter} FGLS iteration(s)",
            "",
            "  Residual correlation (from Sigma_hat):",
        ]
        corr = self.sigma / np.sqrt(np.outer(np.diag(self.sigma), np.diag(self.sigma)))
        lines.append("    " + corr.round(4).to_string().replace("\n", "\n    "))
        for name, params in self.fgls_params.items():
            se = self.fgls_se[name]
            lines.append(f"\n  Equation '{name}' (FGLS):")
            for term in params.index:
                lines.append(f"    {term:<14}{params[term]: .4f}   (se {se[term]:.4f})")
        return "\n".join(lines)


def _cholesky_inverse(sigma: FloatArray) -> FloatArray:
    """C^{-1} where Sigma = C C' — a tiny (M x M) linear solve, not an X'X inversion."""
    c = np.linalg.cholesky(sigma)
    return np.linalg.solve(c, np.eye(sigma.shape[0]))


def _fgls_round(
    y_list: Sequence[FloatArray], x_padded_list: Sequence[FloatArray], sigma: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One FGLS pass: whiten by Sigma^{-1/2} across equations, then a single lstsq.

    Returns the stacked coefficient vector, its covariance
    ``(Z_white'Z_white)^{-1}``, and the per-equation residuals (stacked,
    same order as ``y_list``).
    """
    m = len(y_list)
    t = y_list[0].shape[0]
    c_inv = _cholesky_inverse(sigma)

    z_white_blocks = []
    y_white_blocks = []
    for i in range(m):
        zw = np.zeros_like(x_padded_list[0])
        yw = np.zeros_like(y_list[0])
        for j in range(m):
            zw = zw + c_inv[i, j] * x_padded_list[j]
            yw = yw + c_inv[i, j] * y_list[j]
        z_white_blocks.append(zw)
        y_white_blocks.append(yw)
    z_white = np.vstack(z_white_blocks)
    y_white = np.concatenate(y_white_blocks)

    theta_raw, _, _, _ = np.linalg.lstsq(z_white, y_white, rcond=None)
    theta: FloatArray = np.asarray(theta_raw, dtype=np.float64)
    ztz = z_white.T @ z_white
    cov: FloatArray = np.asarray(np.linalg.inv(ztz), dtype=np.float64)

    resid_stacked = np.empty((t, m), dtype=np.float64)
    for i in range(m):
        resid_stacked[:, i] = y_list[i] - x_padded_list[i] @ theta

    return theta, cov, resid_stacked


class SystemARDL:
    r"""SUR-ECM: several single-equation ARDL/ECMs, estimated jointly by FGLS.

    Parameters
    ----------
    equations : dict
        ``{name: (y, x, order)}``, one entry per equation. Every ``y``
        must share the same length and the same time index — a system
        with partially disjoint dates between equations is out of
        scope (see the module's own limits note).
    det : {'none', 'const', 'trend'}, default 'const'
        Applied to every equation.
    iterate : bool, default True
        Repeat the FGLS round until the coefficients converge
        (equivalent to maximum likelihood under normality) rather than
        stopping after one GLS pass.
    max_iter : int, default 50
    tol : float, default 1e-8
        Convergence criterion: max absolute change in the stacked
        coefficient vector between rounds.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.system import SystemARDL
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> shared = rng.standard_normal(n)
    >>> x1 = pd.Series(rng.standard_normal(n).cumsum(), name="x1")
    >>> x2 = pd.Series(rng.standard_normal(n).cumsum(), name="x2")
    >>> y1 = np.zeros(n)
    >>> y2 = np.zeros(n)
    >>> for t in range(1, n):
    ...     e1 = 0.3 * shared[t] + rng.standard_normal() * 0.2
    ...     e2 = 0.3 * shared[t] + rng.standard_normal() * 0.2
    ...     y1[t] = y1[t - 1] - 0.5 * (y1[t - 1] - 1.0 * x1.iloc[t - 1]) + e1
    ...     y2[t] = y2[t - 1] - 0.5 * (y2[t - 1] - 1.5 * x2.iloc[t - 1]) + e2
    >>> res = SystemARDL({
    ...     "eq1": (y1, pd.DataFrame({"x1": x1}), (1, 1)),
    ...     "eq2": (y2, pd.DataFrame({"x2": x2}), (1, 1)),
    ... }).fit()
    >>> sorted(res.equations)
    ['eq1', 'eq2']
    """

    def __init__(
        self,
        equations: dict[str, tuple[ArrayLike, ArrayLike, tuple[int, Any]]],
        det: Literal["none", "const", "trend"] = "const",
        iterate: bool = True,
        max_iter: int = 50,
        tol: float = 1e-8,
    ) -> None:
        if len(equations) < 2:
            raise ValueError("SystemARDL needs at least two equations.")
        self.equations_spec = equations
        self.det = det
        self.iterate = bool(iterate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self) -> SystemARDLResults:
        """Estimate every equation by OLS, then iterate FGLS.

        Returns
        -------
        SystemARDLResults
        """
        names_eq = list(self.equations_spec)
        ols_results: dict[str, ARDLResults] = {}
        models: dict[str, ARDL] = {}
        for name, (y, x, order) in self.equations_spec.items():
            models[name] = ARDL(y, x, order=order, det=self.det)

        common_hold_back = max(m.hold_back for m in models.values())
        for name, (y, x, order) in self.equations_spec.items():
            models[name] = ARDL(
                y,
                x,
                order=order,
                det=self.det,
                hold_back=common_hold_back,
            )
            ols_results[name] = models[name].fit()

        designs = {}
        y_deps = {}
        names_by_eq: dict[str, list[str]] = {}
        for name, model in models.items():
            design, y_dep, param_names = model._build_design()
            designs[name] = design
            y_deps[name] = y_dep
            names_by_eq[name] = param_names

        t_obs = y_deps[names_eq[0]].shape[0]
        for name in names_eq:
            if y_deps[name].shape[0] != t_obs:
                raise ValueError(
                    "All equations must share the same estimation sample length; "
                    f"'{name}' has {y_deps[name].shape[0]}, expected {t_obs}."
                )

        widths = [designs[name].shape[1] for name in names_eq]
        total_k = sum(widths)
        offsets = np.cumsum([0, *widths])
        slices = {
            name: slice(int(offsets[i]), int(offsets[i + 1]))
            for i, name in enumerate(names_eq)
        }

        x_padded_list = []
        y_list = []
        for name in names_eq:
            padded = np.zeros((t_obs, total_k), dtype=np.float64)
            padded[:, slices[name]] = designs[name]
            x_padded_list.append(padded)
            y_list.append(y_deps[name])

        resid0 = np.column_stack(
            [
                y_deps[name] - designs[name] @ ols_results[name]._params
                for name in names_eq
            ]
        )
        sigma = (resid0.T @ resid0) / t_obs

        theta: FloatArray = np.zeros(total_k)
        cov: FloatArray = np.eye(total_k)
        n_iter = 0
        max_rounds = self.max_iter if self.iterate else 1
        for _ in range(max_rounds):
            n_iter += 1
            theta_new, cov, resid_stacked = _fgls_round(y_list, x_padded_list, sigma)
            sigma = (resid_stacked.T @ resid_stacked) / t_obs
            if n_iter > 1 and np.max(np.abs(theta_new - theta)) < self.tol:
                theta = theta_new
                break
            theta = theta_new
            if not self.iterate:
                break

        fgls_params = {
            name: pd.Series(theta[slices[name]], index=names_by_eq[name], name="coef")
            for name in names_eq
        }
        se_all = np.sqrt(np.diag(cov))
        fgls_se = {
            name: pd.Series(se_all[slices[name]], index=names_by_eq[name], name="se")
            for name in names_eq
        }

        sigma_df = pd.DataFrame(sigma, index=names_eq, columns=names_eq)

        return SystemARDLResults(
            equations=ols_results,
            sigma=sigma_df,
            fgls_params=fgls_params,
            fgls_se=fgls_se,
            n_iter=n_iter,
            names=names_by_eq,
            _theta_stacked=theta,
            _cov_stacked=cov,
            _slices=slices,
        )
