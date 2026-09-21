r"""ARDL-GARCH: UECM mean, GARCH variance (Engle 1982, Bollerslev 1986).

Every inference tool elsewhere in this library (spec 03's standard
errors, spec 10's bounds test, spec 04's restriction tests) assumes a
homoskedastic residual variance, or at best a robust HC/HAC covariance
(spec 03 ``cov_type``) that corrects the *average* variance over a
window without modelling its dynamics. On high-frequency financial data
the conditional variance is often the object of interest in its own
right, or affects how precisely the long-run relation can be pinned
down. This module jointly estimates the conditional mean (a standard
UECM, spec 03) and the conditional variance of its residuals (GARCH),
by maximum likelihood:

.. math::
    \Delta y_t = \det + \lambda y_{t-1} + \gamma' x_{t-1} +
    \sum(\ldots) + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t,
    \quad z_t \sim \text{i.i.d.}(0, 1)

    \sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2 +
    \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2

Like specs 30/41/42/35, "ARDL-GARCH" has no single founding paper
combining both frameworks under this name — this module documents the
combination it uses: mean and variance are estimated **jointly**, not
in two separate steps (two-step estimation is biased under
GARCH-in-mean, and less efficient even without it).

**Implementation strategy.** CLAUDE.md's own architecture notes already
list `arch` as an optional dependency (lazily imported, used elsewhere
for bootstrap and as spec 27's external validation reference) — this
module keeps that same status rather than promoting it to a hard
runtime dependency, and reuses `arch.univariate.LS` (a plain linear
mean model with an attachable volatility process) as the joint
estimation engine, exactly the strategy spec 35's Markov-switching
module applies to `statsmodels`. The mean design (deterministic terms,
the error-correction term ``y.L1``, the level regressors ``x{j}.L1``,
the short-run difference terms) is built with this library's own naming
conventions and passed to `arch` with its own constant suppressed
(``constant=False``) so the columns are exactly this library's UECM
design, not `arch`'s own.

**Order convention, explicit.** The spec's own :math:`\sigma_t^2 =
\omega + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^p
\beta_j \sigma_{t-j}^2` indexes the ARCH terms by ``q`` and the GARCH
terms by ``p`` — the reverse of `arch`'s own ``GARCH(p, o, q)`` where
``p`` counts the ARCH terms and ``q`` the GARCH terms. This module's
``garch_order=(p, q)`` follows the **spec's** convention (``p`` GARCH
terms, ``q`` ARCH terms) and translates internally.

**Out of scope (v1).** GARCH-in-mean (spec 43 Section 2.4) is not
implemented: `arch`'s `LS` mean model has no feedback term from the
variance equation into the mean, and building one would need a custom
joint-likelihood optimiser outside `arch`'s public API — documented in
``docs/DEVIATIONS.md`` rather than silently dropped. Multivariate GARCH
and NARDL/QARDL combinations are out of scope per the spec itself
(Section 6).

References
----------
.. [1] Engle, R. F. (1982). Autoregressive conditional
       heteroscedasticity with estimates of the variance of United
       Kingdom inflation. *Econometrica*, 50(4), 987-1007.
.. [2] Bollerslev, T. (1986). Generalized autoregressive conditional
       heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.utils import check_series

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

Det = Literal["none", "const", "trend"]
GarchType = Literal["garch", "egarch", "gjr"]

__all__ = ["ARDLGarch", "ARDLGarchResults"]

_IGARCH_THRESHOLD = 0.98


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
class ARDLGarchResults:
    """Outcome of :class:`ARDLGarch`.

    Attributes
    ----------
    mean_params, mean_se : pandas.Series
        UECM mean-equation coefficients and standard errors, indexed by
        design term name (``const``, ``y.L1``, ``x{j}.L1``, ...).
    garch_params : pandas.Series
        ``omega``, ``alpha[i]``, ``gamma[i]`` (asymmetric term, GJR/
        EGARCH only), ``beta[j]`` — or just ``sigma2`` for the
        degenerate constant-variance case.
    conditional_variance : pandas.Series
        :math:`\\sigma_t^2` over the estimation sample.
    longrun : pandas.DataFrame
        ``theta = -gamma/lambda`` and its standard error, from the
        delta method applied to the **joint** MLE covariance matrix
        (mean and variance parameters together) — not the OLS
        covariance spec 03's ``ARDLResults.longrun`` uses, since the
        residual variance is no longer constant.
    persistence : float
        ``sum(alpha) + sum(beta)`` (``+ 0.5 * sum(gamma)`` for the
        asymmetric term, the standard GJR/EGARCH convention).
    llf : float
    names : list of str
        Mean design term names.
    """

    mean_params: pd.Series
    mean_se: pd.Series
    garch_params: pd.Series
    conditional_variance: pd.Series
    longrun: pd.DataFrame
    persistence: float
    llf: float
    names: list[str] = field(repr=False)

    def summary(self) -> str:
        """Readable report: mean equation, GARCH equation, persistence, long run."""
        lines = [
            f"ARDL-GARCH (Engle 1982, Bollerslev 1986) - llf={self.llf:.4f}, "
            f"persistence={self.persistence:.4f}",
            "",
            "  Mean equation:",
        ]
        for name in self.names:
            lines.append(
                f"    {name:<14}{self.mean_params[name]: .4f}   "
                f"(se {self.mean_se[name]:.4f})"
            )
        lines.append("\n  Variance equation:")
        for name, value in self.garch_params.items():
            lines.append(f"    {name:<14}{value: .4f}")
        lines.append("\n  Long run (theta, se):")
        lines.append("    " + self.longrun.round(4).to_string().replace("\n", "\n    "))
        return "\n".join(lines)

    def plot_volatility(self, ax: object = None):  # type: ignore[no-untyped-def]
        """Plot the conditional variance over the sample. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "plot_volatility requires matplotlib, an optional dependency. "
                "Install it with: pip install matplotlib"
            ) from exc
        if ax is None:  # pragma: no cover - trivial plumbing
            _, ax = plt.subplots(figsize=(9, 4))
        axes: object = ax
        axes.plot(  # type: ignore[attr-defined]
            self.conditional_variance.index, self.conditional_variance.to_numpy()
        )
        axes.set_ylabel("sigma_t^2")  # type: ignore[attr-defined]
        axes.set_xlabel("t")  # type: ignore[attr-defined]
        return axes


class ARDLGarch:
    r"""ARDL/UECM mean with a GARCH-family conditional variance, jointly estimated.

    Parameters
    ----------
    y, x : array_like
        As in :class:`pyardl.core.ardl.ARDL`.
    order : tuple of (int, int), default (1, 1)
        ``(p, q)`` for the mean equation, the same uniform-lag
        convention as :func:`pyardl.regularized.select_order_regularized`.
    det : {'none', 'const', 'trend'}, default 'const'
    garch_order : tuple of (int, int), default (1, 1)
        ``(p, q)`` in the **spec's** convention: ``p`` GARCH
        (lagged-variance) terms, ``q`` ARCH (lagged squared-residual)
        terms — the reverse of `arch`'s own ``GARCH(p, o, q)`` argument
        order (translated internally). ``(0, 0)`` fits a constant
        (homoskedastic) variance instead — the degenerate case used to
        lock this module against plain :class:`pyardl.core.ardl.ARDL`.
    garch_type : {'garch', 'egarch', 'gjr'}, default 'garch'
        ``'gjr'`` and ``'egarch'`` add an asymmetric term (the leverage
        effect: volatility reacts more to negative shocks) — not to be
        confused with NARDL's (spec 17) asymmetry, which is in the
        mean, not the variance.
    in_mean : bool, default False
        GARCH-in-mean feedback. **Not implemented** — raises
        ``NotImplementedError`` if ``True`` (spec 43 Section 2.4, see
        ``docs/DEVIATIONS.md``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.volatility import ARDLGarch
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x = pd.Series(rng.standard_normal(n).cumsum(), name="x")
    >>> y = np.zeros(n)
    >>> sigma2 = 0.04
    >>> for t in range(1, n):
    ...     y[t] = y[t - 1] - 0.5 * (y[t - 1] - 1.2 * x.iloc[t - 1])
    ...     e = rng.standard_normal() * np.sqrt(sigma2)
    ...     sigma2 = 0.01 + 0.1 * e**2 + 0.8 * sigma2
    ...     y[t] += e
    >>> res = ARDLGarch(y, pd.DataFrame({"x": x}), order=(1, 1)).fit()
    >>> res.persistence < 1
    True
    """

    def __init__(
        self,
        y: ArrayLike,
        x: ArrayLike,
        order: tuple[int, int] = (1, 1),
        det: Det = "const",
        garch_order: tuple[int, int] = (1, 1),
        garch_type: GarchType = "garch",
        in_mean: bool = False,
    ) -> None:
        if in_mean:
            raise NotImplementedError(
                "GARCH-in-mean (in_mean=True) is not implemented in this version "
                "(spec 43 Section 2.4) -- see docs/DEVIATIONS.md."
            )
        if garch_type not in ("garch", "egarch", "gjr"):
            raise ValueError(
                f"garch_type={garch_type!r} must be 'garch', 'egarch' or 'gjr'."
            )
        p, q = order
        if p < 1:
            raise ValueError(
                "order[0] (p) must be >= 1: no error-correction term otherwise."
            )
        gp, gq = garch_order
        if gp < 0 or gq < 0:
            raise ValueError("garch_order entries must be >= 0.")
        if (gp == 0) != (gq == 0):
            raise ValueError("garch_order must be (0, 0) or both entries >= 1.")

        self.y = y
        self.x = x
        self.p = int(p)
        self.q = int(q)
        self.det: Det = det
        self.garch_p = int(gp)
        self.garch_q = int(gq)
        self.garch_type: GarchType = garch_type

    def fit(self) -> ARDLGarchResults:
        """Jointly maximise the Gaussian conditional likelihood.

        Returns
        -------
        ARDLGarchResults
        """
        try:
            from arch.univariate import EGARCH, GARCH, LS, ConstantVariance
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ARDLGarch requires the 'arch' package, an optional dependency. "
                "Install it with: pip install arch"
            ) from exc

        y_arr, x_arr, index, _, x_names = check_series(self.y, self.x)
        if x_arr is None:
            raise ValueError("ARDLGarch needs at least one regressor.")

        design, target, names = _build_design(y_arr, x_arr, self.p, self.q, self.det)
        exog_df = pd.DataFrame(design, columns=names)

        mod = LS(target, x=exog_df, constant=False)
        if self.garch_p == 0 and self.garch_q == 0:
            mod.volatility = ConstantVariance()
        elif self.garch_type == "garch":
            mod.volatility = GARCH(p=self.garch_q, o=0, q=self.garch_p)
        elif self.garch_type == "gjr":
            mod.volatility = GARCH(p=self.garch_q, o=1, q=self.garch_p)
        else:
            mod.volatility = EGARCH(p=self.garch_q, o=1, q=self.garch_p)

        res = mod.fit(disp="off")

        mean_params = res.params[names]
        mean_se = res.std_err[names]
        garch_names = [n for n in res.params.index if n not in names]
        garch_params = res.params[garch_names]

        alpha_sum = sum(v for n, v in garch_params.items() if n.startswith("alpha"))
        beta_sum = sum(v for n, v in garch_params.items() if n.startswith("beta"))
        gamma_sum = sum(v for n, v in garch_params.items() if n.startswith("gamma"))
        persistence = float(alpha_sum + beta_sum + 0.5 * gamma_sum)
        if persistence > _IGARCH_THRESHOLD:
            warnings.warn(
                f"persistence={persistence:.4f} is close to or above 1: "
                "near-integrated variance (IGARCH), a frequent edge case in "
                "financial data -- shocks to volatility barely decay "
                "(spec 43 Section 4).",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

        conditional_variance = pd.Series(
            np.asarray(res.conditional_volatility, dtype=np.float64) ** 2,
            index=index[-target.shape[0] :] if index is not None else None,
            name="conditional_variance",
        )

        full_cov = np.asarray(res.param_cov, dtype=np.float64)
        param_index = list(res.params.index)
        lambda_pos = param_index.index("y.L1")

        longrun_rows = []
        for j, xname in enumerate(x_names):
            gamma_name = f"x{j}.L1"
            gamma_pos = param_index.index(gamma_name)
            lam = mean_params["y.L1"]
            gamma = mean_params[gamma_name]
            theta = -gamma / lam
            grad = np.zeros(full_cov.shape[0])
            grad[lambda_pos] = gamma / lam**2
            grad[gamma_pos] = -1.0 / lam
            var_theta = float(grad @ full_cov @ grad)
            longrun_rows.append(
                {"regressor": xname, "theta": theta, "se": np.sqrt(max(var_theta, 0.0))}
            )
        longrun = pd.DataFrame(longrun_rows).set_index("regressor")

        return ARDLGarchResults(
            mean_params=mean_params,
            mean_se=mean_se,
            garch_params=garch_params,
            conditional_variance=conditional_variance,
            longrun=longrun,
            persistence=persistence,
            llf=float(res.loglikelihood),
            names=names,
        )
