r"""Pooled Mean Group and Dynamic Fixed Effects (spec 23).

Between the two extremes of spec 22 — pool everything (inconsistent
under heterogeneity) or pool nothing (consistent but noisy) — Pesaran,
Shin and Smith (1999) put a middle term that turns out to be the one
applied work actually wants: **constrain the long-run coefficients to be
equal across individuals, leave every short-run dynamic free.**

.. math::

    \Delta y_{it} = \lambda_i (y_{i,t-1} - \theta' x_{i,t-1})
      + \sum_{s=1}^{p-1} \psi_{is} \Delta y_{i,t-s}
      + \sum_{s=0}^{q-1} \omega_{is}' \Delta x_{i,t-s}
      + \mu_i + \varepsilon_{it}

The economics behind the restriction is that long-run relations often
come from theory — a budget constraint, an arbitrage condition, a
technology — which applies to everyone, while the speed at which each
country returns to it plainly does not.

Estimation: concentrate, then alternate
---------------------------------------
The likelihood has :math:`k` common parameters and :math:`4N` individual
ones, but it concentrates beautifully. **Given** :math:`\theta`, define
the error-correction term

.. math:: \xi_{it}(\theta) = y_{i,t-1} - \theta' x_{i,t-1}

and every individual block is an ordinary least-squares regression of
:math:`\Delta y_i` on :math:`[\xi_i(\theta), \Delta W_i]`. **Given** the
dynamics, :math:`\theta` solves one stacked weighted least-squares
problem, each individual weighted by :math:`\lambda_i / \sigma_i`.

Alternating the two is the back-fitting algorithm, and it is what
``xtpmg`` does. A quasi-Newton maximiser of the same concentrated
likelihood is available as ``method='newton'``; the two must agree, and
a test checks that they do.

The concentrated log-likelihood is worth writing down because it is the
object that settles disputes:

.. math::

    \ell(\theta) = -\tfrac{1}{2} \sum_i T_i
      \left[ \log 2\pi + 1 + \log \hat\sigma_i^2(\theta) \right]

Maximising it is minimising :math:`\sum_i T_i \log \hat\sigma_i^2`. When
two implementations disagree about :math:`\hat\theta`, this number says
which one is further from the optimum — see OBS-21, where it caught a
reference implementation stopping short of its own maximum.

One assumption is doing real work here
--------------------------------------
The likelihood is a product over individuals, which assumes they are
**independent of each other**. Common shocks — a world cycle, a
commodity price — violate it, and then both MG and PMG are biased. That
is not a caveat to bury, and nothing in this module corrects for it yet,
so :meth:`PMGResults.summary` says so out loud rather than leaving the
reader to assume it was handled.

References
----------
.. [1] Pesaran, M. H., Shin, Y. & Smith, R. P. (1999). Pooled mean group
       estimation of dynamic heterogeneous panels. *Journal of the
       American Statistical Association*, 94(446), 621-634.
.. [2] Blackburne, E. F. & Frank, M. W. (2007). Estimation of nonstationary
       heterogeneous panels. *The Stata Journal*, 7(2), 197-208.
.. [3] Hausman, J. A. (1978). Specification tests in econometrics.
       *Econometrica*, 46(6), 1251-1271.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import stats

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.panel.container import PanelData, panel_from_frame
from pyardl.panel.mg import MeanGroup, MeanGroupResults

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]

__all__ = ["DFE", "PMG", "DFEResults", "HausmanResult", "PMGResults", "compare"]

DetType = Literal["none", "const", "trend"]
Method = Literal["backfitting", "newton"]


# ----------------------------------------------------------------------
# Design construction, shared by PMG and DFE.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Block:
    """One individual's error-correction design.

    ``dy`` is the target; ``y_lag`` and ``x_lag`` build the level term;
    ``w`` holds the short-run regressors and the deterministic terms.
    """

    key: object
    dy: FloatArray
    y_lag: FloatArray
    x_lag: FloatArray
    w: FloatArray
    w_names: tuple[str, ...]

    @property
    def nobs(self) -> int:
        return int(self.dy.size)


def _build_blocks(panel: PanelData, p: int, q: int, det: DetType) -> list[_Block]:
    """Error-correction design per individual, on its own common sample.

    All lags are taken **within** an individual. Building them on the
    stacked panel would splice the end of one individual's history onto
    the start of the next, which is both wrong and invisible.
    """
    blocks: list[_Block] = []
    for unit in panel:
        y = unit.y.to_numpy()
        x = unit.x.to_numpy()
        n = y.size
        start = max(p, q)
        if n - start < 2:
            continue
        dy_full = np.diff(y)
        dx_full = np.diff(x, axis=0)
        # Row t of the design corresponds to observation index t in the
        # original series, for t = start .. n-1.
        idx = np.arange(start, n)
        dy = dy_full[idx - 1]
        y_lag = y[idx - 1]
        x_lag = x[idx - 1]
        cols: list[FloatArray] = []
        names: list[str] = []
        if det in ("const", "trend"):
            cols.append(np.ones(idx.size))
            names.append("const")
        if det == "trend":
            cols.append(idx.astype(np.float64))
            names.append("trend")
        for s in range(1, p):
            cols.append(dy_full[idx - 1 - s])
            names.append(f"D.{unit.y.name}.L{s}")
        for j, xname in enumerate(panel.x_names):
            for s in range(q):
                cols.append(dx_full[idx - 1 - s, j])
                names.append(f"D.{xname}.L{s}" if s else f"D.{xname}")
        w = np.column_stack(cols) if cols else np.empty((idx.size, 0))
        blocks.append(
            _Block(
                key=unit.key,
                dy=dy,
                y_lag=y_lag,
                x_lag=x_lag,
                w=w,
                w_names=tuple(names),
            )
        )
    return blocks


def _individual_step(
    blocks: list[_Block], theta: FloatArray
) -> tuple[FloatArray, list[FloatArray], FloatArray, FloatArray]:
    """Given theta, one OLS per individual. Returns lambda, gamma, sigma2, ssr."""
    lam = np.empty(len(blocks))
    gam: list[FloatArray] = []
    sigma2 = np.empty(len(blocks))
    ssr = np.empty(len(blocks))
    for i, b in enumerate(blocks):
        xi = b.y_lag - b.x_lag @ theta
        design = np.column_stack([xi, b.w])
        beta, *_ = np.linalg.lstsq(design, b.dy, rcond=None)
        resid = b.dy - design @ beta
        lam[i] = beta[0]
        gam.append(beta[1:])
        ssr[i] = float(resid @ resid)
        # Maximum-likelihood normalisation: the concentrated likelihood
        # divides by T_i, not by the residual degrees of freedom. Using
        # the unbiased variance here would change both the weights and
        # the reported log-likelihood.
        sigma2[i] = ssr[i] / b.nobs
    return lam, gam, sigma2, ssr


def _theta_step(
    blocks: list[_Block],
    lam: FloatArray,
    gam: list[FloatArray],
    sigma2: FloatArray,
) -> FloatArray:
    """Given the dynamics, update theta by stacked weighted least squares.

    From ``dy_i = W_i g_i + lam_i (y_lag_i - X_lag_i theta) + e_i``,

        ``u_i = dy_i - W_i g_i - lam_i y_lag_i = -lam_i X_lag_i theta + e_i``

    so theta solves a least-squares problem with design
    ``(lam_i / sigma_i) X_lag_i`` and target ``-u_i / sigma_i``, stacked
    over individuals. Solved by ``lstsq`` on the stack rather than by
    forming and inverting the normal equations.
    """
    rows: list[FloatArray] = []
    targets: list[FloatArray] = []
    for b, li, gi, si in zip(blocks, lam, gam, np.sqrt(sigma2), strict=True):
        u = b.dy - b.w @ gi - li * b.y_lag
        rows.append((li / si) * b.x_lag)
        targets.append(-u / si)
    theta, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(targets), rcond=None)
    return np.asarray(theta, dtype=np.float64)


def _concentrated_loglik(blocks: list[_Block], theta: FloatArray) -> float:
    """Log-likelihood with the individual blocks profiled out."""
    _, _, sigma2, _ = _individual_step(blocks, theta)
    nobs = np.array([b.nobs for b in blocks], dtype=np.float64)
    with np.errstate(divide="ignore"):
        return float(-0.5 * np.sum(nobs * (np.log(2 * np.pi) + 1 + np.log(sigma2))))


def _theta_covariance(
    blocks: list[_Block],
    theta: FloatArray,
    lam: FloatArray,
    sigma2: FloatArray,
) -> FloatArray:
    r"""Asymptotic covariance of theta_hat: the expected profile information.

    The mean function of individual :math:`i` is

    .. math:: m_i = \lambda_i (y_{i,-1} - X_{i,-1}\theta) + W_i \gamma_i

    so its derivatives are :math:`\partial m_i/\partial\theta =
    -\lambda_i X_{i,-1}`, :math:`\partial m_i/\partial\lambda_i =
    \xi_i(\theta)` and :math:`\partial m_i/\partial\gamma_i = W_i`. The
    information matrix is block-arrow — one shared :math:`\theta` block,
    one block per individual — and the covariance of
    :math:`\hat\theta` is its Schur complement:

    .. math::

        V(\hat\theta) = \Big[ \sum_i \frac{\lambda_i^2}{\sigma_i^2}
            X_{i,-1}' M_{[\xi_i, W_i]} X_{i,-1} \Big]^{-1}

    **The projection must annihilate** :math:`\xi_i` **as well as**
    :math:`W_i`. Both :math:`\lambda_i` and :math:`\gamma_i` are
    estimated, so both derivative directions have to be swept out.
    Removing only :math:`W_i` overstates the information and returns a
    standard error about 5% too small on the reference panel. That was
    the first version of this function, and it was caught by comparing
    against the numerical Hessian of :func:`_concentrated_loglik`, not
    by any test of internal consistency. The suite keeps that
    comparison.
    """
    k = theta.size
    info = np.zeros((k, k))
    for b, li, s2 in zip(blocks, lam, sigma2, strict=True):
        xl: FloatArray = b.x_lag
        xi = (b.y_lag - b.x_lag @ theta)[:, None]
        nuisance = np.column_stack([xi, b.w]) if b.w.shape[1] else xi
        coef, *_ = np.linalg.lstsq(nuisance, xl, rcond=None)
        xl = np.asarray(xl - nuisance @ coef, dtype=np.float64)
        info += (li**2 / s2) * (xl.T @ xl)
    return np.asarray(np.linalg.pinv(info), dtype=np.float64)


def _observed_theta_covariance(
    blocks: list[_Block], theta: FloatArray, step: float = 1e-5
) -> FloatArray:
    """Covariance from the OBSERVED profile information.

    The numerical Hessian of :func:`_concentrated_loglik` by central
    differences. Asymptotically equivalent to :func:`_theta_covariance`
    and about 2% larger on the reference panel — the finite-sample gap
    between observed and expected information.

    Reachable as ``vcov='observed'``. The default is the expected form
    because that is the estimator Pesaran, Shin and Smith published and
    the one ``xtpmg`` implements, so it is what makes pyardl comparable
    to the reference implementations. It is **not** the better-covering
    of the two: measured at N=25, T=60, the observed form covers closer
    to nominal. Both under-cover; see the panel documentation for the
    table.
    """
    k = theta.size
    hess = np.empty((k, k))
    base = _concentrated_loglik(blocks, theta)
    for i in range(k):
        for j in range(i, k):
            ei: FloatArray = np.zeros(k, dtype=np.float64)
            ej: FloatArray = np.zeros(k, dtype=np.float64)
            ei[i] = step
            ej[j] = step
            plus_i: FloatArray = np.asarray(theta + ei, dtype=np.float64)
            minus_i: FloatArray = np.asarray(theta - ei, dtype=np.float64)
            if i == j:
                value = (
                    _concentrated_loglik(blocks, plus_i)
                    - 2 * base
                    + _concentrated_loglik(blocks, minus_i)
                ) / step**2
            else:
                f_pp = _concentrated_loglik(
                    blocks, np.asarray(theta + ei + ej, dtype=np.float64)
                )
                f_pm = _concentrated_loglik(
                    blocks, np.asarray(theta + ei - ej, dtype=np.float64)
                )
                f_mp = _concentrated_loglik(
                    blocks, np.asarray(theta - ei + ej, dtype=np.float64)
                )
                f_mm = _concentrated_loglik(
                    blocks, np.asarray(theta - ei - ej, dtype=np.float64)
                )
                value = (f_pp - f_pm - f_mp + f_mm) / (4 * step**2)
            hess[i, j] = hess[j, i] = value
    return np.asarray(np.linalg.pinv(-hess), dtype=np.float64)


@dataclass(frozen=True)
class HausmanResult:
    """Outcome of the MG-versus-PMG specification test.

    Attributes
    ----------
    statistic : float
        The Hausman statistic.
    dof : int
        Degrees of freedom — the **rank** of the inverted variance
        difference, which is ``k`` when that difference is positive
        definite and smaller when a pseudo-inverse was needed.
    pvalue : float
        Right-tail probability under the chi-squared null.
    used_pseudo_inverse : bool
        Whether the variance difference failed to be positive definite.
        This happens often in practice and is not a bug: the difference
        is only guaranteed positive definite asymptotically, and in
        finite samples it need not be.
    """

    statistic: float
    dof: int
    pvalue: float
    used_pseudo_inverse: bool
    diff: pd.Series = field(repr=False)

    @property
    def decision(self) -> str:
        """Verdict at 5%, in words rather than a bare boolean."""
        if self.pvalue < 0.05:
            return "reject homogeneity: prefer MG"
        return "do not reject homogeneity: PMG is consistent and efficient"

    def summary(self) -> str:
        lines = [
            "Hausman test, MG versus PMG (Pesaran, Shin & Smith 1999)",
            "  H0: the long-run coefficients are common across individuals",
            "      (PMG consistent AND efficient; MG consistent but noisy)",
            "  H1: they are not (only MG is consistent)",
            "",
            f"  chi2({self.dof}) = {self.statistic:.4f}   p = {self.pvalue:.4f}",
            f"  {self.decision}",
        ]
        if self.used_pseudo_inverse:
            lines.append(
                "  NOTE the variance difference was not positive definite, so a "
                "pseudo-inverse was used and the degrees of freedom are its "
                "rank. This is common in finite samples; treat the p-value as "
                "indicative rather than exact."
            )
        return "\n".join(lines)


def hausman(mg: MeanGroupResults, pmg: PMGResults) -> HausmanResult:
    """Test long-run homogeneity by comparing MG and PMG.

    Under the null both estimators are consistent but PMG is efficient,
    so their difference has variance :math:`V_{MG} - V_{PMG}`. Under the
    alternative only MG is consistent and the difference grows.

    Parameters
    ----------
    mg : MeanGroupResults
    pmg : PMGResults
        Both fitted on the same panel and the same regressors.

    Returns
    -------
    HausmanResult

    Raises
    ------
    ValueError
        If the two fits do not share the same regressors — comparing
        coefficients on different variables would produce a number with
        no meaning.

    Notes
    -----
    The variance difference is frequently **not** positive definite in
    finite samples. Rather than fail, or silently return a negative
    statistic, a pseudo-inverse is used and the fact is recorded on the
    result and printed in ``summary()``, as Stata's implementation does.
    """
    names = list(pmg.longrun.index)
    if list(mg.longrun.index) != names:
        raise ValueError(
            f"MG and PMG were fitted on different regressors: "
            f"{list(mg.longrun.index)} vs {names}."
        )
    diff = mg.longrun["theta"].to_numpy() - pmg.longrun["theta"].to_numpy()
    v_mg = np.diag(mg.longrun["se"].to_numpy() ** 2)
    v_pmg = np.asarray(pmg.cov_theta, dtype=np.float64)
    delta = v_mg - v_pmg

    eigenvalues = np.linalg.eigvalsh((delta + delta.T) / 2.0)
    positive_definite = bool(np.all(eigenvalues > 1e-12 * max(1.0, eigenvalues.max())))
    if positive_definite:
        inv = np.linalg.pinv(delta)
        dof = delta.shape[0]
    else:
        inv = np.linalg.pinv(delta)
        dof = int(np.linalg.matrix_rank(delta))
        warnings.warn(
            "The Hausman variance difference V(MG) - V(PMG) is not positive "
            "definite, so a pseudo-inverse was used and the degrees of "
            "freedom are its rank. The difference is only guaranteed "
            "positive definite asymptotically; in finite samples this is "
            "common. Read the p-value as indicative.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )
    stat = float(diff @ inv @ diff)
    # A negative statistic is possible once a pseudo-inverse is in play.
    # Reporting it as-is, with p = 1, is more honest than clipping it to
    # zero and pretending the test ran cleanly.
    dof = max(dof, 1)
    pvalue = float(stats.chi2.sf(stat, dof)) if stat > 0 else 1.0
    return HausmanResult(
        statistic=stat,
        dof=dof,
        pvalue=pvalue,
        used_pseudo_inverse=not positive_definite,
        diff=pd.Series(diff, index=pd.Index(names, name="regressor"), name="mg-pmg"),
    )


@dataclass(frozen=True)
class PMGResults:
    """Pooled Mean Group estimates."""

    longrun: pd.DataFrame
    cov_theta: FloatArray = field(repr=False)
    adjustment: pd.Series
    lambda_i: pd.Series
    shortrun: pd.DataFrame
    sigma2_i: pd.Series
    loglik: float
    vcov_kind: str
    n_iter: int
    converged: bool
    iterations: pd.DataFrame = field(repr=False)
    panel: PanelData = field(repr=False)
    method: str = "backfitting"

    @property
    def n_units(self) -> int:
        return int(self.lambda_i.size)

    @property
    def nobs(self) -> int:
        return int(self.shortrun["nobs"].sum())

    @property
    def non_adjusting(self) -> pd.Index:
        """Individuals whose adjustment speed is not negative."""
        return self.lambda_i.index[self.lambda_i >= 0]

    def hausman_vs_mg(self, mg: MeanGroupResults) -> HausmanResult:
        """Convenience wrapper over :func:`hausman`."""
        return hausman(mg, self)

    def summary(self) -> str:
        lines = [
            f"Pooled Mean Group (Pesaran, Shin & Smith 1999) - "
            f"{self.n_units} individuals, {self.nobs} observations",
            f"  method: {self.method}, "
            f"{'converged' if self.converged else 'DID NOT CONVERGE'} "
            f"in {self.n_iter} iterations",
            f"  log-likelihood: {self.loglik:.6f}",
            "  long-run coefficients POOLED; short-run dynamics free",
            "",
            "  Long-run coefficients (common)",
            f"    {'':<12}{'theta':>12}{'se':>12}{'z':>10}{'p':>10}",
        ]
        for name, row in self.longrun.iterrows():
            lines.append(
                f"    {str(name):<12}{row['theta']:>12.4f}{row['se']:>12.4f}"
                f"{row['z']:>10.3f}{row['pvalue']:>10.4f}"
            )
        lines += [
            "",
            f"  Mean adjustment speed: {self.adjustment['lambda']:.4f} "
            f"(se {self.adjustment['se']:.4f}, between-individual)",
        ]
        n_bad = len(self.non_adjusting)
        if n_bad:
            lines.append(
                f"  WARNING {n_bad} of {self.n_units} individuals have "
                f"lambda_i >= 0: they do not error-correct."
            )
        if not self.converged:
            lines.append(
                "  WARNING the iteration did not converge; the estimates below "
                "are wherever it stopped, not a maximum."
            )
        lines += [
            "",
            "  Assumes individuals are independent of each other. Common "
            "shocks (a world cycle, a commodity price) break it and bias "
            "both PMG and MG. Nothing in this module corrects for that "
            "yet; a CD test on the residuals is the usual way to find "
            "out whether it bites.",
        ]
        return "\n".join(lines)


class PMG:
    """Pooled Mean Group estimator.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel.
    y, X, id, time : str / sequence of str
        Column names, as in :class:`~pyardl.panel.MeanGroup`.
    order : tuple
        ``(p, q)``, common to every individual. Unlike MG there is no
        per-individual selection mode: the long-run coefficients are
        pooled, so the specification that produces them has to be the
        same everywhere for the restriction to mean anything.
    det : {'none', 'const', 'trend'}
        Deterministic terms in each individual equation.
    vcov : {'expected', 'observed'}
        Which profile information to invert. ``'expected'`` is the
        analytic Schur complement of Pesaran, Shin and Smith, which is
        what ``xtpmg`` computes. ``'observed'`` is the numerical Hessian
        of the concentrated likelihood. They are asymptotically
        equivalent and differ by a couple of percent in finite samples.
    method : {'backfitting', 'newton'}
        ``'backfitting'`` alternates the two closed-form steps, as
        ``xtpmg`` does. ``'newton'`` maximises the same concentrated
        likelihood with a quasi-Newton method. They must agree; a test
        checks it.
    tol : float
        Convergence tolerance on the largest change in ``theta``.
    max_iter : int
        Iteration cap.
    min_obs : int
        Individuals with fewer usable rows are excluded.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(10):
    ...     lam = -0.5 + 0.05 * rng.normal()
    ...     x = np.cumsum(rng.normal(size=50))
    ...     y = np.zeros(50)
    ...     for t in range(1, 50):
    ...         y[t] = y[t-1] + lam * (y[t-1] - 0.8 * x[t-1]) + rng.normal(scale=.3)
    ...     rows.append(pd.DataFrame({"id": i, "t": np.arange(50), "y": y, "x": x}))
    >>> res = PMG(pd.concat(rows, ignore_index=True), y="y", X=["x"],
    ...           id="id", time="t", order=(1, 1)).fit()
    >>> res.converged
    True
    >>> bool(abs(res.longrun.loc["x", "theta"] - 0.8) < 0.1)
    True
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        X: Sequence[str],
        id: str,  # noqa: A002 - matches the spec's public API
        time: str,
        order: tuple[int, int] = (1, 1),
        det: DetType = "const",
        method: Method = "backfitting",
        vcov: Literal["expected", "observed"] = "expected",
        tol: float = 1e-8,
        max_iter: int = 200,
        min_obs: int = 15,
    ) -> None:
        if method not in ("backfitting", "newton"):
            raise ValueError(
                f"method must be 'backfitting' or 'newton', got {method!r}."
            )
        if vcov not in ("expected", "observed"):
            raise ValueError(f"vcov must be 'expected' or 'observed', got {vcov!r}.")
        if det not in ("none", "const", "trend"):
            raise ValueError(f"det must be 'none', 'const' or 'trend', got {det!r}.")
        p, q = int(order[0]), int(order[1])
        if p < 1:
            raise ValueError(
                f"p must be at least 1 for an error-correction representation, got {p}."
            )
        if q < 1:
            raise ValueError(f"q must be at least 1, got {q}.")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}.")

        self.df = df
        self.y_name = y
        self.x_names = tuple(str(c) for c in X)
        self.id_col = id
        self.time_col = time
        self.p, self.q = p, q
        self.det: DetType = det
        self.method: Method = method
        self.vcov = vcov
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.min_obs = int(min_obs)
        self.panel = panel_from_frame(
            df, y=y, x=list(X), id_col=id, time_col=time, min_obs=min_obs
        )

    def _start(self, blocks: list[_Block]) -> FloatArray:
        """Start from theta_MG, as the specification prescribes.

        The concentrated likelihood is not globally concave in theta, so
        the starting point matters. theta_MG is consistent under both
        the null and the alternative, which makes it the natural place
        to begin: the iteration only has to travel the efficiency gap,
        never the whole space.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mg = MeanGroup(
                    self.df,
                    y=self.y_name,
                    X=list(self.x_names),
                    id=self.id_col,
                    time=self.time_col,
                    order=(self.p, self.q),
                    det=self.det,
                    min_obs=self.min_obs,
                ).fit()
            theta0 = mg.longrun["theta"].to_numpy()
            if np.all(np.isfinite(theta0)):
                return np.asarray(theta0, dtype=np.float64)
        except (ValueError, np.linalg.LinAlgError):
            pass
        # Fallback: pooled static regression of y on x, individual means
        # removed. Crude, but finite and in the right neighbourhood.
        cols, target = [], []
        for b in blocks:
            xl = b.x_lag - b.x_lag.mean(axis=0)
            cols.append(xl)
            target.append(b.y_lag - b.y_lag.mean())
        theta0, *_ = np.linalg.lstsq(
            np.vstack(cols), np.concatenate(target), rcond=None
        )
        return np.asarray(theta0, dtype=np.float64)

    def fit(self) -> PMGResults:
        """Estimate the pooled long-run coefficients.

        Returns
        -------
        PMGResults

        Raises
        ------
        ValueError
            If fewer than two individuals survive, or if the design of
            an individual is singular.
        """
        blocks = _build_blocks(self.panel, self.p, self.q, self.det)
        if len(blocks) < 2:
            raise ValueError(
                f"Only {len(blocks)} individual(s) have enough observations for "
                f"an ARDL({self.p},{self.q}); pooling needs at least two."
            )
        theta = self._start(blocks)
        log: list[dict[str, float]] = []
        converged = False
        n_iter = 0

        if self.method == "backfitting":
            for n_iter in range(1, self.max_iter + 1):
                lam, gam, sigma2, _ = _individual_step(blocks, theta)
                new = _theta_step(blocks, lam, gam, sigma2)
                delta = float(np.max(np.abs(new - theta)))
                theta = new
                log.append(
                    {
                        "iter": n_iter,
                        "delta": delta,
                        "loglik": _concentrated_loglik(blocks, theta),
                    }
                )
                if delta < self.tol:
                    converged = True
                    break
        else:
            from scipy.optimize import minimize

            def negative_loglik(t: FloatArray) -> float:
                return -_concentrated_loglik(blocks, np.asarray(t, dtype=np.float64))

            out = minimize(
                negative_loglik,
                theta,
                method="BFGS",
                options={"gtol": self.tol, "maxiter": self.max_iter},
            )
            theta = np.asarray(out.x, dtype=np.float64)
            converged = bool(out.success)
            n_iter = int(out.nit)
            log.append(
                {"iter": n_iter, "delta": float("nan"), "loglik": -float(out.fun)}
            )

        if not converged:
            warnings.warn(
                f"PMG did not converge in {self.max_iter} iterations "
                f"(method={self.method}). The estimates are wherever the "
                "iteration stopped, not a maximum of the likelihood. Raise "
                "max_iter, or check whether some individuals fail to "
                "error-correct.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

        lam, gam, sigma2, ssr = _individual_step(blocks, theta)
        cov = (
            _theta_covariance(blocks, theta, lam, sigma2)
            if self.vcov == "expected"
            else _observed_theta_covariance(blocks, theta)
        )
        se = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, theta / se, np.nan)
        crit = float(stats.norm.ppf(0.975))
        longrun = pd.DataFrame(
            {
                "theta": theta,
                "se": se,
                "z": z,
                "pvalue": 2.0 * stats.norm.sf(np.abs(z)),
                "ci_lower": theta - crit * se,
                "ci_upper": theta + crit * se,
            },
            index=pd.Index(list(self.x_names), name="regressor"),
        )

        keys = pd.Index([b.key for b in blocks], name="id")
        lambda_i = pd.Series(lam, index=keys, name="lambda")
        n = lam.size
        # The pooled theta has a likelihood-based standard error, but the
        # MEAN of the lambda_i is a group average like any other, so its
        # dispersion is the between-individual one (spec 22).
        lam_se = float(np.sqrt(np.sum((lam - lam.mean()) ** 2) / (n * (n - 1))))
        adjustment = pd.Series(
            {
                "lambda": float(lam.mean()),
                "se": lam_se,
                "share_non_adjusting": float((lam >= 0).mean()),
            },
            name="adjustment",
        )
        shortrun = pd.DataFrame(
            {
                "lambda": lam,
                "sigma2": sigma2,
                "ssr": ssr,
                "nobs": [b.nobs for b in blocks],
                **{
                    name: [g[j] for g in gam]
                    for j, name in enumerate(blocks[0].w_names)
                },
            },
            index=keys,
        )
        if int((lam >= 0).sum()):
            warnings.warn(
                f"{int((lam >= 0).sum())} of {n} individuals have lambda_i >= 0: "
                "they do not error-correct, so the pooled long-run "
                "coefficient is being identified partly off individuals that "
                "never return to it.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

        return PMGResults(
            longrun=longrun,
            cov_theta=cov,
            adjustment=adjustment,
            lambda_i=lambda_i,
            shortrun=shortrun,
            sigma2_i=pd.Series(sigma2, index=keys, name="sigma2"),
            loglik=_concentrated_loglik(blocks, theta),
            vcov_kind=self.vcov,
            n_iter=n_iter,
            converged=converged,
            iterations=pd.DataFrame(log),
            panel=self.panel,
            method=self.method,
        )


@dataclass(frozen=True)
class DFEResults:
    """Dynamic fixed effects estimates."""

    longrun: pd.DataFrame
    adjustment: pd.Series
    params: pd.Series
    bse: pd.Series
    loglik: float
    nobs: int
    n_units: int
    panel: PanelData = field(repr=False)

    def summary(self) -> str:
        lines = [
            f"Dynamic fixed effects - {self.n_units} individuals, "
            f"{self.nobs} observations",
            "  EVERY coefficient pooled except the intercepts",
            "",
            f"    {'':<12}{'theta':>12}{'se':>12}",
        ]
        for name, row in self.longrun.iterrows():
            lines.append(f"    {str(name):<12}{row['theta']:>12.4f}{row['se']:>12.4f}")
        lines += [
            "",
            f"  Adjustment speed: {self.adjustment['lambda']:.4f} "
            f"(se {self.adjustment['se']:.4f})",
            "",
            "  Consistent ONLY under slope homogeneity. Under heterogeneous "
            "dynamics it is inconsistent even as N and T both grow, and its "
            "bias does not shrink with T - see the panel documentation for "
            "the measurement. Reported for the standard comparison table, "
            "not as a recommendation.",
        ]
        return "\n".join(lines)


class DFE:
    """Dynamic fixed effects: everything pooled but the intercepts.

    Included for the MG/PMG/DFE table that panel papers report, and
    because seeing its bias next to the others is the clearest argument
    for not using it. Estimated by the within transformation, which
    absorbs the individual intercepts exactly.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        X: Sequence[str],
        id: str,  # noqa: A002
        time: str,
        order: tuple[int, int] = (1, 1),
        det: DetType = "const",
        min_obs: int = 15,
    ) -> None:
        if det not in ("none", "const", "trend"):
            raise ValueError(f"det must be 'none', 'const' or 'trend', got {det!r}.")
        self.p, self.q = int(order[0]), int(order[1])
        if self.p < 1 or self.q < 1:
            raise ValueError(f"order must be at least (1, 1), got {order}.")
        self.x_names = tuple(str(c) for c in X)
        self.det: DetType = det
        self.panel = panel_from_frame(
            df, y=y, x=list(X), id_col=id, time_col=time, min_obs=min_obs
        )

    def fit(self) -> DFEResults:
        """Estimate by within-transformed least squares."""
        blocks = _build_blocks(self.panel, self.p, self.q, self.det)
        if len(blocks) < 2:
            raise ValueError(
                f"Only {len(blocks)} individual(s) have enough observations."
            )
        # The intercept is absorbed by the within transformation, so it
        # must not also sit in the design.
        drop = {"const"}
        keep = [j for j, n in enumerate(blocks[0].w_names) if n not in drop]
        names = ["ec"] + [f"theta.{n}" for n in self.x_names]
        names += [blocks[0].w_names[j] for j in keep]

        rows, target = [], []
        for b in blocks:
            design = np.column_stack([b.y_lag, b.x_lag, b.w[:, keep]])
            rows.append(design - design.mean(axis=0))
            target.append(b.dy - b.dy.mean())
        big_x = np.vstack(rows)
        big_y = np.concatenate(target)
        beta, *_ = np.linalg.lstsq(big_x, big_y, rcond=None)
        resid = big_y - big_x @ beta
        nobs = big_y.size
        k = big_x.shape[1] + len(blocks)  # + the absorbed intercepts
        dof = nobs - k
        sigma2 = float(resid @ resid) / dof
        cov = sigma2 * np.linalg.pinv(big_x.T @ big_x)
        se = np.sqrt(np.diag(cov))

        lam = float(beta[0])
        n_x = len(self.x_names)
        theta = -beta[1 : 1 + n_x] / lam
        # Delta method for theta = -b_x / lam.
        grad = np.zeros((n_x, beta.size))
        for j in range(n_x):
            grad[j, 0] = beta[1 + j] / lam**2
            grad[j, 1 + j] = -1.0 / lam
        theta_cov = grad @ cov @ grad.T
        longrun = pd.DataFrame(
            {"theta": theta, "se": np.sqrt(np.diag(theta_cov))},
            index=pd.Index(list(self.x_names), name="regressor"),
        )
        loglik = float(
            -0.5 * nobs * (np.log(2 * np.pi) + 1 + np.log(float(resid @ resid) / nobs))
        )
        return DFEResults(
            longrun=longrun,
            adjustment=pd.Series(
                {"lambda": lam, "se": float(se[0])}, name="adjustment"
            ),
            params=pd.Series(beta, index=pd.Index(names, name="term")),
            bse=pd.Series(se, index=pd.Index(names, name="term")),
            loglik=loglik,
            nobs=nobs,
            n_units=len(blocks),
            panel=self.panel,
        )


def compare(
    df: pd.DataFrame,
    y: str,
    X: Sequence[str],
    id: str,  # noqa: A002
    time: str,
    order: tuple[int, int] = (1, 1),
    det: DetType = "const",
    min_obs: int = 15,
) -> tuple[pd.DataFrame, HausmanResult]:
    """Run MG, PMG and DFE on one panel and tabulate the long run.

    The table panel papers report, plus the Hausman test that says which
    of the first two to believe.

    Returns
    -------
    table : pandas.DataFrame
        One row per estimator per regressor: ``theta``, ``se``, ``t``.
    hausman : HausmanResult

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> rows = []
    >>> for i in range(12):
    ...     lam = -0.5 + 0.05 * rng.normal()
    ...     x = np.cumsum(rng.normal(size=50))
    ...     y = np.zeros(50)
    ...     for t in range(1, 50):
    ...         y[t] = y[t-1] + lam * (y[t-1] - 0.8 * x[t-1]) + rng.normal(scale=.3)
    ...     rows.append(pd.DataFrame({"id": i, "t": np.arange(50), "y": y, "x": x}))
    >>> table, h = compare(pd.concat(rows, ignore_index=True), y="y", X=["x"],
    ...                    id="id", time="t")
    >>> sorted(set(table.index.get_level_values("estimator")))
    ['DFE', 'MG', 'PMG']
    """
    kw = {"y": y, "X": list(X), "id": id, "time": time, "min_obs": min_obs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mg = MeanGroup(df, order=order, det=det, **kw).fit()  # type: ignore[arg-type]
        pmg = PMG(df, order=order, det=det, **kw).fit()  # type: ignore[arg-type]
        dfe = DFE(df, order=order, det=det, **kw).fit()  # type: ignore[arg-type]

    frames = []
    for label, res in (("MG", mg), ("PMG", pmg), ("DFE", dfe)):
        block = res.longrun[["theta", "se"]].copy()
        block["t"] = block["theta"] / block["se"]
        block["estimator"] = label
        frames.append(block.reset_index().set_index(["estimator", "regressor"]))
    table = pd.concat(frames)
    return table, hausman(mg, pmg)
