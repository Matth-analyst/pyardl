r"""The quantile ARDL of Cho, Kim & Shin (2015).

An ARDL describes the *conditional mean*. A QARDL estimates the same
error-correction model at a grid of quantiles, so every parameter becomes
a function of :math:`\tau`:

.. math::

    Q_{\Delta y_t}(\tau \mid \mathcal{F}_{t-1})
      = d_t(\tau) + \lambda(\tau) y_{t-1}
      + \sum_j \gamma_j(\tau) x_{j,t-1}
      + \sum_i \psi_i(\tau) \Delta y_{t-i}
      + \sum_{j,i} \omega_{j,i}(\tau) \Delta x_{j,t-i},

with long-run coefficients :math:`\theta_j(\tau) = -\gamma_j(\tau) /
\lambda(\tau)`.

That is not a cosmetic generalisation. The long-run relation may hold in
the lower tail and not in the middle; adjustment may be fast after bad
shocks and slow after good ones. A mean regression averages all of that
into one number and reports it as *the* relationship.

**The design is the one the rest of the library uses.** The columns come
from the same builder as the classical bounds test, so a QARDL at
:math:`\tau = 0.5` and an ARDL are the same specification estimated under
two different losses — not two models that happen to resemble each other.

Two inference routes, because two different things are being asked:

``'mbb'`` (default)
    Moving-block bootstrap. Blocks of **rows** are resampled, target and
    design together, so the dependence between neighbouring dates
    survives and the design-target link is never broken. It returns the
    *joint* law of the coefficients across quantiles, which is what the
    constancy and symmetry tests need — a per-quantile covariance cannot
    express how :math:`\theta(0.1)` and :math:`\theta(0.9)` move
    together.
``'kernel'``
    The per-quantile sparsity estimator. Fast, and enough when only one
    quantile is being read, but it says nothing across quantiles.

References
----------
.. [1] Cho, J. S., Kim, T. & Shin, Y. (2015). Quantile cointegration in
       the autoregressive distributed-lag modeling framework. *Journal of
       Econometrics*, 188(1), 281-300.
.. [2] Koenker, R. (2005). *Quantile Regression*. Cambridge University
       Press.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.qardl.estimate import quantile_regression

__all__ = ["QARDL", "QARDLResults"]

Inference = Literal["mbb", "kernel"]
Calibration = Literal["null", "mbb", "chi2"]

#: Default quantile grid: the spec's, and the one this literature plots.
DEFAULT_TAUS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))

#: Below this, the adjustment speed is treated as indistinguishable from
#: zero and the long-run coefficient — a ratio with it in the
#: denominator — is not reported.
LAMBDA_TOL = 1e-8

FloatArray = npt.NDArray[np.float64]


def _block_length(n_obs: int) -> int:
    r"""Default moving-block length, :math:`\lceil T^{1/3} \rceil`.

    The usual rule of thumb: long enough to carry the short-run
    dependence, short enough that the blocks are many.
    """
    return max(2, int(np.ceil(n_obs ** (1.0 / 3.0))))


def _moving_block_indices(
    n_obs: int, block: int, rng: np.random.Generator
) -> FloatArray:
    """One resampled index vector, built from overlapping blocks."""
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n_obs]


@dataclass(frozen=True)
class QARDLResults:
    """Outcome of a QARDL fit."""

    model: QARDL
    taus: tuple[float, ...]
    names: tuple[str, ...]
    _params: FloatArray = field(repr=False)
    _covs: FloatArray = field(repr=False)
    _draws: FloatArray | None = field(repr=False)
    inference: Inference
    n_boot: int
    seed: int | None
    block_length: int | None
    nobs: int
    _fit: Any = field(repr=False)
    _target: FloatArray = field(repr=False)
    _design: FloatArray = field(repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    @property
    def coefficients(self) -> pd.DataFrame:
        """Every coefficient, one row per quantile."""
        return pd.DataFrame(
            self._params,
            index=pd.Index(self.taus, name="tau"),
            columns=list(self.names),
        )

    @property
    def lam(self) -> pd.Series:
        r"""The adjustment speed :math:`\lambda(\tau)`."""
        return self.coefficients[self._fit.lam_name]

    def _index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not a coefficient of this model; available: "
                f"{list(self.names)}."
            ) from None

    def _level_name(self, variable: str) -> str:
        """Level term of a regressor, whichever lag convention applies."""
        for candidate in (f"{variable}.L1", f"{variable}.L0"):
            if candidate in self.names:
                return candidate
        raise KeyError(
            f"No level term for {variable!r}; regressors are "
            f"{[n for n in self._fit.tested if n != self._fit.lam_name]}."
        )

    @property
    def regressors(self) -> tuple[str, ...]:
        """Names of the regressors carrying a long-run coefficient."""
        return tuple(
            n.rsplit(".L", 1)[0] for n in self._fit.tested if n != self._fit.lam_name
        )

    # ------------------------------------------------------------------
    def _theta_from(self, params: FloatArray, variable: str) -> FloatArray:
        r""" ":math:`\theta = -\gamma/\lambda` for one variable.

        ``params`` may be one ``(n_tau, k)`` block or a stack of them.
        Where :math:`\lambda` is numerically zero the ratio is not
        defined and the entry becomes ``NaN``: a long-run coefficient
        read off a vanishing adjustment speed is an artefact of division,
        not an estimate.
        """
        lam = params[..., self._index_of(self._fit.lam_name)]
        gamma = params[..., self._index_of(self._level_name(variable))]
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = -gamma / lam
        return np.where(np.abs(lam) < LAMBDA_TOL, np.nan, theta)

    def longrun(self, variable: str | None = None, bands: str = "rows") -> pd.DataFrame:
        r"""Long-run coefficients :math:`\theta_j(\tau)`, with bands.

        Parameters
        ----------
        variable : str, optional
            Restrict to one regressor. Defaults to all of them.
        bands : {"rows", "fixed-design", "delta"}
            How the interval is built when the fit used ``'mbb'``.

            ``"rows"`` (default) resamples blocks of rows, design
            included. ``"fixed-design"`` resamples only the residuals,
            in blocks, around each quantile's own fit. ``"delta"``
            forces the per-quantile delta method even under ``'mbb'``.

            **The default was chosen by measuring coverage, and the
            measurement contradicted what this module expected.**
            OBS-14 had established that row resampling inflates the
            spread of a *contrast between quantiles* by 1.36, and
            inferred that the bands were correspondingly too wide. They
            are not. On 150 replications of a DGP with a known
            ``theta(tau)``, at a nominal 90% (standard error 2.45
            points):

            =============== ================ ==================
            route           homogeneous DGP  tau-varying DGP
            =============== ================ ==================
            ``rows``        90.7 - 94.0 %    88.0 - 92.7 %
            ``fixed-design``  80.7 - 82.7 %    76.7 - 81.3 %
            ``delta``       88.7 - 92.0 %    88.0 - 90.0 %
            =============== ================ ==================

            Holding the design fixed **removes a real source of
            variability**: here the regressor is random, and the
            sampling variation of ``theta_hat(tau)`` genuinely includes
            it. The argument that the design is not random belongs to
            the *contrast* problem, where it was verified; it does not
            transfer to the level of the coefficient. ``"fixed-design"``
            is kept because the comparison is worth reproducing, and it
            is documented as under-covering rather than removed
            silently. See OBS-29 and
            ``validation/spec18_band_coverage.py``.

            Under ``inference='kernel'`` there are no draws and the
            delta method is used whatever this says.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``tau``; for each variable, the estimate and the
            bounds of a 90% interval. Under ``'mbb'`` the interval comes
            from the bootstrap draws — no delta method, no linearisation
            of a ratio near zero. Under ``'kernel'`` it comes from the
            delta method at each quantile.

        Warns
        -----
        PyardlMethodologyWarning
            When the adjustment speed is indistinguishable from zero at
            some quantile, so the ratio is not defined there.
        """
        if bands not in ("fixed-design", "rows", "delta"):
            raise ValueError(
                f'bands must be "fixed-design", "rows" or "delta", got {bands!r}.'
            )
        if bands == "fixed-design":
            source = self._fixed_design_draws()
        elif bands == "rows":
            source = self._draws
        else:
            source = None

        variables = self.regressors if variable is None else (variable,)
        out: dict[str, FloatArray] = {}
        degenerate: list[float] = []
        for name in variables:
            point = self._theta_from(self._params, name)
            out[name] = point
            degenerate += [
                t for t, v in zip(self.taus, point, strict=True) if np.isnan(v)
            ]
            if source is not None:
                drawn = self._theta_from(source, name)
                out[f"{name}_lower"] = np.nanquantile(drawn, 0.05, axis=0)
                out[f"{name}_upper"] = np.nanquantile(drawn, 0.95, axis=0)
            else:
                se = self._delta_se(name)
                out[f"{name}_lower"] = point - 1.645 * se
                out[f"{name}_upper"] = point + 1.645 * se

        if degenerate:
            warnings.warn(
                f"The adjustment speed is indistinguishable from zero at "
                f"tau in {sorted(set(degenerate))}: the long-run coefficient "
                "is a ratio with it in the denominator, so it is reported as "
                "NaN rather than as a very large number.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )
        return pd.DataFrame(out, index=pd.Index(self.taus, name="tau"))

    def _delta_se(self, variable: str) -> FloatArray:
        r"""Standard error of :math:`\theta(\tau)` by the delta method."""
        from pyardl.utils import _delta_method

        i_lam = self._index_of(self._fit.lam_name)
        i_gam = self._index_of(self._level_name(variable))
        out = np.empty(len(self.taus))
        for t in range(len(self.taus)):
            theta_hat = self._params[t, [i_gam, i_lam]]
            if abs(theta_hat[1]) < LAMBDA_TOL:
                # The ratio is not defined, so neither is its variance.
                # Linearising around a vanishing denominator would return
                # a number, and that number would be meaningless.
                out[t] = np.nan
                continue
            v_hat = self._covs[t][np.ix_([i_gam, i_lam], [i_gam, i_lam])]

            def g(v: FloatArray) -> FloatArray:
                return np.array([-v[0] / v[1]])

            _, cov_g = _delta_method(g, theta_hat, v_hat)
            out[t] = float(np.sqrt(cov_g[0, 0]))
        return out

    # ------------------------------------------------------------------
    def _require_draws(self, what: str) -> FloatArray:
        if self._draws is None:
            raise ValueError(
                f"{what} needs the joint law of the coefficients across "
                "quantiles, which the per-quantile kernel estimator cannot "
                "give. Refit with inference='mbb'."
            )
        return self._draws

    def _null_draws(self, n_boot: int | None = None) -> FloatArray | None:
        """Draws under the null of no quantile variation, computed once.

        Held in a small cache on the result: the two joint tests need the
        same draws, and computing them twice would double the cost for
        nothing while giving two slightly different p-values for the same
        data.
        """
        if self._draws is None:
            return None
        cached = self._cache.get("null_draws")
        if cached is not None:
            return np.asarray(cached, dtype=np.float64)
        rng = np.random.default_rng(
            (self.seed or 0) + 987_654_321  # a stream of its own
        )
        block = self.block_length or _block_length(self.nobs)
        drawn = _null_contrast_draws(
            self._target,
            self._design,
            self.taus,
            n_boot if n_boot is not None else self.n_boot,
            block,
            rng,
        )
        self._cache["null_draws"] = drawn
        return drawn

    def _fixed_design_draws(self, n_boot: int | None = None) -> FloatArray | None:
        """Band draws with the design held fixed, computed once and cached.

        Lazy on purpose. A fit that is never asked for bands should not
        pay for a second bootstrap, and most fits are made for the
        constancy test rather than for the figure.
        """
        if self._draws is None:
            return None
        cached: FloatArray | None = self._cache.get("band_draws")
        if cached is not None and (n_boot is None or cached.shape[0] == n_boot):
            return cached
        block = self.block_length or _block_length(self.nobs)
        rng = np.random.default_rng(self.seed)
        drawn = _band_draws(
            self._target,
            self._design,
            self.taus,
            self._params,
            self.n_boot if n_boot is None else n_boot,
            block,
            rng,
        )
        self._cache["band_draws"] = drawn
        return drawn

    def wald_constancy(
        self,
        variable: str | None = None,
        calibration: Calibration = "null",
    ) -> pd.DataFrame:
        r"""Test whether the long run is the same at every quantile.

        :math:`H_0: \theta_j(\tau_1) = \dots = \theta_j(\tau_m)` — the
        signature test of this framework. Rejecting it says the long-run
        relation is not a single number: it differs across the
        distribution, and reporting a mean-regression estimate would
        average away the finding.

        The contrasts are :math:`\theta(\tau_i) - \theta(\tau_1)`, and
        their covariance comes from the bootstrap draws, which is why
        this needs ``inference='mbb'``.

        Returns
        -------
        pandas.DataFrame
            One row per variable: the chi-squared statistic, its degrees
            of freedom, the p-value and a 5% verdict.
        """
        from scipy.stats import chi2

        draws = self._require_draws("wald_constancy")
        null_draws = self._null_draws() if calibration == "null" else None
        variables = self.regressors if variable is None else (variable,)
        rows = []
        for name in variables:
            point = self._theta_from(self._params, name)
            drawn = self._theta_from(draws, name)
            null_theta = (
                None if null_draws is None else self._theta_from(null_draws, name)
            )
            stat, dof, pvalue = _contrast_test(
                point,
                drawn,
                null_theta,
                [(i, 0) for i in range(1, len(self.taus))],
                calibration,
            )
            rows.append(
                {
                    "variable": name,
                    "stat": stat,
                    "df": dof,
                    "pvalue": pvalue,
                    "decision": _verdict(pvalue, "varies with tau", "constant"),
                }
            )
        del chi2
        return pd.DataFrame(rows).set_index("variable")

    def symmetry_test(
        self,
        variable: str | None = None,
        calibration: Calibration = "null",
    ) -> pd.DataFrame:
        r"""Test :math:`\theta_j(\tau) = \theta_j(1-\tau)`.

        Whether the relation treats the two tails alike. A grid without
        matching pairs cannot answer this, and the test says so rather
        than pairing quantiles that are not mirror images.
        """
        draws = self._require_draws("symmetry_test")
        null_draws = self._null_draws() if calibration == "null" else None
        pairs = []
        for i, tau in enumerate(self.taus):
            mirror = round(1.0 - tau, 10)
            if tau < 0.5 and mirror in [round(t, 10) for t in self.taus]:
                pairs.append((i, [round(t, 10) for t in self.taus].index(mirror)))
        if not pairs:
            raise ValueError(
                "The quantile grid holds no mirror pair (tau, 1-tau), so "
                "symmetry cannot be tested on it. Use a grid symmetric "
                "about 0.5."
            )
        variables = self.regressors if variable is None else (variable,)
        rows = []
        for name in variables:
            point = self._theta_from(self._params, name)
            drawn = self._theta_from(draws, name)
            null_theta = (
                None if null_draws is None else self._theta_from(null_draws, name)
            )
            stat, dof, pvalue = _contrast_test(
                point, drawn, null_theta, pairs, calibration
            )
            rows.append(
                {
                    "variable": name,
                    "stat": stat,
                    "df": dof,
                    "pvalue": pvalue,
                    "decision": _verdict(pvalue, "asymmetric", "symmetric"),
                }
            )
        return pd.DataFrame(rows).set_index("variable")

    def cointegration_test(
        self, tau: float = 0.5, n_boot: int = 299, seed: int | None = None
    ) -> pd.Series:
        r"""Test :math:`\lambda(\tau) = 0` at one quantile.

        The t ratio on the adjustment speed, exactly as in the classical
        framework — and, exactly as there, with a **non-standard**
        distribution: the regressors are integrated, so no tabulated t
        applies. The critical values are therefore generated here, by
        regenerating data under a null with the level terms deleted and
        re-estimating the quantile regression on each sample.

        Left-tailed: rejection needs a *negative* estimate, an actual
        pull back towards equilibrium.

        Returns
        -------
        pandas.Series
            ``lambda``, ``t_stat``, the bootstrap critical values and the
            verdict.
        """
        from pyardl.bootstrap import estimate_null_dgp
        from pyardl.bootstrap.dgp import simulate_paths
        from pyardl.bootstrap.resample import resample_residuals

        if tau not in self.taus:
            raise ValueError(
                f"tau={tau} was not estimated; the grid is {list(self.taus)}."
            )
        position = self.taus.index(tau)
        i_lam = self._index_of(self._fit.lam_name)
        lam_hat = float(self._params[position, i_lam])
        se = float(np.sqrt(self._covs[position][i_lam, i_lam]))
        t_stat = lam_hat / se

        rng = np.random.default_rng(seed)
        model = self.model
        dgp = estimate_null_dgp(
            model._y_arr,
            model._x_arr,
            p=model.p,
            q=model.q_tuple,
            case=model.case,
        )
        burn_in = 50
        n_obs = model._y_arr.shape[0]
        n_eq = 1 + dgp.n_regressors
        block = np.empty((n_boot, burn_in + n_obs, n_eq))
        for b in range(n_boot):
            block[b] = resample_residuals(dgp.residuals, burn_in + n_obs, rng)
        y_star, x_star = simulate_paths(
            dgp, block, y0=model._y_arr[0], x0=model._x_arr[0], burn_in=burn_in
        )

        drawn = np.empty(n_boot)
        drawn.fill(np.nan)
        for b in range(n_boot):
            try:
                fit = model._uecm(y_star[b], x_star[b])
                target = fit.design @ fit.params.to_numpy() + fit.resid
                params, cov = quantile_regression(target, fit.design, tau)
                j = list(fit.names).index(fit.lam_name)
                drawn[b] = params[j] / float(np.sqrt(cov[j, j]))
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                continue
        usable = drawn[np.isfinite(drawn)]
        critical = {a: float(np.quantile(usable, a)) for a in (0.10, 0.05, 0.01)}
        pvalue = float((1 + np.sum(usable <= t_stat)) / (usable.size + 1))
        return pd.Series(
            {
                "tau": tau,
                "lambda": lam_hat,
                "t_stat": t_stat,
                "cv_10": critical[0.10],
                "cv_5": critical[0.05],
                "cv_1": critical[0.01],
                "pvalue": pvalue,
                "n_boot": int(usable.size),
                "decision": (
                    "cointegration" if t_stat < critical[0.05] else "no_cointegration"
                ),
            },
            name=f"cointegration at tau={tau}",
        )

    # ------------------------------------------------------------------
    def plot_coefficients(
        self, variable: str | None = None, show_lambda: bool = True
    ) -> Any:
        r"""Plot :math:`\theta_j(\tau)` and :math:`\lambda(\tau)`.

        The canonical figure of this literature. A flat line is the
        finding that a mean regression would have been enough; a sloped
        one is the finding that it would not.

        Raises
        ------
        ImportError
            If matplotlib, an optional dependency, is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Plotting requires matplotlib, an optional dependency. "
                "Install it with: pip install pyardl[plot]"
            ) from exc

        variables = self.regressors if variable is None else (variable,)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyardlMethodologyWarning)
            table = self.longrun()
        n_panel = len(variables) + (1 if show_lambda else 0)
        fig, axes = plt.subplots(
            n_panel, 1, figsize=(7.0, 2.6 * n_panel), sharex=True, squeeze=False
        )
        flat = list(axes[:, 0])
        taus = np.asarray(self.taus)

        for ax, name in zip(flat, variables, strict=False):
            ax.plot(taus, table[name], color="#2A2F86", marker="o", markersize=3)
            ax.fill_between(
                taus,
                table[f"{name}_lower"],
                table[f"{name}_upper"],
                color="#2A2F86",
                alpha=0.15,
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel(rf"$\theta_{{{name}}}(\tau)$")

        if show_lambda:
            ax = flat[-1]
            ax.plot(
                taus, self.lam.to_numpy(), color="#B4451F", marker="o", markersize=3
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel(r"$\lambda(\tau)$")

        flat[-1].set_xlabel(r"$\tau$")
        flat[0].set_title("Quantile ARDL - coefficients across the distribution")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Readable report of the fit and the two joint tests."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyardlMethodologyWarning)
            longrun = self.longrun()
        lines = [
            f"QARDL (Cho, Kim & Shin 2015) - case {self.model.case}, "
            f"{len(self.taus)} quantiles, {self.nobs} observations",
            f"  inference: {self.inference}"
            + (
                f", B={self.n_boot}, block={self.block_length}, seed={self.seed}"
                if self.inference == "mbb"
                else ""
            ),
            "",
            f"  {'tau':>6}{'lambda':>10}"
            + "".join(f"{'theta_' + v:>14}" for v in self.regressors),
        ]
        i_lam = self._index_of(self._fit.lam_name)
        for i, tau in enumerate(self.taus):
            row = f"  {tau:>6}{self._params[i, i_lam]:>10.4f}"
            for name in self.regressors:
                row += f"{longrun[name].iloc[i]:>14.4f}"
            lines.append(row)

        if self._draws is not None:
            lines += ["", "  joint tests across quantiles"]
            for label, frame in (
                ("constancy", self.wald_constancy()),
                ("symmetry", self.symmetry_test()),
            ):
                for name, row in frame.iterrows():
                    lines.append(
                        f"  {label:>10}  {name:>10}  chi2({int(row['df'])}) = "
                        f"{row['stat']:.4f}   p = {row['pvalue']:.4f}   "
                        f"{row['decision']}"
                    )
        else:
            lines += [
                "",
                "  joint tests unavailable: the kernel estimator gives no law",
                "  across quantiles. Refit with inference='mbb'.",
            ]
        return "\n".join(lines)


def _band_draws(
    target: FloatArray,
    design: FloatArray,
    taus: tuple[float, ...],
    params: FloatArray,
    n_boot: int,
    block: int,
    rng: np.random.Generator,
) -> FloatArray:
    r"""Sampling law of :math:`\hat\beta(\tau)`, design held **fixed**.

    Same lesson as :func:`_null_contrast_draws`, applied to the bands.
    Resampling rows of the design mixes blocks of an **integrated**
    regressor: the blocks keep the local dependence but not the
    stochastic trend, which is what makes an I(1) series an I(1) series.
    The bootstrap designs come out more erratic than the real one, the
    estimates more dispersed, and the bands too wide by the same factor
    of 1.36 that OBS-14 measured on the contrasts.

    So the design is not resampled. Only the residuals are, in blocks,
    and **around each quantile's own fit** rather than around a common
    one:

    .. math::

        y^{*}_\tau = X \hat\beta(\tau) + \hat u_\tau[\text{idx}],
        \qquad \hat u_\tau = y - X\hat\beta(\tau).

    Bootstrapping every quantile around the median fit instead would be
    simpler, and it is what the null route does — deliberately, because
    that *is* the null being tested there. Here it would impose the very
    thing the bands are supposed to display: a location shift, in which
    the slopes do not vary with tau. The band would then be narrow
    exactly where the interesting case is.

    **This route is NOT the default, and the reason is measured.** It
    covers 77 % to 83 % at a nominal 90 %, against 88 % to 94 % for the
    row resampling it was written to replace. Holding the design fixed
    removes a source of variability that is real here: the regressor is
    random, and the sampling law of ``theta_hat(tau)`` includes it. The
    "the design is not random" argument was verified for the *contrast*
    between quantiles (OBS-14) and does not carry over to the level of
    the coefficient. It is kept for the comparison, not for use. OBS-29.

    The residual :math:`\hat u_\tau` has its tau-quantile at zero by
    construction, so the resampled sample has the same conditional
    quantile at tau as the fit it came from, and the draw is centred on
    the estimate rather than on some other quantile of it.
    """
    n_obs = design.shape[0]
    drawn = np.empty((n_boot, len(taus), design.shape[1]))
    drawn.fill(np.nan)
    fitted = design @ params.T  # (n_obs, n_tau)
    for i, tau in enumerate(taus):
        resid = target - fitted[:, i]
        for b in range(n_boot):
            idx = _moving_block_indices(n_obs, block, rng)
            star = fitted[:, i] + resid[idx]
            try:
                drawn[b, i], _ = quantile_regression(star, design, tau)
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                continue
    return drawn


def _null_contrast_draws(
    target: FloatArray,
    design: FloatArray,
    taus: tuple[float, ...],
    n_boot: int,
    block: int,
    rng: np.random.Generator,
) -> FloatArray:
    r"""Contrasts simulated under the null of no quantile variation.

    Resampling **rows** of the design is what broke the previous route:
    the regressors are integrated, and shuffling blocks of rows destroys
    the stochastic trend that makes them what they are. Measured, the
    resulting spread was 1.36 times the true sampling spread — which
    deflates any Wald statistic built on it by a factor of 1.85, and the
    test then almost never fires.

    So the design is held **fixed** — it is not random, and the
    literature does not treat it as such — and only the innovations are
    resampled, in blocks, from the median fit. Under the null that the
    coefficients do not vary with :math:`	au`, that is exactly the data
    generating process, which is the same principle the rest of the
    library follows: simulate under the null being tested, not under the
    estimated model.

    Returns
    -------
    numpy.ndarray, shape (n_boot, n_tau, k)
        Coefficients at every quantile, on every regenerated sample.
    """
    median = float(np.median(taus)) if len(taus) % 2 else 0.5
    reference = min(taus, key=lambda t: abs(t - median))
    beta0, _ = quantile_regression(target, design, reference)
    resid = target - design @ beta0

    n_obs = design.shape[0]
    drawn = np.empty((n_boot, len(taus), design.shape[1]))
    drawn.fill(np.nan)
    for b in range(n_boot):
        idx = _moving_block_indices(n_obs, block, rng)
        star = design @ beta0 + resid[idx]
        for i, tau in enumerate(taus):
            try:
                drawn[b, i], _ = quantile_regression(star, design, tau)
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                continue
    return drawn


def _contrast_test(
    point: FloatArray,
    drawn: FloatArray,
    null_drawn: FloatArray | None,
    pairs: list[tuple[int, int]],
    calibration: Calibration = "null",
) -> tuple[float, int, float]:
    r"""Test that a set of differences are jointly zero.

    The statistic is the usual quadratic form
    :math:`W = d' \hat V^{-1} d`. What differs is what it is compared
    against, and the choice was **measured**, not assumed:

    ``'chi2'``
        The asymptotic reference. It presumes :math:`\hat V` is the
        sampling covariance of the contrasts. Measured under a
        homogeneous null, this rejects **0.5%** of the time at a nominal
        5% — the moving-block bootstrap variance is larger than the
        sampling one, so the statistic is deflated and the test almost
        never fires.
    ``'bootstrap'`` (default)
        The same quadratic form recomputed on each bootstrap draw,
        recentred on the bootstrap mean, and the observed value read
        against *that* distribution. Whatever scale :math:`\hat V`
        carries, it carries it in both places and cancels. This is the
        textbook remedy for exactly the failure above, and it is why it
        is the default.
    """
    from scipy.stats import chi2

    diff = np.array([point[i] - point[j] for i, j in pairs])
    drawn_diff = np.column_stack([drawn[:, i] - drawn[:, j] for i, j in pairs])
    usable = drawn_diff[np.all(np.isfinite(drawn_diff), axis=1)]
    if not np.all(np.isfinite(diff)) or usable.shape[0] <= len(pairs):
        # Either the point estimate is undefined somewhere, or fewer
        # usable draws than contrasts: the covariance would be singular
        # by construction. Returning NaN says so; returning a number
        # would invent one.
        return float("nan"), len(pairs), float("nan")
    cov = np.atleast_2d(np.cov(usable, rowvar=False))
    try:
        solved = np.linalg.solve(cov, diff)
    except np.linalg.LinAlgError:  # pragma: no cover - singular contrast
        return float("nan"), len(pairs), float("nan")
    stat = float(diff @ solved)
    dof = len(pairs)

    if calibration == "chi2":
        return stat, dof, float(chi2.sf(stat, dof))

    if calibration == "mbb":
        reference = usable
    else:
        if null_drawn is None:
            return stat, dof, float("nan")
        reference = np.column_stack(
            [null_drawn[:, i] - null_drawn[:, j] for i, j in pairs]
        )
        reference = reference[np.all(np.isfinite(reference), axis=1)]
    if reference.shape[0] <= len(pairs):
        return stat, dof, float("nan")

    centred = reference - reference.mean(axis=0)
    try:
        drawn_stat = np.einsum("bi,bi->b", centred, np.linalg.solve(cov, centred.T).T)
    except np.linalg.LinAlgError:  # pragma: no cover - singular contrast
        return float("nan"), dof, float("nan")
    pvalue = float((1 + np.sum(drawn_stat >= stat)) / (drawn_stat.size + 1))
    return stat, dof, pvalue


def _verdict(pvalue: float, reject: str, keep: str) -> str:
    if not np.isfinite(pvalue):
        return "unavailable"
    return reject if pvalue < 0.05 else keep


class QARDL:
    r"""Quantile ARDL, with the option of asymmetric decomposition.

    Parameters
    ----------
    y, x : array_like
        Dependent variable and regressors.
    order : tuple
        ``(p, q)`` of the error-correction model.
    taus : sequence of float, optional
        Quantile grid; defaults to 0.05 to 0.95 in steps of 0.05.
    asym : sequence of str, optional
        Regressors to split into partial sums before estimating — the
        QNARDL of the two frameworks combined. Reuses the decomposition
        of :mod:`pyardl.nardl`, identity check included.
    case : int, default 3
        Deterministic case, PSS numbering.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = np.cumsum(rng.normal(size=n))
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = y[t - 1] - 0.4 * (y[t - 1] - 1.5 * x[t - 1]) + rng.normal()
    >>> res = QARDL(pd.Series(y, name="y"), pd.DataFrame({"x": x}),
    ...             order=(1, 1), taus=(0.25, 0.5, 0.75)).fit(
    ...                 inference="kernel")
    >>> res.taus
    (0.25, 0.5, 0.75)
    """

    def __init__(
        self,
        y: npt.ArrayLike,
        x: npt.ArrayLike,
        order: tuple[int, int | dict[str, int]] = (1, 1),
        taus: Sequence[float] = DEFAULT_TAUS,
        asym: Sequence[str] | None = None,
        case: int = 3,
    ) -> None:
        from pyardl.utils import check_series

        y_arr, x_arr, index, y_name, x_names = check_series(y, x)
        if x_arr is None:
            raise ValueError("A QARDL needs at least one regressor.")
        if case not in (1, 2, 3, 4, 5):
            raise ValueError(f"case must be 1..5 (PSS numbering), got {case}.")

        grid = tuple(float(t) for t in taus)
        if not grid:
            raise ValueError("taus is empty: there is no quantile to estimate.")
        if any(not 0.0 < t < 1.0 for t in grid):
            raise ValueError(f"every tau must lie strictly in (0, 1), got {grid}.")
        if len(set(grid)) != len(grid):
            raise ValueError(f"taus holds duplicates: {grid}.")
        self.taus = tuple(sorted(grid))

        self.case = int(case)
        self._y_name = y_name
        self.asym = tuple(asym) if asym is not None else ()

        if self.asym:
            from pyardl.nardl.decompose import partial_sums

            unknown = [a for a in self.asym if a not in x_names]
            if unknown:
                raise ValueError(
                    f"asym names {unknown} are not regressors; available: "
                    f"{list(x_names)}."
                )
            columns: dict[str, FloatArray] = {}
            for j, name in enumerate(x_names):
                series = pd.Series(x_arr[:, j], index=index, name=name)
                if name in self.asym:
                    pos, neg = partial_sums(series)
                    columns[str(pos.name)] = pos.to_numpy()
                    columns[str(neg.name)] = neg.to_numpy()
                else:
                    columns[name] = x_arr[:, j]
            self._x_names = tuple(columns)
            self._x_arr = np.column_stack(list(columns.values()))
        else:
            self._x_names = tuple(x_names)
            self._x_arr = x_arr

        self._y_arr = y_arr
        p, q = order
        if isinstance(q, dict):
            missing = [c for c in self._x_names if c not in q]
            if missing:
                raise ValueError(f"order dict is missing {missing}.")
            self.q_tuple = tuple(int(q[c]) for c in self._x_names)
        else:
            self.q_tuple = tuple(int(q) for _ in self._x_names)
        self.p = int(p)

    def _uecm(self, y_arr: FloatArray, x_arr: FloatArray) -> Any:
        """The error-correction design, built exactly as elsewhere."""
        from pyardl.bounds.pss import _estimate_uecm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _estimate_uecm(
                y_arr,
                x_arr,
                self._x_names,
                self._y_name,
                self.p,
                self.q_tuple,
                self.case,
            )

    def fit(
        self,
        inference: Inference = "mbb",
        n_boot: int = 299,
        seed: int | None = None,
        block_length: int | None = None,
    ) -> QARDLResults:
        r"""Estimate the model at every quantile of the grid.

        Parameters
        ----------
        inference : {'mbb', 'kernel'}
            ``'mbb'`` resamples blocks of rows and returns the joint law
            across quantiles — required by the constancy and symmetry
            tests. ``'kernel'`` uses the per-quantile sparsity estimator:
            fast, and silent about anything joint.
        n_boot : int, default 299
            Bootstrap replications. 299 gives a p-value resolution of
            1/300, ample at conventional levels; raise it when reporting
            a p-value that sits near a threshold. The cost is linear:
            999 replications take roughly three times as long.
        seed : int, optional
            Drawn from entropy and **recorded** when omitted.
        block_length : int, optional
            Moving-block length; defaults to the ceiling of ``T^(1/3)``.

        Returns
        -------
        QARDLResults
        """
        if inference not in ("mbb", "kernel"):
            raise ValueError(f"inference must be 'mbb' or 'kernel', got {inference!r}.")
        fit = self._uecm(self._y_arr, self._x_arr)
        target = fit.design @ fit.params.to_numpy() + fit.resid
        design = fit.design
        names = tuple(str(n) for n in fit.names)

        params = np.empty((len(self.taus), design.shape[1]))
        covs = np.empty((len(self.taus), design.shape[1], design.shape[1]))
        for i, tau in enumerate(self.taus):
            params[i], covs[i] = quantile_regression(target, design, tau)

        draws: FloatArray | None = None
        used_seed = seed
        used_block = None
        if inference == "mbb":
            if n_boot < 2:
                raise ValueError(f"n_boot must be at least 2, got {n_boot}.")
            if used_seed is None:
                entropy = np.random.SeedSequence().entropy
                used_seed = int(entropy) % (2**63) if isinstance(entropy, int) else 0
            rng = np.random.default_rng(used_seed)
            n_est = design.shape[0]
            used_block = (
                _block_length(n_est) if block_length is None else int(block_length)
            )
            if not 1 <= used_block <= n_est:
                raise ValueError(
                    f"block_length={used_block} does not fit {n_est} rows."
                )
            drawn = np.empty((n_boot, len(self.taus), design.shape[1]))
            drawn.fill(np.nan)
            for b in range(n_boot):
                idx = _moving_block_indices(n_est, used_block, rng)
                y_b, x_b = target[idx], design[idx]
                for i, tau in enumerate(self.taus):
                    try:
                        drawn[b, i], _ = quantile_regression(y_b, x_b, tau)
                    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                        continue
            draws = drawn

        return QARDLResults(
            model=self,
            taus=self.taus,
            names=names,
            _params=params,
            _covs=covs,
            _draws=draws,
            inference=inference,
            n_boot=n_boot if inference == "mbb" else 0,
            seed=used_seed if inference == "mbb" else None,
            block_length=used_block,
            nobs=int(design.shape[0]),
            _fit=fit,
            _target=target,
            _design=design,
        )
