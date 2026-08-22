r"""The nonlinear ARDL of Shin, Yu & Greenwood-Nimmo (2014).

A regressor is split into cumulated rises and cumulated falls, and the
error-correction model is written on the two pieces:

.. math::

    \Delta y_t = d_t + \lambda y_{t-1}
                 + \gamma^{+} x^{+}_{t-1} + \gamma^{-} x^{-}_{t-1}
                 + \sum_i \psi_i \Delta y_{t-i}
                 + \sum_i (\omega^{+}_i \Delta x^{+}_{t-i}
                          + \omega^{-}_i \Delta x^{-}_{t-i})
                 + \varepsilon_t,

with asymmetric long-run coefficients
:math:`\theta^{+} = -\gamma^{+}/\lambda` and
:math:`\theta^{-} = -\gamma^{-}/\lambda`.

**Nothing new is estimated here.** Once the regressors are decomposed,
the model is a linear ARDL, and every piece of the library applies to it
unchanged: the same least squares, the same Wald machinery, the same
bounds test, the same bootstrap. That is the point of the framework, and
it is why this module is a thin layer rather than a parallel
implementation.

The asymmetry is a *restriction to be tested*, never assumed:
:math:`\gamma^{+} = \gamma^{-}` in the long run,
:math:`\sum_i \omega^{+}_i = \sum_i \omega^{-}_i` in the short run. When
neither is rejected, the honest reading is that the data do not support
an asymmetric model, and the summary says so rather than leaving the
reader to notice.

References
----------
.. [1] Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). Modelling
       asymmetric cointegration and dynamic multipliers in a nonlinear
       ARDL framework. In *Festschrift in Honor of Peter Schmidt*
       (pp. 281-314). Springer.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.nardl.decompose import Threshold, partial_sums

__all__ = ["NARDL", "NARDLBoundsResults", "NARDLResults"]


def _wald(
    params: pd.Series,
    cov: npt.NDArray[np.float64],
    restrictions: dict[str, float],
    df_resid: int,
) -> tuple[float, float]:
    """F statistic and p-value of a single linear contrast.

    ``restrictions`` maps parameter names to the weights of one contrast,
    e.g. ``{"a": 1.0, "b": -1.0}`` to test ``a = b``.
    """
    from scipy.stats import f as f_dist

    names = list(params.index)
    row = np.zeros(len(names))
    for name, weight in restrictions.items():
        row[names.index(name)] = weight
    diff = float(row @ params.to_numpy())
    var = float(row @ cov @ row)
    if var <= 0:  # pragma: no cover - a degenerate covariance
        return float("nan"), float("nan")
    stat = diff**2 / var
    return stat, float(f_dist.sf(stat, 1, df_resid))


def _wald_joint(
    params: pd.Series,
    cov: npt.NDArray[np.float64],
    contrasts: list[dict[str, float]],
    df_resid: int,
) -> tuple[float, float]:
    """F statistic of several contrasts imposed jointly."""
    from scipy.stats import f as f_dist

    names = list(params.index)
    rows = []
    for contrast in contrasts:
        row = np.zeros(len(names))
        for name, weight in contrast.items():
            row[names.index(name)] = weight
        rows.append(row)
    r_mat = np.vstack(rows)
    diff = r_mat @ params.to_numpy()
    middle = r_mat @ cov @ r_mat.T
    try:
        solved = np.linalg.solve(middle, diff)
    except np.linalg.LinAlgError:  # pragma: no cover - singular contrast
        return float("nan"), float("nan")
    q = r_mat.shape[0]
    stat = float(diff @ solved) / q
    return stat, float(f_dist.sf(stat, q, df_resid))


def _autoregressive_prefix(names: Sequence[str]) -> str:
    """Name of the dependent variable, read off its own lagged term."""
    for name in names:
        if name.endswith(".L1"):
            return str(name)[: -len(".L1")]
    raise ValueError(  # pragma: no cover - a fitted ARDL always has one
        "No autoregressive term found; the multipliers need the ARDL form."
    )


def _multiplier_path(
    params: npt.NDArray[np.float64],
    names: Sequence[str],
    column: str,
    h: int,
    y_prefix: str,
) -> npt.NDArray[np.float64]:
    r"""Cumulated response of ``y`` to a unit step in one regressor.

    Recursion on the ARDL form
    :math:`y_t = \sum_i \phi_i y_{t-i} + \sum_j \beta_j x_{t-j}`, driven
    by a step that switches to one at ``t = 0`` and stays there. The
    path converges to :math:`\sum_j \beta_j / (1 - \sum_i \phi_i)` — the
    long-run coefficient — which is what the tests check.

    ``params`` holds one parameter vector or a stack of them, and the
    result carries one row per vector. The recursion is sequential in
    ``t`` but **independent across draws**, so the loop runs over
    horizons with every draw advanced together. That is what the
    confidence bands need: a thousand paths, not a thousand Python loops.
    """
    stacked = np.atleast_2d(np.asarray(params, dtype=np.float64))
    phi_idx = [i for i, n in enumerate(names) if str(n).startswith(f"{y_prefix}.L")]
    beta_idx = [i for i, n in enumerate(names) if str(n).startswith(f"{column}.L")]
    phi = stacked[:, phi_idx]
    beta = stacked[:, beta_idx]
    if beta.shape[1] == 0:  # pragma: no cover - guarded by the caller
        raise KeyError(f"No ARDL terms found for {column!r}.")

    n_draw, p = phi.shape
    q = beta.shape[1]
    # The step switches to one at t = 0 and stays there, so the exogenous
    # contribution at horizon t is the sum of the first min(t+1, q) betas.
    cumulated = np.cumsum(beta, axis=1)
    driver = np.empty((n_draw, h + 1))
    for t in range(h + 1):
        driver[:, t] = cumulated[:, min(t, q - 1)]

    y = np.zeros((n_draw, h + 1 + p))
    for t in range(h + 1):
        value = driver[:, t].copy()
        for i in range(p):
            value += phi[:, i] * y[:, t + p - 1 - i]
        y[:, t + p] = value
    return y[:, p:]


@dataclass(frozen=True)
class NARDLResults:
    """Outcome of a NARDL fit."""

    model: NARDL
    _fit: Any = field(repr=False)
    _ardl_res: Any = field(repr=False)
    asym: tuple[str, ...]
    threshold: Threshold

    # ------------------------------------------------------------------
    @property
    def params(self) -> pd.Series:
        """Coefficients of the error-correction model."""
        return self._fit.params

    @property
    def uecm(self) -> pd.DataFrame:
        """Coefficients, standard errors and t-ratios of the UECM."""
        se = np.sqrt(np.diag(self._fit.cov))
        coef = self.params
        return pd.DataFrame({"coef": coef, "se": se, "t": coef.to_numpy() / se})

    @property
    def nobs(self) -> int:
        return int(self._fit.nobs)

    @property
    def lam(self) -> float:
        r"""The adjustment speed :math:`\lambda`."""
        return float(self.params[self._fit.lam_name])

    def _level_name(self, column: str) -> str:
        """Name of the level term of a transformed regressor."""
        for candidate in (f"{column}.L1", f"{column}.L0"):
            if candidate in self.params.index:
                return candidate
        raise KeyError(f"No level term found for {column!r}.")  # pragma: no cover

    # ------------------------------------------------------------------
    @property
    def longrun_asym(self) -> pd.DataFrame:
        r"""Asymmetric long-run coefficients and their difference.

        :math:`\theta^{\pm} = -\gamma^{\pm}/\lambda`, with standard errors
        from the delta method applied to
        :math:`(\gamma^{+}, \gamma^{-}, \lambda)` **jointly**: the three
        come from one regression and their covariance is not diagonal, so
        treating them as independent would understate the uncertainty on
        the difference — which is the quantity the whole model is about.
        """
        from pyardl.utils import _delta_method

        names = list(self.params.index)
        i_lam = names.index(self._fit.lam_name)
        rows = []
        for base in self.asym:
            idx = [
                names.index(self._level_name(f"{base}_pos")),
                names.index(self._level_name(f"{base}_neg")),
                i_lam,
            ]
            theta_hat = self.params.to_numpy()[idx]
            v_hat = self._fit.cov[np.ix_(idx, idx)]

            def g(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                gamma_pos, gamma_neg, lam = t
                return np.array(
                    [-gamma_pos / lam, -gamma_neg / lam, (-gamma_pos + gamma_neg) / lam]
                )

            est, cov_g = _delta_method(g, theta_hat, v_hat)
            se = np.sqrt(np.diag(cov_g))
            rows.append(
                {
                    "variable": base,
                    "theta_pos": est[0],
                    "se_pos": se[0],
                    "theta_neg": est[1],
                    "se_neg": se[1],
                    "difference": est[2],
                    "se_difference": se[2],
                }
            )
        return pd.DataFrame(rows).set_index("variable")

    # ------------------------------------------------------------------
    def asymmetry_tests(self) -> pd.DataFrame:
        r"""Wald tests of long-run and short-run symmetry.

        Four tests per decomposed regressor, because the literature uses
        four and they do not ask the same question:

        ``longrun_gamma``
            :math:`\gamma^{+} = \gamma^{-}`. Standard practice, and the
            numerically stable form: a linear contrast on fitted
            coefficients.
        ``longrun_theta``
            :math:`\theta^{+} = \theta^{-}`, through the delta method on
            the ratio. Equivalent to the previous one **only when**
            :math:`\lambda \ne 0`. Both are reported so that a
            disagreement is visible instead of being settled by a
            default.
        ``shortrun_additive``
            :math:`\sum_i \omega^{+}_i = \sum_i \omega^{-}_i`: asymmetry
            in the cumulated short-run response.
        ``shortrun_strong``
            :math:`\omega^{+}_i = \omega^{-}_i` for every ``i``, imposed
            jointly — the responses must match lag by lag, not on
            average.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(variable, test)``, with ``stat``, ``pvalue``
            and a 5% ``decision``.

        Notes
        -----
        ``shortrun_strong`` needs the same number of lagged differences
        on both sides. Under unpaired orders the test has no meaning, and
        the row reports ``NaN`` rather than a number built on mismatched
        terms.
        """
        from scipy.stats import f as f_dist

        params = self.params
        cov = self._fit.cov
        df_resid = int(self._fit.df_resid)
        longrun = self.longrun_asym
        rows: list[tuple[str, str, float, float]] = []

        for base in self.asym:
            pos_level = self._level_name(f"{base}_pos")
            neg_level = self._level_name(f"{base}_neg")

            stat, pvalue = _wald(
                params, cov, {pos_level: 1.0, neg_level: -1.0}, df_resid
            )
            rows.append((base, "longrun_gamma", stat, pvalue))

            diff = float(longrun.loc[base, "difference"])
            se_diff = float(longrun.loc[base, "se_difference"])
            if se_diff > 0:
                stat = (diff / se_diff) ** 2
                rows.append(
                    (base, "longrun_theta", stat, float(f_dist.sf(stat, 1, df_resid)))
                )
            else:  # pragma: no cover - degenerate covariance
                rows.append((base, "longrun_theta", float("nan"), float("nan")))

            pos_diffs = [
                n for n in params.index if str(n).startswith(f"D.{base}_pos.L")
            ]
            neg_diffs = [
                n for n in params.index if str(n).startswith(f"D.{base}_neg.L")
            ]

            if pos_diffs and neg_diffs:
                contrast: dict[str, float] = dict.fromkeys(pos_diffs, 1.0)
                contrast.update(dict.fromkeys(neg_diffs, -1.0))
                stat, pvalue = _wald(params, cov, contrast, df_resid)
                rows.append((base, "shortrun_additive", stat, pvalue))
            else:
                rows.append((base, "shortrun_additive", float("nan"), float("nan")))

            if pos_diffs and len(pos_diffs) == len(neg_diffs):
                contrasts = [
                    {str(p): 1.0, str(n): -1.0}
                    for p, n in zip(pos_diffs, neg_diffs, strict=True)
                ]
                stat, pvalue = _wald_joint(params, cov, contrasts, df_resid)
                rows.append((base, "shortrun_strong", stat, pvalue))
            else:
                rows.append((base, "shortrun_strong", float("nan"), float("nan")))

        frame = pd.DataFrame(rows, columns=["variable", "test", "stat", "pvalue"])
        frame["decision"] = np.where(
            frame["pvalue"].isna(),
            "unavailable",
            np.where(frame["pvalue"] < 0.05, "asymmetric", "symmetric"),
        )
        return frame.set_index(["variable", "test"])

    def suggests_symmetric_model(self, alpha: float = 0.05) -> bool:
        """Whether no asymmetry was found, on any test, for any regressor.

        When this is ``True`` the extra parameters bought nothing: a
        symmetric ARDL describes the same data with fewer of them, and is
        the model to report.
        """
        pvalues = self.asymmetry_tests()["pvalue"].dropna()
        return bool(len(pvalues) > 0 and bool((pvalues >= alpha).all()))

    # ------------------------------------------------------------------
    def bounds_test(self, alpha: float = 0.05) -> NARDLBoundsResults:
        r"""Cointegration test on the NARDL error-correction model.

        The statistic is the usual overall F on the level terms. What
        differs is the **critical value**: the PSS tables assume ``k``
        regressors behaving like independent random walks, and two
        partial sums of one series do not. Reading a NARDL statistic
        against them distorts the size in both directions — 7.3% at a
        nominal 5% counting the pieces, 2.6% counting the variable —
        where a genuine two-regressor model is correctly sized at 4.8%.

        So the value used here is simulated for this null specifically
        (:mod:`pyardl.critical_values.syg2014`), and it is a **single**
        value rather than a pair: decomposing a stationary series yields
        trending pieces, so the I(0) bound describes no world the
        decomposition can produce.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level; one of 0.10, 0.05, 0.01.

        Returns
        -------
        NARDLBoundsResults
        """
        from pyardl.bounds.pss import _wald_f
        from pyardl.critical_values.syg2014 import nardl_critical_value

        n_sym = len(self.model.transformed.columns) - 2 * len(self.asym)
        if n_sym:
            raise NotImplementedError(
                "The simulated critical values cover models where every "
                f"regressor is decomposed; this one keeps {n_sym} symmetric "
                "regressor(s). Extend validation/spec17_nardl_cv.py before "
                "reading a value that was not simulated for this design."
            )
        stat = _wald_f(self._fit)
        critical = {
            a: nardl_critical_value(self.model.case, len(self.asym), a)
            for a in (0.10, 0.05, 0.01)
        }
        return NARDLBoundsResults(
            f_stat=stat,
            critical=critical,
            alpha=alpha,
            case=self.model.case,
            k_asym=len(self.asym),
            nobs=self.nobs,
        )

    # ------------------------------------------------------------------
    def dynamic_multipliers(
        self,
        h: int = 40,
        r: int = 1000,
        seed: int | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        r"""Cumulated response of ``y`` to a unit rise and to a unit fall.

        The multiplier at horizon ``h`` is the cumulated effect on ``y``
        of a one-unit permanent step in :math:`x^{+}` (resp.
        :math:`x^{-}`), obtained by recursion on the ARDL form. As ``h``
        grows, :math:`m^{+}_h \to \theta^{+}` and
        :math:`m^{-}_h \to \theta^{-}`: the multipliers are the path the
        long-run coefficients take to get there, which is what makes them
        worth plotting rather than tabulating.

        Bands come from **parameter simulation**: draw ``r`` parameter
        vectors from :math:`N(\hat\theta, \hat V)`, recompute the whole
        trajectory for each, take pointwise quantiles. They are therefore
        **pointwise, not simultaneous** — a trajectory may leave them at
        some horizon without contradicting them, and reading them as a
        band the whole path stays inside would overstate what they say.

        Parameters
        ----------
        h : int, default 40
            Longest horizon.
        r : int, default 1000
            Number of parameter draws.
        seed : int, optional
            Drawn from entropy and **recorded** in ``attrs['seed']`` when
            omitted, so a figure stays reproducible after the fact.
        alpha : float, default 0.05
            Band level; 0.05 gives a 95% pointwise band.

        Returns
        -------
        pandas.DataFrame
            Indexed by horizon, with ``m_pos``, ``m_neg``, ``difference``
            and the matching ``*_lower`` / ``*_upper`` columns. With
            several decomposed regressors the columns carry a first level
            naming the variable.
        """
        if h < 1:
            raise ValueError(f"h must be at least 1, got {h}.")
        if r < 2:
            raise ValueError(f"r must be at least 2 to form a band, got {r}.")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha}.")

        if seed is None:
            entropy = np.random.SeedSequence().entropy
            seed = int(entropy) % (2**63) if isinstance(entropy, int) else 0
        rng = np.random.default_rng(seed)

        params = np.asarray(self._ardl_res._params, dtype=np.float64)
        cov = np.asarray(self._ardl_res._cov_params, dtype=np.float64)
        names = [str(n) for n in self._ardl_res._param_names]
        y_prefix = _autoregressive_prefix(names)

        draws = rng.multivariate_normal(params, cov, size=r)
        frames = []
        for base in self.asym:
            point: dict[str, npt.NDArray[np.float64]] = {}
            paths: dict[str, npt.NDArray[np.float64]] = {}
            for side in ("pos", "neg"):
                column = f"{base}_{side}"
                point[side] = _multiplier_path(params, names, column, h, y_prefix)[0]
                paths[side] = _multiplier_path(draws, names, column, h, y_prefix)
            diff_draws = paths["pos"] - paths["neg"]
            block = pd.DataFrame(
                {
                    "m_pos": point["pos"],
                    "m_pos_lower": np.quantile(paths["pos"], alpha / 2, axis=0),
                    "m_pos_upper": np.quantile(paths["pos"], 1 - alpha / 2, axis=0),
                    "m_neg": point["neg"],
                    "m_neg_lower": np.quantile(paths["neg"], alpha / 2, axis=0),
                    "m_neg_upper": np.quantile(paths["neg"], 1 - alpha / 2, axis=0),
                    "difference": point["pos"] - point["neg"],
                    "difference_lower": np.quantile(diff_draws, alpha / 2, axis=0),
                    "difference_upper": np.quantile(diff_draws, 1 - alpha / 2, axis=0),
                },
                index=pd.RangeIndex(h + 1, name="horizon"),
            )
            if len(self.asym) > 1:
                block.columns = pd.MultiIndex.from_product([[base], block.columns])
            frames.append(block)

        out = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
        out.attrs["seed"] = int(seed)
        out.attrs["r"] = int(r)
        out.attrs["alpha"] = float(alpha)
        return out

    def plot_multipliers(
        self,
        h: int = 40,
        r: int = 1000,
        seed: int | None = None,
        alpha: float = 0.05,
        variable: str | None = None,
    ) -> Any:
        r"""Plot the asymmetric dynamic multipliers.

        The canonical figure of the NARDL literature: the two cumulated
        responses with their bands, and beneath them the difference
        :math:`m^{+} - m^{-}` with its own band.

        The lower panel is the one that answers the question. Two curves
        that look far apart may still have overlapping bands; asymmetry
        shows when the band on the *difference* excludes zero, not when
        the eye judges the gap.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ImportError
            If matplotlib, an optional dependency, is not installed.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Plotting requires matplotlib, an optional dependency. "
                "Install it with: pip install pyardl[plot]"
            ) from exc

        base = self.asym[0] if variable is None else str(variable)
        if base not in self.asym:
            raise ValueError(
                f"{base!r} was not decomposed; asymmetric variables are "
                f"{list(self.asym)}."
            )
        table = self.dynamic_multipliers(h=h, r=r, seed=seed, alpha=alpha)
        if isinstance(table.columns, pd.MultiIndex):
            table = table[base]

        fig, (top, bottom) = plt.subplots(
            2, 1, figsize=(7.5, 7.0), sharex=True, height_ratios=[2, 1]
        )
        horizon = table.index.to_numpy()

        top.plot(horizon, table["m_pos"], color="#2A2F86", label="$m^{+}$")
        top.fill_between(
            horizon,
            table["m_pos_lower"],
            table["m_pos_upper"],
            color="#2A2F86",
            alpha=0.15,
        )
        top.plot(
            horizon,
            table["m_neg"],
            color="#B4451F",
            linestyle="--",
            label="$m^{-}$",
        )
        top.fill_between(
            horizon,
            table["m_neg_lower"],
            table["m_neg_upper"],
            color="#B4451F",
            alpha=0.15,
        )
        top.axhline(0.0, color="black", linewidth=0.8)
        top.set_ylabel("cumulated response")
        top.set_title(f"Dynamic multipliers - {base}")
        top.legend(frameon=False)

        bottom.plot(horizon, table["difference"], color="#333333")
        bottom.fill_between(
            horizon,
            table["difference_lower"],
            table["difference_upper"],
            color="#333333",
            alpha=0.15,
        )
        bottom.axhline(0.0, color="black", linewidth=0.8)
        bottom.set_ylabel("$m^{+} - m^{-}$")
        bottom.set_xlabel("horizon")
        # Les horizons sont des periodes entieres : des graduations a
        # 2.5 suggereraient des dates qui n'existent pas.
        bottom.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Readable report of the fit, the long run and the symmetry tests."""
        longrun = self.longrun_asym
        tests = self.asymmetry_tests()
        lines = [
            f"NARDL (Shin, Yu & Greenwood-Nimmo 2014) - case {self.model.case}, "
            f"asymmetric: {', '.join(self.asym)}, threshold={self.threshold}",
            f"  observations: {self.nobs}   lambda = {self.lam:.4f}",
            "",
            "  long-run coefficients",
            f"  {'variable':>10}{'theta+':>10}{'se':>9}{'theta-':>10}{'se':>9}"
            f"{'difference':>12}{'se':>9}",
        ]
        for base, row in longrun.iterrows():
            lines.append(
                f"  {base:>10}{row['theta_pos']:>10.4f}{row['se_pos']:>9.4f}"
                f"{row['theta_neg']:>10.4f}{row['se_neg']:>9.4f}"
                f"{row['difference']:>12.4f}{row['se_difference']:>9.4f}"
            )
        lines += [
            "",
            "  symmetry tests",
            f"  {'variable':>10}{'test':>20}{'stat':>10}{'pvalue':>10}  verdict (5%)",
        ]
        for (base, test), row in tests.iterrows():
            lines.append(
                f"  {base:>10}{test:>20}{row['stat']:>10.4f}{row['pvalue']:>10.4f}"
                f"  {row['decision']}"
            )
        if self.suggests_symmetric_model():
            lines += [
                "",
                "  No asymmetry was rejected. The extra parameters bought "
                "nothing here:",
                "  a symmetric ARDL describes the same data with fewer of them.",
            ]
        return "\n".join(lines)


@dataclass(frozen=True)
class NARDLBoundsResults:
    """Outcome of the bounds test on a NARDL.

    Attributes
    ----------
    f_stat : float
        Overall F on the level terms.
    critical : dict
        Simulated critical values at the 10%, 5% and 1% levels. A single
        value per level, not a pair — see :meth:`NARDLResults.bounds_test`.
    decision : str
        ``'cointegration'`` or ``'no_cointegration'``. There is no
        inconclusive state here: with one critical value instead of two
        bounds, there is no zone to fall between.
    """

    f_stat: float
    critical: dict[float, float]
    alpha: float
    case: int
    k_asym: int
    nobs: int

    @property
    def decision(self) -> str:
        return (
            "cointegration"
            if self.f_stat > self.critical[self.alpha]
            else "no_cointegration"
        )

    def summary(self) -> str:
        """Readable report of the test."""
        lines = [
            f"NARDL bounds test - case {self.case}, {self.k_asym} decomposed "
            f"variable(s), {self.nobs} observations",
            "  critical values simulated for the decomposed null "
            "(pyardl.critical_values.syg2014)",
            "",
            f"  F = {self.f_stat:.4f}   decision ({self.alpha:.0%}): {self.decision}",
            "",
            f"  {'alpha':>7}{'critical':>12}",
        ]
        for a in (0.10, 0.05, 0.01):
            lines.append(f"  {a:>7}{self.critical[a]:>12.4f}")
        return "\n".join(lines)


class NARDL:
    r"""Nonlinear ARDL with asymmetric partial sums.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like
        Regressors. A :class:`pandas.DataFrame` keeps its column names,
        which is how ``asym`` refers to them.
    asym : sequence of str, optional
        Which regressors to decompose. Defaults to **all** of them.
    order : tuple
        ``(p, q)`` as for :class:`~pyardl.core.ardl.ARDL`. ``q`` may be a
        dict keyed by *transformed* column name (``oil_pos``,
        ``oil_neg``) when the two sides take different lag orders.
    case : int, default 3
        Deterministic case, in the PSS numbering.
    threshold : float or {'mean'}, default 0.0
        Threshold of the decomposition. Anything other than zero
        introduces a linear drift; see :mod:`pyardl.nardl.decompose`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = np.cumsum(rng.normal(size=n))
    >>> y = np.zeros(n)
    >>> for t in range(1, n):
    ...     y[t] = 0.6 * y[t - 1] + 0.5 * x[t] + rng.normal(scale=0.3)
    >>> res = NARDL(pd.Series(y, name="y"), pd.DataFrame({"x": x}),
    ...             order=(1, 1)).fit()
    >>> res.asym
    ('x',)
    >>> sorted(res.model.transformed.columns)
    ['x_neg', 'x_pos']
    """

    def __init__(
        self,
        y: npt.ArrayLike,
        x: npt.ArrayLike,
        asym: Sequence[str] | None = None,
        order: tuple[int, int | dict[str, int]] = (1, 1),
        case: int = 3,
        threshold: Threshold = 0.0,
    ) -> None:
        from pyardl.utils import check_series

        y_arr, x_arr, index, y_name, x_names = check_series(y, x)
        if x_arr is None:
            raise ValueError("A NARDL needs at least one regressor to decompose.")
        if case not in (1, 2, 3, 4, 5):
            raise ValueError(f"case must be 1..5 (PSS numbering), got {case}.")

        self._y_series = pd.Series(y_arr, index=index, name=y_name)
        self.case = int(case)
        self.threshold: Threshold = threshold

        chosen = tuple(x_names) if asym is None else tuple(str(a) for a in asym)
        unknown = [a for a in chosen if a not in x_names]
        if unknown:
            raise ValueError(
                f"asym names {unknown} are not regressors; available: {list(x_names)}."
            )
        if not chosen:
            raise ValueError(
                "asym is empty: a NARDL with nothing decomposed is an ARDL. "
                "Use pyardl.ARDL instead."
            )
        self.asym = chosen

        columns: dict[str, pd.Series] = {}
        with warnings.catch_warnings():
            # A non-zero threshold warns once per column; one is enough to
            # make the point, and k of them would bury it.
            warnings.simplefilter("once", PyardlMethodologyWarning)
            for j, name in enumerate(x_names):
                series = pd.Series(x_arr[:, j], index=index, name=name)
                if name in chosen:
                    pos, neg = partial_sums(series, threshold=threshold)
                    columns[str(pos.name)] = pos
                    columns[str(neg.name)] = neg
                else:
                    columns[name] = series
        self.transformed = pd.DataFrame(columns, index=index)

        p, q = order
        if isinstance(q, dict):
            missing = [c for c in self.transformed.columns if c not in q]
            if missing:
                raise ValueError(
                    f"order dict is missing the transformed columns {missing}. "
                    "Decomposed regressors are named <name>_pos and <name>_neg."
                )
            q_map = {str(c): int(q[str(c)]) for c in self.transformed.columns}
        else:
            q_map = {str(c): int(q) for c in self.transformed.columns}
        self.p = int(p)
        self.q_map = q_map

    def fit(self) -> NARDLResults:
        """Estimate the model and return the results object.

        Two views of the *same* fit are built: the ARDL parameterisation,
        which the multipliers recurse on, and the error-correction one,
        which the Wald tests and the bounds test read. They are
        reparameterisations of one regression, and a test asserts that
        their residuals coincide — otherwise the multipliers and the
        tests could quietly describe different models.
        """
        from pyardl.bounds.pss import _estimate_uecm
        from pyardl.core.ardl import ARDL

        det = {1: "none", 2: "const", 3: "const", 4: "trend", 5: "trend"}[self.case]
        ardl_res = ARDL(
            self._y_series,
            self.transformed,
            order=(self.p, dict(self.q_map)),
            det=det,  # type: ignore[arg-type]
        ).fit()

        names = tuple(str(c) for c in self.transformed.columns)
        fit = _estimate_uecm(
            self._y_series.to_numpy(dtype=np.float64),
            self.transformed.to_numpy(dtype=np.float64),
            names,
            str(self._y_series.name),
            self.p,
            tuple(self.q_map[name] for name in names),
            self.case,
        )
        return NARDLResults(
            model=self,
            _fit=fit,
            _ardl_res=ardl_res,
            asym=self.asym,
            threshold=self.threshold,
        )
