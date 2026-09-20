r"""Pedroni (1999, 2004) and Westerlund (2007): panel cointegration tests.

CS-ARDL/CS-DL (:mod:`pyardl.panel.csardl`) and MG/PMG
(:mod:`pyardl.panel.mg`, :mod:`pyardl.panel.pmg`) **estimate** a
panel long run, implicitly assuming it exists. Nothing so far **tests**
the panel-cointegration hypothesis itself before estimating — the panel
analogue of the bounds test
(:func:`pyardl.bounds.bounds_test`) or Engle-Granger
(:func:`pyardl.cointegration.engle_granger.engle_granger`) is missing.

Two families, with **opposite** null hypotheses, both implemented —
their results are not meant to coincide mechanically, and comparing
them is a diagnostic in its own right, not a redundancy:

- **Pedroni**: :math:`H_0` = no cointegration (a residual unit root for
  every individual) — the panel analogue of Engle-Granger, with a
  heterogeneous first-stage :math:`\beta_i` (unlike a pooled panel
  regression — consistent with the project's Mean-Group philosophy).
- **Westerlund**: :math:`H_0` = no cointegration, built on the
  **existence of an error-correction term** rather than a residual unit
  root — closer in spirit to the ECM t-test (Banerjee-Dolado-Mestre)
  transposed to a panel.

Scope of this version
----------------------
Pedroni's original seven statistics are standardised by theoretical
moments tabulated in Pedroni (1999, 2004) by number of regressors and
deterministic case; this project does not hold that table with a
verified provenance (CLAUDE.md rule 9), so — as for Gregory-Hansen and
Enders-Siklos — critical values here come **only from a bootstrap**,
which sidesteps the moments entirely. Two representative statistics are
implemented rather than all seven: ``panel_adf`` (within-dimension,
pooling residuals across every individual into one regression, common
:math:`\rho`) and ``group_adf`` (between-dimension, the cross-sectional
average of each individual's own ADF t-statistic, no common
:math:`\rho` assumed) — the two most commonly reported in applied work.
Westerlund similarly exposes ``group_tau`` and ``panel_tau`` (both
t-type statistics, unambiguous to construct); ``group_alpha`` and
``panel_alpha`` need normalisation constants from Westerlund (2007) not
reproduced here without a verified source. See ``docs/DEVIATIONS.md``.

The null DGP for the bootstrap mirrors
:mod:`pyardl.cointegration.gregory_hansen`: each individual's own
``[y_i, x_i]`` system is regenerated independently (a VAR-in-differences
fitted per individual, no cointegrating relationship by construction),
never a resampling of a derived residual in isolation — the same lesson
:mod:`pyardl.cointegration.enders_siklos` had to relearn (see
``docs/QUESTIONS.md``).

References
----------
.. [1] Pedroni, P. (1999). Critical values for cointegration tests in
       heterogeneous panels with multiple regressors. *Oxford Bulletin
       of Economics and Statistics*, 61(S1), 653-670.
.. [2] Pedroni, P. (2004). Panel cointegration: asymptotic and finite
       sample properties of pooled time series tests with an
       application to the PPP hypothesis. *Econometric Theory*, 20(3),
       597-625.
.. [3] Westerlund, J. (2007). Testing for error correction in panel
       data. *Oxford Bulletin of Economics and Statistics*, 69(6),
       709-748.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from pyardl.bootstrap.dgp import _fit_marginal_var
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.panel.container import PanelData, panel_from_frame
from pyardl.unitroot.gls import adf_regression, select_lags

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]

__all__ = [
    "PedroniResults",
    "WesterlundResults",
    "pedroni",
    "westerlund",
]

Det = Literal["const", "trend"]
CVSource = Literal["bootstrap"]


def _static_residuals(y: FloatArray, x: FloatArray, det: Det) -> FloatArray:
    """Step-one heterogeneous OLS residuals for one individual."""
    n = y.shape[0]
    cols = [x, np.ones((n, 1))]
    if det == "trend":
        cols.append(np.arange(1, n + 1, dtype=np.float64)[:, None])
    design = np.column_stack(cols)
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return np.asarray(y - design @ beta, dtype=np.float64)


def _panel_adf(residuals: Sequence[FloatArray], max_lags: int | None, ic: str) -> float:
    """Pooled ADF t-stat: one rho common to every individual's residuals."""
    targets = []
    levels = []
    lag_cols: list[list[FloatArray]] = []
    max_p = 0
    per_unit_lags = []
    for u in residuals:
        lags, _ = select_lags(u, method=ic, max_lags=max_lags)  # type: ignore[arg-type]
        per_unit_lags.append(lags)
        max_p = max(max_p, lags)
    for u, lags in zip(residuals, per_unit_lags, strict=True):
        du = np.diff(u)
        n = du.size
        if n - lags < lags + 2:
            continue
        targets.append(du[lags:])
        levels.append(u[lags:-1])
        cols = [du[lags - j : n - j] for j in range(1, lags + 1)]
        # Pad every individual to the common max lag with zero columns so
        # the pooled design has the same width for every row.
        while len(cols) < max_p:
            cols.append(np.zeros_like(du[lags:]))
        lag_cols.append(cols)

    target = np.concatenate(targets)
    level = np.concatenate(levels)
    if max_p > 0:
        lag_design = np.vstack(
            [
                np.column_stack(cols) if cols else np.empty((t.size, 0))
                for cols, t in zip(lag_cols, targets, strict=True)
            ]
        )
        design = np.column_stack([level, lag_design])
    else:
        design = level[:, None]

    coefs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coefs
    n_obs, k = design.shape
    sigma2 = float(resid @ resid) / (n_obs - k)
    xtx_inv = np.linalg.inv(design.T @ design)
    se_rho = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    return float(coefs[0]) / se_rho


def _group_adf(residuals: Sequence[FloatArray], max_lags: int | None, ic: str) -> float:
    """Cross-sectional average of each individual's own ADF t-statistic."""
    tstats = []
    for u in residuals:
        lags, _ = select_lags(u, method=ic, max_lags=max_lags)  # type: ignore[arg-type]
        fit = adf_regression(u, lags)
        tstats.append(fit.tstat)
    return float(np.mean(tstats))


@dataclass(frozen=True)
class PedroniResults:
    """Outcome of a Pedroni panel cointegration test.

    Attributes
    ----------
    panel_adf : float
        Within-dimension statistic: pooled ADF t-stat, common
        :math:`\\rho` across individuals.
    group_adf : float
        Between-dimension statistic: cross-sectional average of each
        individual's own ADF t-statistic, no common :math:`\\rho`.
    critical_values : dict
        ``{'panel_adf': {...}, 'group_adf': {...}}``, level to critical
        value, from ``cv_source``.
    n_units : int
    trend : str
    cv_source : str
    n_boot : int
    seed : int or None
    """

    panel_adf: float
    group_adf: float
    critical_values: dict[str, dict[float, float]]
    n_units: int
    trend: Det
    cv_source: CVSource
    n_boot: int
    seed: int | None

    def decision(self, stat: str = "panel_adf", alpha: float = 0.05) -> str:
        """``'cointegration'`` or ``'no_cointegration'`` at ``alpha``, left-tailed."""
        if stat not in ("panel_adf", "group_adf"):
            raise ValueError(f"stat={stat!r} must be 'panel_adf' or 'group_adf'.")
        value = self.panel_adf if stat == "panel_adf" else self.group_adf
        cv = self.critical_values[stat][alpha]
        return "cointegration" if value < cv else "no_cointegration"

    def summary(self) -> str:
        """Readable report of the test."""
        lines = [
            f"Pedroni panel cointegration test (1999, 2004) - {self.n_units} "
            f"units, trend={self.trend}, {self.n_boot} bootstrap reps",
            "  H0: no cointegration (residual unit root, every individual)",
        ]
        pairs = (("panel_adf", self.panel_adf), ("group_adf", self.group_adf))
        for stat, value in pairs:
            cv = "  ".join(
                f"{int(a * 100)}%: {v:.4f}"
                for a, v in sorted(self.critical_values[stat].items())
            )
            lines.append(
                f"  {stat} = {value:.4f}   decision (5%): {self.decision(stat, 0.05)}"
                f"\n    critical values (left tail)   {cv}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class WesterlundResults:
    """Outcome of a Westerlund panel error-correction test.

    Attributes
    ----------
    group_tau : float
        Group-mean statistic: cross-sectional average of each
        individual's own t-statistic on the error-correction term.
    panel_tau : float
        Panel (pooled) statistic: pooled t-statistic, common adjustment
        speed across individuals.
    critical_values : dict
        ``{'group_tau': {...}, 'panel_tau': {...}}``.
    n_units : int
    cv_source : str
    n_boot : int
    seed : int or None
    cd_warning : str or None
        Set when the caller supplied a cross-sectional dependence test
        result (:func:`pyardl.panel.cd_test`) that rejected independence
        (spec 37 §2.3): classical Pedroni/Westerlund assume independent
        individuals, and this test was requested anyway.
    """

    group_tau: float
    panel_tau: float
    critical_values: dict[str, dict[float, float]]
    n_units: int
    cv_source: CVSource
    n_boot: int
    seed: int | None
    cd_warning: str | None = None

    def decision(self, stat: str = "group_tau", alpha: float = 0.05) -> str:
        """``'cointegration'`` or ``'no_cointegration'`` at ``alpha``, left-tailed."""
        if stat not in ("group_tau", "panel_tau"):
            raise ValueError(f"stat={stat!r} must be 'group_tau' or 'panel_tau'.")
        value = self.group_tau if stat == "group_tau" else self.panel_tau
        cv = self.critical_values[stat][alpha]
        return "cointegration" if value < cv else "no_cointegration"

    def summary(self) -> str:
        """Readable report of the test."""
        lines = [
            f"Westerlund panel error-correction test (2007) - {self.n_units} "
            f"units, {self.n_boot} bootstrap reps",
            "  H0: no error correction (no cointegration)",
        ]
        if self.cd_warning:
            lines.append(f"  WARNING {self.cd_warning}")
        pairs = (("group_tau", self.group_tau), ("panel_tau", self.panel_tau))
        for stat, value in pairs:
            cv = "  ".join(
                f"{int(a * 100)}%: {v:.4f}"
                for a, v in sorted(self.critical_values[stat].items())
            )
            lines.append(
                f"  {stat} = {value:.4f}   decision (5%): {self.decision(stat, 0.05)}"
                f"\n    critical values (left tail)   {cv}"
            )
        return "\n".join(lines)


def _bootstrap_units(
    rng: np.random.Generator, panel: PanelData, var_order: int, n: int
) -> list[tuple[FloatArray, FloatArray]]:
    """One replicate: regenerate every individual's [y_i, x_i] under H0.

    Each individual's own null DGP is a VAR-in-differences fitted to its
    OWN observed [y_i, x_i] (:func:`pyardl.bootstrap.dgp._fit_marginal_var`,
    reused unchanged) — independent I(1) series by construction, no
    cointegration. Individuals stay cross-sectionally independent in
    this null, matching the classical (non-CD-robust) assumption both
    tests are built on.
    """
    out = []
    for unit in panel:
        y_arr = unit.y.to_numpy()
        x_arr = unit.x.to_numpy()
        stacked = np.column_stack([y_arr, x_arr])
        d_stacked = np.diff(stacked, axis=0)
        const, ar, resid = _fit_marginal_var(d_stacked, var_order)
        resid = resid - resid.mean(axis=0)
        k = stacked.shape[1]
        draws = rng.integers(0, resid.shape[0], size=n - 1)
        innovations = resid[draws]
        dz = np.empty((n - 1, k), dtype=np.float64)
        history = (
            np.tile(stacked[0], (var_order, 1)) if var_order > 0 else np.empty((0, k))
        )
        for t in range(n - 1):
            val = const.copy()
            for i in range(var_order):
                val = val + ar[i] @ history[-(i + 1)]
            val = val + innovations[t]
            dz[t] = val
            if var_order > 0:
                history = np.vstack([history, val])[1:]
        z0 = stacked[0]
        z_path = np.vstack([z0, z0 + np.cumsum(dz, axis=0)])
        out.append((z_path[:, 0], z_path[:, 1:]))
    return out


def pedroni(
    df: pd.DataFrame,
    y: str,
    X: Sequence[str],
    id: str,  # noqa: A002 - matches the spec's public API
    time: str,
    det: Det = "const",
    max_lags: int | None = None,
    ic: str = "aic",
    var_order: int = 1,
    n_boot: int = 499,
    seed: int | None = None,
    min_obs: int = 15,
) -> PedroniResults:
    r"""Pedroni (1999, 2004) panel cointegration test.

    Parameters
    ----------
    df, y, X, id, time, min_obs : see :func:`pyardl.panel.panel_from_frame`.
    det : {'const', 'trend'}, default 'const'
        Deterministic terms of the heterogeneous step-one regression.
    max_lags, ic : passed to
        :func:`~pyardl.unitroot.gls.select_lags` for each individual's
        ADF lag order.
    var_order : int, default 1
        Lag order of each individual's bootstrap null VAR-in-differences.
    n_boot : int, default 499
        Bootstrap replications for the critical values.
    seed : int, optional

    Returns
    -------
    PedroniResults

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.panel import pedroni
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(10):
    ...     x = np.cumsum(rng.standard_normal(100))
    ...     u = np.zeros(100)
    ...     for t in range(1, 100):
    ...         u[t] = -0.5 * u[t - 1] + rng.standard_normal() * 0.3
    ...     y = 1.5 * x + u
    ...     rows.append(pd.DataFrame({"id": i, "t": np.arange(100), "y": y, "x": x}))
    >>> panel = pd.concat(rows, ignore_index=True)
    >>> res = pedroni(panel, y="y", X=["x"], id="id", time="t", n_boot=99, seed=0)
    >>> res.decision("panel_adf", 0.05)
    'cointegration'
    """
    if det not in ("const", "trend"):
        raise ValueError(f"det={det!r} must be 'const' or 'trend'.")
    panel = panel_from_frame(
        df, y=y, x=list(X), id_col=id, time_col=time, min_obs=min_obs
    )
    if panel.n_units < 2:
        raise ValueError("Pedroni needs at least two individuals.")

    residuals = [
        _static_residuals(unit.y.to_numpy(), unit.x.to_numpy(), det) for unit in panel
    ]
    panel_stat = _panel_adf(residuals, max_lags, ic)
    group_stat = _group_adf(residuals, max_lags, ic)

    rng = np.random.default_rng(seed)
    n = int(panel.sample_sizes.min())
    panel_boot = np.empty(n_boot, dtype=np.float64)
    group_boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        regen = _bootstrap_units(rng, panel, var_order, n)
        boot_resid = [_static_residuals(y_b, x_b, det) for y_b, x_b in regen]
        panel_boot[b] = _panel_adf(boot_resid, max_lags, ic)
        group_boot[b] = _group_adf(boot_resid, max_lags, ic)

    cvs = {
        "panel_adf": {a: float(np.quantile(panel_boot, a)) for a in (0.01, 0.05, 0.10)},
        "group_adf": {a: float(np.quantile(group_boot, a)) for a in (0.01, 0.05, 0.10)},
    }

    return PedroniResults(
        panel_adf=panel_stat,
        group_adf=group_stat,
        critical_values=cvs,
        n_units=panel.n_units,
        trend=det,
        cv_source="bootstrap",
        n_boot=n_boot,
        seed=seed,
    )


def _ecm_tstat(y: FloatArray, u: FloatArray) -> float:
    """t-statistic on the error-correction coefficient of one individual's ECM."""
    dy = np.diff(y)
    lagged = u[:-1]
    design = lagged[:, None]
    coefs, _, _, _ = np.linalg.lstsq(design, dy, rcond=None)
    resid = dy - design @ coefs
    n_obs, k = design.shape
    sigma2 = float(resid @ resid) / (n_obs - k)
    xtx_inv = np.linalg.inv(design.T @ design)
    se = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    return float(coefs[0]) / se if se > 0 else 0.0


def _panel_ecm_tstat(units: Sequence[tuple[FloatArray, FloatArray]]) -> float:
    """Pooled t-statistic: one adjustment speed common to every individual."""
    targets = []
    levels = []
    for dy, lagged in units:
        targets.append(dy)
        levels.append(lagged)
    target = np.concatenate(targets)
    design = np.concatenate(levels)[:, None]
    coefs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coefs
    n_obs, k = design.shape
    sigma2 = float(resid @ resid) / (n_obs - k)
    xtx_inv = np.linalg.inv(design.T @ design)
    se = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    return float(coefs[0]) / se if se > 0 else 0.0


def westerlund(
    df: pd.DataFrame,
    y: str,
    X: Sequence[str],
    id: str,  # noqa: A002 - matches the spec's public API
    time: str,
    det: Det = "const",
    var_order: int = 1,
    n_boot: int = 499,
    seed: int | None = None,
    min_obs: int = 15,
    cd_pvalue: float | None = None,
) -> WesterlundResults:
    r"""Westerlund (2007) panel error-correction test.

    Parameters
    ----------
    df, y, X, id, time, min_obs : see :func:`pyardl.panel.panel_from_frame`.
    det : {'const', 'trend'}, default 'const'
        Deterministic terms of the first-stage long-run regression used
        to build each individual's error-correction term.
    var_order, n_boot, seed : as in :func:`pedroni`.
    cd_pvalue : float, optional
        p-value of a prior :func:`pyardl.panel.cd_test` on the same
        panel. When given and below 0.05, ``cd_warning`` is set: spec 37
        §2.3 requires flagging classical panel cointegration tests
        applied where cross-sectional independence was rejected, rather
        than letting the violated assumption pass silently.

    Returns
    -------
    WesterlundResults

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from pyardl.panel import westerlund
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(10):
    ...     x = np.cumsum(rng.standard_normal(100))
    ...     u = np.zeros(100)
    ...     for t in range(1, 100):
    ...         u[t] = -0.5 * u[t - 1] + rng.standard_normal() * 0.3
    ...     y = 1.5 * x + u
    ...     rows.append(pd.DataFrame({"id": i, "t": np.arange(100), "y": y, "x": x}))
    >>> panel = pd.concat(rows, ignore_index=True)
    >>> res = westerlund(panel, y="y", X=["x"], id="id", time="t", n_boot=99, seed=0)
    >>> res.decision("group_tau", 0.05)
    'cointegration'
    """
    if det not in ("const", "trend"):
        raise ValueError(f"det={det!r} must be 'const' or 'trend'.")
    panel = panel_from_frame(
        df, y=y, x=list(X), id_col=id, time_col=time, min_obs=min_obs
    )
    if panel.n_units < 2:
        raise ValueError("Westerlund needs at least two individuals.")

    def _fit_all(
        units_yx: Sequence[tuple[FloatArray, FloatArray]],
    ) -> tuple[float, float]:
        tstats = []
        ecm_parts = []
        for y_arr, x_arr in units_yx:
            u = _static_residuals(y_arr, x_arr, det)
            tstats.append(_ecm_tstat(y_arr, u))
            dy = np.diff(y_arr)
            ecm_parts.append((dy, u[:-1]))
        group = float(np.mean(tstats))
        panel_stat = _panel_ecm_tstat(ecm_parts)
        return group, panel_stat

    observed_units = [(unit.y.to_numpy(), unit.x.to_numpy()) for unit in panel]
    group_stat, panel_stat = _fit_all(observed_units)

    rng = np.random.default_rng(seed)
    n = int(panel.sample_sizes.min())
    group_boot = np.empty(n_boot, dtype=np.float64)
    panel_boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        regen = _bootstrap_units(rng, panel, var_order, n)
        group_boot[b], panel_boot[b] = _fit_all(regen)

    cvs = {
        "group_tau": {a: float(np.quantile(group_boot, a)) for a in (0.01, 0.05, 0.10)},
        "panel_tau": {a: float(np.quantile(panel_boot, a)) for a in (0.01, 0.05, 0.10)},
    }

    cd_warning = None
    if cd_pvalue is not None and cd_pvalue < 0.05:
        cd_warning = (
            f"cross-sectional dependence test rejected independence "
            f"(p={cd_pvalue:.4f}); classical Westerlund assumes "
            "independent individuals — see pyardl.panel.cd_test and spec 37 §2.3."
        )
        warnings.warn(cd_warning, PyardlMethodologyWarning, stacklevel=2)

    return WesterlundResults(
        group_tau=group_stat,
        panel_tau=panel_stat,
        critical_values=cvs,
        n_units=panel.n_units,
        cv_source="bootstrap",
        n_boot=n_boot,
        seed=seed,
        cd_warning=cd_warning,
    )
