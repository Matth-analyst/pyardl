r"""Unified cointegration analysis: NARDL x Fourier x bootstrap.

The 2026 state of the art (Roudane's Stata module AARDL; the CRAN
package fbardl) combines three methodological advances — the Fourier
approximation of smooth breaks, bootstrap inference, and the asymmetric
partial-sum decomposition — inside the three-test framework of Sam,
McNown & Goh (2019). This module is the orchestration layer: it owns
**no estimator and no distribution of its own**. Every cell of the
combination matrix is delegated to, or assembled from, bricks validated
by their own specs (10-20), and the one genuinely new responsibility is
choosing the *right critical values for each cell*, which is exactly
where applied work goes wrong.

The rules, centralised in :func:`resolve_critical_values`:

- **Fourier terms present** => never the PSS/KS tables alone. The
  frequency was searched on the data, so the null distribution must be
  simulated with the search inside the loop (the Davies problem,
  measured in OBS-15: selection multiplies the size of a nominal 5% test
  to 24.6%).
- **Asymmetric decomposition present** => never the tables at face
  value either: two partial sums of one series are not two independent
  I(1) regressors, and both conventions for counting them distort the
  size (OBS-13/OBS-14). The tabulated route uses the simulated
  Shin-Yu-Greenwood-Nimmo values of spec 17.
- **Bootstrap** is available for every cell and is the recommended
  route for any combination without a validated tabulated source. The
  null DGP follows McNown, Sam & Goh (2018): the conditional equation
  under the joint null plus a marginal VAR, one joint null for all
  three statistics (OBS-8). Fourier terms, being deterministic, stay in
  the null model and in every regenerated path; the decomposition is
  re-applied to each regenerated path, because partial sums are a
  function of the data, not data themselves.

References
----------
.. [1] Sam, C. Y., McNown, R. & Goh, S. K. (2019). An augmented
       autoregressive distributed lag bounds test for cointegration.
       *Economic Modelling*, 80, 130-141.
.. [2] McNown, R., Sam, C. Y. & Goh, S. K. (2018). Bootstrapping the
       autoregressive distributed lag test for cointegration.
       *Applied Economics*, 50(13), 1509-1521.
.. [3] Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). Modelling
       asymmetric cointegration and dynamic multipliers in a nonlinear
       ARDL framework. In *Festschrift in Honor of Peter Schmidt*.
.. [4] Banerjee, P., Arcabic, V. & Lee, H. (2017). Fourier ADL
       cointegration test to approximate smooth breaks. *Economic
       Modelling*, 67, 114-124.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pyardl.bootstrap.resample import ResampleScheme
from pyardl.bounds.classification import classify
from pyardl.exceptions import PyardlMethodologyWarning

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

__all__ = ["UnifiedResults", "cointegration_analysis", "resolve_critical_values"]

_ALPHAS = (0.10, 0.05, 0.01)
_CHUNK = 200

Inference = Literal["bounds", "bootstrap"]


def resolve_critical_values(
    asym: bool, fourier: bool, inference: str
) -> tuple[str, str]:
    """Name the critical-value source a cell of the matrix must use.

    Parameters
    ----------
    asym : bool
        Whether any regressor is decomposed into partial sums.
    fourier : bool
        Whether Fourier terms enter the deterministic part.
    inference : {'bounds', 'bootstrap'}
        Requested inference route.

    Returns
    -------
    source : str
        Machine-readable key: ``'pss_tables'``, ``'simulated_fourier'``,
        ``'simulated_syg'`` or ``'bootstrap'``.
    reason : str
        One sentence saying why, quotable in a methods section.

    Raises
    ------
    ValueError
        For the one cell with no validated non-bootstrap source:
        asymmetric decomposition combined with Fourier terms under
        ``inference='bounds'``. Raising is the design: substituting a
        neighbouring table silently is how sizes get distorted.

    Examples
    --------
    >>> resolve_critical_values(False, False, "bounds")[0]
    'pss_tables'
    >>> resolve_critical_values(True, True, "bootstrap")[0]
    'bootstrap'
    """
    if inference == "bootstrap":
        return (
            "bootstrap",
            "Critical values bootstrapped under the joint null DGP; valid "
            "for every combination, and the only validated source for this "
            "one."
            if (asym and fourier)
            else "Critical values bootstrapped under the joint null DGP "
            "(McNown, Sam & Goh 2018).",
        )
    if asym and fourier:
        raise ValueError(
            "No tabulated or pre-simulated critical values exist for the "
            "combination of an asymmetric decomposition with Fourier terms: "
            "the partial sums change the effective regressor count and the "
            "frequency search changes the null distribution, and no "
            "published table covers both at once. Use "
            "inference='bootstrap', which simulates the null of this exact "
            "specification."
        )
    if fourier:
        return (
            "simulated_fourier",
            "Critical values simulated with the frequency search inside the "
            "loop; the PSS/KS tables do not cover a searched frequency "
            "(Davies problem, OBS-15).",
        )
    if asym:
        return (
            "simulated_syg",
            "Critical values simulated for the partial-sum null; the PSS "
            "tables assume independent I(1) regressors, which two partial "
            "sums of one series are not (OBS-13/14).",
        )
    return (
        "pss_tables",
        "Tabulated bounds apply: no searched frequency, no decomposition.",
    )


def _fourier_matrix(
    t_values: FloatArray, span: int, freq: float, fourier_k: int
) -> FloatArray:
    """Sine/cosine columns at arbitrary dates, ``fourier_terms``'s convention.

    ``t_values`` may run into the burn-in (non-positive dates): like the
    trend of the null model, the sinusoid is the same function of time
    extended backwards, which is the only choice that leaves no
    discontinuity at the join.
    """
    cols = []
    for m in range(1, fourier_k + 1):
        w = 2.0 * np.pi * freq * m / span
        cols.append(np.sin(w * t_values))
        cols.append(np.cos(w * t_values))
    return np.column_stack(cols)


@dataclass(frozen=True)
class _BootOutcome:
    """What the bootstrap engine returns for one cell."""

    frequency: float | None
    selection: pd.DataFrame | None
    f_stat: float
    t_stat: float
    f_indep_stat: float
    f_critical: dict[float, float]
    t_critical: dict[float, float]
    f_indep_critical: dict[float, float]
    f_pvalue: float
    t_pvalue: float
    f_indep_pvalue: float
    n_boot: int
    n_failed: int


@dataclass(frozen=True)
class _CellConfig:
    """One resolved cell of the combination matrix."""

    asym: tuple[str, ...]
    fourier_k: int
    freq: float | None  # None => searched on the data
    grid: tuple[float, ...]
    inference: str
    cv_source: str
    cv_reason: str

    @property
    def label(self) -> str:
        model = "nardl" if self.asym else "ardl"
        det = f"+fourier(k={self.fourier_k})" if self.fourier_k else ""
        return f"{model}{det}/{self.inference}"


@dataclass(frozen=True)
class UnifiedResults:
    """Outcome of one cell of the unified matrix.

    Fields are ``None`` when the cell's validated critical-value source
    does not cover that statistic — the tabulated NARDL route carries
    only the overall F, the simulated Fourier route only the ``t``. The
    bootstrap route carries all three; that asymmetry of coverage is a
    property of the literature, reported rather than papered over.

    Attributes
    ----------
    f_stat, t_stat, f_indep_stat : float or None
        The three statistics of the Sam-McNown-Goh framework.
    f_critical, t_critical, f_indep_critical : dict or None
        Critical values per level, from the source named by
        ``cv_source``.
    decision_f, decision_t, decision_indep : str or None
        Per-test verdicts.
    classification : str or None
        The joint classification of spec 15, available when all three
        tests could run.
    detail
        The underlying result object of the delegated brick —
        ``BoundsTestResults``, ``BootstrapBoundsResults``,
        ``FourierBoundsResults`` or ``NARDLBoundsResults`` — carrying
        everything cell-specific (estimates, diagnostics, seeds).
    """

    label: str
    cv_source: str
    cv_reason: str
    case: int
    alpha: float
    nobs: int
    order: tuple[int, dict[str, int]]
    frequency: float | None
    f_stat: float | None
    t_stat: float | None
    f_indep_stat: float | None
    f_critical: dict[float, float] | None
    t_critical: dict[float, float] | None
    f_indep_critical: dict[float, float] | None
    f_pvalue: float | None
    t_pvalue: float | None
    f_indep_pvalue: float | None
    decision_f: str | None
    decision_t: str | None
    decision_indep: str | None
    classification: str | None
    reason: str | None
    seed: int | None
    n_boot: int | None
    detail: object = field(repr=False)
    _rerun: dict[str, object] = field(repr=False, default_factory=dict)

    def summary(self) -> str:
        """Publication-style text summary of the cell."""
        p, q_map = self.order
        lines = [
            f"Unified cointegration analysis - cell {self.label}",
            f"  case {self.case}, {self.nobs} observations, "
            f"ECM({p}; {', '.join(f'{k}:{v}' for k, v in q_map.items())})",
            f"  critical values: {self.cv_source} - {self.cv_reason}",
        ]
        if self.frequency is not None:
            lines.append(f"  Fourier frequency: {self.frequency:g}")
        lines.append("")
        rows = [
            ("F_overall", self.f_stat, self.f_critical, self.decision_f),
            ("t_BDM", self.t_stat, self.t_critical, self.decision_t),
            ("F_indep", self.f_indep_stat, self.f_indep_critical, self.decision_indep),
        ]
        for name, stat, crit, dec in rows:
            if stat is None:
                lines.append(f"  {name:<10} not covered by this source")
                continue
            cv = (
                f"critical ({int(self.alpha * 100)}%) = {crit[self.alpha]:.4f}"
                if crit
                else ""
            )
            lines.append(f"  {name:<10} = {stat:8.4f}   {cv}   -> {dec}")
        lines.append("")
        if self.classification is not None:
            lines.append(f"  classification: {self.classification}")
            if self.reason:
                lines.append(f"  {self.reason}")
        else:
            lines.append(
                "  classification: unavailable - this source does not cover "
                "all three tests; use inference='bootstrap' for the full "
                "triplet."
            )
        return "\n".join(lines)

    def compare(
        self,
        cells: tuple[tuple[bool, bool], ...] = (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ),
        **overrides: Any,
    ) -> pd.DataFrame:
        """Run sibling cells on the same data and tabulate the verdicts.

        This is the robustness table applied papers publish: the same
        relationship estimated linear and asymmetric, with and without a
        smooth break, side by side. Each row is one cell.

        Parameters
        ----------
        cells : tuple of (asym, fourier) pairs
            Which cells to run. Defaults to the full 2x2 for the current
            inference route.
        **overrides
            Passed through to :func:`cointegration_analysis` (for
            instance a smaller ``n_boot`` to keep the table cheap).

        Returns
        -------
        pandas.DataFrame
            One row per cell: statistics, decisions, classification. A
            cell whose critical-value source does not exist for this
            inference route (NARDL+Fourier under ``'bounds'``) is
            reported as unavailable rather than silently skipped.
        """
        base: dict[str, Any] = dict(self._rerun)
        base.update(overrides)
        y = base.pop("y")
        x = base.pop("x")
        all_names: tuple[str, ...] = base.pop("_x_names")
        rows: list[dict[str, object]] = []
        for use_asym, use_fourier in cells:
            kwargs: dict[str, Any] = dict(base)
            kwargs["asym"] = list(all_names) if use_asym else None
            kwargs["fourier"] = {"k": 1} if use_fourier else None
            try:
                res = cointegration_analysis(y, x, **kwargs)
            except ValueError as exc:
                rows.append(
                    {
                        "cell": f"{'nardl' if use_asym else 'ardl'}"
                        f"{'+fourier' if use_fourier else ''}"
                        f"/{kwargs.get('inference', 'bounds')}",
                        "classification": f"unavailable: {exc}",
                    }
                )
                continue
            rows.append(
                {
                    "cell": res.label,
                    "F_overall": res.f_stat,
                    "t_BDM": res.t_stat,
                    "F_indep": res.f_indep_stat,
                    "decision_F": res.decision_f,
                    "decision_t": res.decision_t,
                    "decision_indep": res.decision_indep,
                    "classification": res.classification,
                }
            )
        return pd.DataFrame(rows).set_index("cell")


def _guard_searched_frequency_bootstrap(fourier_k: int, searched: bool) -> None:
    """Warn about the one combination measured to over-reject.

    Measured on 2000 replications under a true null (spec21_size.py and
    spec21_fourier_arbitration.py): searching the frequency *and*
    carrying the fitted Fourier component into the estimated null DGP
    gives 7.1% (F) and 7.5% (t) at a nominal 5%, standard error 0.49
    point. Fixing the frequency brings both back inside the band
    (5.50/5.35 at f=1, 4.85/4.85 at f=2), and so does removing the
    component from the null model.

    Neither half causes it alone. The bootstrap re-runs the search in
    every replication, so the search itself is calibrated; what is not
    is that the null DGP was estimated *conditional on the frequency
    the search had already won*. See OBS-19.
    """
    if fourier_k and searched:
        warnings.warn(
            "Bootstrapping a searched Fourier frequency over-rejects: "
            "measured 7.1% (F) and 7.5% (t) at a nominal 5% on 2000 "
            "replications (standard error 0.49 point). The search is "
            "re-run in every replication, but the null DGP was estimated "
            "at the frequency the search had already won, so the "
            "regenerated paths carry a component fitted to this sample. "
            "Passing an explicit fourier={'freq': ...} is correctly "
            "sized (5.5% / 5.4% at f=1). See OBS-19.",
            PyardlMethodologyWarning,
            stacklevel=3,
        )


def _guard_overparameterised(n_est: int, n_par: int) -> None:
    """Warn when the specification is rich beyond what T can support."""
    if n_par > 0 and n_est / n_par < 5.0:
        warnings.warn(
            f"{n_est} usable observations for {n_par} parameters "
            f"(ratio {n_est / n_par:.1f} < 5). A decomposition, Fourier "
            "terms and lags each look cheap on their own; together they "
            "are not. Every statistic below is computed, but its finite-"
            "sample distribution is poorly approximated at this ratio.",
            PyardlMethodologyWarning,
            stacklevel=3,
        )


def _decide(stat: float, cv: float, tail: str) -> str:
    if tail == "upper":
        return "cointegration" if stat > cv else "no_cointegration"
    return "cointegration" if stat < cv else "no_cointegration"


def _bootstrap_cell(
    y_arr: FloatArray,
    z_arr: FloatArray,
    x_marg: FloatArray,
    threshold: float,
    asym_positions: tuple[int, ...],
    p: int,
    q_cond: tuple[int, ...],
    case: int,
    conditional: bool,
    config: _CellConfig,
    n_boot: int,
    resample: ResampleScheme,
    var_order: int,
    burn_in: int,
    seed: int,
) -> _BootOutcome:
    """Bootstrap the full triplet for any cell of the matrix.

    Assembled entirely from validated bricks: the null DGP and path
    simulation of spec 14/16 (``pyardl.bootstrap.dgp``), the batched
    statistics of the same specs (``pyardl.bootstrap.batch``, with the
    ``extra`` deterministic plane added by spec 20), and the residual
    resampling of spec 14. What is new here is only the wiring:

    - the marginal VAR runs on the **original** regressors while the
      conditional equation carries the partial sums (their increments
      are sign-constrained; a VAR fitted to them would generate paths
      that are partial sums of nothing);
    - the Fourier columns stay in the null model and in every
      regenerated path — a deterministic term does not vanish when the
      level relationship does;
    - when the frequency was searched on the data, every replication
      re-runs the same search (OBS-15).
    """
    from pyardl.bootstrap.batch import batch_uecm_statistics
    from pyardl.bootstrap.dgp import estimate_null_dgp, simulate_paths
    from pyardl.bootstrap.resample import resample_residuals

    rng = np.random.default_rng(seed)
    n_obs = y_arr.shape[0]
    k_marg = x_marg.shape[1]
    t_obs = np.arange(1, n_obs + 1, dtype=np.float64)

    def expand(dx_t: FloatArray) -> FloatArray:
        """One period of conditional increments from marginal ones."""
        out = []
        for j in range(k_marg):
            if j in asym_positions:
                centred = dx_t[:, j] - threshold
                out.append(np.maximum(centred, 0.0))
                out.append(np.minimum(centred, 0.0))
            else:
                out.append(dx_t[:, j])
        return np.column_stack(out)

    def levels_from(x_star: FloatArray) -> FloatArray:
        """Re-apply the decomposition to regenerated original-scale paths."""
        if not asym_positions:
            return x_star
        b = x_star.shape[0]
        cols = []
        for j in range(k_marg):
            if j in asym_positions:
                d = np.diff(x_star[:, :, j], axis=1) - threshold
                zero = np.zeros((b, 1))
                cols.append(
                    np.concatenate(
                        [zero, np.cumsum(np.maximum(d, 0.0), axis=1)], axis=1
                    )
                )
                cols.append(
                    np.concatenate(
                        [zero, np.cumsum(np.minimum(d, 0.0), axis=1)], axis=1
                    )
                )
            else:
                cols.append(x_star[:, :, j])
        return np.stack(cols, axis=2)

    # --- observed statistics, frequency search included -------------------
    def stats_at(freq: float | None) -> tuple[float, float, float, float]:
        extra = None
        if config.fourier_k and freq is not None:
            cols = _fourier_matrix(t_obs, n_obs, freq, config.fourier_k)
            extra = cols[None, :, :]
        f, t, fi, ssr, ok = batch_uecm_statistics(
            y_arr[None, :], z_arr[None, :, :], p, q_cond, case, conditional, extra
        )
        if not ok[0]:
            raise ValueError(
                "The observed specification could not be estimated "
                "(singular design). Reduce the lag orders or the number "
                "of decomposed regressors."
            )
        return float(f[0]), float(t[0]), float(fi[0]), float(ssr[0])

    selection: pd.DataFrame | None = None
    freq_used: float | None
    if config.fourier_k == 0:
        freq_used = None
        f_obs, t_obs_stat, fi_obs, _ = stats_at(None)
    elif config.freq is not None:
        freq_used = config.freq
        f_obs, t_obs_stat, fi_obs, _ = stats_at(freq_used)
    else:
        table = []
        for cand in config.grid:
            f_at, t_at, fi_at, ssr_at = stats_at(cand)
            table.append((cand, ssr_at, f_at, t_at, fi_at))
        table.sort(key=lambda r: r[1])
        selection = pd.DataFrame([(r[0], r[1]) for r in table], columns=["freq", "ssr"])
        freq_used, _, f_obs, t_obs_stat, fi_obs = table[0]

    # --- null DGP: decomposed conditional equation, original marginal -----
    det_obs = (
        _fourier_matrix(t_obs, n_obs, freq_used, config.fourier_k)
        if config.fourier_k and freq_used is not None
        else None
    )
    dgp = estimate_null_dgp(
        y_arr,
        z_arr,
        p=p,
        q=q_cond,
        case=case,
        var_order=var_order,
        conditional=conditional,
        x_marginal=x_marg if asym_positions else None,
        det=det_obs,
    )

    n_periods = burn_in + n_obs
    det_paths = (
        _fourier_matrix(
            np.arange(1 - burn_in, n_obs + 1, dtype=np.float64),
            n_obs,
            freq_used,
            config.fourier_k,
        )
        if det_obs is not None and freq_used is not None
        else None
    )
    if config.fourier_k == 0:
        candidates: tuple[float, ...] = ()
    elif freq_used is not None and (config.freq is not None or selection is None):
        candidates = (freq_used,)
    else:
        candidates = config.grid

    f_star = np.empty(n_boot)
    t_star = np.empty(n_boot)
    i_star = np.empty(n_boot)
    kept = 0
    n_failed = 0
    n_eq = 1 + dgp.n_regressors
    done = 0
    while done < n_boot:
        size = min(_CHUNK, n_boot - done)
        block = np.empty((size, n_periods, n_eq))
        for i in range(size):
            block[i] = resample_residuals(dgp.residuals, n_periods, rng, resample)
        y_blk, x_blk = simulate_paths(
            dgp,
            block,
            y0=y_arr[0],
            x0=x_marg[0],
            burn_in=burn_in,
            expand=expand if asym_positions else None,
            det_paths=det_paths,
        )
        z_blk = levels_from(x_blk)
        f_b: FloatArray = np.full(size, np.nan)
        t_b: FloatArray = np.full(size, np.nan)
        i_b: FloatArray = np.full(size, np.nan)
        ok: FloatArray = np.zeros(size, dtype=np.float64)
        if not candidates:
            f_b, t_b, i_b, _ssr_b, ok = batch_uecm_statistics(
                y_blk, z_blk, p, q_cond, case, conditional
            )
        else:
            # Same search as the observed call, replication by
            # replication: the winning frequency of a null sample is the
            # one that minimises ITS OWN residual sum of squares.
            running_ssr: FloatArray = np.full(size, np.inf)
            for cand in candidates:
                cols = _fourier_matrix(t_obs, n_obs, cand, config.fourier_k)
                extra = np.broadcast_to(cols, (size, n_obs, cols.shape[1])).copy()
                f_c, t_c, i_c, ssr_c, ok_c = batch_uecm_statistics(
                    y_blk, z_blk, p, q_cond, case, conditional, extra
                )
                better = np.logical_and(ok_c.astype(bool), ssr_c < running_ssr)
                running_ssr = np.asarray(
                    np.where(better, ssr_c, running_ssr), dtype=np.float64
                )
                f_b = np.asarray(np.where(better, f_c, f_b), dtype=np.float64)
                t_b = np.asarray(np.where(better, t_c, t_b), dtype=np.float64)
                i_b = np.asarray(np.where(better, i_c, i_b), dtype=np.float64)
                ok = np.asarray(
                    np.logical_or(ok.astype(bool), better), dtype=np.float64
                )
        mask = ok.astype(bool)
        n_ok = int(mask.sum())
        f_star[kept : kept + n_ok] = f_b[mask]
        t_star[kept : kept + n_ok] = t_b[mask]
        i_star[kept : kept + n_ok] = i_b[mask]
        kept += n_ok
        n_failed += size - n_ok
        done += size

    if kept < n_boot // 2:
        raise ValueError(
            f"Only {kept} of {n_boot} bootstrap replications could be "
            "estimated. The specification is probably too rich for the "
            "sample."
        )
    if n_failed:
        warnings.warn(
            f"{n_failed} of {n_boot} bootstrap replications were discarded "
            f"as unestimable; the critical values rest on the remaining "
            f"{kept}.",
            PyardlMethodologyWarning,
            stacklevel=3,
        )
    f_kept = f_star[:kept]
    t_kept = t_star[:kept]
    i_kept = i_star[:kept]

    return _BootOutcome(
        frequency=freq_used,
        selection=selection,
        f_stat=f_obs,
        t_stat=t_obs_stat,
        f_indep_stat=fi_obs,
        f_critical={a: float(np.quantile(f_kept, 1 - a)) for a in _ALPHAS},
        t_critical={a: float(np.quantile(t_kept, a)) for a in _ALPHAS},
        f_indep_critical={a: float(np.quantile(i_kept, 1 - a)) for a in _ALPHAS},
        f_pvalue=float((1 + np.sum(f_kept >= f_obs)) / (kept + 1)),
        t_pvalue=float((1 + np.sum(t_kept <= t_obs_stat)) / (kept + 1)),
        f_indep_pvalue=float((1 + np.sum(i_kept >= fi_obs)) / (kept + 1)),
        n_boot=kept,
        n_failed=n_failed,
    )


def cointegration_analysis(
    y: ArrayLike,
    x: ArrayLike,
    asym: list[str] | None = None,
    fourier: dict[str, Any] | None = None,
    inference: Inference = "bounds",
    case: int = 3,
    order: tuple[int, int | dict[str, int]] | None = None,
    conditional: bool = True,
    threshold: float = 0.0,
    n_boot: int = 2999,
    alpha: float = 0.05,
    seed: int | None = None,
    resample: ResampleScheme = "iid",
    var_order: int = 1,
    burn_in: int = 50,
) -> UnifiedResults:
    r"""One entry point for the whole ARDL cointegration matrix.

    The three switches — ``asym`` (linear vs partial-sum NARDL),
    ``fourier`` (smooth break vs none) and ``inference`` (tabulated or
    simulated bounds vs bootstrap) — span the eight configurations of
    the unified framework. Each cell delegates to the brick validated
    for it and, above all, gets the critical values that cell requires:
    see :func:`resolve_critical_values` for the rules and the one
    combination that is refused rather than approximated.

    Parameters
    ----------
    y, x : array_like
        Dependent variable and level regressors.
    asym : list of str, optional
        Regressor names to decompose into partial sums. ``None`` keeps
        the model linear.
    fourier : dict, optional
        ``{"k": 1, "freq": "auto", "grid": (1, 2, 3, 4, 5)}``. ``k`` is
        the number of harmonic pairs; ``freq`` a fixed fundamental
        frequency or ``"auto"`` to search the grid by minimum SSR.
        ``None`` (default) includes no Fourier terms.
    inference : {'bounds', 'bootstrap'}
        ``'bounds'``: the tabulated or pre-simulated route of the
        corresponding single-cell test. ``'bootstrap'``: the joint-null
        bootstrap, available for every cell and the only route carrying
        the full three-test triplet in every cell.
    case : int, default 3
        Deterministic case, PSS numbering.
    order : tuple, optional
        ``(p, q)``; ``q`` may be a dict keyed by (transformed) column
        name. Selected by information criterion when omitted.
    conditional : bool, default True
        Conditional (contemporaneous differences in) or unconditional
        UECM.
    threshold : float, default 0.0
        Threshold of the partial-sum decomposition.
    n_boot : int, default 2999
        Bootstrap replications, or simulation draws for the Fourier
        route.
    alpha : float, default 0.05
        Level used for the reported decisions.
    seed : int, optional
        Seed for any simulated route; recorded in the result.
    resample, var_order, burn_in
        Bootstrap knobs, as in
        :func:`pyardl.bootstrap.bootstrap_bounds_test`.

    Returns
    -------
    UnifiedResults

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(7)
    >>> n = 150
    >>> x = np.cumsum(rng.normal(size=n))
    >>> y = np.zeros(n)
    >>> for i in range(1, n):
    ...     y[i] = y[i - 1] - 0.4 * (y[i - 1] - x[i - 1]) + rng.normal(scale=0.5)
    >>> res = cointegration_analysis(
    ...     pd.Series(y, name="y"), pd.DataFrame({"x": x})
    ... )
    >>> res.label
    'ardl/bounds'
    >>> res.classification
    'cointegration'
    """
    from pyardl.utils import check_series

    if inference not in ("bounds", "bootstrap"):
        raise ValueError(
            f"inference must be 'bounds' or 'bootstrap', got {inference!r}."
        )
    if alpha not in _ALPHAS:
        raise ValueError(f"alpha must be one of {_ALPHAS}, got {alpha}.")

    fourier_cfg: dict[str, Any] = dict(fourier) if fourier else {}
    unknown_keys = set(fourier_cfg) - {"k", "freq", "grid"}
    if unknown_keys:
        raise ValueError(
            f"fourier keys {sorted(unknown_keys)} are not understood; "
            "expected 'k', 'freq', 'grid'."
        )
    fourier_k = int(fourier_cfg.get("k", 1)) if fourier else 0
    if fourier and fourier_k < 1:
        raise ValueError(f"fourier k must be >= 1, got {fourier_k}.")
    freq_setting = fourier_cfg.get("freq", "auto")
    grid = tuple(float(g) for g in fourier_cfg.get("grid", (1, 2, 3, 4, 5)))
    if fourier and not grid and freq_setting == "auto":
        raise ValueError("fourier grid is empty: no frequency to search.")
    fixed_freq: float | None = (
        None if (not fourier or freq_setting == "auto") else float(freq_setting)
    )

    cv_source, cv_reason = resolve_critical_values(bool(asym), bool(fourier), inference)

    y_arr, x_arr, _, y_name, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("cointegration_analysis needs at least one regressor.")
    if asym is not None:
        missing = [a for a in asym if a not in x_names]
        if missing:
            raise ValueError(
                f"asym names {missing} are not regressors; available: {list(x_names)}."
            )

    config = _CellConfig(
        asym=tuple(asym) if asym else (),
        fourier_k=fourier_k,
        freq=fixed_freq,
        grid=grid,
        inference=inference,
        cv_source=cv_source,
        cv_reason=cv_reason,
    )
    rerun: dict[str, object] = {
        "y": y,
        "x": x,
        "_x_names": tuple(x_names),
        "inference": inference,
        "case": case,
        "conditional": conditional,
        "threshold": threshold,
        "n_boot": n_boot,
        "alpha": alpha,
        "seed": seed,
        "resample": resample,
        "var_order": var_order,
        "burn_in": burn_in,
        "order": order,
    }

    if seed is None:
        entropy = np.random.SeedSequence().entropy
        seed = int(entropy) % (2**63) if isinstance(entropy, int) else 0

    # ------------------------------------------------------------------
    # Delegated single-cell routes ('bounds' inference).
    # ------------------------------------------------------------------
    if inference == "bounds" and not asym and not fourier:
        from pyardl.bounds import bounds_test

        bt = bounds_test(
            y, x, case=case, order=order, alpha=alpha, conditional=conditional
        )
        classification, reason = classify(
            bt.decision_f, bt.decision_t, bt.decision_indep
        )
        return UnifiedResults(
            label=config.label,
            cv_source=cv_source,
            cv_reason=cv_reason,
            case=case,
            alpha=alpha,
            nobs=int(bt._fit.nobs),
            order=bt.order,
            frequency=None,
            f_stat=bt.f_stat,
            t_stat=bt.t_stat,
            f_indep_stat=bt.f_indep_stat,
            f_critical=None,
            t_critical=None,
            f_indep_critical=None,
            f_pvalue=None,
            t_pvalue=None,
            f_indep_pvalue=None,
            decision_f=bt.decision_f,
            decision_t=bt.decision_t,
            decision_indep=bt.decision_indep,
            classification=classification,
            reason=reason,
            seed=None,
            n_boot=None,
            detail=bt,
            _rerun=rerun,
        )

    if inference == "bounds" and fourier and not asym:
        from pyardl.fourier import fourier_bounds_test

        fb = fourier_bounds_test(
            y,
            x,
            case=case,
            order=order if order is not None else (1, 1),
            fourier_k=fourier_k,
            freq=fixed_freq if fixed_freq is not None else "auto",
            grid=grid,
            alpha=alpha,
            n_sims=n_boot,
            seed=seed,
        )
        return UnifiedResults(
            label=config.label,
            cv_source=cv_source,
            cv_reason=cv_reason,
            case=case,
            alpha=alpha,
            nobs=fb.nobs,
            order=(
                order[0] if order is not None else 1,
                dict.fromkeys(x_names, order[1] if order is not None else 1)  # type: ignore[arg-type]
                if order is None or not isinstance(order[1], dict)
                else dict(order[1]),
            ),
            frequency=fb.frequency,
            f_stat=None,
            t_stat=fb.t_stat,
            f_indep_stat=None,
            f_critical=None,
            t_critical=fb.critical,
            f_indep_critical=None,
            f_pvalue=None,
            t_pvalue=fb.pvalue,
            f_indep_pvalue=None,
            decision_f=None,
            decision_t=fb.decision,
            decision_indep=None,
            classification=None,
            reason=None,
            seed=fb.seed,
            n_boot=fb.n_sims,
            detail=fb,
            _rerun=rerun,
        )

    if inference == "bounds" and asym and not fourier:
        from pyardl.nardl import NARDL

        model = NARDL(
            y,
            x,
            asym=list(asym),
            order="auto" if order is None else order,
            case=case,
            threshold=threshold,
        )
        fit = model.fit()
        nb = fit.bounds_test(alpha=alpha)
        return UnifiedResults(
            label=config.label,
            cv_source=cv_source,
            cv_reason=cv_reason,
            case=case,
            alpha=alpha,
            nobs=fit.nobs,
            order=(model.p, dict(model.q_map)),
            frequency=None,
            f_stat=nb.f_stat,
            t_stat=None,
            f_indep_stat=None,
            f_critical=nb.critical,
            t_critical=None,
            f_indep_critical=None,
            f_pvalue=None,
            t_pvalue=None,
            f_indep_pvalue=None,
            decision_f=nb.decision,
            decision_t=None,
            decision_indep=None,
            classification=None,
            reason=None,
            seed=None,
            n_boot=None,
            detail=nb,
            _rerun=rerun,
        )

    # ------------------------------------------------------------------
    # Bootstrap route: one engine, every cell.
    # ------------------------------------------------------------------
    if not asym and not fourier:
        from pyardl.bootstrap import bootstrap_bounds_test

        bb = bootstrap_bounds_test(
            y,
            x,
            case=case,
            order=order,
            n_boot=n_boot,
            resample=resample,
            seed=seed,
            var_order=var_order,
            burn_in=burn_in,
            conditional=conditional,
        )
        classification, reason = bb.classification(alpha)
        return UnifiedResults(
            label=config.label,
            cv_source=cv_source,
            cv_reason=cv_reason,
            case=case,
            alpha=alpha,
            nobs=int(bb.classical._fit.nobs),
            order=bb.classical.order,
            frequency=None,
            f_stat=bb.f_stat,
            t_stat=bb.t_stat,
            f_indep_stat=bb.f_indep_stat,
            f_critical=bb.f_critical,
            t_critical=bb.t_critical,
            f_indep_critical=bb.f_indep_critical,
            f_pvalue=bb.f_pvalue,
            t_pvalue=bb.t_pvalue,
            f_indep_pvalue=bb.f_indep_pvalue,
            decision_f=bb.decision_f(alpha),
            decision_t=bb.decision_t(alpha),
            decision_indep=bb.decision_indep(alpha),
            classification=classification,
            reason=reason,
            seed=bb.seed,
            n_boot=bb.n_boot,
            detail=bb,
            _rerun=rerun,
        )

    # Assemble the transformed conditional columns.
    from pyardl.nardl.decompose import partial_sums

    z_cols: dict[str, FloatArray] = {}
    asym_positions: list[int] = []
    pos = 0
    for j, name in enumerate(x_names):
        if asym and name in asym:
            series = pd.Series(x_arr[:, j], name=name)
            with warnings.catch_warnings():
                warnings.simplefilter("once", PyardlMethodologyWarning)
                pos_part, neg_part = partial_sums(series, threshold=threshold)
            z_cols[str(pos_part.name)] = pos_part.to_numpy()
            z_cols[str(neg_part.name)] = neg_part.to_numpy()
            asym_positions.append(j)
            pos += 2
        else:
            z_cols[name] = x_arr[:, j]
            pos += 1
    z_names = list(z_cols)
    z_arr = np.column_stack(list(z_cols.values()))

    # Lag orders on the transformed model.
    if order is None:
        if asym:
            from pyardl.nardl import NARDL

            model = NARDL(
                y, x, asym=list(asym), order="auto", case=case, threshold=threshold
            )
            p_order = model.p
            q_map = dict(model.q_map)
        else:
            from pyardl.bounds import bounds_test

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bt0 = bounds_test(y, x, case=case, conditional=conditional)
            p_order, q_sel = bt0.order
            q_map = dict(q_sel)
    else:
        p_order = int(order[0])
        q_spec = order[1]
        if isinstance(q_spec, dict):
            missing = [c for c in z_names if c not in q_spec]
            if missing:
                raise ValueError(
                    f"order dict is missing the transformed columns {missing}."
                )
            q_map = {c: int(q_spec[c]) for c in z_names}
        else:
            q_map = dict.fromkeys(z_names, int(q_spec))
    q_cond = tuple(q_map[c] for c in z_names)

    from pyardl.fourier.bounds import _n_parameters

    n_est = y_arr.shape[0] - max(p_order, max(q_cond, default=0))
    n_par = _n_parameters(p_order, q_cond, case, len(z_names)) + 2 * fourier_k
    _guard_overparameterised(n_est, n_par)
    _guard_searched_frequency_bootstrap(fourier_k, fixed_freq is None)

    out = _bootstrap_cell(
        y_arr,
        z_arr,
        x_arr,
        float(threshold),
        tuple(asym_positions),
        p_order,
        q_cond,
        case,
        conditional,
        config,
        n_boot,
        resample,
        var_order,
        burn_in,
        seed,
    )
    decision_f = _decide(out.f_stat, out.f_critical[alpha], "upper")
    decision_t = _decide(out.t_stat, out.t_critical[alpha], "lower")
    decision_i = _decide(
        out.f_indep_stat,
        out.f_indep_critical[alpha],
        "upper",
    )
    classification, reason = classify(decision_f, decision_t, decision_i)

    return UnifiedResults(
        label=config.label,
        cv_source=cv_source,
        cv_reason=cv_reason,
        case=case,
        alpha=alpha,
        nobs=n_est,
        order=(p_order, q_map),
        frequency=out.frequency,
        f_stat=out.f_stat,
        t_stat=out.t_stat,
        f_indep_stat=out.f_indep_stat,
        f_critical=out.f_critical,
        t_critical=out.t_critical,
        f_indep_critical=out.f_indep_critical,
        f_pvalue=out.f_pvalue,
        t_pvalue=out.t_pvalue,
        f_indep_pvalue=out.f_indep_pvalue,
        decision_f=decision_f,
        decision_t=decision_t,
        decision_indep=decision_i,
        classification=classification,
        reason=reason,
        seed=seed,
        n_boot=out.n_boot,
        detail=out,
        _rerun=rerun,
    )
