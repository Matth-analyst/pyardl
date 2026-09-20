r"""Mean-Group QARDL: quantile long-run coefficients, averaged across individuals.

Spec 36: not a founding paper's method but the combination `pyardl`
retains — same logic as Mean-Group NARDL (:mod:`pyardl.panel.mg_nardl`,
spec 30), transposed to the quantile axis instead of the sign axis.
QARDL (:mod:`pyardl.qardl`, Cho, Kim & Shin 2015) answers "does the
long-run link depend on the quantile of the conditional distribution?"
on **one** series; nothing aggregates that question over a panel.

**No new estimator.** Every individual is fit with
:class:`pyardl.qardl.QARDL` exactly as in the single-series case, on the
**same** quantile grid for every individual (a panel unbalanced in
``tau`` has no economic meaning — a model constraint, not an oversight);
this module owns the panel bookkeeping and reuses
:func:`pyardl.panel.mg._aggregate`, applied **separately at each tau**,
so the output is a surface :math:`\hat\theta_{MG}(\tau)` rather than a
point.

Scope of this version
----------------------
Only the individual-fit-plus-Mean-Group-aggregation core (spec 36
§2.1-2.2) is implemented, using ``inference='kernel'`` (delta method)
for every individual fit. Three parts of the spec are **not
implemented** in this version:

- ``inference='mbb'`` aggregated across individuals (spec 36 §2.4):
  aggregating each individual's own moving-block-bootstrap draws into a
  group bootstrap distribution needs threading per-individual draws
  through the aggregator, not implemented here.
- The group constancy and symmetry tests (spec 36 §2.3): both need a
  variance construction that accounts for the correlation of
  :math:`\hat\theta_{MG}(\tau)` across neighbouring :math:`\tau` (the
  same group mean varying continuously), which the spec explicitly asks
  not to approximate by reusing the individual-level test unchanged.
- ``plot_coefficients()`` (a Mean-Group band over
  :meth:`pyardl.qardl.QARDLResults.plot_coefficients`).

See ``docs/DEVIATIONS.md``.

References
----------
.. [1] Cho, J. S., Kim, T. & Shin, Y. (2015). Quantile cointegration in
       the autoregressive distributed-lag modeling framework. *Journal
       of Econometrics*, 188(1), 281-300.
.. [2] Pesaran, M. H. & Smith, R. (1995). Estimating long-run
       relationships from dynamic heterogeneous panels. *Journal of
       Econometrics*, 68(1), 79-113.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.panel.container import PanelData, panel_from_frame
from pyardl.panel.mg import _aggregate
from pyardl.qardl.model import DEFAULT_TAUS, QARDL

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from typing import Literal

    from pyardl.qardl.model import QARDLResults

    Aggregator = Literal["mean", "median", "trimmed"]

__all__ = ["MeanGroupQARDL", "MeanGroupQARDLResults"]


@dataclass(frozen=True)
class MeanGroupQARDLResults:
    """Mean-Group QARDL estimates: one long-run surface per regressor.

    Attributes
    ----------
    longrun_mg : dict of pandas.DataFrame
        ``{variable: DataFrame}``, indexed by ``tau``, columns
        ``theta``, ``se``, ``t``, ``pvalue`` — the Mean-Group long-run
        coefficient at every quantile, aggregated **separately at each
        tau** (never a single pooled number across taus).
    theta_i : dict of pandas.DataFrame
        ``{variable: DataFrame}`` of individual :math:`\\hat\\theta_i(\\tau)`,
        indexed by unit, one column per ``tau``, that were averaged.
    individual : dict
        ``{key: QARDLResults}`` — every individual QARDL fit.
    taus : tuple of float
    panel : PanelData
    aggregator : str
    failed : dict
        ``{key: reason}`` for individuals whose QARDL could not be fitted.
    """

    longrun_mg: dict[str, pd.DataFrame]
    theta_i: dict[str, pd.DataFrame] = field(repr=False)
    individual: dict[object, QARDLResults] = field(repr=False)
    taus: tuple[float, ...] = ()
    panel: PanelData = field(repr=False, default=None)  # type: ignore[assignment]
    aggregator: str = "mean"
    n_effective: int = 0
    failed: dict[object, str] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        """Individuals actually averaged (excludes :attr:`failed`)."""
        any_var = next(iter(self.theta_i.values()))
        return int(any_var.shape[0])

    def summary(self) -> str:
        """Publication-style summary of the group quantile surface."""
        lines = [
            f"Mean-Group QARDL - {self.n_units} individuals, "
            f"{len(self.taus)} quantiles, aggregator={self.aggregator}",
            f"  dependent: {self.panel.y_name}",
        ]
        for name, table in self.longrun_mg.items():
            lines.append(f"\n  {name}:")
            lines.append(f"    {'tau':>6}{'theta':>12}{'se':>12}{'p':>10}")
            for tau, row in table.iterrows():
                lines.append(
                    f"    {tau:>6.2f}{row['theta']:>12.4f}"
                    f"{row['se']:>12.4f}{row['pvalue']:>10.4f}"
                )
        if self.failed:
            lines.append(f"\n  {len(self.failed)} individual(s) could not be fitted:")
            lines.extend(f"    {k!r}: {v}" for k, v in self.failed.items())
        return "\n".join(lines)


class MeanGroupQARDL:
    """Mean-Group estimator for a panel of quantile (QARDL) individuals.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel.
    y : str
    X : sequence of str
    id, time : str
    taus : sequence of float, optional
        Quantile grid, applied identically to every individual.
    order : tuple
        ``(p, q)`` applied to every individual.
    case : int, default 3
        PSS deterministic case, applied to every individual.
    aggregator, trim, min_obs :
        As in :class:`pyardl.panel.mg.MeanGroup`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(12):
    ...     x = np.cumsum(rng.normal(size=100))
    ...     y = np.zeros(100)
    ...     for t in range(1, 100):
    ...         y[t] = y[t - 1] - 0.4 * (y[t - 1] - 1.5 * x[t - 1]) + rng.normal()
    ...     rows.append(
    ...         pd.DataFrame({"id": i, "t": np.arange(100), "y": y, "x": x})
    ...     )
    >>> panel = pd.concat(rows, ignore_index=True)
    >>> res = MeanGroupQARDL(panel, y="y", X=["x"], id="id", time="t",
    ...                       taus=(0.25, 0.5, 0.75), order=(1, 1)).fit()
    >>> res.taus
    (0.25, 0.5, 0.75)
    >>> res.n_units
    12
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y: str,
        X: Sequence[str],
        id: str,  # noqa: A002 - matches the spec's public API
        time: str,
        taus: Sequence[float] = DEFAULT_TAUS,
        order: tuple[int, int | dict[str, int]] = (1, 1),
        case: int = 3,
        aggregator: Aggregator = "mean",
        trim: float = 0.1,
        min_obs: int = 15,
    ) -> None:
        if aggregator not in ("mean", "median", "trimmed"):
            raise ValueError(
                f"aggregator must be 'mean', 'median' or 'trimmed', got {aggregator!r}."
            )
        if not 0.0 <= trim < 0.5:
            raise ValueError(f"trim must be in [0, 0.5), got {trim}.")

        self.panel = panel_from_frame(
            df, y=y, x=list(X), id_col=id, time_col=time, min_obs=min_obs
        )
        self.taus = tuple(sorted(float(t) for t in taus))
        self.order = order
        self.case = int(case)
        self.aggregator: Aggregator = aggregator
        self.trim = float(trim)

    def _fit_one(self, y: pd.Series, x: pd.DataFrame) -> QARDLResults:
        return QARDL(y, x, order=self.order, taus=self.taus, case=self.case).fit(
            inference="kernel"
        )

    def fit(self) -> MeanGroupQARDLResults:
        """Estimate every individual QARDL, then average at each quantile.

        Returns
        -------
        MeanGroupQARDLResults

        Raises
        ------
        ValueError
            If fewer than two individuals can be fitted.
        """
        fits: dict[object, QARDLResults] = {}
        failed: dict[object, str] = {}
        theta_rows: dict[str, dict[object, pd.Series]] = {
            name: {} for name in self.panel.x_names
        }

        for unit in self.panel:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = self._fit_one(unit.y, unit.x)
            except (ValueError, np.linalg.LinAlgError) as exc:
                failed[unit.key] = f"{type(exc).__name__}: {exc}"
                continue
            ok = True
            for name in self.panel.x_names:
                point = fit.longrun(variable=name)[name]
                if not np.all(np.isfinite(point.to_numpy())):
                    ok = False
                    break
                theta_rows[name][unit.key] = point
            if not ok:
                failed[unit.key] = "non-finite long-run coefficients"
                continue
            fits[unit.key] = fit

        if len(fits) < 2:
            raise ValueError(
                f"Only {len(fits)} individual(s) could be fitted, and the "
                "Mean-Group standard error is the dispersion ACROSS "
                f"individuals: with fewer than two there is none. Failures: "
                f"{failed}."
            )
        if failed:
            warnings.warn(
                f"{len(failed)} individual(s) could not be fitted and are "
                f"absent from the group average: {failed}.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

        keys = list(fits)
        index = pd.Index(keys, name="id")
        theta_i: dict[str, pd.DataFrame] = {}
        longrun_mg: dict[str, pd.DataFrame] = {}
        n_eff = 0

        for name in self.panel.x_names:
            mat = pd.DataFrame(
                {k: theta_rows[name][k] for k in keys}
            ).T  # index=unit, columns=tau
            mat.index = index
            theta_i[name] = mat

            theta, var, n_eff = _aggregate(mat.to_numpy(), self.aggregator, self.trim)
            se = np.sqrt(var)
            with np.errstate(divide="ignore", invalid="ignore"):
                tstat = np.where(se > 0, theta / se, np.nan)
            dof = max(n_eff - 1, 1)
            pvalue = 2.0 * stats.t.sf(np.abs(tstat), dof)
            longrun_mg[name] = pd.DataFrame(
                {"theta": theta, "se": se, "t": tstat, "pvalue": pvalue},
                index=pd.Index(self.taus, name="tau"),
            )

        return MeanGroupQARDLResults(
            longrun_mg=longrun_mg,
            theta_i=theta_i,
            individual=fits,
            taus=self.taus,
            panel=self.panel,
            aggregator=self.aggregator,
            n_effective=n_eff,
            failed=failed,
        )
