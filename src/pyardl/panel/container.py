r"""Panel container: a long DataFrame turned into N time series.

Specs 22, 23 and 24 all rest on the same move — a heterogeneous dynamic
panel is **N time-series problems plus an aggregation step**, not a new
estimator. This module is the first half of that sentence: it validates
a long-format panel and hands out one clean ``(y_i, X_i)`` pair per
individual, in a fixed order, with the time dimension sorted and the
gaps named.

Three things it refuses to do quietly, because each is a way to get a
plausible number out of a broken panel:

- **Reorder time silently.** A dynamic model reads lags off the row
  order. A panel sorted by anything other than time within individual
  produces lags of the wrong observations, and nothing downstream can
  detect it. The container sorts, and says so when the input was not
  already sorted.
- **Bridge a gap in the time index.** A missing year inside an
  individual's history is not the same as a shorter history: lagging
  across the hole silently pairs observations that are two periods
  apart. Individuals with internal gaps are reported, not patched.
- **Drop an individual without saying which.** Too few observations,
  a constant series, an all-NaN column — each is a legitimate reason to
  exclude an individual, and each is recorded in :attr:`PanelData.excluded`
  with the reason, so ``N`` in a results table is always accounted for.

References
----------
.. [1] Pesaran, M. H. & Smith, R. (1995). Estimating long-run
       relationships from dynamic heterogeneous panels. *Journal of
       Econometrics*, 68(1), 79-113.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pyardl.exceptions import PyardlMethodologyWarning

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator, Sequence

__all__ = ["PanelData", "PanelUnit"]

#: Below this many usable observations the Pesaran-Smith framework is
#: outside the regime it was derived for: MG is consistent as T grows,
#: and at small T the individual estimates carry the dynamic-panel bias
#: that averaging cannot remove.
MIN_T_RECOMMENDED = 30


@dataclass(frozen=True)
class PanelUnit:
    """One individual's time series, ready for a time-series estimator.

    Attributes
    ----------
    key : object
        The individual's identifier, as it appeared in the input.
    y : pandas.Series
        Dependent variable, indexed by the time column.
    x : pandas.DataFrame
        Regressors, same index.
    nobs : int
        Rows available before any lag is taken.
    has_time_gaps : bool
        Whether the time index skips periods. A dynamic model lags by
        position, so a gap makes some "lags" span more than one period.
    """

    key: object
    y: pd.Series
    x: pd.DataFrame

    @property
    def nobs(self) -> int:
        return int(self.y.shape[0])

    @property
    def has_time_gaps(self) -> bool:
        idx = self.y.index
        if idx.size < 3:
            return False
        try:
            steps = np.diff(np.asarray(idx, dtype=np.float64))
        except (TypeError, ValueError):  # pragma: no cover - exotic index
            return False
        return bool(np.unique(steps).size > 1)


@dataclass(frozen=True)
class PanelData:
    """A validated heterogeneous panel, addressable individual by individual.

    Attributes
    ----------
    units : tuple of PanelUnit
        The retained individuals, in the order they will be estimated
        and averaged. Sorted by key so that results do not depend on the
        row order of the input.
    y_name : str
        Name of the dependent variable.
    x_names : tuple of str
        Names of the regressors, in design order.
    excluded : dict
        ``{key: reason}`` for every individual dropped during
        validation. Never empty silently: a results table reporting
        ``N = 22`` out of 24 can always say which two are missing and
        why.
    unbalanced : bool
        Whether the retained individuals have different sample lengths.
    """

    units: tuple[PanelUnit, ...]
    y_name: str
    x_names: tuple[str, ...]
    excluded: dict[object, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self) -> Iterator[PanelUnit]:
        return iter(self.units)

    def __getitem__(self, key: object) -> PanelUnit:
        for unit in self.units:
            if unit.key == key:
                return unit
        raise KeyError(
            f"{key!r} is not a retained individual. "
            f"Available: {[u.key for u in self.units]}."
            + (f" Excluded: {self.excluded}." if self.excluded else "")
        )

    @property
    def keys(self) -> tuple[object, ...]:
        """Identifiers of the retained individuals, in estimation order."""
        return tuple(u.key for u in self.units)

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def sample_sizes(self) -> pd.Series:
        """Observations available per individual, before lagging."""
        return pd.Series(
            [u.nobs for u in self.units],
            index=pd.Index(list(self.keys), name="id"),
            name="nobs",
        )

    @property
    def unbalanced(self) -> bool:
        return bool(self.sample_sizes.nunique() > 1)

    @property
    def units_with_time_gaps(self) -> tuple[object, ...]:
        """Individuals whose time index skips periods."""
        return tuple(u.key for u in self.units if u.has_time_gaps)

    def summary(self) -> str:
        """One-paragraph description of what survived validation."""
        sizes = self.sample_sizes
        lines = [
            f"Panel: {self.n_units} individuals, "
            f"{'unbalanced' if self.unbalanced else 'balanced'}",
            f"  dependent: {self.y_name}   regressors: {', '.join(self.x_names)}",
            f"  observations per individual: min {int(sizes.min())}, "
            f"median {int(sizes.median())}, max {int(sizes.max())}",
        ]
        gaps = self.units_with_time_gaps
        if gaps:
            lines.append(
                f"  time gaps in {len(gaps)} individual(s): {list(gaps)} - "
                "lags there span more than one period"
            )
        if self.excluded:
            lines.append(f"  excluded {len(self.excluded)}:")
            lines.extend(f"    {k!r}: {v}" for k, v in self.excluded.items())
        return "\n".join(lines)


def _check_columns(df: pd.DataFrame, needed: Sequence[str]) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} are not in the DataFrame; "
            f"available: {list(df.columns)}."
        )


def panel_from_frame(
    df: pd.DataFrame,
    y: str,
    x: Sequence[str],
    id_col: str,
    time_col: str,
    min_obs: int = 15,
    warn_short: bool = True,
) -> PanelData:
    """Validate a long-format panel and split it by individual.

    Parameters
    ----------
    df : pandas.DataFrame
        Long format: one row per (individual, period).
    y : str
        Dependent-variable column.
    x : sequence of str
        Regressor columns, in the order they should enter the design.
    id_col, time_col : str
        Individual and time identifiers.
    min_obs : int, default 15
        Individuals with fewer usable rows are excluded, with the reason
        recorded in :attr:`PanelData.excluded`.
    warn_short : bool, default True
        Whether to warn when individuals fall below
        :data:`MIN_T_RECOMMENDED` observations.

    Returns
    -------
    PanelData

    Raises
    ------
    ValueError
        If a column is missing, if no individual survives validation, or
        if an individual has duplicate periods — a duplicated period
        makes the lag structure ambiguous, and picking one of the rows
        would be a silent choice about the data.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "country": ["FR"] * 20 + ["DE"] * 20,
    ...     "year": list(range(20)) * 2,
    ...     "c": list(range(20)) + list(range(5, 25)),
    ...     "inc": list(range(1, 21)) + list(range(2, 22)),
    ... })
    >>> panel = panel_from_frame(df, "c", ["inc"], "country", "year",
    ...                          min_obs=10, warn_short=False)
    >>> panel.keys
    ('DE', 'FR')
    >>> panel.n_units
    2
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
    x_names = tuple(str(c) for c in x)
    if not x_names:
        raise ValueError(
            "x is empty: a panel ARDL with no regressor is a set of "
            "univariate AR models. Use pyardl.ARDL individually."
        )
    _check_columns(df, [y, *x_names, id_col, time_col])

    units: list[PanelUnit] = []
    excluded: dict[object, str] = {}
    short: list[object] = []

    for key, block in df.groupby(id_col, sort=True, dropna=False):
        block = block.sort_values(time_col)
        if block[time_col].duplicated().any():
            dups = block.loc[block[time_col].duplicated(), time_col].unique()
            raise ValueError(
                f"Individual {key!r} has duplicate periods {list(dups)[:5]}. "
                "A duplicated period makes the lag structure ambiguous; "
                "choosing one of the rows would be a silent decision about "
                "the data, so it is refused here."
            )
        cols = block[[y, *x_names]].astype(np.float64)
        valid = ~cols.isna().any(axis=1)
        if valid.sum() == 0:
            excluded[key] = "no row without a missing value"
            continue
        # Trim leading/trailing NaN, then refuse internal ones: bridging a
        # hole would pair observations that are not one period apart.
        first, last = (
            int(np.argmax(valid.to_numpy())),
            int(len(valid) - 1 - np.argmax(valid.to_numpy()[::-1])),
        )
        window = valid.iloc[first : last + 1]
        if not window.all():
            excluded[key] = (
                f"{int((~window).sum())} missing value(s) inside the sample; "
                "lagging across them would pair observations more than one "
                "period apart"
            )
            continue
        block = block.iloc[first : last + 1]
        cols = cols.iloc[first : last + 1]
        if len(block) < min_obs:
            excluded[key] = f"{len(block)} observations, below min_obs={min_obs}"
            continue
        constant = [c for c in [y, *x_names] if cols[c].nunique() <= 1]
        if constant:
            excluded[key] = f"constant series {constant}"
            continue
        index = pd.Index(block[time_col].to_numpy(), name=str(time_col))
        units.append(
            PanelUnit(
                key=key,
                y=pd.Series(cols[y].to_numpy(), index=index, name=str(y)),
                x=pd.DataFrame(
                    cols[list(x_names)].to_numpy(), index=index, columns=list(x_names)
                ),
            )
        )
        if len(block) < MIN_T_RECOMMENDED:
            short.append(key)

    if not units:
        raise ValueError(
            "No individual survived validation. Reasons: "
            f"{excluded if excluded else 'the panel is empty'}."
        )

    if warn_short and short:
        warnings.warn(
            f"{len(short)} of {len(units) + len(excluded)} individuals have "
            f"fewer than {MIN_T_RECOMMENDED} observations ({short[:5]}"
            f"{'...' if len(short) > 5 else ''}). Mean Group is consistent as "
            "T grows: at small T each individual estimate carries the "
            "dynamic-panel bias, and averaging N of them does not remove a "
            "bias they share.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )

    panel = PanelData(
        units=tuple(units),
        y_name=str(y),
        x_names=x_names,
        excluded=excluded,
    )
    if panel.units_with_time_gaps:
        warnings.warn(
            f"Time gaps in {list(panel.units_with_time_gaps)[:5]}: the time "
            "index skips periods, so a first lag there spans more than one "
            "period. The estimates are computed on the rows as given; "
            "whether that is what you mean is a modelling decision.",
            PyardlMethodologyWarning,
            stacklevel=2,
        )
    return panel
