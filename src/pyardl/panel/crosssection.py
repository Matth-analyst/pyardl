r"""Cross-sectional averages and the CD test (spec 24).

Specs 22 and 23 assume individuals are independent of each other. They
are usually not: a world business cycle, a commodity price, a common
policy shock hit every country at once. Write that as an unobserved
factor structure,

.. math:: y_{it} = \beta_i' x_{it} + \gamma_i' f_t + \varepsilon_{it}

and the problem is immediate — :math:`f_t` is unobserved, correlated
with :math:`x_{it}`, so omitting it biases :math:`\hat\beta_i` and no
amount of data repairs it.

Pesaran's move is to **stop trying to observe the factors**. Average the
observed variables across individuals at each date: because the
loadings :math:`\gamma_i` average to something non-degenerate, those
cross-sectional averages span the same space as :math:`f_t`
asymptotically. Add them as regressors and the factor is controlled for
without ever being estimated.

Two things this module owes the caller
--------------------------------------
**The averages must be honest about who was present.** In an unbalanced
panel the composition of the average changes from date to date, and a
mean over 40 countries is not the same object as a mean over 12. The
count is returned alongside the average, and a sharply varying
composition raises a warning rather than quietly changing what the
regressor means.

**The CD test has a direction that is easy to get backwards.** Its null
is *no* cross-sectional dependence. It is used twice, and the two uses
want opposite answers: before augmenting, a rejection motivates the
whole exercise; after augmenting, a *failure* to reject is the good
outcome. :class:`CDResult` states which reading applies rather than
leaving a bare p-value to be interpreted from memory.

References
----------
.. [1] Pesaran, M. H. (2006). Estimation and inference in large
       heterogeneous panels with a multifactor error structure.
       *Econometrica*, 74(4), 967-1012.
.. [2] Pesaran, M. H. (2015). Testing weak cross-sectional dependence in
       large panels. *Econometric Reviews*, 34(6-10), 1089-1117.
.. [3] Chudik, A. & Pesaran, M. H. (2015). Common correlated effects
       estimation of heterogeneous dynamic panel data models with weakly
       exogenous regressors. *Journal of Econometrics*, 188(2), 393-420.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from pyardl.exceptions import PyardlMethodologyWarning

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import numpy.typing as npt

    FloatArray = npt.NDArray[np.float64]

__all__ = ["CDResult", "cd_test", "cross_section_averages", "default_cs_lags"]


def default_cs_lags(n_obs: int) -> int:
    """Lags of the cross-sectional averages: ``floor(T**(1/3))``.

    The rule of thumb of Chudik and Pesaran (2015). In a *dynamic*
    panel the contemporaneous averages no longer span the factor space
    on their own — the lagged dependent variable drags the factor's own
    history into the equation — so lags of the averages are needed, and
    their number must grow with ``T`` but slowly.

    Computed with :func:`numpy.cbrt`, not ``n_obs ** (1/3)``. The two
    differ exactly where it matters: at a perfect cube the power form
    lands *just below* the integer, and the floor then loses a lag.
    Measured — ``64 ** (1/3) == 3.99999999999999956`` and
    ``1000 ** (1/3) == 9.99999999999999822``, so the naive rule returns
    3 and 9 where it means 4 and 10. A silently shorter lag list is a
    different specification, not a rounding detail.

    Examples
    --------
    >>> [default_cs_lags(t) for t in (27, 64, 100, 125, 1000)]
    [3, 4, 4, 5, 10]
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be positive, got {n_obs}.")
    return int(np.floor(np.cbrt(float(n_obs))))


def cross_section_averages(
    df: pd.DataFrame,
    variables: Sequence[str],
    id_col: str,
    time_col: str,
    lags: int = 0,
    weights: str | None = None,
    warn_composition: bool = True,
) -> pd.DataFrame:
    """Per-period cross-sectional averages, optionally lagged.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel.
    variables : sequence of str
        Columns to average. Typically the dependent variable *and* the
        regressors: the factor loads on both, so averaging only ``x``
        leaves part of the factor in the error.
    id_col, time_col : str
        Individual and time identifiers.
    lags : int, default 0
        How many lags of each average to add. ``0`` gives the
        contemporaneous averages only, which is the static CCE case.
    weights : str, optional
        Column of weights. ``None`` gives equal weights, which is the
        standard choice and the one the asymptotics are derived for.
    warn_composition : bool, default True
        Whether to warn when the number of individuals entering the
        average varies sharply across periods.

    Returns
    -------
    pandas.DataFrame
        Indexed by period. Columns ``cs_<var>`` and ``cs_<var>_L<k>``,
        plus ``cs_count`` — the number of individuals that entered each
        average. The count is not decoration: in an unbalanced panel it
        is the difference between a regressor that means "the world" and
        one that means "whoever reported this year".

    Raises
    ------
    ValueError
        If a column is missing, if ``lags`` is negative, or if the
        weights contain a non-positive value — a zero weight silently
        drops an individual, and a negative one is not an average.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "id": ["a", "a", "b", "b"],
    ...     "t": [0, 1, 0, 1],
    ...     "y": [1.0, 3.0, 3.0, 5.0],
    ... })
    >>> out = cross_section_averages(df, ["y"], "id", "t")
    >>> out["cs_y"].tolist()
    [2.0, 4.0]
    >>> out["cs_count"].tolist()
    [2, 2]
    """
    if lags < 0:
        raise ValueError(f"lags must be non-negative, got {lags}.")
    names = [str(v) for v in variables]
    if not names:
        raise ValueError(
            "variables is empty: with nothing to average there is no "
            "approximation of the common factors."
        )
    needed = [*names, id_col, time_col] + ([weights] if weights else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} are not in the DataFrame; "
            f"available: {list(df.columns)}."
        )

    work = df[needed].copy()
    if weights is not None:
        w = work[weights].astype(np.float64)
        if not np.all(np.isfinite(w)) or (w <= 0).any():
            raise ValueError(
                "weights must be finite and strictly positive: a zero weight "
                "silently removes an individual from the average, and a "
                "negative one does not produce an average at all."
            )

    out: dict[str, pd.Series] = {}
    grouped = work.groupby(time_col, sort=True)
    for name in names:
        if weights is None:
            out[f"cs_{name}"] = grouped[name].mean()
        else:

            def _wmean(block: pd.DataFrame, col: str = name) -> float:
                vals = block[col].to_numpy(dtype=np.float64)
                wts = block[weights].to_numpy(dtype=np.float64)
                ok = np.isfinite(vals)
                if not ok.any():
                    return float("nan")
                return float(np.average(vals[ok], weights=wts[ok]))

            # Les colonnes sont selectionnees AVANT `apply`, ce qui
            # exclut la cle de groupe sans avoir besoin de
            # `include_groups=False` — un argument que pandas n'a
            # introduit qu'en 2.2, alors que le plancher declare est
            # 2.1. Sur 2.1 il n'etait pas ignore : il etait transmis a
            # `_wmean`, qui levait un TypeError. Le job `floors` de la
            # CI est ce qui l'a attrape.
            columns = [name] if weights == name else [name, weights]
            out[f"cs_{name}"] = grouped[columns].apply(_wmean)

    counts = grouped[names[0]].count()
    frame = pd.DataFrame(out)
    frame.index.name = str(time_col)

    for k in range(1, lags + 1):
        for name in names:
            frame[f"cs_{name}_L{k}"] = frame[f"cs_{name}"].shift(k)

    frame["cs_count"] = counts.astype(int)

    if warn_composition and len(counts) > 1:
        lo, hi = int(counts.min()), int(counts.max())
        # A composition that halves is a different average, not a noisier
        # one: the threshold is on the ratio, not on the spread.
        if lo > 0 and hi / lo >= 1.5:
            warnings.warn(
                f"The cross-sectional averages are taken over {lo} to {hi} "
                "individuals depending on the period. They approximate the "
                "common factors only if the composition is roughly stable; "
                "here it is not, so early and late averages are not the same "
                "object. See the cs_count column.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )
    return frame


@dataclass(frozen=True)
class CDResult:
    """Outcome of Pesaran's CD test.

    Attributes
    ----------
    statistic : float
        Standard normal under the null.
    pvalue : float
        Two-sided.
    n_units, n_pairs : int
        Individuals retained, and pairs actually used.
    mean_abs_correlation : float
        Average absolute pairwise residual correlation. The statistic
        can be near zero because correlations genuinely are, or because
        positive and negative ones cancel; this number tells the two
        apart, and they call for different conclusions.
    """

    statistic: float
    pvalue: float
    n_units: int
    n_pairs: int
    mean_abs_correlation: float
    dropped: dict[object, str] = field(default_factory=dict)

    @property
    def rejects(self) -> bool:
        """Whether the null of no cross-sectional dependence is rejected at 5%."""
        return bool(self.pvalue < 0.05)

    def summary(self, context: str = "") -> str:
        """Report the verdict *and* which reading of it applies.

        Parameters
        ----------
        context : {'', 'before', 'after'}
            ``'before'`` reads a rejection as the motivation for
            augmenting with cross-sectional averages; ``'after'`` reads
            a rejection as the augmentation having been insufficient.
            The same p-value means opposite things in the two, which is
            why the direction is stated rather than assumed.
        """
        lines = [
            "Pesaran CD test for cross-sectional dependence",
            "  H0: residuals are cross-sectionally independent",
            f"  CD = {self.statistic:.4f}   p = {self.pvalue:.4f}   "
            f"({self.n_units} individuals, {self.n_pairs} pairs)",
            f"  mean |pairwise correlation| = {self.mean_abs_correlation:.4f}",
        ]
        verdict = "reject" if self.rejects else "do not reject"
        if context == "before":
            lines.append(
                f"  {verdict} at 5%: "
                + (
                    "a common factor is present, so MG and PMG are biased and "
                    "the cross-sectional augmentation is warranted."
                    if self.rejects
                    else "no evidence of a common factor; the augmentation "
                    "spends parameters for nothing."
                )
            )
        elif context == "after":
            lines.append(
                f"  {verdict} at 5%: "
                + (
                    "dependence SURVIVES the augmentation — more lags of the "
                    "averages, or more factors than the averages can span."
                    if self.rejects
                    else "the augmentation absorbed the dependence, which is "
                    "the outcome it is there for."
                )
            )
        else:
            lines.append(f"  {verdict} at 5%")
        if self.dropped:
            lines.append(f"  excluded {len(self.dropped)}: {self.dropped}")
        return "\n".join(lines)


def cd_test(
    residuals: pd.DataFrame | dict[object, pd.Series],
    min_overlap: int = 5,
) -> CDResult:
    r"""Pesaran's CD test for cross-sectional dependence.

    .. math::

        CD = \sqrt{\frac{2}{N(N-1)}}
             \sum_{i<j} \sqrt{T_{ij}}\, \hat\rho_{ij}

    where :math:`\hat\rho_{ij}` is the correlation of the residuals of
    individuals :math:`i` and :math:`j` over their :math:`T_{ij}` common
    dates. Standard normal under the null of independence, and — the
    property that makes it usable here — valid for *large N with small
    T*, which is the regime where a per-individual estimator leaves few
    residuals each.

    Parameters
    ----------
    residuals : pandas.DataFrame or dict
        Residuals per individual. A DataFrame is read as one column per
        individual indexed by period; a dict maps identifier to a Series
        indexed by period. Pairs are matched **on the index**, not by
        position: two individuals with different sample windows must
        not have their residuals lined up by row number.
    min_overlap : int, default 5
        Pairs sharing fewer periods are skipped, and recorded.

    Returns
    -------
    CDResult

    Raises
    ------
    ValueError
        If fewer than two individuals are usable, or if no pair has
        enough overlap.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> f = rng.normal(size=60)
    >>> dep = pd.DataFrame({f"u{i}": f + 0.3 * rng.normal(size=60)
    ...                     for i in range(8)})
    >>> bool(cd_test(dep).rejects)
    True
    >>> ind = pd.DataFrame({f"u{i}": rng.normal(size=60) for i in range(8)})
    >>> bool(cd_test(ind).rejects)
    False
    """
    frame = pd.DataFrame(residuals) if isinstance(residuals, dict) else residuals.copy()
    dropped: dict[object, str] = {}
    keep: list[object] = []
    for col in frame.columns:
        series = frame[col].dropna()
        if series.size < min_overlap:
            dropped[col] = f"{series.size} usable residuals"
        elif not np.isfinite(series.to_numpy(dtype=np.float64)).all():
            dropped[col] = "non-finite residuals"
        elif float(series.std(ddof=1)) <= 0.0:
            dropped[col] = "constant residuals"
        else:
            keep.append(col)
    if len(keep) < 2:
        raise ValueError(
            f"The CD test needs at least two usable individuals, got "
            f"{len(keep)}. Excluded: {dropped}."
        )

    frame = frame[keep]
    cols = list(frame.columns)

    # Chemin rapide : sans valeur manquante, toutes les paires partagent
    # les memes dates, donc la matrice de correlation se calcule en une
    # fois. Le chemin general ci-dessous fabrique un DataFrame PAR PAIRE,
    # ce qui coute O(N^2) constructions pandas : mesure a 818 ms pour
    # N = 30 et 3.4 s pour N = 60, contre quelques millisecondes ici.
    # Les deux donnent le meme nombre — un test le verifie.
    if not frame.isna().to_numpy().any():
        values = frame.to_numpy(dtype=np.float64)
        n_t = values.shape[0]
        if n_t >= min_overlap:
            corr = np.corrcoef(values, rowvar=False)
            upper = np.triu_indices(len(cols), k=1)
            pair_rho = corr[upper]
            finite = np.isfinite(pair_rho)
            if finite.any():
                n_all = len(cols)
                fast_stat = float(
                    np.sqrt(2.0 / (n_all * (n_all - 1)))
                    * np.sqrt(n_t)
                    * pair_rho[finite].sum()
                )
                return CDResult(
                    statistic=fast_stat,
                    pvalue=float(2.0 * stats.norm.sf(abs(fast_stat))),
                    n_units=n_all,
                    n_pairs=int(finite.sum()),
                    mean_abs_correlation=float(np.abs(pair_rho[finite]).mean()),
                    dropped=dropped,
                )

    total = 0.0
    abs_total = 0.0
    n_pairs = 0
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            pair = frame[[cols[a], cols[b]]].dropna()
            if pair.shape[0] < min_overlap:
                continue
            u = pair.iloc[:, 0].to_numpy(dtype=np.float64)
            v = pair.iloc[:, 1].to_numpy(dtype=np.float64)
            su, sv = u.std(ddof=1), v.std(ddof=1)
            if su <= 0 or sv <= 0:
                continue
            rho = float(np.corrcoef(u, v)[0, 1])
            if not np.isfinite(rho):
                continue
            total += np.sqrt(pair.shape[0]) * rho
            abs_total += abs(rho)
            n_pairs += 1

    if n_pairs == 0:
        raise ValueError(
            f"No pair of individuals shares at least {min_overlap} periods, "
            "so no correlation could be computed."
        )
    n = len(cols)
    statistic = float(np.sqrt(2.0 / (n * (n - 1))) * total)
    return CDResult(
        statistic=statistic,
        pvalue=float(2.0 * stats.norm.sf(abs(statistic))),
        n_units=n,
        n_pairs=n_pairs,
        mean_abs_correlation=float(abs_total / n_pairs),
        dropped=dropped,
    )
