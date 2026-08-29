r"""Almon (1965): polynomial distributed lags, and the restriction as a test.

A finite distributed lag,

.. math::

    y_t = \alpha + \sum_{i=0}^{q} \beta_i x_{t-i} + \varepsilon_t,

is estimable by least squares for any ``q``. The trouble is that
:math:`x_t, x_{t-1}, \ldots` are nearly collinear, so the individual
:math:`\hat\beta_i` come out imprecise and often alternate in sign for
no economic reason.

Almon's answer is to make the weights lie on a polynomial of degree
``r < q`` in the lag index,

.. math::

    \beta_i = \gamma_0 + \gamma_1 i + \cdots + \gamma_r i^r,
    \qquad \beta = H\gamma, \quad H_{ij} = i^j,

which turns ``q + 1`` free coefficients into ``r + 1``. The regression
is then an ordinary least squares of ``y`` on the *Almon variables*
:math:`z_j = \sum_i i^j x_{t-i}`, and :math:`\hat\beta = H\hat\gamma`
with :math:`\hat V(\hat\beta) = H \hat V(\hat\gamma) H'` — exactly, since
the map is linear.

**The restriction is not free, and this module never lets it pass as
free.** ``polynomial_restriction_test`` puts it against the
unrestricted finite lag model, because a smooth lag distribution
obtained by assuming smoothness is not evidence of smoothness.

Two implementation points that matter more than they look:

**Conditioning.** :math:`H_{ij} = i^j` with ``q = 12`` and ``r = 4``
holds entries up to :math:`12^4`, and the columns are near-collinear.
The internal basis is therefore :math:`(i/q)^j`, or Chebyshev on
request; :math:`\gamma` changes meaning with the basis, :math:`\beta`
does not, and the tests pin the two bases to agree on :math:`\beta`.

**Endpoint constraints** are linear in :math:`\gamma`, so they are
imposed by reparameterising onto the null space of the constraint matrix
rather than by penalising or by dropping terms. That keeps the estimator
exactly least squares under the constraint.

References
----------
.. [1] Almon, S. (1965). The distributed lag between capital
       appropriations and expenditures. *Econometrica*, 33(1), 178-196.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.utils import check_series, lag_matrix

__all__ = ["AlmonModel", "AlmonResults"]

FloatArray = npt.NDArray[np.float64]
Endpoint = Literal["none", "head", "tail", "both"]
Basis = Literal["power", "chebyshev"]


def _basis_matrix(q: int, r: int, basis: Basis, node: FloatArray) -> FloatArray:
    """Rows of the polynomial basis evaluated at the given lag indices.

    ``node`` holds the lag indices, already scaled to ``[0, 1]`` by ``q``
    for the power basis or mapped to ``[-1, 1]`` for Chebyshev. Both are
    reparameterisations of the same polynomial space: they change what
    ``gamma`` means and leave ``beta`` alone.
    """
    if basis == "power":
        return np.vander(node, r + 1, increasing=True).astype(np.float64)
    if basis == "chebyshev":
        scaled = 2.0 * node - 1.0
        cols = [np.ones_like(scaled), scaled]
        for _ in range(2, r + 1):
            cols.append(2.0 * scaled * cols[-1] - cols[-2])
        return np.column_stack(cols[: r + 1]).astype(np.float64)
    raise ValueError(f'basis must be "power" or "chebyshev", got {basis!r}.')


class AlmonModel:
    r"""Polynomial (Almon) distributed lag.

    Parameters
    ----------
    y : array-like, shape (T,)
        Dependent variable.
    x : array-like, shape (T,)
        The single distributed-lag regressor.
    q : int
        Highest lag included.
    r : int
        Degree of the polynomial the weights follow. Must be ``< q``,
        otherwise the "restriction" restricts nothing.
    endpoint : {"none", "head", "tail", "both"}
        Force the weight to vanish just outside the window:
        ``"head"`` imposes :math:`\beta_{-1} = 0`, ``"tail"``
        :math:`\beta_{q+1} = 0`. Each constraint costs one degree of
        freedom.
    basis : {"power", "chebyshev"}
        Internal parameterisation. Both give the same ``lag_weights``;
        Chebyshev is better conditioned at high ``r``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n, q = 300, 8
    >>> x = rng.normal(size=n)
    >>> true = np.array([0.1 * i * (8 - i) for i in range(q + 1)])
    >>> y = np.convolve(x, true)[:n] + 1.0 + rng.normal(scale=0.3, size=n)
    >>> res = AlmonModel(pd.Series(y, name="y"), pd.Series(x, name="x"),
    ...                  q=8, r=2).fit()
    >>> res.lag_weights.round(2).tolist()
    [0.02, 0.71, 1.21, 1.51, 1.61, 1.51, 1.21, 0.71, 0.02]

    The weights used to build ``y`` were ``0.1 i (8 - i)``, a genuine
    quadratic: ``[0.0, 0.7, 1.2, 1.5, 1.6, 1.5, 1.2, 0.7, 0.0]``. When
    the restriction is true it is worth imposing, and
    ``polynomial_restriction_test`` does not reject (``p = 0.54``).
    """

    def __init__(
        self,
        y: npt.ArrayLike,
        x: npt.ArrayLike,
        q: int,
        r: int,
        endpoint: Endpoint = "none",
        basis: Basis = "power",
    ) -> None:
        q, r = int(q), int(r)
        if r >= q:
            raise ValueError(
                f"r={r} must be strictly smaller than q={q}: with r >= q the "
                "polynomial has as many free coefficients as the lags it is "
                "supposed to restrict, so it restricts nothing."
            )
        if r < 0 or q < 1:
            raise ValueError(f"q must be >= 1 and r >= 0, got q={q}, r={r}.")
        if endpoint not in ("none", "head", "tail", "both"):
            raise ValueError(
                f'endpoint must be "none", "head", "tail" or "both", got {endpoint!r}.'
            )
        n_constraints = {"none": 0, "head": 1, "tail": 1, "both": 2}[endpoint]
        if r + 1 - n_constraints < 1:
            raise ValueError(
                f"endpoint={endpoint!r} imposes {n_constraints} constraint(s) "
                f"on r+1={r + 1} coefficients, leaving nothing to estimate. "
                "Raise r."
            )

        y_arr, x_arr, index, y_name, x_names = check_series(y, x)
        if x_arr is None or x_arr.shape[1] != 1:
            raise ValueError("AlmonModel takes exactly one distributed-lag regressor.")
        self._y = y_arr
        self._x = x_arr[:, 0]
        self._index = index
        self._y_name = y_name
        self._x_name = x_names[0]
        self.q, self.r = q, r
        self.endpoint: Endpoint = endpoint
        self.basis: Basis = basis

        n_obs = y_arr.shape[0] - q
        n_params = 1 + (r + 1) - n_constraints
        if n_obs - n_params <= 0:
            raise ValueError(
                f"Not enough observations: q={q} leaves {n_obs} rows for "
                f"{n_params} parameters."
            )
        if q > y_arr.shape[0] / 3:
            warnings.warn(
                f"q={q} exceeds a third of the sample ({y_arr.shape[0]} "
                "observations). The lag window is eating the data; "
                "AlmonModel.select_order can pick q on an information "
                "criterion instead.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    def _lag_design(
        self, hold_back: int | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """``y`` and the ``(T - hold_back, q + 1)`` matrix of lags of ``x``.

        ``hold_back`` larger than ``q`` is what makes several candidate
        ``q`` comparable: they are then all estimated on the *same*
        rows. Comparing information criteria across different samples is
        the classic way to select the largest ``q`` by accident.
        """
        hb = self.q if hold_back is None else int(hold_back)
        lags = lag_matrix(self._x, self.q, first_lag=0)
        if hb > self.q:
            lags = lags[hb - self.q :]
        return self._y[hb:], lags.astype(np.float64)

    def _h_matrix(self) -> FloatArray:
        """``H``, mapping ``gamma`` to the ``q + 1`` lag weights."""
        node: FloatArray = np.asarray(
            np.arange(self.q + 1, dtype=np.float64) / float(self.q), dtype=np.float64
        )
        return _basis_matrix(self.q, self.r, self.basis, node)

    def _constraint_matrix(self) -> FloatArray:
        """Rows ``R`` such that ``R gamma = 0`` is the endpoint condition.

        Each row is the basis evaluated one step *outside* the window —
        at ``i = -1`` or ``i = q + 1`` — which is what "the weight has
        already died there" means. The node is scaled the same way as in
        ``H``, so the constraint follows the basis instead of silently
        assuming the power one.
        """
        rows: list[FloatArray] = []
        if self.endpoint in ("head", "both"):
            rows.append(
                _basis_matrix(self.q, self.r, self.basis, np.array([-1.0 / self.q]))[0]
            )
        if self.endpoint in ("tail", "both"):
            rows.append(
                _basis_matrix(
                    self.q, self.r, self.basis, np.array([(self.q + 1.0) / self.q])
                )[0]
            )
        if not rows:
            return np.zeros((0, self.r + 1), dtype=np.float64)
        return np.vstack(rows).astype(np.float64)

    def fit(
        self, cov_type: str = "nonrobust", hold_back: int | None = None
    ) -> AlmonResults:
        """Estimate by least squares on the Almon variables.

        Parameters
        ----------
        cov_type : {"nonrobust", "hac"}
            ``"hac"`` uses a Newey-West covariance, which is common on
            quarterly data where the disturbance is serially correlated
            even after the lag structure is modelled.
        hold_back : int, optional
            Drop this many initial observations instead of ``q``. Used by
            :meth:`select_order` to force a common sample.
        """
        if cov_type not in ("nonrobust", "hac"):
            raise ValueError(
                f'cov_type must be "nonrobust" or "hac", got {cov_type!r}.'
            )
        y_dep, lags = self._lag_design(hold_back)
        h_mat = self._h_matrix()
        transform = h_mat
        constraints = self._constraint_matrix()
        null_basis: FloatArray | None = None
        if constraints.shape[0]:
            from scipy.linalg import null_space

            null_basis = np.asarray(null_space(constraints), dtype=np.float64)
            transform = np.asarray(h_mat @ null_basis, dtype=np.float64)

        z_mat = lags @ transform
        design = np.column_stack([np.ones(y_dep.shape[0]), z_mat])
        if np.linalg.cond(design) > 1e12:
            warnings.warn(
                "The Almon variables are badly conditioned "
                f"(cond > 1e12) at q={self.q}, r={self.r}. Try "
                'basis="chebyshev", or a smaller r.',
                PyardlMethodologyWarning,
                stacklevel=2,
            )
        coefs, *_ = np.linalg.lstsq(design, y_dep, rcond=None)
        resid = y_dep - design @ coefs
        n_obs, k = design.shape
        xtx_inv = np.linalg.pinv(design.T @ design)
        if cov_type == "nonrobust":
            cov = float(resid @ resid) / (n_obs - k) * xtx_inv
        else:
            n_lags = int(np.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0)))
            meat = np.zeros((k, k), dtype=np.float64)
            xu = design * resid[:, None]
            meat += xu.T @ xu
            for lag in range(1, n_lags + 1):
                weight = 1.0 - lag / (n_lags + 1.0)
                gamma = xu[lag:].T @ xu[:-lag]
                meat += weight * (gamma + gamma.T)
            cov = xtx_inv @ meat @ xtx_inv
        # beta = H gamma is LINEAR, so the covariance transports exactly:
        # no delta method, no approximation.
        weights = transform @ coefs[1:]
        cov_weights = transform @ cov[1:, 1:] @ transform.T
        return AlmonResults(
            model=self,
            _coefs=coefs.astype(np.float64),
            _cov=np.asarray(cov, dtype=np.float64),
            _weights=np.asarray(weights, dtype=np.float64),
            _cov_weights=np.asarray(cov_weights, dtype=np.float64),
            _resid=resid.astype(np.float64),
            _lags=lags,
            _transform=np.asarray(transform, dtype=np.float64),
            cov_type=cov_type,
            extra={"null_space": null_basis},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def select_order(
        y: npt.ArrayLike,
        x: npt.ArrayLike,
        max_q: int = 12,
        max_r: int = 4,
        ic: str = "aic",
        endpoint: Endpoint = "none",
        basis: Basis = "power",
    ) -> pd.DataFrame:
        """Grid over ``(q, r)`` on a **common sample**, sorted by the criterion.

        Every candidate is estimated on ``t = max_q + 1 .. T``, not on
        its own longest available sample. Otherwise a larger ``q``
        competes on fewer observations, its likelihood is not comparable,
        and the criterion silently rewards whichever model happened to
        drop the hardest rows. Specs 02 §4 and 05 §3 both call this out
        because it is the mistake that keeps being made.
        """
        if ic not in ("aic", "bic", "hqic"):
            raise ValueError(f'ic must be "aic", "bic" or "hqic", got {ic!r}.')
        if max_r >= max_q:
            raise ValueError(f"max_r={max_r} must be smaller than max_q={max_q}.")
        rows = []
        for q in range(1, int(max_q) + 1):
            for r in range(0, min(int(max_r), q - 1) + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", PyardlMethodologyWarning)
                        res = AlmonModel(
                            y, x, q=q, r=r, endpoint=endpoint, basis=basis
                        ).fit(hold_back=int(max_q))
                except (ValueError, np.linalg.LinAlgError):
                    continue
                rows.append(
                    {
                        "q": q,
                        "r": r,
                        "nobs": res.nobs,
                        "aic": res.aic,
                        "bic": res.bic,
                        "hqic": res.hqic,
                    }
                )
        if not rows:  # pragma: no cover - only if every candidate fails
            raise ValueError("No candidate (q, r) could be estimated.")
        table = pd.DataFrame(rows).sort_values(ic).reset_index(drop=True)
        if table["nobs"].nunique() != 1:  # pragma: no cover - invariant
            raise AssertionError("Candidates were not estimated on a common sample.")
        return table


@dataclass(frozen=True)
class AlmonResults:
    """Outcome of an :class:`AlmonModel` fit."""

    model: AlmonModel
    _coefs: FloatArray = field(repr=False)
    _cov: FloatArray = field(repr=False)
    _weights: FloatArray = field(repr=False)
    _cov_weights: FloatArray = field(repr=False)
    _resid: FloatArray = field(repr=False)
    _lags: FloatArray = field(repr=False)
    _transform: FloatArray = field(repr=False)
    cov_type: str = "nonrobust"
    extra: dict[str, Any] = field(default_factory=dict)

    # -------------------------- basics --------------------------------
    @property
    def nobs(self) -> int:
        return int(self._resid.shape[0])

    @property
    def params_gamma(self) -> pd.Series:
        """The estimated polynomial coefficients, in the internal basis.

        Their *values* depend on ``basis`` and on the endpoint
        reparameterisation; :attr:`lag_weights` does not. Read the
        weights, not these.
        """
        names = ["const"] + [f"gamma{j}" for j in range(self._coefs.size - 1)]
        return pd.Series(self._coefs, index=names, name="coef")

    @property
    def intercept(self) -> float:
        return float(self._coefs[0])

    @property
    def lag_weights(self) -> pd.Series:
        r"""The :math:`\hat\beta_i`, ``i = 0 .. q``."""
        return pd.Series(
            self._weights,
            index=pd.RangeIndex(self.model.q + 1, name="lag"),
            name="beta",
        )

    @property
    def bse_lag_weights(self) -> pd.Series:
        return pd.Series(
            np.sqrt(np.diag(self._cov_weights)),
            index=pd.RangeIndex(self.model.q + 1, name="lag"),
            name="se",
        )

    @property
    def resid(self) -> pd.Series:
        return pd.Series(self._resid, name="resid")

    @property
    def ssr(self) -> float:
        return float(self._resid @ self._resid)

    @property
    def llf(self) -> float:
        n = self.nobs
        return float(-n / 2 * (np.log(2 * np.pi * self.ssr / n) + 1))

    @property
    def _k_ic(self) -> int:
        return int(self._coefs.size) + 1

    @property
    def aic(self) -> float:
        return float(-2 * self.llf + 2 * self._k_ic)

    @property
    def bic(self) -> float:
        return float(-2 * self.llf + np.log(self.nobs) * self._k_ic)

    @property
    def hqic(self) -> float:
        return float(-2 * self.llf + 2 * np.log(np.log(self.nobs)) * self._k_ic)

    # -------------------------- multipliers ---------------------------
    @property
    def impact_multiplier(self) -> float:
        return float(self._weights[0])

    @property
    def longrun_multiplier(self) -> float:
        r""" ":math:`\sum_i \beta_i`, a **linear** form — the error is exact.

        Unlike the Koyck long run, which is a ratio and needs the delta
        method, this one is a sum of the estimated weights. Its variance
        is :math:`\iota' V(\hat\beta) \iota` with no approximation
        anywhere.
        """
        return float(np.sum(self._weights))

    @property
    def se_longrun_multiplier(self) -> float:
        ones = np.ones(self._weights.size, dtype=np.float64)
        return float(np.sqrt(ones @ self._cov_weights @ ones))

    def interim_multiplier(self, h: int) -> tuple[float, float]:
        """Cumulated effect through lag ``h``, with its exact error."""
        if not 0 <= h <= self.model.q:
            raise ValueError(f"h must lie in [0, q={self.model.q}], got {h}.")
        selector = np.zeros(self._weights.size, dtype=np.float64)
        selector[: h + 1] = 1.0
        value = float(selector @ self._weights)
        se = float(np.sqrt(selector @ self._cov_weights @ selector))
        return value, se

    def mean_lag(self) -> tuple[float, float]:
        r""":math:`\sum i\beta_i / \sum \beta_i`, by the delta method.

        A ratio, so unlike the multipliers above it needs one. Returns
        ``nan`` when the weights sum to (numerically) zero: the average
        delay of an effect that nets out to nothing is not a quantity.
        """
        total = float(np.sum(self._weights))
        if abs(total) < 1e-12:
            return float("nan"), float("nan")
        lags = np.arange(self._weights.size, dtype=np.float64)
        numer = float(lags @ self._weights)
        value = numer / total
        grad = (lags * total - numer) / total**2
        se = float(np.sqrt(grad @ self._cov_weights @ grad))
        return value, se

    # -------------------------- tests ---------------------------------
    def _unrestricted(self) -> tuple[float, int]:
        """SSR and residual degrees of freedom of the free finite lag."""
        y_dep = self.model._y[self.model._y.shape[0] - self.nobs :]
        design = np.column_stack([np.ones(self.nobs), self._lags])
        coefs, *_ = np.linalg.lstsq(design, y_dep, rcond=None)
        resid = y_dep - design @ coefs
        return float(resid @ resid), self.nobs - design.shape[1]

    def polynomial_restriction_test(self) -> pd.Series:
        """Is the polynomial shape supported, or merely assumed?

        Compares this fit against the **unrestricted** finite distributed
        lag by an F test. A rejection means the smooth lag distribution
        the model produced is an artefact of the restriction rather than
        a feature of the data — at which point the honest moves are to
        raise ``r``, or to leave the polynomial family altogether for the
        free lag or an ARDL.

        Returns
        -------
        pandas.Series
            ``statistic``, ``df_num``, ``df_denom``, ``pvalue``.
        """
        ssr_free, df_free = self._unrestricted()
        df_num = int(self.model.q + 1) - int(self._coefs.size - 1)
        if df_num <= 0 or df_free <= 0:  # pragma: no cover - guarded by __init__
            return pd.Series(
                {
                    "statistic": float("nan"),
                    "df_num": float(df_num),
                    "df_denom": float(df_free),
                    "pvalue": float("nan"),
                }
            )
        stat = ((self.ssr - ssr_free) / df_num) / (ssr_free / df_free)
        return pd.Series(
            {
                "statistic": float(stat),
                "df_num": float(df_num),
                "df_denom": float(df_free),
                "pvalue": float(stats.f.sf(stat, df_num, df_free)),
            },
            name="polynomial_restriction",
        )

    def lags_are_jointly_zero(self) -> pd.Series:
        r"""F test of :math:`\beta_0 = \cdots = \beta_q = 0`.

        Equivalent to ``gamma = 0``, and tested on gamma because that is
        where the parameters are free: ``beta`` lives on an
        ``(r+1)``-dimensional surface, so a Wald test written on its
        ``q+1`` components would use a singular covariance.
        """
        gamma = self._coefs[1:]
        v_gamma = self._cov[1:, 1:]
        df_num = int(gamma.size)
        df_denom = self.nobs - int(self._coefs.size)
        stat = float(gamma @ np.linalg.pinv(v_gamma) @ gamma) / df_num
        return pd.Series(
            {
                "statistic": stat,
                "df_num": float(df_num),
                "df_denom": float(df_denom),
                "pvalue": float(stats.f.sf(stat, df_num, df_denom)),
            },
            name="lags_jointly_zero",
        )

    def endpoint_test(self) -> pd.Series:
        """F test of the endpoint constraints against the same model without them.

        Raises
        ------
        ValueError
            If the model was fitted with ``endpoint="none"``: there is
            no constraint to test.
        """
        if self.model.endpoint == "none":
            raise ValueError(
                'The model was fitted with endpoint="none": there is no '
                "constraint to test. Fit with endpoint='head'/'tail'/'both' "
                "and call this on that result."
            )
        free = AlmonModel(
            pd.Series(self.model._y, name=self.model._y_name),
            pd.Series(self.model._x, name=self.model._x_name),
            q=self.model.q,
            r=self.model.r,
            endpoint="none",
            basis=self.model.basis,
        ).fit(cov_type=self.cov_type)
        df_num = {"head": 1, "tail": 1, "both": 2}[self.model.endpoint]
        df_denom = free.nobs - int(free._coefs.size)
        stat = ((self.ssr - free.ssr) / df_num) / (free.ssr / df_denom)
        return pd.Series(
            {
                "statistic": float(stat),
                "df_num": float(df_num),
                "df_denom": float(df_denom),
                "pvalue": float(stats.f.sf(stat, df_num, df_denom)),
            },
            name="endpoint_restriction",
        )

    # -------------------------- diagnostics ---------------------------
    def diagnostics(self) -> pd.DataFrame:
        """Residual tests.

        Durbin-Watson is **valid here** and reported, unlike in the Koyck
        model: there is no lagged dependent variable among the
        regressors, which is the condition that invalidates it.
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson, jarque_bera

        resid = self._resid
        design = np.column_stack([np.ones(self.nobs), self._lags @ self._transform])
        lb = acorr_ljungbox(resid, lags=[max(1, min(10, self.nobs // 5))])
        jb_stat, jb_p, _, _ = jarque_bera(resid)
        bp = het_breuschpagan(resid, design)
        return pd.DataFrame(
            {
                "statistic": [
                    float(durbin_watson(resid)),
                    float(lb["lb_stat"].iloc[0]),
                    jb_stat,
                    bp[0],
                ],
                "pvalue": [
                    float("nan"),
                    float(lb["lb_pvalue"].iloc[0]),
                    jb_p,
                    bp[1],
                ],
            },
            index=["durbin_watson", "ljung_box", "jarque_bera", "breusch_pagan"],
        )

    # -------------------------- forecast ------------------------------
    def forecast(self, x_future: npt.ArrayLike) -> pd.DataFrame:
        """Forecast ``y`` over a given future path of ``x``.

        No recursion is needed — there is no lagged ``y`` — so the
        forecast is a direct weighted sum of future and past ``x``. The
        interval is the usual regression one; unlike the Koyck case
        there is no MA(1) term to carry forward.
        """
        path = np.atleast_1d(np.asarray(x_future, dtype=np.float64))
        if path.ndim != 1 or path.size == 0:
            raise ValueError("x_future must be a non-empty 1-D path for the regressor.")
        q = self.model.q
        history = np.concatenate([self.model._x[-q:], path]) if q else path
        sigma2 = self.ssr / (self.nobs - self._coefs.size)
        point = np.empty(path.size, dtype=np.float64)
        for h in range(path.size):
            window = history[h + q :: -1][: q + 1]
            point[h] = self.intercept + float(window @ self._weights)
        se = np.full(path.size, np.sqrt(sigma2), dtype=np.float64)
        return pd.DataFrame(
            {
                "forecast": point,
                "se": se,
                "lower": point - 1.959963984540054 * se,
                "upper": point + 1.959963984540054 * se,
            },
            index=pd.RangeIndex(1, path.size + 1, name="horizon"),
        )

    # -------------------------- bridges -------------------------------
    def to_fdl(self) -> pd.DataFrame:
        """The unrestricted finite lag on the same rows, for comparison.

        Printing the two side by side is the point of the Almon model:
        the free weights show what the data say on their own, the
        polynomial weights show what the restriction did to them.
        """
        y_dep = self.model._y[self.model._y.shape[0] - self.nobs :]
        design = np.column_stack([np.ones(self.nobs), self._lags])
        coefs, *_ = np.linalg.lstsq(design, y_dep, rcond=None)
        resid = y_dep - design @ coefs
        n, k = design.shape
        cov = float(resid @ resid) / (n - k) * np.linalg.pinv(design.T @ design)
        return pd.DataFrame(
            {"beta": coefs[1:], "se": np.sqrt(np.diag(cov)[1:])},
            index=pd.RangeIndex(self.model.q + 1, name="lag"),
        )

    def plot_lag_distribution(self) -> Any:
        """The weights with their pointwise band.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ImportError
            If matplotlib, an optional dependency, is missing.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Plotting requires matplotlib, an optional dependency. "
                "Install it with: pip install pyardl[plot]"
            ) from exc

        weights = self.lag_weights
        se = self.bse_lag_weights
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.fill_between(
            weights.index,
            weights - 1.96 * se,
            weights + 1.96 * se,
            alpha=0.2,
            color="C0",
            linewidth=0,
            label="95%",
        )
        ax.plot(weights.index, weights, "o-", color="C0", label="Almon")
        free = self.to_fdl()["beta"]
        ax.plot(free.index, free, "x--", color="0.5", label="free lag")
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        ax.set_xlabel("lag")
        ax.set_ylabel("weight")
        ax.set_title(f"Lag distribution - q={self.model.q}, r={self.model.r}")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        return fig

    def summary(self) -> str:
        """Publication-style report of the fit."""
        model = self.model
        test = self.polynomial_restriction_test()
        lines = [
            f"Almon polynomial distributed lag - {self.nobs} observations",
            f"  q={model.q}, r={model.r}, endpoint={model.endpoint!r}, "
            f"basis={model.basis!r}, cov_type={self.cov_type!r}",
            "",
            f"{'':>10}{'beta':>12}{'se':>12}{'t':>10}",
        ]
        se = self.bse_lag_weights
        for lag, weight in self.lag_weights.items():
            t = weight / se[lag] if se[lag] > 0 else float("nan")
            lines.append(f"    L{lag:<5}{weight:>12.4f}{se[lag]:>12.4f}{t:>10.3f}")
        lines += [
            "",
            f"  long run: {self.longrun_multiplier:.4f} "
            f"(se {self.se_longrun_multiplier:.4f})",
            f"  polynomial restriction: F({test['df_num']:.0f}, "
            f"{test['df_denom']:.0f}) = {test['statistic']:.3f}, "
            f"p = {test['pvalue']:.4f}",
        ]
        if test["pvalue"] < 0.05:
            lines.append(
                "  -> REJECTED: the smooth shape above is the restriction "
                "talking, not the data."
            )
        return "\n".join(lines)
