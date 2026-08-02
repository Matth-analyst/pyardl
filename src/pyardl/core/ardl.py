r"""ARDL(p, q_1, ..., q_k) estimation by ordinary least squares.

The model is

    y_t = det_t + sum_{i=1}^{p} phi_i y_{t-i}
          + sum_j sum_{i=0}^{q_j} beta_{j,i} x_{j,t-i} + eps_t

where ``det_t`` collects the deterministic terms (intercept, linear
trend). The ARDL is the general form from which the static regression,
the pure-differences model, finite distributed lags and the
error-correction representation all follow: one estimator, several views
(:meth:`ARDLResults.to_ecm`, :attr:`ARDLResults.longrun`,
:attr:`ARDLResults.adjustment`).

Two conventions are worth knowing:

- The design matrix columns are ordered ``const, trend, y.L1..y.Lp,
  x_j.L0..x_j.Lq_j, fixed regressors``, matching
  ``statsmodels.tsa.ardl.ARDL`` (coefficients agree to 1e-10).
- ``nobs`` is the size of the actual estimation sample (``T -
  hold_back``). ``statsmodels`` instead reports ``T - p`` and computes
  the log-likelihood and information criteria on that, even when
  ``max(q_j) > p``. pyardl uses the real estimation sample throughout,
  which is what makes information criteria comparable across candidates
  in :meth:`ARDL.select_order`. The two conventions coincide as soon as
  ``p >= max(q_j)``.

References
----------
Hendry, D. F., Pagan, A. R. & Sargan, J. D. (1984). "Dynamic
Specification", *Handbook of Econometrics*, vol. 2, ch. 18.
Pesaran, M. H. & Shin, Y. (1998). "An Autoregressive Distributed Lag
Modelling Approach to Cointegration Analysis", in *Econometrics and
Economic Theory in the 20th Century*, Cambridge University Press.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

from pyardl.core.transforms import (
    ARDLParams,
    ECMParams,
    ardl_to_ecm,
    half_life,
    longrun_coefs,
    longrun_covariance,
    speed_of_adjustment,
)
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.utils import check_series, lag_matrix

FloatArray = npt.NDArray[np.float64]

DetType = Literal["none", "const", "trend"]
CovType = Literal["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"]


def _parse_order(
    order: tuple[int, int | dict[str, int]] | int,
    x_names: tuple[str, ...],
) -> tuple[int, dict[str, int]]:
    """Normalise ``order`` into ``(p, {name: q_j})``."""
    if isinstance(order, int):
        if x_names:
            raise ValueError(
                "An integer order is only accepted for a pure AR model (no x "
                "regressors); with x, pass order=(p, q)."
            )
        p, q_spec = order, {}
    else:
        p, q_raw = order
        if isinstance(q_raw, dict):
            unknown = set(q_raw) - set(x_names)
            if unknown:
                raise ValueError(f"Unknown regressor names in order: {sorted(unknown)}")
            q_spec = {name: int(q_raw[name]) for name in x_names}
        else:
            q_spec = {name: int(q_raw) for name in x_names}
    if p < 0:
        raise ValueError("p must be >= 0.")
    for name, qj in q_spec.items():
        if qj < 0:
            raise ValueError(f"q[{name}] must be >= 0.")
    return int(p), q_spec


class ARDL:
    """ARDL(p, q_1, ..., q_k) model estimated by ordinary least squares.

    Parameters
    ----------
    y : array-like, shape (T,)
        Dependent variable.
    x : array-like, shape (T, k), optional
        Distributed-lag regressors; a DataFrame is recommended so that
        column names carry through to the output. ``None`` fits a pure
        AR(p) model.
    order : tuple (p, q)
        ``p`` is the number of lags of y (``p=0`` gives a finite
        distributed lag model with no dynamics). ``q`` is either an int,
        applied to every regressor, or a dict ``{name: q_j}``.
    det : {"none", "const", "trend"}
        Deterministic terms: none, an intercept, or an intercept plus a
        linear trend (``"trend"`` always includes the intercept).
    seasonal : bool
        Not implemented yet.
    fixed_regressors : array-like, shape (T, m), optional
        Variables entered without lags, such as dummies.
    hold_back : int, optional
        Number of initial observations excluded from estimation (at least
        ``max(p, max q_j)``). Used to force a common sample across
        candidate models during order selection.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> x = pd.DataFrame({"x": rng.normal(size=100).cumsum()})
    >>> y = pd.Series(rng.normal(size=100), name="y") + 0.5 * x["x"]
    >>> res = ARDL(y, x, order=(1, 1)).fit()
    >>> res.params.index.tolist()
    ['const', 'y.L1', 'x.L0', 'x.L1']
    """

    def __init__(
        self,
        y: npt.ArrayLike,
        x: npt.ArrayLike | None = None,
        order: tuple[int, int | dict[str, int]] | int = (1, 0),
        det: DetType = "const",
        seasonal: bool = False,
        fixed_regressors: npt.ArrayLike | None = None,
        hold_back: int | None = None,
    ) -> None:
        if seasonal:
            raise NotImplementedError("Seasonal dummies are not implemented yet.")
        if det not in ("none", "const", "trend"):
            raise ValueError('det must be "none", "const" or "trend".')

        y_arr, x_arr, index, y_name, x_names = check_series(y, x)
        self._y = y_arr
        self._x = x_arr
        self._index = index
        self._y_name = y_name
        self._x_names = x_names
        self.det: DetType = det

        self.p, self._q = _parse_order(order, x_names)
        self.q: tuple[int, ...] = tuple(self._q[name] for name in x_names)

        self._fixed: FloatArray | None = None
        self._fixed_names: tuple[str, ...] = ()
        if fixed_regressors is not None:
            fixed = np.asarray(fixed_regressors, dtype=np.float64)
            if fixed.ndim == 1:
                fixed = fixed[:, None]
            if fixed.shape[0] != y_arr.shape[0]:
                raise ValueError("fixed_regressors has a different length from y.")
            if isinstance(fixed_regressors, pd.DataFrame):
                self._fixed_names = tuple(str(c) for c in fixed_regressors.columns)
            else:
                self._fixed_names = tuple(f"z.{j}" for j in range(fixed.shape[1]))
            self._fixed = fixed

        start_required = max([self.p, *self.q]) if self.q else self.p
        if hold_back is None:
            hold_back = start_required
        elif hold_back < start_required:
            raise ValueError(
                f"hold_back={hold_back} is smaller than max(p, max q)={start_required}."
            )
        self.hold_back = int(hold_back)

        n_est = y_arr.shape[0] - self.hold_back
        n_params = (
            (0 if det == "none" else 1)
            + (1 if det == "trend" else 0)
            + self.p
            + sum(qj + 1 for qj in self.q)
            + len(self._fixed_names)
        )
        if n_est <= n_params:
            raise ValueError(
                f"Not enough observations: the estimation sample has {n_est} "
                f"points for {n_params} parameters."
            )

    # ------------------------------------------------------------------
    # Design matrix (column order matches statsmodels, see module docstring)
    # ------------------------------------------------------------------
    def _build_design(self) -> tuple[FloatArray, FloatArray, list[str]]:
        y, x = self._y, self._x
        n = y.shape[0]
        hb = self.hold_back
        cols: list[FloatArray] = []
        names: list[str] = []

        if self.det in ("const", "trend"):
            cols.append(np.ones(n - hb))
            names.append("const")
        if self.det == "trend":
            cols.append(np.arange(hb + 1, n + 1, dtype=np.float64))
            names.append("trend")

        if self.p > 0:
            y_lags = lag_matrix(y, hb, first_lag=1)[:, : self.p]
            cols.extend(y_lags.T)
            names.extend(f"{self._y_name}.L{i}" for i in range(1, self.p + 1))

        if x is not None:
            for j, name in enumerate(self._x_names):
                x_lags = lag_matrix(x[:, j], hb, first_lag=0)[:, : self.q[j] + 1]
                cols.extend(x_lags.T)
                names.extend(f"{name}.L{i}" for i in range(self.q[j] + 1))

        if self._fixed is not None:
            cols.extend(self._fixed[hb:].T)
            names.extend(self._fixed_names)

        design = np.column_stack(cols)
        y_dep = y[hb:]
        return design, y_dep, names

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def fit(
        self,
        cov_type: CovType = "nonrobust",
        cov_kwds: dict[str, int] | None = None,
    ) -> ARDLResults:
        """Fit the model by OLS and check the residuals for autocorrelation.

        Valid long-run inference in an ARDL requires enough lags for the
        errors to be white noise. A Ljung-Box test is therefore run
        automatically after every fit, and a
        :class:`~pyardl.exceptions.PyardlMethodologyWarning` is issued when
        it rejects. This is a condition of validity, not an option.

        Parameters
        ----------
        cov_type : {"nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"}
            Covariance estimator. ``HAC`` accepts ``cov_kwds={"nlags": m}``
            (Newey-West); a data-driven default is used otherwise.
        cov_kwds : dict, optional
            Extra arguments for the covariance estimator.

        Returns
        -------
        ARDLResults
        """
        results = self._fit(cov_type=cov_type, cov_kwds=cov_kwds)
        lb_p = results._ljungbox_pvalue()
        if lb_p < 0.05:
            warnings.warn(
                f"Autocorrelated residuals (Ljung-Box p={lb_p:.4f} < 0.05): "
                "long-run inference is not reliable. Increase p/q or revisit "
                "the specification.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )
        return results

    def _fit(
        self,
        cov_type: CovType = "nonrobust",
        cov_kwds: dict[str, int] | None = None,
    ) -> ARDLResults:
        """Fit without the autocorrelation check (used internally)."""
        design, y_dep, names = self._build_design()
        n_est, k = design.shape

        coefs, _, rank, _ = np.linalg.lstsq(design, y_dep, rcond=None)
        if rank < k:
            warnings.warn(
                "Singular design matrix (perfect collinearity): minimum-norm "
                "coefficients are returned and the covariance is unreliable.",
                PyardlMethodologyWarning,
                stacklevel=3,
            )
        resid = y_dep - design @ coefs
        ssr = float(resid @ resid)

        # inv(X'X) obtained from the QR factorisation, never by inverting
        # X'X directly, which is numerically far less stable.
        q_mat, r_mat = np.linalg.qr(design)
        r_inv = scipy.linalg.solve_triangular(r_mat, np.eye(k))
        xtx_inv = r_inv @ r_inv.T

        df_resid = n_est - k
        scale = ssr / df_resid
        if cov_type == "nonrobust":
            cov = scale * xtx_inv
        elif cov_type in ("HC0", "HC1", "HC2", "HC3"):
            u2 = resid**2
            if cov_type == "HC1":
                u2 = u2 * n_est / df_resid
            elif cov_type in ("HC2", "HC3"):
                leverage = np.sum(q_mat**2, axis=1)
                power = 1 if cov_type == "HC2" else 2
                u2 = u2 / (1.0 - leverage) ** power
            meat = (design * u2[:, None]).T @ design
            cov = xtx_inv @ meat @ xtx_inv
        elif cov_type == "HAC":
            nlags = (cov_kwds or {}).get(
                "nlags", int(np.floor(4 * (n_est / 100.0) ** (2.0 / 9.0)))
            )
            xu = design * resid[:, None]
            meat = (xu.T @ xu).astype(np.float64)
            for lag in range(1, nlags + 1):
                w = 1.0 - lag / (nlags + 1.0)
                gamma = xu[lag:].T @ xu[:-lag]
                meat += w * (gamma + gamma.T)
            cov = xtx_inv @ meat @ xtx_inv
        else:
            raise ValueError(f"Unknown cov_type: {cov_type!r}")

        return ARDLResults(
            model=self,
            _params=coefs.astype(np.float64),
            _cov_params=cov.astype(np.float64),
            _param_names=names,
            _resid=resid.astype(np.float64),
            _ssr=ssr,
            cov_type=cov_type,
        )

    # ------------------------------------------------------------------
    # Order selection
    # ------------------------------------------------------------------
    @staticmethod
    def select_order(
        y: npt.ArrayLike,
        x: npt.ArrayLike,
        max_p: int,
        max_q: int,
        ic: Literal["aic", "bic", "hq"] = "aic",
        search: Literal["grid", "per_variable"] = "grid",
        det: DetType = "const",
        min_p: int = 1,
    ) -> ARDLOrderSelection:
        """Select the lag orders by information criterion.

        All candidates are estimated on the **same** sample,
        ``t = max(max_p, max_q)+1 .. T``. This matters: comparing
        information criteria computed on different numbers of
        observations is meaningless, and it is an easy mistake to make
        when each candidate is allowed to use its own maximal sample. The
        selected model is then re-estimated on the largest sample its own
        order allows.

        Parameters
        ----------
        y, x : array-like
            Data. ``x`` is required here.
        max_p, max_q : int
            Grid bounds: ``p`` in ``min_p..max_p``, each ``q_j`` in
            ``0..max_q``.
        ic : {"aic", "bic", "hq"}
            Criterion used to rank candidates. All three are reported in
            the output table.
        search : {"grid", "per_variable"}
            ``"grid"`` explores the full cartesian product.
            ``"per_variable"`` optimises ``p`` and then each ``q_j`` in
            turn, which keeps the problem tractable when the number of
            regressors makes the full grid explode.
        det : {"none", "const", "trend"}
            Deterministic terms, see :class:`ARDL`.
        min_p : int
            Lower bound for ``p``.

        Returns
        -------
        ARDLOrderSelection
            Ranked table of candidates, best order, and the re-estimated
            best model.
        """
        if ic not in ("aic", "bic", "hq"):
            raise ValueError('ic must be "aic", "bic" or "hq".')
        _, x_arr, _, _, x_names = check_series(y, x)
        if x_arr is None:
            raise ValueError("select_order requires x regressors.")
        k = x_arr.shape[1]
        hold_back = max(max_p, max_q)

        def eval_candidate(p: int, q_tuple: tuple[int, ...]) -> dict[str, float]:
            q_dict = dict(zip(x_names, q_tuple, strict=True))
            res = ARDL(y, x, order=(p, q_dict), det=det, hold_back=hold_back)._fit()
            row: dict[str, float] = {"p": p}
            for name, qj in q_dict.items():
                row[f"q_{name}"] = qj
            row.update(
                aic=res.aic, bic=res.bic, hq=res.hqic, llf=res.llf, nobs=res.nobs
            )
            return row

        rows: list[dict[str, float]] = []
        if search == "grid":
            for p in range(min_p, max_p + 1):
                for q_tuple in product(range(max_q + 1), repeat=k):
                    rows.append(eval_candidate(p, q_tuple))
        elif search == "per_variable":
            current_p = max_p
            current_q = [max_q] * k
            seen: set[tuple[int, ...]] = set()

            def eval_and_log(p: int, q_t: tuple[int, ...]) -> float:
                key = (p, *q_t)
                row = eval_candidate(p, q_t)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
                return row[ic]

            for _ in range(10):  # iterate until the orders stop changing
                changed = False
                best_p = min(
                    range(min_p, max_p + 1),
                    key=lambda p: eval_and_log(p, tuple(current_q)),
                )
                if best_p != current_p:
                    current_p, changed = best_p, True
                for j in range(k):

                    def ic_for(qj: int, j: int = j, p: int = current_p) -> float:
                        trial = current_q.copy()
                        trial[j] = qj
                        return eval_and_log(p, tuple(trial))

                    best_qj = min(range(max_q + 1), key=ic_for)
                    if best_qj != current_q[j]:
                        current_q[j], changed = best_qj, True
                if not changed:
                    break
        else:
            raise ValueError('search must be "grid" or "per_variable".')

        table = pd.DataFrame(rows).sort_values(ic, kind="stable").reset_index(drop=True)
        best = table.iloc[0]
        best_p = int(best["p"])
        best_q = {name: int(best[f"q_{name}"]) for name in x_names}

        best_model = ARDL(y, x, order=(best_p, best_q), det=det).fit()
        return ARDLOrderSelection(
            table=table, ic=ic, best_order=(best_p, best_q), best_model=best_model
        )

    # ------------------------------------------------------------------
    # General-to-specific reduction
    # ------------------------------------------------------------------
    @staticmethod
    def gets(
        y: npt.ArrayLike,
        x: npt.ArrayLike,
        max_p: int,
        max_q: int,
        alpha: float = 0.05,
        det: DetType = "const",
    ) -> GETSResults:
        """General-to-specific reduction of an over-parameterised model.

        Starts from ``(max_p, max_q)`` and repeatedly drops the least
        significant *terminal* lag, as long as three conditions hold:
        its p-value exceeds ``alpha``, the residual diagnostics stay
        clean (Ljung-Box and Breusch-Pagan above 5%), and an F test of
        the accumulated restrictions against the general model does not
        reject. The full reduction path is recorded in
        :attr:`GETSResults.reduction_path`, so the sequence of decisions
        can be audited.

        Only the terminal lag of each variable is a candidate for
        removal, which keeps the lag structure contiguous and the result
        a genuine ARDL(p, q) model.

        Parameters
        ----------
        y, x : array-like
            Data.
        max_p, max_q : int
            Starting orders of the general model.
        alpha : float
            Significance level used both for dropping a lag and for the
            cumulated F test.
        det : {"none", "const", "trend"}
            Deterministic terms, see :class:`ARDL`.

        Returns
        -------
        GETSResults
        """
        _, x_arr, _, y_name, x_names = check_series(y, x)
        if x_arr is None:
            raise ValueError("gets requires x regressors.")
        hold_back = max(max_p, max_q)

        def fit_cand(p: int, q_list: list[int]) -> ARDLResults:
            q_dict = dict(zip(x_names, q_list, strict=True))
            return ARDL(y, x, order=(p, q_dict), det=det, hold_back=hold_back)._fit()

        general = fit_cand(max_p, [max_q] * len(x_names))
        current_p, current_q = max_p, [max_q] * len(x_names)
        current = general
        path: list[dict[str, object]] = []

        while True:
            # Terminal lags eligible for removal
            candidates: list[tuple[str, float]] = []
            pvals = current.pvalues
            if current_p >= 1:
                candidates.append(
                    (f"{y_name}.L{current_p}", float(pvals[f"{y_name}.L{current_p}"]))
                )
            for j, name in enumerate(x_names):
                if current_q[j] >= 1:
                    candidates.append(
                        (
                            f"{name}.L{current_q[j]}",
                            float(pvals[f"{name}.L{current_q[j]}"]),
                        )
                    )
            if not candidates:
                break
            drop_name, drop_p = max(candidates, key=lambda c: c[1])
            if drop_p <= alpha:
                break

            # Try the reduction
            trial_p, trial_q = current_p, current_q.copy()
            if drop_name.startswith(f"{y_name}.L"):
                trial_p -= 1
            else:
                var = drop_name.rsplit(".L", 1)[0]
                trial_q[x_names.index(var)] -= 1
            trial = fit_cand(trial_p, trial_q)

            lb_p = trial._ljungbox_pvalue()
            bp_p = trial._breuschpagan_pvalue()
            f_p = _f_test_nested(general, trial)
            ok = lb_p > 0.05 and bp_p > 0.05 and f_p > alpha
            path.append(
                {
                    "dropped": drop_name,
                    "pvalue": drop_p,
                    "ljungbox_p": lb_p,
                    "breuschpagan_p": bp_p,
                    "cumulative_f_p": f_p,
                    "accepted": ok,
                    "aic": trial.aic,
                }
            )
            if not ok:
                break
            current_p, current_q, current = trial_p, trial_q, trial

        final_q = dict(zip(x_names, current_q, strict=True))
        final = ARDL(y, x, order=(current_p, final_q), det=det).fit()
        return GETSResults(
            final_model=final,
            final_order=(current_p, final_q),
            reduction_path=pd.DataFrame(
                path,
                columns=[
                    "dropped",
                    "pvalue",
                    "ljungbox_p",
                    "breuschpagan_p",
                    "cumulative_f_p",
                    "accepted",
                    "aic",
                ],
            ),
            general_model=general,
        )


def _f_test_nested(general: ARDLResults, restricted: ARDLResults) -> float:
    """p-value of the F test of the accumulated restrictions."""
    if restricted.nobs != general.nobs:
        raise ValueError("Nested F test requires the same estimation sample.")
    n_restr = len(general.params) - len(restricted.params)
    if n_restr == 0:
        return 1.0
    df2 = general.nobs - len(general.params)
    f_stat = ((restricted.ssr - general.ssr) / n_restr) / (general.ssr / df2)
    from scipy.stats import f as f_dist

    return float(f_dist.sf(f_stat, n_restr, df2))


@dataclass(frozen=True)
class ARDLOrderSelection:
    """Outcome of :meth:`ARDL.select_order`.

    Attributes
    ----------
    table : pandas.DataFrame
        All candidates, sorted by the selection criterion, with AIC, BIC,
        HQ, log-likelihood and sample size for each.
    ic : str
        Criterion used for the ranking.
    best_order : tuple
        ``(p, {name: q_j})`` of the selected model.
    best_model : ARDLResults
        Selected model, re-estimated on its own maximal sample.
    """

    table: pd.DataFrame
    ic: str
    best_order: tuple[int, dict[str, int]]
    best_model: ARDLResults

    def top(self, n: int = 5) -> pd.DataFrame:
        """Return the ``n`` best candidates.

        Inspecting a few near-optimal specifications, rather than trusting
        the single best one, is good practice: information criteria often
        separate the top candidates by very little.
        """
        return self.table.head(n)


@dataclass(frozen=True)
class GETSResults:
    """Outcome of :meth:`ARDL.gets`.

    Attributes
    ----------
    final_model : ARDLResults
        Reduced model, re-estimated on its maximal sample.
    final_order : tuple
        ``(p, {name: q_j})`` reached at the end of the reduction.
    reduction_path : pandas.DataFrame
        One row per attempted removal, with the p-value of the dropped
        lag, the residual diagnostics, the cumulated F test and whether
        the step was accepted.
    general_model : ARDLResults
        The initial over-parameterised model.
    """

    final_model: ARDLResults
    final_order: tuple[int, dict[str, int]]
    reduction_path: pd.DataFrame
    general_model: ARDLResults


@dataclass
class ARDLResults:
    """Results of an ARDL fit.

    Besides the usual regression output (``params``, ``bse``, ``tvalues``,
    ``pvalues``, ``resid``, ``aic``/``bic``/``hqic``, ``rsquared``), this
    object exposes the error-correction views of the same fit:
    :meth:`to_ecm`, :attr:`longrun` and :attr:`adjustment`.
    """

    model: ARDL
    _params: FloatArray
    _cov_params: FloatArray
    _param_names: list[str]
    _resid: FloatArray
    _ssr: float
    cov_type: str
    _cache: dict[str, object] = field(default_factory=dict, repr=False)

    # -------------------------- basic statistics ----------------------
    @property
    def params(self) -> pd.Series:
        return pd.Series(self._params, index=self._param_names, name="coef")

    @property
    def cov_params_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._cov_params, index=self._param_names, columns=self._param_names
        )

    @property
    def bse(self) -> pd.Series:
        return pd.Series(
            np.sqrt(np.diag(self._cov_params)), index=self._param_names, name="se"
        )

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        from scipy.stats import t as t_dist

        df = self.nobs - len(self._params)
        return pd.Series(
            2 * t_dist.sf(np.abs(self.tvalues), df),
            index=self._param_names,
            name="pvalue",
        )

    @property
    def resid(self) -> pd.Series:
        index = (
            self.model._index[self.model.hold_back :]
            if self.model._index is not None
            else pd.RangeIndex(self.model.hold_back, len(self.model._y))
        )
        return pd.Series(self._resid, index=index, name="resid")

    @property
    def fittedvalues(self) -> pd.Series:
        return pd.Series(
            self.model._y[self.model.hold_back :] - self._resid,
            index=self.resid.index,
            name="fitted",
        )

    @property
    def nobs(self) -> int:
        """Size of the actual estimation sample.

        Differs from the statsmodels convention when ``max(q) > p``; see
        the module documentation.
        """
        return len(self._resid)

    @property
    def ssr(self) -> float:
        return self._ssr

    @property
    def sigma2(self) -> float:
        """Variance ML des erreurs : SSR / nobs."""
        return self._ssr / self.nobs

    @property
    def llf(self) -> float:
        return float(-self.nobs / 2 * (np.log(2 * np.pi * self.sigma2) + 1))

    @property
    def _k_ic(self) -> int:
        return len(self._params) + 1  # + sigma2, convention statsmodels

    @property
    def aic(self) -> float:
        return -2 * self.llf + 2 * self._k_ic

    @property
    def bic(self) -> float:
        return -2 * self.llf + float(np.log(self.nobs)) * self._k_ic

    @property
    def hqic(self) -> float:
        return -2 * self.llf + 2 * self._k_ic * float(np.log(np.log(self.nobs)))

    @property
    def rsquared(self) -> float:
        y_dep = self.model._y[self.model.hold_back :]
        tss = float(np.sum((y_dep - y_dep.mean()) ** 2))
        return 1.0 - self._ssr / tss

    @property
    def rsquared_adj(self) -> float:
        k = len(self._params)
        return 1.0 - (1.0 - self.rsquared) * (self.nobs - 1) / (self.nobs - k)

    # -------------------------- dynamic stability ---------------------
    @property
    def ar_roots(self) -> npt.NDArray[np.complex128]:
        """Roots of the autoregressive polynomial 1 - phi_1 L - ... - phi_p L^p."""
        phi = self._phi_values()
        if phi.shape[0] == 0:
            return np.array([], dtype=np.complex128)
        return np.roots(np.concatenate(([1.0], -phi))[::-1]).astype(np.complex128)

    @property
    def is_stable(self) -> bool:
        """Whether all autoregressive roots lie outside the unit circle.

        If they do not, the dynamics are explosive or contain a unit root,
        the long-run quantities have no equilibrium interpretation, and a
        :class:`~pyardl.exceptions.PyardlMethodologyWarning` is issued.
        """
        roots = self.ar_roots
        if roots.shape[0] == 0:
            return True
        stable = bool(np.all(np.abs(roots) > 1.0))
        if not stable:
            warnings.warn(
                "Unstable dynamics: at least one autoregressive root lies on "
                "or inside the unit circle, so the long-run quantities have no "
                "equilibrium interpretation.",
                PyardlMethodologyWarning,
                stacklevel=2,
            )
        return stable

    # -------------------------- error-correction views ----------------
    def _phi_values(self) -> FloatArray:
        p = self.model.p
        y_name = self.model._y_name
        if p == 0:
            return np.array([], dtype=np.float64)
        return np.array([self.params[f"{y_name}.L{i}"] for i in range(1, p + 1)])

    @property
    def ardl_params(self) -> ARDLParams:
        """Parameters packaged for the error-correction algebra.

        Can be passed directly to
        :func:`~pyardl.core.transforms.ardl_to_ecm` and to the long-run
        helpers; the covariance matrix is carried along.
        """
        model = self.model
        if model.p == 0:
            raise ValueError(
                "p=0: with no lagged y there is no error-correction form "
                "(this is a pure distributed-lag model)."
            )
        if model._fixed is not None:
            raise NotImplementedError(
                "The error-correction views are not defined when the model "
                "has fixed regressors: they are neither phi nor beta "
                "coefficients."
            )
        beta = []
        pos = (
            (1 if model.det in ("const", "trend") else 0)
            + (1 if model.det == "trend" else 0)
            + model.p
        )
        for qj in model.q:
            beta.append(self._params[pos : pos + qj + 1])
            pos += qj + 1
        return ARDLParams(
            p=model.p,
            q=model.q,
            phi=self._phi_values(),
            beta=tuple(beta),
            const=(
                float(self.params["const"]) if model.det in ("const", "trend") else 0.0
            ),
            trend=(float(self.params["trend"]) if model.det == "trend" else 0.0),
            has_const=model.det in ("const", "trend"),
            has_trend=model.det == "trend",
            x_names=model._x_names,
            cov_params=self._cov_params,
        )

    def to_ecm(self) -> ECMParams:
        """Return the error-correction view of this fit.

        An exact reparameterisation: same data, same residuals, same fit,
        expressed in terms of the adjustment speed and the level
        coefficients instead of the raw lag polynomials.
        """
        return ardl_to_ecm(self.ardl_params)

    @property
    def longrun(self) -> pd.DataFrame:
        """Long-run coefficients with delta-method standard errors.

        Returns
        -------
        pandas.DataFrame
            One row per regressor, with columns ``theta`` and ``se``.
        """
        params = self.ardl_params
        theta = longrun_coefs(params)
        cov_theta = longrun_covariance(params)
        return pd.DataFrame(
            {"theta": theta, "se": np.sqrt(np.diag(cov_theta))}, index=theta.index
        )

    @property
    def adjustment(self) -> pd.Series:
        """Adjustment speed, its standard error, and the half-life.

        Returns
        -------
        pandas.Series
            ``lambda`` (negative under error correction), ``se`` and
            ``half_life``, the number of periods needed to absorb half of
            a shock.
        """
        params = self.ardl_params
        lam = speed_of_adjustment(params)
        # var(lam) = 1' V_phi 1, since lam = -1 + sum(phi)
        p = self.model.p
        n_lead = (1 if self.model.det in ("const", "trend") else 0) + (
            1 if self.model.det == "trend" else 0
        )
        v_phi = self._cov_params[n_lead : n_lead + p, n_lead : n_lead + p]
        se_lam = float(np.sqrt(np.ones(p) @ v_phi @ np.ones(p)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hl = half_life(params)
        return pd.Series(
            {"lambda": lam, "se": se_lam, "half_life": hl}, name="adjustment"
        )

    # -------------------------- diagnostics ---------------------------
    def _ljungbox_lags(self) -> int:
        return max(1, min(10, self.nobs // 5))

    def _ljungbox_pvalue(self) -> float:
        lb = acorr_ljungbox(self._resid, lags=[self._ljungbox_lags()])
        return float(lb["lb_pvalue"].iloc[0])

    def _breuschpagan_pvalue(self) -> float:
        design, _, _ = self.model._build_design()
        if not (design[:, 0] == 1.0).all():
            design = np.column_stack([np.ones(design.shape[0]), design])
        return float(het_breuschpagan(self._resid, design)[1])

    def diagnostics(self) -> pd.DataFrame:
        """Residual diagnostics: Ljung-Box, Jarque-Bera and Breusch-Pagan.

        Returns
        -------
        pandas.DataFrame
            Test statistic and p-value for autocorrelation, normality and
            heteroskedasticity of the residuals.
        """
        lb_lags = self._ljungbox_lags()
        lb = acorr_ljungbox(self._resid, lags=[lb_lags])
        jb_stat, jb_p, _, _ = jarque_bera(self._resid)
        bp_p = self._breuschpagan_pvalue()
        return pd.DataFrame(
            {
                "statistic": [
                    float(lb["lb_stat"].iloc[0]),
                    float(jb_stat),
                    np.nan,
                ],
                "pvalue": [float(lb["lb_pvalue"].iloc[0]), float(jb_p), bp_p],
            },
            index=[f"Ljung-Box({lb_lags})", "Jarque-Bera", "Breusch-Pagan"],
        )

    def stability(self, alpha: float = 0.05) -> pd.DataFrame:
        """CUSUM and CUSUM-of-squares tests for parameter constancy.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level of the boundaries. One of 0.10, 0.05, 0.01.

        Returns
        -------
        pandas.DataFrame
            One row per test, with ``stable``, ``max_excess`` and
            ``first_crossing``.

        Notes
        -----
        Every long-run coefficient this object reports assumes the
        parameters did not move over the sample. When they did, the
        long-run estimate is a blend of two regimes rather than an
        equilibrium. These tests check that assumption; see
        :mod:`pyardl.diagnostics` for what each one can and cannot see.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from pyardl.core.ardl import ARDL
        >>> rng = np.random.default_rng(0)
        >>> x = pd.Series(rng.standard_normal(150), name="x")
        >>> y = pd.Series(np.zeros(150), name="y")
        >>> for t in range(1, 150):
        ...     y.iloc[t] = 0.5 * y.iloc[t - 1] + x.iloc[t] + rng.standard_normal()
        >>> res = ARDL(y, x, order=(1, 1))._fit()
        >>> res.stability()["stable"].tolist()
        [True, True]
        """
        from pyardl.diagnostics import stability_tests

        design, y_dep, _ = self.model._build_design()
        return stability_tests(y_dep, design, alpha=alpha)

    # -------------------------- presentation --------------------------
    def summary(self) -> str:
        """Return a publication-style summary of the fit as a string."""
        q_desc = ", ".join(
            f"{name}:{qj}"
            for name, qj in zip(self.model._x_names, self.model.q, strict=True)
        )
        header = (
            f"ARDL({self.model.p}; {q_desc}) — det={self.model.det}, "
            f"cov={self.cov_type}\n"
            f"nobs={self.nobs}, R2={self.rsquared:.4f}, "
            f"R2_adj={self.rsquared_adj:.4f}\n"
            f"llf={self.llf:.4f}, AIC={self.aic:.4f}, BIC={self.bic:.4f}, "
            f"HQIC={self.hqic:.4f}\n"
            f"stable={self.is_stable}\n"
        )
        table = pd.DataFrame(
            {
                "coef": self.params,
                "se": self.bse,
                "t": self.tvalues,
                "P>|t|": self.pvalues,
            }
        )
        return header + str(table.to_string(float_format=lambda v: f"{v: .6f}"))
