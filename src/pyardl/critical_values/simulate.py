r"""Monte Carlo engine for bounds-test critical values.

Simulates the null distribution of the F and t statistics for a given
deterministic case, number of regressors and sample size, following the
design of Pesaran, Shin & Smith (2001):

- under the null, ``y`` is a random walk (``lam = 0``, ``gamma = 0``);
- for the lower bound, the regressors are i.i.d. draws (all I(0));
- for the upper bound, they are independent random walks (all I(1)).

Useful when a configuration is not tabulated anywhere: an unusual
significance level, more regressors than the published tables cover, or
an arbitrary sample size. It is also how the shipped tables were
independently cross-checked.

Least squares is done through a batched QR factorisation, never by
inverting X'X. The random generator takes an explicit seed, and every
simulation parameter is recorded on the result object so that a set of
critical values can always be traced back to how it was produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_CASE_DET = {1: "none", 2: "const", 3: "const", 4: "trend", 5: "trend"}
_CASE_RESTRICTED = {2: "const", 4: "trend"}


@dataclass(frozen=True)
class SimulatedBounds:
    """Simulated quantiles of the bounds-test statistics.

    Every simulation parameter is recorded on the object, so a set of
    critical values always carries enough information to be reproduced
    exactly.
    """

    case: int
    k: int
    t_obs: int
    n_sims: int
    seed: int
    i1: bool  # True for the all-I(1) bound, False for the all-I(0) bound
    chunk: int  # draws are generated in batches, so the batch size is part
    # of what makes a run reproducible
    alphas: tuple[float, ...]
    f_quantiles: dict[float, float]
    t_quantiles: dict[float, float]
    f_stats: FloatArray = field(repr=False)
    t_stats: FloatArray = field(repr=False)
    # Third statistic of Sam, McNown & Goh (2019): gamma = 0. Added after
    # the object was first released, so it carries defaults and every
    # earlier field keeps its position.
    f_indep_quantiles: dict[float, float] = field(default_factory=dict)
    f_indep_stats: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)

    def f_cv(self, alpha: float) -> float:
        """Critical value of the F statistic at level ``alpha``."""
        return self.f_quantiles[alpha]

    def f_indep_cv(self, alpha: float) -> float:
        """Critical value of ``F_indep`` at level ``alpha``.

        The third test of Sam, McNown & Goh (2019): the levels of the
        independent variables are jointly zero.
        """
        return self.f_indep_quantiles[alpha]

    def t_cv(self, alpha: float) -> float:
        """Critical value of the t statistic at level ``alpha``.

        The t test is left-tailed, so this is the ``alpha`` quantile.
        """
        return self.t_quantiles[alpha]


def simulate_bounds(
    case: int,
    k: int,
    t_obs: int = 1000,
    n_sims: int = 40_000,
    seed: int = 0,
    i1: bool = True,
    alphas: tuple[float, ...] = (0.10, 0.05, 0.025, 0.01),
    chunk: int = 2_000,
) -> SimulatedBounds:
    """Simulate critical values for the bounds test under the null.

    Parameters
    ----------
    case : int
        Deterministic case, 1 to 5. In cases 2 and 4 the restricted
        deterministic term is part of the tested vector.
    k : int
        Number of regressors; ``k = 0`` tests ``y_{t-1}`` alone.
    t_obs : int
        Length of each simulated series. 1000 reproduces the asymptotic
        convention; use 30 to 80 for small-sample bounds.
    n_sims : int
        Number of replications. More replications means tighter
        quantiles: the Monte Carlo error falls with the square root.
    seed : int
        Seed of the random generator; recorded on the result.
    i1 : bool
        ``True`` generates I(1) regressors (upper bound), ``False``
        i.i.d. ones (lower bound).
    alphas : tuple of float
        Significance levels at which quantiles are returned.
    chunk : int
        Batch size of the vectorised least squares; memory use is
        roughly ``chunk * t_obs * (k + 3)``. Since draws are generated in
        batches, the same ``(seed, n_sims, chunk)`` always yields exactly
        the same statistics.

    Returns
    -------
    SimulatedBounds
        Quantiles for both statistics, plus the full set of draws.

    Examples
    --------
    >>> sb = simulate_bounds(case=3, k=1, t_obs=200, n_sims=200, seed=42)
    >>> sb.seed, sb.n_sims, sb.case, sb.i1
    (42, 200, 3, True)
    >>> 0 < sb.f_cv(0.05) < 20
    True
    """
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case must be between 1 and 5, got {case}.")
    if k < 0:
        raise ValueError("k must be >= 0.")
    if t_obs < 20:
        raise ValueError("t_obs must be >= 20.")

    rng = np.random.default_rng(seed)
    det = _CASE_DET[case]
    n_det = {"none": 0, "const": 1, "trend": 2}[det]
    n_restr = k + 1 + (1 if case in _CASE_RESTRICTED else 0)
    k_par = n_det + 1 + k
    n_eff = t_obs - 1  # Δy_t, t = 2..T
    lam_pos = n_det  # position de y_{t-1} dans le design

    # Deterministic columns, identical across replications
    det_cols = np.empty((n_eff, n_det))
    if n_det >= 1:
        det_cols[:, 0] = 1.0
    if n_det == 2:
        det_cols[:, 1] = np.arange(2, t_obs + 1, dtype=np.float64)

    # Restricted design: only the deterministics that are not tested
    restr_idx = list(range(n_det))
    if case in _CASE_RESTRICTED:
        restr_idx = restr_idx[:-1]  # the last deterministic term is tested

    # F_indep keeps y_{t-1} and drops the levels of the regressors (plus
    # the restricted deterministic under cases 2 and 4, which belongs to
    # the cointegrating vector). k restrictions, or k+1 in those cases.
    n_restr_indep = k + (1 if case in _CASE_RESTRICTED else 0)
    indep_restr_idx = [*restr_idx, lam_pos]

    f_stats = np.empty(n_sims)
    t_stats = np.empty(n_sims)
    f_indep_stats = np.empty(n_sims)

    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        eps = rng.standard_normal((m, t_obs))
        y = np.cumsum(eps, axis=1)  # random walk under the null
        dy = np.diff(y, axis=1)  # = eps[:, 1:]
        y_lag = y[:, :-1]

        design = np.empty((m, n_eff, k_par))
        design[:, :, :n_det] = det_cols
        design[:, :, lam_pos] = y_lag
        if k > 0:
            x_innov = rng.standard_normal((m, t_obs, k))
            x = np.cumsum(x_innov, axis=1) if i1 else x_innov
            design[:, :, lam_pos + 1 :] = x[:, :-1, :]

        # --- unrestricted regression (batched QR) ---
        q_u, r_u = np.linalg.qr(design)
        qty = np.einsum("stk,st->sk", q_u, dy)
        coefs = np.linalg.solve(r_u, qty[:, :, None])[:, :, 0]
        ssr_u = np.einsum("st,st->s", dy, dy) - np.einsum("sk,sk->s", qty, qty)

        # --- restricted regression (null imposed) ---
        if restr_idx:
            q_r, _ = np.linalg.qr(design[:, :, restr_idx])
            qty_r = np.einsum("stk,st->sk", q_r, dy)
            ssr_r = np.einsum("st,st->s", dy, dy) - np.einsum("sk,sk->s", qty_r, qty_r)
        else:
            ssr_r = np.einsum("st,st->s", dy, dy)

        # --- restricted regression for F_indep (gamma = 0) ---
        if n_restr_indep > 0:
            q_i, _ = np.linalg.qr(design[:, :, indep_restr_idx])
            qty_i = np.einsum("stk,st->sk", q_i, dy)
            ssr_i = np.einsum("st,st->s", dy, dy) - np.einsum("sk,sk->s", qty_i, qty_i)
        else:
            ssr_i = ssr_u

        df = n_eff - k_par
        f_stats[done : done + m] = ((ssr_r - ssr_u) / n_restr) / (ssr_u / df)
        if n_restr_indep > 0:
            f_indep_stats[done : done + m] = ((ssr_i - ssr_u) / n_restr_indep) / (
                ssr_u / df
            )
        else:
            f_indep_stats[done : done + m] = np.nan

        # --- t statistic on y_{t-1}: standard error from R^{-1} ---
        r_inv = np.linalg.solve(r_u, np.broadcast_to(np.eye(k_par), r_u.shape))
        xtx_inv_lam = np.einsum("sj,sj->s", r_inv[:, lam_pos, :], r_inv[:, lam_pos, :])
        se_lam = np.sqrt(ssr_u / df * xtx_inv_lam)
        t_stats[done : done + m] = coefs[:, lam_pos] / se_lam

        done += m

    f_q = {a: float(np.quantile(f_stats, 1 - a)) for a in alphas}
    t_q = {a: float(np.quantile(t_stats, a)) for a in alphas}
    f_i_q = (
        {a: float(np.quantile(f_indep_stats, 1 - a)) for a in alphas}
        if n_restr_indep > 0
        else {}
    )
    return SimulatedBounds(
        case=case,
        k=k,
        t_obs=t_obs,
        n_sims=n_sims,
        seed=seed,
        i1=i1,
        chunk=chunk,
        alphas=tuple(alphas),
        f_quantiles=f_q,
        t_quantiles=t_q,
        f_stats=f_stats,
        t_stats=t_stats,
        f_indep_quantiles=f_i_q,
        f_indep_stats=f_indep_stats,
    )
