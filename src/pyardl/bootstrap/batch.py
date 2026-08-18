r"""Batched estimation of the error-correction model.

The bootstrap re-estimates the same specification on ``B`` regenerated
samples. Those fits share everything except the data: identical column
layout, identical dimensions, identical tested vector. Only the numbers
differ.

That is exactly the shape of problem NumPy's stacked linear algebra is
for. Instead of ``B`` calls into LAPACK on matrices of a few hundred
rows — where the per-call overhead rivals the arithmetic — one call
handles the whole stack:

.. math::

    X = QR, \qquad \hat\beta = R^{-1} Q' y

**QR, never the normal equations.** Forming :math:`(X'X)^{-1}` squares
the condition number of the design; on lagged levels of integrated
series, which is precisely what an error-correction model regresses on,
that is not a theoretical worry. The same rule governs every estimator
in this library.

The design layout is a mirror of the single-sample estimator, column for
column, and a test asserts that the two agree — otherwise the bootstrap
would silently be testing a different model from the one the observed
statistic came from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]

__all__ = ["batch_uecm_statistics"]

# Imported, never re-declared: the deterministic layout of the five
# cases has exactly one definition in this library, and the batched
# estimator must not be able to drift from it.
from pyardl.bounds.pss import (  # noqa: E402
    _CASE_DET,
    _CASE_RESTRICTED_DET,
)


def _build_designs(
    y: FloatArray,
    x: FloatArray,
    p: int,
    q: tuple[int, ...],
    case: int,
) -> tuple[FloatArray, FloatArray, list[int], int]:
    """Stack the design matrices of every replication.

    The column order mirrors the single-sample estimator exactly, and
    the names are carried alongside so the tested positions are *derived*
    rather than assumed — the same way the scalar estimator derives them.

    Returns
    -------
    design : numpy.ndarray, shape (B, n_est, k_par)
    target : numpy.ndarray, shape (B, n_est)
    tested : list of int
        Column positions entering the F test.
    lam_pos : int
        Column position of the lagged level of ``y``.
    """
    n_rep, n_obs = y.shape
    k = x.shape[2]
    start = max([p, *q]) if q else p
    dy = np.diff(y, axis=1)
    dx = np.diff(x, axis=1)

    cols: list[FloatArray] = []
    names: list[str] = []
    tested_names: list[str] = []

    det = _CASE_DET[case]
    if det in ("const", "trend"):
        cols.append(np.ones((n_rep, n_obs - start)))
        names.append("const")
    if det == "trend":
        trend = np.arange(start + 1, n_obs + 1, dtype=np.float64)
        cols.append(np.broadcast_to(trend, (n_rep, n_obs - start)).copy())
        names.append("trend")
    if case in _CASE_RESTRICTED_DET:
        # Under cases 2 and 4 the restricted deterministic term is part
        # of the tested vector, giving k+2 restrictions instead of k+1.
        tested_names.append(_CASE_RESTRICTED_DET[case])

    lam_name = "y.L1"
    cols.append(y[:, start - 1 : n_obs - 1])
    names.append(lam_name)
    tested_names.append(lam_name)

    for j in range(k):
        if q[j] == 0:
            # Contemporaneous level, the q_j = 0 convention of the
            # single-sample estimator.
            cols.append(x[:, start:n_obs, j])
            names.append(f"x{j}.L0")
        else:
            cols.append(x[:, start - 1 : n_obs - 1, j])
            names.append(f"x{j}.L1")
        tested_names.append(names[-1])

    for i in range(1, p):
        cols.append(dy[:, start - i - 1 : n_obs - i - 1])
        names.append(f"D.y.L{i}")
    for j in range(k):
        for i in range(q[j]):
            cols.append(dx[:, start - i - 1 : n_obs - i - 1, j])
            names.append(f"D.x{j}.L{i}")

    design = np.stack(cols, axis=2)
    target = dy[:, start - 1 :]
    tested = [names.index(name) for name in tested_names]
    return design, target, tested, names.index(lam_name)


def batch_uecm_statistics(
    y: FloatArray,
    x: FloatArray,
    p: int,
    q: tuple[int, ...],
    case: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Wald F and t statistics for a stack of regenerated samples.

    Parameters
    ----------
    y : numpy.ndarray, shape (B, T)
        Regenerated dependent variables, one row per replication.
    x : numpy.ndarray, shape (B, T, k)
        Regenerated regressors.
    p, q : ...
        Lag orders of the error-correction model.
    case : int
        Deterministic case, 1 to 5.

    Returns
    -------
    f_stat : numpy.ndarray, shape (B,)
    t_stat : numpy.ndarray, shape (B,)
    ok : numpy.ndarray of bool, shape (B,)
        ``False`` where the replication could not be estimated — a
        singular design, or a non-finite statistic. Those replications
        are dropped by the caller and counted, never replaced.

    Notes
    -----
    The two statistics are the ones of the classical test:

    .. math::

        F = \frac{\hat\theta' \hat V_\theta^{-1} \hat\theta}{m},
        \qquad t = \frac{\hat\lambda}{\mathrm{se}(\hat\lambda)}

    where :math:`\theta` collects the tested coefficients and :math:`m`
    counts them. Both come from the same fit, so they cannot describe
    different models.

    A replication whose design is singular produces a non-finite
    statistic rather than an exception: with ``B`` in the thousands,
    stopping the whole run because one regenerated sample degenerated
    would be worse than dropping it and saying so.
    """
    design, target, tested, lam_pos = _build_designs(y, x, p, q, case)
    n_rep, n_est, k_par = design.shape
    if n_est <= k_par:
        raise ValueError(
            f"Each regenerated sample leaves {n_est} rows for {k_par} "
            "parameters: the model cannot be estimated."
        )

    # Stacked QR — one LAPACK call for the whole batch, and never the
    # normal equations.
    q_mat, r_mat = np.linalg.qr(design)
    qty = np.einsum("bij,bi->bj", q_mat, target)

    # A singular replication would make the triangular solve blow up;
    # it is detected on R's diagonal and neutralised beforehand so the
    # solve stays finite for the whole batch.
    diag = np.abs(np.diagonal(r_mat, axis1=1, axis2=2))
    scale = diag.max(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.asarray(
            np.all(diag > np.finfo(np.float64).eps * 100.0 * scale, axis=1),
            dtype=bool,
        )
    safe = np.where(ok[:, None, None], r_mat, np.eye(k_par))

    beta = np.linalg.solve(safe, qty[:, :, None])[:, :, 0]
    resid = target - np.einsum("bij,bj->bi", design, beta)
    ssr = np.einsum("bi,bi->b", resid, resid)
    sigma2 = ssr / (n_est - k_par)

    # (X'X)^{-1} = R^{-1} R^{-T}, obtained by triangular inversion.
    r_inv = np.linalg.solve(safe, np.broadcast_to(np.eye(k_par), safe.shape))
    xtx_inv = np.einsum("bij,bkj->bik", r_inv, r_inv)
    cov = sigma2[:, None, None] * xtx_inv

    idx = np.asarray(tested, dtype=np.intp)
    theta = beta[:, idx]
    v_sub = cov[np.ix_(np.arange(n_rep), idx, idx)]
    with np.errstate(invalid="ignore", divide="ignore"):
        solved = np.linalg.solve(
            np.where(ok[:, None, None], v_sub, np.eye(idx.size)),
            theta[:, :, None],
        )[:, :, 0]
        f_stat = np.einsum("bi,bi->b", theta, solved) / idx.size
        t_stat = beta[:, lam_pos] / np.sqrt(cov[:, lam_pos, lam_pos])

    ok = np.asarray(ok & np.isfinite(f_stat) & np.isfinite(t_stat), dtype=bool)
    return (
        np.asarray(f_stat, dtype=np.float64),
        np.asarray(t_stat, dtype=np.float64),
        np.asarray(ok, dtype=bool),
    )
