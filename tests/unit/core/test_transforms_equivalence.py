"""Verrou n°1 (piège connu) : équivalence des régressions
ARDL <-> ECM. Ce test doit être écrit AVANT l'implémentation de
``pyardl.core.transforms`` et verrouille les formules de passage de la
spec 03 §2.2 (source d'erreurs n°1 du projet).

Spec 03 §6.1.2 : sur données simulées, estimer par OLS (a) la forme ARDL
et (b) la forme ECM -> résidus identiques (1e-10), SSR identique, et
coefficients liés exactement par les formules de passage.

Les deux régressions ci-dessous sont construites indépendamment de
``pyardl.core.transforms`` (matrices de dessin bâties à la main dans ce
fichier de test), afin que la comparaison soit une vérification croisée
réelle et non une tautologie.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.core.transforms import ARDLParams, ardl_to_ecm


def _simulate_ardl(
    rng: np.random.Generator,
    *,
    t_obs: int,
    p: int,
    q: tuple[int, ...],
    has_trend: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """DGP ARDL(p, q_1..q_k) stationnaire, construit récursivement (test only)."""
    k = len(q)
    phi_true = rng.uniform(-0.3, 0.3, size=p)
    beta_true = [rng.uniform(-1.0, 1.0, size=qj + 1) for qj in q]
    const_true = rng.uniform(-1.0, 1.0)
    trend_true = rng.uniform(-0.01, 0.01) if has_trend else 0.0

    burn = 100
    n = t_obs + burn
    x = rng.normal(size=(n, k))
    eps = rng.normal(scale=0.5, size=n)
    y = np.zeros(n)
    start_t = max(p, (max(q) + 1) if k else 0)
    for t in range(start_t, n):
        val = const_true + trend_true * t
        for i in range(p):
            val += phi_true[i] * y[t - i - 1]
        for j in range(k):
            for i in range(q[j] + 1):
                val += beta_true[j][i] * x[t - i, j]
        val += eps[t]
        y[t] = val

    return (
        y[burn:],
        x[burn:],
        phi_true,
        np.array(beta_true, dtype=object) if k else np.array([]),
        const_true,
        trend_true,
    )


@pytest.mark.parametrize("seed", range(8))
def test_ardl_ecm_regression_equivalence(seed: int) -> None:
    """Spec 03 §6.1.2 : résidus ARDL et ECM identiques (verrou n°1).

    (Marque ``needs_review`` levée le 2026-07-07 : la convention
    q_j = 0 est confirmée par R ARDL::uecm sur les données danoises —
    niveau contemporain, résidus ardl/uecm identiques à 1.6e-14, cf.
    tests/replication/test_spec03.py et docs/QUESTIONS.md.)
    """
    rng = np.random.default_rng(seed)
    t_obs = 400
    p = int(rng.integers(1, 4))
    k = int(rng.integers(1, 3))
    q = tuple(int(v) for v in rng.integers(0, 4, size=k))
    has_trend = bool(rng.integers(0, 2))

    y, x, *_ = _simulate_ardl(rng, t_obs=t_obs, p=p, q=q, has_trend=has_trend)
    t_index = np.arange(t_obs, dtype=np.float64)

    max_lag = max(p, max(q) + 1)
    start = max_lag

    # --- (a) régression ARDL brute : y_t sur [1, t, y_{t-1..t-p}, x_{j,t-i}] ---
    cols_ardl = [np.ones(t_obs - start)]
    if has_trend:
        cols_ardl.append(t_index[start:])
    for i in range(1, p + 1):
        cols_ardl.append(y[start - i : t_obs - i])
    for j in range(k):
        for i in range(q[j] + 1):
            cols_ardl.append(x[start - i : t_obs - i, j])
    x_ardl = np.column_stack(cols_ardl)
    y_dep = y[start:]
    coefs_ardl, *_ = np.linalg.lstsq(x_ardl, y_dep, rcond=None)
    resid_ardl = y_dep - x_ardl @ coefs_ardl

    pos = 0
    const_hat = coefs_ardl[pos]
    pos += 1
    trend_hat = 0.0
    if has_trend:
        trend_hat = coefs_ardl[pos]
        pos += 1
    phi_hat = coefs_ardl[pos : pos + p]
    pos += p
    beta_hat = []
    for j in range(k):
        beta_hat.append(coefs_ardl[pos : pos + q[j] + 1])
        pos += q[j] + 1

    # --- (b) régression ECM : Δy_t sur [1, t, y_{t-1}, x_{j,·}, Δy, Δx] ---
    dy = np.diff(y)
    dx = np.diff(x, axis=0)

    cols_ecm = [np.ones(t_obs - start)]
    if has_trend:
        cols_ecm.append(t_index[start:])
    cols_ecm.append(y[start - 1 : t_obs - 1])
    for j in range(k):
        if q[j] == 0:
            # q_j = 0 : pas de dynamique propre, gamma_j multiplie x_{j,t}
            # (contemporain), pas x_{j,t-1} (cf. docs/QUESTIONS.md).
            cols_ecm.append(x[start:t_obs, j])
        else:
            cols_ecm.append(x[start - 1 : t_obs - 1, j])
    for i in range(1, p):
        cols_ecm.append(dy[start - i - 1 : t_obs - i - 1])
    n_omega = list(q)
    for j in range(k):
        for i in range(n_omega[j]):
            cols_ecm.append(dx[start - i - 1 : t_obs - i - 1, j])
    x_ecm = np.column_stack(cols_ecm)
    y_dep_ecm = dy[start - 1 : t_obs - 1]
    coefs_ecm, *_ = np.linalg.lstsq(x_ecm, y_dep_ecm, rcond=None)
    resid_ecm = y_dep_ecm - x_ecm @ coefs_ecm

    # Verrou n°1 : résidus et SSR identiques (spec 03 §6.1.2).
    np.testing.assert_allclose(resid_ardl, resid_ecm, atol=1e-10)
    np.testing.assert_allclose(np.sum(resid_ardl**2), np.sum(resid_ecm**2), atol=1e-10)

    # --- (c) coefficients ECM = transformation des coefficients ARDL estimés ---
    params = ARDLParams(
        p=p,
        q=q,
        phi=phi_hat,
        beta=tuple(beta_hat),
        const=float(const_hat),
        trend=float(trend_hat),
        has_const=True,
        has_trend=has_trend,
    )
    ecm = ardl_to_ecm(params)

    pos2 = 1 + (1 if has_trend else 0)
    lam_hat = coefs_ecm[pos2]
    pos2 += 1
    gamma_hat = coefs_ecm[pos2 : pos2 + k]
    pos2 += k
    psi_hat = coefs_ecm[pos2 : pos2 + (p - 1)]
    pos2 += p - 1
    omega_hat = []
    for j in range(k):
        omega_hat.append(coefs_ecm[pos2 : pos2 + n_omega[j]])
        pos2 += n_omega[j]

    assert ecm.lam == pytest.approx(lam_hat, abs=1e-8)
    np.testing.assert_allclose(ecm.gamma, gamma_hat, atol=1e-8)
    np.testing.assert_allclose(ecm.psi, psi_hat, atol=1e-8)
    for j in range(k):
        np.testing.assert_allclose(ecm.omega[j], omega_hat[j], atol=1e-8)
