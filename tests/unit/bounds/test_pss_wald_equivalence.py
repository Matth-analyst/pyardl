"""Verrou spec 10 §7.2 (écrit AVANT l'implémentation) : le F du bounds
test calculé par Wald sur la régression non contrainte — avec le
déterministe restreint DANS le vecteur testé pour les cas II et IV
(spec 10 §3.1, piège connu) — doit être identique à 1e-10 au F
calculé par SSR entre régression contrainte et non contrainte,
construites À LA MAIN dans ce fichier (vérification croisée réelle).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ardlpy.bounds import bounds_test


def _dgp(seed: int, n: int = 250, k: int = 2) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({f"x{j}": rng.normal(size=n).cumsum() for j in range(k)})
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (
            0.2
            + 0.6 * y[t - 1]
            + 0.1 * y[t - 2]
            + 0.4 * x.iloc[t, 0]
            - 0.1 * x.iloc[t - 1, 0]
            + 0.2 * x.iloc[t, 1]
            + rng.normal(scale=0.5)
        )
    return pd.Series(y, name="y"), x


def _manual_f(
    y: pd.Series,
    x: pd.DataFrame,
    p: int,
    q: int,
    case: int,
) -> tuple[float, float]:
    """(F par SSR, t_BDM) construits à la main : régression UECM non
    contrainte vs contrainte sous H0 du cas demandé."""
    yv = y.to_numpy()
    xv = x.to_numpy()
    n, k = xv.shape
    start = max(p, q)
    dy = np.diff(yv)
    dx = np.diff(xv, axis=0)

    # --- blocs communs (court terme) ---
    short_run: list[np.ndarray] = []
    for i in range(1, p):
        short_run.append(dy[start - i - 1 : n - i - 1])
    for j in range(k):
        for i in range(q):
            short_run.append(dx[start - i - 1 : n - i - 1, j])

    # --- déterministes et niveaux ---
    const_col = np.ones(n - start)
    trend_col = np.arange(start + 1, n + 1, dtype=np.float64)
    y_lag = yv[start - 1 : n - 1]
    x_lags = [xv[start - 1 : n - 1, j] for j in range(k)]

    if case == 1:
        unrestricted = [y_lag, *x_lags, *short_run]
        restricted = list(short_run)
        n_restr = k + 1
        t_pos = 0
    elif case == 2:
        unrestricted = [const_col, y_lag, *x_lags, *short_run]
        restricted = list(short_run)  # H0 : c0 = lambda = gamma = 0
        n_restr = k + 2
        t_pos = 1
    elif case == 3:
        unrestricted = [const_col, y_lag, *x_lags, *short_run]
        restricted = [const_col, *short_run]
        n_restr = k + 1
        t_pos = 1
    elif case == 4:
        unrestricted = [const_col, trend_col, y_lag, *x_lags, *short_run]
        restricted = [const_col, *short_run]  # H0 : c1 = lambda = gamma = 0
        n_restr = k + 2
        t_pos = 2
    elif case == 5:
        unrestricted = [const_col, trend_col, y_lag, *x_lags, *short_run]
        restricted = [const_col, trend_col, *short_run]
        n_restr = k + 1
        t_pos = 2
    else:
        raise ValueError(case)

    y_dep = dy[start - 1 :]

    def ssr_of(cols: list[np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
        design = np.column_stack(cols) if cols else np.empty((len(y_dep), 0))
        if design.shape[1] == 0:
            return float(y_dep @ y_dep), np.array([]), design
        coefs, *_ = np.linalg.lstsq(design, y_dep, rcond=None)
        resid = y_dep - design @ coefs
        return float(resid @ resid), coefs, design

    ssr_u, coefs_u, design_u = ssr_of(unrestricted)
    ssr_r, *_ = ssr_of(restricted)

    n_est, k_u = design_u.shape
    df = n_est - k_u
    f_stat = ((ssr_r - ssr_u) / n_restr) / (ssr_u / df)

    # t_BDM sur y_{t-1} (se nonrobust)
    xtx_inv = np.linalg.inv(design_u.T @ design_u)  # test only
    se = float(np.sqrt(ssr_u / df * xtx_inv[t_pos, t_pos]))
    t_stat = float(coefs_u[t_pos] / se)
    return float(f_stat), t_stat


@pytest.mark.parametrize("case", [2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_wald_equals_restricted_regression_cases_ii_iv(case: int, seed: int) -> None:
    """Spec 10 §7.2 : cas II et IV — le déterministe restreint fait
    partie du vecteur testé (k+2 restrictions)."""
    y, x = _dgp(seed)
    res = bounds_test(y, x, case=case, order=(2, 2))
    f_manual, t_manual = _manual_f(y, x, p=2, q=2, case=case)
    assert res.f_stat == pytest.approx(f_manual, abs=1e-10)
    assert res.t_stat == pytest.approx(t_manual, abs=1e-10)


@pytest.mark.parametrize("case", [1, 3, 5])
def test_wald_equals_restricted_regression_unrestricted_cases(case: int) -> None:
    """Cas I, III, V : k+1 restrictions, déterministes hors du test."""
    y, x = _dgp(seed=7)
    res = bounds_test(y, x, case=case, order=(2, 2))
    f_manual, t_manual = _manual_f(y, x, p=2, q=2, case=case)
    assert res.f_stat == pytest.approx(f_manual, abs=1e-10)
    assert res.t_stat == pytest.approx(t_manual, abs=1e-10)
