"""Spec 03 §6.2 — validation externe : concordance avec
``statsmodels.tsa.ardl.UECM.from_ardl`` sur la reparamétrisation
ARDL -> ECM (résout aussi le point ambigu de §2.2 sur omega_{j,0},
cf. docs/QUESTIONS.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.ardl import ARDL, UECM

from ardlpy.core.transforms import ARDLParams, ardl_to_ecm


@pytest.mark.parametrize("seed", range(5))
def test_ardl_to_ecm_matches_statsmodels_uecm(seed: int) -> None:
    """theta, lambda, gamma, psi, omega de ardl_to_ecm = ceux de statsmodels
    UECM.from_ardl, sur un ARDL(p, q) estimé par statsmodels (q_j >= 1)."""
    rng = np.random.default_rng(seed)
    n = 400
    p = int(rng.integers(1, 4))
    q_x = int(rng.integers(1, 4))  # q_j >= 1 : statsmodels l'exige (cf. QUESTIONS.md)

    x_v = rng.normal(size=n)
    y = np.zeros(n)
    phi_true = rng.uniform(-0.3, 0.3, size=p)
    beta_true = rng.uniform(-1.0, 1.0, size=q_x + 1)
    start = max(p, q_x + 1)
    for t in range(start, n):
        val = 0.2
        for i in range(p):
            val += phi_true[i] * y[t - i - 1]
        for i in range(q_x + 1):
            val += beta_true[i] * x_v[t - i]
        val += rng.normal(scale=0.3)
        y[t] = val

    y_s = pd.Series(y, name="y")
    x_df = pd.DataFrame({"x": x_v})

    ardl_res = ARDL(y_s, lags=p, exog=x_df, order={"x": q_x}, trend="c").fit()
    uecm_res = UECM(y_s, lags=p, exog=x_df, order={"x": q_x}, trend="c").fit()

    phi_hat = np.array([ardl_res.params[f"y.L{i}"] for i in range(1, p + 1)])
    beta_hat = np.array([ardl_res.params[f"x.L{i}"] for i in range(q_x + 1)])
    const_hat = float(ardl_res.params["const"])

    params = ARDLParams(
        p=p,
        q=(q_x,),
        phi=phi_hat,
        beta=(beta_hat,),
        const=const_hat,
        has_const=True,
        has_trend=False,
    )
    ecm = ardl_to_ecm(params)

    lam_sm = uecm_res.params["y.L1"]
    gamma_sm = uecm_res.params["x.L1"]
    psi_sm = np.array([uecm_res.params[f"D.y.L{i}"] for i in range(1, p)])
    omega_sm = np.array([uecm_res.params[f"D.x.L{i}"] for i in range(q_x)])

    assert ecm.lam == pytest.approx(lam_sm, abs=1e-6)
    assert ecm.gamma[0] == pytest.approx(gamma_sm, abs=1e-6)
    np.testing.assert_allclose(ecm.psi, psi_sm, atol=1e-6)
    np.testing.assert_allclose(ecm.omega[0], omega_sm, atol=1e-6)
