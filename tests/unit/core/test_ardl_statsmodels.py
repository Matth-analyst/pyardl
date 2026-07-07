"""Verrou spec 05 §6.5 (validation externe critique) : concordance totale
avec ``statsmodels.tsa.ardl.ARDL`` — mêmes données, mêmes ordres ->
coefficients à 1e-10. Écrit AVANT l'implémentation de ardlpy.core.ardl.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.ardl import ARDL as SM_ARDL

from ardlpy.core.ardl import ARDL

_TREND_MAP = {"none": "n", "const": "c", "trend": "ct"}


def _dgp(seed: int, n: int = 200, k: int = 2) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({f"x{j}": rng.normal(size=n).cumsum() for j in range(k)})
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = (
            0.3
            + 0.6 * y[t - 1]
            + 0.4 * x.iloc[t, 0]
            - 0.2 * x.iloc[t - 1, 0]
            + 0.1 * x.iloc[t, 1]
            + rng.normal(scale=0.5)
        )
    return pd.Series(y, name="y"), x


@pytest.mark.parametrize("det", ["none", "const", "trend"])
@pytest.mark.parametrize(
    ("p", "q"),
    [(1, {"x0": 1, "x1": 1}), (2, {"x0": 3, "x1": 1}), (3, {"x0": 0, "x1": 2})],
)
def test_params_match_statsmodels_1e10(det: str, p: int, q: dict[str, int]) -> None:
    """Spec 05 §6.5 : coefficients, bse, résidus identiques à 1e-10."""
    y, x = _dgp(seed=hash((p, det)) % 2**31)

    res = ARDL(y, x, order=(p, q), det=det).fit()
    sm_res = SM_ARDL(y, lags=p, exog=x, order=q, trend=_TREND_MAP[det]).fit()

    assert res.params.index.tolist() == sm_res.params.index.tolist()
    np.testing.assert_allclose(
        res.params.values, sm_res.params.values, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(res.bse.values, sm_res.bse.values, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(res.resid), np.asarray(sm_res.resid), rtol=0, atol=1e-10
    )


def test_loglik_and_ic_match_statsmodels() -> None:
    """llf/aic/bic/hqic alignés sur les conventions statsmodels (nécessaire
    pour que select_order soit comparable à ardl_select_order)."""
    y, x = _dgp(seed=7)
    res = ARDL(y, x, order=(2, {"x0": 1, "x1": 2}), det="const").fit()
    sm_res = SM_ARDL(y, lags=2, exog=x, order={"x0": 1, "x1": 2}, trend="c").fit()

    assert res.llf == pytest.approx(sm_res.llf, abs=1e-8)
    assert res.aic == pytest.approx(sm_res.aic, abs=1e-8)
    assert res.bic == pytest.approx(sm_res.bic, abs=1e-8)
    assert res.hqic == pytest.approx(sm_res.hqic, abs=1e-8)


def test_order_as_int_broadcasts() -> None:
    """order=(p, q_scalaire) : même q pour tous les régresseurs."""
    y, x = _dgp(seed=11)
    res = ARDL(y, x, order=(1, 2), det="const").fit()
    sm_res = SM_ARDL(y, lags=1, exog=x, order=2, trend="c").fit()
    np.testing.assert_allclose(
        res.params.values, sm_res.params.values, rtol=0, atol=1e-10
    )
