# System ARDL (SUR-ECM)

`pyardl.system`

Every other estimator in the library is single-equation: one `y`,
possibly repeated across individuals in a panel (`pyardl.panel`), but
never several dependent variables *simultaneously linked* through their
errors. `SystemARDL` covers that case: two or more ARDL equations whose
residuals are correlated contemporaneously, estimated jointly by
Feasible Generalized Least Squares — Seemingly Unrelated Regressions
(Zellner 1962).

**Distinct from Johansen.** `pyardl.cointegration.johansen` estimates a
full VAR system where every variable is endogenous and the number of
cointegrating relations is itself estimated. `SystemARDL` instead takes
**already-specified single-equation ECMs** (each cointegrating relation
assumed known, and tested separately with the bounds test), linked only
through the contemporaneous correlation of their residuals — a more
restrictive model, but one that never needs a system-wide cointegration
rank chosen.

## `SystemARDL(equations, det="const", iterate=True, max_iter=50, tol=1e-8)`

```python
from pyardl.system import SystemARDL

res = SystemARDL({
    "money": (y1, x1, (1, 1)),
    "credit": (y2, x2, (1, 1)),
}, det="const", iterate=True).fit()
```

| Argument | Meaning |
|---|---|
| `equations` | `{name: (y, x, order)}`, one entry per equation |
| `det` | applied to every equation |
| `iterate` | repeat the FGLS round to convergence (equivalent to ML under normality) rather than a single GLS pass |

Every `y` must share the same length and time index — a system with
partially disjoint dates between equations is out of scope.

### Estimation algorithm

1. Fit every equation separately by plain OLS, reusing `pyardl.core.ardl.ARDL`
   unchanged, with a common `hold_back` so all equations share the same
   estimation sample.
2. Estimate the residual covariance `Sigma_hat` from the OLS residuals.
3. Re-estimate the whole system by GLS, whitening the stacked design by
   `Sigma_hat^{-1/2}` (a Cholesky-based `M x M` solve — small relative
   to the sample size) and solving the whitened system with a single
   `numpy.linalg.lstsq` call.
4. Repeat steps 2-3 to convergence when `iterate=True`.

If every equation shares the exact same regressor matrix, SUR reduces
to equation-by-equation OLS exactly (the classical Zellner 1962 result)
— `pyardl` reproduces this to numerical precision
(`tests/unit/system/test_system_ardl.py::TestIdenticalRegressorsIdentity`).

## `SystemARDLResults`

| Member | Returns |
|---|---|
| `equations` | `{name: ARDLResults}` — the per-equation OLS fit, for comparison |
| `sigma` | `Sigma_hat`, the residual covariance, as a `pandas.DataFrame` |
| `fgls_params` / `fgls_se` | `{name: pandas.Series}` — FGLS coefficients and standard errors per equation |
| `n_iter` | FGLS iterations run |
| `names` | `{name: [term, ...]}` — parameter names in the order they appear in the stacked system |

### `.efficiency_gain(name)`

Ratio of the single-equation OLS standard errors to the FGLS ones, for
one equation. Above 1 means FGLS is more precise — the gain Zellner's
method promises, larger with stronger residual correlation and more
different regressors between equations. Not guaranteed to exceed 1 in
every finite sample (the SUR efficiency gain can be small or negative
when `Sigma_hat` is poorly estimated relative to the number of
equations); this reports what was measured on the fitted data, not what
theory promises asymptotically.

### `.test_cross_equation_restriction(r_matrix, r)`

Wald test of `H0: R theta = r` on the **raw stacked UECM coefficients**
(adjustment speeds, level coefficients, short-run terms) — a
restriction like `lambda_1 = lambda_2` is linear in these and directly
testable, generalising the single-equation long-run restriction test
(`pyardl.restrictions`) to a system.

!!! warning "Scope: raw parameters, not the long-run ratio"
    A restriction on the **long-run** coefficient
    `theta = -gamma/lambda` across equations (rather than on `gamma` or
    `lambda` directly) would need the delta method extended to the
    stacked system's covariance — **not implemented**. See
    `docs/DEVIATIONS.md`.

Returns `WaldResult(stat, df, pvalue)`.

### `.summary()`

Publication-style report: residual correlation matrix and per-equation
FGLS coefficients with standard errors.

## References

Zellner, A. (1962). An efficient method of estimating seemingly
unrelated regressions and tests for aggregation bias. *Journal of the
American Statistical Association*, 57(298), 348-368.
