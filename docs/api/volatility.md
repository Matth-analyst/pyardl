# ARDL-GARCH

`pyardl.volatility`

Every inference tool elsewhere in this library (spec 03's standard
errors, spec 10's bounds test, spec 04's restriction tests) assumes a
homoskedastic residual variance, or at best a robust HC/HAC covariance
that corrects the *average* variance over a window without modelling
its dynamics. `ARDLGarch` jointly estimates the conditional mean (a
standard UECM) and the conditional variance of its residuals (GARCH):

```
Delta y_t = det + lambda*y_{t-1} + gamma'x_{t-1} + Sigma(...) + eps_t,  eps_t = sigma_t*z_t
sigma_t^2 = omega + Sigma_i alpha_i*eps_{t-i}^2 + Sigma_j beta_j*sigma_{t-j}^2
```

Mean and variance are estimated **jointly**, not in two separate
steps — two-step estimation is biased under GARCH-in-mean, and less
efficient even without it.

## Implementation strategy

`arch` is already an optional dependency of this project (lazily
imported, used elsewhere for bootstrap validation) — `ARDLGarch` keeps
that status rather than promoting it to a hard runtime dependency, and
reuses `arch.univariate.LS` (a plain linear mean model with an
attachable volatility process) as the joint estimation engine, the same
strategy `pyardl.markov_switching` applies to `statsmodels`.

## `ARDLGarch(y, x, order=(1, 1), det="const", garch_order=(1, 1), garch_type="garch", in_mean=False)`

```python
from pyardl.volatility import ARDLGarch

res = ARDLGarch(y, x, order=(1, 1), garch_order=(1, 1), garch_type="gjr").fit()
```

!!! warning "`garch_order` convention"
    `garch_order=(p, q)` follows the **spec's** convention (`p` GARCH
    lagged-variance terms, `q` ARCH lagged-squared-residual terms) —
    the reverse of `arch`'s own `GARCH(p, o, q)` argument order,
    translated internally.

| Argument | Meaning |
|---|---|
| `garch_order` | `(0, 0)` fits a constant (homoskedastic) variance instead — the degenerate case that locks this module against plain `ARDL` |
| `garch_type` | `"garch"`, `"egarch"`, or `"gjr"` (asymmetric/leverage term — not to be confused with NARDL's asymmetry, which is in the mean, not the variance) |
| `in_mean` | **not implemented** — raises `NotImplementedError` if `True` |

## `ARDLGarchResults`

| Member | Returns |
|---|---|
| `mean_params` / `mean_se` | UECM mean-equation coefficients and standard errors |
| `garch_params` | `omega`, `alpha[i]`, `gamma[i]` (asymmetric term), `beta[j]` |
| `conditional_variance` | `sigma_t^2` over the sample |
| `longrun` | `theta = -gamma/lambda` and its standard error |
| `persistence` | `sum(alpha) + sum(beta) + 0.5*sum(gamma)` |

!!! note "`longrun` standard errors are not spec 03's"
    The delta method here is applied to the **joint** MLE covariance
    matrix (mean and variance parameters together, `arch`'s
    `param_cov`) — never the OLS covariance `ARDLResults.longrun` uses,
    since the residual variance is no longer constant.

A `PyardlMethodologyWarning` is raised when `persistence > 0.98`
(near-integrated variance, IGARCH — a frequent edge case in financial
data, the same discipline as `is_stable` for the mean).

`.summary()` and `.plot_volatility()` (requires matplotlib) are also
available.

## Out of scope

GARCH-in-mean (`in_mean=True`): `arch`'s `LS` mean model has no
feedback term from the variance equation into the mean, and building
one would need a custom joint-likelihood optimiser outside `arch`'s
public API. Multivariate GARCH and NARDL/QARDL combinations.

## References

Engle, R. F. (1982). Autoregressive conditional heteroscedasticity
with estimates of the variance of United Kingdom inflation.
*Econometrica*, 50(4), 987-1007.

Bollerslev, T. (1986). Generalized autoregressive conditional
heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
