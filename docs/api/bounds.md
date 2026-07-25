# Bounds test

`pyardl.bounds`

## `bounds_test(y, x, case=3, order=None, ic="aic", max_p=4, max_q=4, alpha=0.05, cv_source="kripfganz", finite_t=False, fixed_regressors=None)`

Tests whether a long-run level relationship exists between `y` and `x`,
without requiring the integration order of the regressors to be known
beforehand.

The test is run on the unrestricted error-correction form

```
Δy_t = det_t + λ y_{t-1} + Σ_j γ_j x_{j,t-1}
       + Σ ψ_i Δy_{t-i} + Σ_j Σ_i ω_{j,i} Δx_{j,t-i} + ε_t
```

with two statistics:

- **`F_overall`** tests `λ = γ_1 = ... = γ_k = 0`, i.e. no level
  relationship. Under cases 2 and 4 the restricted deterministic term is
  part of the tested vector, giving `k+2` restrictions instead of `k+1`.
- **`t_BDM`** tests `λ = 0` alone. Left-tailed: rejection requires a
  negative estimate, i.e. an actual pull back towards equilibrium.

### The five deterministic cases

| `case` | Intercept | Trend | Typical use |
|---|---|---|---|
| 1 | none | none | demeaned data only |
| 2 | restricted | none | no trend anywhere |
| 3 | unrestricted | none | **the usual choice** |
| 4 | unrestricted | restricted | trending data, no trend in the relation |
| 5 | unrestricted | unrestricted | trending data and relation |

### Arguments

| Argument | Meaning |
|---|---|
| `order` | `(p, q)` lag orders; if omitted they are selected with `ic`, `max_p`, `max_q` |
| `alpha` | level driving the reported decisions; the `bounds` table always shows 10%, 5% and 1% |
| `cv_source` | `"kripfganz"` (default), `"pss"` or `"narayan"` — see [critical values](critical-values.md) |
| `fixed_regressors` | unlagged variables such as dummies; never part of the tested vector, and ignored by automatic order selection |
| `finite_t` | experimental and unvalidated; requires `cv_source="kripfganz"` |

### Assumptions

The test is valid if the regressors are weakly exogenous, are not
cointegrated among themselves, no series is I(2), and the residuals are
not autocorrelated. Only the last is checked automatically.

## `BoundsTestResults`

### Decisions

Each statistic is compared with a *pair* of critical values, so the
verdict has three states, never a boolean:

| Attribute | Values |
|---|---|
| `decision_f` | `"cointegration"`, `"no_cointegration"`, `"inconclusive"` |
| `decision_t` | same, or `None` when no t bounds exist for the case and source |
| `decision_joint` | the above plus `"degenerate_suspicion"`, or `None` |

`decision_joint` requires **both** tests to agree:

| F | t | Joint |
|---|---|---|
| rejects | rejects | `cointegration` |
| rejects | does not | `degenerate_suspicion` |
| does not | does not | `no_cointegration` |
| any other disagreement | | `inconclusive` |

`degenerate_suspicion` means the level terms are jointly significant but
`y` shows no error-correction force. The apparent relationship is likely
carried by the regressors alone; this is not cointegration, and a
warning is issued.

### Other attributes

| Attribute | Contents |
|---|---|
| `f_stat`, `t_stat` | the statistics |
| `bounds` | DataFrame of lower/upper bounds for F and t at 10%, 5%, 1% |
| `p_values` | Series with `p_I0`, `p_I1` (and `t_p_I0`, `t_p_I1` under `finite_t`), or `None` |
| `uecm` | coefficients, standard errors and t-ratios of the fitted model |
| `case`, `k`, `order`, `alpha`, `cv_source` | the settings used |

### `adjustment(alpha=0.05)`

Returns `lambda`, `se`, `ci_lower`, `ci_upper`.

The confidence interval is **only** produced when `decision_joint` is
`"cointegration"`. Otherwise the bounds are `NaN` and a warning is
issued: under the null, the distribution of this estimator is
non-standard, so the usual normal interval would be misleading. The
point estimate and its standard error remain available either way.

### `diagnostics()`

Ljung-Box, Jarque-Bera and Breusch-Pagan on the error-correction
residuals.

### `summary()`

Readable report: both statistics, their p-values at each bound, the
decisions, and the bounds at all three levels. When the F decision is
inconclusive, the p-value interval is shown so the result can be read on
a continuous scale:

```text
F_overall = 5.4724   decision (5%): inconclusive, p in [0.0594, 0.0313]
```

## A note on `q_j = 0`

A regressor with no lags of its own enters the tested vector through its
contemporaneous level `x_{j,t}`. This does not affect the asymptotics:
`x_{j,t} = x_{j,t-1} + Δx_{j,t}` and the difference is stationary, so
only the dating shifts by one period.
