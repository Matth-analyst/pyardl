# ARDL estimation

`pyardl.core.ardl`

## `ARDL(y, x, order, det="const", fixed_regressors=None, hold_back=None)`

Fits

```
y_t = det_t + Σ φ_i y_{t-i} + Σ_j Σ_i β_{j,i} x_{j,t-i} + ε_t
```

by ordinary least squares.

| Argument | Meaning |
|---|---|
| `y` | dependent variable, shape `(T,)` |
| `x` | regressors, shape `(T, k)`; a DataFrame keeps column names in the output. `None` fits a pure AR(p) |
| `order` | `(p, q)` where `q` is an int applied to all regressors, or a dict `{name: q_j}` |
| `det` | `"none"`, `"const"` (default) or `"trend"` (which includes the intercept) |
| `fixed_regressors` | variables entered without lags, e.g. dummies |
| `hold_back` | initial observations to exclude, used to force a common sample |

`q_j = 0` is allowed: the regressor then enters contemporaneously with
no short-run dynamics of its own. (`statsmodels` rejects this case;
`pyardl` and Stata's `ardl` both support it.)

### `.fit(cov_type="nonrobust", cov_kwds=None)`

Returns [`ARDLResults`](#ardlresults). `cov_type` accepts `"nonrobust"`,
`"HC0"` to `"HC3"`, and `"HAC"` (Newey-West, with
`cov_kwds={"nlags": m}`; a data-driven default is used otherwise).

Every fit runs a Ljung-Box test on the residuals and warns if it
rejects. Long-run inference in an ARDL is only valid with enough lags to
whiten the errors, so this is a condition of validity rather than an
optional diagnostic.

## `ARDL.select_order(y, x, max_p, max_q, ic="aic", search="grid", det="const", min_p=1)`

Searches lag orders by information criterion.

- `ic` — `"aic"`, `"bic"` or `"hq"`; all three appear in the output table.
- `search` — `"grid"` explores the full cartesian product;
  `"per_variable"` optimises `p` and then each `q_j` in turn, which
  stays tractable when the grid would explode.

All candidates are estimated on the same sample, `t = max(max_p, max_q)+1 .. T`.
This is deliberate: information criteria computed on different numbers
of observations are not comparable, and letting each candidate use its
own maximal sample silently biases the choice. The winner is then
re-estimated on the largest sample its own order allows.

Returns `ARDLOrderSelection` with:

| Attribute | Contents |
|---|---|
| `table` | all candidates ranked by `ic`, with AIC, BIC, HQ, log-likelihood, `nobs` |
| `best_order` | `(p, {name: q_j})` |
| `best_model` | the re-estimated `ARDLResults` |
| `top(n)` | the `n` best candidates |

## `ARDL.gets(y, x, max_p, max_q, alpha=0.05, det="const")`

General-to-specific reduction. Starts from `(max_p, max_q)` and
repeatedly drops the least significant terminal lag, provided its
p-value exceeds `alpha`, the residual diagnostics stay clean, and an F
test of the accumulated restrictions against the general model does not
reject.

Only terminal lags are candidates, which keeps the lag structure
contiguous and the result a genuine ARDL(p, q).

Returns `GETSResults` with `final_model`, `final_order`,
`general_model`, and `reduction_path` — one row per attempted removal,
recording the p-value, the diagnostics, the cumulated F test and whether
the step was accepted. The path makes the reduction auditable instead of
a black box.

## `pyardl.regularized.select_order_regularized(y, x, max_p=4, max_q=4, method='elastic_net', l1_ratio=1.0, alpha='cv', det='const')`

An alternative to `select_order` for a large number of candidate
regressors, where the exhaustive/sequential grid search
(`(max_q+1)^k` candidates for `search="grid"`) becomes combinatorially
expensive — not a replacement: `select_order` stays the right tool for
a small `k` with a clean information-criterion reading; regularization
is for large `k`, or for selecting *which* level regressors belong in
the long run.

```python
from pyardl.regularized import select_order_regularized

res = select_order_regularized(y, x, max_p=4, max_q=4,
                                 method="elastic_net", alpha="cv")
res.selected_order      # (p, {name: q_j}), retro-deduced from surviving coefficients
res.best_model            # ARDLResults, re-estimated by plain OLS — never with the penalty
res.coefficients_path      # theta(alpha) over the grid, for plot_regularization_path()
```

Fits **one** model at `(max_p, max_q)` with a penalty that shrinks
irrelevant lag coefficients to zero, instead of choosing the order by
discrete search. `method="lasso"` is the `l1_ratio=1.0` special case of
Elastic Net, offered directly since pure LASSO's instability under
correlated level regressors is exactly what Elastic Net (`l1_ratio<1`)
corrects — implement both, and see the difference on correlated data
rather than assume it (`tests/unit/regularized/test_model.py::TestElasticNetVsLasso`).

**Penalty weights are block-differentiated, never uniform.** The
deterministic terms and the error-correction coefficient `lambda*y.L1`
are *never* penalised — penalising the adjustment speed itself would
bias the long-run reading, exactly the quantity this whole library
exists to get right. Only the short-run lag terms and the long-run
levels themselves are candidates for shrinkage.

**`best_model` is always re-estimated by plain OLS, never with the
penalty.** The regularisation path chooses the order; it never produces
the reported coefficients — reporting penalised coefficients as final
would bias the long-run standard errors downward in an undocumented
way, a known trap in the post-selection literature. Classical inference
on `best_model` still ignores the selection's own uncertainty
("post-selection inference", a distinct literature not addressed here).

`alpha="cv"` selects the penalty by **rolling-origin** (expanding
window) cross-validation, never a random split, which would break the
time structure. Estimation is a plain NumPy cyclic coordinate descent
(Friedman, Hastie & Tibshirani 2010) — no new runtime dependency;
`scikit-learn`/`glmnet` are used only in the external validation
script, never at runtime.

**Out of scope**: regularization applied to a panel (specs 22-24 — each
individual would potentially need its own `alpha`, not addressed) and
to NARDL/QARDL (joint selection of order and asymmetry/quantile, a
distinct combinatorial problem). See `docs/DEVIATIONS.md`.

## `ARDLResults`

### Regression output

`params`, `bse`, `tvalues`, `pvalues`, `resid`, `fittedvalues`,
`cov_params_matrix`, `nobs`, `ssr`, `sigma2`, `llf`, `aic`, `bic`,
`hqic`, `rsquared`, `rsquared_adj`.

!!! note "`nobs` convention"
    `nobs` is the size of the actual estimation sample, `T - hold_back`.
    `statsmodels` reports `T - p` and computes the log-likelihood and
    information criteria on that, even when `max(q_j) > p`. The two
    agree as soon as `p >= max(q_j)`; using the real sample is what
    makes the criteria comparable in `select_order`.

### Error-correction views

| Member | Returns |
|---|---|
| `to_ecm()` | `ECMParams` — exact reparameterisation, identical residuals |
| `longrun` | DataFrame with `theta` and `se` per regressor |
| `adjustment` | Series with `lambda`, `se`, `half_life` |
| `ardl_params` | `ARDLParams` container, covariance included |

Not available when `p = 0` (no lagged `y`, so no error-correction form)
or when the model has fixed regressors, which map to neither the `φ` nor
the `β` coefficients. Both cases raise an explicit error.

### Stability

`ar_roots` gives the roots of `1 - φ_1 L - ... - φ_p L^p`. `is_stable`
is `True` when all lie outside the unit circle; otherwise it warns, as
the long-run quantities then have no equilibrium interpretation.

### `diagnostics()`

Ljung-Box (autocorrelation), Jarque-Bera (normality) and Breusch-Pagan
(heteroskedasticity), with p-values.

### `summary()`

Publication-style report as a string.
