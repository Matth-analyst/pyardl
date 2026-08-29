# Distributed lags — Koyck and Almon

`pyardl.distributed_lags`

Before the ARDL, the question was how to fit an effect spread over many
periods without estimating one free coefficient per period. The two
answers that survived are still worth having, and both are here — not as
museum pieces, but because each makes a point the general model hides.

## Koyck: the transformation that breaks least squares

Assume the weights decline geometrically, `β_i = β₀ λ^i`. Taking
`y_t − λ y_{t-1}` telescopes the infinite sum into three parameters:

```
y_t = α(1−λ) + β₀ x_t + λ y_{t-1} + u_t,   u_t = ε_t − λ ε_{t-1}
```

**And the transformation is exactly what breaks OLS.** The error `u_t`
contains `ε_{t-1}`, which is also inside `y_{t-1}`, so
`Cov(y_{t-1}, u_t) = −λσ² ≠ 0`. Least squares here is not inefficient,
it is *inconsistent*: the bias does not shrink with the sample.

Fit the same simulated data — true `(α, β₀, λ) = (2.0, 0.8, 0.6)` — three
ways:

| method | α | β₀ | λ |
|---|---|---|---|
| `"ols"` | 1.9752 | 0.8498 | **0.4727** |
| `"iv"` (default) | 1.9986 | 0.8537 | **0.6010** |
| `"ml"` | 1.9931 | 0.8442 | **0.5860** |

OLS misses `λ` by a quarter of its value. Enlarging the sample does not
help: measured over 20 replications, the bias is 0.118 at `T = 2000` and
0.121 at `T = 8000`, with a Monte Carlo standard error of 0.002.

```pycon
>>> import numpy as np, pandas as pd
>>> from pyardl.distributed_lags import KoyckModel
>>> rng = np.random.default_rng(0)
>>> n = 400
>>> x = rng.normal(size=n)
>>> e = rng.normal(scale=0.5, size=n)
>>> y = np.zeros(n)
>>> for t in range(1, n):
...     y[t] = 2.0 * 0.4 + 0.8 * x[t] + 0.6 * y[t - 1] + e[t] - 0.6 * e[t - 1]
>>> res = KoyckModel(pd.Series(y, name="y"), pd.Series(x, name="x")).fit()
>>> res.multipliers().round(4)
             value      se
impact      0.8537  0.0297
longrun     2.1397  0.2073
mean_lag    1.5065  0.2235
median_lag  1.3615  0.1583

```

`"ols"` is kept, and warns on every fit. Removing it would hide the one
thing this model exists to teach.

### Read the autocorrelation tests against the right null

Under `"ols"` and `"iv"` the residuals **are** `u_t`, which is MA(1) by
construction. Rejecting white noise there confirms the model rather than
contradicting it — Ljung-Box gives `p = 0.0000` for IV on the data above
and `p = 0.9025` for ML. Only under `"ml"` are the residuals the
structural `ε_t`, and only there does a rejection mean the geometric
assumption itself is failing.

Durbin-Watson is not reported at all: it is biased towards "no
autocorrelation" when a lagged dependent variable is a regressor, which
is every Koyck model. Durbin's `h` replaces it, with an automatic
fallback to Durbin's alternative test when `n·var(λ̂) ≥ 1` makes `h`
undefined — and the index says which one ran.

### Where the delta method stops working

The long-run multiplier `β₀/(1−λ)` is a **ratio**, so its standard error
is a linearisation. Against a parametric bootstrap:

| n | delta se | bootstrap se | gap |
|---|---|---|---|
| 400 | 0.241 | 0.309 | **22% too small** |
| 1000 | 0.1335 | 0.1353 | 1.4% |
| 3000 | 0.0821 | 0.0835 | 1.6% |

At `n = 400` an interval built on it undercovers. This is a property of
linearising a ratio, not a defect, and the test suite pins both halves —
the agreement at 1000 *and* the shortfall at 400 — so neither can drift
away unnoticed.

## Almon: the restriction, and the test of the restriction

Keep the lag finite and put the weights on a polynomial of degree
`r < q`:

```
β_i = γ₀ + γ₁i + ... + γ_r i^r,    β = Hγ,    H[i,j] = i^j
```

`q+1` free coefficients become `r+1`, the regression is ordinary least
squares on the Almon variables `z_j = Σ_i i^j x_{t-i}`, and `β = Hγ` with
`V(β) = H V(γ) H'` — **exactly**, since the map is linear.

That linearity is worth noticing: unlike the Koyck long run, the Almon
long run `Σβ_i` is a linear form, so its standard error involves no
approximation and does not degrade in small samples.

```pycon
>>> from pyardl.distributed_lags import AlmonModel
>>> from pyardl.datasets import load_denmark
>>> d = load_denmark()
>>> res = AlmonModel(d["LRM"], d["LRY"], q=4, r=2).fit()
>>> print(res.summary())
Almon polynomial distributed lag - 51 observations
  q=4, r=2, endpoint='none', basis='power', cov_type='nonrobust'
<BLANKLINE>
                  beta          se         t
    L0          1.6751      0.3470     4.828
    L1          0.3802      0.1527     2.490
    L2         -0.2621      0.2780    -0.943
    L3         -0.2520      0.1612    -1.563
    L4          0.4107      0.3327     1.234
<BLANKLINE>
  long run: 1.9518 (se 0.1804)
  polynomial restriction: F(2, 45) = 0.361, p = 0.6990

```

**An Almon model always produces a smooth lag distribution. That is what
it was asked for.** The only informative question is whether the shape is
in the data or only in the assumption, which is why
`polynomial_restriction_test` runs against the *unrestricted* finite lag
and appears in every summary. Here it does not reject.

`to_fdl()` shows what the restriction bought:

```text
       beta      se
lag
0    1.9552  0.4824
1   -0.1672  0.6888
2    0.0100  0.6921
3   -0.1344  0.6930
4    0.2925  0.4726
```

Free weights with standard errors of 0.69 on estimates of 0.01 — the
collinearity problem Almon set out to solve, visible. The restricted
errors are less than half as large, and the F says the price was not
paid in bias.

### Two implementation details that are not cosmetic

**Conditioning.** `H[i,j] = i^j` at `q = 12, r = 4` holds entries up to
`12⁴` in near-collinear columns. The internal basis is `(i/q)^j`, or
Chebyshev on request. `γ` changes meaning with the basis; `β` does not,
and the tests pin the two to agree to 1e-8. The effect is measurable
against `dLagM`, which uses the raw basis: the weights agree to 1.8e-13
but the *standard errors* only to 1.2e-9 — the extra digits lost are the
conditioning.

**Endpoint constraints** are linear in `γ`, so `endpoint="head"` /
`"tail"` / `"both"` are imposed by reparameterising onto the null space
of the constraint matrix, not by penalty. The constraint then holds to
1e-10 rather than approximately. One subtlety worth stating because it
caught me: `"head"` constrains `β₋₁ = 0`, one step *outside* the window —
a weight distribution that vanishes at `i = 0` does not satisfy it.

### Choosing `(q, r)`

`select_order` grids both on a **common sample** — every candidate
estimated on `t = max_q+1 .. T`. Otherwise a larger `q` competes on
fewer observations, its criterion is not comparable, and the selection
silently rewards whichever model dropped the hardest rows.

## Cross-checked against R

`dLagM` 1.1.13 (GPL-2.1), on the Danish data pyardl already ships.

| quantity | agreement |
|---|---|
| Almon `β` vs `polyDlm` | **1.8e-13** |
| Almon standard errors | 1.2e-9 |
| Almon SSR | 7.4e-15 |
| Almon `β`, Chebyshev basis | 5.0e-14 |

**The Koyck comparison did not agree, and finding out why was the useful
part.** `koyckDlm` returns an `ivreg` object, and an `ivreg` carries its
instrument formula:

```
y.t ~ Y.1 + X.t | Y.1 + X.t_1
```

`Y.1` appears on **both** sides of the bar — it is treated as exogenous,
and `X.t` is the variable being instrumented. But in a Koyck model the
endogenous regressor is `y_{t-1}`: it is what the transformation
correlates with the error. pyardl instruments it with `x_{t-1}`, after
Liviatan (1963).

Adopting that instrument set inside pyardl reproduces `dLagM`'s
coefficients to better than 1e-8, which is what turns "two packages
disagree" into "here is the single decision they differ on". The cost of
that decision, on a DGP where `λ = 0.6` is known, 400 replications at
`T = 2000`:

| estimator | mean | bias | bias / MC error |
|---|---|---|---|
| Liviatan (pyardl) | 0.6001 | +0.0001 | 0.2 |
| the other instrument set | 0.5052 | −0.0948 | 190.4 |
| plain OLS | 0.5620 | −0.0380 | 148.5 |

Instrumenting the wrong regressor is worse than instrumenting nothing.
OBS-26.

**On the Danish data, pyardl warns.** `LRY` is highly persistent, so
`x_{t-1}` explains `y_{t-1}` poorly and the first-stage F is **3.70**,
well under 10. Danish money demand is bad ground for a Koyck model, and
the software should say so rather than leave the reader to notice.

## Where these sit in the library

Both are special cases of the ARDL, and both results objects carry a
bridge so the restriction can be seen as a restriction:

- `KoyckResults.to_ardl()` — the equivalent ARDL(1, 0), *exactly* (the
  OLS coefficients match to 1e-12, and `β₀/(1−λ)` is the same number as
  the ARDL's `θ = Σβ/(1−Σφ)`).
- `AlmonResults.to_fdl()` — the unrestricted finite lag on the same rows.

## References

- Koyck, L. M. (1954). *Distributed Lags and Investment Analysis*.
  North-Holland.
- Almon, S. (1965). The distributed lag between capital appropriations
  and expenditures. *Econometrica*, 33(1), 178-196.
- Liviatan, N. (1963). Consistent estimation of distributed lags.
  *International Economic Review*, 4(1), 44-52.
- Durbin, J. (1970). Testing for serial correlation in least-squares
  regression when some of the regressors are lagged dependent variables.
  *Econometrica*, 38(3), 410-421.
