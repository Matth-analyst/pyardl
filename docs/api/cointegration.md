# Engle-Granger cointegration test

`pyardl.cointegration`

The original cointegration test, and still the one most applied papers
report. It is provided here for comparison — the bounds test is the
recommended route, for reasons this page makes concrete.

## The idea

If two I(1) series are cointegrated, some linear combination of them is
stationary. So estimate that combination, then test its residuals for a
unit root:

```
step 1:  y_t = det_t + x_t'β + u_t
step 2:  Δû_t = ρ û_{t-1} + Σ ξ_i Δû_{t-i} + e_t
```

The null is **no cointegration**, so a large negative t-ratio on `ρ` is
evidence *for* a long-run relationship.

## `engle_granger(y, x, trend='c', max_lags=None, ic='aic', fit_ecm=False)`

```python
from pyardl.cointegration import engle_granger

res = engle_granger(y, x)
print(res.summary())
```

```text
Engle-Granger test (1987) - trend 'c', 2 variables, lags=1, nobs=200
  statistic = -11.3705   p-value = 0.0000
  decision (5%): cointegration
  critical values (left tail)   1%: -3.9520  5%: -3.3669  10%: -3.0657
  H0: no cointegration

  Long-run coefficients (point estimates only, no inference):
    x0           1.5016
    const       -0.0961
```

### Do not run inference on step one

The first-step coefficients are **super-consistent** — they converge
faster than an ordinary regression coefficient — but their distribution
is non-standard and they carry a second-order bias that does not vanish
at the usual rate. The standard errors an ordinary regression would
report are simply wrong here, which is why this function does not report
any.

For long-run inference use the ARDL route, where the delta method
applies:

```python
ARDL(y, x, order=(2, 2)).fit().longrun
```

### `fit_ecm=True`

Estimates the second-step error-correction model,
`Δy_t = α û_{t-1} + Δx_t'γ + ε_t`. Here the usual standard errors *are*
legitimate: because the first-step estimate converges faster, the
estimation error it injects into this regression is asymptotically
negligible.

## Why this is not the recommended tool

Three limitations are structural, not implementation details. Each one
is a reason the bounds test exists.

**The normalisation is arbitrary.** Regressing `y` on `x` and regressing
`x` on `y` are different tests, and they can disagree — our test suite
demonstrates it rather than asserting it. Nothing in the method says
which one is right.

**Only one relationship can be found.** With three or more variables
several cointegrating vectors may exist. This procedure sees at most one
and gives no warning that others were missed.

**Every series must be I(1).** A mixture of I(0) and I(1) regressors
invalidates the test. Establishing that beforehand is itself a sequence
of tests, each with its own error rate — see
[unit-root pre-tests](unitroot.md). The bounds test drops this
requirement entirely.

## Critical values

The statistic is computed on *estimated* residuals. Estimating the
cointegrating vector uses up information, so the null distribution
shifts left — and further left as the number of regressors grows. Using
ordinary Dickey-Fuller critical values here is a real error: it
over-rejects, increasingly with `k`.

`pyardl.critical_values.mackinnon` serves the response surfaces of
MacKinnon (1994, 2010), which account for this.

| Case | Available |
|---|---|
| `trend='c'`, `'ct'`, `'ctt'` | yes, up to 12 variables |
| `trend='n'` | **no** — MacKinnon published none |

Under `trend='n'` the values are returned as `NaN` with a warning, and
`decision()` raises rather than deciding. No value is borrowed from a
neighbouring deterministic case.

The surfaces are cross-checked against an independent in-house
simulation of the null — 54 cells over two deterministic cases, three
values of `k`, three sample sizes and three levels — all within three
standard errors of the simulated quantile. Details, along with two
rounding conventions where `pyardl` and `statsmodels` differ, are in
[`PROVENANCE.md`](https://github.com/Matth-analyst/pyardl/blob/main/src/pyardl/critical_values/PROVENANCE.md).

## Comparing with the bounds test

Running both on the same data is a reasonable habit. When they agree,
the conclusion is robust to the I(1) assumption. When they disagree, the
bounds test is the one that did not have to assume it — and the
disagreement itself is worth reporting.
