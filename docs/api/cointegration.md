# Engle-Granger, Gregory-Hansen and Bai-Perron cointegration/break tests

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

## Gregory-Hansen: cointegration with an unknown regime shift

Engle-Granger assumes the cointegrating vector is **stable** over the
whole sample. If the relationship actually shifts regime partway through
(a crisis, a change of monetary policy), the static step-one regression
averages both regimes, the residual looks non-stationary, and the test
wrongly concludes there is no cointegration. Gregory & Hansen (1996)
answer by leaving the break point unknown and estimated from the data: a
grid search over candidate break fractions, an Engle-Granger step-two ADF
test at each one, and the test statistic is the smallest (most negative)
ADF found over the grid.

No new numerical engine: step two reuses the Engle-Granger ADF regression
unchanged; only the step-one design gains regime-shift regressors, and
the critical values differ, because a supremum over a grid of unknown
break dates has a non-standard asymptotic distribution (the "problem of
Davies", also met at the Fourier frequency pre-test).

### `gregory_hansen(y, x, model='C', trim=0.15, cv_source='bootstrap', n_boot=999, seed=None)`

```python
from pyardl.cointegration import gregory_hansen

res = gregory_hansen(y, x, model="C/S", n_boot=999, seed=0)
print(res.summary())
res.break_fraction   # tau_hat
res.grid              # (tau, adf_stat) for every candidate — the margin
                        # of the retained break, not just the minimum
res.decision(0.05)
```

Four break specifications, all implemented (no silent fallback to one):

| Model | Equation |
|---|---|
| `'C'` | level shift: `y_t = μ1 + μ2·φ_t(τ) + x_t'β + u_t` |
| `'C/T'` | level shift + trend |
| `'C/S'` | level **and slope** shift: `x_t'β1 + φ_t(τ)·x_t'β2` also enters |
| `'C/S/T'` | `C/S` plus a trend |

where `φ_t(τ) = 1{t > ⌊nτ⌋}` is the regime indicator. Choosing the wrong
specification is not cosmetic: on a DGP where only the slope changes,
only `'C/S'` and `'C/S/T'` detect it — `'C'` and `'C/T'` have no
regressor to absorb a slope break and under-perform, which is exactly
what the test suite checks rather than asserts.

### Critical values: bootstrap only, for now

The published Gregory-Hansen asymptotic table (their original Table 1,
by number of regressors and by specification) is **not currently
encoded**: pyardl does not hold a verified copy of that table with a
documented provenance, and CLAUDE.md rule 9 forbids writing a
critical-value table from memory. `cv_source='table'` therefore raises
`NotImplementedError` rather than guessing — see `docs/QUESTIONS.md`.

`cv_source='bootstrap'` (the default, and the only implemented route)
regenerates the null — series that are jointly I(1) with **no**
cointegrating relationship, by fitting a VAR-in-differences to the
stacked system `[y, x]` and simulating from it — and reruns the same
grid search on every replicate to build the null distribution of `ADF*`.
This is more expensive than a table lookup (the grid search itself runs
once per bootstrap replicate) but does not depend on a table limited to
a few values of `k`.

### Limits

A single break only — for multiple structural breaks, see
[`bai_perron`](#bai-perron-multiple-structural-breaks) below. The test
remains residual-based, so it inherits Engle-Granger's structural limits
(arbitrary normalisation, at most one relationship detected). And, as
with any pre-tested break point, `ADF*(τ̂)` must never be compared to an
ordinary single-break-free ADF table — the search itself changes the law
of the statistic.

## Bai-Perron: multiple structural breaks

Gregory-Hansen finds at most one break. A series spanning decades often
crosses several regimes (oil shocks, financial crises, exchange-rate
regime changes) — a single tested break where there are two or three
under-counts them. Bai & Perron (1998, 2003) generalise the question:
how many breaks, and where, estimated jointly by minimising the residual
sum of squares over every admissible partition of the sample.

The model estimated within each segment is an ordinary OLS regression,
exactly as in Gregory-Hansen. What is new is the search algorithm — a
dynamic programme over partitions, O(T²) rather than an exhaustive
O(T^m) search — and the inference on the *number* of breaks.

### `bai_perron(y, x, max_breaks=5, trim=0.15, selection='ic', ic='bic', n_boot=199, alpha=0.05, seed=None)`

```python
from pyardl.cointegration import bai_perron

res = bai_perron(y, x, max_breaks=5, selection="ic", ic="bic")
print(res.summary())
res.n_breaks         # m_hat
res.break_fractions   # one per estimated break
res.segment_params     # beta_j per segment, m_hat + 1 of them
```

`x` gets its own validation here rather than reusing
[`check_series`](#comparing-with-the-bounds-test) — unlike Gregory-Hansen
or the bounds test, Bai-Perron's model is `y_t = x_t'β_j + u_t` and a
constant regressor is a perfectly ordinary choice, not a degenerate one.
Include any constant/trend column explicitly; nothing is added for you.

### Two routes to choose the number of breaks

`selection='ic'` minimises a standard AIC/BIC on `SSR(m)/T` over
`m = 0, ..., max_breaks`. `selection='sequential'` starts at 0 breaks
and, at each step, tests whether one more break reduces SSR by more than
chance — via a **residual bootstrap** p-value, not a table lookup —
stopping the first time the test fails to reject at `alpha`.

### What is not implemented, and why

The spec names three routes (sequential, UDmax/WDmax, information
criteria) and two specific information criteria (Yao's 1988 modified
BIC, the Liu-Wu-Zidek criterion). Two things are deliberately missing
rather than approximated under a misleading name:

- **`selection='udmax'` raises `NotImplementedError`.** The weighted
  UDmax/WDmax test needs Bai & Perron's (1998) tabulated critical
  values and weights, which this project does not hold with a verified
  provenance — the same discipline as Gregory-Hansen's
  `cv_source='table'`. An *unweighted* version could have been written,
  but shipping it under the name `udmax` would promise a test it is not.
- **`ic='bic'`/`'aic'` are the textbook criteria**, not Yao's modified
  BIC or LWZ — their exact penalty terms are not reproduced here without
  a verified source. Still consistent, just not guaranteed to coincide
  with the modified variants in finite samples.

See `docs/QUESTIONS.md` and `docs/DEVIATIONS.md` for the full reasoning.

### Scope: diagnostic, not (yet) a multi-break cointegration test

The spec distinguishes two uses, deliberately not to be conflated: a
**diagnostic** on any already-specified regression (what this function
does — a classic residual bootstrap, no I(1) structure assumed), and a
**cointegration** generalisation of Gregory-Hansen to `m` breaks
(applying Bai-Perron to Engle-Granger's step-one residuals, which would
need the same I(1) bootstrap null as `gregory_hansen`, doubling the
implementation). Only the diagnostic use is implemented. Consequently,
`bai_perron` at `m=1` is **not yet** guaranteed to coincide with
`gregory_hansen` — they operate on different objects until the
cointegration use is added.
