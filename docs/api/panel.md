# Heterogeneous panels — MG, PMG, DFE, CS-ARDL

`pyardl.panel`

Everything before this page is one country. This one is many, and the
first thing to say about many is a negative result.

## Why pooling is not a shortcut

Take a dynamic panel whose coefficients genuinely differ across
individuals. The natural instinct is to pool: impose a common dynamic,
gain degrees of freedom, accept a little bias for a lot of precision.

Pesaran and Smith (1995) showed that trade is not available. Forcing a
common dynamic when the dynamics differ pushes the heterogeneity into
the error term, where it becomes serial correlation *correlated with the
lagged dependent variable*. That contaminates every coefficient, long-run
ones included, and it does not shrink — so pooled estimators are
**inconsistent even as N and T both go to infinity**. It is not a loss of
efficiency. It is a loss of consistency, and more data does not fix it.

Measured on a heterogeneous DGP with `theta_bar = 0.75`, 2000
replications (Monte Carlo standard error ≈ 0.0014):

| | T = 50 | T = 100 |
|---|---|---|
| Mean Group bias | −0.0050 | **−0.0026** |
| Dynamic fixed effects bias | +0.0242 | **+0.0273** |

The MG bias shrinks as T grows. The DFE bias does not — it drifts
slightly *up*. Raising N does not help it either: at N = 50, T = 100 the
DFE bias is +0.0300. That table is the reason this module exists.

The fix is almost embarrassingly simple: estimate each individual's ARDL
separately, then average.

```
theta_MG = (1/N) Σ theta_i
```

Each `theta_i` is consistent as its own `T_i` grows, so the average is
consistent with no homogeneity assumption at all.

## Use

```python
from pyardl.panel import MeanGroup

res = MeanGroup(df, y="y", X=["x"], id="id", time="t", order=(1, 1)).fit()
print(res.summary())
```

```text
Mean Group estimation (Pesaran & Smith 1995) - 30 individuals
  dependent: y   balanced panel, T from 60 to 60
  aggregator: mean over 30 individuals
  standard errors: BETWEEN-individual dispersion (not pooled within-individual)

  Long-run coefficients
                       theta          se         t         p
    x                 0.6764      0.0269    25.140    0.0000

  Adjustment speed: -0.4098 (se 0.0141)
```

The panel is a long DataFrame: one row per (individual, period).
`order='auto'` selects the lag orders per individual instead of imposing
a common one — both modes exist because they answer different questions.
A common order makes the individual coefficients directly comparable;
per-individual selection lets each dynamic be what its own data say.

## The standard error is not what you expect

This is the part worth reading twice, and it is the trap the
specification flags.

Every individual fit returns a perfectly good standard error for its own
`theta_i`. Combining them into a standard error for the average is
natural, looks like what you do everywhere else — and is wrong.

The variance of `theta_MG` comes from **how much the estimates differ
across individuals**:

```
V(theta_MG) = Σ (theta_i − theta_MG)² / (N(N−1))
```

The reason is conceptual, not arithmetic. `theta_MG` does not estimate a
common coefficient; it estimates the **mean of a distribution** of
coefficients. Its sampling variability is dominated by the fact that the
`theta_i` genuinely differ — a fact about the world, which more data
measures better without making it go away.

### What the naive version costs

Coverage of a nominal 95% interval, 2000 replications (standard error
0.49 point):

| | between-individual | naive (pooled within) |
|---|---|---|
| N=20, T=50 | 95.2% | 54.0% |
| N=20, T=100 | 94.0% | **27.3%** |
| N=50, T=50 | 94.5% | 53.1% |
| N=50, T=100 | 95.2% | **29.2%** |

The correct construction holds nominal everywhere. The naive one gives
54%, then 27%.

**And notice the direction.** It does not merely fail — it gets *worse
as T grows*, halving each time T doubles. An interval that is wrong is
one thing; an interval that becomes more wrong the more data you collect
is another, so the mechanism was measured rather than guessed (300
replications per T, N = 20):

| T | `se_between` | `se_naive` | ratio |
|---|---|---|---|
| 50 | 0.04813 | 0.01845 | 2.61 |
| 100 | 0.04491 | 0.00859 | 5.23 |
| 200 | 0.04469 | 0.00415 | 10.76 |
| 400 | 0.04462 | 0.00200 | **22.26** |

`se_between` converges to a non-zero constant — factor 0.999 between
T = 200 and T = 400 — because it estimates the true dispersion of the
`theta_i`, a property of the population that has nothing to do with how
long the series are. `se_naive` halves at every doubling.

The rate matters: the observed factor is **0.483, not 0.707**. That is
the `1/T` rate of superconsistency, which is what long-run coefficients
get when the regressors are integrated. So the gap between the two grows
like `T`, without bound, and the naive coverage does not converge to a
bad number — it converges to **zero**. Recorded as OBS-20.

Only the between-individual construction is implemented here. The naive
one exists nowhere in the library except as a control in the validation
script.

## MG cross-checked against R

The individual ARDL fits were already validated against the R package
`ARDL` in spec 05. What is new here is the aggregation, and that is
checked against `plm::pmg(model="mg")`, which computes exactly this
average and exactly this between-individual variance:

| quantity | agreement |
|---|---|
| group coefficients | 5.6e-16 |
| between-individual standard errors | 4.9e-17 |
| long-run `theta` and its se | 2.2e-16 |
| adjustment speed and its se | 5.6e-16 |

The specification asks for 1e-4. The R script also re-derives the
aggregation by hand and confirms `plm` matches it to 0.000e+00, so the
reference is readable rather than a black box.

The comparison panel is **generated by pyardl itself** from a fixed seed
(`validation/external/spec22_make_panel.py`): no third-party dataset is
redistributed, and the file is byte-identical on both sides. A second,
independent check on the Produc panel of Munnell (1990), which ships
with `plm` and is therefore *not* bundled here, agreed to 4.5e-14 on all
four coefficients and standard errors.

## What the container refuses to do quietly

A broken panel produces plausible numbers, which is why validation
happens before estimation and every exclusion is named.

- **Time is sorted, always.** A dynamic model reads its lags off the row
  order. A panel sorted by anything else silently lags the wrong
  observations and nothing downstream can detect it.
- **A gap inside an individual's history excludes it.** A missing year is
  not a shorter history: lagging across the hole pairs observations two
  periods apart and calls it a one-period lag. Leading and trailing NaNs
  are trimmed; internal ones are not bridged.
- **Duplicate periods are an error**, not a row to pick.
- **Every dropped individual is recorded** in `panel.excluded` with its
  reason, so an `N` in a results table can always be accounted for.

Unbalanced panels are fine — the framework needs large `T`, not equal
`T`. Below 30 observations per individual the call warns: at small `T`
each individual estimate carries the dynamic-panel bias, and averaging N
of them does not remove a bias they all share.

## Reading the result

```python
res.longrun          # theta, se, t, p, ci_lower, ci_upper
res.coefficients     # group mean of the raw ARDL coefficients
res.adjustment       # mean lambda, its se, share of non-adjusting units
res.theta_i          # the individual estimates that were averaged
res.individual["u03"]  # the full ARDLResults for one individual
res.heterogeneity()  # spread of the theta_i, per coefficient
res.non_adjusting    # individuals with lambda_i >= 0
```

`heterogeneity()` is the table that says whether pooling would have been
defensible at all — a small dispersion relative to the mean is the case
where MG and a pooled estimator agree, and a large one is the case
Pesaran and Smith wrote the paper about:

```text
       mean        sd       min    median       max        cv
x  0.676414  0.147372  0.383456  0.705594  0.978701  0.217872
```

**Individuals with `lambda_i >= 0` are kept and named.** Error correction
requires a negative adjustment speed; an individual without one is not
returning to any long-run relation, so its `theta_i` is not a long-run
coefficient in the sense being averaged. Dropping them would be
selecting on the outcome, so they stay in the average, the call warns,
and `res.non_adjusting` lists them.

`res.coefficients` is empty under `order='auto'` when the selected orders
differ: coefficients from different specifications are not the same
quantity, and averaging them would produce a number about nothing. The
long-run `theta` remains comparable, which is why it is still reported.

## Pooled Mean Group — the middle term

`pyardl.panel.PMG`

Pool everything and you lose consistency; pool nothing and you pay for
it in noise. Pesaran, Shin and Smith (1999) put a third option between
them, and it is the one applied work actually reaches for: **constrain
the long-run coefficients to be equal, leave every short-run dynamic
free.**

```
Dy_it = lambda_i (y_{i,t-1} - theta' x_{i,t-1})
        + short-run terms_i + mu_i + e_it
```

The economics is that long-run relations often come from theory — a
budget constraint, an arbitrage condition — which applies to everyone,
while the speed at which each country returns to it plainly does not.

```python
from pyardl.panel import PMG

res = PMG(df, y="y", X=["x"], id="id", time="t", order=(1, 1)).fit()
print(res.summary())
```

```text
Pooled Mean Group (Pesaran, Shin & Smith 1999) - 25 individuals, 1475 observations
  method: backfitting, converged in 29 iterations
  log-likelihood: -636.177651
  long-run coefficients POOLED; short-run dynamics free

  Long-run coefficients (common)
                       theta          se         z         p
    x                 0.7520      0.0065   115.652    0.0000

  Mean adjustment speed: -0.4344 (se 0.0142, between-individual)
```

### How it is estimated

The likelihood has `k` common parameters and `4N` individual ones, but
it concentrates. **Given** `theta`, each individual block is an ordinary
least-squares regression of `Dy_i` on `[xi_i(theta), DW_i]`, where
`xi_it = y_{i,t-1} - theta'x_{i,t-1}`. **Given** the dynamics, `theta`
solves one stacked weighted least-squares problem, each individual
weighted by `lambda_i / sigma_i`. Alternating the two is back-fitting,
which is what `xtpmg` does; `method='newton'` maximises the same
concentrated likelihood directly, and a test pins the two together to
1e-6.

The iteration starts from `theta_MG`, which is consistent under both the
null and the alternative — so it only ever has to travel the efficiency
gap, never the whole space. `res.iterations` keeps the log.

### The variance formula, and a bug worth naming

The covariance of `theta` is the Schur complement of the block-arrow
information matrix:

```
V(theta) = [ sum_i (lambda_i^2 / sigma_i^2) X_i' M_[xi_i, W_i] X_i ]^-1
```

**The projection must sweep out `xi_i` as well as `W_i`.** Both
`lambda_i` and `gamma_i` are estimated, so both derivative directions
have to go. The first version of this function projected on `W_i` only
and returned a standard error about **5% too small** — narrow intervals,
inflated `t`, over-rejection.

Nothing internal would have caught it. The number was finite, positive,
the right order of magnitude, stable and reproducible. It was caught by
comparing against the **numerical Hessian of the concentrated
log-likelihood** — the definition of the profile information, computed
independently of the formula. After the fix, the formula reproduces
`ardlverse` to **8e-12**. The test suite keeps that comparison. OBS-21.

`vcov='observed'` exposes the numerical Hessian itself. The two are
asymptotically equivalent and differ by about 2% here; the default is
the analytic one because it is what PSS published and what `xtpmg`
computes, which is what makes the cross-check meaningful.

## What PMG buys — and where it breaks

Measured on 2000 replications, N = 25, T = 60, standard error 0.49
point, varying the dispersion of the true `theta_i`.

**Under exact homogeneity it delivers what it promises:**

| `theta_sd` | MG bias | PMG bias | var MG / var PMG |
|---|---|---|---|
| 0.00 | −0.50% | **−0.14%** | **2.41x** |
| 0.10 | −0.58% | +2.55% | 0.61x |
| 0.25 | −0.12% | +15.26% | 0.36x |

A bias of 0.14% where the specification asked for under 1%, and a 2.41x
efficiency gain over MG. Then it degrades, fast.

**And this is the part to read twice:**

| `theta_sd` | MG coverage | PMG coverage | Hausman rejects |
|---|---|---|---|
| 0.00 | 94.5% | 92.3% | 8.6% |
| 0.10 | 94.2% | **36.2%** | 18.6% |
| 0.25 | 94.8% | **7.3%** | 59.8% |

At `theta_sd = 0.10` — a 13% dispersion around 0.75, nothing exotic for
a panel of countries — PMG is biased by 2.55%, its 95% interval covers
**36%**, and its efficiency advantage has *inverted*: at 0.61x it is now
less precise than MG as well as biased.

**And the guard does not fire.** At that same dispersion the Hausman
test rejects only **18.6%** of the time. In more than four samples out of
five where PMG is already materially wrong, the standard diagnostic
answers "PMG is fine". That is not a theoretical grey zone; it is the
regime the estimator is used in.

The Hausman size is not exact either: 8.6% against a nominal 5% under
perfect homogeneity, seven standard errors high. It over-rejects when it
should not and under-rejects when it should — both directions make its
verdict less informative. Recorded as OBS-22.

MG, by contrast, does not move: bias between −0.58% and −0.12%, coverage
94.2–94.8% throughout. That is the difference between a consistent
estimator and an efficient-under-a-condition one, in numbers.

**What to do with that.** Do not read a non-significant Hausman as a
green light for PMG. The dispersion of the MG estimates —
`mg_res.heterogeneity()` — is more direct and depends on no test at all:
if the coefficient of variation is visible, PMG is already in trouble
whether or not Hausman says so.

## The Hausman test

```python
from pyardl.panel import hausman

result = hausman(mg_res, pmg_res)
print(result.summary())
```

```text
Hausman test, MG versus PMG (Pesaran, Shin & Smith 1999)
  H0: the long-run coefficients are common across individuals
      (PMG consistent AND efficient; MG consistent but noisy)
  H1: they are not (only MG is consistent)

  chi2(1) = 0.0025   p = 0.9603
  do not reject homogeneity: PMG is consistent and efficient
```

The variance difference `V(MG) - V(PMG)` is only guaranteed positive
definite asymptotically, and in finite samples frequently is not — 1.8%
of replications here. Rather than fail, or return a negative statistic
without comment, a pseudo-inverse is used, the degrees of freedom become
its rank, and `used_pseudo_inverse` records the fact so `summary()` can
say the p-value is indicative.

## The comparison table

```python
from pyardl.panel import compare

table, hausman_result = compare(df, y="y", X=["x"], id="id", time="t")
```

```text
                        theta        se           t
estimator regressor
MG        x          0.752359  0.009881   76.140136
PMG       x          0.751988  0.006502  115.652117
DFE       x          0.753622  0.008550   88.140645
```

`DFE` — dynamic fixed effects, everything pooled but the intercepts — is
included because panel papers report the three side by side, and because
seeing its bias next to the others is the clearest argument against it.
Its `summary()` says so rather than presenting it as an option.

## PMG and DFE cross-checked against R

`ardlverse::panel_ardl()` states that it replicates Stata's `xtpmg`,
which is the reference the specification names; Stata is not available
here. On a panel pyardl generates itself from a fixed seed:

| quantity | agreement |
|---|---|
| PMG `theta` | 1.9e-08 |
| PMG standard error | 2.1e-10 |
| PMG adjustment speed | 5.7e-09 |
| PMG log-likelihood | 4.1e-12 |
| DFE `theta`, se, `lambda` | 1.1e-16 |

The specification asks for 1e-3.

**One methodological note worth keeping.** The first comparison showed a
2.7e-07 gap on `theta` — small enough to shrug at, or to absorb by
loosening one's own tolerance to "match the reference". What prevented
that was the **concentrated log-likelihood**: both implementations
maximise the same function, so it ranks them, and it was *lower* at
`ardlverse`'s estimate than at pyardl's. Re-running the reference at
`tol=1e-8` instead of its default `1e-6` moved it to within 6.7e-10,
with identical log-likelihoods.

When two implementations disagree, splitting the difference is not an
inference, and neither is deferring to the published one. If both
optimise an explicit objective, compute it — the coefficients alone
looked like agreement. A package default is not a specification. OBS-21.

## One assumption is doing real work

The likelihood is a product over individuals, which assumes they are
**independent of each other**. Common shocks — a world cycle, a
commodity price — break it, and then both MG and PMG are biased.
Nothing in this module corrects for that, and `summary()` says so rather
than letting the reader assume it was handled.

## When individuals are not independent — CS-ARDL and CS-DL

`pyardl.panel.CSARDL`, `pyardl.panel.CSDL`

Everything above assumes individuals are independent of each other. They
usually are not. A world business cycle, a commodity price, a common
policy shock — one thing moves everyone at once. Write it as a factor:

```
y_it = beta_i' x_it + gamma_i' f_t + e_it
```

`f_t` is unobserved and correlated with `x_it`, so omitting it biases
every `theta_i` *before* the averaging of spec 22 even begins. MG and
PMG are both affected, and more data does not help.

Pesaran's move is to stop trying to observe the factor. Average the
observed variables across individuals at each date: since the loadings
average to something non-degenerate, those **cross-sectional averages
span the same space as the factor** asymptotically. Add them as
regressors and the factor is controlled for without ever being
estimated.

### Two estimators

**CS-ARDL** keeps the dynamics and adds the averages plus `p_z` of their
lags — the lags matter because in a dynamic panel the lagged dependent
variable drags the factor's own history into the equation. The long run
is rebuilt from the short-run coefficients, as in spec 03.

```python
from pyardl.panel import CSARDL

res = CSARDL(df, y="y", X=["x"], id="country", time="year",
             order=(1, 1), cs_lags="auto").fit()
print(res.summary())
```

```text
CS-ARDL (Chudik & Pesaran 2015) - 30 individuals
  ARDL(1, 1) augmented with cross-sectional averages and 4 lag(s) of them
  standard errors: BETWEEN-individual dispersion (Mean Group)

  Long-run coefficients
                       theta          se         t         p
    x                 0.8058      0.0042   193.139    0.0000

  Mean adjustment speed: -1.0451 (se 0.0203)
```

The adjustment speed of −1.05 on this particular panel is not a
pathology: the reference DGP is *static*, so fitting an ARDL(1,1) to it
finds an autoregressive coefficient near zero and an adjustment that
completes within the period. On a genuinely dynamic panel it lands where
you would expect.

**CS-DL** skips the dynamics entirely: `y` on `x`, a few lagged
*differences* of `x`, and the averages. The coefficient on `x` **is**
the long-run coefficient — no ratio of estimated dynamics is ever
formed, which is exactly what makes it robust to getting the lag order
wrong.

```python
from pyardl.panel import CSDL

res = CSDL(df, y="y", X=["x"], id="country", time="year").fit()
```

The price is stated rather than hidden: the truncation is innocuous only
if adjustment is fast enough, and **CS-DL cannot report an adjustment
speed at all**. Its `summary()` says so instead of leaving a blank where
a number would be expected.

Both aggregate the Mean Group way, so the standard error is the
between-individual dispersion of spec 22 — not anything pooled from
within.

### `cs_lags="auto"` and a floating-point trap

The default is `floor(T**(1/3))`, the rule of thumb of Chudik and
Pesaran. Computed with `numpy.cbrt`, not `T ** (1/3)`, and the
difference is not cosmetic: at a perfect cube the power form lands just
*below* the integer, so the floor loses a lag.

```
64 ** (1/3)   == 3.99999999999999956   -> floor 3, should be 4
1000 ** (1/3) == 9.99999999999999822   -> floor 9, should be 10
```

A silently shorter lag list is a different specification, not a rounding
detail. The test suite pins both forms against each other.

### Collinearity is handled by a rule, not by the solver

With `k+1` averages, `p_z` lags of each and a modest `T`, the individual
design is often rank-deficient. Dropping columns is unavoidable. Doing
it *by whatever the linear algebra happens to prefer today* is not: a
different BLAS on another machine would keep different columns and
report different long-run coefficients from the same data — a result
that looks like a result and is not reproducible.

So the rule is fixed and stated. Columns are examined **left to right in
a declared order** — deterministic terms, own lags, own regressors, then
the cross-sectional averages from contemporaneous to most-lagged — and a
column is dropped when it adds nothing to the rank of what precedes it.

**The averages come last on purpose.** They are the approximation, so
when something must go it should be the approximation rather than the
model. Every drop is recorded in `res.dropped_columns` and mentioned in
`summary()`.

### The CD test, and its direction

```python
from pyardl.panel import cd_test

before = cd_test(residuals_of_a_plain_MG)
print(before.summary(context="before"))

after = res.cd_test()
print(after.summary(context="after"))
```

Pesaran's CD test has a null of *no* cross-sectional dependence, and it
is used twice with **opposite desired answers**: before augmenting, a
rejection is what motivates the whole exercise; after augmenting, a
*failure* to reject is the good outcome. The same p-value means
different things in the two, so `summary(context=...)` states which
reading applies rather than leaving it to memory:

```text
Pesaran CD test for cross-sectional dependence
  H0: residuals are cross-sectionally independent
  CD = 54.9540   p = 0.0000   (30 individuals, 435 pairs)
  mean |pairwise correlation| = 0.2968
  reject at 5%: a common factor is present, so MG and PMG are biased and the cross-sectional augmentation is warranted.
```

Run *after* the augmentation, on the same panel, the picture changes but
does not become clean:

```text
  CD = -6.1169   p = 0.0000   (30 individuals, 435 pairs)
  mean |pairwise correlation| = 0.1043
  reject at 5%: dependence SURVIVES the augmentation - more lags of the averages, or more factors than the averages can span.
```

The average absolute correlation fell from 0.30 to 0.10 — most of the
dependence is gone — but 435 pairs give the test enough power to see
what remains, so it still rejects. That is worth reporting rather than
smoothing over: the augmentation is an approximation, and the CD test is
honest about how good an approximation it was here.

`mean_abs_correlation` sits next to the statistic for a reason: a CD near
zero can mean correlations genuinely are near zero, or that positive and
negative ones cancelled. Those call for different conclusions, so both
numbers are reported.

Pairs are matched **on the index**, never by position — two individuals
with different sample windows must not have their residuals lined up by
row number, which would correlate different dates.

### What the augmentation is worth

On the reference panel — a common factor entering both `y` and `x`,
heterogeneous loadings, true `theta = 0.80`:

| estimator | `theta` | error |
|---|---|---|
| Mean Group (spec 22), no augmentation | **1.1938** | **+49%** |
| CS-ARDL | 0.8058 | +0.7% |
| CS-DL | 0.8059 | +0.7% |

A 49% error is not a loss of efficiency that a larger sample would
repair; it is the omitted factor being correlated with the regressor.
That single row is the reason this module exists.

### And what it costs — which is not nothing

A dimensioned Monte Carlo (1000 replications, N = 30, T = 80, Monte
Carlo standard error ≈ 0.0004) over the strength of the loading, on a
**dynamic** DGP:

| `gamma` | MG | bias | CS-ARDL | bias | CS-DL | bias |
|---|---|---|---|---|---|---|
| 0.0 | 0.7973 | −0.0027 | 0.7917 | −0.0083 | 0.7732 | **−0.0268** |
| 0.3 | 0.7972 | −0.0028 | 0.7931 | −0.0069 | 0.7729 | **−0.0271** |
| 0.6 | 0.7966 | −0.0034 | 0.7916 | −0.0084 | 0.7705 | **−0.0295** |

Two things here contradict what the introduction above would lead you to
expect, and both are worth knowing.

**MG is not biased in this DGP, at any loading.** Its bias sits at
−0.003 throughout. The reason is in the data, not the estimator: here
the factor enters `y` through its *difference*, `gamma*(f_t − f_{t−1})`,
which is I(0) and leaves the long-run relation alone — while `x`
contains `gamma*f`, which is I(1). The factor is present, it is
correlated with the regressor, and it biases nothing.

On the reference panel, where the factor enters `y` in **levels**, the
bias is the 49% shown above. So it is not the presence of a common
factor that breaks MG — it is whether that factor's contribution to `y`
is **persistent**. A large but transitory common shock leaves the long
run intact.

**The augmentation costs something even when it is not needed.** At
`gamma = 0` — no factor at all — CS-DL carries a bias of −0.027, 3.4% of
the coefficient, and CS-ARDL −0.008. Reaching for CS-DL reflexively is
not free.

### A warning about reading the CD test after augmentation

In the same Monte Carlo, the CD test rejects **100% of the time after
augmentation, including at `gamma = 0`**:

| `gamma` | CD rejects before | CD rejects after |
|---|---|---|
| 0.0 | 17.9% | **100%** |
| 0.3 | 100% | **100%** |
| 0.6 | 100% | **100%** |

The augmentation does not merely fail to absorb the dependence — it
*induces* some. Every individual is regressed on averages that contain
its own `y` (at N = 30, a weight of 1/30), which creates a mechanical
**negative** correlation between residuals. The signature is visible on
the reference panel: `CD = −6.12`, a negative statistic, with a mean
absolute correlation of only 0.10.

**So a significant CD on CS-ARDL residuals does not mean "factors
remain".** It may be nothing but the self-inclusion effect. Reading it
as a diagnostic of residual factors leads to adding lags that buy
nothing. Recorded as OBS-23.

### Cross-checked against R — and the limits of that check

This is the part to read before trusting the numbers.

`plm::pcce(model="mg")` implements the **static** CCE of Pesaran (2006),
which is exactly the special case of CS-DL with no lagged differences
and no lags of the averages. On a panel pyardl generates itself:

| quantity | agreement |
|---|---|
| group `theta` | 1.1e-16 |
| between-individual standard error | 1.5e-16 |
| individual `theta_i` | ~1e-15 |

The R script also re-derives the aggregation by hand from the individual
OLS and lands on the same digits, so the reference is readable rather
than a black box.

**That validates three things and no more**: the construction of the
cross-sectional averages, the augmented per-individual regression, and
the Mean Group aggregation with its between-individual variance.

**The dynamic half has no external reference here.** Lags of the
averages, and the long run rebuilt from short-run coefficients, are
covered by the internal tests and the Monte Carlo only. The
specification names Stata's `xtdcce2` as the reference; Stata is not
available in this environment and no R package implements the full
CS-ARDL — which is precisely why the specification calls this a blank
area. A `.do` script is provided at
`validation/external/spec24_xtdcce2.do` for the day it can be run, and
no reference values have been invented in the meantime.

## What comes next

The panel branch is complete: MG, PMG, DFE, CS-ARDL and CS-DL all share
one container and one per-individual loop, and one aggregation rule.

What is *not* here, and is flagged rather than omitted silently: the
strong/weak dependence exponent of Bailey, Kapetanios and Pesaran, which
the specification places outside the first version; and an external
reference for the dynamic half of CS-ARDL, which waits on Stata.

## References

- Chudik, A. & Pesaran, M. H. (2015). Common correlated effects estimation
  of heterogeneous dynamic panel data models with weakly exogenous
  regressors. *Journal of Econometrics*, 188(2), 393-420.
- Chudik, A., Mohaddes, K., Pesaran, M. H. & Raissi, M. (2016). Long-run
  effects in large heterogeneous panel data models with cross-sectionally
  correlated errors. *Advances in Econometrics*, 36, 85-135.
- Pesaran, M. H. (2006). Estimation and inference in large heterogeneous
  panels with a multifactor error structure. *Econometrica*, 74(4),
  967-1012.
- Pesaran, M. H. (2015). Testing weak cross-sectional dependence in large
  panels. *Econometric Reviews*, 34(6-10), 1089-1117.
- Pesaran, M. H. & Smith, R. (1995). Estimating long-run relationships
  from dynamic heterogeneous panels. *Journal of Econometrics*, 68(1),
  79-113.
- Pesaran, M. H., Shin, Y. & Smith, R. P. (1999). Pooled mean group
  estimation of dynamic heterogeneous panels. *Journal of the American
  Statistical Association*, 94(446), 621-634.
- Munnell, A. H. (1990). Why has productivity growth declined?
  Productivity and public investment. *New England Economic Review*,
  Jan/Feb, 3-22.
