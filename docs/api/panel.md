# Heterogeneous panels — Mean Group

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

## Cross-checked against R

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

## What comes next

Spec 23 (Pooled Mean Group) constrains the long-run coefficients to be
equal while leaving the short-run dynamics free — the middle ground
between this module and full pooling — and adds the Hausman test that
decides between the two. Spec 24 drops the assumption that individuals
are independent of each other. Both reuse this container and this
per-individual loop.

## References

- Pesaran, M. H. & Smith, R. (1995). Estimating long-run relationships
  from dynamic heterogeneous panels. *Journal of Econometrics*, 68(1),
  79-113.
- Pesaran, M. H., Shin, Y. & Smith, R. P. (1999). Pooled mean group
  estimation of dynamic heterogeneous panels. *Journal of the American
  Statistical Association*, 94(446), 621-634.
- Munnell, A. H. (1990). Why has productivity growth declined?
  Productivity and public investment. *New England Economic Review*,
  Jan/Feb, 3-22.
