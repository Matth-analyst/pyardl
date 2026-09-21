# Unit-root pre-tests

`pyardl.unitroot`

The bounds test is unusual in tolerating a mixture of I(0) and I(1)
regressors — that is what makes it attractive. It has one hard limit:
**it is not valid if any series is I(2)**. Its distribution theory does
not cover that case, and it will not warn you, because it cannot tell.
It will simply return a number.

Screening for that is step zero of the workflow.

## Which tests, and why not the ADF

The classical ADF removes the mean or trend by ordinary least squares.
Under a near-unit root that estimate is poor, and the test loses most of
its power exactly where power matters — against roots close to but below
one. It also over-rejects badly when the errors carry a negative
moving-average component, which macroeconomic data often do.

The two tests here fix those two failures:

- **DF-GLS** (Elliott, Rothenberg & Stock 1996) detrends under a *local
  alternative* instead of under the null, recovering the lost power.
- **The M tests** (Ng & Perron 2001) add an autoregressive long-run
  variance and a modified lag criterion, which is what removes the size
  distortion.

## `report(data, trend='c', alpha=0.05, method='bic')`

The everyday entry point: pre-test every column and tabulate the
verdicts.

```python
>>> from pyardl.unitroot import report
>>> from pyardl.datasets import load_denmark
>>> data = load_denmark()[["LRM", "LRY"]]
>>> print(report(data).reset_index().to_string(index=False))
variable order dfgls_level decision_level dfgls_diff decision_diff mzt_level lags_level
     LRM  I(1)   -1.047542      unit_root  -2.378078    stationary -1.152137          2
     LRY  I(1)   -0.819062      unit_root  -4.630152    stationary -0.813225          0

```

A `PyardlMethodologyWarning` is raised as soon as one series is
I(2)-suspect. That is deliberate: the failure it guards against is
silent.

### The sequential protocol

Testing the level alone cannot answer the question — failing to reject a
unit root in the level is compatible with I(1) *and* with I(2). So each
series is tested twice:

| Level | First difference | Verdict |
|---|---|---|
| rejects | (not needed) | I(0) |
| does not reject | rejects | I(1) |
| does not reject | does not reject | **I(2) suspect** |

**"Suspect" is meant literally.** Failing to reject twice is also what a
short, noisy sample looks like. The report says "suspect" rather than
asserting an order of integration the data cannot establish.

## `dfgls(y, trend='c', lags=None, method='maic', max_lags=None)`

```python
>>> import numpy as np
>>> from pyardl.unitroot import dfgls
>>> rng = np.random.default_rng(0)
>>> y = np.cumsum(rng.standard_normal(200))
>>> res = dfgls(y, trend="ct")
>>> print(res.summary())
DF-GLS (Elliott, Rothenberg & Stock 1996) - trend 'ct', lags=0 (maic), nobs=199
  statistic = -1.5443   decision (5%): unit_root
  critical values (left tail)   1%: -3.5142  5%: -2.9390  10%: -2.6494
  H0: the series has a unit root

```

Left-tailed: a large negative statistic is evidence *against* the unit
root. **Failing to reject is not evidence of a unit root** — it is
absence of evidence, which is exactly why the sequential protocol above
exists.

Use `trend="ct"` when the series clearly trends. Testing a trending
series under `"c"` will almost never reject, whatever the truth.

## `ng_perron(y, trend='c', lags=None, method='maic', max_lags=None)`

Four statistics sharing one long-run variance estimate — same `y` as
above, this time with `trend="c"`:

```python
>>> from pyardl.unitroot import ng_perron
>>> res = ng_perron(y, trend="c")
>>> print(res.summary())
Ng-Perron M tests (2001) - trend 'c', lags=0 (maic), nobs=199
  long-run variance (autoregressive): 0.9272
<BLANKLINE>
  statistic        value    5% bound  decision (5%)
  MZa            -2.6804     -8.6303  unit_root
  MZt            -1.1461     -2.0150  unit_root
  MSB             0.4276      0.2295  unit_root
  MPT             9.0981      3.0866  unit_root
<BLANKLINE>
  H0: the series has a unit root (reject when below)

```

All four are lower-tail: reject when the statistic falls below its
bound. `MZa` and `MZt` are large and negative under stationarity;
`MSB` and `MPT` are positive and shrink towards zero. The rule is
uniform, so there is no direction to get wrong.

`s2_ar` is reported because all four statistics are only as good as that
one estimate.

## Lag selection, and one thing worth knowing

MAIC is the heart of Ng & Perron's contribution: with a negative
moving-average component the plain AIC picks too few lags and the test
over-rejects massively.

But MAIC has a cost, and it is measurable. Its penalty term is large
precisely when the series looks stationary, so it over-selects on I(0)
data — 6.1 lags on white noise against 0.0 for BIC. In the sequential
report that costs power at the differencing stage, turning genuine I(1)
series into false I(2) suspicions. Measured over 40 replications of
length 250:

| criterion | I(0) correct | I(1) correct | I(2) flagged |
|---|---|---|---|
| MAIC | 29/40 | 32/40 | 37/40 |
| BIC | 40/40 | 40/40 | 35/40 |

So the defaults split along the job being done: `report` and
`integration_order` — screening, before you know anything about the
data — default to **BIC**; `dfgls` and `ng_perron` — one series, tested
deliberately — default to **MAIC**. Override either whenever you have a
reason; what would be wrong is to choose by inertia.

`select_lags` also returns the criterion value at every candidate order,
so the choice can be inspected rather than trusted.

## Where the critical values come from

Both tables are simulated in-house, and their provenance differs in one
important respect.

**DF-GLS** has a second source: the response surfaces shipped with
`arch`. The two agree within Monte Carlo error for `T >= 100`. At
`T = 50` they differ by up to 0.037, and an independent size experiment
settles it: our values deliver 5.07% at the nominal 5% level, the
response surface 4.72%. The statistics themselves match to 1e-15 — the
gap is a fitting artefact at the edge of the surface's range.

**The M statistics have no second source at all.** Neither `arch` nor
`statsmodels` implements them. The verification is therefore internal
and rests on a property of the statistics: `MZt` shares the limiting
distribution of DF-GLS, so the two independently simulated tables must
converge as `T` grows. They do — the gap falls from 0.155 at `T = 100`
to 0.006 at `T = 2000`.

Details in
[`PROVENANCE.md`](https://github.com/Matth-analyst/pyardl/blob/main/src/pyardl/critical_values/PROVENANCE.md).
