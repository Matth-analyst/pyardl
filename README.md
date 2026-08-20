<div align="center">

# pyardl

**ARDL models, bounds tests for cointegration, and critical values you can trace back to their source.**

[![CI](https://github.com/Matth-analyst/pyardl/actions/workflows/ci.yml/badge.svg)](https://github.com/Matth-analyst/pyardl/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![codecov](https://codecov.io/gh/Matth-analyst/pyardl/branch/main/graph/badge.svg)](https://codecov.io/gh/Matth-analyst/pyardl)

</div>

---

## The problem this solves

You have macroeconomic series and you suspect a long-run relationship. The
classical route — Engle-Granger, Johansen — asks you to establish the
integration order of every series first, and is invalid if you get it wrong or
if the orders are mixed.

Mixed orders are the normal case. Here is what happens on the Danish
money-demand data shipped with this package:

```python
from pyardl.datasets import load_denmark
from pyardl.unitroot import report

data = load_denmark()
print(report(data[["LRM", "LRY", "IBO", "IDE"]]))
```

```text
         order dfgls_level decision_level dfgls_diff decision_diff mzt_level
variable
LRM       I(1)   -1.047542      unit_root  -2.378078    stationary -1.152137
LRY       I(1)   -0.819062      unit_root  -4.630152    stationary -0.813225
IBO       I(1)   -1.744509      unit_root  -2.925082    stationary  -1.64134
IDE       I(0)   -2.445024     stationary        NaN               -2.612435
```

Three I(1) series and one I(0). Engle-Granger has no right to run on this, and
when you run it anyway it finds nothing:

```text
Engle-Granger test (1987) - trend 'c', 4 variables, lags=3, nobs=55
  statistic = -3.3147   p-value = 0.2611
  decision (5%): no_cointegration
```

The bounds test of Pesaran, Shin & Smith does not need the orders to be known,
and on the same data it finds the relationship:

```text
F_overall = 6.2059   decision (5%): cointegration
t_BDM     = -4.5479   decision (5%): cointegration
F_indep   = 8.1619   decision (5%): cointegration

CLASSIFICATION (5%): cointegration
```

That gap is the reason this library exists.

Engle-Granger and Johansen ship here too — you should be able to compare — but
they are the point of reference, not the recommended route.

---

## Install

`pyardl` is not on PyPI yet. Install from source:

```bash
pip install git+https://github.com/Matth-analyst/pyardl.git
```

Or clone it, which is what you want if you intend to read the code:

```bash
git clone https://github.com/Matth-analyst/pyardl.git
cd pyardl
pip install -e ".[dev,plot,bootstrap]"
```

Requires Python 3.11+. Runtime dependencies are numpy, scipy, pandas and
statsmodels — nothing else. `matplotlib` (extra `plot`) and `arch` (extra
`bootstrap`) are optional and imported lazily.

---

## Sixty seconds

```python
from pyardl.bounds import bounds_test
from pyardl.datasets import load_denmark

data = load_denmark()
res = bounds_test(data["LRM"], data[["LRY", "IBO", "IDE"]], case=3)
print(res.summary())
```

```text
Bounds test (Pesaran, Shin & Smith 2001) - case 3, k=3, ECM(3; LRY:1, IBO:3, IDE:2), critical values: kripfganz

F_overall = 6.2059   decision (5%): cointegration
F p-values: p_I0 = 0.0005, p_I1 = 0.0039
t_BDM     = -4.5479   decision (5%): cointegration
F_indep   = 8.1619   decision (5%): cointegration

CLASSIFICATION (5%): cointegration
  F_overall, t_BDM and F_indep all reject: the level terms are jointly significant, y adjusts back towards equilibrium, and the regressors carry the long-run relationship.

        F_I0   F_I1   t_I0   t_I1  F_indep_I0  F_indep_I1
alpha                                                    
0.10   2.730  3.747 -2.570 -3.460       2.084       3.864
0.05   3.229  4.322 -2.860 -3.780       2.619       4.646
0.01   4.311  5.543 -3.430 -4.370       3.814       6.311
```

---

## Table of contents

- [Design principles](#design-principles)
- [The workflow](#the-workflow)
  - [Step 0 — Screen for I(2)](#step-0--screen-for-i2)
  - [Step 1 — Choose lag orders](#step-1--choose-lag-orders)
  - [Step 2 — Run the bounds test](#step-2--run-the-bounds-test)
  - [Step 3 — Read the long run](#step-3--read-the-long-run)
  - [Step 4 — Test what theory says](#step-4--test-what-theory-says)
  - [Step 5 — Check the model held still](#step-5--check-the-model-held-still)
  - [Step 6 — Remove the inconclusive zone](#step-6--remove-the-inconclusive-zone)
- [API reference](#api-reference)
  - [Bootstrap bounds test](#bootstrap-bounds-test)
  - [Johansen test](#johansen-test)
  - [VECM simulator](#vecm-simulator)
- [Validation](#validation)
- [Compatibility](#compatibility)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citing](#citing)
- [References](#references)

---

## Design principles

Four rules shape the API. They are the reason to prefer this library over
writing the same tests yourself.

**An inconclusive result stays inconclusive.** The bounds test compares a
statistic against a *pair* of critical values, so the verdict has three states —
`cointegration`, `no_cointegration`, `inconclusive` — and never collapses to a
boolean. When the answer lands in the middle you get the p-value interval, so
you can see how close it was.

**Nothing is silently substituted.** Ask for a critical value that does not
exist — a level nobody tabulated, more regressors than the tables cover,
`trend='n'` for Engle-Granger — and you get an exception naming a source that
does have it. Never a neighbouring cell quietly returned in its place.

**Invalid inference is refused, not decorated.** A confidence interval on the
speed of adjustment is produced only once cointegration is established; before
that its distribution is non-standard, so you get `NaN` and a warning.
First-step Engle-Granger coefficients come with no standard errors at all,
because the usual ones are wrong there.

**Every number is traceable.** Every critical value ships with its exact source,
and every table was cross-checked against an independent one or against an
in-house Monte Carlo engine. See [Validation](#validation).

---

## The workflow

### Step 0 — Screen for I(2)

The bounds test tolerates a mix of I(0) and I(1). It is **invalid if any series
is I(2)**, and it cannot detect that itself — it would return a number.

```python
from pyardl.unitroot import report

print(report(data))          # DF-GLS on the level, then on the difference
```

Each series is tested twice, because failing to reject a unit root in the level
is compatible with I(1) *and* I(2):

| Level | First difference | Verdict |
|---|---|---|
| rejects | (not needed) | I(0) |
| does not reject | rejects | I(1) |
| does not reject | does not reject | **I(2) suspect** |

"Suspect" is literal. A double failure to reject is also what a short, noisy
sample looks like. A `PyardlMethodologyWarning` fires when it happens.

### Step 1 — Choose lag orders

```python
from pyardl.core.ardl import ARDL

sel = ARDL.select_order(y, x, max_p=3, max_q=3, ic="bic")
print(sel.top(4).round(3))
```

```text
   p  q_LRY  q_IBO  q_IDE      aic      bic       hq      llf  nobs
0  3      1      0      0 -248.783 -231.222 -242.050  133.391    52
1  3      1      0      1 -247.580 -228.067 -240.099  133.790    52
2  3      2      0      0 -247.304 -227.792 -239.823  133.652    52
3  3      1      1      0 -246.834 -227.322 -239.354  133.417    52
```

Every candidate is estimated on the **same sample** — note the constant `nobs`
column. Comparing information criteria computed on different numbers of
observations is a silent, common mistake that biases the choice towards short
lags. Look at `top(n)` rather than only the winner: criteria often separate the
leading specifications by very little.

### Step 2 — Run the bounds test

```python
from pyardl.bounds import bounds_test

res = bounds_test(y, x, case=3, order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}))
```

Three statistics, and **all three** must reject:

- **`F_overall`** tests that all level terms are jointly zero.
- **`t_BDM`** tests the adjustment coefficient alone. Left-tailed: rejection
  requires a *negative* estimate, an actual pull back towards equilibrium.
- **`F_indep`** tests the regressors' levels alone. Without it, two situations
  that are not cointegration pass for it.

```python
label, reason = res.classification()
```

| `F_overall` | `t_BDM` | `F_indep` | `classification()` |
|---|---|---|---|
| rejects | rejects | rejects | `cointegration` |
| rejects | rejects | does not | `degenerate_1` |
| rejects | does not | rejects | `degenerate_2` |
| does not | does not | does not | `no_cointegration` |
| anything else | | | `inconclusive` |

**`degenerate_1`** — `y` adjusts towards its own past while the regressors
carry nothing. What looks like error correction is `y` returning to a constant.

**`degenerate_2`** — the regressors' levels are jointly significant, but
nothing pulls `y` back. No mechanism restores the relationship, so nothing
holds it together.

The mapping is total: every combination of three three-state verdicts lands on
a named outcome, and `reason` says in one sentence which test decided. The
older two-test `decision_joint` is still there, but it can only *suspect* a
degeneracy — with two tests, the information needed to tell them apart does not
exist.

### Step 3 — Read the long run

```python
model = ARDL(y, x, order=(3, {"LRY": 1, "IBO": 3, "IDE": 2})).fit()
print(model.longrun.round(4))
print(res.adjustment().round(4))
```

```text
      theta      se
LRY  0.9965  0.1239
IBO -4.5381  0.5203
IDE  2.8915  0.9951

lambda     -0.4169
se          0.0917
ci_lower   -0.5965
ci_upper   -0.2372
```

Standard errors come from the delta method with an analytical gradient. About
42% of a disequilibrium is corrected each quarter. The confidence interval
appears only because cointegration was established.

### Step 4 — Test what theory says

An income elasticity of 0.9965 *looks* like one. With a standard error of 0.124,
so would 0.85. Ask properly:

```python
out = model.test_longrun_restriction([[1.0, 0.0, 0.0]], 1.0, impose=True)
print(out.summary())
```

```text
Long-run restriction test - Wald chi2(1) = 0.0008, p = 0.9773
  decision (5%): not_rejected
  R.theta - r = [-0.0035]

  imposed: F = 0.0008, p = 0.9774
  SSR unrestricted = 0.014228, restricted = 0.014229
```

With `impose=True` the two level terms collapse into `(LRM − LRY)` — the
velocity of money, a ratio theory expects to be stationary. The restriction
costs nothing in fit and buys an interpretable model plus a degree of freedom.
This is the discipline of Davidson, Hendry, Srba & Yeo (1978).

### Step 5 — Check the model held still

A long-run coefficient estimated across a structural break is an average of two
regimes, not an equilibrium.

```python
print(model.stability())
```

```text
                  stable  max_excess  first_crossing
test
CUSUM               True         0.0             NaN
CUSUM-of-squares    True         0.0             NaN
```

Both are reported, always, because they fail differently — see
[Stability diagnostics](#stability-diagnostics).

---

### Step 6 — Remove the inconclusive zone

The bounds test compares a statistic against *two* critical values, because the
true distribution depends on integration orders nobody knows. When the
statistic lands between them the answer is **inconclusive** — and on the sample
sizes this literature works with, that happens often enough to be a practical
problem.

The bootstrap builds the distribution instead of bracketing it: regenerate the
data many times under a null that is true by construction, recompute the
statistic each time, read the critical value off the result.

```python
from pyardl.bootstrap import bootstrap_bounds_test

res = bootstrap_bounds_test(
    y, x, case=3, order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}),
    n_boot=2999, seed=42,
)
print(res.summary())
```

```text
Bootstrap bounds test (McNown, Sam & Goh 2018) - case 3, B=2999, resample='iid', seed=42

F_overall = 6.2059   bootstrap p = 0.0123   decision (5%): cointegration
t_BDM     = -4.5479   bootstrap p = 0.0110   decision (5%): cointegration
F_indep   = 8.1619   bootstrap p = 0.0090   decision (5%): cointegration

  CLASSIFICATION (5%): cointegration
  F_overall, t_BDM and F_indep all reject: the level terms are jointly significant, y adjusts back towards equilibrium, and the regressors carry the long-run relationship.

  bootstrap critical values
    alpha           F           t     F_indep
      0.1      4.1105     -3.3914      4.8594
     0.05      4.8319     -3.7780      5.7467
     0.01      6.5195     -4.5739      7.8825

  bootstrap against classical bounds (5%)
        test      stat   boot cv   boot p     I(0)     I(1)  boot             bounds           
   F_overall    6.2059    4.8319   0.0123    3.229    4.322  cointegration    cointegration    
       t_BDM   -4.5479   -3.7780   0.0110   -2.860   -3.780  cointegration    cointegration    
     F_indep    8.1619    5.7467   0.0090    2.619    4.646  cointegration    cointegration    

  classification: bootstrap -> cointegration, bounds -> cointegration
```

Both routes are reported side by side, because a disagreement between them is
itself a result: `res.comparison()` returns it as a frame,
`res.agrees_with_bounds()` as a boolean.

**What the bootstrap buys, measured** — 1000 replications, `T = 100`, on the
four canonical systems:

| DGP | bootstrap correct | bounds correct | bounds inconclusive |
|---|---|---|---|
| cointegration | 100.0% | 100.0% | 0.0% |
| degenerate_1 | 99.4% | 93.2% | 5.5% |
| degenerate_2 | 96.3% | 99.8% | 0.1% |
| no cointegration | 91.5% | 71.3% | 24.8% |

Full guidance on which route to believe, including what happens when they
disagree, is in [Bootstrap or classical bounds?](docs/bootstrap-or-bounds.md).

Almost all of the gain is the disappearance of the inconclusive zone, and it
shows only where that zone is wide. Where neither route hesitates, the
bootstrap adds nothing — and under a type 2 degeneracy it is confidently wrong
3.7% of the time against the bounds' 0.1%. Deciding has a price, and it is
recorded rather than advertised away.

---

## API reference

### Unit-root pre-tests

`pyardl.unitroot`

| Function | Purpose |
|---|---|
| `report(data, trend, alpha, method)` | Sequential screening, one row per variable |
| `integration_order(y, ...)` | Same, for a single series |
| `dfgls(y, trend, lags, method, max_lags)` | DF-GLS test (Elliott, Rothenberg & Stock 1996) |
| `ng_perron(y, ...)` | The four M statistics (Ng & Perron 2001) |
| `gls_detrend`, `ols_detrend`, `adf_regression`, `select_lags` | The shared machinery, exposed |

The classical ADF removes the mean by ordinary least squares, which is what
costs it most of its power under a near-unit root. DF-GLS detrends under a
*local alternative* instead and recovers it. The M tests add an autoregressive
long-run variance and a modified lag criterion, which is what removes the size
distortion the ADF suffers when the errors carry a negative moving-average
component.

```text
Ng-Perron M tests (2001) - trend 'c', lags=2 (maic), nobs=52
  long-run variance (autoregressive): 0.0050

  statistic        value    5% bound  decision (5%)
  MZa            -3.9863     -9.3800  unit_root
  MZt            -1.1521     -2.1023  unit_root
  MSB             0.2890      0.2205  unit_root
  MPT             6.4313      2.8557  unit_root

  H0: the series has a unit root (reject when below)
```

All four are lower-tail, so there is no direction to get wrong.

**On lag selection.** `maic` is the default for `dfgls` and `ng_perron`: it is
what protects against a negative MA component, and it is Ng & Perron's central
contribution. It has a measurable cost, though — its penalty is large exactly
when a series looks stationary, so it over-selects on I(0) data. The screening
functions `report` and `integration_order` therefore default to `bic`, which
classifies clean data better. Over 40 replications of length 250:

| criterion | I(0) correct | I(1) correct | I(2) flagged |
|---|---|---|---|
| BIC | 40/40 | 40/40 | 35/40 |
| MAIC | 29/40 | 32/40 | 37/40 |

### ARDL estimation

`pyardl.core.ardl`

```python
ARDL(y, x, order=(p, q), det="const", seasonal=False, seasonal_periods=4,
     fixed_regressors=None, hold_back=None).fit(cov_type="nonrobust")
```

| Argument | Meaning |
|---|---|
| `order` | `(p, q)` with `q` an int or a dict `{name: q_j}` |
| `det` | `"none"`, `"const"`, `"trend"` (which includes the intercept) |
| `seasonal` | adds `s-1` seasonal dummies (`s` when `det="none"`) |
| `fixed_regressors` | variables entered without lags, e.g. dummies |
| `hold_back` | initial observations excluded, to force a common sample |
| `cov_type` | `"nonrobust"`, `"HC0"`–`"HC3"`, `"HAC"` |

`q_j = 0` is supported: the regressor enters contemporaneously with no dynamics
of its own. `statsmodels` rejects this case; `pyardl` and Stata's `ardl` both
accept it.

Every fit runs a Ljung-Box test and warns when it rejects. Valid long-run
inference requires enough lags to whiten the errors, so this is a condition of
validity, not an optional diagnostic.

**Results.** `params`, `bse`, `tvalues`, `pvalues`, `resid`, `fittedvalues`,
`aic`/`bic`/`hqic`, `rsquared`, plus the error-correction views: `to_ecm()`,
`longrun`, `adjustment`, `ar_roots`, `is_stable`, `diagnostics()`,
`stability()`, `test_longrun_restriction()`, `summary()`.

**Order selection.** `ARDL.select_order(...)` searches by grid or per-variable.
`ARDL.gets(...)` performs a general-to-specific reduction over terminal lags,
guarded by residual diagnostics and an F test, and returns the full
`reduction_path` so the reduction is auditable rather than a black box.

### Bounds test

```python
pyardl.bounds.bounds_test(
    y, x, case=3, order=None, ic="aic", max_p=4, max_q=4,
    alpha=0.05, cv_source="kripfganz", finite_t=False, fixed_regressors=None,
    conditional=True,
)
```

The five deterministic cases of PSS:

| `case` | Intercept | Trend | Use |
|---|---|---|---|
| 1 | none | none | demeaned data only |
| 2 | restricted | none | no trend anywhere |
| 3 | unrestricted | none | **the usual choice** |
| 4 | unrestricted | restricted | trending data, no trend in the relation |
| 5 | unrestricted | unrestricted | trending data and relation |

Under cases 2 and 4 the restricted deterministic term is part of the tested
vector, giving `k+2` restrictions instead of `k+1`.

**Results.** `f_stat`, `t_stat`, `f_indep_stat`, `decision_f`, `decision_t`,
`decision_indep`, `classification()`, `decision_joint` (the older two-test
verdict, kept for continuity), `bounds`, `p_values`, `uecm`,
`adjustment(alpha)`, `stability(alpha)`, `diagnostics(alpha)`, `conditional`,
`summary()`.

`diagnostics()` reports residual tests *and* both stability tests:

```text
                    statistic  pvalue
Ljung-Box(10)         12.2814  0.2667
Jarque-Bera           85.2392  0.0000
Breusch-Pagan             NaN  0.9731
CUSUM(5%) excess       0.0000     NaN
CUSUMSQ(5%) excess     0.0000     NaN
```

The stability rows carry no p-value, and the column is `NaN` rather than a
plausible-looking number: they are boundary-crossing procedures, not statistics
with a null distribution to integrate.

**Assumptions.** The test is valid if the regressors are weakly exogenous, are
not cointegrated among themselves, no series is I(2), and the residuals are not
autocorrelated. Only the last is checked automatically — hence step 0.

### Critical values

`pyardl.critical_values`

Because the limiting distribution depends on the unknown integration order of
the regressors, critical values come in pairs: a lower bound assuming all
regressors are I(0), an upper bound assuming all are I(1).

| `cv_source` | Use for | Coverage |
|---|---|---|
| `"kripfganz"` | everyday work — the default | cases 1–5, `k = 1..10`, F, any level, with p-values |
| `"pss"` | reproducing published results exactly | cases 1–5, `k = 0..10`, F and t, 10/5/2.5/1% |
| `"narayan"` | small samples, `30 ≤ T ≤ 80` | cases 2, 3, 5, `k ≤ 7`, F, 10/5/1% |

Asymptotic bounds over-reject when `T` is between 30 and 80 — where annual data
lands. Using them there produces spurious findings.

Also available: `simulate_bounds(...)`, a reproducible Monte Carlo engine for
configurations no table covers, recording seed, replications and batch size on
the result so a run reproduces exactly; and `bde1975`, `ers1996`,
`ngperron2001`, `mackinnon` for the other tests' bounds.

### Long-run restrictions

```python
ARDLResults.test_longrun_restriction(R, r, impose=False)
```

Wald test of `R θ = r` on the long-run coefficients, using the same delta-method
covariance as the standard errors in `.longrun`, so the two cannot disagree.
The discrepancy `R θ − r` is returned **signed**.

With `impose=True` the error-correction model is re-estimated with `θ_j = 1`
applied — the level term becomes the ratio `(y − x_j)` — and a regression F test
compares the two residual sums of squares. That comparison is legitimate only
because the unrestricted error-correction design reproduces the ARDL regression
exactly, residuals identical to 1e-10; a test verifies it across lag orders and
deterministic cases rather than assuming it.

The verdict is `not_rejected`, never `accept`.

### Stability diagnostics

`pyardl.diagnostics`

| Function | Detects |
|---|---|
| `cusum(y, x, alpha)` | a shift in the **mean** of the coefficients |
| `cusumsq(y, x, alpha)` | a change in **variance** |
| `stability_tests(y, x, alpha)` | both, in one table |
| `recursive_residuals(y, x)` | the standardised one-step-ahead prediction errors |
| `plot_cusum`, `plot_cusumsq` | the two canonical graphs, bands included |

**The two are not interchangeable.** A break in the slope on a zero-mean
regressor leaves the recursive residuals centred on zero: the CUSUM path stays
flat and reports stability, however large the break, while the inflated variance
pushes the CUSUM of squares straight out of its band. On 20 simulated samples
with exactly that break, the CUSUM said "stable" 20 times out of 20 and the
CUSUM of squares detected it 20 times out of 20.

Reporting only the CUSUM — as much applied work does — leaves an entire family
of common instabilities untested. `pyardl` always produces both.

Results carry `stable`, `max_excess` (how far from stability, not merely
whether) and `crossings` (when the break happened).

### Bootstrap bounds test

```python
pyardl.bootstrap.bootstrap_bounds_test(
    y, x, case=3, order=None, n_boot=2999, resample="iid", seed=None,
    var_order=1, burn_in=50, store_distribution=False, conditional=True,
)
```

The verdict is **binary**: no inconclusive zone. The p-value is
`(1 + #)/(B + 1)` and never exactly zero — `B` replications cannot resolve more
than `1/(B+1)`. A replication that cannot be estimated is counted and reported,
never replaced by a fresh draw, which would bias the distribution towards
estimable samples.

Same seed, same critical values, bit for bit. When no seed is given, one is
drawn from entropy and **recorded**, so any run can be reproduced after the
fact.

All three statistics are drawn under the **same joint null**. That is a
measured choice, not a reading: giving each test its own weaker null inflates
size to 9.3% at a nominal 5% for the `t`, and to 8.5% for `F_indep`. See OBS-8
and the deviation note in `DEVIATIONS`.

**Results.** `f_stat`, `t_stat`, `f_indep_stat`, the matching `*_critical` and
`*_pvalue`, `classification(alpha)`, `comparison(alpha)`,
`agrees_with_bounds(alpha)`, `classical`, `distribution`, `summary()`.

Building blocks are exposed, because a bootstrap you cannot inspect is a
bootstrap you cannot debug: `estimate_null_dgp`, `simulate_paths`,
`simulate_path`, `resample_residuals`.

**Cost.** 0.19 to 1.81 s for a full test at `B = 2999`, depending on the
specification. Both hot paths are vectorised across replications and the `B`
fits are solved by one stacked QR — never the normal equations, which would
square the condition number of a design built on lagged levels of integrated
series.

### Conditional and unconditional models

`conditional=False`, on `bounds_test` and `bootstrap_bounds_test` alike, drops
the contemporaneous differences of the regressors and changes nothing else —
the distinction of Bertelli, Vacca and Zoia (2022). The tested vector is
untouched, so the two forms test the same restriction on two specifications.

The setting is threaded through the observed statistic, the null model, the
regenerated data and each replication. If the null model kept `Δx_t` while the
statistic did not, the simulated null would not be the null being tested — and
nothing in the output would say so.

The convention was measured against `bootCT`, which reports its own
unconditional statistic: of two candidate specifications, only one reproduces
it, to 1e-12.

### Johansen test

```python
pyardl.cointegration.johansen(y, det_order=0, k_ar_diff=1, alpha=0.05, method="trace")
```

```text
Johansen test (1988, 1991) - 4 variables (LRM, LRY, IBO, IDE), det_order=0, k_ar_diff=1

        H0       trace     cv 5%      maxeig     cv 5%
     r = 0     48.8037   47.8545     31.5136   27.5858
    r <= 1     17.2902   29.7961     10.1453   21.1314
    r <= 2      7.1449   15.4943      6.5889   14.2639
    r <= 3      0.5560    3.8415      0.5560    3.8415

selected rank (trace, 5%): 1
```

A thin wrapper over `statsmodels`, plus what it leaves to the caller: the
**sequential decision** (stop at the *first* non-rejection — continuing past it
is a different procedure with a different size), the result object, and
normalised cointegrating vectors.

`check_no_cointegration_among_x(x, ...)` checks the assumption the bounds test
makes and never reveals on its own: that the regressors are not cointegrated
among themselves. It warns, naming the number of relations found.

Measured (OBS-10): the trace statistic **over-selects** the rank — 87.8%
correct against `maxeig`'s 92.5% on a rank-1 DGP — and never under-selects.
`trace` remains the default because it is what the applied literature reports;
a borderline rank deserves a second reading by `maxeig`, and both are always
computed.

Deterministic conventions differ across implementations and the naming is a
trap: `urca`'s `ecdet="none"` matches `det_order=0`, **not** `det_order=-1`.
The correspondence was established by running both sides, not by reading either
manual — see [`docs/api/johansen.md`](docs/api/johansen.md).

### VECM simulator

```python
pyardl.simulate.vecm_ardl(n_obs, alpha, beta, gammas=(), case=3, sigma=None, ...)
pyardl.simulate.degenerate_system(kind, k=1, speed=-0.4)
```

One generator for every Monte Carlo study in the library, so a disagreement
between two validation studies is a disagreement about estimators rather than
about data. Writing `Π = α β'` makes the rank *chosen* rather than hoped for,
and the reported `rank` is the rank of `Π` — a zero `alpha` creates no
relation, and saying otherwise would claim one the data do not contain.

`degenerate_system` builds the canonical systems the three-test framework has
to tell apart. Stability is deliberately **not** enforced: an explosive system
is a legitimate thing to simulate.

### Engle-Granger

```python
pyardl.cointegration.engle_granger(y, x, trend="c", max_lags=None,
                                   ic="aic", fit_ecm=False)
```

Provided for comparison, and because much of the literature reports it. Three
limitations are structural:

- **The normalisation is arbitrary.** Regressing `y` on `x` and `x` on `y` are
  different tests and can disagree; the test suite demonstrates it.
- **Only one relationship can be found**, with no warning that others exist.
- **Every series must be I(1)** — as the opening example shows, that is often
  false, and the test fails silently rather than complaining.

First-step coefficients are reported without standard errors: they are
super-consistent but non-standard, so the usual ones would be wrong.

### ARDL ↔ ECM algebra

`pyardl.core.transforms`

`ardl_to_ecm` and `ecm_to_ardl` are exact inverses: fit either representation on
the same data and you get identical residuals. Also `longrun_coefs`,
`longrun_covariance` (delta method, analytical gradient), `speed_of_adjustment`
and `half_life`. Degenerate configurations return `NaN` with a warning rather
than a number produced by dividing by something near zero.

### Utilities and datasets

```python
from pyardl.utils import diff, lag_matrix, check_series
from pyardl.datasets import load_denmark, load_pss2001
```

`diff(x, d=1, D=0, s=4)` applies `(1-L)^d (1-L^s)^D`. A `Series` keeps the tail
of its index, so a differenced series stays attached to its dates instead of
silently shifting by `d + D·s` periods.

`load_denmark()` — Danish money demand, quarterly.
`load_pss2001()` — the UK wage-price data of Pesaran, Shin & Smith (2001).

---

## Validation

This is the part worth reading before trusting any number.

**Against reference implementations.** Coefficients, standard errors and
residuals agree with `statsmodels` to 1e-10, and with the R package `ARDL` to
1e-6. The UK wage equation of PSS (2001) is reproduced to 1e-4 on the F and t
statistics. DF-GLS agrees with `arch` to 1e-8 across sample sizes, trends and
lag orders. Engle-Granger agrees with `statsmodels.tsa.stattools.coint` to
1e-13.

**Critical values.** Every shipped table documents its source, its transcription
channel and its cross-check in
[`PROVENANCE.md`](src/pyardl/critical_values/PROVENANCE.md). Where a second
published source exists it is used; where none does, the table is generated by
an in-house Monte Carlo engine with recorded seeds and verified against a
theoretical limit. The comparison criterion is derived from the Monte Carlo
standard error of each quantile — published tables carry their own simulation
error, so a flat tolerance is not defensible.

**What the protocol surfaced.** A typo in a published R package's transcription
(`11.60` for `1.60`), three cells where independent sources disagree with a
printed table, a response surface that is conservative at the edge of its fitted
range, and two rounding conventions where reference implementations differ from
the published rule. All documented rather than smoothed over.

**Conventions settled by measurement, not by reading.** Three times, a
specification admitted two readings and the choice was made by measuring both:
which null the bootstrap draws from (a per-test null inflates size to 9.3% at a
nominal 5%), what the unconditional model actually removes (only one of two
candidate specifications reproduces `bootCT`'s own statistic, to 1e-12), and
which Johansen statistic meets the criterion the specification itself sets.
Each is recorded in the project's validation register with the numbers that
decided it — including one hypothesis that the data refuted, kept in the record
with its full trajectory rather than quietly replaced by the conclusion.

**Limits, recorded rather than smoothed over.** `F_indep` is oversized at
`T = 100` — 6.5% at a nominal 5%, where the `t` holds its size. The bootstrap's
decisiveness costs accuracy under a type 2 degeneracy. The bounds of `F_indep`
are simulated in-house because the published ones are behind an access barrier,
so their cross-checks are structural rather than external, and that is weaker.
None of this is hidden in a footnote: it is OBS-9, OBS-11 and OBS-12 of that
same register, and the summary of OBS-12 is a page of the documentation in its
own right — [Bootstrap or classical bounds?](docs/bootstrap-or-bounds.md).

**Test suite.** 707 tests plus 38 doctests, `mypy --strict` clean, on Linux,
Windows and macOS across Python 3.11–3.13. Monte Carlo experiments run nightly
at full replication counts.

---

## Compatibility

| | |
|---|---|
| Python | 3.11, 3.12, 3.13 |
| OS | Linux, Windows, macOS |
| Required | numpy, scipy, pandas, statsmodels |
| Optional | matplotlib (`plot`), arch (`bootstrap`) |

Tested against numpy 2.5 and pandas 3.0.

---

## Roadmap

Released:

- **0.1.0** — ARDL/UECM estimation, bounds test with the five deterministic
  cases, joint F and t decision, PSS critical values.
- **0.2.0** — small-sample and response-surface critical values with p-values,
  CUSUM/CUSUMSQ stability, DF-GLS and Ng-Perron pre-tests, long-run restriction
  testing and seasonality, Engle-Granger.
- **0.3.0** — bootstrap ARDL with no inconclusive zone, the three-test
  framework that names both degeneracies, the Johansen system test and its
  regressor diagnostic, conditional/unconditional models, and one VECM
  simulator for every Monte Carlo study.

Planned:

- **0.4** — NARDL (asymmetric).
- **0.5+** — Fourier ARDL, dynamic simulations, QARDL, heterogeneous panels
  (MG, PMG, CS-ARDL).

---

## Contributing

Issues and pull requests are welcome. Before opening a PR:

```bash
pip install -e ".[dev,plot,bootstrap]"
ruff check src tests && ruff format --check src tests
mypy src/pyardl
pytest -m "not slow" --doctest-modules src/pyardl tests --cov=pyardl
```

Two expectations specific to this project. Every statistical claim needs a test
that would fail if the claim were false — not a smoke test. And **no numerical
value is ever written from memory**: critical values, docstring examples,
doctest expectations and figures quoted in documentation are all computed by a
real run and pasted from it.

---

## Citing

If you use `pyardl` in published work, please cite the software (see
[`CITATION.cff`](CITATION.cff)) as well as the methodological articles behind
the part you used — they are listed in each module's docstring.

---

## References

- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). Bounds testing approaches to
  the analysis of level relationships. *Journal of Applied Econometrics*, 16(3),
  289–326.
- Banerjee, A., Dolado, J. & Mestre, R. (1998). Error-correction mechanism tests
  for cointegration in a single-equation framework. *Journal of Time Series
  Analysis*, 19(3), 267–283.
- Narayan, P. K. (2005). The saving and investment nexus for China. *Applied
  Economics*, 37(17), 1979–1990.
- Kripfganz, S. & Schneider, D. C. (2020). Response surface regressions for
  critical value bounds and approximate p-values in equilibrium correction
  models. *Oxford Bulletin of Economics and Statistics*, 82(6), 1456–1481.
- Brown, R. L., Durbin, J. & Evans, J. M. (1975). Techniques for testing the
  constancy of regression relationships over time. *JRSS B*, 37(2), 149–192.
- Elliott, G., Rothenberg, T. J. & Stock, J. H. (1996). Efficient tests for an
  autoregressive unit root. *Econometrica*, 64(4), 813–836.
- Ng, S. & Perron, P. (2001). Lag length selection and the construction of unit
  root tests with good size and power. *Econometrica*, 69(6), 1519–1554.
- Davidson, J. E. H., Hendry, D. F., Srba, F. & Yeo, S. (1978). Econometric
  modelling of the aggregate time-series relationship between consumers'
  expenditure and income in the United Kingdom. *The Economic Journal*, 88(352),
  661–692.
- Engle, R. F. & Granger, C. W. J. (1987). Co-integration and error correction.
  *Econometrica*, 55(2), 251–276.
- MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's
  University Working Paper 1227.
- Hendry, D. F., Pagan, A. R. & Sargan, J. D. (1984). Dynamic specification.
  *Handbook of Econometrics*, vol. 2.

---

<div align="center">

MIT licensed · [Documentation](docs/) · [Changelog](CHANGELOG.md)

</div>
