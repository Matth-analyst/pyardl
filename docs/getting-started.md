# Getting started

This walkthrough goes from a dataset to a defensible conclusion about a
long-run relationship. It uses the Danish money-demand data shipped with
the package.

## The question

You have a set of macroeconomic series and you suspect they move
together in the long run. The classical cointegration tests ask you to
establish the integration order of every series first — and they are not
valid if you get it wrong, or if the series are of mixed order.

The bounds test avoids that. It works whether the regressors are I(0),
I(1), or a mixture, at the cost of an answer that is sometimes
inconclusive.

## 1. Load the data

```python
>>> from pyardl.datasets import load_denmark
>>> data = load_denmark()
>>> y = data["LRM"]                        # log real money (M2)
>>> x = data[["LRY", "IBO", "IDE"]]        # income, bond rate, deposit rate

```

Your own data works the same way: a pandas `Series` for the dependent
variable and a `DataFrame` for the regressors. Column names carry
through to every output, so name them meaningfully.

## 2. Choose the lag orders

Lag orders matter more than they look. Too few lags leave autocorrelated
residuals, which invalidates the inference; too many waste degrees of
freedom.

```python
from pyardl.core import ARDL

sel = ARDL.select_order(y, x, max_p=4, max_q=4, ic="bic")
print(sel.best_order)
print(sel.top(5))
```

Two things worth knowing here. First, every candidate is estimated on
the *same* sample, so the information criteria are comparable — comparing
criteria computed on different numbers of observations is a common and
silent mistake. Second, look at `top(5)` and not only at the winner:
criteria often separate the leading specifications by very little, and
checking that your conclusions survive a nearby specification is cheap
insurance.

## 3. Pick the deterministic case

The bounds test comes in five variants, depending on how the intercept
and trend enter:

| Case | Intercept | Trend | When |
|---|---|---|---|
| 1 | none | none | rare; only for demeaned data |
| 2 | restricted | none | no trend in the data or the relationship |
| 3 | unrestricted | none | **the usual choice** |
| 4 | unrestricted | restricted | trending data, no trend in the long-run relation |
| 5 | unrestricted | unrestricted | trending data and relationship |

Case 3 is the default choice in most applied work. If your series
clearly trend, compare cases 4 and 5, and check whether the trend term
is significant.

## 4. Run the test

```python
>>> from pyardl.bounds import bounds_test
>>> res = bounds_test(y, x, case=3, order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}))
>>> print(res.summary())
Bounds test (Pesaran, Shin & Smith 2001) - case 3, k=3, ECM(3; LRY:1, IBO:3, IDE:2), critical values: kripfganz
<BLANKLINE>
F_overall = 6.2059   decision (5%): cointegration
F p-values: p_I0 = 0.0005, p_I1 = 0.0039
t_BDM     = -4.5479   decision (5%): cointegration
F_indep   = 8.1619   decision (5%): cointegration
<BLANKLINE>
CLASSIFICATION (5%): cointegration
  F_overall, t_BDM and F_indep all reject: the level terms are jointly significant, y adjusts back towards equilibrium, and the regressors carry the long-run relationship.
<BLANKLINE>
        F_I0   F_I1   t_I0   t_I1  F_indep_I0  F_indep_I1
alpha                                                    
0.10   2.730  3.747 -2.570 -3.460       2.084       3.864
0.05   3.229  4.322 -2.860 -3.780       2.619       4.646
0.01   4.311  5.543 -3.430 -4.370       3.814       6.311

```

If you omit `order`, the lag orders are selected automatically.

## 5. Read the result

Each statistic is compared with **two** critical values, not one:
the `_I0` bound assumes all regressors are I(0), the `_I1` bound
assumes all are I(1). The truth lies between, so for each of the three
statistics:

- beyond the upper bound → reject;
- below the lower bound → do not reject;
- in between → **inconclusive** for that statistic.

Rejecting `F_overall` alone is not enough to claim cointegration: it
only says the level terms are not *all* jointly zero, and there are two
distinct ways for that to happen with no genuine long-run relationship
at all. Sam, McNown & Goh (2019) add a third test, `F_indep` (are the
regressors' levels jointly significant on their own?), which — combined
with `t_BDM` (does `y` pull back towards equilibrium?) — tells the two
degeneracies apart:

| classification | `F_overall` | `t_BDM` | `F_indep` | reading |
|---|---|---|---|---|
| `cointegration` | reject | reject | reject | a genuine long-run relationship |
| `degenerate_1` | reject | reject | do not reject | `y` adjusts towards its own past; the regressors carry nothing |
| `degenerate_2` | reject | do not reject | reject | the regressors' levels matter, but `y` never adjusts back |
| `no_cointegration` | do not reject | — | — | no level relationship |
| `inconclusive` | — | — | — | the three tests do not settle the question |

Here `F = 6.21`, `t = -4.55` and `F_indep = 8.16` all reject at 5%, so
`res.classification()` returns `("cointegration", "...")`, with the
reason spelled out in `summary()`'s `CLASSIFICATION` block.

When the answer is inconclusive, the p-value interval tells you how
close you are:

```text
F_overall = 5.4724   decision (5%): inconclusive, p in [0.0594, 0.0313]
```

Here the statistic would be significant at 5% if the regressors were
I(0) (`p = 0.031`) but not if they were I(1) (`p = 0.059`). Since both
are near 5%, the verdict genuinely hinges on the integration order — and
a pre-test on the regressors is the way forward.

## 6. Interpret the relationship

Once cointegration is established, the long-run coefficients become
meaningful:

```python
model = ARDL(y, x, order=(3, {"LRY": 1, "IBO": 3, "IDE": 2})).fit()
print(model.longrun.round(4))
```

```text
      theta      se
LRY  0.9965  0.1239
IBO -4.5381  0.5203
IDE  2.8915  0.9951
```

A one-percent rise in real income raises long-run money demand by
almost exactly one percent — a unit income elasticity, which is what
theory predicts.

How fast does the system return to equilibrium after a shock?

```python
print(res.adjustment().round(4))
```

```text
lambda     -0.4169
se          0.0917
ci_lower   -0.5965
ci_upper   -0.2372
```

About 42% of a disequilibrium is corrected each quarter. The negative
sign is what makes it error *correction*; a positive value would mean
the system diverges.

Note that the confidence interval only appears because cointegration was
established. Ask for it when it has not been, and you get `NaN` plus a
warning: under the null the distribution of this estimator is
non-standard, so the usual interval would be misleading.

## 7. Check the diagnostics

```python
print(res.diagnostics().round(4))
```

Autocorrelated residuals are the main threat here, and they are checked
automatically: if the Ljung-Box test rejects, a warning is issued
whether or not you look at the diagnostics table. The fix is usually
more lags.

For scripted analyses, promote these warnings to errors so a silently
unreliable result cannot slip through:

```python
import warnings
from pyardl.exceptions import PyardlMethodologyWarning

warnings.filterwarnings("error", category=PyardlMethodologyWarning)
```

## Small samples

Asymptotic critical values over-reject when `T` is between roughly 30
and 80 observations — the situation annual data often lands in. Use the
small-sample bounds instead:

```python
res = bounds_test(y, x, case=3, cv_source="narayan")
```

These cover cases 2, 3 and 5, and the F statistic only.

## Reproducing published results

To match numbers printed in a paper, use the published tables exactly as
they appear rather than the more precise default:

```python
res = bounds_test(y, x, case=3, cv_source="pss")
```

## What to check before trusting the test

The bounds test assumes the regressors are weakly exogenous, that they
are not cointegrated among themselves, that no series is I(2), and that
the residuals are not autocorrelated. Only the last one is checked for
you. The I(2) case matters most: the test's distribution theory does not
cover it, so run a unit-root test on any series you are unsure about.
