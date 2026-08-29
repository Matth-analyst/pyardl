# Efficient long-run estimators — DOLS, FMOLS, CCR

`pyardl.cointegration`

Static OLS on cointegrated variables is consistent. Engle and Granger's
superconsistency makes its bias vanish at rate `1/T` rather than
`1/sqrt(T)`, and a great deal of applied work stops there.

**Its inference is not consistent, and that is a different problem.**
The regressors are correlated with the equation error, the error is
serially correlated, and the two interact to leave a second-order bias.
The `t` statistics have a non-standard distribution, so reading them
against a Student table overstates significance — sometimes by a lot.

Measured on 1000 replications of a cointegrated DGP with both
endogeneity and autocorrelation, a nominal 95% interval from static OLS
covers:

| T | OLS, naive se | OLS, HAC se |
|---|---|---|
| 100 | 42.1% | 71.1% |
| 200 | 38.4% | 71.8% |
| 400 | 41.4% | 71.8% |

**Forty-two percent.** Not conservative — wrong. And the second column
is the part worth pausing on: fixing the standard error with a HAC
correction gets you to 71% and stops there, at every sample size. The
variance was never the only problem. The second-order bias survives any
amount of standard-error repair, and removing it is what these three
estimators are for.

## The three routes

**DOLS** (Stock & Watson) adds leads *and* lags of `Δx` to the static
regression. The leads are the point — they absorb the feedback running
from the equation error to future regressor changes, which *is* the
endogeneity. What is left is serial correlation, handled by a HAC
standard error. Conceptually simplest, and on the evidence below the one
to reach for first.

```pycon
>>> from pyardl.cointegration import dols
>>> from pyardl.datasets import load_denmark
>>> d = load_denmark()
>>> res = dols(d["LRM"], d[["LRY", "IBO", "IDE"]],
...            n_leads=2, n_lags=2, bandwidth=5)
>>> print(res.summary())
DOLS long-run estimates - 50 observations
  2 lead(s) and 2 lag(s) of the differenced regressors; HAC standard errors (bartlett, bandwidth 5)
  t statistics are asymptotically standard normal
<BLANKLINE>
                     theta          se         z         p
    LRY             1.2214      0.0878    13.919    0.0000
    IBO            -3.8353      0.3406   -11.259    0.0000
    IDE             2.6308      0.7922     3.321    0.0009

```

**FMOLS** (Phillips & Hansen) leaves the regression alone and corrects
the data and the estimator: `y` is purged of the part the regressor
innovations explain in the long run, and an explicit bias term is
subtracted.

```pycon
>>> from pyardl.cointegration import fmols
>>> res = fmols(d["LRM"], d[["LRY", "IBO", "IDE"]], bandwidth=5)
>>> round(float(res.longrun.loc["LRY", "theta"]), 6)
1.26569

```

**CCR** (Park) transforms `y` and `x` so that ordinary least squares on
the transformed data is already efficient. Asymptotically equivalent to
FMOLS; in finite samples they differ, which is why reporting both is
informative.

## What they buy, measured

Same 1000 replications, `theta = 1.5`:

| T | OLS | DOLS | FMOLS | CCR |
|---|---|---|---|---|
| 100 | +0.1380 | **+0.0087** | +0.0358 | +0.0321 |
| 200 | +0.0773 | **+0.0013** | +0.0094 | +0.0085 |
| 400 | +0.0397 | **+0.0013** | +0.0038 | +0.0036 |

And coverage of a nominal 95% interval:

| T | DOLS | FMOLS | CCR |
|---|---|---|---|
| 100 | 88.8% | 88.9% | 87.4% |
| 200 | 91.5% | **92.2%** | 91.1% |
| 400 | **93.4%** | **94.5%** | **93.7%** |

At T = 400 all three meet **both** of the specification's criteria: the
92–97% coverage band, and bias under 10% of the OLS bias (3.3%, 9.6%,
9.1%). DOLS is the most reliable across the range — its bias is
essentially gone by T = 200.

**Prewhitening is why those numbers look like that, and it is on by
default.** Without it the same study gave 90.4% / 88.7% / 87.6%
coverage and missed both criteria. The first version of this page
reported that shortfall as a property of the estimators at modest `T`.
It was not: it was a missing component. See below.

The bandwidth was fixed at 8 throughout. Tuning it until the coverage
band was reached would have been choosing the result — the prewhitening
was found by asking what was *absent*, not by tuning what was present.

## The comparison table

The robustness block applied papers report, in one call:

```pycon
>>> from pyardl.cointegration import compare_longrun
>>> from pyardl.core import ARDL
>>> fit = ARDL(d["LRM"], d[["LRY", "IBO", "IDE"]],
...            order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}), det="const").fit()
>>> table = compare_longrun(d["LRM"], d[["LRY", "IBO", "IDE"]],
...                         ardl_results=fit, bandwidth=5,
...                         n_leads=2, n_lags=2)
>>> sorted(set(table.index.get_level_values("method")))
['ARDL', 'CCR', 'DOLS', 'FMOLS']

```

```text
                   theta      se        t
method regressor
ARDL   LRY        0.9965  0.1239   8.0405
       IBO       -4.5381  0.5203  -8.7222
       IDE        2.8915  0.9951   2.9058
DOLS   LRY        1.2214  0.0878  13.9187
       IBO       -3.8353  0.3406 -11.2590
       IDE        2.6308  0.7922   3.3207
FMOLS  LRY        1.2657  0.1326   9.5474
       IBO       -3.8893  0.4620  -8.4179
       IDE        4.0343  0.9709   4.1552
CCR    LRY        1.1498  0.1179   9.7500
       IBO       -4.2805  0.3657 -11.7063
       IDE        4.0856  0.8628   4.7353
```

The four agree on every **sign** and on every **verdict** — all three
regressors are significant under all four estimators — and they disagree
substantially on **magnitude**. The income elasticity runs from 1.00
(ARDL) to 1.27 (FMOLS); the deposit-rate coefficient from 2.63 to 4.09,
a factor of 1.55.

A paper reporting one of these rows as *the* long-run relation is
reporting a choice of estimator as though it were a finding. Printing
the table is what makes the choice visible.

**This table is itself an argument for prewhitening.** Computed without
it, FMOLS and CCR put the deposit-rate `t` at 1.08 and 1.06 — *not*
significant — flatly contradicting ARDL and DOLS. The disagreement was
not economics; it was an underestimated long-run covariance feeding the
bias correction. Turn prewhitening off with `prewhiten=False` and the
old table comes back.

## The long-run covariance, and a transpose that matters

FMOLS and CCR both need the long-run covariance matrices of the
residuals stacked with the differenced regressors.
`pyardl.utils.longrun_covariance_kernel` computes them:

```pycon
>>> import numpy as np
>>> from pyardl.utils import longrun_covariance_kernel
>>> rng = np.random.default_rng(0)
>>> out = longrun_covariance_kernel(rng.normal(size=(500, 2)), bandwidth=6)
>>> out.omega.shape, round(float(out.bandwidth), 1)
((2, 2), 6.0)

```

Four kernels (Bartlett, Parzen, quadratic-spectral, truncated) and two
automatic bandwidth rules (Andrews, Newey-West), all checked against
`cointReg::getLongRunVar` to 5.3e-15.

## Prewhitening, and why it is the default

A kernel estimate of the long-run covariance is biased downward when the
series is persistent — the kernel weights down exactly the
autocovariances that carry the persistence. Andrews and Monahan's remedy
is to strip a VAR(1) first, run the kernel on what is left, and put the
persistence back.

The size of the effect, measured directly on the brick: for an AR(1)
with coefficient 0.8 the true `Omega` is 25.0. The plain kernel estimate
at bandwidth 6 returns **11.68 — 53% too low**. With prewhitening,
26.82.

A long-run variance halved gives standard errors divided by 1.4, and
that is precisely the coverage gap. `prewhiten=True` is therefore the
default on `dols`, `fmols` and `ccr`; `longrun_covariance_kernel` keeps
`False` so the plain estimate stays available and the two can be
compared.

### CCR is a fixed point, not a substitution

Park's transformation of `y` depends on `theta` — the quantity being
estimated. The textbook statement substitutes a first-stage estimate
once, and pyardl originally used static OLS there, carrying that
estimator's finite-sample bias straight into the transformation.

Iterating to convergence takes CCR from 15% of the OLS bias to 9% at
T = 400, and its coverage from 93.0% to 93.7% — the difference between
missing the specification's bias criterion and meeting it. Two checks
that this is the fixed point and not an artefact of the iteration count:
starting from the FMOLS estimate instead of OLS reaches the *same* place
(+0.0034 either way), and convergence is geometric at roughly 1/16 per
step.

`res.n_iter` and `res.converged` record what happened. The count differs
sharply between simulation and real data — median 9 over 200 simulated
samples at T = 200, but **34** on Danish money demand, where T = 55 with
three regressors sits closer to the edge of the contraction. The cap is
200 rather than the 50 the simulation alone would have suggested.

A detail that cost a second bug. Prewhitening recolours the **long-run**
matrices, `Omega` and `Delta`, through `(I-A)^-1 . (I-A')^-1`. It does
**not** apply to `Sigma`, the contemporaneous covariance — that identity
is about the spectrum at frequency zero. Recolouring `Sigma` too made
CCR *worse than plain OLS* while DOLS and FMOLS improved, and CCR is the
only one of the three that reads `Sigma`. One estimator degrading while
the others improve points straight at the quantity it alone uses.

**The one-sided matrix has two conventions in circulation** that differ
by a transpose:

```
Delta = Gamma_0 + sum_j k_j Gamma_j        (naive)
Delta = Gamma_0 + sum_j k_j Gamma_j'       (what FMOLS needs)
```

Both give the **same** `Omega`, since `Omega = Delta + Delta' - Gamma_0`
is invariant to the transposition. So checking `Omega` — the natural
first check — does not separate them. Here `Omega` agreed to 5.6e-16
while `theta` was wrong.

And the two estimators do not use the same one: FMOLS needs the second,
Park's CCR needs the first. Nothing in either formula flags it; they
come from papers with different notation. What exposed it was the
**asymptotic equivalence** of the two estimators used as a detector —
with the wrong convention CCR removed almost none of the OLS bias
(+0.0370 against +0.0384 for plain OLS at T = 400) where FMOLS removed
most of it. Two estimators that must coincide and did not. OBS-24.

## Cross-checked against R

| quantity | agreement |
|---|---|
| long-run covariance, four kernels | 5.3e-15 |
| FMOLS `theta` | 2.0e-11 |
| FMOLS standard errors | 8.6e-13 |
| DOLS `theta` | 7.2e-14 |
| DOLS standard errors | 2.3e-12 |

Reference: `cointReg` 0.2.0 (GPL-3), on the Danish data pyardl already
ships. **CCR has no external reference** — `cointReg` does not implement
it — so it rests on convergence to the true coefficient as `T` grows and
on agreement with FMOLS, both of which are in the test suite.

Two conventions were pinned by measurement rather than assumed: the
transpose above, and the fact that the **bias correction scales with the
full sample size** `T`, not with the `T-1` rows the regression uses.
Solving for the `lambda+` that cointReg's published coefficients imply
gave a ratio of exactly 55/54 on all three coefficients — that constancy
is what identified the convention rather than a search.

## References

- Phillips, P. C. B. & Hansen, B. E. (1990). Statistical inference in
  instrumental variables regression with I(1) processes. *Review of
  Economic Studies*, 57(1), 99-125.
- Stock, J. H. & Watson, M. W. (1993). A simple estimator of
  cointegrating vectors in higher order integrated systems.
  *Econometrica*, 61(4), 783-820.
- Park, J. Y. (1992). Canonical cointegrating regressions.
  *Econometrica*, 60(1), 119-143.
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation
  consistent covariance matrix estimation. *Econometrica*, 59(3),
  817-858.
