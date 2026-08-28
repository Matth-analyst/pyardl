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
    LRY             1.2214      0.0841    14.522    0.0000
    IBO            -3.8353      0.3265   -11.747    0.0000
    IDE             2.6308      0.7593     3.465    0.0005

```

**FMOLS** (Phillips & Hansen) leaves the regression alone and corrects
the data and the estimator: `y` is purged of the part the regressor
innovations explain in the long run, and an explicit bias term is
subtracted.

```pycon
>>> from pyardl.cointegration import fmols
>>> res = fmols(d["LRM"], d[["LRY", "IBO", "IDE"]], bandwidth=5)
>>> round(float(res.longrun.loc["LRY", "theta"]), 6)
1.290357

```

**CCR** (Park) transforms `y` and `x` so that ordinary least squares on
the transformed data is already efficient. Asymptotically equivalent to
FMOLS; in finite samples they differ, which is why reporting both is
informative.

## What they buy, measured

Same 1000 replications, `theta = 1.5`:

| T | OLS | DOLS | FMOLS | CCR |
|---|---|---|---|---|
| 100 | +0.1380 | **+0.0087** | +0.0565 | +0.0611 |
| 200 | +0.0773 | **+0.0013** | +0.0232 | +0.0261 |
| 400 | +0.0397 | **+0.0013** | +0.0108 | +0.0118 |

And coverage of a nominal 95% interval:

| T | DOLS | FMOLS | CCR |
|---|---|---|---|
| 100 | 83.7% | 80.8% | 79.0% |
| 200 | 88.1% | 85.9% | 84.0% |
| 400 | **90.4%** | 88.7% | 87.6% |

Two things to take from this, and the second is a caveat the
specification did not anticipate.

**DOLS dominates on this DGP.** Its bias is 3.3% of the OLS bias at
T = 400, where FMOLS and CCR sit around 27–30%. The specification asked
for under 10%; only DOLS delivers it.

**None of the three reaches nominal coverage at these sample sizes.**
The specification expected 92–97%; the best is 90.4%, and the shortfall
shrinks steadily with `T`. The intervals are still a little too narrow
in finite samples. That is a property of the estimators, not of this
implementation — the coefficients agree with the `cointReg` reference to
1e-11 — and reporting it is more useful than the reassurance the
specification assumed. Recorded as OBS-24.

The bandwidth was fixed at 8 throughout. Tuning it until the coverage
band was reached would have been choosing the result.

## The comparison table

The robustness block applied papers report, in one call:

```pycon
>>> from pyardl.cointegration import compare_longrun
>>> from pyardl.core.ardl import ARDL
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
DOLS   LRY        1.2214  0.0841  14.5217
       IBO       -3.8353  0.3265 -11.7468
       IDE        2.6308  0.7593   3.4646
FMOLS  LRY        1.2904  0.1228  10.5051
       IBO       -3.0068  0.4281  -7.0237
       IDE        0.9702  0.8996   1.0785
CCR    LRY        1.2877  0.1260  10.2204
       IBO       -3.0210  0.4441  -6.8029
       IDE        1.0150  0.9597   1.0576
```

On Danish money demand the four do **not** agree, and the disagreement
is the result. The income elasticity runs from 1.00 (ARDL) to 1.29
(FMOLS); the bond-rate semi-elasticity from −3.01 to −4.54. Most
strikingly, the deposit rate is significant under ARDL and DOLS
(`t` = 2.9 and 3.5) and **not** under FMOLS and CCR (`t` ≈ 1.06).

A paper reporting only one of these rows would be reporting a choice of
estimator as though it were a finding. That is the case for printing the
table.

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
