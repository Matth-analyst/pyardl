# Dynamic simulation — turning coefficients into a trajectory

`pyardl.simulate.dynardl_simulate` · `ARDLResults.dynardl_simulate`

Here is an estimated ARDL, printed the way applied papers print it:

```pycon
>>> from pyardl.core import ARDL
>>> from pyardl.datasets import load_denmark
>>> d = load_denmark()
>>> fit = ARDL(d["LRM"], d[["LRY", "IBO", "IDE"]],
...            order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}), det="const").fit()
>>> fit.params[["LRM.L1", "LRM.L2", "LRM.L3", "IBO.L0", "IBO.L3"]].round(4)
LRM.L1    0.3192
LRM.L2    0.5326
LRM.L3   -0.2687
IBO.L0   -1.0785
IBO.L3   -0.9947
Name: coef, dtype: float64

```

Now answer the question a reader actually has: **if the bond rate rises
by one standard deviation and stays there, what happens to money demand,
and how fast?** The effect is spread across three lags of the dependent
variable and four of the regressor. It is in that table. It is not
*readable* in that table.

```pycon
>>> sim = fit.dynardl_simulate("IBO", size="1sd", t0=5, horizon=40,
...                            r=2000, seed=25)
>>> print(sim.summary())
Dynamic simulation - step shock of 0.0307989 on IBO
  t0 = 5, horizon = 40, 2000 parameter draws, seed = 25
  innovations: off (they cancel out of the response either way)
  baseline equilibrium y* = 11.768926
<BLANKLINE>
  response at h = 40: -0.139771
  95% band: [-0.184923, -0.110393]
  long-run target theta * dx = -0.139769

```

Two numbers are worth pausing on. The response at `h = 40` is
`-0.139771` and the algebraic long-run target is `-0.139769`: the
simulation has arrived where the coefficient table says it should, and
the agreement is a **test**, not a coincidence — it is the first thing
the test suite checks, to 1e-6.

And the trajectory is what the table could not show:

```pycon
>>> sim.summary_df["response"].loc[[0, 5, 6, 10, 20, 40], ["point", "lo_95", "hi_95"]].round(4)
          point   lo_95   hi_95
horizon                        
0        0.0000  0.0000  0.0000
5       -0.0332 -0.0523 -0.0146
6       -0.0471 -0.0690 -0.0260
10      -0.1263 -0.1529 -0.0991
20      -0.1400 -0.1811 -0.1108
40      -0.1398 -0.1849 -0.1104

```

Ninety percent of the long-run effect has landed by `h = 10`, five
periods after the shock — consistent with the half-life of 1.29 the
adjustment table reports, and impossible to read off the coefficients.

## Two blocks, and why the response is the interesting one

`summary_df` carries `response` and `level`. **`response` is a paired
difference**: each draw produces a shocked trajectory and a no-shock
counterfactual, and the reported quantity is their difference, draw by
draw. Everything common to the two branches cancels *exactly* rather
than approximately — the intercept, the deterministic trend, the
seasonal dummies, the level of the other regressors, and the starting
point.

That is why the response before the shock is exactly `0.0000`, not
`1e-14`, and why the band there has zero width. Each draw is started at
**its own** implied equilibrium rather than at `ŷ*(θ̂)`, so the no-shock
branch is flat for every draw instead of drifting in from a starting
point borrowed from a different parameter vector.

## Forecast uncertainty cancels out. Exactly.

`stochastic=True` adds innovations `N(0, σ̂²)` to the recursion. They are
drawn once per draw and used in **both** branches, because an innovation
is a property of the world, not of the intervention.

The model is linear in `y`, so the difference between the branches does
not depend on them at all:

```pycon
>>> import numpy as np
>>> noisy = fit.dynardl_simulate("IBO", size="1sd", t0=5, horizon=40,
...                              r=2000, seed=25, stochastic=True)
>>> gap = float(np.max(np.abs(noisy.summary_df["response"].to_numpy()
...                           - sim.summary_df["response"].to_numpy())))
>>> gap < 1e-13
True

```

The identity is exact in exact arithmetic; what is left — `1.4e-14`
here — is the rounding of the two summations, not a Monte Carlo
discrepancy that would shrink with more draws. What forecast
uncertainty *does* widen is the band on the level, which is the honest
place for it to show up.

## Impulse, and the return to equilibrium

```pycon
>>> imp = fit.dynardl_simulate("IBO", shock_type="impulse", size="1sd",
...                            t0=5, horizon=40, r=2000, seed=25)
>>> imp.summary_df[("response", "point")].loc[[5, 6, 10, 20, 40]].round(4)
horizon
5    -0.0332
6    -0.0139
10   -0.0187
20   -0.0004
40   -0.0000
Name: (response, point), dtype: float64

```

A temporary rise leaves no trace: `longrun_target` is `nan` for an
impulse, because there is nothing for the response to converge to but
zero.

## `mean` is not `point`, and the gap is informative

Each block reports both the trajectory at `θ̂` and the mean across
draws. They differ — `-0.1398` against `-0.1424` at `h = 40` above —
because the recursion is **non-linear in the autoregressive
coefficients**. The long run is a ratio, `Σβ / (1 − Σφ)`, and the mean
of a ratio is not the ratio of the means.

The gap widens with the horizon and with the persistence of the model.
At long horizons the mean is the *worse* summary of the two: a small
share of draws are explosive, and an explosive path dominates an
average. On the Danish fit above, 2.3% of draws have an autoregressive
root outside the unit circle at `r = 20000` — enough to pull the mean
visibly away from both the point path and the median while the quantile
bands stay put. Read `point`, use `mean` to notice when the two
disagree.

## Do the bands cover?

A band produced by pushing parameter draws through a non-linear
recursion is easy to compute and hard to justify. It was measured
(`validation/spec25_montecarlo.py`, 1000 replications of an ARDL(1,1)
with known coefficients, so the true response is known exactly):

| h | true response | 75% | 90% | 95% |
|---|---|---|---|---|
| 5 (impact) | 0.800000 | 75.7% | 89.3% | 93.7% |
| 6 | 0.980000 | 72.6% | 89.2% | 94.8% |
| 10 | 1.215008 | 73.3% | 88.7% | 94.3% |
| 60 (long run) | 1.250000 | 73.5% | 89.4% | 95.0% |

The Monte Carlo standard error on a rate near 95% is 0.69 point at this
size, fixed before the study rather than after it. The 95% band holds
across the whole trajectory, including at the long-run horizon where the
quantity is a ratio and asymmetry could have bitten. The 75% band runs
about two points low at intermediate horizons — reported as measured.

## NARDL: shocking one branch

A NARDL is a linear ARDL in the decomposed regressors, so this applies
to it unchanged — and that is exactly what makes it interesting here.
Shocking `x_pos` alone *is* the counterfactual "x rises and never
falls", which is the asymmetry the model was fitted to represent.

```python
sim_up = nardl_res.dynardl_simulate("x_pos", size=1.0, t0=0)
sim_down = nardl_res.dynardl_simulate("x_neg", size=1.0, t0=0)
```

A step of size one on `x_pos` reproduces the `m_pos` column of
`dynamic_multipliers` exactly — the test suite checks the two to 1e-10.
Two routes to the same object that disagreed would mean one of them was
wrong.

## Bootstrap draws instead of the normal

`param_draws` accepts a `(r, n_params)` array, so bootstrap replications
from specs 14/16 can drive the figure instead of `N(θ̂, V̂)`. A paper
whose bounds test rests on a bootstrap and whose figure rests on
asymptotic normality is quietly using two notions of uncertainty; this
is the hook that lets it use one.

## Cross-checked against R

`dynamac` 0.1.12 (GPL-2), on the Danish data pyardl already ships — no
third-party dataset involved.

| quantity | agreement |
|---|---|
| the 13 regression coefficients | 3.5e-14 |
| baseline `ŷ*` | 5.8e-05 |
| central path at `h = 61` | within dynamac's own seed-to-seed spread |

The coefficient check is the exact one: `dynardl(..., ec = FALSE)`
estimates the same regression, so agreement to machine precision is
required rather than hoped for. The trajectory check is not exact,
because dynamac produces its central path by Monte Carlo; the script
runs three seeds to *measure* that noise (0.005 at `h = 61`) instead of
assuming it, and pyardl's exact limit falls inside it.

**The bands are not comparable term by term, and the test does not
pretend they are.** dynamac holds the pre-shock level nearly fixed — its
95% band there is 0.012 wide — while pyardl starts each draw at its own
equilibrium, so its band on the *level* also carries the sampling
dispersion of `ŷ*`. Two different questions about the level, neither one
wrong. The band on the **response** is a paired difference and is
untouched by the difference.

## References

- Jordan, S. & Philips, A. Q. (2018). Cointegration testing and dynamic
  simulations of autoregressive distributed lag models. *The Stata
  Journal*, 18(4), 902-923.
- Philips, A. Q. (2018). Have your cake and eat it too? Cointegration
  and dynamic inference from autoregressive distributed lag models.
  *American Journal of Political Science*, 62(1), 230-244.
