# Bootstrap bounds test

`pyardl.bootstrap`

The classical bounds test compares a statistic against *two* critical
values, because the true distribution depends on integration orders
nobody knows. When the statistic lands between them the answer is
**inconclusive** — and on the sample sizes this literature works with,
that happens often enough to be a practical problem.

The bootstrap removes the ambiguity at its source. Rather than
bracketing the distribution, it **builds** it: regenerate the data many
times under a null that is true by construction, recompute the statistic
each time, and read the critical value off the result. The regenerated
data inherit the integration orders, the short-run dynamics and the
error covariance of the sample at hand, so the critical value is
specific to that sample and there is nothing left to bracket.

## `bootstrap_bounds_test(y, x, case=3, order=None, n_boot=2999, resample="iid", seed=None, var_order=1, burn_in=50, store_distribution=False)`

```python
from pyardl.bootstrap import bootstrap_bounds_test
from pyardl.datasets import load_denmark

data = load_denmark()
res = bootstrap_bounds_test(
    data["LRM"], data[["LRY", "IBO", "IDE"]],
    case=3, order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}),
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

## Both routes, side by side

The classical bounds are reported alongside — they are what the
literature quotes, and a disagreement between the two routes is itself a
result. `res.comparison()` returns it as a frame:

| | statistic | boot_cv | boot_p | boot_decision | bound_I0 | bound_I1 | bound_decision |
|---|---|---|---|---|---|---|---|
| F_overall | 6.2059 | 4.8319 | 0.0123 | cointegration | 3.2290 | 4.3223 | cointegration |
| t_BDM | -4.5479 | -3.7780 | 0.0110 | cointegration | -2.8600 | -3.7800 | cointegration |
| F_indep | 8.1619 | 5.7467 | 0.0090 | cointegration | 2.6191 | 4.6457 | cointegration |

`res.agrees_with_bounds()` answers the same question in one boolean.
When it is `False`, neither route is wrong: the bootstrap has no
inconclusive zone and the bounds do, so they can legitimately differ.
That is a reason to report both, not to pick the one you like.

Where the classical route tabulates nothing — the `t` under cases II and
IV — the cell reads `unavailable` rather than being left blank, since a
blank cell reads as a non-rejection.

### What the arguments do

| Argument | Meaning |
|---|---|
| `n_boot` | replications; 2999 is the article's default |
| `resample` | `"iid"`, or `"wild"` when the residuals look heteroskedastic |
| `seed` | drawn from entropy and **recorded** when omitted, so a run stays reproducible after the fact |
| `var_order` | lag order of the marginal VAR that regenerates the regressors |
| `burn_in` | initial periods discarded from each regenerated path |
| `store_distribution` | keeps the simulated statistics on the result |

## How the null is built

Two models are estimated on your data.

**The conditional model, under the null.** The null of the overall F
test is `λ = γ₁ = … = γₖ = 0`. Imposing it means **deleting the level
terms**: the model becomes a regression in differences. The short-run
coefficients are re-estimated under that restriction — not simply
carried over from the unrestricted fit with the level coefficients set
to zero, which would be a different model.

**The marginal model for the regressors.** They cannot be held fixed
across replications: they are I(1), and the statistic's distribution
depends on their stochastic behaviour. They are regenerated from a VAR
in first differences, which reproduces integrated regressors with the
observed short-run dependence and no cointegration among them.

**And one rule that governs the resampling.** The residuals of the two
blocks are contemporaneously correlated — that is what weak exogeneity
is about, and it is rarely exactly zero. So a draw picks a **date**, and
the whole residual vector travels together. Drawing each equation
independently would destroy that correlation and inflate the critical
values in the optimistic direction, with nothing in the output to
suggest anything went wrong.

## What you get, and what you should not read into it

The verdict is **binary**: there is no inconclusive zone left.

The p-value is `(1 + #{as extreme}) / (B + 1)` and therefore never
exactly zero. That is deliberate: reporting `p = 0` would claim more
resolution than `B` replications can provide. With `B = 2999` the
smallest reportable p-value is `1/3000`.

A replication that cannot be estimated — a regenerated sample whose
design happens to be singular — is **counted and dropped**, never
replaced by a fresh draw. Replacing it would quietly bias the
distribution towards the samples that happen to be estimable. If any
were dropped, the summary says so.

## Reproducibility

Same seed, same critical values, bit for bit. The seed, the number of
replications, the resampling scheme, the marginal VAR order and the
burn-in are all recorded on the result, because a bootstrap critical
value nobody else can reproduce is not a result anyone can check.

## Cost

The whole point of `B` replications is that each one re-estimates the
model. Measured on this implementation:

| T | k | ms per replication | one test at B = 2999 |
|---|---|---|---|
| 60 | 1 | 0.117 | 0.35 s |
| 100 | 3 | 0.148 | 0.44 s |
| 200 | 3 | 0.227 | 0.68 s |
| 200 | 5 | 0.604 | 1.81 s |

Both hot paths are vectorised across replications: the regeneration
advances all paths together, and the `B` least-squares fits are solved
by a single stacked QR — never through the normal equations, which
would square the condition number of a design built on lagged levels of
integrated series.

## Validation

Checked against the R package **bootCT** on the Danish data, `B = 2000`
on both sides. The observed statistics agree to **4e-10**, and the
decisions agree at all three levels. The bootstrap critical values for
`F` differ by 0.6% to 13%, which is what two bootstraps with different
generators produce.

The `t` critical values differ more — 21% to 30%, systematically in the
direction that makes ours more demanding. That gap has since been
**resolved by measurement**, not left as a caveat.

Both statistics are bootstrapped under the same joint null,
`λ = γ = 0`. The alternative — building the `t` distribution under the
weaker null `λ = 0` alone, leaving the level terms of the regressors
free — sounds more faithful to what the `t` test asks, and is almost
certainly what the other implementation does. It is also wrong: over 400
Monte Carlo samples it rejects **9.3% of the time at a nominal 5%** in
case III, and 8.3% in case V, where the joint null gives 3.5% and 5.0%.

So our bounds are not conservative by accident; they are built under the
null that holds the size. The full experiment, including a hypothesis
about the mechanism that the data refuted, is recorded as OBS-8 in the
[validation register](../VALIDATION_OBSERVATIONS.md).

Details, including a convention trap in `bootCT`'s lag argument that
makes a naive comparison disagree by 58% while comparing two different
models, are in
[`PROVENANCE.md`](https://github.com/Matth-analyst/pyardl/blob/main/src/pyardl/critical_values/PROVENANCE.md).

## What the bootstrap buys, measured

> For the decision itself — which route to believe on your own data —
> see [Bootstrap or classical bounds?](../bootstrap-or-bounds.md).

1000 replications, `T = 100`, `B = 299`, case III, correlated
innovations, on the four canonical systems of `pyardl.simulate`:

| DGP | bootstrap correct | bounds correct | bounds inconclusive |
|---|---|---|---|
| cointegration | 100.0% | 100.0% | 0.0% |
| degenerate_1 | 99.4% | 93.2% | 5.5% |
| degenerate_2 | 96.3% | 99.8% | 0.1% |
| no cointegration | 91.5% | 71.3% | 24.8% |

Almost all of the gain is the disappearance of the inconclusive zone,
and it shows up exactly where that zone is wide. Under no cointegration
the bounds leave a quarter of the samples without a verdict; the
bootstrap settles nearly all of them correctly.

Where neither route hesitates — clear cointegration — the bootstrap adds
nothing at all.

**And it has a cost.** Under a type 2 degeneracy the bootstrap calls
cointegration 3.7% of the time against the bounds' 0.1%. Deciding also
means being confidently wrong where the bounds stayed silent, and that
error runs in the worst direction: a degeneracy read as a relationship.
Details in OBS-12, and the related small-sample fragility of `F_indep`
in OBS-11.

## Building blocks

The pieces are exposed, because a bootstrap you cannot inspect is a
bootstrap you cannot debug:

| Function | Purpose |
|---|---|
| `estimate_null_dgp(y, x, p, q, case, var_order)` | the two models above |
| `simulate_paths(dgp, innovations, y0, x0, burn_in)` | regenerate `B` samples at once |
| `simulate_path(...)` | one sample; delegates to the batched routine, so the two cannot drift apart |
| `resample_residuals(residuals, n_draw, rng, scheme)` | row-wise resampling, `iid` or `wild` |
