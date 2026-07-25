# Critical values

`pyardl.critical_values`

Because the limiting distributions of the bounds test depend on the
unknown integration order of the regressors, critical values come in
pairs: a lower bound assuming all regressors are I(0) and an upper bound
assuming all are I(1).

## Choosing a source

| `cv_source` | Use it for | Coverage |
|---|---|---|
| `"kripfganz"` | everyday work — the default | cases 1-5, `k = 1..10`, F, any level, with p-values |
| `"pss"` | reproducing published results exactly | cases 1-5, `k = 0..10`, F and t, 10/5/2.5*/1% |
| `"narayan"` | small samples, `30 ≤ T ≤ 80` | cases 2, 3, 5, `k ≤ 7`, F, 10/5/1% |

\* the 2.5% level comes from internal simulation, not from the printed
tables.

**Why `"kripfganz"` by default.** It is response-surface based, fitted
on far more replications than the original tables (32 million per
configuration against 40 000), so it is more precise, available at any
significance level, and yields p-values rather than only reject/not
reject. The t bounds still come from the published tables, since the
response-surface material covers the F statistic only.

**Why `"pss"` still exists.** Its job is fidelity, not precision: it
returns exactly what the article printed, so you can reproduce a
published table down to the last decimal. Those values carry the Monte
Carlo error of the original work — roughly ±0.05 at the usual levels,
up to ±0.15 in the 1% tail.

**Why `"narayan"` matters.** Asymptotic bounds over-reject noticeably
when `T` is between 30 and 80 — precisely where annual data tends to
land. Using them there produces spurious cointegration findings.

## `get_bounds(stat, case, k, alpha, cv_source="pss", t_obs=None)`

Returns `(lower, upper)`.

```python
from pyardl.critical_values import get_bounds

get_bounds("F", case=3, k=1, alpha=0.05)
# (4.94, 5.73)

get_bounds("F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=40)
# (5.26, 6.16)
```

For the t statistic the test is left-tailed, so the "upper" bound is the
more negative of the two.

`t_obs` is required for `"narayan"`, which interpolates linearly between
the tabulated sizes and falls back to the asymptotic bounds, with a
warning, outside 30-80.

Requesting a combination a source does not cover raises an error naming
a source that does. Nothing is ever silently substituted with a
neighbouring cell.

## `pvalue_bounds(f_stat, case, k)`

Approximate p-values of the F statistic at both bounds.

```python
from pyardl.critical_values import pvalue_bounds

p_i0, p_i1 = pvalue_bounds(6.0, case=3, k=1)
```

Read them as: `p_i1 <= alpha` means cointegration, `p_i0 > alpha` means
no rejection, anything in between is the inconclusive zone.

## `simulate_bounds(case, k, t_obs=1000, n_sims=40_000, seed=0, i1=True, alphas=..., chunk=2_000)`

Monte Carlo engine for configurations no published table covers: an
unusual significance level, more regressors than the tables reach, or an
arbitrary sample size.

```python
from pyardl.critical_values import simulate_bounds

sb = simulate_bounds(case=3, k=2, t_obs=45, n_sims=100_000, seed=42)
sb.f_cv(0.05), sb.t_cv(0.05)
sb.seed, sb.n_sims, sb.chunk      # everything needed to reproduce the run
```

Under the null, `y` is a random walk; the regressors are i.i.d. draws
for the lower bound and independent random walks for the upper one.

Every simulation parameter is recorded on the result, including `chunk`:
draws are generated in batches, so the same `(seed, n_sims, chunk)`
always yields exactly the same statistics. A set of critical values can
therefore always be traced back to how it was produced.

## Provenance

Every shipped table documents its exact source, how it was transcribed,
and how it was cross-checked, in
[`PROVENANCE.md`](https://github.com/Matth-analyst/pyardl/blob/main/src/pyardl/critical_values/PROVENANCE.md).

The tables were verified cell by cell against independent sources
(response surfaces, Dickey-Fuller critical values) and, where no second
published source existed, against the internal Monte Carlo engine. The
comparison criterion is derived from the Monte Carlo standard error of
each quantile rather than a flat tolerance, since published tables carry
their own simulation error.
