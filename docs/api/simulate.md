# VECM simulator

`pyardl.simulate`

Every Monte Carlo study in this library draws its data from one
generator. Before that, each specification carried its own — which is
how two studies end up disagreeing for a reason nobody can locate: not
because the estimators differ, but because the *data* did.

```
Δz_t = Π z_{t-1} + Σ Γ_i Δz_{t-i} + d_t + ε_t,    Π = α β'
```

with `z_t = (y_t, x_1t, …, x_kt)'`.

## `vecm_ardl(n_obs, alpha, beta, gammas=(), case=3, sigma=None, ...)`

```python
import numpy as np
from pyardl.simulate import vecm_ardl, degenerate_system

alpha, beta = degenerate_system(None, k=1, speed=-0.5)   # cointegration
sim = vecm_ardl(300, alpha=alpha, beta=beta,
                sigma=np.array([[1.0, 0.6], [0.6, 1.0]]), seed=42)

sim.y        # the dependent variable
sim.x        # the regressors
sim.rank     # 1
sim.lam      # the true λ
sim.seed     # 42, recorded even when drawn from entropy
```

Writing `Π = α β'` is what makes the rank **chosen** rather than hoped
for: `beta` holds the long-run relations, one per column, and `alpha`
the speed at which each equation adjusts to them.

`rank` reports the rank of `Π`, not the number of columns you passed. A
zero `alpha` creates no relation, and reporting the column count there
would claim a relation the data do not contain.

## The canonical systems

`degenerate_system(kind, k, speed)` returns the `(alpha, beta)` pair for
each case the three-test framework has to tell apart:

| `kind` | System | `Π[0]` for `k=1` |
|---|---|---|
| `None` | genuine cointegration | `[-0.4, 0.4]` |
| `1` | type 1: `λ ≠ 0`, `γ = 0` | `[-0.4, 0.0]` |
| `2` | type 2: `γ ≠ 0`, `λ = 0` | `[0.0, -0.4]` |

For no cointegration at all there is nothing to build: pass a zero
system to `vecm_ardl` directly.

## What is guaranteed, and what is not

**Guaranteed**: what comes back is what the parameters describe. Shapes
are checked, and a deterministic term the requested case does not carry
is **refused** rather than absorbed — passing a trend to case 3 is a
specification error, not a detail.

**Not guaranteed**: stability. An explosive system is a legitimate thing
to simulate — to check that a test never calls it cointegration, for
instance — so no stationarity condition is imposed behind your back.

## Why `sigma` deserves attention

With a diagonal `sigma`, the conditional and unconditional models
coincide: `Δx_t` carries no information about `Δy_t` beyond the lags.
Any study meant to distinguish them needs correlated innovations, which
is also the realistic case — that correlation is what weak exogeneity is
about.

## How the simulator is tested

Not on what it prints. On what estimators recover from it:

- the Johansen test (spec 07) finds the injected rank;
- the three-test classification (spec 15) never calls an injected
  degeneracy `cointegration`;
- a zero-`Π` system is detected as cointegrated no more often than
  Johansen's own false-detection rate (OBS-10) — the generator invents
  no relation.

That is the only check that catches an error in the construction of the
DGP itself.

## References

- Bertelli, S., Vacca, G. & Zoia, M. (2022). Bootstrap cointegration
  tests in ARDL models. *Economic Modelling*, 116, 105987.
- Johansen, S. (1991). Estimation and hypothesis testing of
  cointegration vectors in Gaussian vector autoregressive models.
  *Econometrica*, 59(6), 1551-1580.
