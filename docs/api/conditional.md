# Conditional and unconditional models

`conditional=True | False`

The error-correction model the bounds test is built on contains the
**contemporaneous** differences of the regressors, `Δx_t`. That is
Pesaran, Shin and Smith's conditional model: it conditions on the
regressors' current movement, which is legitimate when they are weakly
exogenous.

Bertelli, Vacca and Zoia (2022) point out that the choice is not
innocuous. Drop those terms and you get the **unconditional** model —
and under some degeneracies the two lead to different conclusions. So
`pyardl` offers both, everywhere the model appears, and says on the
output which one produced the numbers.

```python
res = bounds_test(y, x, case=3, order=(3, 1))          # conditional
res = bounds_test(y, x, case=3, order=(3, 1), conditional=False)
```

The default is `True` — the PSS framework, and what the applied
literature reports.

## What exactly changes

Only the contemporaneous differences are removed. Everything else — the
deterministic terms, the level terms, the lagged differences of `y`, the
lagged differences of the regressors — stays in place, in the same
order.

| Term | conditional | unconditional |
|---|---|---|
| `const`, `trend` | yes | yes |
| `y.L1`, `x.L1` (tested) | yes | yes |
| `D.y.L1 … L(p-1)` | yes | yes |
| `D.x.L0` | **yes** | **no** |
| `D.x.L1 … L(q-1)` | yes | yes |

The tested vector is untouched, which is what makes the two comparable:
they test the same restriction on two specifications, rather than two
restrictions.

## How that convention was established

Not by reading it. "Unconditional means without `Δx_t`" is the easy
half; the question is what happens to the *rest* of the specification —
are the remaining short-run terms shifted to lags `1..q_j`, keeping
their number, or is one simply removed?

`bootCT` reports both versions of `F_indep` itself, so the answer could
be measured. On the Danish data, case III, order (3; 1, 3, 2), it gives
`3.40560040391862`. Two candidates were computed here:

| candidate | `F_indep` |
|---|---|
| conditional design minus `Δx_t`, nothing else changed | **3.405600** |
| the VECM equation for `y` | 3.327761 |

The first matches to 1e-12; the second does not match at all. The
convention follows the measurement. The same run reproduces bootCT's
conditional statistic (`8.161935`) to 4e-13, so both columns agree.

That check is a permanent test, not a one-off:
`tests/replication/test_spec16.py`.

## In the bootstrap

`conditional` is threaded through the whole path — the observed
statistic, the estimated null model, the regenerated data and the
re-estimation on each replication. That is not tidiness: if the null
model kept `Δx_t` while the statistic was computed without it, the
simulated null would not be the null being tested, and nothing in the
output would say so.

```python
res = bootstrap_bounds_test(y, x, order=(2, 2), conditional=False)
```

## When it matters

The two models coincide when the innovations of the `y` equation and of
the regressors are contemporaneously uncorrelated — then `Δx_t` carries
no information about `Δy_t` beyond what the lags already hold. The more
correlated they are, the further apart the two specifications drift.

That correlation is exactly what weak exogeneity is about, and it is
rarely zero in practice. If the two forms disagree on your data, that
disagreement is information: report it rather than picking the one you
prefer.

## References

- Bertelli, S., Vacca, G. & Zoia, M. (2022). Bootstrap cointegration
  tests in ARDL models. *Economic Modelling*, 116, 105987.
- Vacca, G. & Bertelli, S. (2024). bootCT: bootstrap cointegration tests
  in ARDL models. *The R Journal*.
