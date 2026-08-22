# The three-test framework

`pyardl.bounds.classification`

The overall `F` test asks whether **all** the level terms are zero. It
rejects if any of them is not — and two of the ways it can reject have
nothing to do with cointegration.

**Degeneracy of type 1** — `λ ≠ 0`, `γ = 0`. The dependent variable
pulls back towards its own past, and the regressors carry nothing. What
looks like error correction is `y` correcting towards a constant.

**Degeneracy of type 2** — `γ ≠ 0`, `λ = 0`. The regressors' levels are
jointly significant, but nothing pulls `y` back. No mechanism restores
the relationship, so nothing holds it together.

Sam, McNown and Goh (2019) close both holes with a third test on the
regressors' levels alone:

| Test | Null | Tail |
|---|---|---|
| `F_overall` | `λ = γ = 0` | right |
| `t_BDM` | `λ = 0` | **left** |
| `F_indep` | `γ = 0` | right |

Cointegration is established **only when all three reject**. Every other
combination has a name, and `pyardl` gives it one rather than leaving
three numbers on the table for the reader to combine.

## Reading the verdict

```python
from pyardl.bounds import bounds_test

res = bounds_test(y, x, case=3, order=(1, 1))
label, reason = res.classification()
```

| `label` | Meaning |
|---|---|
| `cointegration` | all three reject |
| `degenerate_1` | `F` and `t` reject, `F_indep` does not |
| `degenerate_2` | `F` and `F_indep` reject, `t` does not |
| `no_cointegration` | none rejects |
| `inconclusive` | anything else, including a test that fell between its bounds or could not be run |

`reason` says in one sentence *which test decided*. The label alone says
what; the reason says why, which is what a reader needs in order to
judge whether to believe it.

The mapping is **total**: every combination of three three-state
verdicts lands on a named outcome, and none falls through to a default.
A silent `else` there would be the most dangerous line in the library —
it would return a verdict nobody chose.

## Where the bounds come from

`F_indep` has no standard distribution either — it depends on the
integration orders — so it is read against a pair of bounds like
`F_overall`.

Sam, McNown and Goh publish theirs behind an access barrier, and the
project rule is that a critical value nobody here computed does not get
encoded. So they are **simulated**, by the same engine that produces the
2.5% PSS bounds, under exactly the PSS null: `y` a random walk,
regressors i.i.d. for the I(0) bound and independent random walks for
the I(1) bound. `F_indep` is computed on the *same* replications as
`F_overall` and `t_BDM`, so the three sets of bounds describe one single
world rather than three loosely related ones.

The **statistic** itself is checked externally: `bootCT::boot_ardl`
reports its own `F_indep` on the Danish data, and the two agree to
**4e-13**. The bounds are internal, and their cross-checks are
structural rather than external — the distinction is recorded as OBS-9.

Coverage is cases 1 to 5, `k = 1..10`, at the 10%, 5% and 1% levels.
Outside that grid `decision_indep` is `None` and the classification
refuses to conclude — **no neighbouring value is substituted**.

## In the bootstrap

`bootstrap_bounds_test` reports the same three tests, with one
difference that matters: there is no inconclusive zone, so
`degenerate_1` and `degenerate_2` are reached cleanly rather than
through a bracket.

All three statistics are bootstrapped under the **same joint null**,
`λ = γ = 0`. That is not a convenience — it is a measured choice.
Building each test's distribution under its own weaker null sounds more
faithful to what each test asks, and it inflates size: 9.3% at a nominal
5% for the `t` in case III, against 3.5% for the joint null. The
experiment, including a hypothesis about the mechanism that the data
refuted, is recorded as OBS-8 in the
[validation register](../VALIDATION_OBSERVATIONS.md).

## The older two-test verdict

`decision_joint` is still on the result object, and still reports
`degenerate_suspicion` when `F` rejects and `t` does not. It is kept for
continuity, but it can only *suspect*: it cannot tell the two
degeneracies apart, because with two tests the information to do so does
not exist. Prefer `classification()`.

## References

- Sam, C. Y., McNown, R. & Goh, S. K. (2019). An augmented autoregressive
  distributed lag bounds test for cointegration. *Economic Modelling*,
  80, 130-141.
- McNown, R., Sam, C. Y. & Goh, S. K. (2018). Bootstrapping the
  autoregressive distributed lag test for cointegration. *Applied
  Economics*, 50(13), 1509-1521.
