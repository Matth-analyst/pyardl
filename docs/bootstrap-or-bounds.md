# When to use the bootstrap rather than the classical bounds

Both routes answer the same question, and `pyardl` runs both on the same
sample. This page says which one to believe, and it answers with
measurements rather than preferences.

## The short answer

Use the **bootstrap** when the classical bounds leave you without a
verdict, and when your sample is large enough for a resampling procedure
to mean anything. Use the **classical bounds** when they already decide,
when you need a number the literature will recognise, or when the sample
is too short for the bootstrap's null model to be estimated with any
precision.

Most of the time the two agree, and the choice does not arise. The
interesting question is what happens when they do not.

## The measurement

1000 replications, `T = 100`, `B = 299`, case III, correlated
innovations, on the four canonical systems of `pyardl.simulate`
(`validation/spec16_montecarlo.py`). Each row is a data-generating
process; each cell is how often a route returned that verdict.

| DGP | route | cointegration | degenerate_1 | degenerate_2 | no cointegration | inconclusive |
|---|---|---|---|---|---|---|
| **cointegration** | bootstrap | **100.0%** | 0.0% | 0.0% | 0.0% | 0.0% |
| | bounds | **100.0%** | 0.0% | 0.0% | 0.0% | 0.0% |
| **degenerate_1** | bootstrap | 0.6% | **99.4%** | 0.0% | 0.0% | 0.0% |
| | bounds | 1.3% | **93.2%** | 0.0% | 0.0% | 5.5% |
| **degenerate_2** | bootstrap | 3.7% | 0.0% | **96.3%** | 0.0% | 0.0% |
| | bounds | 0.1% | 0.0% | **99.8%** | 0.0% | 0.1% |
| **no cointegration** | bootstrap | 2.2% | 2.8% | 0.8% | **91.5%** | 2.7% |
| | bounds | 2.3% | 1.2% | 0.4% | **71.3%** | 24.8% |

The bold cell on each line is the correct answer for that DGP.

## Reading it

**Where the bootstrap wins, it wins on the inconclusive column, and
nowhere else.** Under no cointegration the bounds leave **24.8%** of
samples without a verdict; the bootstrap has no inconclusive zone and
settles nearly all of them correctly — 91.5% against 71.3%. That gap of
twenty points is almost exactly the inconclusive column moving into the
correct one.

**Where nothing is ambiguous, the bootstrap adds nothing.** Under clear
cointegration both routes are right 100% of the time. Paying `B`
re-estimations to confirm a verdict the bounds already gave is a waste
of everyone's afternoon.

**And it has a cost, in the worst direction.** Under a type 2
degeneracy — the regressors' levels matter but nothing pulls `y` back —
the bootstrap claims cointegration **3.7%** of the time against the
bounds' **0.1%**. Removing the inconclusive zone does not create
information; it forces a decision where there was none, and some of
those decisions are wrong. Here the error means reading a degeneracy as
a genuine relationship, which is the mistake the three-test framework
exists to prevent.

That is the trade in one sentence: **the bootstrap converts "I don't
know" into an answer, and a small share of those answers are confidently
wrong.**

## When the two disagree

They agree 100%, 93.8%, 96.4% and 76.8% of the time on the four DGPs
above. The disagreement is concentrated exactly where the bounds are
inconclusive — the two routes do not diverge on the cases either of them
decides cleanly.

`pyardl` reports both rather than choosing for you:

```python
res = bootstrap_bounds_test(y, x, case=3, order=(1, 1), n_boot=2999, seed=42)

res.comparison()          # one row per test, both routes side by side
res.agrees_with_bounds()  # the same question as a boolean
```

When `agrees_with_bounds()` is `False`, neither route is broken. The
honest reading is that your sample does not settle the question, and the
bootstrap's verdict is the more fragile of the two — it is the one that
had to decide.

## What the bootstrap cannot fix

- **A short sample.** The null model is estimated from your data. At
  `T = 60` there is little to estimate it from, and a bootstrap built on
  a poorly estimated null is not more reliable than a tabulated bound —
  it is less, and it looks more confident.
- **A misspecified lag order.** Autocorrelated residuals invalidate both
  routes equally. `bounds_test` warns; the warning applies to the
  bootstrap too.
- **`F_indep` in small samples.** It is oversized at `T = 100` — 6.5% at
  a nominal 5%, where the `t` holds its size. A `degenerate_1` verdict
  that rests on `F_indep` narrowly failing to reject deserves suspicion
  at that sample size. See OBS-11.
- **Cointegration among the regressors.** Neither route detects it.
  `check_no_cointegration_among_x` does.

## Practical rule

1. Run `bounds_test`. If it decides, and the diagnostics are clean, you
   are done.
2. If it returns `inconclusive`, run `bootstrap_bounds_test` on the same
   specification. Report both, and say which one you followed.
3. If the two disagree on a *named* verdict — not merely
   bounds-inconclusive against a bootstrap decision — treat the result
   as weak evidence whichever way it points, and say so.
4. Never report the bootstrap verdict alone because it is the one you
   liked. `comparison()` exists so that the reader sees what you saw.

## Sources

The figures on this page come from
`validation/results/spec16_montecarlo.txt` and its JSON companion, and
are recorded as **OBS-12** in
[`VALIDATION_OBSERVATIONS.md`](VALIDATION_OBSERVATIONS.md). The
article's own Monte Carlo tables are behind an access barrier, so the
specification's numeric criterion could not be checked against them;
what is verified here is the set of qualitative claims, measured on our
own data-generating processes.
