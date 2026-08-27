# Unified analysis — one entry point for the whole matrix

`pyardl.unified.cointegration_analysis`

By 2026 applied work rarely runs a plain bounds test. It runs a bounds
test *with* a smooth-break correction, *with* an asymmetric
decomposition, *with* bootstrap critical values — the combination that
Roudane's Stata module and the CRAN package `fbardl` package up. Three
binary switches, eight configurations:

```python
from pyardl.unified import cointegration_analysis

res = cointegration_analysis(
    y, x,
    asym=["oil"],            # partial-sum decomposition
    fourier={"k": 1},        # smooth break
    inference="bootstrap",   # joint-null bootstrap
)
print(res.summary())
```

This module **owns no estimator and no distribution of its own.** Every
cell delegates to, or is assembled from, bricks already validated by
their own specs. Its one real responsibility is the thing applied work
gets wrong: giving each cell the critical values that cell actually
requires.

## The rule that justifies the module

Critical values are not a detail you inherit from whichever table is
nearest. Two of the three switches change the null distribution:

| cell | `inference="bounds"` | `inference="bootstrap"` |
|---|---|---|
| plain ARDL | PSS tables | joint-null bootstrap |
| + Fourier | simulated, search in the loop | joint-null bootstrap |
| + decomposition | simulated SYG values | joint-null bootstrap |
| **both** | **refused** | joint-null bootstrap |

`resolve_critical_values(asym, fourier, inference)` returns the source
and a one-sentence reason you can quote in a methods section:

```python
>>> from pyardl.unified import resolve_critical_values
>>> source, reason = resolve_critical_values(False, True, "bounds")
>>> source
'simulated_fourier'
```

**The refused cell is the point.** There is no published table for an
asymmetric decomposition combined with a searched Fourier frequency —
the partial sums change the effective regressor count *and* the search
changes the null distribution, and nothing in the literature covers both
at once. Substituting a neighbouring table would run, would look
plausible, and would be wrong by an unknown amount. So the call raises,
and names the way out:

```text
ValueError: No tabulated or pre-simulated critical values exist for the
combination of an asymmetric decomposition with Fourier terms: ... Use
inference='bootstrap', which simulates the null of this exact
specification.
```

## Where the bootstrap route comes from

The bootstrap cell is one engine for all eight configurations, built
from the validated pieces of specs 14, 16, 17 and 20 rather than written
afresh. Three wiring decisions carry the methodology:

**The marginal VAR runs on the original regressors, not the partial
sums.** The increments of a partial sum are sign-constrained by
construction; a VAR fitted to them would generate paths whose increments
have no such constraint — partial sums of nothing. So the engine
regenerates `x` at its original scale and **re-applies the
decomposition to every regenerated path**, exactly as the observed call
does.

**Fourier terms stay in the null model.** A deterministic term does not
vanish when the level relationship does. Their coefficients are
estimated under the joint null and carried into every regenerated path,
extended backwards through the burn-in the same way the trend is — the
only choice that leaves no discontinuity at the join.

**When the frequency was searched, every replication searches too.**
The Davies lesson of OBS-15 survives composition: a critical value
computed at a fixed frequency does not apply to the result of a search.

### The lock

An orchestration layer that quietly re-implements its bricks is worse
than no layer at all. So the engine is pinned to the brick it
generalises: with the decomposition and the Fourier terms switched off,
`_bootstrap_cell` and `bootstrap_bounds_test` must agree **to machine
precision on the same seed** — same null DGP, same innovations, same
paths. Measured on 999 replications:

| quantity | absolute difference |
|---|---|
| `F_overall` | 1.4e-14 |
| `t_BDM` | 5.3e-15 |
| `F_indep` | 0.0 |
| all nine critical values | 0.0 |

That test is in the suite, and it is the one to run first if the engine
is ever touched.

## What the combination costs, measured

Each brick was measured correctly sized *on its own*. That says nothing
about their combination — and the natural hypothesis, that distortions
compound so the richest cell is the worst, is a hypothesis. It was
measured: 2000 replications under a true null, T = 100, nominal 5%,
standard error **0.49 point**, so the two-standard-error band is
[4.03%, 5.97%].

| cell | `F_overall` | `t_BDM` | `F_indep` | joint |
|---|---|---|---|---|
| ardl | 6.15% | 5.45% | 5.50% | 1.70% |
| nardl | 5.30% | 5.85% | 4.35% | 2.05% |
| ardl+fourier | **7.10%** | **7.50%** | 4.90% | 2.10% |
| nardl+fourier | **7.35%** | **8.50%** | 5.05% | 3.10% |

**The hypothesis was wrong in an informative way.** The decomposition
does not distort: the `nardl` row sits at nominal on all three
statistics. Everything comes from the Fourier terms.

And that contradicts the Fourier-ADL page, where the same ingredient
measured **3.5%** — under-rejection — with simulated critical values and
the search inside the loop. Two protocols, opposite verdicts on the same
component. So it was arbitrated rather than explained away:

| arm | `F_overall` | `t_BDM` |
|---|---|---|
| frequency searched, terms in the null DGP | 7.10% | 7.50% |
| frequency fixed at 1 | 5.50% | 5.35% |
| frequency fixed at 2 | 4.85% | 4.85% |
| searched, terms removed from the null DGP | 5.55% | 5.60% |

**Neither half causes it alone.** Remove the search: back to nominal.
Remove the Fourier component from the null model: back to nominal. Both
together over-reject.

The mechanism follows. The bootstrap does re-run the search in every
replication, so the search itself is calibrated — that is the lesson of
the Davies problem, correctly applied. What is not calibrated is that
the null DGP was estimated **conditional on the frequency the search had
already won**. Its Fourier coefficients absorb the component fitted to
the observed sample, noise included; the regenerated paths carry a wave
tailored to that sample; and each replication then re-searches on paths
where the winning frequency is already present and easy to find. Their
statistics come out systematically less extreme than the observed one,
whose advantage came from fitting noise the replications inherit as
signal.

**The general lesson is larger than this cell.** Putting a selection
step inside the bootstrap loop is not enough to calibrate it. The world
you simulate from must also not have been fitted to that selection's
outcome. Recorded as OBS-19.

So the combination is kept — it is what applied work uses — and it
warns, with the measured number and the correctly sized way out:

```python
res = cointegration_analysis(y, x, fourier={"k": 1, "freq": 1.0},
                             inference="bootstrap")   # no warning
```

The simulated-critical-value route of the Fourier-ADL page is not
affected and carries no such warning.

### One result worth keeping

The joint classification stays **conservative in every cell**, 1.70% to
3.10%, including the cell whose individual tests over-reject most. The
three-test conjunction of Sam, McNown and Goh absorbs the individual
distortion: requiring all three to agree protects, at the cost of power.
That is a measured argument for the three-test framework rather than an
assumed one.

## Reading the result

```text
Unified cointegration analysis - cell nardl/bootstrap
  case 3, 139 observations, ECM(1; oil_pos:1, oil_neg:1)
  critical values: bootstrap - Critical values bootstrapped under the joint null DGP (McNown, Sam & Goh 2018).

  F_overall  =  48.8409   critical (5%) = 5.2345   -> cointegration
  t_BDM      = -11.5503   critical (5%) = -3.7236   -> cointegration
  F_indep    =  72.2202   critical (5%) = 6.3858   -> cointegration

  classification: cointegration
  F_overall, t_BDM and F_indep all reject: the level terms are jointly significant, y adjusts back towards equilibrium, and the regressors carry the long-run relationship.
```

```python
res.classification     # the joint verdict of the three-test framework
res.decision_f         # per-test verdicts
res.cv_source          # which critical values were used
res.cv_reason          # and why, in one sentence
res.detail             # the underlying brick's own result object
```

Fields are `None` when the cell's source does not cover that statistic:
the tabulated NARDL route carries only the overall `F`, the simulated
Fourier route only the `t`. `summary()` says `not covered by this
source` rather than leaving a blank — and rather than filling the cell
with a critical value from somewhere else. Only the bootstrap route
carries the full triplet in every cell, which is a property of the
literature, reported rather than papered over.

### The robustness table

`compare()` runs the sibling cells on the same data and tabulates the
verdicts — the table applied papers publish, without the copy-paste:

```python
table = res.compare()
```

```text
                              F_overall      t_BDM     F_indep classification
cell
ardl/bootstrap                68.671716 -11.384565  135.139582  cointegration
ardl+fourier(k=1)/bootstrap   74.351005 -11.870718  146.531275  cointegration
nardl/bootstrap               48.840908 -11.550257   72.220166  cointegration
nardl+fourier(k=1)/bootstrap  52.506816 -12.078331   77.758432  cointegration
```

A cell with no valid source appears in the table marked
`unavailable: ...` rather than silently dropping out of it.

## A guard worth having

A decomposition, a pair of sinusoids and a few lags each look cheap
alone. Together, on ninety observations, they are not. When the ratio of
usable observations to parameters falls below five, the call warns:
every statistic is still computed, but its finite-sample distribution is
poorly approximated at that ratio, and saying so is more useful than a
confident number.

## References

- Sam, C. Y., McNown, R. & Goh, S. K. (2019). An augmented autoregressive
  distributed lag bounds test for cointegration. *Economic Modelling*,
  80, 130-141.
- McNown, R., Sam, C. Y. & Goh, S. K. (2018). Bootstrapping the
  autoregressive distributed lag test for cointegration. *Applied
  Economics*, 50(13), 1509-1521.
- Shin, Y., Yu, B. & Greenwood-Nimmo, M. (2014). Modelling asymmetric
  cointegration and dynamic multipliers in a nonlinear ARDL framework.
- Banerjee, P., Arčabić, V. & Lee, H. (2017). Fourier ADL cointegration
  test to approximate smooth breaks. *Economic Modelling*, 67, 114-124.
