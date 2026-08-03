# Testing an economic restriction

Estimating a long-run coefficient tells you what the data say. Theory
usually says something more specific — that an elasticity should equal
one, that two coefficients should sum to zero, that a ratio should be
stationary. Those are testable claims, and testing them is the point of
the exercise.

This walkthrough follows the method of Davidson, Hendry, Srba & Yeo
(1978): find the long-run relationship, test the restriction theory
implies, and — if the data do not reject it — impose it and read the
model that results.

!!! note "About the data"
    DHSY worked on UK consumption and income. Those series come from a
    1978 article behind an access barrier, and no freely redistributable
    version was identified, so they are **not** shipped with `pyardl`.

    This page therefore demonstrates their *method* on the Danish
    money-demand data included with the package. The economics is
    parallel — DHSY's restriction makes `log(C/Y)` the error-correction
    term, ours makes it `log(M/Y)`, the velocity of money — and both are
    ratios that theory expects to be stationary. What follows is a
    demonstration of the method, not a replication of the paper.

## 1. Estimate the relationship

```python
from pyardl.core.ardl import ARDL
from pyardl.datasets import load_denmark

data = load_denmark()
y = data["LRM"]                      # log real money
x = data[["LRY", "IBO", "IDE"]]      # income, bond rate, deposit rate

res = ARDL(y, x, order=(3, {"LRY": 1, "IBO": 3, "IDE": 2})).fit()
print(res.longrun.round(4))
```

```text
      theta      se
LRY  0.9965  0.1239
IBO -4.5381  0.5203
IDE  2.8915  0.9951
```

The income elasticity is 0.9965. That is strikingly close to one — but
"close to one" is an impression, not a result. The standard error is
0.124, so a value of 0.85 or 1.15 would look much the same. The question
has to be asked properly.

## 2. State the restriction

Quantity theory implies a unit income elasticity of money demand: a one
percent rise in real income raises money demand by one percent, leaving
velocity unchanged in the long run. In matrix form, with three
regressors:

```
R = [1  0  0],    r = 1
```

## 3. Test it

```python
out = res.test_longrun_restriction([[1.0, 0.0, 0.0]], 1.0)
print(out.summary())
```

```text
Long-run restriction test - Wald chi2(1) = 0.0008, p = 0.9773
  decision (5%): not_rejected
  R.theta - r = [-0.0035]
```

The restriction is not rejected, and not by a narrow margin.

Two details of the output are deliberate. The discrepancy is reported
**signed** — here −0.0035, so the estimate sits marginally *below* one
— because the direction of a violation is often more informative than
its size. And the verdict reads `not_rejected`, never `accept`: with 55
quarterly observations, failing to reject is comfortably compatible with
a true elasticity of 0.9 that the test simply cannot see.

## 4. Impose it

Testing a restriction and adopting it are different decisions. Having
failed to reject unit elasticity, DHSY rewrote their model around the
ratio. The same move here:

```python
out = res.test_longrun_restriction([[1.0, 0.0, 0.0]], 1.0, impose=True)
print(out.restricted_params.round(4).head(4))
```

```text
const           2.6159
(LRM-LRY).L1   -0.4176
IBO.L1         -1.8925
IDE.L1          1.2070
```

The two level terms `LRM_{t-1}` and `LRY_{t-1}` have collapsed into a
single regressor, `(LRM − LRY)_{t-1}` — the log of the money-income
ratio, that is, **the velocity of money**. What was a pair of estimated
coefficients is now one economically named quantity, and the model has
gained a degree of freedom.

Its coefficient, −0.4176, is the speed of adjustment: about 42% of any
deviation of velocity from its equilibrium is corrected within a
quarter.

## 5. Check what the restriction cost

```python
print(out.summary())
```

```text
  imposed: F = 0.0008, p = 0.9774
  SSR unrestricted = 0.014228, restricted = 0.014229
```

The residual sum of squares moved from 0.014228 to 0.014229 — the fourth
significant digit. The restriction is free: it buys an interpretable
model and a degree of freedom at no measurable cost in fit.

The F test compares those two sums of squares directly, and agrees with
the Wald test. They are not the same statistic — the Wald is asymptotic
and nonlinear in the parameters, the F is exact under normality — so
their agreement is a genuine check rather than a tautology.

That comparison is only legitimate because the unrestricted
error-correction model reproduces the original ARDL regression exactly:
same sample, same residuals, same sum of squares, verified to 1e-10. If
the two were fitted on different samples — which is what happens if
lagged differences are built carelessly — the F statistic would be
meaningless while still looking perfectly plausible.

## 6. Check the model held still

A long-run coefficient estimated across a structural break is an average
of two regimes, not an equilibrium. Before believing any of the above:

```python
print(res.stability())
```

```text
                  stable  max_excess  first_crossing
test
CUSUM               True         0.0             NaN
CUSUM-of-squares    True         0.0             NaN
```

Both tests pass, and both matter: they detect different failures, and
one can be blind where the other is not. See
[stability diagnostics](api/diagnostics.md).

## What to take away

The restriction was worth stating. Without it, the model reports an
income elasticity of 0.9965 ± 0.124 and leaves the reader to decide
whether that is one. With it, the model says something economists can
argue about: velocity is stationary, and deviations from it correct at
42% per quarter.

That is the DHSY discipline — write the model so the economics is
testable, then test it.
