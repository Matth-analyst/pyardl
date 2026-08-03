# Long-run restrictions and seasonality

Davidson, Hendry, Srba & Yeo (1978) contributed a discipline rather than
an estimator: write the model so the economics is testable, then test
it. Their consumption function turns on one such restriction — a
long-run elasticity of consumption to income equal to one, which lets
the level term be written as the ratio `log(C/Y)`.

Three tools follow from that, and all three are useful well beyond the
original paper.

## `ARDLResults.test_longrun_restriction(R, r, impose=False)`

A Wald test of `R θ = r` on the long-run coefficients.

```python
res = ARDL(y, x, order=(2, 2)).fit()
out = res.test_longrun_restriction([[1.0]], 1.0)
print(out.summary())
```

```text
Long-run restriction test - Wald chi2(1) = 0.4187, p = 0.5176
  decision (5%): not_rejected
  R.theta - r = [0.0731]
```

The covariance of `θ̂` comes from the delta method with the analytical
gradient — the same one behind the standard errors in `.longrun`, so the
test and the reported standard errors cannot disagree with each other.

`discrepancy` is returned **signed**, so you see the direction of the
violation and not merely its size.

The verdict is `not_rejected`, never `accept`. Failing to reject a
restriction is not evidence that it holds, and on the sample lengths
this literature works with the distinction matters.

### `impose=True`

Testing a restriction and adopting it are different questions. The
second is what DHSY actually did: having failed to reject unit
elasticity, they rewrote the model around the ratio, gaining a degree of
freedom and an error-correction term with a direct interpretation.

```python
out = res.test_longrun_restriction([[1.0]], 1.0, impose=True)
print(out.restricted_params)
```

The restricted model replaces the two level terms `y_{t-1}` and
`x_{t-1}` by the single regressor `(y - x)_{t-1}`, and an F test
compares the two residual sums of squares.

That F test is only meaningful because the unrestricted error-correction
design reproduces the ARDL regression **exactly** — same sample, same
residuals, same sum of squares. That identity is verified to 1e-10
across lag orders and deterministic cases, not assumed.

Imposition is supported for the homogeneity restriction `θ_j = 1` only.
Anything else raises an explicit error rather than quietly imposing
something different from what was tested: joint restrictions have no
ratio representation in the error-correction form.

## `pyardl.utils.diff(x, d=1, D=0, s=4)`

The operator `(1-L)^d (1-L^s)^D`.

```python
from pyardl.utils import diff

diff(consumption, d=1, D=1, s=4)
```

Seasonal differencing is not a convenience for removing a seasonal
pattern. On quarterly data `Δ₄ y_t = y_t - y_{t-4}` removes a fixed
seasonal pattern *and* a unit root at once, which is why DHSY built
their model on it. Combining the two, `d=1, D=1`, removes a seasonal
pattern that itself drifts.

Pass a `Series` and the result keeps the tail of its index, so it stays
attached to the dates it belongs to instead of silently shifting by
`d + D·s` periods — a mistake that is invisible until you merge two
frames.

## Seasonal dummies

```python
ARDL(y, x, order=(1, 1), seasonal=True, seasonal_periods=4)
```

`s-1` dummies are added when an intercept is present, `s` when
`det="none"`: the full set would be perfectly collinear with the
constant.

The season of each observation is taken from its position in the
**original** series, not in the estimation sample. Otherwise two models
with different lag orders would assign the same observation to different
quarters, and their seasonal coefficients would not be comparable.

## A note on what is not here

The spec for this module also calls for a UK consumption dataset, to
reproduce the original DHSY equation. It is **not shipped**. The data
come from a 1978 article behind an access barrier, and no freely
redistributable source was identified.

Fabricating a plausible "UK consumption" series would have been worse
than inventing a critical value: an invented table can be caught by
cross-checking, whereas an invented dataset carrying the name of a real
source looks authentic, can be checked against nothing, and would
contaminate every result built on it.
