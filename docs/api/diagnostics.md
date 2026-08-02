# Stability diagnostics

`pyardl.diagnostics`

Fitting an ARDL over thirty years of data assumes the relationship did
not change over those thirty years. That assumption is rarely stated and
almost never checked — yet if it fails, the long-run coefficients are an
average of two regimes rather than an equilibrium, and the bounds test
conclusion is about a relationship that never existed in that form.

The two tests of Brown, Durbin & Evans (1975) check it. Both are run
automatically as part of `bounds_test(...).diagnostics()`.

## Recursive residuals

Everything is built on one idea. Estimate the model on the first `t-1`
observations, predict `y_t`, and standardise the prediction error:

```
w_t = (y_t - x_t' b_{t-1}) / sqrt(1 + x_t' (X'X)_{t-1}^{-1} x_t)
```

Under constant parameters these are i.i.d. normal, whatever the data
look like. A break makes them drift or change scale.

### `recursive_residuals(y, x)`

```python
from pyardl.diagnostics import recursive_residuals

w = recursive_residuals(y, design)   # T - k values
```

The recursion updates `(X'X)^{-1}` in place by the Sherman-Morrison
identity rather than re-estimating the model `T-k` times, so the cost is
linear in `T` instead of cubic. This is exact, not an approximation: it
matches a full re-estimation, and `statsmodels`, to machine precision.

## `cusum(y, x, alpha=0.05)`

Cumulates the residuals: `W_t = Σ w_s / σ̂`, compared with the lines

```
± [ a·sqrt(n) + 2a(t-k)/sqrt(n) ],    n = T - k
```

The bands widen with `t` on purpose — a random walk spreads as `sqrt(t)`,
so constant-width bands would reject far too often late in the sample.

Because it cumulates the residuals themselves, this test reacts to a
shift in the **mean**: a jump in the intercept, or a slow drift.

## `cusumsq(y, x, alpha=0.05)`

Cumulates their squares: `S_t = Σ w_s² / Σ w_s²`, running from 0 to 1 by
construction, compared with `(t-k)/(T-k) ± c₀`.

Because it cumulates squares, it reacts to a change in **variance**.

## The two tests are not interchangeable

This is the part worth internalising. A break in the **slope** on a
zero-mean regressor leaves the recursive residuals centred on zero: the
CUSUM path stays flat and the test reports stability, while the inflated
variance pushes the CUSUM of squares straight out of its band.

In our test suite, on 20 simulated samples with exactly that break, the
CUSUM concluded "stable" 20 times out of 20 and the CUSUM of squares
detected it 20 times out of 20.

Reporting only the CUSUM — as much applied work does — leaves an entire
family of common instabilities untested. `pyardl` therefore always
produces both.

## `stability_tests(y, x, alpha=0.05)`

Both tests in one table.

```python
from pyardl.diagnostics import stability_tests

print(stability_tests(y, design))
```

```text
                  stable  max_excess  first_crossing
test
CUSUM               True         0.0             NaN
CUSUM-of-squares    True         0.0             NaN
```

`max_excess` is the largest excursion beyond the band, zero when stable
— it says *how far* from stability the model is, not merely whether it
crossed. `first_crossing` locates *when* the break happened, which a
boolean cannot.

## On the results object

Both `ARDLResults` and `BoundsTestResults` expose `.stability(alpha)`.
For the bounds test the two tests also appear in `.diagnostics()`, as
rows `CUSUM(5%) excess` and `CUSUMSQ(5%) excess`.

Those rows carry **no p-value**, and the column is `NaN` rather than a
plausible-looking number. These are boundary-crossing procedures, not
statistics with a null distribution to integrate; inventing a p-value
would misrepresent what the test does.

## Plots

```python
from pyardl.diagnostics import cusum, plot_cusum

fig = plot_cusum(cusum(y, design))
```

`plot_cusum` and `plot_cusumsq` draw the two canonical graphs, bands
included. They require matplotlib, an optional dependency; a clear
`ImportError` is raised if it is missing.

## Significance levels

Only 10%, 5% and 1% exist. Brown, Durbin & Evans tabulated three values
of `a`, and the CUSUMSQ table is simulated at the same three levels. Any
other value raises an error rather than interpolating something that has
no meaning.

## Where the critical values come from

The CUSUM coefficient `a` is published (0.850 / 0.948 / 1.143). The
boundary sequence it produces was cross-checked against `statsmodels`
and agrees **exactly**.

The CUSUMSQ half-width `c₀` is simulated rather than transcribed. That
is not a fallback: the statistic is genuinely distribution-free, so its
law can be computed to any precision. It was cross-checked against its
own asymptotic limit (the Kolmogorov distribution), which it approaches
monotonically from below — meaning the fallback used beyond the
simulated grid widens the band and makes the test conservative, never
liberal.

Full details, including a documented convention difference with
`statsmodels` about where the recursion starts, are in
[`PROVENANCE.md`](https://github.com/Matth-analyst/pyardl/blob/main/src/pyardl/critical_values/PROVENANCE.md).
