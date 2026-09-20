# MIDAS-ARDL — mixed sampling frequencies

`pyardl.midas`

The rest of the library assumes `y` and `x` are observed at the **same**
frequency. In practice a quarterly `y` (GDP) often comes with a monthly
or daily `x` (equity indices, confidence surveys) — aggregating `x` down
to `y`'s frequency (a monthly-to-quarterly average) throws away
information and imposes equal weights on every observation within the
period, an assumption never tested. MIDAS (Ghysels, Santa-Clara &
Valkanov 2004) answers by keeping `x` at its native frequency and
modelling the weights linking its fine observations to each aggregated
`y` observation as a low-dimensional parametric function — the same idea
as the [Almon weights](distributed-lags.md) already in the library,
applied here to a frequency problem rather than a classical finite lag.

```python
from pyardl.midas import MIDASARDL, midas_weights

w = midas_weights(theta=[0.0, -0.15], k_max=12, form="almon_exp")

res = MIDASARDL(y, x_high_freq, freq_ratio=3, k_max=12,
                 form="almon_exp", order=(1, 1)).fit()
res.theta_hat      # fitted weight-function parameters
res.weights          # b(k; theta_hat), k = 0..k_max
res.longrun           # theta = -gamma/lambda, on the aggregated regressor
res.summary()
```

`x_high_freq` is the full high-frequency series (length `len(y) *
freq_ratio`), ordered so the last high-frequency observation within
low-frequency period `t` sits at `x_high_freq[(t+1)*freq_ratio - 1]`.

## Two weight forms, both implemented

- `"almon_exp"` (the default, and the form most used in applied
  macro-finance): `b(k) ∝ exp(theta_1*k + theta_2*k^2)` — always
  positive. A rigid special case of the polynomial Almon weights: the
  same shrinking-parameters idea, exponential rather than polynomial in
  `k`, because the exponential form is what the MIDAS literature
  standardised on rather than a re-derivation of the same trick under a
  new name.
- `"beta"` (Ghysels et al. 2004): weights built from a Beta density on
  `[0, 1]`, two shape parameters — a different, more flexible family.

## Estimation: concentrated, not a general nonlinear solver

For *fixed* weight parameters `theta`, the aggregated regressor `z_t =
sum_k b(k; theta) x_{t - k/m}` is just a number for every `t`, and the
outer UECM in `(y_{t-1}, z_{t-1}, ...)` is then ordinary linear least
squares. `MIDASARDL` exploits that — the same trick
[STAR-ARDL](threshold.md) uses for its own two nonlinear parameters:
search only over `theta` (two parameters, either weight form), seeded
from a coarse grid to avoid local optima, exactly as spec 40 §2.3
requires.

## Choosing `k_max`

`k_max` (the highest high-frequency lag included) is a modelling
choice, not selected automatically — too small truncates real
information, too large adds nothing once `b(k; theta_hat)` has already
decayed to near zero. Check the stability of `res.longrun` as `k_max`
grows rather than fixing an arbitrary value; the fitted weights
themselves (`res.weights`) are the most direct diagnostic of whether
`k_max` was generous enough.

## Downstream consistency

At `freq_ratio=1, k_max=0` the aggregation is trivial (`weights = [1.0]`,
`z_t = x_t`) and `MIDASARDL` coincides with a plain `ARDL(y, x,
order=(1, 1))` to machine precision — verified in
`tests/unit/midas/test_model.py::TestDownstreamConsistency`, the same
discipline as ARDL↔ECM (spec 03) and `partial_sums_multi` against
`partial_sums` (spec 28 §4).

## Limits

Weak identification of `theta` in small samples (the same
grid-plus-local-optimum risk STAR-ARDL documents). Out of scope: several
regressors at different frequencies simultaneously, U-MIDAS (free,
non-parametric weights — loses the whole point of MIDAS's parsimony),
and DFM-MIDAS (dynamic factors upstream, outside the single-equation
ARDL frame this library covers).
