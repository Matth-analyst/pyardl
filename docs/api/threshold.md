# Threshold regression — an estimated regime split

`pyardl.threshold`

NARDL ([nardl.md](nardl.md)) decomposes a regressor around a threshold
**fixed at zero a priori**, on the regressor's *own change*. Hansen
(1999, 2000) answers a different question: what regime split actually
fits the data — a threshold on a **transition variable** `q_t` (which
can be `x` itself, a third series, or `y_{t-1}` — self-exciting TAR in
the strict sense), **estimated** rather than chosen, and applied to a
threshold, not a change.

```python
from pyardl.threshold import threshold_ardl

res = threshold_ardl(y, x, transition=q, delay=1, trim=0.15, n_boot=999, seed=0)
print(res.summary())
res.gamma_hat            # estimated threshold
res.gamma_ci               # asymptotic no-rejection region (Hansen 2000)
res.regime_params           # beta_1 (q_{t-d} <= gamma_hat), beta_2 (> gamma_hat)
res.linearity_stat            # H0: beta_1 = beta_2 (no threshold effect)
res.grid                       # (gamma, ssr) for every candidate — the margin
                                 # of the retained threshold, not just the minimum
res.decision(0.05)               # 'threshold' or 'linear'
```

`x` gets its own validation (`pyardl.utils.check_regressors_allow_constant`)
rather than the usual `check_series` — a constant regressor is a normal
choice here, not a degenerate one (the model is `y_t = x_t'β_j + ε_t`
per regime, and `x` commonly needs its own intercept). Include any
constant/trend column explicitly; nothing is added for you.

## Two regimes, split on `q_{t-d}`

```
y_t = x_t'β1 + ε_t   if q_{t-d} <= gamma
y_t = x_t'β2 + ε_t   if q_{t-d} >  gamma
```

`gamma` is estimated by grid search over the candidate values of `q`
within `trim`/`1-trim` empirical quantiles (never the whole support —
the trim is part of identification, same discipline as
[Gregory-Hansen](cointegration.md#gregory-hansen-cointegration-with-an-unknown-regime-shift)
and [Bai-Perron](cointegration.md#bai-perron-multiple-structural-breaks)),
minimising the pooled residual sum of squares `SSR(gamma)`.

## Linearity test: bootstrap only, and why

The statistic testing `H0: beta1 = beta2` has no standard asymptotic
distribution — `gamma` is only identified under the alternative (the
"problem of Davies", the same family as Gregory-Hansen's break search
and the Fourier frequency pre-test). Its p-value comes from a
**fixed-design residual bootstrap** (Hansen 1996): fit the restricted
linear model, resample its residuals with the design (`x`, `q`) held
fixed, and rerun the same grid search on each replicate.

This is a genuinely different null construction than
[Gregory-Hansen](cointegration.md) or
[Enders-Siklos](cointegration.md#enders-siklos-asymmetric-tar--m-tar-adjustment)
use, and deliberately so: those test **cointegration** between I(1)
series, where the null (independent random walks) requires regenerating
the primitive series and rerunning the whole procedure — resampling a
derived residual's own differences in isolation there measurably
under-sizes the test (a bug caught and fixed in `enders_siklos`, see
`docs/QUESTIONS.md`). Hansen's threshold regression assumes **stationary**
regressors from the start; holding the design fixed and resampling
residuals is the textbook-correct bootstrap for that setting, not a
shortcut.

## What `gamma_ci` is, and its verification status

`gamma_ci` is Hansen's (2000) likelihood-ratio-based no-rejection
region — built from the shape of `SSR(gamma)` near its minimum, not a
symmetric interval around a point estimate. The width constant is
derived from the known limiting distribution of the LR sequence but has
not been checked against the original paper in this environment; see
`docs/QUESTIONS.md`. `gamma_hat` itself does not depend on this constant
and has been verified exactly against R `pdR::SMPLSplit_het` (B. E.
Hansen's own code for the 2000 paper).

## Distinct from NARDL, Gregory-Hansen and Bai-Perron

Three different models share the "grid search + problem of Davies"
family without being substitutable:

| | Threshold on | Searched? |
|---|---|---|
| NARDL | `Δx_t`'s own sign | No — fixed at 0 |
| Threshold regression (this module) | a transition variable `q_{t-d}` | Yes — `gamma_hat` |
| Gregory-Hansen / Bai-Perron | a *date* | Yes — break fraction(s) |

Comparing a Threshold-ARDL fit against NARDL on the same data (e.g.
`transition=x`, forcing `gamma=0`) is a reasonable sanity check, but the
two do not coincide exactly: Threshold-ARDL lets the *entire* dynamic
change regime, while NARDL only decomposes the level of `x`.

## Limits

Two regimes only — multiple thresholds exist in the literature (Hansen
1999 §5, Gonzalo & Pitarakis 2002) but are out of scope here. The delay
`d` of the transition variable is assumed known (chosen by the caller,
not searched over). The bootstrap reruns a full grid search per
replicate — measure a typical runtime before raising `n_boot` far past
the default on a large sample.

## STAR-ARDL: a smooth transition instead of a sharp one

`threshold_ardl` (above) switches regime all at once, exactly at
`gamma_hat`. STAR-ARDL (Teräsvirta 1994) replaces the step indicator
with a **continuous** transition function — economically more plausible
for aggregate phenomena — at the cost of an extra slope parameter
(`gamma`, how fast the transition happens) that is genuinely harder to
pin down than a sharp threshold.

```python
from pyardl.threshold import star_ardl

res = star_ardl(y, x, transition_var=q, form="lstar", order=(1, 1))
res.gamma_hat, res.c_hat     # transition speed and centre
res.linearity_pvalue          # Terasvirta LM test — a standard asymptotic
                                # test here, no bootstrap needed (see below)
res.longrun_at([-1, 0, 1])      # theta(q) evaluated on a grid — the
                                  # signature STAR output, a curve rather
                                  # than two regime numbers
```

Two forms, both implemented — pick the one that matches the economics,
or let `form="auto"` decide via an approximation of Teräsvirta's (1994)
nested-test sequence on the linearity test's own auxiliary regression:

- `"lstar"` (logistic): `G(q) = 1 / (1 + exp(-gamma*(q - c)))` — an
  asymmetric regime change, one side is fundamentally different from
  the other.
- `"estar"` (exponential): `G(q) = 1 - exp(-gamma*(q - c)^2)` —
  symmetric around `c`, the middle regime differs from both tails
  equally.

### Estimation: concentrated, not a general nonlinear solver

For a *fixed* `(gamma, c)` the model is linear in everything else — an
ordinary least-squares fit on `[y_{t-1}, G*y_{t-1}, x_{t-1},
G*x_{t-1}, ...]`. `star_ardl` exploits that: it searches only over the
two genuinely nonlinear parameters, starting from the best of a coarse
`start_grid × start_grid` search to avoid the local optima a single
gradient start would risk, then refines with Nelder-Mead on the
profiled SSR.

### Why the linearity test needs no bootstrap here

Both `threshold_ardl` and Enders-Siklos need a bootstrap because their
threshold/break is unidentified under the null (the "problem of
Davies"). Teräsvirta's trick avoids it: a third-order Taylor expansion
of `G` around `gamma=0` removes `gamma` from the null altogether, so
the resulting LM test has a standard asymptotic F distribution — a
genuine methodological difference from the rest of this module, not an
oversight.

### Limits

Weak identification of `gamma` and `c` *separately* is common in small
samples when the transition is nearly sharp or nearly absent — a
property of the STAR literature generally, not specific to this
implementation. Report both together with the linearity test rather
than trusting a lone point estimate. The `form="auto"` decision rule
approximates Teräsvirta's nested F-test sequence rather than
reproducing its exact construction; see `docs/DEVIATIONS.md`.
