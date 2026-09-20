# Bayesian ARDL (Minnesota prior)

`pyardl.bayesian`

Every other estimator in the library is frequentist: coefficients are
point estimates, uncertainty comes from the asymptotic normal law or
from bootstrap resampling (specs 14/16). `BayesianARDL` complements
that with two things a frequentist fit does not give directly:

- a **prior** that distant short-run lags matter less than recent ones
  (the Minnesota prior, Litterman 1986, designed for BVARs and adapted
  here to the single-equation UECM);
- a **posterior distribution** of the long-run coefficient
  `theta = -gamma/lambda`, obtained by direct simulation (every
  posterior draw of `(gamma, lambda)` gives a draw of `theta`) rather
  than the delta method's first-order approximation
  (`ARDLResults.longrun`, spec 03) — useful in particular when `lambda`
  is weakly identified, a case where the delta method is known to
  degrade.

## `BayesianARDL(y, x, order, det="const", prior="minnesota", tau="cv", decay=1.0, n_draws=5000, seed=None)`

```python
from pyardl.bayesian import BayesianARDL

res = BayesianARDL(y, x, order=(2, 2), tau="cv", n_draws=5000, seed=0).fit()
```

| Argument | Meaning |
|---|---|
| `order` | `(p, q)`, `q` applied uniformly to every regressor — same convention as `select_order_regularized` |
| `tau` | prior scale: a float, or `"cv"` (rolling-origin forecast error) / `"evidence"` (marginal likelihood) to select it from a grid |
| `decay` | Minnesota decay rate `d`: informative-term prior variance is `tau^2 / lag^d` |
| `n_draws` | posterior draws |
| `seed` | seeds `numpy.random.Generator` |

**Prior scope, deliberately narrow.** Only the short-run terms
(`D.y.Li`, `D.x{j}.Li`) carry the informative Minnesota prior. The
deterministic terms, the error-correction coefficient `lambda`
(`y.L1`) and the long-run levels `x{j}.L1` get a diffuse (large fixed
variance) prior — the same non-penalisation discipline as
`pyardl.regularized` (spec 41), for the same reason: a short-run
shrinkage choice must never bias the long-run reading.

!!! note "Deviation from classical Minnesota"
    The classical Minnesota prior centers a VAR's own first lag at 1 (a
    random-walk prior). Here `lambda` already carries all of the
    persistence (diffuse, never shrunk), and `D.y.Li` represents
    short-run noise around the error-correction mechanism rather than
    the level's own persistence — every informative coefficient is
    centered at prior mean zero instead. See `docs/DEVIATIONS.md`.

### Estimation

Closed-form: a conjugate normal-inverse-gamma posterior under a
Gaussian likelihood and independent normal priors per coefficient — no
MCMC. The posterior mean is obtained by an **augmented least-squares**
solve (the prior precision enters as extra pseudo-rows, then a single
`numpy.linalg.lstsq` call), not by explicitly inverting `X'X`; the
small `k x k` triangular factor from that same QR decomposition is
reused (not inverted) to draw posterior samples.

## `BayesianARDLResults`

| Member | Returns |
|---|---|
| `posterior_params` | `n_draws x k` `DataFrame`, one column per UECM design term |
| `posterior_mean` / `posterior_std` | per-term posterior summary |
| `longrun_posterior` | `n_draws x n_regressors` `DataFrame` of `theta^(s) = -gamma^(s)/lambda^(s)` |
| `tau`, `decay` | hyperparameters used (the selected value when `tau` was `"cv"`/`"evidence"`) |
| `tau_grid`, `tau_criterion` | the searched grid and its CV error / negative log evidence, or `None` for a fixed `tau` |

### `.credible_interval(alpha=0.05)`

Empirical `(alpha/2, 1-alpha/2)` quantiles for every design term and
every long-run `theta`.

### `.compare_to_delta_method(alpha=0.05)`

Bayesian credible interval vs. the frequentist delta-method interval
(`ARDL(...).fit().longrun`), side by side. The two need not agree — a
diagnostic, not a claim that either is "more correct"; documented to
degrade in particular when `lambda` is weakly identified.

### `.summary()`

Publication-style report: posterior mean/std per term, and per
long-run `theta`.

## Limits (read before presenting a result)

The choice of prior (`tau`, `decay`) influences small-sample results —
never present a Bayesian result without a sensitivity check across at
least two `tau` values, standard practice in applied Bayesian work. The
diffuse prior on `lambda` and the levels means this module adds nothing
specifically Bayesian to the long-run reading beyond the posterior
simulation of §2.3 above — documented honestly rather than oversold.

## References

Litterman, R. B. (1986). Forecasting with Bayesian vector
autoregressions: five years of experience. *Journal of Business &
Economic Statistics*, 4(1), 25-38.

Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and conditional
projection using realistic prior distributions. *Econometric Reviews*,
3(1), 1-100.
