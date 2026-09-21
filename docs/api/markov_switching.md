# Markov-Switching ARDL

`pyardl.markov_switching`

STAR (`pyardl.star`, spec 34) makes the regime depend on an
**observed** transition variable. `ms_ardl` makes it depend on an
unobserved state `S_t` following a first-order Markov chain — relevant
when the regime change is driven by a latent economic state
(recession/expansion, financial stress) with no single reliable
observed proxy, at the cost of a much heavier estimation.

```
Delta y_t = det(S_t) + lambda(S_t) y_{t-1} + gamma(S_t) x_{t-1} + Sigma(...) + eps_t
eps_t | S_t ~ N(0, sigma^2(S_t))
```

## Implementation strategy

Every other spec in this library reuses an OLS regression inside a
grid search or a light nonlinear optimisation on top of bricks already
validated elsewhere. This spec is the exception: the Hamilton filter,
the Kim (1994) smoother, and the EM loop it requires are a genuinely
new estimation engine. Rather than reimplement them, `ms_ardl` wraps
`statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`
— an already-validated implementation of exactly this engine, and
`statsmodels` is already a required runtime dependency of this
project. The UECM design matrix (deterministic terms, `y.L1`,
`x{j}.L1`, the short-run difference terms) is built with this
library's own naming conventions and handed to it as `exog`, every
coefficient switching by regime.

## `ms_ardl(y, x, order=(1, 1), n_states=2, det="const", method="em", n_starts=3, max_iter=1000, tol=1e-6, seed=None)`

```python
from pyardl.markov_switching import ms_ardl

res = ms_ardl(y, x, order=(1, 1), n_states=2, n_starts=5, seed=0)
```

| Argument | Meaning |
|---|---|
| `order` | `(p, q)`, `q` applied uniformly to every regressor |
| `n_states` | number of regimes `K`; a warning is raised above `K=3` (`K*(K-1)` transition probabilities, identification degrades fast) |
| `method` | `"em"` fits by **pure** EM to convergence; `"direct"` fits by BFGS from the default start, no EM at all |
| `n_starts` | restarts from perturbed starting values, keeps the highest likelihood — EM convergence is sensitive to the start on short series |

!!! note "`method='em'` is pure EM, not an EM warm-up"
    A short EM warm-up followed by direct BFGS maximisation of the
    likelihood was found to diverge on ordinary synthetic ARDL-scaled
    data during this module's own test development, while plain EM to
    convergence recovered the true regime parameters reliably. See
    `docs/DEVIATIONS.md`.

Restarts are drawn with an explicit `numpy.random.Generator` (CLAUDE.md
rule 2) rather than `statsmodels`' own `search_reps`, which draws from
the global NumPy random state.

## `MSARDLResults`

| Member | Returns |
|---|---|
| `transition_matrix` | `P[i, j] = P(S_t=j \| S_{t-1}=i)`, rows sum to 1 |
| `regime_params` | `{state: pandas.Series}`, one per regime, plus `sigma2` |
| `filtered_probs` / `smoothed_probs` | `P(S_t=j \| info up to t)` vs. `P(S_t=j \| all data, Kim 1994)` — never the same quantity |
| `expected_duration` | `1 / (1 - P[i, i])` per state |
| `llf` | maximised log-likelihood |

`.summary()` and `.plot_regimes()` (requires matplotlib) are also
available.

## Out of scope

Markov chains of order > 1, time-varying transition probabilities
(TVTP), and the regime-specific long-run ratio `theta(S_t) =
-gamma(S_t)/lambda(S_t)` — its inference would need the delta method
combined with the latent-regime uncertainty, not addressed here.

## References

Hamilton, J. D. (1989). A new approach to the economic analysis of
nonstationary time series and the business cycle. *Econometrica*,
57(2), 357-384.

Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
*Journal of Econometrics*, 60(1-2), 1-22.
