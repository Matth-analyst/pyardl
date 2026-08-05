# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/1.1.0/).
This project follows [semantic versioning](https://semver.org/).

## [0.2.0] — 2026-08-05

Second release. Modern critical values with p-values, parameter-stability
diagnostics, unit-root pre-tests, long-run restriction testing, and the
Engle-Granger test for comparison.

### Added — stability diagnostics (`pyardl.diagnostics`)

- `recursive_residuals` — standardised one-step-ahead prediction errors,
  computed by Sherman-Morrison rank-one updates rather than `T-k` full
  re-estimations. Exact, and verified against `statsmodels` to 1e-10.
- `cusum` and `cusumsq` — the two parameter-constancy tests of Brown,
  Durbin & Evans (1975). Each returns the path, its boundaries, the
  verdict, the largest excursion beyond the band, and the positions
  where the path leaves it, so a break can be located and not merely
  detected.
- `stability_tests` — both tests in one table.
- `plot_cusum` / `plot_cusumsq` — the two canonical graphs, bands
  included. Requires matplotlib, an optional dependency.
- `BoundsTestResults.stability()` and `ARDLResults.stability()`.
- `bounds_test(...).diagnostics()` now reports both stability tests
  alongside the residual diagnostics. The rows are added, not
  substituted, and carry no p-value: they are boundary-crossing
  procedures, so the column is `NaN` rather than an invented number.

### Added — Engle-Granger test (`pyardl.cointegration`)

- `engle_granger(y, x, trend, max_lags, ic, fit_ecm)` — the two-step
  procedure, with the second-step error-correction model on request.
  Statistic and p-value agree with `statsmodels.tsa.stattools.coint` to
  1e-13 across deterministic cases, numbers of regressors and sample
  sizes.
- First-step coefficients are reported as point estimates with **no
  standard errors**: they are super-consistent but non-standard, so the
  usual ones would be wrong. The summary says so.
- `critical_values.mackinnon` — response surfaces of MacKinnon (1994,
  2010), which correct for the fact that the statistic is computed on
  estimated residuals. Under `trend='n'`, where no surfaces were
  published, values are `NaN` with a warning and `decision()` raises
  rather than deciding.
- The surfaces are cross-checked against an independent in-house
  simulation of the null: 54 cells, all within three standard errors of
  the simulated quantile.
- Two rounding conventions differ from `statsmodels` and are documented
  in `PROVENANCE.md`: the Schwert rule for the maximum lag (we round
  down, as published; they round up), and the sample size at which the
  surface is evaluated. The first can flip a near-tie in the AIC and
  move the statistic from -14.11 to -7.82, which is why the concordance
  tests pass `max_lags` explicitly on both sides.

### Added — long-run restrictions and seasonality (`pyardl.core.restrictions`)

- `ARDLResults.test_longrun_restriction(R, r, impose=False)` — Wald test
  of `R theta = r` on the long-run coefficients, using the same
  delta-method covariance as the standard errors in `.longrun`, so the
  two cannot disagree. The discrepancy `R theta - r` is returned signed.
- `impose=True` re-estimates the error-correction model with the
  homogeneity restriction `theta_j = 1` applied — the level term becomes
  the ratio `(y - x_j)` — and reports the regression F test. The
  unrestricted design reproduces the ARDL regression exactly (residuals
  to 1e-10), which is what makes that F test legitimate; a test verifies
  it across lag orders and deterministic cases. Any other restriction
  raises rather than imposing something different from what was tested.
- `utils.diff(x, d, D, s)` — the operator `(1-L)^d (1-L^s)^D`. A Series
  keeps the tail of its index, so a differenced series stays attached to
  its dates instead of silently shifting.
- `ARDL(..., seasonal=True, seasonal_periods=4)` — seasonal dummies,
  `s-1` of them when an intercept is present. The season of each
  observation is read from its position in the original series, so
  `hold_back` cannot relabel the quarters.
- The verdict is `not_rejected`, never `accept`.

### Decided — the DHSY method travels, the DHSY data do not

The walkthrough for this feature is built on the Danish money-demand
data already shipped, not on UK consumption. That is a deliberate
choice, settled and closed, not a gap waiting to be filled.

The original DHSY series come from a 1978 article behind an access
barrier, and no freely redistributable version exists. Reconstructing
something similar from current ONS releases was considered and rejected:
those are revised vintages on a different period, so the result would
not have been the DHSY data either — it would have carried the same
caveat while adding reconstruction choices that can be checked against
nothing.

Nothing is lost methodologically. The Danish data make the same point
with the same structure: the long-run income elasticity of money demand
is 0.9965, the homogeneity restriction is not rejected (Wald 0.0008,
p = 0.977), and imposing it turns the level term into `(LRM - LRY)` —
the velocity of money, a ratio theory expects to be stationary, exactly
as `log(C/Y)` is in the original paper. See
[Testing an economic restriction](docs/long-run-restrictions.md), where
DHSY is credited as the source of the *method* and never of the data.

### Added — unit-root pre-tests (`pyardl.unitroot`)

- `dfgls` — the DF-GLS test of Elliott, Rothenberg & Stock (1996).
  Verified against `arch` to 1e-8 across sample sizes, trends and lag
  orders.
- `ng_perron` — the four M statistics (MZa, MZt, MSB, MPT), sharing an
  autoregressive long-run variance. All are lower-tail, so there is no
  direction to get wrong.
- `gls_detrend`, `ols_detrend`, `adf_regression`, `select_lags` —
  the shared machinery, exposed rather than hidden. `select_lags`
  implements MAIC and MBIC alongside AIC, BIC and the sequential t
  rule, compares every candidate on a common sample, and returns the
  criterion value at each order so the choice can be inspected.
- `report` / `integration_order` — sequential level-then-difference
  screening, classifying each series as I(0), I(1) or I(2)-suspect. A
  `PyardlMethodologyWarning` fires on any I(2) suspicion: the bounds
  test is invalid on I(2) data and cannot detect it itself.

### Added — critical values

- `critical_values.ers1996.dfgls_critical_values` and
  `critical_values.ngperron2001.m_critical_values`, simulated in-house
  over `T = 25..2000` for both deterministic cases, interpolated in
  `1/T`.
- The two families were verified differently, because their
  availability differs. DF-GLS has a second source (`arch`) and agrees
  with it within Monte Carlo error for `T >= 100`. The M statistics
  have **no second implementation anywhere**, so verification is
  internal: `MZt` shares the limiting distribution of DF-GLS, and the
  two independently simulated tables converge as `T` grows — a gap of
  0.155 at `T = 100` down to 0.006 at `T = 2000`.

### Notes

- The first observation is not quasi-differenced during GLS detrending.
  Getting this wrong moves the statistic by 45% on a random walk; the
  convention is locked by a test.
- Lag selection runs on the OLS-detrended series, not the GLS-detrended
  one. Selecting on the latter makes every criterion over-select and
  costs the test half its power — measured, and corroborated by
  `arch`'s own implementation.
- The two screening entry points, `report` and `integration_order`,
  default to BIC; the two targeted tests, `dfgls` and `ng_perron`,
  default to MAIC. MAIC over-selects on stationary data, which costs
  classification accuracy when screening (29/40 against 40/40 for BIC
  on I(0) series), but it is what protects against a negative
  moving-average component once a series is being tested deliberately.
  The measured trade-off appears in `help(report)`, not only in the
  documentation.

### Added — critical values (`pyardl.critical_values.bde1975`)

- `cusum_a` — the published boundary coefficients (0.850 / 0.948 /
  1.143). The resulting boundary sequence agrees exactly with
  `statsmodels`.
- `cusumsq_c0` — half-width of the CUSUM-of-squares band, from a
  simulated table covering `n = 4..1000`, interpolated in `1/sqrt(n)`.
  The statistic is distribution-free, so the table is computed rather
  than transcribed; it was cross-checked against its own asymptotic
  Kolmogorov limit, which it approaches monotonically from below.
  Beyond the grid, that asymptotic approximation is used with an
  explicit warning — it widens the band, making the test conservative.

### Notes

- Only the 10%, 5% and 1% levels exist for either test. Any other level
  raises an error instead of interpolating a value with no meaning.
- The CUSUM and the CUSUM of squares are not two views of the same
  thing: a slope break on a zero-mean regressor is invisible to the
  first and obvious to the second. Both are therefore always reported.
- Convention difference with `statsmodels`, documented in
  `PROVENANCE.md`: the recursion starts at `t = k+1` as in the original
  article, not at `t = k` where the recursive residual is zero by
  construction.

## [0.1.0] — 2026-07-25

First public release. Covers ARDL estimation, the exact error-correction
reparameterisation, the Pesaran-Shin-Smith bounds test with its five
deterministic cases, and three sources of critical values.

### ARDL estimation — `pyardl.core.ardl`

- `ARDL(y, x, order, det=..., fixed_regressors=..., hold_back=...)`,
  estimated by least squares through QR. Robust covariances:
  `HC0`-`HC3` and `HAC` (Newey-West).
- `q_j = 0` supported — the regressor enters contemporaneously with no
  dynamics of its own.
- `ARDL.select_order` — grid or per-variable search over AIC, BIC and
  HQ. All candidates are estimated on a common sample, so the criteria
  are comparable; the winner is then re-estimated on the largest sample
  its own order allows.
- `ARDL.gets` — general-to-specific reduction restricted to terminal
  lags, guarded by residual diagnostics and an F test against the
  general model. The full `reduction_path` is returned.
- Every fit runs a Ljung-Box test on the residuals and warns when it
  rejects.
- `ARDLResults` exposes `to_ecm()`, `longrun`, `adjustment`,
  `ar_roots`, `is_stable`, `diagnostics()` and `summary()`.

### ARDL ↔ ECM algebra — `pyardl.core.transforms`

- `ardl_to_ecm` / `ecm_to_ardl`, exact in both directions: the two
  representations share the same residuals to machine precision.
- `longrun_coefs`, `longrun_covariance` (delta method with analytical
  gradient), `speed_of_adjustment`, `half_life`.
- Degenerate configurations return NaN with a `DegenerateCaseWarning`
  rather than a number produced by dividing by something near zero.

### Bounds test — `pyardl.bounds`

- `bounds_test` on the unrestricted error-correction form, all five
  deterministic cases. Under cases 2 and 4 the restricted deterministic
  term is part of the tested vector.
- `F_overall` and `t_BDM`. The t test is left-tailed and requires a
  negative adjustment estimate.
- Three-state decisions — `cointegration`, `no_cointegration`,
  `inconclusive` — never a boolean.
- `decision_joint` requires both statistics to agree, and reports
  `degenerate_suspicion` when the F test rejects but the t test does
  not.
- `adjustment(alpha)` returns a confidence interval on λ only once
  cointegration is established; otherwise NaN with a warning, since the
  distribution is non-standard under the null.
- `fixed_regressors` for dummies and other unlagged variables, kept out
  of the tested vector.
- `diagnostics()` and a publication-style `summary()` that displays the
  p-value interval when the verdict is inconclusive.

### Critical values — `pyardl.critical_values`

- `get_bounds(stat, case, k, alpha, cv_source, t_obs)` dispatching over
  three sources:
  - `"kripfganz"` (default) — response surfaces, any significance
    level, with p-values via `pvalue_bounds`;
  - `"pss"` — the published tables reproduced exactly, for matching
    printed results;
  - `"narayan"` — small-sample bounds for `30 ≤ T ≤ 80`, linearly
    interpolated in `T`.
- Any combination a source does not cover raises an explicit error
  naming a source that does. Nothing is silently substituted.
- `simulate_bounds` — Monte Carlo engine for configurations no table
  covers. All parameters, including the batch size, are recorded on the
  result, so a run reproduces exactly.
- Every shipped table documents its source and cross-check in
  `src/pyardl/critical_values/PROVENANCE.md`.

### Datasets — `pyardl.datasets`

- `load_denmark()` — Danish money demand.
- `load_pss2001()` — the UK wage-price data of Pesaran, Shin & Smith
  (2001).

### Validation

- The UK wage equation of Pesaran, Shin & Smith (2001) is reproduced:
  F statistics under cases 4 and 5, the case-5 t statistic, and the
  error-correction coefficients.
- Coefficients, standard errors and residuals agree with `statsmodels`
  to 1e-10, and with the R package `ARDL` to 1e-6.
- The published critical-value tables were checked cell by cell against
  the internal simulation engine, using a tolerance derived from the
  Monte Carlo standard error of each quantile.
- Monte Carlo size and power experiments are run as part of the test
  suite.

### Known limitations

- `bounds_test(finite_t=True)` is experimental and unvalidated. It
  requires external material that is not yet available, and should not
  be used.
- The t bounds are asymptotic in every source; only the F bounds have a
  small-sample variant.
- Weak exogeneity of the regressors, absence of cointegration among
  them, and the absence of I(2) series are assumptions of the test and
  are **not** checked automatically. Residual autocorrelation is.
