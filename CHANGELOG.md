# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/1.1.0/).
This project follows [semantic versioning](https://semver.org/).

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
