# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/1.1.0/).
This project follows [semantic versioning](https://semver.org/).

## [Unreleased]

### Added — Unified analysis (`pyardl.unified`)

- `cointegration_analysis` — one entry point for the eight
  configurations of the ARDL matrix: linear or asymmetric, with or
  without Fourier terms, tabulated/simulated bounds or joint-null
  bootstrap. The module owns no estimator of its own; each cell
  delegates to the brick validated for it.
- `resolve_critical_values` — the rule that justifies the module,
  centralised and testable: which critical-value source each cell
  requires, and a one-sentence reason quotable in a methods section.
  The one combination with no validated non-bootstrap source
  (asymmetric decomposition *and* Fourier terms under
  `inference='bounds'`) **raises** rather than borrowing a neighbouring
  table.
- `UnifiedResults.compare()` — the robustness table applied papers
  publish: the same relationship across sibling cells, with unavailable
  cells reported rather than silently dropped.
- The bootstrap route is one engine for every cell. It is pinned to the
  brick it generalises: with the decomposition and Fourier terms off it
  reproduces `bootstrap_bounds_test` to machine precision on the same
  seed (largest difference 1.4e-14; all nine critical values identical).

### Added — Fourier terms for smooth structural change (`pyardl.fourier`)

- `fourier_terms`, `select_frequency`, `fourier_orthogonality` — the
  deterministic building block of Becker, Enders & Lee (2006), plus a
  way to see how far a fractional frequency is from orthogonal to the
  intercept (at `f = 0.5` on 200 observations, the sine sums to 127
  rather than 0, so the two share the same variation).
- `fourier_f_test` — is a smooth deterministic component present at all?
- `fourier_kpss` — stationarity *around* such a component, the pre-test
  that tells a drifting mean apart from a unit root.
- `fourier_bounds_test` — the Fourier-ADL cointegration test of
  Banerjee, Arčabić & Lee (2017): the UECM augmented with the sinusoids,
  tested by the left-tailed `t` on the error-correction coefficient. All
  five deterministic cases; critical values simulated with the frequency
  search inside the loop; `selection` exposes the whole grid; a pre-test
  says whether the two extra parameters are buying anything at all.

### Measured

- **The Fourier-ADL does not recover the power its authors claim for
  it.** Both tests are correctly sized first (5.0% for the plain bounds
  test, 3.5% for the Fourier-ADL under two independent random walks, 200
  replications), so the comparison is fair. Under true cointegration
  with a smooth logistic break the Fourier-ADL is *behind*: 93 vs 99%
  with no break, 77 vs 91% at amplitude 3, 59 vs 59% at amplitude 6
  (150 replications, standard error 3.7 points). A four-configuration
  scan found every difference within ±5 points. The library ships the
  test and documents the measurement rather than the claim. OBS-17.
- **A pre-test read against the wrong null answered yes every time.**
  The Fourier relevance pre-test initially called `fourier_f_test` on
  `y`. That test simulates its null on white noise while `y` is
  integrated, so it declared the terms significant in 100% of
  replications, break or no break. It now compares two fits of the same
  model against the null simulated in the same loop: 0% significant with
  no break, 13% at amplitude 3, 35% at amplitude 6. OBS-18.
- **A frequency chosen on the data multiplies the size by five.** Under
  the null the frequency is not identified, so picking the best of a
  grid is a search, not an estimate. Measured on 2000 replications: 4.8%
  rejection with the frequency fixed in advance, **24.6%** with it
  selected, against a nominal 5%. The correct 95% quantile is 5.05 where
  the tabulated F gives 3.04. Both tests therefore simulate their own
  critical values with the search inside the loop. OBS-15.
- **What a Fourier component absorbs, and what it leaves.** The R² of a
  single frequency on a logistic break tops out at 0.86, not the 0.9 the
  literature suggests, and the unexplained remainder is persistent
  enough that a Fourier KPSS still rejects stationarity. Against the
  plain KPSS, though, the statistic falls by 66% to 83%. The unit test
  checks that reduction — which is true — rather than a non-rejection
  that is not. OBS-16.

### Added — QARDL, a relation that can differ across the distribution (`pyardl.qardl`)

- `QARDL(y, x, order, taus, asym, case).fit(inference, n_boot, seed)` —
  the framework of Cho, Kim & Shin (2015). Every parameter, the
  adjustment speed included, becomes a function of the quantile. The
  design comes from the same builder as the classical bounds test, so a
  QARDL at the median and an ARDL are one specification read under two
  losses.
- `res.longrun()`, `res.wald_constancy()` (the signature test: is the
  long run the same at every quantile?), `res.symmetry_test()`,
  `res.cointegration_test(tau)` with bootstrap critical values,
  `res.plot_coefficients()`, `res.summary()`.
- `asym=[...]` composes with the partial-sum decomposition of
  `pyardl.nardl` — the QNARDL, where the response may differ both
  between rises and falls and across the distribution.
- Two inference routes, because they answer different questions: a
  moving-block bootstrap giving the **joint** law across quantiles
  (required by the constancy and symmetry tests), and the per-quantile
  kernel estimator. The joint tests refuse to run on the latter rather
  than quietly assuming independence between quantiles.
- `λ(τ)` indistinguishable from zero gives `NaN` and a warning, never a
  very large number: the long run is a ratio with it in the denominator.

### Measured

- **statsmodels' quantile regression does not reach the optimum at its
  default tolerance.** Quantile regression is a linear program, so the
  optimum is exact and any estimate can be scored on the check loss.
  Scored that way, the default `p_tol=1e-6` misses the loss by up to
  3.4e-03 and the coefficients by up to 2.6e-02, silently. Raising
  `max_iter` alone changes nothing; it is the tolerance. The library runs
  at `p_tol=1e-10`, and every estimate is checked against the
  linear-programming optimum in the test suite.
- **Row-resampling breaks integrated regressors.** The constancy test
  first shipped with a moving-block bootstrap over the rows of the
  design, and rejected 0.5% of the time at a nominal 5% under a
  homogeneous null. The expected culprit — a noisy covariance from too
  few draws — was refuted: quadrupling the draws moved it only to 1.0%.
  The bootstrap spread is 1.36x the true sampling spread, because
  shuffling blocks of an I(1) regressor destroys the stochastic trend
  that defines it. The tests now draw under the **null** with the design
  held fixed, which brings the rate to 3.0%. That is better, not proven
  correct: at 200 replications 3.0% sits 1.3 standard errors from 5%.
  OBS-14 records the whole arc, hypothesis included.
- When the solver stops on its iteration cap, the warning is earned
  rather than assumed: the exact optimum is computed and the alert raised
  only if the gap is real, in which case the exact solution is returned.
  Measured on nearly collinear designs the cap fires on about one fit in
  a hundred, with the estimate at the optimum anyway.

## [0.4.0] — 2026-08-22

Fourth release. Asymmetric ARDL: responses that differ between rises and
falls, the tests that decide whether the asymmetry is real, and the
figure the literature reports.

### Added — NARDL, asymmetric responses (`pyardl.nardl`)

- `NARDL(y, x, asym=[...], order=..., case=..., threshold=...)` and
  `.fit()` — the framework of Shin, Yu & Greenwood-Nimmo (2014). A
  regressor is split into cumulated rises and falls, and the model is
  then estimated as a linear ARDL, so every existing brick applies to it
  unchanged.
- `partial_sums(x, threshold)` and `decomposition_error(...)`, the
  numerical core and the identity that locks it: `x = x₀ + x⁺ + x⁻`,
  verified to 1e-12 before anything else in the module. A non-zero
  threshold turns it into `x = x₀ + x⁺ + x⁻ + c·t` and warns, because the
  drift then moves into the deterministic part of the model.
- `order="auto"` selects the lag orders by information criterion on the
  transformed model. `asym_lags="paired"` (default) keeps the two sides
  of a decomposed variable on the same order — which halves the grid and
  keeps the strong short-run test meaningful — and `"free"` lets them
  differ. Every candidate is estimated on the same sample, and one the
  sample cannot support is skipped rather than scored as if it had lost
  on merit.
- `res.longrun_asym` — `θ⁺`, `θ⁻` and their difference, with delta-method
  standard errors taken on `(γ⁺, γ⁻, λ)` **jointly**: they come from one
  regression and are correlated at 0.93 to 0.99, so treating them as
  independent would misstate the uncertainty on exactly the quantity the
  model is about.
- `res.asymmetry_tests()` — four Wald tests, long run and short run, with
  `shortrun_strong` reported as unavailable rather than computed on
  mismatched terms when the two sides carry different lag orders.
  `res.suggests_symmetric_model()` says plainly when the extra parameters
  bought nothing.
- `res.dynamic_multipliers(h, r, seed, alpha)` and
  `res.plot_multipliers(...)` — the signature output of this literature,
  with bands from parameter simulation. The seed is recorded when
  omitted, so a figure can be regenerated after the fact. The bands are
  pointwise, and the documentation says so.
- **The first figure in the library.** Rendered from a real run and
  regenerated by `validation/spec17_figure.py`, never hand-drawn.

### Performance

- The multiplier recursion advances every parameter draw together
  instead of looping in Python once per draw: **11x** faster at
  `h = 40, r = 1000` and **17x** at `h = 100, r = 5000` (2.17 s to
  0.13 s). The naive loop is kept in the tests as the reference the
  vectorised version must match, to 1e-12.

### Added — critical values for the NARDL null (`critical_values.syg2014`)

- Simulated for this null specifically: cases 1-5, one to three
  decomposed variables, 100 000 replications each. A **single** critical
  value per level, not a pair.
- This replaces both conventions the literature uses, and the reason is
  measured rather than argued: reading a NARDL statistic against the PSS
  tables rejects 7.3% of the time at a nominal 5% counting the pieces, or
  2.6% counting the variable, where a genuine two-regressor model is
  correctly sized at 4.8%. With the simulated values, case III lands at
  5.7%. Recorded as OBS-13.

### Infrastructure

- The version now lives in `src/pyardl/__init__.py` and nowhere else.
  Two copies of a version number always drift eventually.
- On a tag push, the CI checks that the tag matches the version the
  built wheel declares. A tag placed without bumping the version
  produces a mislabelled release that installs and imports perfectly —
  nothing else would catch it.
- The dependency-skip guard reads the skip reasons from the run that
  already happened (`-rs`) instead of re-running the whole suite. It was
  doing that on each of the nine matrix cells.
- The package description no longer advertises QARDL, Fourier ARDL or
  panels, which are not implemented.

### Validated

- Against the R package `nardl` on its own inflation/food data:
  coefficients agree to **2.4e-14** and standard errors to 1.7e-12, once
  the package's convention is translated. That convention was read from
  its column construction, not inferred from its coefficient names —
  `lxp` stacks the *contemporaneous level* of `x⁺` and its lags, and its
  cumulation starts without the initial zero.

## [0.3.0] — 2026-08-21

Third release, and the end of phase 3: bootstrap inference, the
three-test framework that names the degeneracies instead of suspecting
them, the Johansen system test, and one simulator for every Monte Carlo
study in the library.

Three conventions in this release were settled by **measurement rather
than by reading**: which null the bootstrap draws from (OBS-8), what the
unconditional model actually removes (locked against bootCT to 1e-12),
and which Johansen statistic meets the specification's own criterion
(OBS-10). Two limits are recorded rather than smoothed over: `F_indep`
is oversized at `T = 100` (OBS-11), and the bootstrap's decisiveness
costs accuracy under a type 2 degeneracy (OBS-12).

### Added — conditional and unconditional models

- `conditional=True | False` on `bounds_test`, `bootstrap_bounds_test`
  and the whole bootstrap path — the distinction of Bertelli, Vacca &
  Zoia (2022). The unconditional form drops the contemporaneous
  differences of the regressors and changes nothing else; the tested
  vector is untouched, so the two forms test the same restriction on two
  specifications.
- The setting is threaded through the observed statistic, the null
  model, the regenerated data and each replication's re-estimation. If
  the null model kept `Dx_t` while the statistic did not, the simulated
  null would not be the null being tested — and nothing in the output
  would say so.
- The convention was **measured, not read**: bootCT reports its own
  unconditional `F_indep` (3.405600 on the Danish data), and of the two
  candidate specifications only one reproduces it, to 1e-12. Locked by
  `tests/replication/test_spec16.py`.
- `res.summary()` states which specification produced the numbers.

### Added — both routes reported side by side

- `res.comparison(alpha)` returns one row per test with the statistic,
  the bootstrap critical value, p-value and verdict, then the classical
  I(0)/I(1) bounds and their verdict. `res.agrees_with_bounds(alpha)`
  answers the same question as a boolean, and `summary()` prints the
  table plus both classifications.
- Where the classical route tabulates nothing — the `t` under cases II
  and IV — the cell reads `unavailable` rather than being left blank: a
  blank cell reads as a non-rejection.
- `bounds` on the classical result now carries the `F_indep` bounds
  alongside those of `F` and `t`, `NaN` outside the simulated grid.

### Added — VECM simulator (`pyardl.simulate`)

- `vecm_ardl(n_obs, alpha, beta, gammas, case, sigma, ...)` — one
  generator for every Monte Carlo study in the library, so a
  disagreement between two validation studies is a disagreement about
  estimators rather than about data.
- Writing `Pi = alpha @ beta.T` makes the rank chosen rather than hoped
  for. The reported `rank` is the rank of `Pi`, not the number of
  columns supplied: a zero `alpha` creates no relation, and saying
  otherwise would claim a relation the data do not contain.
- `degenerate_system(kind, k, speed)` builds the canonical systems the
  three-test framework has to tell apart, written once.
- Seeds are recorded even when drawn from entropy. Deterministic terms a
  case does not carry are refused rather than absorbed.
- Verified by what estimators recover from it, not by what it prints:
  Johansen finds the injected rank, and the three-test classification
  never calls an injected degeneracy cointegration.

### Measured

- What the bootstrap actually buys, on 1000 replications over the four
  canonical systems: almost entirely the removal of the inconclusive
  zone. Under no cointegration the bounds leave 24.8% of samples without
  a verdict and the bootstrap settles them (91.5% correct against
  71.3%); under clear cointegration both give 100% and the bootstrap
  adds nothing. It also costs something — under a type 2 degeneracy the
  bootstrap claims cointegration 3.7% of the time against 0.1%. OBS-12.
- The article's own tables are behind an access barrier, so the
  specification's numeric criterion could not be checked against them.
  That is stated rather than papered over; what is verified is the set
  of qualitative claims, measured on our own DGPs.
- The specification asks for three distinct constrained null DGPs, one
  per test. Measured on 1200 paired Monte Carlo samples, that variant
  over-rejects for `F_indep` too (8.5% against 6.7% in case III,
  McNemar p = 1e-05), in the same direction as OBS-8. One joint null is
  kept; the deviation and its evidence are in `docs/DEVIATIONS.md`.
- The same run showed that `F_indep` is itself oversized at `T = 100` —
  6.4-6.7% at a nominal 5%, where the `t` holds its size. Recorded as
  OBS-11 rather than left unsaid; the first 400-sample pass had
  suggested 4.75%, and the discrepancy was checked to be sampling luck,
  not a bug, before being written down.

### Added — Johansen test (`pyardl.cointegration`)

- `johansen(y, det_order, k_ar_diff, alpha, method)` — a thin wrapper
  over statsmodels' computation, plus what it leaves to the caller: the
  **sequential rank decision** (stop at the first non-rejection, checked
  against `select_coint_rank` at every level and both methods), a result
  object with `.summary()`, and cointegrating vectors normalised so two
  runs can be compared.
- `check_no_cointegration_among_x(x, ...)` — the bounds test assumes the
  regressors are not cointegrated among themselves and gives no sign
  when they are. This checks it and warns, naming the number of
  relations found.
- Limits are refused, not approximated: more than 12 variables, an
  untabulated `alpha`, a single series, or a `det_order` outside
  `{-1, 0, 1}` all raise.
- Measured and documented (OBS-10): the trace statistic **over-selects**
  the rank (87.8% correct against maxeig's 92.5% on a rank-1 DGP) and
  never under-selects. `trace` remains the default because it is what
  the applied literature reports; the behaviour is documented rather
  than hidden behind a default chosen to make a test pass.
- The deterministic correspondence with `urca::ca.jo` was established by
  **running both sides** across six variants: `ecdet="none"` matches
  `det_order=0`, not `det_order=-1`, despite its name. Statistics agree
  to 1e-9; the critical values come from different tabulations and
  differ by under 1%, which is enough to flip a borderline decision.

### Added — the three-test framework (`pyardl.bounds.classification`)

- `F_indep`, the third test of Sam, McNown & Goh (2019), on the
  regressors' levels alone (`γ = 0`). Reported by `bounds_test` and by
  `bootstrap_bounds_test` as `f_indep_stat` / `decision_indep`.
- `res.classification()` returns a **named** verdict and the reason for
  it: `cointegration`, `degenerate_1`, `degenerate_2`,
  `no_cointegration` or `inconclusive`. The two degeneracies are now
  told apart instead of merely suspected — with two tests the
  information to separate them does not exist.
- The mapping from three three-state verdicts to a verdict is total: no
  combination falls through to a default.
- `decision_joint` (the two-test verdict, with its
  `degenerate_suspicion` state) stays on the result object for
  continuity, but `classification()` supersedes it.
- Bounds for `F_indep`: simulated, cases 1-5, `k = 1..10`, at 10/5/1%.
  The published bounds are behind an access barrier and the project does
  not encode a critical value it has not computed. They come from the
  same engine and the same replications as the `F` and `t` bounds, so
  the three describe one single null world. Outside the grid the test is
  reported as unavailable — no neighbouring value is substituted.
- In the bootstrap, all three statistics are drawn under the **same
  joint null**, per the size experiment recorded as OBS-8.

### Added — bootstrap bounds test (`pyardl.bootstrap`)

- `bootstrap_bounds_test(y, x, case, order, n_boot, resample, seed, ...)`
  — the procedure of McNown, Sam & Goh (2018). Instead of bracketing the
  null distribution between an I(0) and an I(1) bound, it builds that
  distribution by regenerating the data under a null that is true by
  construction. The verdict is binary: **there is no inconclusive zone**.
- Both resampling schemes: `"iid"`, and `"wild"` for heteroskedastic
  residuals. Residuals are drawn **by date**, so the contemporaneous
  correlation between the conditional equation and the marginal block
  survives — drawing equations independently would inflate the critical
  values in the optimistic direction with nothing in the output to show
  it.
- The p-value is `(1 + #)/(B + 1)` and never exactly zero: `B`
  replications cannot resolve more than `1/(B+1)`.
- A replication that cannot be estimated is counted and reported, never
  replaced by a fresh draw — replacing it would bias the distribution
  towards estimable samples.
- Reproducibility is a property of the result, not of luck: the seed is
  drawn from entropy and **recorded** when the caller omits it, so any
  run can be reproduced after the fact. Same seed, same critical values,
  bit for bit.
- Building blocks exposed: `estimate_null_dgp`, `simulate_paths`,
  `simulate_path`, `resample_residuals`.

### Performance

- Both hot paths are vectorised across replications. The regeneration
  advances all paths together, and the `B` least-squares fits are solved
  by one stacked QR — never the normal equations, which would square the
  condition number of a design built on lagged levels of integrated
  series.
- Measured end to end: 20x to 53x faster than the first working version.
  A test at `B = 2999` runs in 0.19-1.81 s depending on the
  specification; the specification's own validation study went from
  hours to minutes. The profile was re-measured after each change and is
  recorded rather than asserted.

### Validation

- Checked against the R package `bootCT`: observed statistics agree to
  4e-10 and decisions agree at all three levels. The `F` bootstrap
  bounds differ by 0.6-13%, as two bootstraps with different generators
  do.
- The `t` bounds differed by 21-30%. Rather than record that as a
  caveat, the question was settled by a size experiment: both statistics
  are bootstrapped under the joint null `λ = γ = 0`, and the plausible
  alternative — a separate, weaker null for the `t` — rejects 9.3% of
  the time at a nominal 5%. Our bounds are demanding because they are
  built under the null that holds the size, not out of caution.

### Fixed

- The Johansen wrapper cast possibly-complex eigenvalues, eigenvectors
  and statistics straight to `float`, discarding any imaginary part in
  silence. It now checks the size of what it discards: rounding-level
  imaginary parts (which some BLAS builds produce) pass through, and a
  genuinely complex eigenvalue raises instead of being truncated — it
  would mean the problem solved is not the one the test assumes. The
  dependency's own `ComplexWarning` no longer escapes to the caller.
  Caught by CI on platforms the development machine did not reproduce.
- The trend of the null model was estimated, stored, and then ignored
  when regenerating data. Under deterministic case 5 the bootstrap
  samples therefore lacked the trend the null model describes, making
  that case's critical values wrong.

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
