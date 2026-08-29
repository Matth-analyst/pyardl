# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/1.1.0/).
This project follows [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] — 2026-08-29

Fifth release, and the one that closes the genealogy. The library now
covers every specification the project set out to implement, from the
distributed-lag models of the 1950s that the ARDL descends from to the
heterogeneous panels and dynamic simulations at its frontier — each with
its own external cross-validation against R or Stata output.

### Changed — the cross-cutting review before the tag

Three inconsistencies that no functional test could have caught, because
each of them worked:

- **`cointegration_analysis(B=...)` is now `n_boot=...`.** It was the
  only public place calling the bootstrap replication count `B`, while
  every other entry point — and the field this very object records,
  `UnifiedResults.n_boot` — used `n_boot`. Breaking change, no alias:
  the inconsistency is better removed now than carried. The printed
  summaries keep `B=2999`, which is display notation from the
  literature, not a parameter name.
- **`ARDLResults` and `BoundsTestResults` are frozen.** They were the
  only two mutable Results objects out of twenty-seven, and the
  architecture had promised immutability throughout. A value rewritten
  after the fact would be reported by `summary()` with exactly the same
  confidence as a computed one. `dataclasses.replace` still works.
- **`pyardl.core` now exports its own names.** It was the library's only
  empty `__init__`, which made `ARDL` — the central class everything
  else is built on — the sole symbol requiring a deep import path.
  `from pyardl.core.ardl import ARDL` keeps working.

### Added — a lazy top-level surface

`from pyardl import ARDL, bounds_test` now works. The re-exports use
PEP 562's module `__getattr__`, so `import pyardl` stays at about a
millisecond instead of paying the ~3.5 s statsmodels import that eager
binding would charge to everyone, including a caller who only wants
`__version__`. `tests/test_public_api.py` checks that in a fresh
interpreter, since inside a pytest session statsmodels is already
loaded and the question no longer arises.

Packaging metadata caught up at the same time: keywords, classifiers,
project URLs, and a description that matches what the library actually
does. Verified on the built wheel — an earlier version of that edit had
silently moved `dependencies` under `[project.urls]`, which would have
shipped a package declaring no requirements at all.


### Added — Distributed lags (`pyardl.distributed_lags`, specs 01 & 02)

- `KoyckModel` — the geometric lag, with three estimators. `"ols"` is
  kept and warns on every fit, because the point of this model is that
  the obvious estimator is *inconsistent*: measured bias on lambda of
  0.118 at T = 2000 and 0.121 at T = 8000 (Monte Carlo error 0.002), so
  it does not shrink. `"iv"` (Liviatan, default) and `"ml"` (the MA(1)
  cross-restriction) recover the truth.
- Durbin's `h` instead of Durbin-Watson, with an automatic fallback to
  Durbin's alternative test when `n * var(lambda)` makes `h` undefined,
  and the index naming which one ran.
- `AlmonModel` (alias `PDL`) — polynomial lags with endpoint
  constraints imposed on the null space (exact to 1e-10, not by
  penalty), a normalised or Chebyshev basis for conditioning, and
  `select_order` on a **common sample**.
- `polynomial_restriction_test` in every Almon summary. An Almon model
  always produces a smooth lag distribution — that is what it was asked
  for — so the only informative question is whether the shape survives a
  comparison with the unrestricted finite lag.
- Bridges to the core: `to_ardl()` matches an ARDL(1, 0) to 1e-12 and
  `to_fdl()` shows the free weights the restriction replaced.
- Cross-validated against R `dLagM` 1.1.13: Almon weights to **1.8e-13**,
  standard errors to 1.2e-9, SSR to 7.4e-15, and the same weights again
  from the Chebyshev basis (5.0e-14).
- **The Koyck comparison disagreed, and the disagreement is documented
  rather than smoothed over.** `koyckDlm`'s own `ivreg` formula,
  `y.t ~ Y.1 + X.t | Y.1 + X.t_1`, puts `Y.1` on both sides of the bar:
  it instruments `X.t` and treats the lagged dependent variable as
  exogenous. Reproducing that instrument set inside pyardl lands on its
  coefficients to 1e-8; on a DGP with `lambda = 0.6` known, it carries a
  bias of −0.095 where Liviatan carries +0.0001, and is more biased than
  plain OLS. OBS-26.

### Added — Dynamic simulation (`pyardl.simulate`, spec 25)

- `dynardl_simulate` / `ARDLResults.dynardl_simulate` /
  `NARDLResults.dynardl_simulate` — run a fitted model forward under a
  counterfactual step or impulse, with bands from parameter draws. The
  interpretation layer of Jordan and Philips (2018): the coefficient
  table says how the pieces fit, this says what happens and when.
- The reported response is a **paired difference** from a no-shock
  counterfactual, computed draw by draw, so the intercept, trend,
  seasonal dummies and starting point cancel exactly rather than
  approximately. Each draw starts at its own implied equilibrium, which
  is what makes the no-shock branch flat for every draw.
- `stochastic=True` adds innovations to both branches. Because the model
  is linear in `y`, they cancel out of the response *exactly* — the
  columns are identical to `stochastic=False` up to rounding (1.4e-14),
  and the test suite pins it. Forecast uncertainty shows up on the
  level, which is where it belongs.
- Band coverage measured, not assumed: 1000 replications of an ARDL(1,1)
  with known coefficients give 93.7% / 94.8% / 94.3% / 95.0% at h = 5,
  6, 10 and 60 for a nominal 95%, with a Monte Carlo standard error of
  0.69 point (`validation/spec25_montecarlo.py`).
- A step of size one on a NARDL's `x_pos` reproduces the `m_pos` column
  of `dynamic_multipliers` to 1e-10 — two routes to the same object.
- `param_draws` accepts bootstrap replications, so a figure and a bounds
  test can rest on one notion of uncertainty rather than two.
- Cross-validated against R `dynamac` 0.1.12 on the Danish data pyardl
  already ships: the 13 regression coefficients to **3.5e-14**, the
  baseline equilibrium to 5.8e-05, and the final level inside dynamac's
  own three-seed spread.

### Fixed — Long-run views read the parameter vector by name (OBS-25)

- `ardl_params` sliced `_params` **positionally**, and seasonal dummies
  sit between the intercept and the lags in the design. With
  `seasonal=True` the beta slice therefore started `s-1` columns early:
  `longrun`, `adjustment`, `half_life` and `to_ecm` were all read off
  the wrong coefficients — silently, with plausible-looking numbers and
  a coherent standard error, since the same shift hit `cov_params`.
  On a test fit, theta came out 0.3768 instead of 1.2745.
  Coefficients are now picked by name and the covariance is restricted
  to the order `ARDLParams.param_vector` documents. Found by the spec-25
  test that forces a numerical recursion and an algebraic formula to
  agree.

### Added — Efficient long-run estimators (`pyardl.cointegration`)

- `dols`, `fmols`, `ccr` — Stock-Watson, Phillips-Hansen and Park. The
  robustness block that sits next to an ARDL long run; static OLS is
  consistent but its inference is not, and these repair it three ways.
- `compare_longrun` — ARDL, DOLS, FMOLS and CCR side by side in one
  call.
- `pyardl.utils.longrun_covariance_kernel` — the transversal brick:
  four kernels, two automatic bandwidth rules, checked against
  `cointReg::getLongRunVar` to **5.3e-15**. Also
  `pyardl.utils.lead_lag_matrix`.
- Cross-validated against `cointReg` 0.2.0: FMOLS `theta` to **2.0e-11**
  and its standard errors to 8.6e-13; DOLS to 7.2e-14 and 2.3e-12. CCR
  has no external reference — `cointReg` does not implement it — so it
  rests on convergence and on agreement with FMOLS.
- `ccr` iterates Park's transformation to a **fixed point** (`res.n_iter`,
  `res.converged`) rather than substituting a first-stage estimate once.
  The transformation depends on `theta`, so a single pass carries the
  static-OLS bias into it: iterating takes CCR from 15% of the OLS bias
  to 9% at T = 400, and its coverage from 93.0% to 93.7%.
- All three estimators meet **both** criteria of the specification at
  T = 400: bias under 10% of the OLS bias (3.3%, 9.6%, 9.1%) and
  coverage inside [92, 97]% (93.4%, 94.5%, 93.7%). Prewhitening
  (Andrews-Monahan, on by default) is what makes DOLS and FMOLS meet
  them; the fixed point is what makes CCR meet them.

### Added — Documentation as tested code

- **`docs/workflow.md`** — the complete methodological sequence on
  Danish money demand: integration orders, no-cointegration-among-x,
  order selection, the three-test bounds procedure, the long run and
  adjustment speed, stability, and three robustness routes. Runs
  end-to-end in 6 s (the specification allows 60).
- **`docs/common-mistakes.md`** — six errors recurring in applied work,
  each *shown running* with the number it produces, and what the API
  does to make committing it deliberate rather than accidental.
- **`docs/glossary.md`** — English/French vocabulary and notation, with
  the two degeneracies distinguished precisely and the two terms this
  project uses deliberately ("inconclusive", "classification").
- Multiple-threshold asymmetries (Greenwood-Nimmo et al.) documented as
  user code on `partial_sums`, with the deterministic-drift caveat — no
  dedicated API in this version, deliberately.
- **The documentation is now executed in CI.** `--doctest-glob=*.md`
  runs every `>>>` block in `docs/`; pages without one are simply not
  collected. This is not ceremony: writing `common-mistakes.md` I
  invented the table comparing the five deterministic cases, and the
  doctest rejected it within the minute. The real spread is wider than
  what I made up — `F` runs from 0.71 to 6.79 across the cases on the
  same data.

### Added — Cross-sectional dependence (`pyardl.panel`)

- `CSARDL` / `CSDL` — the estimators of Chudik & Pesaran (2015, 2016).
  The individual regression is augmented with the cross-sectional
  averages of the variables, which span the unobserved factor space;
  CS-ARDL keeps the dynamics and rebuilds the long run from them, CS-DL
  reads the long run straight off the coefficient on `x` and cannot
  report an adjustment speed — its `summary()` says so.
- `cross_section_averages` / `default_cs_lags` — the averages, their
  lags, and the `floor(T**(1/3))` rule. The count of individuals
  entering each average is returned alongside it, and a sharply varying
  composition warns: in an unbalanced panel a mean over 40 countries and
  a mean over 12 are not the same regressor.
- `cd_test` — Pesaran's CD test, with `summary(context='before'|'after')`
  because the same p-value means opposite things on either side of the
  augmentation. Pairs are matched on the index, never by position.
- Collinear columns are dropped by a **declared left-to-right rule**,
  averages last, every drop recorded in `res.dropped_columns` — not by
  whatever the solver prefers, which would give different long-run
  coefficients on different machines from the same data.
- Cross-validated against `plm::pcce(model="mg")` to **1.1e-16** on the
  group coefficient and 1.5e-16 on the between-individual standard
  error. That covers the *static* CCE case only; the dynamic half has no
  external reference — Stata's `xtdcce2` is the one the specification
  names, a `.do` script is provided, and no values were invented.

### Added — Pooled Mean Group, DFE and Hausman (`pyardl.panel`)

- `PMG` — the estimator of Pesaran, Shin & Smith (1999): long-run
  coefficients pooled, short-run dynamics free, by concentrated maximum
  likelihood. Back-fitting (as `xtpmg`) and a quasi-Newton path over the
  same concentrated likelihood, pinned together to 1e-6; iteration log
  kept; started from `theta_MG`.
- `DFE` — dynamic fixed effects, for the MG/PMG/DFE table panel papers
  report. Its `summary()` says it is there for comparison, not as a
  recommendation.
- `hausman` / `PMGResults.hausman_vs_mg` — the homogeneity test, with
  the pseudo-inverse path recorded rather than hidden when the variance
  difference is not positive definite (1.8% of replications measured).
- `compare` — MG, PMG and DFE on one panel plus the Hausman verdict.
- Cross-validated against `ardlverse::panel_ardl` (which replicates
  Stata's `xtpmg`): PMG `theta` to 1.9e-08, its standard error to
  2.1e-10, the log-likelihood to 4.1e-12; DFE to 1.1e-16. The spec asks
  for 1e-3.

### Added — Heterogeneous panels (`pyardl.panel`)

- `MeanGroup` — the estimator of Pesaran & Smith (1995): one ARDL per
  individual, then average. Common or per-individual lag orders; mean,
  median or trimmed aggregation for small N; individual fits kept and
  addressable; `heterogeneity()` for the spread of the estimates.
- `panel_from_frame` / `PanelData` — the container specs 22-24 share. It
  sorts time, refuses to bridge internal gaps or duplicate periods, and
  records every excluded individual with its reason so the `N` in a
  results table can always be accounted for.
- Cross-validated against `plm::pmg(model="mg")` to **5.6e-16** on the
  group coefficients and **4.9e-17** on the between-individual standard
  errors, on a panel pyardl generates itself from a fixed seed. A second
  check on the Produc panel of Munnell (1990) agreed to 4.5e-14.

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

- **It is persistence, not the factor, that breaks Mean Group.** On a
  dynamic DGP where the common factor enters `y` through its difference
  — I(0) — MG is unbiased at every loading strength (bias −0.003 from
  `gamma = 0` to `0.6`, 1000 replications, Monte Carlo standard error
  0.0004), even though the factor is present and correlated with the
  regressor. The bias appears only when the factor's contribution to `y`
  is persistent. That distinction is in neither formulation of the
  result I would have copied; it came out of a DGP that refused to
  produce the expected bias. OBS-23.
- **The augmentation is not free, and the CD test misleads after it.**
  At `gamma = 0` — no factor — CS-DL still carries a −0.027 bias (3.4%)
  and CS-ARDL −0.008. And the CD test rejects 100% of the time *after*
  augmentation even with no factor: regressing each individual on
  averages containing its own `y` induces a mechanical negative
  correlation between residuals (the reference panel shows `CD = −6.12`
  with mean absolute correlation 0.10). A significant CD on CS-ARDL
  residuals therefore does not mean factors remain. OBS-23.
- **A common factor costs 49%, and the augmentation recovers it.** On the
  reference panel — factor entering both `y` and `x`, heterogeneous
  loadings, true `theta = 0.80` — a plain Mean Group returns **1.1938**.
  CS-ARDL returns 0.8058 and CS-DL 0.8059. The CD test rejects before
  the augmentation (mean absolute pairwise correlation 0.30) and still
  rejects after it (0.10): most of the dependence is absorbed, not all,
  and the test is honest about the remainder rather than being read as a
  clean bill of health.
- **`floor(T**(1/3))` loses a lag at every perfect cube** if written as
  `T ** (1/3)`: measured, `64 ** (1/3) == 3.99999999999999956` and
  `1000 ** (1/3) == 9.99999999999999822`, so the naive form returns 3
  and 9 where the rule means 4 and 10. `numpy.cbrt` is exact. A shorter
  lag list is a different specification, not a rounding detail.
- **The Hausman test is asleep exactly where PMG has already failed.**
  Under exact long-run homogeneity PMG delivers: bias −0.14% (the spec
  asked for under 1%) and a **2.41x** efficiency gain over MG. But at a
  13% dispersion of the true coefficients — unremarkable for a panel of
  countries — PMG is biased +2.55%, its 95% interval covers **36%**, and
  its efficiency advantage has inverted to 0.61x. At that same
  dispersion the Hausman test rejects only **18.6%** of the time: in more
  than four samples out of five where PMG is materially wrong, the
  standard diagnostic says it is fine. Its size is not exact either
  (8.6% against a nominal 5%). MG meanwhile does not move — bias between
  −0.58% and −0.12%, coverage 94.2–94.8% throughout. 2000 replications,
  standard error 0.49 point. OBS-22.
- **Two implementations disagreed; the likelihood settled it.** The PMG
  cross-check first showed a 2.7e-07 gap on `theta` — small enough to
  absorb by loosening one's own tolerance. Instead the concentrated
  log-likelihood was computed at both estimates: it was *lower* at the
  reference's, which had stopped at its default `tol=1e-6`. Re-run at
  1e-8 it agrees to 6.7e-10. Separately, the same comparison caught a
  real bug: the PMG variance formula projected out only the short-run
  regressors, forgetting that `lambda_i` is estimated too, and returned
  a standard error 5% too small. Nothing internal would have flagged it;
  the numerical Hessian of the concentrated likelihood did. OBS-21.
- **An interval that gets more wrong the more data you collect.** The
  Mean Group standard error comes from the dispersion *across*
  individuals, not from pooling the individual standard errors. The
  naive construction is not absurd — each individual variance is correct
  for its own coefficient — but it covers 54% at T=50 and **27%** at
  T=100 against a nominal 95% (2000 replications, standard error 0.49
  point), while the correct one holds 94-95% throughout. The mechanism
  was measured, not guessed: `se_between` converges to a non-zero
  constant (factor 0.999 from T=200 to T=400) while `se_naive` halves at
  every doubling of T — the 1/T rate of superconsistency under
  integrated regressors, not 1/sqrt(T). The gap grows like T, so the
  naive coverage tends to zero. OBS-20.
- **Pooling costs consistency, not efficiency — measured.** On the same
  heterogeneous DGP the dynamic-fixed-effects bias does not shrink with
  T (+0.0242 at T=50, +0.0273 at T=100; Monte Carlo standard error
  0.0014) and raising N does not help (+0.0300 at N=50). The Mean Group
  bias does shrink (−0.0050 then −0.0026).
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
