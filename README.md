# pyardl

ARDL models and bounds tests for cointegration in Python.

`pyardl` estimates autoregressive distributed lag models, converts them
to their error-correction form, and runs the Pesaran-Shin-Smith bounds
test for the existence of a long-run relationship — the test you reach
for when you do not know in advance whether your series are I(0) or
I(1).

It fills a gap: the bounds test is standard in applied econometrics and
available in Stata and R, but Python support has been partial. `pyardl`
covers all five deterministic cases, gives you three critical-value
sources including modern response surfaces with p-values, and never
reduces an inconclusive result to a yes/no answer.

```python
from pyardl.bounds import bounds_test
from pyardl.datasets import load_denmark

data = load_denmark()
res = bounds_test(
    data["LRM"],                       # log real money demand
    data[["LRY", "IBO", "IDE"]],       # income, bond rate, deposit rate
    case=3,                            # unrestricted intercept, no trend
    order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}),
)
print(res.summary())
```

```text
Bounds test (Pesaran, Shin & Smith 2001) - case 3, k=3, ECM(3; LRY:1, IBO:3, IDE:2), critical values: kripfganz

F_overall = 6.2059   decision (5%): cointegration
F p-values: p_I0 = 0.0005, p_I1 = 0.0039
t_BDM     = -4.5479   decision (5%): cointegration
joint decision (F and t): cointegration

        F_I0   F_I1   t_I0   t_I1
alpha
0.10   2.730  3.747 -2.570 -3.460
0.05   3.229  4.322 -2.860 -3.780
0.01   4.311  5.543 -3.430 -4.370
```

## Installation

```bash
pip install -e .
```

Requires Python 3.11+ and numpy, scipy, pandas, statsmodels. Not yet on
PyPI; install from a clone for now.

## What it does

**Estimation** — `pyardl.core.ardl.ARDL` fits ARDL(p, q) models by OLS
with robust covariance options (HC0-HC3, Newey-West), reports the usual
regression output, and checks the residuals for autocorrelation
automatically, because long-run inference is not valid without it.

```python
from pyardl.core.ardl import ARDL

res = ARDL(y, x, order=(2, {"income": 1, "price": 2})).fit()
res.longrun        # long-run coefficients with delta-method std. errors
res.adjustment     # speed of adjustment and half-life
res.to_ecm()       # exact error-correction reparameterisation
res.is_stable      # are all AR roots outside the unit circle?
res.diagnostics()  # Ljung-Box, Jarque-Bera, Breusch-Pagan
```

**Order selection** — a grid or per-variable search over AIC, BIC and
HQ. All candidates are estimated on the same sample, so the criteria are
actually comparable; the selected model is then re-estimated on its own
maximal sample.

```python
sel = ARDL.select_order(y, x, max_p=4, max_q=4, ic="bic")
sel.best_order
sel.top(5)         # inspect near-optimal specifications too
```

**General-to-specific reduction** — `ARDL.gets` reduces an
over-parameterised model while keeping the residual diagnostics clean,
and records every step in `reduction_path` so the sequence can be
audited.

**Bounds testing** — all five deterministic cases, the F and t
statistics, and a three-state verdict (`cointegration`,
`no_cointegration`, `inconclusive`) that is never collapsed into a
boolean. When the F and t tests disagree in the specific way that
signals a degenerate relationship, that is reported too rather than
hidden.

**Critical values** — three sources, selected with `cv_source`:

| Source | Use it for | Coverage |
|---|---|---|
| `"kripfganz"` (default) | everyday work; precise values at any level, with p-values | cases 1-5, k = 1-10, F |
| `"pss"` | reproducing published results exactly | cases 1-5, k = 0-10, F and t |
| `"narayan"` | small samples, 30 ≤ T ≤ 80 | cases 2, 3, 5, k ≤ 7, F |

Every shipped table documents its exact source and how it was
cross-checked in
[`PROVENANCE.md`](src/pyardl/critical_values/PROVENANCE.md). A Monte
Carlo engine (`simulate_bounds`) is also available for configurations
no published table covers.

## Design choices

A few decisions that shape how the library behaves:

- **Inconclusive means inconclusive.** The bounds test genuinely has a
  region where no conclusion follows. `pyardl` reports it as such, and
  shows the p-value interval so the result can be read on a continuous
  scale.
- **Methodological warnings are real warnings.** Autocorrelated
  residuals, an adjustment speed with the wrong sign, unstable dynamics:
  each raises a `PyardlMethodologyWarning` you can turn into an error
  with `warnings.filterwarnings("error", category=...)`.
- **Confidence intervals are withheld when they would be invalid.** The
  interval for the adjustment speed only appears once cointegration is
  established, since its distribution is non-standard under the null.
- **Nothing is silently substituted.** Asking for a critical value
  outside a source's coverage raises an error naming a source that does
  cover it, rather than quietly returning a neighbouring cell.

## Validation

Estimation results agree with `statsmodels.tsa.ardl` to 1e-10, and with
the R `ARDL` package to 1e-6 on the Danish dataset. The bounds test
reproduces the UK real-wage application of Pesaran, Shin & Smith (2001)
to 1e-4 on both statistics. The shipped critical value tables were
cross-checked cell by cell against independent sources and an internal
Monte Carlo engine.

```bash
pytest -m "not slow"     # full suite
pytest -m slow           # long Monte Carlo runs
```

## Status

Version 0.1.0. The estimation core, the error-correction algebra, the
bounds test and the critical-value machinery are implemented, tested
and validated. Work continues on parameter-stability tests, unit-root
pre-tests, bootstrap inference, and the non-linear and panel
extensions.

## References

Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). "Bounds Testing
Approaches to the Analysis of Level Relationships", *Journal of Applied
Econometrics*, 16(3), 289-326.

Kripfganz, S. & Schneider, D. C. (2020). "Response Surface Regressions
for Critical Value Bounds and Approximate p-values in Equilibrium
Correction Models", *Oxford Bulletin of Economics and Statistics*,
82(6), 1456-1481.

Narayan, P. K. (2005). "The saving and investment nexus for China:
evidence from cointegration tests", *Applied Economics*, 37(17),
1979-1990.

Banerjee, A., Dolado, J. & Mestre, R. (1998). "Error-correction
Mechanism Tests for Cointegration in a Single-equation Framework",
*Journal of Time Series Analysis*, 19(3), 267-283.

## Licence

MIT — see [LICENSE](LICENSE). The encoded critical values come from
published tables; their provenance and licensing are documented in
[`PROVENANCE.md`](src/pyardl/critical_values/PROVENANCE.md). No
third-party material is redistributed without a clear licence.
