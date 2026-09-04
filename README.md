<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pyardl-lockup-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/pyardl-lockup-light.svg">
  <img src="docs/assets/pyardl-lockup-light.svg" alt="pyardl" width="300">
</picture>

**ARDL models, bounds tests for cointegration, and critical values you can trace back to their source.**

[![CI](https://github.com/Matth-analyst/pyardl/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/Matth-analyst/pyardl/actions/workflows/ci.yml?query=event%3Apush)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## The problem this solves

You have macroeconomic series and you suspect a long-run relationship. The
classical route — Engle-Granger, Johansen — asks you to establish the
integration order of every series first, and is invalid if you get it wrong or
if the orders are mixed.

Mixed orders are the normal case. Here is what happens on the Danish
money-demand data shipped with this package:

```python
from pyardl.datasets import load_denmark
from pyardl.unitroot import report

data = load_denmark()
print(report(data[["LRM", "LRY", "IBO", "IDE"]]))
```

```text
         order dfgls_level decision_level dfgls_diff decision_diff mzt_level
variable
LRM       I(1)   -1.047542      unit_root  -2.378078    stationary -1.152137
LRY       I(1)   -0.819062      unit_root  -4.630152    stationary -0.813225
IBO       I(1)   -1.744509      unit_root  -2.925082    stationary  -1.64134
IDE       I(0)   -2.445024     stationary        NaN               -2.612435
```

Three I(1) series and one I(0). Engle-Granger has no right to run on this, and
when you run it anyway it finds nothing:

```text
Engle-Granger test (1987) - trend 'c', 4 variables, lags=3, nobs=55
  statistic = -3.3147   p-value = 0.2611
  decision (5%): no_cointegration
```

The bounds test of Pesaran, Shin & Smith does not need the orders to be known,
and on the same data it finds the relationship:

```text
F_overall = 6.2059   decision (5%): cointegration
t_BDM     = -4.5479   decision (5%): cointegration
F_indep   = 8.1619   decision (5%): cointegration

CLASSIFICATION (5%): cointegration
```

That gap is the reason this library exists. Engle-Granger and Johansen ship
here too — you should be able to compare — but they are the point of
reference, not the recommended route.

---

## Install

`pyardl` is not on PyPI yet. Install from source:

```bash
pip install git+https://github.com/Matth-analyst/pyardl.git
```

Or clone it, which is what you want if you intend to read the code:

```bash
git clone https://github.com/Matth-analyst/pyardl.git
cd pyardl
pip install -e ".[dev,plot,bootstrap]"
```

Requires Python 3.11+. Runtime dependencies are numpy, scipy, pandas and
statsmodels — nothing else. `matplotlib` (extra `plot`) and `arch` (extra
`bootstrap`) are optional and imported lazily. An optional native Rust kernel
speeds up the bootstrap; it is never required (see
[Native backend](docs/api/backend.md)).

---

## Sixty seconds

```python
from pyardl.bounds import bounds_test
from pyardl.datasets import load_denmark

data = load_denmark()
res = bounds_test(data["LRM"], data[["LRY", "IBO", "IDE"]], case=3)
print(res.summary())
```

```text
Bounds test (Pesaran, Shin & Smith 2001) - case 3, k=3, ECM(3; LRY:1, IBO:3, IDE:2), critical values: kripfganz

F_overall = 6.2059   decision (5%): cointegration
F p-values: p_I0 = 0.0005, p_I1 = 0.0039
t_BDM     = -4.5479   decision (5%): cointegration
F_indep   = 8.1619   decision (5%): cointegration

CLASSIFICATION (5%): cointegration
  F_overall, t_BDM and F_indep all reject: the level terms are jointly significant, y adjusts back towards equilibrium, and the regressors carry the long-run relationship.

        F_I0   F_I1   t_I0   t_I1  F_indep_I0  F_indep_I1
alpha
0.10   2.730  3.747 -2.570 -3.460       2.084       3.864
0.05   3.229  4.322 -2.860 -3.780       2.619       4.646
0.01   4.311  5.543 -3.430 -4.370       3.814       6.311
```

---

## Design principles

Four rules shape the API, and are the reason to prefer this library over
writing the same tests yourself.

**An inconclusive result stays inconclusive.** The bounds test compares a
statistic against a *pair* of critical values, so the verdict has three
states — `cointegration`, `no_cointegration`, `inconclusive` — and never
collapses to a boolean.

**Nothing is silently substituted.** Ask for a critical value that does not
exist and you get an exception naming a source that does have it, never a
neighbouring cell quietly returned in its place.

**Invalid inference is refused, not decorated.** A confidence interval on the
speed of adjustment is produced only once cointegration is established;
before that you get `NaN` and a warning.

**Every number is traceable.** Every critical value ships with its exact
source and a cross-check against an independent source or an in-house Monte
Carlo engine — see [Validation](#validation).

---

## The workflow

The full sequence — with the *why* of each step, the common error at each
one, and every code block executed in CI — lives in
**[docs/workflow.md](docs/workflow.md)**, which runs end to end on Danish
money demand in a few seconds. Its companions:

- **[Common mistakes](docs/common-mistakes.md)** — six errors that recur in
  published applications, each shown running with the number it produces.
- **[Bootstrap or classical bounds?](docs/bootstrap-or-bounds.md)** — when
  each route is right, measured rather than asserted.
- **[Glossary](docs/glossary.md)** — the vocabulary in English and French,
  with the notation.

In brief, the workflow is: screen every series for I(2) (the bounds test is
invalid, and silent, if one is present) → select lag orders on a common
sample → run the bounds test and read its three-test classification → read
the long-run coefficients and test what theory predicts about them → check
CUSUM *and* CUSUM-of-squares, since they fail differently → fall back to the
bootstrap version if the classical test lands in its inconclusive zone.

---

## What's in the library

Every model below has its own documentation page with the full API, worked
examples, and the measurements behind its design choices. This table is the
map — the pages are where the detail lives.

| Model | Answers | Docs |
|---|---|---|
| **ARDL / UECM** | Estimate the model, read the long-run relationship, test restrictions on it | [ardl.md](docs/api/ardl.md), [restrictions.md](docs/api/restrictions.md), [transforms.md](docs/api/transforms.md) |
| **Bounds test (PSS)** | Is there cointegration, without knowing the integration orders in advance? | [bounds.md](docs/api/bounds.md), [three-tests.md](docs/api/three-tests.md) |
| **Bootstrap bounds test** | Remove the classical test's inconclusive zone | [bootstrap.md](docs/api/bootstrap.md), [conditional.md](docs/api/conditional.md) |
| **Critical values** | Response-surface, small-sample and simulated tables, every one sourced | [critical-values.md](docs/api/critical-values.md) |
| **Unit-root pre-tests** | DF-GLS and Ng-Perron, needed before trusting a bounds test | [unitroot.md](docs/api/unitroot.md) |
| **Stability diagnostics** | CUSUM and CUSUM-of-squares — they catch different breaks | [diagnostics.md](docs/api/diagnostics.md) |
| **Engle-Granger / Johansen** | The classical alternatives, for comparison | [cointegration.md](docs/api/cointegration.md), [johansen.md](docs/api/johansen.md) |
| **NARDL** | Does `y` respond differently to rises and falls in `x`? | [nardl.md](docs/api/nardl.md) |
| **QARDL** | Does the long run change across the distribution of `y`, not just its mean? | [qardl.md](docs/api/qardl.md) |
| **Fourier ARDL / Fourier-ADL** | Cointegration under a smooth, undated structural break | [fourier.md](docs/api/fourier.md), [fourier-adl.md](docs/api/fourier-adl.md) |
| **Unified analysis** | One entry point over asymmetry × Fourier × bootstrap, 8 configurations | [unified.md](docs/api/unified.md) |
| **Heterogeneous panels** | Mean Group, PMG, DFE, CS-ARDL, CS-DL, and the Hausman test between them | [panel.md](docs/api/panel.md) |
| **DOLS / FMOLS / CCR** | Long-run inference that static OLS on cointegrated data does not give you | [efficient.md](docs/api/efficient.md) |
| **Koyck / Almon** | The distributed-lag models the whole family descends from | [distributed-lags.md](docs/api/distributed-lags.md) |
| **Dynamic simulation** | Trace what happens to `y` if a regressor moves and stays there | [dynardl.md](docs/api/dynardl.md) |
| **VECM simulator** | One reproducible data generator behind every Monte Carlo study in the library | [simulate.md](docs/api/simulate.md) |
| **Native backend** | An optional Rust kernel for the bootstrap — what profiling said to port, and by how much | [backend.md](docs/api/backend.md) |

A representative taste, so the table above is not just names: on the
Danish data, three tests must all reject for `bounds_test` to call it
cointegration —

```python
label, reason = res.classification()
```

| `F_overall` | `t_BDM` | `F_indep` | `classification()` |
|---|---|---|---|
| rejects | rejects | rejects | `cointegration` |
| rejects | rejects | does not | `degenerate_1` — y reverts to its own past, not to the regressors |
| rejects | does not | rejects | `degenerate_2` — the regressors matter, but nothing pulls y back |
| does not | does not | does not | `no_cointegration` |
| anything else | | | `inconclusive` |

The mapping is total, and `reason` says in one sentence which test decided.
Full account, including why two tests cannot tell the degeneracies apart, in
[three-tests.md](docs/api/three-tests.md).

---

## Validation

This is the part worth reading before trusting any number, in full, in
[docs/VALIDATION_OBSERVATIONS.md](docs/VALIDATION_OBSERVATIONS.md). The short
version:

- **Against reference implementations.** Coefficients and residuals agree
  with `statsmodels` to 1e-10 and with R's `ARDL` package to 1e-6. PSS
  (2001)'s UK wage equation is reproduced to 1e-4. Panel estimators are
  cross-checked against `plm`, `ardlverse` and (where those tools do not
  reach) documented Stata scripts.
- **Critical values.** Every shipped table cites its exact source and its
  cross-check in
  [PROVENANCE.md](src/pyardl/critical_values/PROVENANCE.md) — a second
  published source where one exists, an in-house Monte Carlo engine with
  recorded seeds where none does.
- **Conventions settled by measurement, not by reading.** Every place a
  specification admitted two readings, the choice was made by measuring
  both and recording the numbers that decided it — never guessed.
- **Limits, stated rather than hidden.** Where a test is oversized, where a
  method underperforms its own literature's claims, where a cross-check is
  structural rather than external — each one is a numbered entry in the
  validation register, not a footnote.
- **Test suite.** 700+ tests plus 38 doctests, `mypy --strict` clean, on
  Linux, Windows and macOS across Python 3.11–3.13. Monte Carlo experiments
  re-run nightly at full replication counts (`event: schedule` in Actions —
  the push-triggered badge above only covers the fast suite).

---

## Compatibility

| | |
|---|---|
| Python | 3.11, 3.12, 3.13 |
| OS | Linux, Windows, macOS |
| Required | numpy, scipy, pandas, statsmodels |
| Optional | matplotlib (`plot`), arch (`bootstrap`), a Rust toolchain (native backend) |

Tested against numpy 2.5 and pandas 3.0, and against a pinned pandas 2.1
floor.

---

## Roadmap

- **0.1.0** — ARDL/UECM estimation, bounds test with the five deterministic
  cases, joint F and t decision, PSS critical values.
- **0.2.0** — small-sample and response-surface critical values with
  p-values, CUSUM/CUSUMSQ stability, DF-GLS and Ng-Perron pre-tests,
  long-run restriction testing and seasonality, Engle-Granger.
- **0.3.0** — bootstrap ARDL with no inconclusive zone, the three-test
  framework, the Johansen system test, conditional/unconditional models,
  one VECM simulator for every Monte Carlo study.
- **0.4.0** — NARDL: asymmetric responses, the four symmetry tests, dynamic
  multipliers with simulated bands, critical values for the decomposed
  null.
- **0.5.0** — QARDL and QNARDL, Fourier ARDL and Fourier-ADL, the unified
  entry point, heterogeneous panels (MG, PMG, DFE, CS-ARDL, CS-DL), the
  efficient long-run estimators (DOLS, FMOLS, CCR), dynamic simulations,
  and the distributed-lag models the family descends from (Koyck, Almon).
  With this release, the 28 planned specifications are all implemented.
- **0.6.0** — performance work, and three measurements that contradicted the
  reasoning behind them: an optional Rust kernel plus an augmented QR
  (**1.53x to 1.80x** on a bootstrap run), the QARDL band coverage study the
  specification had always asked for, and a declared pandas floor that had
  stopped being true. Full account in [CHANGELOG.md](CHANGELOG.md).

Next: **0.7+** — whatever the validation register turns up. The
specifications are implemented and the performance profile is flat; what
remains is use, and the questions use raises.

---

## Contributing

Issues and pull requests are welcome. Before opening a PR:

```bash
pip install -e ".[dev,plot,bootstrap]"
ruff check src tests && ruff format --check src tests
mypy src/pyardl
pytest -m "not slow" -ra --doctest-modules src/pyardl tests docs --cov=pyardl
```

Two expectations specific to this project. Every statistical claim needs a
test that would fail if the claim were false — not a smoke test. And **no
numerical value is ever written from memory**: critical values, docstring
examples, doctest expectations and figures quoted in documentation are all
computed by a real run and pasted from it.

---

## Citing

If you use `pyardl` in published work, please cite the software (see
[CITATION.cff](CITATION.cff)) as well as the methodological articles behind
the part you used — they are listed in each module's docstring and doc page.

The central reference is Pesaran, M. H., Shin, Y. & Smith, R. J. (2001).
Bounds testing approaches to the analysis of level relationships. *Journal
of Applied Econometrics*, 16(3), 289–326. Every other article this library
implements is cited where it is used, in the linked doc page for that model.

---

<div align="center">

MIT licensed · [Documentation](docs/) · [Changelog](CHANGELOG.md)

</div>
