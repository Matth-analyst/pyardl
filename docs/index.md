# pyardl

ARDL models, bounds tests for cointegration, and modern critical values
in Python.

`pyardl` estimates autoregressive distributed lag models, converts them
exactly to their error-correction form, and tests for a long-run level
relationship using the Pesaran, Shin & Smith (2001) bounds procedure —
with critical values you can trace back to their source.

```python
from pyardl.bounds import bounds_test
from pyardl.datasets import load_denmark

data = load_denmark()
res = bounds_test(data["LRM"], data[["LRY", "IBO", "IDE"]], case=3)
print(res.summary())
```

## Where to go

**Start here**

- **[Getting started](getting-started.md)** — from a dataset to a
  defensible conclusion, step by step.
- **[The complete workflow](workflow.md)** — the whole methodological
  sequence on Danish money demand, every code block executed in CI.
- **[Common mistakes](common-mistakes.md)** — six errors that recur in
  published applications, each one shown running.
- **[Glossary](glossary.md)** — the vocabulary in English and French,
  with the notation.

**Deciding**

- **[Bootstrap or classical bounds?](bootstrap-or-bounds.md)** — which
  inference to use, and when the choice changes the answer.
- **[Testing an economic restriction](long-run-restrictions.md)** —
  putting a hypothesis on the long run rather than reading it off.
- **[Validation register](VALIDATION_OBSERVATIONS.md)** — every
  hypothesis this project stated and then refuted by measurement.

**Core**

- **[ARDL estimation](api/ardl.md)** — fitting, order selection,
  general-to-specific reduction.
- **[ARDL ↔ ECM algebra](api/transforms.md)** — the exact
  reparameterisation and the long-run quantities it yields.
- **[Bounds test](api/bounds.md)** — the five deterministic cases, the
  three-state decision, the joint F/t verdict.
- **[Three-test framework](api/three-tests.md)** — naming the two
  degeneracies instead of calling them cointegration.
- **[Critical values](api/critical-values.md)** — which source to use
  and why.
- **[Bootstrap bounds test](api/bootstrap.md)** — inference with no
  inconclusive zone.
- **[Conditional vs unconditional](api/conditional.md)** — the
  specification choice hiding inside "the" bounds test.
- **[Long-run restrictions](api/restrictions.md)**,
  **[stability diagnostics](api/diagnostics.md)** and
  **[unit-root pre-tests](api/unitroot.md)**.

**Beyond the linear ARDL**

- **[NARDL](api/nardl.md)** — asymmetric long-run and short-run
  responses, with the symmetry tests that make asymmetry a finding.
- **[QARDL](api/qardl.md)** — a long run that is allowed to differ
  across quantiles.
- **[Fourier terms](api/fourier.md)** and
  **[Fourier-ADL cointegration](api/fourier-adl.md)** — smooth breaks
  without dating them.
- **[Unified analysis](api/unified.md)** — one entry point over the
  eight configurations, and the one combination it refuses.
- **[Heterogeneous panels](api/panel.md)** — MG, PMG, DFE, CS-ARDL and
  CS-DL, with the Hausman test between them.

**Alternatives and interpretation**

- **[Efficient long-run estimators](api/efficient.md)** — DOLS, FMOLS,
  CCR, and what static OLS inference actually costs.
- **[Engle-Granger](api/cointegration.md)** and
  **[Johansen](api/johansen.md)** — the classical routes, for
  comparison.
- **[Dynamic simulation](api/dynardl.md)** — what happens to `y`, and
  when, if a regressor moves and stays there.
- **[Distributed lags](api/distributed-lags.md)** — Koyck and Almon,
  the roots of the genealogy.
- **[VECM simulator](api/simulate.md)** — one data generator for every
  Monte Carlo study in the library.

## What makes it different

An inconclusive bounds test stays inconclusive: the decision has three
states, never a boolean. A confidence interval on the speed of
adjustment is refused until cointegration is established. Every critical
value ships with its exact source, and every table was cross-checked
against an independent one or against the internal Monte Carlo engine.

## Install

```bash
pip install pyardl
```

Requires Python 3.11+, numpy, scipy, pandas and statsmodels.
`matplotlib` and `arch` are optional (`pip install "pyardl[plot]"`).

Every symbol lives in the submodule that owns it — `pyardl.bounds`,
`pyardl.nardl`, `pyardl.panel` — and the documentation uses those paths
throughout. The headline entry points are also re-exported at the top
level for convenience:

```python
from pyardl import ARDL, bounds_test, NARDL, load_denmark
```

Those re-exports are **lazy**: `import pyardl` costs about a
millisecond, and statsmodels is loaded only when you first touch a name
that needs it.
