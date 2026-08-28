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

- **[Getting started](getting-started.md)** — from a dataset to a
  defensible conclusion, step by step.
- **[The complete workflow](workflow.md)** — the whole methodological
  sequence on Danish money demand, every code block executed in CI.
- **[Common mistakes](common-mistakes.md)** — six errors that recur in
  published applications, each one shown running.
- **[Glossary](glossary.md)** — the vocabulary in English and French,
  with the notation.
- **[ARDL estimation](api/ardl.md)** — fitting, order selection,
  general-to-specific reduction.
- **[Bounds test](api/bounds.md)** — the five deterministic cases, the
  three-state decision, the joint F/t verdict.
- **[Critical values](api/critical-values.md)** — which source to use
  and why.
- **[ARDL ↔ ECM algebra](api/transforms.md)** — the exact
  reparameterisation and the long-run quantities it yields.

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
