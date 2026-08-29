# The complete workflow

An ARDL analysis is a sequence, and most of what goes wrong in published
work goes wrong *between* the steps rather than inside them. This page
walks the whole sequence once, on real data, saying at each stage what
the step is for, how pyardl does it, and what the common error is.

**Every `>>>` block on this page is executed in CI.** If the library
changes so that a number here is no longer what it says, the build
fails. The whole page runs on the Danish money-demand data in under ten
seconds.

## The data

Denmark, 1974Q1–1987Q3: real money `LRM`, real income `LRY`, the bond
rate `IBO`, the deposit rate `IDE`. The canonical money-demand example
of Johansen and Juselius.

```pycon
>>> from pyardl.datasets import load_denmark
>>> d = load_denmark()
>>> y = d["LRM"]
>>> x = d[["LRY", "IBO", "IDE"]]
>>> d.shape
(55, 5)

```

## Step 1 — Integration orders: check nothing is I(2)

**Why.** The bounds test tolerates a mix of I(0) and I(1) regressors —
that is its whole appeal. It does *not* tolerate I(2): the critical
values are simply not derived for it, and a test run on I(2) data
returns a number with no distribution behind it.

So the pre-test is not there to classify every series. It is there to
rule out I(2).

```pycon
>>> from pyardl.unitroot import integration_order
>>> {c: integration_order(d[c])["order"] for c in ["LRM", "LRY", "IBO", "IDE"]}
{'LRM': 'I(1)', 'LRY': 'I(1)', 'IBO': 'I(1)', 'IDE': 'I(0)'}

```

A mix of I(1) and I(0), no I(2). This is precisely the case where the
bounds test earns its keep and Engle-Granger or Johansen cannot be used
directly.

**The common error.** Reading these as a licence to proceed *because
everything is I(1)*. The ARDL framework does not require that, and
insisting on it usually means differencing a series that did not need
it.

## Step 2 — No cointegration among the regressors

**Why.** The bounds test conditions on `x`. If two regressors are
themselves cointegrated, there is more than one long-run relation in the
system, and the single equation cannot identify the one you want.

```pycon
>>> from pyardl.cointegration import check_no_cointegration_among_x
>>> check_no_cointegration_among_x(x).selected_rank
0

```

Rank 0: no cointegrating vector among `LRY`, `IBO`, `IDE`. The
single-equation framework is legitimate here.

**The common error.** Skipping this entirely. It is the step most often
missing from applied papers, and when it fails the long-run coefficients
are not wrong so much as *meaningless* — they mix two relations.

## Step 3 and 4 — Order selection and the three-test bounds procedure

Order selection and the test are one call, because the orders must be
selected on the **common sample** for the information criteria to be
comparable at all.

```pycon
>>> from pyardl.bounds import bounds_test
>>> res = bounds_test(y, x, case=3, max_p=4, max_q=4, ic="aic")
>>> res.order
(3, {'LRY': 1, 'IBO': 3, 'IDE': 2})
>>> [round(float(v), 4) for v in (res.f_stat, res.t_stat, res.f_indep_stat)]
[6.2059, -4.5479, 8.1619]
>>> res.classification()[0]
'cointegration'

```

Three statistics, not one. `F_overall` tests the level terms jointly,
`t_BDM` tests that `y` actually adjusts, and `F_indep` tests that the
regressors carry the relation. All three reject here, which is the only
configuration that supports an unambiguous reading.

```pycon
>>> res.decision_f, res.decision_t, res.decision_indep
('cointegration', 'cointegration', 'cointegration')

```

**The common error, and it is the big one.** Concluding from `F_overall`
alone. A significant `F` with a non-significant `t` is a *degenerate
case*, not evidence of cointegration — see
[the three-test page](api/three-tests.md). pyardl returns the
classification rather than a bare `F` precisely so that reporting only
the `F` takes deliberate effort.

## Step 5 — The long run and the speed of adjustment

Only now, with cointegration established, does it make sense to read the
long-run coefficients.

```pycon
>>> from pyardl.core.ardl import ARDL
>>> p, q = res.order
>>> fit = ARDL(y, x, order=(p, q), det="const").fit()
>>> fit.longrun.round(4).to_dict()["theta"]
{'LRY': 0.9965, 'IBO': -4.5381, 'IDE': 2.8915}
>>> round(float(fit.adjustment["lambda"]), 4)
-0.4169
>>> round(float(fit.adjustment["half_life"]), 4)
1.2852

```

An income elasticity of **0.9965** — the textbook unit elasticity of
money demand, recovered without imposing it. The adjustment speed is
−0.42, so half of any disequilibrium is absorbed in about 1.3 quarters.

**The common error.** Reporting a confidence interval on `lambda`
*before* establishing cointegration. Under the null of no cointegration
that interval has no coverage at all; it is a number formatted like a
result.

## Step 6 — Diagnostics and stability

```pycon
>>> from pyardl.diagnostics import stability_tests
>>> design, y_dep, _ = fit.model._build_design()
>>> stability_tests(y_dep, design)["stable"].to_dict()
{'CUSUM': True, 'CUSUM-of-squares': True}

```

Both stable. A model that fails CUSUM-of-squares has a variance that
moves, and its long-run coefficients describe no single regime.

## Step 7 — Robustness: three ways of being wrong differently

A single test that rejects is a result. The same conclusion from
estimators that fail differently is an argument.

### Bootstrap critical values

Tabulated bounds are asymptotic. At `T = 55` that matters.

```pycon
>>> from pyardl.bootstrap import bootstrap_bounds_test
>>> bb = bootstrap_bounds_test(y, x, case=3, order=(p, q), n_boot=999, seed=42)
>>> [round(float(v), 4) for v in (bb.f_stat, bb.f_critical[0.05])]
[6.2059, 4.7272]
>>> bb.classification(0.05)[0]
'cointegration'

```

The bootstrap critical value (4.73) is *above* the tabulated upper bound
(4.32), so the asymptotic table was mildly optimistic — and the
conclusion survives anyway.

### Asymmetry: does the sign of a change matter?

```pycon
>>> from pyardl.nardl import NARDL
>>> na = NARDL(y, x, asym=["IBO"], order=(2, 1), case=3).fit()
>>> na.asymmetry_tests()["decision"].unique().tolist()
['symmetric']
>>> na.suggests_symmetric_model()
True

```

Every asymmetry test says symmetric. **The right conclusion is to go
back to the linear model** — a NARDL fitted here would spend parameters
to estimate a distinction the data do not support.

### Efficient long-run estimators

The fourth robustness route, and the one that most often changes the
reading. Static OLS on cointegrated data has invalid inference; DOLS,
FMOLS and CCR repair it three different ways, and
[their page](api/efficient.md) has the measurements.

```pycon
>>> from pyardl.cointegration import compare_longrun
>>> table = compare_longrun(y, x, ardl_results=fit, bandwidth=5,
...                         n_leads=2, n_lags=2)
>>> sorted(set(table.index.get_level_values("method")))
['ARDL', 'CCR', 'DOLS', 'FMOLS']
>>> [round(float(v), 4) for v in table.xs("IDE", level="regressor")["theta"]]
[2.8915, 2.6308, 4.0343, 4.0856]

```

The four agree on the sign and on significance, and disagree on
magnitude by a factor of 1.55 — the deposit-rate coefficient runs from
2.63 to 4.09.

**The common error.** Reporting whichever of these four rows supports
the story. A paper that shows one row is presenting a choice of
estimator as a finding; the spread is part of the result.

### Smooth structural change: read the pre-test first

```pycon
>>> from pyardl.fourier import fourier_bounds_test
>>> fb = fourier_bounds_test(y, x, case=3, order=(1, 1), n_sims=999, seed=42)
>>> fb.decision
'no_cointegration'
>>> fb.fourier_is_warranted
False

```

This is the most instructive result on the page, so read the two lines
together. The Fourier-ADL says **no cointegration** — flatly
contradicting every other test above. And its own pre-test says the
Fourier terms are **not warranted**: there is no smooth break in these
data.

So the verdict on the first line should not be read at all. Two
parameters were spent on a component that is not there, the test lost
power accordingly, and the resulting non-rejection is an artefact of the
specification rather than a fact about Denmark.

**The common error.** Running several tests and reporting the one that
agrees with the story. The pre-test exists so that "this test does not
apply here" is a checkable statement rather than a matter of taste.

## What this workflow does not cover

pyardl does not yet implement two parts of the standard sequence, and
saying so is more useful than a silent gap:

- **Dynamic simulations** in the sense of Jordan & Philips — the NARDL
  module has `dynamic_multipliers`, but the general simulation-based
  interpretation of an ARDL is absent.
- **Koyck and Almon** distributed lags — historical, not needed for the
  sequence above.

## Multiple asymmetries

Greenwood-Nimmo, Shin, van Treeck and Yu extend the partial-sum
decomposition to **several thresholds**, splitting changes into more
than two regimes — small rises, large rises, small falls, large falls.

pyardl exposes the building block rather than a dedicated API: apply
`partial_sums` at each threshold and pass the resulting columns as
ordinary regressors.

```pycon
>>> import numpy as np, pandas as pd
>>> from pyardl.nardl import partial_sums
>>> ibo = d["IBO"]
>>> lo_pos, lo_neg = partial_sums(ibo, threshold=0.0, name="IBO_lo")
>>> hi_pos, hi_neg = partial_sums(ibo, threshold=0.01, name="IBO_hi")
>>> multi = pd.DataFrame({
...     "IBO_small_up": lo_pos - hi_pos,
...     "IBO_large_up": hi_pos,
...     "IBO_down": lo_neg,
... })
>>> list(multi.columns)
['IBO_small_up', 'IBO_large_up', 'IBO_down']
>>> bool(np.isfinite(multi.to_numpy()).all())
True

```

`IBO_large_up` accumulates the rises above one percentage point;
`IBO_small_up` the rest. A non-zero threshold introduces a deterministic
drift into the decomposition — the library warns about it — so the
long-run coefficients are read net of a trend. That is a modelling
choice, not a detail, which is why there is no one-line API for it in
this version.

## Where to go next

- [The three-test framework](api/three-tests.md) — why one `F` is not enough.
- [Common mistakes](common-mistakes.md) — the same errors, collected.
- [Bootstrap or classical bounds?](bootstrap-or-bounds.md) — when the
  tables are not enough.
- [Glossary](glossary.md) — the vocabulary, in French and English.

## References

- Nkoro, E. & Uko, A. K. (2016). Autoregressive Distributed Lag (ARDL)
  cointegration technique: application and interpretation. *Journal of
  Statistical and Econometric Methods*, 5(4), 63-91.
- Greenwood-Nimmo, M., Shin, Y., van Treeck, T. & Yu, B. (2013). The
  decoupling of monetary policy from long-term rates in the U.S. and
  Germany. Working paper.
- Johansen, S. & Juselius, K. (1990). Maximum likelihood estimation and
  inference on cointegration. *Oxford Bulletin of Economics and
  Statistics*, 52(2), 169-210.
