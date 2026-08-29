# Common mistakes in applied ARDL work

Six errors that recur in published bounds-test applications. Each one is
shown here running, on real or simulated data, with the number it
produces — and with what pyardl does to make committing it require
deliberate effort rather than mere inattention.

**Every `>>>` block is executed in CI.** These are not cautionary tales;
they are reproducible.

```pycon
>>> import warnings
>>> import numpy as np, pandas as pd
>>> from pyardl.datasets import load_denmark
>>> d = load_denmark()
>>> y, x = d["LRM"], d[["LRY", "IBO", "IDE"]]

```

## 1. Concluding from the overall F alone

**The error.** Report `F_overall`, find it above the upper bound,
declare cointegration. This is the single most common mistake in the
literature, and it is not conservative — it is simply wrong in a
specific, nameable way.

`F_overall` tests that the level terms are *jointly* zero. It rejects in
two situations that mean different things: a genuine long-run relation,
and a **degenerate case** where the regressors carry level effects but
`y` does not adjust to them. In the second, there is no error-correction
mechanism and nothing to interpret as a long run.

Sam, McNown and Goh add two more tests to separate them: `t_BDM` on the
adjustment coefficient, and `F_indep` on the regressors' level terms.

```pycon
>>> from pyardl.bounds import bounds_test
>>> res = bounds_test(y, x, case=3, max_p=4, max_q=4, ic="aic")
>>> res.decision_f, res.decision_t, res.decision_indep
('cointegration', 'cointegration', 'cointegration')

```

**What pyardl does.** `bounds_test` computes all three whether you ask or
not, and `classification()` returns a named verdict rather than a bare
statistic. To report only the `F` you have to reach past the
classification and pull out `f_stat` — which is possible, but it is no
longer the path of least resistance.

```pycon
>>> res.classification()[0]
'cointegration'

```

The classification also distinguishes the degenerate cases by name, so a
result that is *not* cointegration says which kind it is instead of
falling back to a vague "inconclusive".

## 2. A confidence interval on the adjustment speed before cointegration is established

**The error.** Fit the ECM, read `lambda` and its standard error, report
`lambda = -0.31 (0.09)` — without having established that a long-run
relation exists.

Under the null of no cointegration, the `t` on `lambda` does **not**
follow a Student distribution. Its distribution is non-standard, which
is exactly why `t_BDM` needs its own tabulated bounds. So the reported
standard error has no coverage: it is a number formatted like a result.

```pycon
>>> from pyardl.core import ARDL
>>> p, q = res.order
>>> fit = ARDL(y, x, order=(p, q), det="const").fit()
>>> round(float(fit.adjustment["lambda"]), 4)
-0.4169

```

Here it *is* legitimate — step 4 established cointegration on all three
tests first. That ordering is the whole point.

**What pyardl does.** The bounds test is a separate call from the ECM
estimation, so the sequence is visible in your own code. The library
cannot stop you from inverting the order, but the
[workflow page](workflow.md) puts the test before the interval and says
why.

## 3. Mixing up the five deterministic cases between software packages

**The error.** Running case 3 in one package and comparing to case 2 in
another, or taking a critical value from a table that numbers the cases
differently. The five PSS cases differ in whether the intercept and
trend are **restricted** into the level relation or left unrestricted,
and cases II and IV put the restricted deterministic *inside the tested
vector* — so the test statistic itself changes, not just the critical
value.

```pycon
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")
...     stats = {c: round(float(bounds_test(y, x, case=c, order=(p, q)).f_stat), 4)
...              for c in (1, 2, 3, 4, 5)}
>>> stats
{1: 0.7109, 2: 5.1168, 3: 6.2059, 4: 5.4306, 5: 6.7853}

```

The same data, the same lag orders, give `F` anywhere from **0.71 to
6.79** depending on the case. Case 1 — no deterministic terms at all —
produces 0.71 against 6.21 for case 3: not a different critical value,
a different test entirely.

Choosing the case is a modelling decision about the deterministic terms,
not a robustness knob. Comparing a statistic computed under one case to
a table published for another is comparing two different tests, and the
table will not object.

Note also the warning cases 2 and 4 emit: PSS do not tabulate `t` when
the deterministics are restricted, so no `t` decision exists there. The
library says so instead of returning a bound from a neighbouring case.

**What pyardl does.** All five cases are implemented exhaustively, never
silently substituted, and `case` is a required-by-default argument on
every bounds call. The result object records which case produced it.

## 4. Asymptotic critical values on a short sample

**The error.** Using the PSS asymptotic bounds at `T = 35`. They are
derived for `T -> infinity`; at macro sample sizes they are too
permissive, and the test over-rejects.

```pycon
>>> from pyardl.bootstrap import bootstrap_bounds_test
>>> bb = bootstrap_bounds_test(y, x, case=3, order=(p, q), n_boot=999, seed=42)
>>> round(float(res.bounds.loc[0.05, "F_I1"]), 4)
4.3223
>>> round(float(bb.f_critical[0.05]), 4)
4.7272

```

On these 55 observations the bootstrap critical value is **4.73** where
the asymptotic upper bound is **4.32** — nearly 10% higher. A statistic
landing between them would be called cointegration by the table and
rejected by the bootstrap. Here `F = 6.21` clears both, so the
conclusion holds; that is a fact about this dataset, not a general
reassurance.

**What pyardl does.** Finite-sample critical values are available from
response surfaces (`cv_source="kripfganz"`, the default) and from the
bootstrap. The default is already not the asymptotic table.

## 5. Undetected I(2) variables

**The error.** Running a bounds test on data containing an I(2) series.
The framework is derived for regressors that are I(0) or I(1) in any
mix; for I(2) the critical values do not exist, and the statistic has no
distribution behind it.

```pycon
>>> from pyardl.unitroot import integration_order
>>> {c: integration_order(d[c])["order"] for c in ["LRM", "LRY", "IBO", "IDE"]}
{'LRM': 'I(1)', 'LRY': 'I(1)', 'IBO': 'I(1)', 'IDE': 'I(0)'}

```

None is I(2). Note also that the mix of I(1) and I(0) is *fine* — the
frequent belief that everything must be I(1) is itself the mirror-image
error, and it leads people to difference a stationary series.

**What pyardl does.** `integration_order` runs DF-GLS and Ng-Perron and
returns the order as a label, so the check is one line and its outcome is
a value you can assert on rather than a table you eyeball.

## 6. A "significant" t with a positive lambda

**The error.** Reading the `t` on the adjustment coefficient as
two-sided, finding it large in absolute value, and declaring
error-correction — when `lambda` is **positive**.

`t_BDM` is a **left-tailed** test. A positive `lambda` means
disequilibrium *grows*: the system diverges. A large positive `t` is
evidence against error correction, not for it, and reporting it as
significant inverts the conclusion.

```pycon
>>> rng = np.random.default_rng(0)
>>> n = 120
>>> xx = np.cumsum(rng.normal(size=n))
>>> yy = np.zeros(n)
>>> for t in range(1, n):
...     yy[t] = yy[t-1] + 0.05 * (yy[t-1] - 0.8 * xx[t-1]) + rng.normal(scale=0.3)
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")
...     div = bounds_test(pd.Series(yy, name="y"),
...                       pd.DataFrame({"x": xx}), case=3, order=(1, 1))
>>> bool(div.t_stat > 0)
True
>>> div.decision_t
'no_cointegration'

```

The adjustment coefficient is positive here by construction, the `t` is
positive, and the left-tailed test correctly refuses to reject. A
two-sided reading of the same number would have called this
cointegration.

**What pyardl does.** The test is implemented one-sided, in the correct
direction, and the decision is returned as a word rather than left to be
derived from a statistic and a table. There is no way to get
`'cointegration'` out of a positive `t`.

## The pattern

Five of these six share a shape: a quantity is computed correctly, and
then read as if it answered a different question. The defence is not
more arithmetic — it is returning **named verdicts** instead of bare
numbers, so that the question being answered travels with the answer.

That is why `bounds_test` returns a classification, why `cd_test` asks
whether you are reading it before or after augmentation, and why the
Fourier test carries a pre-test saying whether it applies at all.

## See also

- [The complete workflow](workflow.md) — the same steps in order, on the
  same data.
- [The three-test framework](api/three-tests.md) — the degenerate cases
  in detail.
- [Bootstrap or classical bounds?](bootstrap-or-bounds.md) — when the
  tables are not enough.
- [Validation register](VALIDATION_OBSERVATIONS.md) — the errors this
  project itself made, and how each was caught.
