# Johansen test

`pyardl.cointegration`

The bounds test asks whether **one** long-run relationship holds between
a dependent variable and its regressors. Johansen's procedure asks how
many hold among a set of variables treated symmetrically. It works on
the VECM

```
Δy_t = Π y_{t-1} + Σ Γ_i Δy_{t-i} + d_t + ε_t
```

where the **rank of Π is the number of cointegrating relations**: rank 0
means none, full rank means the system was stationary to begin with, and
anything in between counts the relations.

`pyardl` does not reimplement it. The computation comes from
`statsmodels.tsa.vector_ar.vecm.coint_johansen`, which is mature and
validated; reimplementing it would add risk and buy nothing. What is
added here is the sequential decision, the result object, and the
diagnostic the bounds framework needs.

## `johansen(y, det_order=0, k_ar_diff=1, alpha=0.05, method="trace")`

```python
from pyardl.cointegration import johansen
from pyardl.datasets import load_denmark

data = load_denmark()[["LRM", "LRY", "IBO", "IDE"]]
res = johansen(data, det_order=0, k_ar_diff=1)
print(res.summary())
```

```text
Johansen test (1988, 1991) - 4 variables (LRM, LRY, IBO, IDE), det_order=0, k_ar_diff=1

        H0       trace     cv 5%      maxeig     cv 5%
     r = 0     48.8037   47.8545     31.5136   27.5858
    r <= 1     17.2902   29.7961     10.1453   21.1314
    r <= 2      7.1449   15.4943      6.5889   14.2639
    r <= 3      0.5560    3.8415      0.5560    3.8415

selected rank (trace, 5%): 1
```

`k_ar_diff` is the number of lagged **differences** in the VECM, that is
`p - 1` where `p` is the lag order of the VAR in levels. Both statistics
are always computed; `method` only chooses which one sets
`selected_rank`, and `res.rank(method=..., alpha=...)` re-reads the
decision without re-estimating.

## The sequential rule

Test `r = 0`, then `r ≤ 1`, and so on, and **stop at the first
non-rejection**. Continuing past it — taking the last rejection, or the
largest rank that rejects — is a different procedure with a different
size, and it is the classic way to misread an implementation that
returns statistics only.

statsmodels returns the statistics; the loop lives here, and it is
checked against `select_coint_rank` at every level and both methods.

## Which statistic to trust

Measured on 1000 replications, VECM of rank 1, three variables, T = 200:

| statistic | rank 1 retained | rank over-stated | rank missed |
|---|---|---|---|
| trace | 87.8% | 12.2% | **0%** |
| maxeig | 92.5% | 7.5% | **0%** |

The trace test **over-selects**: it adds directions, it never removes
them. Under a rank-0 DGP it claims at least one relation 8.3% of the
time against a nominal 5%, where `maxeig` gives 5.8%.

The default stays `trace`, because that is what the applied literature
reports. But a rank retained by the trace deserves a second reading by
`maxeig` — and since both are always computed, that costs nothing. The
full figures are in the project's validation register (OBS-10), kept
with the specifications rather than shipped.

## Deterministic conventions — read this before comparing with R

The naming across implementations is a trap, and the difference is not
cosmetic: it changes the statistics *and* the critical values. The
correspondence below was **established by running both sides** — six
`urca::ca.jo` variants (3 `ecdet` × 2 `spec`) against three
`det_order` values — not by reading either manual.

| pyardl / statsmodels | meaning | `urca::ca.jo` |
|---|---|---|
| `det_order=-1` | no deterministic term at all | no equivalent |
| `det_order=0` | **unrestricted constant** in the VECM | `ecdet="none"` |
| `det_order=1` | linear trend | no exact equivalent |

Two things follow, and both bite in practice:

- `ecdet="none"` in `urca` does **not** mean "no deterministic terms".
  It means the constant stays unrestricted in the VECM. It matches
  `det_order=0`, not `det_order=-1`.
- `ecdet="const"` and `ecdet="trend"` put the term **inside** the
  cointegrating space (restricted). `coint_johansen` has no equivalent,
  so no pairing is offered rather than an approximate one.

`urca`'s `spec` argument (`"transitory"` vs `"longrun"`) changes the
parameterisation of the `Γ` matrices but **not** the test statistics —
verified, both agree with ours to 1e-9.

One caveat that survives the correspondence: the **critical values come
from different tabulations** (MacKinnon-Haug-Michelis in statsmodels,
Osterwald-Lenum in `urca`). On the Danish data the 5% trace bound for
`r = 0` is 47.85 here against 48.28 there — under 1%, but enough to flip
a decision when the statistic lands between them, as it nearly does
here (48.80).

## `check_no_cointegration_among_x(x, ...)`

The bounds test conditions on the regressors and tests for a *single*
relation, the one involving `y`. If the regressors are cointegrated
among themselves, that premise fails and the tabulated distribution no
longer applies — and the bounds test gives no sign of it. Hence a
separate check:

```python
from pyardl.cointegration import check_no_cointegration_among_x

res = check_no_cointegration_among_x(data[["LRY", "IBO", "IDE"]])
```

It returns the full `JohansenResults` so you can inspect it, `None` when
there is nothing to test (a single regressor), and raises a
`PyardlMethodologyWarning` naming the number of relations found when the
retained rank exceeds zero.

## Cointegrating vectors

`res.beta` holds one vector per column, each normalised so its first
element is 1. The normalisation is arbitrary — only the *space* the
vectors span is identified — and it exists so two runs can be compared
at all. A first element numerically indistinguishable from zero is left
unscaled rather than divided by, which would manufacture enormous
coefficients out of noise.

## Limits, stated rather than worked around

- Critical values are tabulated for at most **12 variables**; beyond
  that `johansen` raises rather than returning an undecidable rank.
- `alpha` is one of 0.10, 0.05, 0.01 — the levels that are tabulated. No
  interpolation is offered.

## References

- Johansen, S. (1988). Statistical analysis of cointegration vectors.
  *Journal of Economic Dynamics and Control*, 12, 231-254.
- Johansen, S. (1991). Estimation and hypothesis testing of
  cointegration vectors in Gaussian vector autoregressive models.
  *Econometrica*, 59(6), 1551-1580.
