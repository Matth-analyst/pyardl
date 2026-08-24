# Fourier terms — smooth structural change

`pyardl.fourier`

Structural breaks are usually handled by dating them: pick the number,
find the locations, add dummies. That works when breaks are sharp and
few, and it goes wrong when they are neither — the dates become
parameters estimated on the same data, and every test downstream
inherits their uncertainty.

Becker, Enders and Lee (2006) take the other route. Approximate the
time-varying deterministic component by a few low-frequency sinusoids:

```
d(t) ≈ a₀ + Σ_f [ a_f·sin(2πft/T) + b_f·cos(2πft/T) ]
```

Two parameters per frequency, no dates to estimate, and a component that
bends smoothly through several changes of unknown shape.

**The proviso is in the word *smooth*.** A Fourier component cannot
represent a jump; fitting one to a sharp break produces a wave that
overshoots on both sides of it. This is a method for gradual regime
change, not for a devaluation.

## The building block

```python
from pyardl.fourier import fourier_terms, select_frequency

terms = fourier_terms(200, 1.0)        # columns sin_1, cos_1
freq, table = select_frequency(y)      # best frequency by SSR, and the ranking
```

`f = 1` completes exactly one cycle over the sample; the period of `f` is
`T/f`. Frequencies beyond the Nyquist limit `T/2` are refused: sampled at
these dates they are indistinguishable from slower ones, so asking for
them is asking for a column that means something other than what it says.

### Integer and fractional frequencies are not the same object

At an **integer** frequency the sine and cosine are exactly orthogonal —
to each other, to the constant, and across frequencies (to 1e-14). At a
**fractional** one none of that holds: at `f = 0.5` on 200 observations
the sine sums to 127 instead of 0, so the component and the intercept
share the same variation and neither is separately interpretable.

The library supports both, because the literature uses both, and exposes
`fourier_orthogonality(n_obs, freq)` so the entanglement is a number you
can look at rather than a property you assume.

## The trap that governs everything here

Choosing the frequency on the data is not estimation. Under the null
that the Fourier terms are absent, `f` is **not identified** — there is
no true value for it to converge to. Picking the best of five is a
*search*, and the statistic at the winning frequency is a maximum over a
grid, not a draw from a fixed distribution. That is the Davies problem,
and the size of the error is not subtle:

| frequency | rejection at a nominal 5% |
|---|---|
| fixed in advance | 4.8% |
| **selected on the data** | **24.6%** |

*(2000 replications, T = 200, white-noise null, integer grid 1–5.)*

One rejection in four where five in a hundred were promised. The correct
95% quantile is **5.05** against the tabulated **3.04**.

A live illustration from the test runs: on one white-noise sample the
statistic came out at **3.52** — above the tabulated 3.04, so rejected by
the textbook route, and comfortably below the true critical value of
4.83.

So both tests here simulate their own critical values **with the search
inside the loop**: each replication re-runs the frequency selection on
its own null sample, exactly as the real call does. The result carries
`freq_estimated`, and `summary()` names which construction was used.

## Is a smooth component there at all?

```python
from pyardl.fourier import fourier_f_test

res = fourier_f_test(y, n_sims=2000, seed=42)
print(res.summary())
```

```text
Fourier F test (Becker, Enders & Lee 2006) - 200 observations, frequency 1 (selected)
  critical values: simulated WITH the frequency search inside the loop, n_sims=2000, seed=42

  statistic = 393.7174   simulated p = 0.0005   decision (5%): reject

    alpha    critical
      0.1      3.8811
     0.05      4.4725
     0.01      5.7306
```

`H₀: a_f = b_f = 0`. Set `freq_estimated=False` with an explicit `freq`
when the frequency comes from theory rather than from the data — the
critical values are then simulated at that fixed frequency, and they
land back on the tabulated F, which the test suite checks.

## Stationarity around a moving mean

```python
from pyardl.fourier import fourier_kpss

res = fourier_kpss(y, n_sims=2000, seed=42)
```

KPSS in spirit: the null is **stationarity**, and large values reject it.
That direction is the opposite of a unit-root test's, and mixing the two
up is the classic way to report the reverse of what the data say.

Its use is as a pre-test. A series that looks non-stationary to an ADF
may simply be stationary around a mean that drifts, and this tells the
two apart without dating anything.

### What it buys, measured — and what it does not

A single Fourier frequency captures most of a logistic break, not all of
it: the R² of the fitted component tops out at **0.86** with one
frequency and **0.88** with two, never the 0.9 the literature suggests.
The unexplained remainder is small but **persistent**, and persistence is
exactly what a KPSS detects.

So on a series that is stationary around a smooth break, the Fourier
KPSS still rejects. The honest comparison is not against the threshold
but against the plain KPSS — what you would run without any of this:

| break amplitude | plain KPSS | Fourier, F = 1 | reduction |
|---|---|---|---|
| 0.5 | 2.468 | 0.424 | **83%** |
| 1.0 | 3.494 | 0.853 | 76% |
| 2.0 | 3.902 | 1.241 | 68% |
| 3.0 | 3.988 | 1.370 | 66% |

*(simulated 5% critical value: 0.288)*

The component removes two thirds to four fifths of the distortion. It
does not remove it. More frequencies help monotonically — 1.37, then
0.83, then 0.58 — and still do not cross the threshold.

Recorded as OBS-15 and OBS-16 in the validation register, with the
measurements that produced these numbers.

## Cost

Both tests simulate their null distribution on every call, because there
is no table that could cover the sample size, the grid, the deterministic
terms and whether the frequency was searched for. `n_sims=2000` is a
couple of seconds; the seed is recorded when you do not supply one, so a
critical value can always be reproduced.

## References

- Becker, R., Enders, W. & Lee, J. (2006). A stationarity test in the
  presence of an unknown number of smooth breaks. *Journal of Time
  Series Analysis*, 27(3), 381-409.
- Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is
  present only under the alternative. *Biometrika*, 74(1), 33-43.
