# Fourier-ADL — cointegration with a smooth break

`pyardl.fourier.fourier_bounds_test`

The bounds test asks whether a level relation holds with a *constant*
intercept. When the intercept drifts — a gradual change of regime, a
slow institutional shift — the drift has nowhere to go but the residual,
the residual looks persistent, and the test concludes there is no error
correction when there is one.

Banerjee, Arčabić and Lee (2017) put the drift back into the
specification instead: the UECM is augmented with the low-frequency
sinusoids of [Becker, Enders and Lee](fourier.md), and the test is the
usual left-tailed `t` on the error-correction coefficient.

```
Δyₜ = a₀ + Σ_f [a_f·sin(2πft/T) + b_f·cos(2πft/T)]
      + λ·yₜ₋₁ + θ'xₜ₋₁ + short-run terms + εₜ
```

`H₀: λ = 0` against `λ < 0`. The sinusoids sit in the **design**, never
in the tested vector: they are deterministic terms of the specification,
not part of the level relation. Putting them in the tested vector would
change the hypothesis.

## Use

```python
from pyardl.fourier import fourier_bounds_test

res = fourier_bounds_test(y, x, case=3, n_sims=2000, seed=42)
print(res.summary())
```

```text
Fourier-ADL cointegration test (Banerjee, Arcabic & Lee 2017) - case 3, k=1, ECM(1; x:1)
  frequency 1 (selected), 149 observations
  critical values: simulated WITH the frequency search inside the loop, n_sims=2000, seed=42

  t_BDM = -11.0935   simulated p = 0.0005   decision (5%): cointegration

    alpha    critical
      0.1     -3.9263
     0.05     -4.2303
     0.01     -4.8164

  pre-test on the Fourier terms: F = 44.1613, p = 0.0005, critical (5%) = 7.9807
  The Fourier terms are significant: a smooth break is present and this test is the right one for it.
```

*(T = 150, λ = −0.3, logistic break of amplitude 4 in the long-run
intercept, seed 20260824.)*

`res.selection` carries the whole grid, so the margin of the winning
frequency is visible rather than implied:

```text
 freq       ssr
  1.0 26.667115
  3.0 41.871068
  4.0 42.767916
  2.0 42.872908
  5.0 43.078421
```

All five PSS deterministic cases are supported, and `freq=1.0` fixes the
frequency when it comes from theory instead of from the data.

## Two reasons the critical values are not standard

The regressors are integrated — the ordinary reason — **and** the
frequency was chosen on the sample. The second is the Davies problem
inherited from [the Fourier page](fourier.md), and it is not a detail:
searching a grid of five stretches the left tail, so the true critical
value is more negative than the one for a fixed frequency. The test
suite checks that ordering directly.

Every call therefore simulates its own null distribution with the
selection re-run inside each replication, exactly as the real call does.
There is no table that could cover the sample size, the grid, the
deterministic case and whether a search took place.

## The pre-test, and why it is not the standalone Fourier F test

`fourier_is_warranted` answers a question the main test cannot: **are the
two extra parameters buying anything?** With no break they are spent for
nothing, and the plain bounds test — with published critical values and
no simulation — is the better instrument. `recommendation` says so in
words.

The first implementation of this pre-test called `fourier_f_test` on `y`.
That was wrong in a way worth recording. `fourier_f_test` simulates its
null on **white noise**, while `y` here is **integrated**: the statistic
was therefore enormous on every sample, and the pre-test declared the
Fourier terms significant in **100% of replications, break or no break**.
A pre-test that always says yes is not a pre-test.

It now compares two fits of the *same* model — with and without the
sinusoids — and reads the F against the null simulated in the same loop.
On the illustration above the 5% critical value is **7.98**, against a
tabulated `F(2, ·)` of **3.08**: the gap is the whole point. Measured
behaviour on the same DGP: **p = 0.98** with no break, **p = 0.0005**
with a break of 4. Recorded as OBS-18.

Across the power study (150 replications, T = 100), the corrected
pre-test now flags the Fourier terms as significant in:

| break amplitude | pre-test significant |
|---|---|
| 0 | 0% |
| 3 | 13% |
| 6 | 35% |

Monotone in the break, and zero when there is none — the shape the old
version could not produce. Note the level: at T = 100 the pre-test misses
a break of 6 about two times in three. It is a guard against spending
parameters on nothing, not a reliable detector of a break that is there.

## What it buys, measured

This is where the honest answer diverges from the advertised one.

**Size, under two independent random walks** (200 replications, T = 100,
nominal 5%):

| test | rejection |
|---|---|
| bounds test, tabulated PSS bounds | 5.0% |
| Fourier-ADL, simulated CV + search | 3.5% |

Both are correctly sized, which is what makes the power comparison below
a fair one rather than a comparison of two different tests.

**Power, under true cointegration with a smooth logistic break**
(150 replications, T = 100, λ = −0.15; standard error ≈ 3.7 points):

| break amplitude | bounds test | Fourier-ADL |
|---|---|---|
| 0 | 99% | 93% |
| 3 | 91% | 77% |
| 6 | 59% | 59% |

The Fourier-ADL does **not** dominate. It costs six points where there is
no break — the price of two parameters, as expected — and it does not
recover the loss at intermediate amplitudes. A four-configuration scan
over amplitude, break sharpness and adjustment speed found every
difference within ±5 points, i.e. inside the noise.

This contradicts the gain claimed for the method, and the register says
so rather than the reverse: OBS-17, with the DGPs that produced it.

**What that means in practice.** Reach for this test when the smooth
break is part of what you are modelling — you want the drifting
intercept estimated, and the pre-test confirms it is there. Do not reach
for it expecting free power against a plain bounds test; on the DGPs
measured here, there is none to collect.

The calibration of that comparison deserves one line, because it is the
kind of choice that silently decides a result. A first attempt at
λ = −0.4 and T = 150 put **both** tests at 100% everywhere, including
with a break of 5 — a comparison between two ceilings says nothing. The
DGP was recalibrated to place the **reference** test in a range where it
is not already perfect. The calibration targets the test being compared
*against*, never the one being evaluated.

## Cost

One call simulates `n_sims` null samples, each re-running the frequency
search — a few seconds at `n_sims=2000`. The seed is recorded when you do
not pass one, so any critical value in a paper can be reproduced.

## References

- Banerjee, P., Arčabić, V. & Lee, H. (2017). Fourier ADL cointegration
  test to approximate smooth breaks with new evidence from crude oil
  market. *Economic Modelling*, 67, 114-124.
- Becker, R., Enders, W. & Lee, J. (2006). A stationarity test in the
  presence of an unknown number of smooth breaks. *Journal of Time
  Series Analysis*, 27(3), 381-409.
- Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is
  present only under the alternative. *Biometrika*, 74(1), 33-43.
- Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). Bounds testing
  approaches to the analysis of level relationships. *Journal of Applied
  Econometrics*, 16(3), 289-326.
