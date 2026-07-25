# ARDL ↔ error-correction algebra

`pyardl.core.transforms`

An ARDL model and its error-correction form are two ways of writing the
same regression. The conversion is exact: fit either representation on
the same data and you get identical residuals, identical sum of squared
residuals, identical fit. Nothing is approximated.

```
ARDL:  y_t = α + δt + Σ φ_i y_{t-i} + Σ_j Σ_i β_{j,i} x_{j,t-i} + ε_t

ECM:   Δy_t = α + δt + λ y_{t-1} + Σ_j γ_j x_{j,t-1}
              + Σ ψ_i Δy_{t-i} + Σ_j Σ_i ω_{j,i} Δx_{j,t-i} + ε_t
```

The point of the second form is interpretation. It separates the
short-run dynamics (the differenced terms) from the long-run
relationship (the level terms), and it puts two economically meaningful
quantities in plain sight:

- **λ**, the speed of adjustment — the fraction of last period's
  disequilibrium corrected each period. Negative under error correction;
  the closer to -1, the faster.
- **θ_j = -γ_j / λ**, the long-run coefficient — the elasticity of `y`
  with respect to `x_j` once everything has settled.

Most users reach this through `ARDLResults.to_ecm()`, `.longrun` and
`.adjustment` rather than calling these functions directly.

## Conversions

### `ardl_to_ecm(params) -> ECMParams`

```python
import numpy as np
from pyardl.core.transforms import ARDLParams, ardl_to_ecm

p = ARDLParams(p=1, q=(1,), phi=np.array([0.5]), beta=(np.array([0.3, 0.2]),))
ecm = ardl_to_ecm(p)
ecm.lam        # -0.5
ecm.gamma[0]   # 0.5
```

### `ecm_to_ardl(params) -> ARDLParams`

Exact inverse. Useful for simulation and bootstrap, where it is natural
to generate data in one representation and test in the other.

## Long-run quantities

### `speed_of_adjustment(params) -> float`

`λ = -(1 - Σ φ_i)`.

### `longrun_coefs(params, tol=1e-8) -> Series`

`θ_j = Σ β_{j,i} / (1 - Σ φ_i)`, one per regressor.

If `|λ| < tol` there is no error-correction force and the long-run
coefficients are not defined: returns NaNs with a
`DegenerateCaseWarning` rather than a meaningless number produced by
dividing by something near zero.

### `longrun_covariance(params, v=None) -> ndarray`

Covariance of the long-run coefficients by the delta method, using the
analytical gradient

```
∂θ_j/∂β_{j,i} = 1 / (1 - Σφ)
∂θ_j/∂φ_i     = θ_j / (1 - Σφ)
```

Requires `cov_params` on the parameter object, or a covariance matrix
passed as `v`.

### `half_life(params) -> float`

`ln(0.5) / ln(1 + λ)` — the number of periods needed to absorb half of a
shock. Only meaningful when `-1 < λ < 0`; otherwise returns NaN with a
`DegenerateCaseWarning`, since there is no geometric convergence to
speak of.

## Containers

### `ARDLParams`

`p`, `q`, `phi`, `beta`, `const`, `trend`, `has_const`, `has_trend`,
`x_names`, `cov_params`.

`param_vector()` stacks the parameters as `const?, trend?, phi,
beta[0], beta[1], ...`. This ordering is the contract used by
`cov_params` and by `longrun_covariance`.

### `ECMParams`

`p`, `q`, `lam`, `gamma`, `psi`, `omega`, plus the same deterministic
and naming fields.

!!! note "When `q_j = 0`"
    `omega[j]` is empty and `gamma_j` multiplies the *contemporaneous*
    level `x_{j,t}` rather than `x_{j,t-1}`. Using a lagged level would
    give the error-correction form one degree of freedom more than the
    original ARDL, and the two would no longer share the same residuals.

    This matches Stata's `ardl`. `statsmodels.tsa.ardl.UECM` rejects
    `q_j = 0` outright; `pyardl` supports it.
