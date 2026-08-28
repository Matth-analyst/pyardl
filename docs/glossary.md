# Glossary — English, French, notation

The documentation of this project is written in English for the code and
its docstrings, and the vocabulary of the ARDL literature is not always
translated consistently between the two languages. This page fixes one
term per concept, so that the same thing is called the same thing
everywhere.

Where a term is genuinely contested — and two of them are — the entry
says so instead of picking silently.

## Core objects

| English | Français | Notation | What it is |
|---|---|---|---|
| autoregressive distributed lag | modèle autorégressif à retards échelonnés | ARDL(p, q) | `y` on its own `p` lags and `q` lags of each regressor |
| error-correction model | modèle à correction d'erreur | ECM / UECM | The exact reparameterisation of an ARDL in differences plus a level term |
| unrestricted ECM | MCE non contraint | UECM | The form the bounds test is run on; no long-run coefficient imposed |
| long-run coefficient | coefficient de long terme | `θ` (theta) | `Σβⱼ / (1 − Σφₛ)` — the eventual effect of a permanent unit change |
| adjustment speed / error-correction coefficient | vitesse d'ajustement / coefficient de correction d'erreur | `λ` (lambda) | Share of disequilibrium absorbed per period. **Must be negative.** |
| half-life | demi-vie | — | Periods to absorb half a shock, `ln(0.5)/ln(1+λ)` |
| deterministic case | cas déterministe | I–V | Which of intercept and trend are restricted into the level relation |

## The tests

| English | Français | Notation | What it is |
|---|---|---|---|
| bounds test | test aux bornes | `F_overall` | Joint test that the level terms are zero |
| BDM t-test | test t de BDM | `t_BDM` | **Left-tailed** test on `λ`. Banerjee, Dolado & Mestre |
| independence test | test d'indépendance | `F_indep` | Joint test on the regressors' level terms only |
| lower / upper bound | borne inférieure / supérieure | I(0) / I(1) | The two critical values bracketing the unknown integration mix |
| inconclusive region | zone non concluante | — | Statistic between the bounds. **A third state, never a boolean.** |
| degenerate case | dégénérescence | type 1 / type 2 | `F` rejects but the relation is not a long run — see below |

### The two degeneracies, precisely

They are easy to confuse because both look like "F rejected but
something is off".

- **Degenerate case 1**: `F_overall` rejects, `t_BDM` does not. The
  level terms are jointly significant but `y` does not adjust. There is
  no error-correction mechanism, so nothing to read as a long run.
- **Degenerate case 2**: `F_overall` and `t_BDM` reject, `F_indep` does
  not. `y` adjusts to its own lagged level, but the regressors carry no
  level effect — the "relation" involves only `y`.

## Cross-sectional dependence

| English | Français | Notation | What it is |
|---|---|---|---|
| common factor | facteur commun | `f_t` | Unobserved shock hitting every individual |
| factor loading | chargement / saturation | `γᵢ` | How strongly individual `i` responds to the factor |
| cross-sectional average | moyenne transversale | `z̄_t` | Mean over individuals at date `t`; proxies the factor space |
| CD test | test CD | — | Pesaran's test. Null is **no** dependence — direction matters |
| between-individual variance | variance inter-individus | — | The Mean Group standard error. **Not** pooled from within |

## Panel estimators

| English | Français | What it constrains |
|---|---|---|
| Mean Group (MG) | Mean Group | Nothing — estimate each individual, then average |
| Pooled Mean Group (PMG) | Pooled Mean Group | Long run common, short run free |
| dynamic fixed effects (DFE) | effets fixes dynamiques | Everything but the intercepts |
| CS-ARDL | CS-ARDL | MG plus cross-sectional averages and their lags |
| CS-DL | CS-DL | Long run read directly, no dynamics estimated |

## Extensions

| English | Français | What it adds |
|---|---|---|
| nonlinear ARDL (NARDL) | ARDL non linéaire | Partial sums split rises from falls |
| partial sum | somme partielle | `x⁺`, `x⁻`: cumulated positive and negative changes |
| quantile ARDL (QARDL) | ARDL quantile | The relation estimated across the conditional distribution |
| Fourier terms | termes de Fourier | Sinusoids approximating a smooth break, no dates estimated |
| Davies problem | problème de Davies | A parameter unidentified under the null — searching for it invalidates tabulated critical values |

## Two terms this project uses deliberately

**"Inconclusive", not "fail to reject".** The bounds test has three
outcomes, not two, because the statistic can fall between the I(0) and
I(1) bounds. Collapsing that into a boolean loses the only information
the region carries: that the answer depends on an integration order you
have not established. Every decision in pyardl is one of
`'cointegration'`, `'no_cointegration'`, `'inconclusive'`.

**"Classification", not "decision".** The three-test framework does not
produce a yes/no; it produces one of several named states, including the
two degeneracies. `bounds_test(...).classification()` returns that name
plus the sentence saying which test decided — because the name alone
says *what*, and a reader needs *why* to judge whether to believe it.

## Notation conventions in this documentation

| Symbol | Meaning |
|---|---|
| `p` | Lags of the dependent variable |
| `q` | Lags of a regressor (may differ per regressor) |
| `k` | Number of regressors |
| `T` | Time-series length |
| `N` | Number of individuals in a panel |
| `φ` (phi) | Autoregressive coefficients of the ARDL |
| `β` (beta) | Distributed-lag coefficients |
| `θ` (theta) | Long-run coefficient |
| `λ` (lambda) | Adjustment speed, negative under error correction |
| `α` (alpha) | Significance level |
| `Δ` | First difference |

## See also

- [The complete workflow](workflow.md) — these terms in sequence, on data.
- [Common mistakes](common-mistakes.md) — what goes wrong when two of
  them are confused.
