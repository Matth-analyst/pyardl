# `ardlpy.critical_values` — valeurs critiques et moteur de simulation

Références : Pesaran, Shin & Smith (2001) — spec 10 ; Narayan (2005),
*Applied Economics* 37(17) — spec
[12_small_sample_critical_values](../references/12_small_sample_critical_values.md).

## Pourquoi des CV petits échantillons

Les bornes asymptotiques de PSS (T = 1000) sont **trop libérales** quand
T ∈ [30, 80] — le cas typique des données annuelles : sur-rejet de H₀,
donc fausses cointégrations. Narayan (2005) a simulé des bornes exactes
par taille d'échantillon (T = 30..80 par pas de 5, cas II/III/V,
k ≤ 7, F seulement).

## Politique de sources (spec 12 §2.4)

| `cv_source` | Fonction | Couverture |
|---|---|---|
| `"pss"` | **reproduction de la littérature** : valeurs publiées servies à l'identique, erreur MC d'origine documentée (~±0.05, jusqu'à ±0.15 à 1 %) | cas I-V, k ≤ 10, F et t, 10/5/2.5*/1 % |
| `"narayan"` | valeurs publiées, 30 ≤ T ≤ 80 (données annuelles courtes) | cas II/III/V, k ≤ 7, F seulement, 10/5/1 % |
| `"kripfganz"` | **DÉFAUT (spec 13, voie A1)** : CV asymptotiques précis (32M réplications) à tout seuil + p-values aux deux bornes, via statsmodels (BSD, rien redistribué) | cas I-V, k = 1..10, F seulement (t : PSS) ; fini-T -> voie A2/B |

\* le seuil 2.5 % provient du moteur de simulation interne (voir
PROVENANCE.md), pas de la transcription des tables publiées.

```python
from ardlpy.critical_values import get_bounds

get_bounds("F", case=3, k=1, alpha=0.05)                                # PSS
get_bounds("F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=47) # interpolé
```

## Surfaces finies-T (voie A2, spec 13)

`bounds_test(finite_t=True)` évalue directement les **coefficients
publiés** de Kripfganz & Schneider (fichier `ardl_surfreg_coefs.dta`,
distribué avec leur package Stata `ardl`) : CV et p-values pour F **et
t**, ajustés à la taille d'échantillon réelle et au nombre de
coefficients de court terme — c'est la source la plus précise
disponible, et la seule à couvrir le t pour tous les cas (les cas
restreints 2/4 sont servis par les surfaces non restreintes 3/5, la
distribution du t n'étant pas affectée par la restriction des
déterministes).

Ce fichier n'a pas de licence explicite : ardlpy ne le redistribue pas.
Premier usage :

```python
from ardlpy.critical_values.ks2020_finite import download_surface_coefs
download_surface_coefs()   # télécharge depuis kripfganz.de, met en cache localement
```

```python
res = bounds_test(y, x, case=3, order=(6, {...}), finite_t=True,
                   fixed_regressors=dummies)
res.p_values   # p_I0, p_I1 (F) + t_p_I0, t_p_I1 (t)
```

Validé à 1e-3 contre la sortie Stata publiée (Kripfganz & Schneider
2023, *Stata Journal* — exemple salaires UK) : voir PROVENANCE.md. Sans
téléchargement préalable, une erreur explicite indique la marche à
suivre (jamais de téléchargement silencieux).

Interpolation linéaire entre les tailles tabulées adjacentes ; hors
plage [30, 80] → repli asymptotique + warning ; toute combinaison non
couverte (cas I/IV chez Narayan, t, k > 7) → exception explicite
orientant vers la bonne source.

## Le moteur `simulate_bounds` (spec 12 §2.3)

```python
from ardlpy.critical_values import simulate_bounds

sb = simulate_bounds(case=3, k=2, t_obs=45, n_sims=100_000, seed=42, i1=True)
sb.f_cv(0.05), sb.t_cv(0.05)   # CV pour une configuration NON tabulée
sb.seed, sb.n_sims, sb.chunk   # paramètres journalisés (reproductibilité)
```

DGP sous H₀ : y marche aléatoire ; x iid (borne I(0)) ou marches
aléatoires indépendantes (borne I(1)) — convention PSS 2001. Moindres
carrés par QR batchée (jamais d'inversion de X'X). **Tous les paramètres
sont journalisés dans l'objet résultat**, y compris `chunk` (le flux
aléatoire est tiré par lots).

Usages : reproduction des tables publiées (recoupement intégral :
`validation/spec12_montecarlo.py`, version pytest `slow` en nightly),
CV pour configurations non tabulées (k > 10, seuils non publiés, T
arbitraire), base du futur backend Rust.

## Provenance et recoupement

Chaque table encodée cite sa source exacte et son canal de transcription
dans [`PROVENANCE.md`](../../src/ardlpy/critical_values/PROVENANCE.md),
et est recoupée soit par une seconde source (Kripfganz-Schneider,
Dickey-Fuller/MacKinnon), soit par le moteur interne — y compris les
cellules qui n'avaient AUCUNE seconde source avant la spec 12 (colonnes
I(1) du t, lignes k=0 du F) : dette QUESTIONS.md spec 10 §4 soldée.
