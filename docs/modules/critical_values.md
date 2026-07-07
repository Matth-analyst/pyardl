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

| `cv_source` | Quand l'utiliser | Couverture |
|---|---|---|
| `"pss"` | reproduction d'anciens papiers ; fallback | cas I-V, k ≤ 10, F et t, 10/5/2.5*/1 % |
| `"narayan"` | 30 ≤ T ≤ 80 (données annuelles courtes) | cas II/III/V, k ≤ 7, F seulement, 10/5/1 % |
| `"kripfganz"` | **deviendra le défaut** (spec 13) : ajuste T continûment, p-values | à venir |

\* le seuil 2.5 % provient du moteur de simulation interne (voir
PROVENANCE.md), pas de la transcription des tables publiées.

```python
from ardlpy.critical_values import get_bounds

get_bounds("F", case=3, k=1, alpha=0.05)                                # PSS
get_bounds("F", case=3, k=1, alpha=0.05, cv_source="narayan", t_obs=47) # interpolé
```

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
