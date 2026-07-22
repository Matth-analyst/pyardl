# Spec 10 — Pesaran, Shin & Smith (2001) : le bounds test — CŒUR DE LA BIBLIOTHÈQUE

## Référence
PSS (2001), *Journal of Applied Econometrics*, 16(3), 289–326.
DOI: 10.1002/jae.616. Clé : `pesaran2001bounds` · Branche : **3. Noyau**.
Module : `pyardl.bounds` · Priorité : **v0.1** — la raison d'être du package.

## 1. Le cadre

UECM (forme conditionnelle) :
Δy_t = det_t + λ y_{t-1} + Σ_{j=1}^{k} γ_j x_{j,t-1}
       + Σ ψ_i Δy_{t-i} + Σ_j Σ ω_{j,i} Δx_{j,t-i} + ε_t

Hypothèses à documenter et diagnostiquer : x faiblement exogènes pour les
paramètres de long terme, pas de cointégration entre les x (→ helper
spec 07), aucune variable I(2) (→ spec 27), erreurs non autocorrélées
(→ garde-fou spec 09).

## 2. Les trois tests

1. **F_overall** : H₀ : λ = γ₁ = ... = γ_k = 0 (pas de relation de niveau).
   Wald/F standard sur l'UECM estimé par OLS.
2. **t_BDM** : H₀ : λ = 0 (t sur le coefficient de y_{t-1}) — cf. spec 11.
3. (Le F sur les γ seuls arrive avec SMG 2019, spec 15 — prévoir la place.)

Distributions non standard, dépendant de : k, du cas déterministe, et du
degré d'intégration des régresseurs → PSS fournissent des **bornes** :
CV sous « tout I(0) » (borne inférieure) et « tout I(1) » (borne
supérieure).

**Règle de décision à implémenter** (pour F et t) :
stat > CV_I(1) → rejet de H₀ (relation de niveaux) ;
stat < CV_I(0) → non-rejet ;
entre les deux → **zone non concluante** (statut de sortie explicite
`inconclusive` — jamais un booléen ; c'est cette zone que les specs 13-16
viennent résorber).

## 3. Les 5 cas déterministes (implémentation exhaustive obligatoire)

| Cas | Constante | Tendance | Restriction testée |
|---|---|---|---|
| I | aucune | aucune | λ, γ |
| II | restreinte (dans la relation de LT) | aucune | λ, γ, c₀ |
| III | non restreinte | aucune | λ, γ |
| IV | non restreinte | restreinte | λ, γ, c₁ |
| V | non restreinte | non restreinte | λ, γ |

> **Note d'implémentation (2026-07-07, reportée depuis la spec 03 /
> docs/QUESTIONS.md)** : cas limite q_j = 0 — un régresseur sans retard
> propre entre dans le vecteur testé {λ, γ_1, ..., γ_k} via son niveau
> CONTEMPORAIN x_{j,t} (pas de terme Δx_{j,t} distinct). x_{j,t} reste
> I(1) sous H₀ : par l'identité exacte x_{j,t} = x_{j,t-1} + Δx_{j,t}
> (écart stationnaire), l'asymptotique du test est inchangée — seule la
> datation du régresseur de niveau diffère d'une période. Convention
> identique à Stata ardl ; statsmodels UECM refuse q_j = 0.

Détails d'implémentation :
1. Cas II et IV : la constante (resp. tendance) entre dans le vecteur testé
   → le F porte sur k+2 restrictions ; la régression s'écrit avec le terme
   déterministe *à l'intérieur* du terme de correction. Implémenter par la
   régression équivalente non contrainte + test de Wald incluant le
   déterministe (équivalence à vérifier par test).
2. Le conteneur de cas (`case ∈ {1..5}`) créé en spec 03 est finalisé ici ;
   toutes les specs ultérieures (13-16, 19-21) le consomment.
3. Mapping doc : correspondances avec les conventions R ARDL / Stata ardl /
   EViews (tableau de la vignette — source récurrente de confusion).

## 4. Valeurs critiques asymptotiques

1. Encoder les tables de bornes asymptotiques (F et t, cas I-V,
   k = 0..10, seuils 1/2.5/5/10 %) dans `pyardl.critical_values.pss2001`
   (données numériques — fichiers .npz versionnés + provenance documentée).
2. Interface : `get_bounds(stat="F"|"t", case, k, alpha)` → (lower, upper).
3. Ces tables sont le fallback ; les surfaces de réponse (spec 13)
   deviendront la source par défaut car elles ajustent T.

## 5. Workflow utilisateur complet (la fonction phare)

```python
res = pyardl.bounds_test(y, X, case=3, p=..., q=... | ic="aic",
                          cv_source="kripfganz"|"pss"|"narayan")
# BoundsTestResults:
#   f_stat, t_stat, bounds (df), decision_f, decision_t ∈
#   {"cointegration", "no_cointegration", "inconclusive"},
#   pvalues (si cv_source le permet, spec 13), uecm (résultats complets),
#   diagnostics (LB, BP, JB, CUSUM), warnings structurés
```

Étapes internes : validation entrées → (option) sélection d'ordre (05) →
estimation UECM (03/05) → F et t → confrontation aux bornes → diagnostics
automatiques → objet résultat avec `summary()` reproduisant la présentation
standard des papiers appliqués (stat, bornes aux 3 seuils, décision).

## 6. Réplication de référence (OBLIGATOIRE avant release v0.1)
Reproduire l'application de PSS 2001 (équation de salaires réels UK) telle
que répliquée par Natsiopoulos & Tzeremes (2022, JAE) avec le package R
ARDL : mêmes données publiques, mêmes ordres → F et t identiques à 1e-4,
mêmes décisions aux 3 cas testés. Ce cas devient le test de non-régression
permanent `tests/test_replication_pss2001.py`.

## 7. Tests
1. Chaque cas I-V : DGP sous H₀ → taille empirique aux bornes (I(0) et
   I(1)) cohérente ; DGP cointégré → puissance croissante en T (grille MC).
2. Wald cas II/IV = régression contrainte équivalente (1e-10).
3. Concordance statsmodels `UECM.bounds_test` (mêmes stats), R `ARDL::
   bounds_f_test/bounds_t_test`, et Stata ardl (valeurs publiées) sur
   données danoises + PSS.
4. Statut inconclusive correctement retourné sur cas construits.

## 8. Liens
Consomme 03, 05, 07, 09, 27. Étendu par : 11 (t), 12-13 (CV), 14-16
(bootstrap/augmenté), 17 (NARDL), 19-21 (Fourier). C'est le hub.
