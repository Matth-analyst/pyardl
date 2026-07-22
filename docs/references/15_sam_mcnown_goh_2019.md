# Spec 15 — Sam, McNown & Goh (2019) : le cadre augmenté à 3 tests

## Référence
SMG (2019), *Economic Modelling*, 80, 130–141.
DOI: 10.1016/j.econmod.2018.11.001. Clé : `sam2019augmented`.
Branche : **5** · Module : `pyardl.bounds` (extension) ·
Priorité : **v0.3** — avec spec 14, le duo différenciateur du package.

## 1. Apport
Complète le cadre PSS en dérivant la distribution limite du **troisième
test** : F_indep, H₀ : γ₁ = ... = γ_k = 0 (nullité jointe des niveaux des
variables indépendantes dans l'UECM), avec ses propres bornes de valeurs
critiques. La cointégration authentique est établie si et seulement si les
TROIS tests rejettent simultanément :

| Test | H₀ | Rejet requis |
|---|---|---|
| F_overall (PSS) | λ = γ = 0 | oui |
| t_BDM (spec 11) | λ = 0 | oui (unilatéral gauche, λ̂<0) |
| F_indep (SMG) | γ = 0 | oui |

Sinon, classification automatique :
- F_overall rejette, t rejette, F_indep non → **dégénérescence type 1**
- F_overall rejette, F_indep rejette, t non → **dégénérescence type 2**
- F_overall seul → non concluant renforcé, etc.

## 2. Implémentation

1. **F_indep** : Wald sur les k coefficients γ de l'UECM déjà estimé
   (aucune nouvelle régression) ; brancher sur les 5 cas déterministes
   (les restrictions incluent les déterministes restreints dans les cas
   II/IV, comme en spec 10 §3.1).
2. **Valeurs critiques de F_indep** : encoder les bornes de SMG
   (asymptotiques, par k et cas) dans critical_values ; les compléter par
   le moteur simulate_bounds (spec 12) pour T finis ; version bootstrap
   via le moteur spec 14 (recommandation par défaut).
3. **Table de décision** : implémenter la classification complète
   ci-dessus dans `BoundsTestResults.classification` ∈
   {"cointegration", "degenerate_1", "degenerate_2", "no_cointegration",
   "inconclusive"} avec justification textuelle (quel test a échoué).
   C'est LA sortie qui manque partout en Python et que les referees
   demandent désormais.
4. **Mode par défaut** : à partir de la v0.3, `bounds_test()` exécute les
   3 tests systématiquement ; la présentation `summary()` affiche le
   triplet + classification (l'utilisateur ne doit pas pouvoir conclure
   « cointégration » sur le seul F_overall sans le voir).

## 3. Tests
1. Les 4 DGP canoniques (cointégré, dég. 1, dég. 2, indépendant) →
   classification correcte dans ≥ 90 % des MC (T = 100, CV bootstrap).
2. F_indep : taille aux bornes sous H₀ par cas et k.
3. Validation externe : R bootCT expose le F sur les variables
   indépendantes — concordance des statistiques observées (1e-6) sur
   données communes ; module Stata aardl (spec 21) pour le triplet.
4. Non-régression : l'exemple SMG (réplication de leur application) si
   les données sont publiquement reconstituables ; sinon DGP de l'article.

## 4. Liens
Fusionne specs 10+11 en un cadre unique ; le bootstrap (14/16) fournit
ses CV recommandés ; NARDL (17) et Fourier (19-21) exigent le même triplet
sur leurs UECM transformés — la table de décision est réutilisée telle
quelle.
