# Spec 05 — Hendry, Pagan & Sargan (1984) : la spécification dynamique et l'ARDL(p,q) général

## Référence
Hendry, Pagan & Sargan (1984), "Dynamic Specification", *Handbook of
Econometrics*, vol. 2, ch. 18. Clé : `hendry1984dynamic` · Branche : **1**.
Module : `pyardl.core` · Classes : `ARDL`, `ARDLOrderSelection`.
Priorité : **v0.1** — c'est l'estimateur central de la bibliothèque.

## 1. Apport
Synthèse théorique : l'ARDL(p, q₁..q_k) est la forme mère dont dérivent
(comme cas particuliers ou reparamétrisations) : statique, différences,
Koyck, FDL/Almon, ECM, autorégressif pur... La bibliothèque encode cette
taxonomie : un seul moteur d'estimation, plusieurs vues.

## 2. L'estimateur ARDL — algorithme complet

Modèle : y_t = det_t + Σ_{i=1}^{p} φ_i y_{t-i} + Σ_j Σ_{i=0}^{q_j} β_{j,i} x_{j,t-i} + ε_t
(det_t = déterministes selon le cas : const, trend, saisonnalité — spec 03/04).

1. **Construction** : `lag_matrix` (spec 02) pour y et chaque x_j ;
   alignement sur t = max(p, q_j)+1..T ; option `fixed_regressors` (variables
   z_t sans retards, ex. dummies).
2. **OLS** via lstsq ; V̂ classique + options `cov_type` ∈ {nonrobust, HC0-3,
   HAC(nlags)} (réutiliser statsmodels.sandwich).
3. **Sorties** : params, bse, résidus, llf, aic/bic/hqic, R², et les vues
   `.to_ecm()`, `.longrun`, `.adjustment` (spec 03).
4. **Stabilité dynamique** : racines du polynôme 1−Σφ_i L^i via
   `np.roots` ; propriété `.is_stable` (toutes hors cercle unité) ;
   warning sinon.

## 3. Sélection d'ordre — `ARDL.select_order`

1. Grille p ∈ 1..max_p, q_j ∈ 0..max_q (k variables → produit cartésien ;
   pour k>3 et max_q>4, l'espace explose → implémenter aussi la recherche
   **par variable** à la statsmodels/EViews : optimiser q_j séquentiellement).
2. **Échantillon commun obligatoire** (piège de la spec 02 §4).
3. Critères : AIC, BIC, HQ ; sortie = tableau trié + top-N (utile pour
   robustesse à la Pesaran, qui recommande de vérifier plusieurs
   spécifications proches).
4. Post-sélection : re-fit du meilleur modèle sur l'échantillon maximal.
5. Performance : boucle candidate vectorisée ; c'est un point chaud →
   candidat au backend Rust (moindres carrés répétés avec mise à jour
   QR par ajout/retrait de colonnes — optimisation v0.4+).

## 4. Stratégie GETS (general-to-specific) — option
`ARDL.gets(alpha=0.05)` : partir de (max_p, max_q), éliminer itérativement
le retard le moins significatif tant que (a) p-value > alpha, (b) les
diagnostics restent propres (Ljung-Box, hétéroscédasticité), (c) le test F
des restrictions cumulées ne rejette pas. Journaliser le chemin de
réduction dans `.reduction_path` (transparence — important pour un usage
recherche). Équivalent visé : `ardl.nardl` R (gets_ardl_uecm).

## 5. API
```python
ARDL(y, X, order=(p, q) | dict, det="const", seasonal=False,
     fixed_regressors=None).fit(cov_type="nonrobust")
ARDL.select_order(y, X, max_p, max_q, ic="aic", search="grid"|"per_variable")
results.to_ecm(); results.longrun; results.is_stable; results.summary()
```

## 6. Tests
1. Cohérence interne : ARDL(1,0) ≡ KoyckModel.to_ardl() (spec 01) ;
   ARDL(0, q) ≡ FDL (spec 02, r=q).
2. Équivalence ARDL/ECM (test verrou spec 03 §6.1.2).
3. select_order sur DGP connu → retrouve l'ordre vrai avec BIC quand
   T grand (consistance, 500 MC).
4. Racines : DGP explosif → is_stable=False.
5. **Validation externe (critique)** : concordance totale avec
   `statsmodels.tsa.ardl.ARDL` (mêmes données, mêmes ordres → coefficients
   à 1e-10) puis avec R `ARDL::ardl` et `auto_ardl` sur les données
   danoises → mêmes ordres sélectionnés, coefficients à 1e-6.

## 7. Liens
Consomme : lag_matrix (02), transforms (03), déterministes (04).
Alimente : tout le reste — le bounds test (10) est une post-estimation
de cette classe ; NARDL (17) et Fourier (19-21) sont des ARDL sur
régresseurs transformés ; PMG (23) réutilise la brique par individu.
