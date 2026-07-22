# Spec 17 — Shin, Yu & Greenwood-Nimmo (2014) : le NARDL (asymétries)

## Référence
Shin, Yu & Greenwood-Nimmo (2014), Festschrift Peter Schmidt, Springer,
281–314. DOI: 10.1007/978-1-4899-8008-3_9. Clé : `shin2014nardl`.
Branche : **6** · Module : `pyardl.nardl` · Priorité : **v0.4** — la
fonctionnalité la plus demandée absente de Python.

## 1. Le modèle

Chaque régresseur asymétrique x est décomposé en sommes partielles autour
d'un seuil c (défaut c = 0 sur les variations) :

x_t = x_0 + x⁺_t + x⁻_t,
x⁺_t = Σ_{s≤t} max(Δx_s − c, 0),   x⁻_t = Σ_{s≤t} min(Δx_s − c, 0)

UECM NARDL :
Δy_t = det + λ y_{t-1} + γ⁺ x⁺_{t-1} + γ⁻ x⁻_{t-1}
       + Σψ_i Δy_{t-i} + Σ(ω⁺_i Δx⁺_{t-i} + ω⁻_i Δx⁻_{t-i}) + ε_t

Coefficients de long terme asymétriques : θ⁺ = −γ⁺/λ, θ⁻ = −γ⁻/λ.
Le point clé du cadre : l'estimation et les tests restent de l'OLS/Wald
standard sur ce modèle transformé — toute l'infrastructure des specs
03-16 se réutilise.

## 2. Implémentation

### 2.1 Décomposition
`partial_sums(x, threshold=0.0 | "mean", on="diff")` → (x⁺, x⁻) ;
option seuil sur la moyenne des Δx ; multi-variables : décomposer un
sous-ensemble seulement (dict par variable) — les autres restent
symétriques. Attention à l'initialisation (x⁺_0 = x⁻_0 = 0) et à la
propagation d'index.

### 2.2 Estimation et sélection
`NARDL(y, X, asym=["oil"], order=..., case=3)` : construit les colonnes
décomposées puis délègue à ARDL/UECM (spec 05/03). Sélection d'ordre par
IC sur le modèle transformé (retards possiblement différents sur x⁺ et
x⁻ — supporter les deux modes : appariés ou libres).

### 2.3 Tests d'asymétrie (Wald, sur l'UECM estimé)
1. **Long terme** : H₀ : θ⁺ = θ⁻ — attention, ratio de coefficients →
   Wald via delta method sur (γ⁺, γ⁻, λ) ; alternative équivalente plus
   stable : H₀ : γ⁺ = γ⁻ (documenter que les deux ne coïncident que sous
   λ ≠ 0 ; retenir γ⁺=γ⁻ par défaut comme le fait la pratique).
2. **Court terme** : H₀ : Σω⁺_i = Σω⁻_i (asymétrie additive) et version
   forte ω⁺_i = ω⁻_i ∀i — les deux variantes existent dans la pratique,
   implémenter les deux avec noms explicites.
3. Si aucune asymétrie n'est rejetée → suggérer le repli ARDL symétrique.

### 2.4 Bounds test NARDL
Le triplet (10/11/15) s'applique à l'UECM NARDL avec k = nombre de
niveaux inclus (x⁺ et x⁻ comptent chacun) — question de convention sur k
pour les CV : documenter les deux pratiques (compter les décomposées
séparément — recommandation par défaut — vs la variable d'origine) et
laisser le choix explicite. Versions bootstrap disponibles via spec 14/16.

### 2.5 Multiplicateurs dynamiques asymétriques (la sortie signature)
Effet cumulé de x⁺ (resp. x⁻) sur y à l'horizon h : calculer par récursion
sur la forme ARDL (spec 03, ecm_to_ardl) la réponse m⁺_h à un step
unitaire positif, m⁻_h au négatif ; m⁺_h → θ⁺, m⁻_h → θ⁻.
IC par simulation : tirer R jeux de paramètres ~ N(θ̂_full, V̂), recalculer
les trajectoires, quantiles ponctuels (méthode standard de la pratique
NARDL). `plot_multipliers()` : trajectoires ± IC + courbe de différence
m⁺ − m⁻ avec son IC (le graphique canonique des papiers NARDL).

## 3. API
```python
res = pyardl.NARDL(y, X, asym=["price"], order="auto", case=3).fit()
res.longrun_asym        # θ+, θ− et se
res.asymmetry_tests()   # LR/SR, les 2 variantes
res.bounds_test(cv="kripfganz"|"bootstrap")
res.dynamic_multipliers(h=40, R=1000, seed=...)
res.plot_multipliers()
```

## 4. Tests
1. DGP NARDL simulé (θ⁺=2, θ⁻=0.5) : récupération des deux coefficients ;
   test d'asymétrie LR : puissance ; DGP symétrique : taille ~5 %.
2. Multiplicateurs : convergence m⁺_∞ → θ⁺ (1e-3 à h=200) ; cas
   symétrique → m⁺ = −m⁻ (miroir).
3. Somme des décompositions : x⁺ + x⁻ + x_0 = x (1e-12).
4. Validation externe : R `ardl.nardl` (nardl_uecm) et `nardl` (cas 3/5
   seulement — documenter que nous couvrons les 5 cas, eux non) sur le
   dataset inflation/alimentation de leur doc → coefficients à 1e-6,
   Wald d'asymétrie concordants ; comparaison avec un exemple Stata
   publié pour les multiplicateurs.

## 5. Liens
Réutilise tout le cœur (03-16) ; QARDL (18) combine quantiles et cette
décomposition (QNARDL) ; spec 21 (aardl) combine NARDL+Fourier+bootstrap ;
les multiplicateurs préparent l'infrastructure de simulation de spec 25.
