# Spec 06 — Engle & Granger (1987) : cointégration et représentation ECM

## Référence
Engle & Granger (1987), *Econometrica*, 55(2), 251–276. DOI: 10.2307/1913236.
Clé : `engle1987cointegration` · Branche : **2. Cointégration classique**.
Module : `ardlpy.cointegration` · Priorité : v0.2 (contexte + test de
référence pour comparaisons).

## 1. Apport et rôle dans la bibliothèque
Formalise : (a) la cointégration — des séries I(1) dont une combinaison
linéaire est I(0) ; (b) le théorème de représentation de Granger — des
séries cointégrées admettent une représentation ECM et réciproquement.
Justifie théoriquement que le terme de rappel de Sargan/DHSY est bien
défini. La bibliothèque implémente le test en deux étapes comme méthode
de comparaison (le cœur restant l'approche bounds).

## 2. Le test Engle-Granger en deux étapes — algorithme

1. **Étape 1 (relation de long terme)** : OLS statique y_t sur [det, x_t]
   → résidus û_t. Coefficients super-convergents mais biais de petit
   échantillon et distribution non standard (documenter : ne PAS faire
   d'inférence sur cette étape — renvoyer vers FMOLS/DOLS spec 08 ou
   l'ARDL).
2. **Étape 2 (test de racine unitaire sur û)** : régression ADF sans
   déterministes sur û_t : Δû_t = ρ û_{t-1} + Σ ξ_i Δû_{t-i} + e_t ;
   statistique t sur ρ.
3. **Valeurs critiques spéciales** (dépendent de k et du cas déterministe
   de l'étape 1) : utiliser les surfaces de réponse de MacKinnon (2010) —
   implémenter la table des coefficients de surface (τ_∞ + b₁/T + b₂/T²)
   dans `ardlpy.critical_values.mackinnon` ; p-values approchées incluses.
4. **ECM en deuxième étape** (option `fit_ecm=True`) : régresser Δy sur
   Δx et û_{t-1} → vitesse d'ajustement à la Engle-Granger.

## 3. API
```python
engle_granger(y, X, det="const", max_ad_lags=None, ic="aic",
              fit_ecm=False) -> EGResults
# EGResults: stat, pvalue, crit (1/5/10%), residuals, longrun_params, ecm
```

## 4. Limites à documenter (vignette)
Normalisation arbitraire (choix de la variable de gauche), une seule
relation de cointégration détectable, pas de test possible si mélange
I(0)/I(1) — exactement les points qui motivent Johansen (07) et surtout
l'approche bounds (10).

## 5. Tests
1. DGP cointégré bivarié (β connu) → rejet fréquent ; DGP marches
   aléatoires indépendantes → taille ~5 % avec les CV MacKinnon (1000 MC).
2. Surfaces MacKinnon : reproduire quelques valeurs tabulées publiées
   (comparaison à statsmodels `coint` qui utilise les mêmes surfaces —
   concordance à 1e-8 sur stat et p-value).
3. Cohérence : û de l'étape 1 identiques à ceux d'une OLS statsmodels.

## 6. Liens
Spec 07 (Johansen, approche système) ; spec 08 (estimateurs efficaces du
long terme) ; spec 10 (bounds : lever la contrainte « tout I(1) ») ;
spec 16 (bootCT inclut un test de Johansen sur les x — même besoin de
briques de cointégration classiques).
