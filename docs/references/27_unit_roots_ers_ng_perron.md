# Spec 27 — Elliott, Rothenberg & Stock (1996), Ng & Perron (2001) : pré-tests de racine unitaire

## Références
ERS (1996), *Econometrica*, 64(4), 813–836 (DF-GLS) ; Ng & Perron (2001),
*Econometrica*, 69(6), 1519–1554 (statistiques M et sélection de retards
MAIC). Clés : `elliott1996ers`, `ng2001lag` · Branche : **11** ·
Module : `ardlpy.unitroot` · Priorité : v0.2 — étape 0 de tout workflow
bounds (garantir l'absence d'I(2)).

## 1. Rôle dans la bibliothèque
Le bounds test admet un mélange I(0)/I(1) mais **exclut les I(2)** : la
bibliothèque doit offrir un pré-test crédible et un rapport automatique.
Les tests modernes (DF-GLS, M de Ng-Perron avec sélection MAIC) dominent
l'ADF standard en puissance locale et en taille — ce sont eux qu'on expose
par défaut.

## 2. Implémentation

1. **DF-GLS (ERS)** :
   a. quasi-différenciation des données sous l'alternative locale
      (c̄ = −7 cas constante, −13.5 cas tendance) ;
   b. dé-trending GLS : régresser la série quasi-différenciée sur les
      déterministes quasi-différenciés → série détrendée ỹ ;
   c. ADF sans déterministes sur ỹ ; CV propres (encoder tables +
      surfaces si disponibles).
   Wrapper de `statsmodels DFGLS`? — statsmodels ne l'a pas ; `arch`
   (Kevin Sheppard) l'a : **dépendance optionnelle `arch`** avec fallback
   implémentation maison (la garder de toute façon comme référence
   croisée : c'est notre politique de double implémentation pour les
   briques critiques).
2. **Ng-Perron** : statistiques MZ_α, MZ_t, MSB, MPT calculées sur la
   série GLS-détrendée, avec variance de long terme autorégressive et
   **sélection de retards MAIC** (le critère modifié — implémenter, c'est
   le cœur de leur contribution : l'AIC standard sous-sélectionne en
   présence de MA négative) ; CV tabulés à encoder.
3. **Rapport intégré** : `ardlpy.unitroot.report(df, det=...)` → tableau
   par variable : niveau et première différence, DF-GLS + MZ_t (+ ADF/KPSS
   standard via statsmodels pour complétude, + option fourier_kpss de
   spec 19), verdict I(0)/I(1)/I(2)-suspect. Warning bloquant du bounds
   test si I(2) suspecté (désactivable explicitement).
4. Politique séquentielle documentée : tester le niveau puis la
   différence ; combiner test de racine (H₀ : I(1)) et KPSS (H₀ : I(0))
   pour un verdict croisé.

## 3. Tests
1. Taille/puissance : AR(1) à ρ ∈ {1, 0.95, 0.9}, T=100 : DF-GLS plus
   puissant que l'ADF (reproduire l'écart qualitatif attendu, MC).
2. MAIC vs AIC sur DGP à composante MA négative : MAIC évite les
   distorsions de taille (l'expérience clé de Ng-Perron).
3. Concordance avec le package `arch` (DFGLS) à 1e-8 et avec les valeurs
   des tables publiées pour les CV ; report() détecte une I(2) simulée.

## 4. Liens
Étape 0 du workflow (10) ; fourier_kpss vient de 19 ; le rapport alimente
la vignette « check-list avant bounds test » (avec Johansen sur les x,
spec 07).
