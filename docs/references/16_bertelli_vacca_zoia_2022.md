# Spec 16 — Bertelli, Vacca & Zoia (2022) et bootCT (2024) : bootstrap conditionnel vs inconditionnel

## Références
Bertelli, Vacca & Zoia (2022), *Economic Modelling*, 116 (preprint
arXiv:2204.04939) ; Vacca & Bertelli (2024), *The R Journal* (package
bootCT). Clés : `bertelli2022bootstrap`, `vacca2024bootct`.
Branche : **5** · Module : `pyardl.bootstrap` + `pyardl.simulate` ·
Priorité : v0.3-v0.4.

## 1. Apport par rapport à McNown et al. (spec 14)

1. **Distinction conditionnel / inconditionnel** : l'UECM conditionnel
   inclut les Δx contemporains (le cadre de PSS) ; l'inconditionnel les
   exclut. Sous certaines dégénérescences, les deux conduisent à des
   conclusions différentes — la bibliothèque doit offrir les deux formes
   et documenter le choix (paramètre `conditional=True` par défaut,
   partout où l'UECM apparaît : specs 10, 14, 15).
2. **Bootstrap dérivé rigoureusement du VECM sous-jacent** : le DGP H₀
   est paramétré au niveau du système (bloc marginal des x + équation
   conditionnelle), avec les trois versions bootstrap (F_overall, t,
   F_indep) et prise en compte des cas déterministes via les paramètres
   liant intercept/tendance au mécanisme de correction.
3. **Simulateur VECM/ARDL** : génération de séries multivariées suivant
   un DGP VECM ou ARDL conditionnel spécifié (rang, dégénérescences,
   cas I-V, corrélations d'innovations) — l'équivalent de sim_vecm_ardl.

## 2. Implémentation

### 2.1 `pyardl.simulate.vecm_ardl(...)` (brique transversale)
Paramètres : n_obs, burn_in, matrices du VECM (α, β → Π de rang choisi),
Γ_i, cas déterministe, Σ des innovations, option dégénérescence type 1/2,
seed. Retour : DataFrame + les paramètres « vrais » (pour les MC).
Toutes nos études Monte Carlo (specs 9-21) migrent vers ce simulateur
unique → cohérence et auditabilité.

### 2.2 Extension du moteur bootstrap (spec 14)
- `conditional=True|False` ;
- restrictions H₀ propres à chacun des 3 tests (trois DGP contraints
  distincts — pas un seul) ;
- gestion des cas II/IV dans la régénération (déterministes restreints).

### 2.3 Politique de sortie
`bootstrap_bounds_test` retourne les trois p-values bootstrap + la
classification (spec 15) calculée sur les décisions bootstrap, plus les
décisions « bornes » pour comparaison — reproduire la présentation type
bootCT pour faciliter la validation croisée.

## 3. Tests
1. Simulateur : moments et rang vérifiés (le Johansen de spec 07 retrouve
   le rang injecté ; les dégénérescences injectées sont retrouvées par la
   classification spec 15).
2. Reproduction de l'étude MC de l'article : tailles et puissances des
   tests bootstrap vs bornes dans les configurations principales —
   accord qualitatif (écarts < 2 points) ; publier le notebook.
3. Validation externe bootCT : sur les mêmes données (l'exemple
   consommation de leur documentation), statistiques observées identiques
   (1e-6) et décisions concordantes ; comparaison conditionnel vs
   inconditionnel reproduite.

## 4. Liens
Unifie 14+15 ; fournit le simulateur qui alimente toutes les validations ;
le portage Rust couvre simulate + bootstrap d'un coup ; spec 21 (Fourier
bootstrap) hérite directement de cette infrastructure.
