# Spec 13 — Kripfganz & Schneider (2020) : surfaces de réponse et p-values approchées

## Référence
Kripfganz & Schneider (2020), *Oxford Bulletin of Economics and Statistics*,
82(6), 1456–1481. DOI: 10.1111/obes.12377. Clé : `kripfganz2020response`.
Branche : **4** · Module : `pyardl.critical_values.ks2020` ·
Priorité : **v0.2** — différenciateur majeur (natif dans Stata, absent de
Python, partiel en R).

## 1. Apport
Plutôt que des tables discrètes, des **régressions de surface de réponse**
estimées sur des simulations massives donnent les CV des bornes F et t
comme fonctions lisses de : la taille d'échantillon, le nombre k de
variables de long terme, le cas déterministe et l'ordre des retards. En
prime : des **p-values approchées** (et non plus seulement des seuils),
pour les deux bornes I(0) et I(1).

## 2. Implémentation — deux voies complémentaires

### 2.1 Voie A (rapide, v0.2) : réutiliser les coefficients publiés
Les auteurs distribuent les coefficients de leurs surfaces (matériel
accompagnant l'article/le package Stata). Étapes :
1. Vérifier la licence de redistribution ; si compatible → intégrer les
   coefficients (fichiers versionnés + provenance) ; sinon → téléchargeur
   à la première utilisation avec cache local, ou voie B.
2. Implémenter l'évaluation : CV(τ) = combinaison des régresseurs de
   surface (fonctions de 1/T, k, cas...) selon la forme fonctionnelle du
   papier ; idem pour la transformation stat → p-value approchée
   (les deux sens : quantile→CV et stat→p).
3. API : `get_bounds(..., cv_source="kripfganz", T, p, k, case)` →
   bornes exactes ajustées ; `pvalue_bounds(stat, ...)` → (p_I0, p_I1).
4. Intégration au workflow spec 10 : les sorties gagnent des p-values ;
   `decision` inchangé mais enrichi (« inconclusive, p ∈ [p_I1, p_I0] »).

### 2.2 Voie B (v0.4+, publiable) : ré-estimer nos propres surfaces
Avec le moteur simulate_bounds (spec 12) porté en Rust :
1. Grille massive (T, k, cas, lags) × n_sims élevé → base de quantiles.
2. Ajuster nos régressions de surface (forme fonctionnelle : polynômes en
   1/T et interactions — sélection par validation croisée).
3. Comparer aux valeurs Stata publiées → note méthodologique dans la doc
   (matériel d'article logiciel). C'est le livrable « statisticien » par
   excellence : reproduction indépendante d'un résultat de référence.

## 3. Tests
1. Voie A : reproduire les CV affichés par Stata ardl (valeurs publiées
   dans l'article du Stata Journal 2023 et la doc) sur plusieurs
   configurations (T, k, cas) à 1e-3 ; p-values recoupées sur les mêmes
   exemples.
2. Cohérence interne : CV(T→∞) → bornes asymptotiques PSS (±0.02) ;
   monotonies attendues en T et k.
3. Voie B : nos surfaces vs coefficients publiés — écarts documentés.

## 4. Liens
Devient la source de CV **par défaut** du bounds test (spec 10) et du
t (spec 11) ; le moteur partagé avec spec 12 est le premier module Rust ;
les p-values alimentent la présentation des résultats des specs 14-16.
