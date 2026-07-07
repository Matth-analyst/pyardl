# Spec 20 — Banerjee, Arčabić & Lee (2017) : le test de cointégration Fourier-ADL

## Référence
Banerjee, Arčabić & Lee (2017), *Economic Modelling* (test Fourier-ADL de
cointégration). Clé : `banerjee2017fourieradl` · Branche : **8** ·
Module : `ardlpy.fourier` · Priorité : v0.5.

## 1. Apport
Injecte les termes de Fourier (spec 19) dans un test de cointégration en
équation unique de type ADL/ECM : la relation de long terme est testée en
présence de ruptures lisses de la composante déterministe. Sans cela, des
ruptures ignorées biaisent le bounds test vers le non-rejet (perte de
puissance) ou faussent sa taille. Le test porte sur le coefficient de
correction d'erreur dans le modèle augmenté des sinusoïdes.

## 2. Implémentation

1. **Modèle** : UECM (spec 10) + termes de Fourier dans les déterministes
   (spec 19), fréquence unique estimée par grille sur la SSR.
2. **Statistique** : t sur λ (comme spec 11) dans le modèle Fourier-augmenté ;
   variantes F disponibles par cohérence avec le triplet (spec 15).
3. **Valeurs critiques** : dépendent de k, du cas, de F (nombre de
   fréquences) et du fait que f est estimée → encoder les tables de
   l'article ET fournir la génération par simulation (moteur spec 12,
   avec re-sélection de f dans chaque réplication — leçon de Davies,
   spec 19 §2.2) ; le bootstrap (spec 14/16, également avec re-sélection)
   est l'option recommandée.
4. **Pré-test de pertinence** : le F-test des termes de Fourier (spec 19
   §2.3) doit être exécuté et affiché — si non significatif, recommander
   le bounds test standard (le test Fourier perd de la puissance quand il
   n'y a pas de rupture) ; encoder cette logique de workflow dans la
   sortie.
5. API : `fourier_bounds_test(y, X, case, order, fourier_k=1,
   freq="auto", cv="table"|"sim"|"bootstrap")` → même objet résultat que
   spec 10 enrichi (freq_selected, fourier_ftest, decision).

## 3. Tests
1. DGP cointégré avec rupture lisse dans la constante : le test standard
   (spec 10) perd de la puissance, le Fourier la récupère (MC comparatif —
   figure pour la doc).
2. DGP sans rupture sous H₀ : taille correcte avec CV « f estimée ».
3. Grille de fréquences : robustesse de la décision au pas de grille.
4. Validation externe : package R fbardl / valeurs de l'article sur
   configurations tabulées (CV à ±0.05) ; concordance des statistiques
   avec le module Stata aardl (spec 21) sur données communes.

## 4. Liens
Assemble 19 (briques) + 10/11 (test) ; le bootstrap vient de 14/16 ;
spec 21 généralise au cadre complet à 3 tests avec toutes les combinaisons.
