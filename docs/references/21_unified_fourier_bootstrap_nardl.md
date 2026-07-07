# Spec 21 — Le cadre unifié moderne : Fourier × Bootstrap × NARDL (aardl 2026, fbardl 2026)

## Références
Roudane (2026), module Stata AARDL (SSC S459609) ; package CRAN fbardl
(2026). Clés : `roudane2026aardl`, `fbardl2026` · Branche : **8** ·
Module : `ardlpy.unified` (couche d'orchestration) · Priorité : v0.6 —
l'aboutissement de la feuille de route.

## 1. Apport
L'état de l'art 2026 combine les trois avancées méthodologiques —
approximation de Fourier (specs 19-20), inférence bootstrap (14/16) et
ARDL non linéaire (17) — dans le cadre à 3 tests de SMG (15). Le module
Stata propose huit types de modèles issus des combinaisons ; c'est la
cible fonctionnelle de notre couche d'orchestration.

## 2. Implémentation : une matrice de combinaisons, pas de nouveau code

Toute la valeur des specs 03-20 se compose ici. Définir l'API produit :

```python
res = ardlpy.cointegration_analysis(
    y, X,
    asym=None | [...],          # NARDL on/off (17)
    fourier=None | dict(k=1, freq="auto"),   # 19-20
    inference="bounds" | "bootstrap",         # 10-13 vs 14/16
    case=3, order="auto", conditional=True,
    B=2999, seed=...)
```

→ 2 (linéaire/NARDL) × 2 (Fourier ou non) × 2 (bornes/bootstrap) =
les 8 configurations du cadre unifié, chacune produisant le **triplet de
tests + classification** (spec 15).

Travail propre à cette spec :
1. **Cohérence des CV par combinaison** : chaque cellule de la matrice a
   sa distribution propre (k effectif changé par la décomposition NARDL,
   f estimée, etc.) — centraliser la logique de choix de CV dans un
   résolveur unique `resolve_critical_values(config)` avec les règles :
   Fourier ⇒ jamais les tables PSS/KS seules ; NARDL ⇒ convention k
   documentée (17 §2.4) ; bootstrap disponible partout et recommandé pour
   toute combinaison non tabulée.
2. **Rapport comparatif** : `res.compare()` — exécuter plusieurs cellules
   (ex. linéaire vs NARDL, avec/sans Fourier) et tabuler triplets et
   classifications côte à côte : c'est le tableau de robustesse que les
   papiers appliqués publient, offert en une ligne.
3. **Garde-fous de sur-spécification** : warning si T petit face au nombre
   de paramètres (décomposition + Fourier + retards) — règle documentée
   (ex. ratio observations/paramètres < 5).

## 3. Tests
1. Chaque cellule de la matrice sur son DGP favorable (généré par le
   simulateur spec 16) → décisions correctes ; matrice complète en test
   nightly.
2. Cohérence descendante : cellule (linéaire, sans Fourier, bornes) ≡
   spec 10 exactement (mêmes objets, mêmes valeurs).
3. Validation externe : module Stata aardl sur données communes — mêmes
   statistiques observées par configuration (1e-4) ; fbardl pour la
   cellule Fourier-bootstrap.

## 4. Liens
Pure orchestration de 10-20 ; c'est la vitrine du package (README,
premier exemple de doc) et le sujet naturel de l'article logiciel
(« un cadre unifié ARDL en Python »).
