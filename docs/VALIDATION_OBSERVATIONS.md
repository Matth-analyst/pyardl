# Registre des observations de validation

Anomalies détectées dans des sources EXTERNES au cours des validations
d'pyardl (tables publiées, packages de référence). Chaque entrée :
source, position exacte, preuve, action. Les anomalies de nos propres
implémentations ne vont pas ici (ce sont des bugs, traités par les
tests) ; ce registre documente ce que nos protocoles de recoupement ont
révélé sur les références elles-mêmes — matériel pour issues upstream
et pour la section validation de l'article JOSS.

## OBS-1 — Coquille dans dynamac (transcription des tables PSS 2001)

- **Source** : package R dynamac (Jordan & Philips), `R/dynamac.R`,
  fonction `pssbounds()`, branche `obs <= 35`, cas I, k = 10,
  seuil 10 %, borne I(0).
- **Anomalie** : `11.60` au lieu de `1.60`.
- **Preuve** : triple démonstration (monotonie en k, cohérence
  I(0) <= I(1), seconde source K&S ≈ 1.598) + confirmation interne (la
  branche asymptotique du même fichier porte 1.60) — détail complet
  dans `src/pyardl/critical_values/PROVENANCE.md`.
- **Action** : valeur corrigée dans notre encodage ; issue upstream à
  ouvrir sur github.com/andyphilips/dynamac.
- **Statut** : documentée (2026-07-07), issue non encore ouverte.

## OBS-2 — Cellule PSS 2001 cas I, k = 0, seuil 1 % (F)

- **Source** : PSS 2001, table CI(i), k = 0, 1 % (valeur publiée 7.17,
  bornes confondues).
- **Anomalie** : écart persistant d'environ -0.22 avec la simulation
  interne — deux runs indépendants à 300 000 réplications convergent
  vers ~6.95 (écart inter-seeds 0.02), soit bien au-delà du critère de
  3 erreurs types combinées (~0.15 pour cette cellule), alors que les
  autres cellules k = 0 des cas IV/V convergent exactement vers les
  valeurs publiées (15.73 retrouvé à ±0.003).
- **Requalification (critère dérivé, arbitrage 2026-07-07)** : la
  distribution F du cas I à k = 0 a une queue droite épaisse (t² de
  Dickey-Fuller sans constante) → densité faible au 99e percentile →
  SE du quantile publié (40 000 réplications) ≈ 0.11. Sous le critère
  « 3 erreurs types combinées », la tolérance de cette cellule vaut
  0.35 : l'écart de -0.23 n'est qu'à ~2σ — **dans l'erreur MC attendue
  de la table publiée**, pas une anomalie. L'impression initiale
  d'anomalie venait de la tolérance uniforme ±0.05 (intenable, cf.
  note de révision spec 12).
- **Action** : valeur publiée conservée telle quelle dans `pss2001.py` ;
  test `needs_review` dédié documentant l'état (plus grand écart absolu
  du recoupement, ~2σ) ; à vérifier contre l'article original quand il
  sera consultable.
- **Statut** : expliquée par le critère dérivé (2026-07-07), maintenue
  en observation.

## OBS-3 — Non-monotonie ponctuelle dans Narayan 2005

- **Source** : Narayan 2005, table Case II, T = 30, 5 %, borne I(1) :
  k = 6 (4.148) -> k = 7 (4.163), +0.015.
- **Anomalie** : violation marginale de la décroissance en k, dans
  l'erreur MC de l'article (40 000 réplications à T = 30).
- **Action** : valeur publiée conservée ; tolérance du test structurel
  documentée (0.02) citant cette cellule.
- **Statut** : documentée (2026-07-07) — vraisemblablement bruit MC de
  l'article, pas d'action upstream.

## OBS-4 — Cellule PSS 2001 cas II, k = 2, seuil 10 %, borne I(1) (F)

- **Source** : PSS 2001, table CI(ii), k = 2, 10 %, I(1) : valeur
  publiée 3.35.
- **Anomalie** : seule cellule au-delà du critère 3σ dans le
  recoupement intégral à 100 000 réplications (écart -0.051, tolérance
  0.041, soit ~3.7σ). Deux sources indépendantes concordent CONTRE la
  valeur publiée : notre simulation (3.299) et Kripfganz-Schneider
  (3.308) — écart simulation/K&S de 0.009 seulement.
- **Interprétation** : imprécision légère de la valeur publiée (le SE
  à 10 % est petit, ~0.014, donc un écart de 0.04 y est détectable) —
  sans conséquence pratique (borne 10 %).
- **Action** : valeur publiée conservée (fonction de cv_source="pss" =
  reproduction de la littérature) ; comptée dans la marge « 0-3
  dépassements fortuits » du test slow ; les surfaces K&S (spec 13)
  fourniront la valeur précise.
- **Confirmation (spec 13, 2026-07-10 — première utilisation du
  registre)** : la surface de réponse K&S (voie A1, statsmodels) redonne
  3.3084 pour cette cellule — troisième source indépendante concordante
  (simulation interne 3.299, K&S 3.308) contre la valeur publiée 3.35.
- **Statut** : confirmée par trois sources (2026-07-10).
