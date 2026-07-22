# Spec 12 — Valeurs critiques en échantillon fini : Narayan (2005) et al.

## Références
Narayan (2005), *Applied Economics*, 37(17), 1979–1990 (les tables petits
échantillons les plus citées) ; Narayan & Smyth (2004) ; Mills & Pentecost
(2001) ; Kanioura & Turner (2005). Clés : `narayan2005saving` etc.
Branche : **4** · Module : `pyardl.critical_values` · Priorité : v0.2.

## 1. Problème traité
Les bornes asymptotiques de PSS (spec 10) sont trop libérales quand
T ∈ [30, 80] (le cas typique des données annuelles) : sur-rejet de H₀ →
fausses cointégrations. Narayan (2005) a simulé des bornes exactes par
taille d'échantillon (T = 30..80 par pas de 5), pour les cas II, III et V,
k = 0..7.

## 2. Implémentation

1. **Encodage des tables de Narayan** : fichiers de données versionnés
   (npz) dans critical_values, source et méthode documentées ; interface
   commune `get_bounds(..., cv_source="narayan", T=...)` : sélectionner la
   table du T le plus proche (interpolation linéaire entre deux T
   adjacents, à documenter ; hors plage → fallback asymptotique + warning).
2. **Restriction de couverture honnête** : cas I et IV non couverts par
   Narayan → erreur explicite orientant vers cv_source="kripfganz"
   (spec 13) ; jamais de substitution silencieuse.
3. **Moteur maison de simulation de CV** (la vraie valeur ajoutée de cette
   spec) : `pyardl.critical_values.simulate_bounds(case, k, T, n_sims,
   seed)` — générer sous H₀ (y : marche aléatoire/proc. I(0) selon la
   borne ; x : marches aléatoires indépendantes), estimer l'UECM, stocker
   F et t, retourner les quantiles. Usages : (a) reproduire les tables
   publiées (validation), (b) fournir des CV pour des configurations non
   tabulées, (c) servir de base au backend Rust (boucle idéale à porter —
   premier module PyO3, cf. architecture).
4. Politique par défaut de la bibliothèque (documentée) :
   cv_source="kripfganz" par défaut (spec 13) ; "narayan" recommandé si
   30 ≤ T ≤ 80 et cas couvert ; "pss" pour reproduction d'anciens papiers.

## 3. Tests
1. simulate_bounds (n=100k, seed fixe) reproduit les bornes PSS
   asymptotiques (T=1000) à ±0.05 et plusieurs cellules publiées de
   Narayan (T=40, 60) à ±0.1.

> **Note de révision (2026-07-07, validée)** : la tolérance uniforme
> ±0.05 du point 1 supposait implicitement que les tables publiées sont
> exactes ; elles portent leur propre erreur MC (PSS 2001 : 40 000
> réplications), maximale dans la queue à 1 % (SE du quantile
> ~0.05-0.16 selon la dispersion de la cellule). Hypothèse intenable :
> 23/528 cellules dépassaient ±0.05 alors que le moteur concorde avec
> Kripfganz-Schneider à ±0.06 partout où K&S existe. Critère corrigé
> (arbitrage utilisateur) : tolérance PAR CELLULE = 3 x l'erreur type
> combinée des deux quantiles (formule et estimation de densité :
> PROVENANCE.md ; scripts : validation/spec12_mc_error.py et
> spec12_montecarlo.py). À 3σ sur 528 cellules, 0-3 dépassements
> fortuits restent attendus ; seuls les dépassements persistants entre
> seeds indépendantes sont des anomalies (registre :
> docs/VALIDATION_OBSERVATIONS.md). La même logique s'applique au
> « ±0.1 » du recoupement des cellules de Narayan (40 000 réplications
> également) : critère par cellule identique.
2. Interpolation : monotonie des CV en T ; hors plage → warning.
3. Cas non couvert → exception documentée.

## 4. Liens
Spec 10 (consommateur) ; spec 13 (les surfaces de réponse subsument ces
tables — cette spec reste utile pour la réplication d'articles et le
moteur de simulation) ; backend Rust (candidat n°1).
