# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Non publié]

### Modifié (spec 13 — CHANGEMENT DE COMPORTEMENT)
- **`bounds_test` : `cv_source` par défaut passe de `"pss"` à
  `"kripfganz"`** (surfaces de réponse, voie A1 via statsmodels). Les
  bornes F par défaut sont désormais les CV asymptotiques précis (32M
  de réplications) au lieu des valeurs publiées PSS 2001 ; les bornes t
  restent celles de PSS (le matériel voie A1 ne couvre pas le t —
  composition documentée). Pour reproduire la littérature à
  l'identique, passer explicitement `cv_source="pss"` (les réplications
  PSS 2001 des tests le font désormais, et restent vertes).

### Ajouté (spec 13, voie A1 — surfaces de réponse et p-values)
- `critical_values.ks2020` : CV asymptotiques du F à TOUT seuil
  (percentiles simulés + inversion numérique des polynômes) et
  `pvalue_bounds(f_stat, case, k)` -> (p_I0, p_I1) — méthodologie
  Kripfganz-Schneider 2020 via le matériel re-simulé de statsmodels
  (BSD, aucun matériel K&S redistribué — PROVENANCE.md).
- `BoundsTestResults.p_values` + `summary()` enrichi : p-values aux
  deux bornes et lecture continue de la zone non concluante
  (« inconclusive, p ∈ [p_I1, p_I0] »).
- Validation contre les valeurs publiées (WP Exeter 1901, annexe D,
  intercepts theta_{0,0}) ; cohérences internes : bornes PSS validées
  en spec 12, monotonies en k et alpha, aller-retour p-value/CV à 1e-8,
  seuil 2.5 % recoupé contre la table interne spec 12 (±0.05) ; OBS-4
  confirmée par la surface (3.3084 — troisième source concordante,
  registre mis à jour).
- Limitations voie A1 explicites : F seulement, k=1..10, asymptotique
  (fini-T -> voie A2 [licence K&S en clarification] ou voie B v0.4+).

### Ajouté (spec 12 — CV petits échantillons + moteur de simulation)
- `critical_values.simulate.simulate_bounds` : moteur de simulation des
  CV sous H0 (QR batchée, jamais d'inversion de X'X ; seed, n_sims,
  chunk et tous paramètres journalisés dans l'objet résultat).
- Recoupement intégral des tables PSS 2001 par le moteur (100 000
  réplications, 528 cellules, y compris celles sans seconde source) :
  527/528 dans le critère dérivé « 3 erreurs types combinées par
  cellule » (formule dans PROVENANCE.md ; dérivation :
  validation/spec12_mc_error.py) — dette QUESTIONS.md spec 10 §4
  soldée. Note de révision dans la spec 12 (tolérance uniforme ±0.05
  intenable). Registre docs/VALIDATION_OBSERVATIONS.md créé (OBS-1
  coquille dynamac, OBS-2 cas I/k=0/1 % requalifiée ~2σ, OBS-3
  non-monotonie Narayan, OBS-4 cas II/k=2/10 % à 3.7σ confirmé par
  K&S).
- Seuil 2.5 % des tables PSS par simulation interne
  (`pss2001_p025.py`, needs_review + test d'encadrement 5 %/1 %).
- Tables de Narayan 2005 (`narayan2005.py`, GÉNÉRÉ par parseur
  programmatique depuis dynamac — jamais de recopie manuelle) :
  cas II/III/V, T=30..80, k<=7, F ; recoupement moteur (T=40/60),
  cohérences structurelles, PROVENANCE.md.
- `get_bounds(stat, case, k, alpha, cv_source, t_obs)` : dispatcher
  pss/narayan/kripfganz avec hiérarchie documentée (pss = valeurs
  publiées à l'identique, reproduction de la littérature ; kripfganz à
  venir = précis, recommandé, futur défaut) ; interpolation linéaire
  en T pour Narayan, repli asymptotique + warning hors [30, 80],
  exceptions explicites (cas I/IV, t, k>7).
- `bounds_test(cv_source="narayan")` : bornes petits échantillons
  (T = nobs de l'UECM) ; pas de décision t (non tabulé) + warning.

## [0.1.0] — 2026-07-07

Jalon de phase 1 (specs 03, 05, 10, 11 + garde-fous 09) : la
réplication PSS 2001 passe.

### Ajouté
- Validation externe R exécutée (R 4.6.1, package ARDL 0.2.5) : les 4
  scripts de `validation/external/` tournent, valeurs de référence dans
  `tests/replication/expected/*.json` (provenance + tolérances
  contractuelles), 10 tests `external` actifs et verts :
  - spec 03 : theta/lambda à 1e-6 (ARDL(3,1,3,2), denmark) ; cas
    q_j=0 confirmé par R (niveau contemporain, résidus identiques) —
    marque needs_review du test d'équivalence levée ;
  - spec 05 : coefficients ARDL(2,2,2,2) à 1e-6 + SSR ; sélection BIC
    identique à auto_ardl sous politique d'échantillon émulée (la
    divergence de politique est documentée dans QUESTIONS.md) ;
  - spec 10 (JALON) : réplication PSS 2001 salaires UK — F cas IV
    (5.9942) et V (5.8015), t cas V (-2.8633) à 1e-4, coefficients
    UECM à 1e-6 (tendance : reparamétrisation t/4 de R documentée),
    décisions conformes à l'article ;
  - spec 11 : t_BDM cas III et V à 1e-6 vs bounds_t_test.
- `ardlpy.datasets` : load_denmark(), load_pss2001() (provenance
  documentée).
- `bounds_test(..., fixed_regressors=...)` : dummies hors du vecteur
  testé (requis par la réplication PSS).
- Corrections journalisées des scripts R (cas IV/V exigent trend()
  dans le modèle — API du package).
- les règles du projet : les scripts R de validation externe sont exécutés automatiquement quand
  Rscript est disponible ; exécution humaine réservée aux outils
  absents (Stata, EViews).
- Spec 11 (Banerjee-Dolado-Mestre 1998) : `decision_joint` sur
  `BoundsTestResults` (concordance F+t ; « F rejette mais pas t » →
  `degenerate_suspicion` + warning renvoyant à la spec 15) ;
  `adjustment(alpha)` avec IC sur lambda conditionnel à la
  cointégration établie (sinon NaN + warning) ; tests d'unilatéralité
  (DGP explosif jamais « cointegration »), MC taille/puissance du t
  seul (cas III/V, fast_mc + slow), concordance t avec statsmodels
  UECM à 1e-6 ; script R external bounds_t_test (données danoises).
- PROVENANCE.md : démonstration complète de la coquille dynamac
  (position, valeur, triple preuve) en vue d'une issue upstream.
- Spec 10 (PSS 2001) : `ardlpy.bounds.bounds_test` — UECM estimé
  directement pour les 5 cas déterministes (II/IV : déterministe
  restreint dans le vecteur testé, équivalence Wald/régression
  contrainte verrouillée à 1e-10), F_overall + t_BDM (unilatéral
  gauche, garde λ̂ < 0), décision à trois états
  (cointegration/no_cointegration/inconclusive), convention q_j=0
  (niveau contemporain, I(1) sous H0), diagnostics, summary type
  publication. Concordance F avec statsmodels UECM.bounds_test (5 cas).
- `ardlpy.critical_values.pss2001` : bornes asymptotiques PSS 2001
  (tables CI/CII, k=0..10, seuils 10/5/1 %) avec PROVENANCE.md et
  recoupement automatisé par seconde source (F : surfaces
  Kripfganz-Schneider ±0.15 ; t I(0) : Dickey-Fuller/MacKinnon ±0.03) ;
  exceptions explicites pour toute couverture manquante (2.5 %, t cas
  II/IV, k>10) ; dette de recoupement simulation documentée (spec 12).
- Monte Carlo taille/puissance du bounds test (fast_mc 200 reps CI +
  slow 1000 reps × 5 cas, exécuté : taille ≤ 6.5 % à la borne I(1),
  puissance croissante en T).
- Script R external du jalon de phase 1 (réplication PSS 2001 salaires
  UK via R ARDL / Natsiopoulos & Tzeremes 2022).
- Spec 05 (Hendry-Pagan-Sargan 1984) : estimateur `ardlpy.core.ardl.ARDL`
  — OLS via lstsq/QR, cov_type nonrobust/HC0-3/HAC, `is_stable`
  (racines AR), vues spec 03 (`ardl_params`, `to_ecm`, `longrun`,
  `adjustment`), `select_order` (grille + per_variable, échantillon
  commun obligatoire, re-fit final sur échantillon maximal),
  `gets` (réduction contiguë journalisée dans `reduction_path`),
  garde-fou d'autocorrélation automatique post-fit (spec 09 §2.2).
- Briques transversales : `utils.lag_matrix` (spec 02),
  `utils.check_series` (spec 01).
- Tests spec 05 : concordance statsmodels à 1e-10 (coefficients, bse,
  résidus ; llf/IC alignés), verrou échantillon commun de select_order,
  consistance BIC (fast_mc + slow), stabilité, pont spec 03, garde-fou
  spec 09 (positif et négatif), covariances robustes vs statsmodels OLS.
- Squelette du projet (`pyproject.toml`, layout `src/ardlpy`, `tests/`,
  `validation/`, configuration ruff/mypy/pytest).
- Spec 03 (Sargan 1964) : algèbre exacte ARDL <-> ECM dans
  `ardlpy.core.transforms` — `ARDLParams`/`ECMParams`, `ardl_to_ecm`,
  `ecm_to_ardl`, `longrun_coefs`, `longrun_covariance`,
  `speed_of_adjustment`, `half_life`.
- Briques transversales : `ardlpy.utils._delta_method` (méthode delta
  générique par différences finies), `ardlpy.exceptions` (
  `ArdlpyMethodologyWarning`, `DegenerateCaseWarning`).
- Tests : verrou n°1 (équivalence des résidus ARDL/ECM, 8 tirages
  aléatoires dont le cas limite q_j = 0), aller-retour (1000 tirages,
  1e-12), long terme par simulation de réponse à un step (1e-6),
  covariance par méthode delta (analytique vs numérique, 1e-6),
  validations d'entrée, concordance avec
  `statsmodels.tsa.ardl.UECM.from_ardl` (1e-6). Couverture 97 % sur le
  nouveau code.

### Documenté
- `docs/QUESTIONS.md` : traitement du cas limite q_j = 0 dans la
  formule de omega_{j,0} (spec 03 §2.2, « source d'erreurs n°1 »),
  confirmé indépendamment par le comportement de
  `statsmodels.tsa.ardl.UECM` (qui refuse q_j = 0).
