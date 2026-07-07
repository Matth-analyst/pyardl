# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Non publié]

### Ajouté
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
