# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Non publié]

### Ajouté
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
