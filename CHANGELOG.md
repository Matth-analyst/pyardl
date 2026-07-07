# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [Non publié]

### Ajouté
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
