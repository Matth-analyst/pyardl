# Index des spécifications d'implémentation — ardlpy

Chaque spec est autonome : modèle, algorithmes étape par étape, API,
cas limites, plan de tests (unitaires + validation externe), liens.

## Par ordre chronologique (la généalogie)

| # | Spec | Module | Priorité |
|---|------|--------|----------|
| 01 | Koyck (1954) — lag géométrique | distributed_lags | v0.5 |
| 02 | Almon (1965) — PDL | distributed_lags | v0.5 |
| 03 | Sargan (1964) — algèbre ARDL↔ECM | core.transforms | **v0.1** |
| 04 | DHSY (1978) — conventions ECM, saisonnalité | core / datasets | v0.1-0.2 |
| 05 | Hendry-Pagan-Sargan (1984) — ARDL(p,q), sélection, GETS | core | **v0.1** |
| 06 | Engle-Granger (1987) — test 2 étapes, CV MacKinnon | cointegration | v0.2 |
| 07 | Johansen (1988/91) — wrapper + check sur les x | cointegration | v0.3 |
| 08 | FMOLS/DOLS/CCR (1990/93) — long terme efficace | cointegration | v0.3 |
| 09 | Pesaran-Shin (1998) — inférence de long terme | core | v0.1-0.2 |
| 10 | **PSS (2001) — bounds test, 5 cas** | bounds | **v0.1** |
| 11 | Banerjee-Dolado-Mestre (1998) — t-test ECM | bounds | v0.1 |
| 12 | CV petits échantillons (Narayan 2005...) + moteur de simulation | critical_values | v0.2 |
| 13 | **Kripfganz-Schneider (2020) — surfaces de réponse, p-values** | critical_values | v0.2 |
| 14 | **McNown-Sam-Goh (2018) — bootstrap ARDL** | bootstrap | v0.3 |
| 15 | **Sam-McNown-Goh (2019) — cadre à 3 tests, dégénérescences** | bounds | v0.3 |
| 16 | Bertelli et al. (2022)/bootCT — conditionnel, simulateur VECM | bootstrap/simulate | v0.3-0.4 |
| 17 | **Shin-Yu-Greenwood-Nimmo (2014) — NARDL** | nardl | v0.4 |
| 18 | Cho-Kim-Shin (2015) — QARDL, QNARDL | qardl | v0.6 |
| 19 | Becker-Enders-Lee (2006) — briques Fourier | fourier | v0.5 |
| 20 | Banerjee-Arčabić-Lee (2017) — Fourier ADL | fourier | v0.5 |
| 21 | Cadre unifié 2026 (aardl/fbardl) — orchestration 8 modèles | unified | v0.6 |
| 22 | Pesaran-Smith (1995) — panel MG | panel | v0.7 |
| 23 | PSS (1999) — PMG + Hausman + DFE | panel | v0.7 |
| 24 | Chudik-Pesaran (2015/16) — CS-ARDL, CS-DL, test CD | panel | v0.8 |
| 25 | Jordan-Philips (2018) — simulations dynamiques | simulate | v0.5 |
| 26 | Brown-Durbin-Evans (1975) — CUSUM/CUSUMSQ | diagnostics | v0.2 |
| 27 | ERS (1996), Ng-Perron (2001) — pré-tests racine unitaire | unitroot | v0.2 |
| 28 | Surveys — vignettes, check-lists, doctests | docs | continue |

## Feuille de route dérivée (ordre d'implémentation)

- **v0.1** : 03 + 05 + 10 + 11 (+ garde-fous 09) → ARDL/UECM + bounds
  test complet 5 cas, réplication PSS 2001 obligatoire.
- **v0.2** : 12 + 13 + 26 + 27 + 04/06 → CV modernes (p-values), stabilité,
  pré-tests, datasets.
- **v0.3** : 14 + 15 + 07 (+ début 16) → bootstrap + 3 tests +
  classification des dégénérescences. Premier module Rust (simulate/boot).
- **v0.4** : 16 + 17 → simulateur VECM, NARDL complet.
- **v0.5** : 19 + 20 + 25 + 01 + 02 → Fourier, simulations dynamiques,
  module historique.
- **v0.6** : 18 + 21 → QARDL/QNARDL, orchestration unifiée. Article JOSS.
- **v0.7-0.8** : 22 + 23 + 24 → panels (MG, PMG, CS-ARDL/CS-DL).

## Briques transversales (créées une fois, consommées partout)

- `utils.lag_matrix`, `lead_lag_matrix` (02, 08)
- `utils._delta_method` (01) ; `utils.check_series` (01)
- `core.transforms` ARDL↔ECM (03) — testé par équivalence de régressions
- conteneur des cas déterministes I-V + Fourier + saisonnalité (03/04/10/19)
- `utils.longrun_covariance_kernel` (08)
- moteur `critical_values.simulate_bounds` (12) → surfaces (13), Fourier (20)
- moteur bootstrap + simulateur VECM (14/16) → NARDL/QARDL/Fourier/unified
- table de décision 3 tests + classification (15) — réutilisée par 17-21
- test CD et moyennes transversales (24)

## Politique de validation externe (résumé)

statsmodels (05, 10), R ARDL (03, 05, 10, 11), dLagM (01, 02),
bootCT (14-16), ardl.nardl/nardl (17), ardlverse (22-23), urca (07),
cointReg (08), arch (27), dynamac (25), Stata ardl/xtpmg/xtdcce2/aardl
(10, 13, 21, 23, 24), tables publiées (12, 13, 19, 20, 26, 27).
Chaque réplication = un test de non-régression permanent dans tests/.
