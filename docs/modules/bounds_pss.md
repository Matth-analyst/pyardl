# `ardlpy.bounds` — le bounds test PSS 2001

Référence : Pesaran, Shin & Smith (2001), "Bounds Testing Approaches to
the Analysis of Level Relationships", *JAE* 16(3), 289-326 — spec
[10_pesaran_shin_smith_2001](../references/10_pesaran_shin_smith_2001.md).

## L'idée

Tester l'existence d'une relation de **niveau** entre y et les x sans
savoir si les régresseurs sont I(0) ou I(1) : les statistiques F et t de
l'UECM ont des distributions non standard bornées par deux cas polaires
(« tout I(0) » / « tout I(1) »). D'où une décision à **trois états** :

| Situation | Décision |
|---|---|
| stat au-delà de la borne I(1) | `"cointegration"` |
| stat en deçà de la borne I(0) | `"no_cointegration"` |
| entre les deux | `"inconclusive"` (résorbée par les specs 13-16) |

Jamais un booléen.

## Exemple

```python
from ardlpy.bounds import bounds_test

res = bounds_test(y, x, case=3, order=(2, 1))     # ou order=None -> sélection
print(res.summary())        # stat, bornes aux 3 seuils, décisions
res.decision_f              # "cointegration" | "no_cointegration" | "inconclusive"
res.uecm                    # tableau UECM complet
res.diagnostics()           # Ljung-Box, Jarque-Bera, Breusch-Pagan
```

## Les 5 cas déterministes (spec 10 §3)

| Cas | Constante | Tendance | Vecteur testé | ardlpy | R ARDL | Stata ardl | EViews |
|---|---|---|---|---|---|---|---|
| I | — | — | λ, γ | `case=1` | `case=1` | `noconstant` | None |
| II | restreinte | — | λ, γ, c₀ | `case=2` | `case=2` | `restricted` (défaut CV) | Rest. constant |
| III | libre | — | λ, γ | `case=3` | `case=3` | défaut | Unrest. constant |
| IV | libre | restreinte | λ, γ, c₁ | `case=4` | `case=4` | `trendvar restricted` | Rest. trend |
| V | libre | libre | λ, γ | `case=5` | `case=5` | `trendvar` | Unrest. trend |

Cas II et IV : le déterministe restreint fait **partie du vecteur testé**
(k+2 restrictions) — vérifié par le test d'équivalence
Wald = régression contrainte à 1e-10
([test_pss_wald_equivalence.py](../../tests/unit/bounds/test_pss_wald_equivalence.py)).

## Décision jointe F + t (spec 11, Banerjee-Dolado-Mestre 1998)

La cointégration exige la concordance des deux tests — `decision_joint` :

| F | t | `decision_joint` |
|---|---|---|
| cointegration | cointegration | `"cointegration"` |
| cointegration | autre | `"degenerate_suspicion"` + warning → spec 15 |
| no_cointegration | no_cointegration | `"no_cointegration"` |
| autre discordance | | `"inconclusive"` |
| (cas II/IV : t non tabulé) | | `None` |

« F rejette mais pas t » signale une dégénérescence de type 1 (les γ
seuls portent la relation, pas de force de rappel en y) ; la
classification formelle arrive avec le cadre à 3 tests de
Sam-McNown-Goh 2019 (spec 15).

**IC sur la vitesse d'ajustement** : `res.adjustment(alpha)` renvoie λ̂,
se et IC — mais l'IC n'est affiché que si `decision_joint ==
"cointegration"` (sinon NaN + warning) : l'IC standard sur λ n'est pas
valide sous H₀ (piège documenté dans les règles du projet).

## Avertissements méthodologiques

- **t_BDM unilatéral GAUCHE** : le rejet exige λ̂ < 0 ; si λ̂ ≥ 0, un
  `DegenerateCaseWarning` est émis et la décision t est
  `"no_cointegration"` (pas de force de rappel — règle du projet).
- **t non tabulé pour les cas II et IV** (PSS 2001 ne publie pas ces
  bornes) : `decision_t = None` + warning explicite ; utiliser le
  F_overall.
- **Hypothèses de validité** (spec 10 §1) : x faiblement exogènes, pas
  de cointégration entre les x (helper spec 07), aucune variable I(2)
  (pré-tests spec 27), erreurs non autocorrélées (garde-fou Ljung-Box
  automatique, spec 09 §2.2).
- **q_j = 0** : le régresseur entre dans le vecteur testé via son niveau
  contemporain x_{j,t}, qui reste I(1) sous H₀ (identité
  x_t = x_{t-1} + Δx_t, écart stationnaire) — même convention que Stata
  ardl ; statsmodels UECM refuse ce cas.

## Valeurs critiques

Bornes asymptotiques PSS 2001 (tables CI/CII), k = 0..10, seuils
10/5/1 % — provenance et recoupement par seconde source documentés dans
[`src/ardlpy/critical_values/PROVENANCE.md`](../../src/ardlpy/critical_values/PROVENANCE.md).
Limitations actuelles (exceptions explicites) : seuil 2.5 % et
recoupement simulation différés à la spec 12 (dette,
[QUESTIONS.md](../QUESTIONS.md)) ; petits échantillons → Narayan
(spec 12) ; p-values et ajustement en T → surfaces de réponse (spec 13).

## Jalon de phase 1

La réplication de l'équation de salaires UK de PSS 2001 (via le package
R ARDL, réplication Natsiopoulos & Tzeremes 2022) est le test de
non-régression obligatoire :
[spec10_pss2001_replication.R](../../validation/external/spec10_pss2001_replication.R)
(exécution humaine requise, valeurs jamais estimées par nous).
