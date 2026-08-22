# Registre des observations de validation

Ce que nos protocoles de recoupement ont révélé sur les MÉTHODES et les
SOURCES que nous consommons. Deux catégories, même format (source,
position exacte, preuve, action, statut) :

1. **Anomalies de sources externes** — tables publiées, packages de
   référence (OBS-1 à OBS-4).
2. **Limites méthodologiques des tests eux-mêmes** — propriétés vraies
   de la méthode, indépendantes de toute implémentation, mises en
   évidence par nos DGP de validation (OBS-5). La « source » est alors
   la structure mathématique du test, pas un document.

Les anomalies de NOS implémentations ne vont pas ici : ce sont des bugs,
traités par les tests. Matériel pour issues upstream et pour la section
validation de l'article JOSS.

> **Note de lecture.** Ce registre est publié ; les scripts qu'il cite
> (`validation/…`) et les fichiers de travail (`docs/DEVIATIONS.md`,
> `docs/QUESTIONS.md`, les spécifications) restent internes. Chaque
> observation porte donc ses chiffres et le paramétrage qui les a
> produits, de sorte qu'elle se lise sans eux. Les résultats numériques
> livrés, eux, sont vérifiables directement : ils sont dans les tests et
> dans `src/pyardl/critical_values/PROVENANCE.md`.

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
- **Requalification (critère dérivé, 2026-07-07)** : la
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

## OBS-5 — Cécité structurelle du CUSUM aux ruptures de pente centrées

- **Source** : propriété méthodologique du test lui-même
  (Brown-Durbin-Evans 1975), et non une source externe. Vraie de toute
  implémentation correcte du CUSUM, y compris statsmodels et EViews.
- **Anomalie** : une rupture portant sur la PENTE d'un régresseur de
  moyenne nulle est structurellement invisible au CUSUM, alors qu'elle
  est le cas d'instabilité le plus courant après la rupture de niveau.
- **Preuve** : DGP à rupture de pente pure à mi-échantillon (saut de +4
  sur x2 centré, T = 200, k = 3), 20 graines indépendantes. Le CUSUM
  conclut « stable » **20 fois sur 20** ; le CUSUMSQ détecte la rupture
  **20 fois sur 20**. Le mécanisme est analytique et non statistique :
  après la rupture les résidus récursifs valent `jump · x_{t,2} + bruit`
  et, `x_2` étant centré, leur espérance reste nulle — la somme cumulée
  ne peut donc pas dériver. Seule la variance change, ce que mesure le
  CUSUMSQ. La cécité n'est pas un défaut de puissance : elle est exacte.
- **Action** : mise en garde ACTIVE, à trois niveaux. (a) Section
  ``Warnings`` de la docstring de ``cusum()``, donc visible dans
  ``help(cusum)`` au moment où l'utilisateur appelle la fonction ;
  (b) docstring du module ``pyardl.diagnostics`` et page utilisateur
  ``docs/api/diagnostics.md``, section dédiée avec le chiffre 20/20 ;
  (c) garde-fou structurel : ``bounds_test(...).diagnostics()`` produit
  TOUJOURS les deux tests, jamais un seul, de sorte qu'un utilisateur
  ne peut pas obtenir par inadvertance le seul CUSUM.
- **Portée** : la pratique répandue, dans la littérature appliquée
  ARDL, de ne publier que le graphique CUSUM laisse donc entièrement
  non testée une famille courante d'instabilités.
- **Trace** : `tests/unit/diagnostics/test_stability.py`,
  `TestSpecialisation::test_slope_break_invisible_to_cusum_seen_by_cusumsq`.
- **Statut** : établie et verrouillée par test (2026-08-03).

## OBS-6 — Surfaces DF-GLS de `arch` conservatrices à T = 50

- **Source** : package `arch` (Sheppard, BSD-3),
  `arch/unitroot/critical_values/dfgls.py`, coefficients
  `dfgls_cv_approx`, évalués à 1/T pour T = 50.
- **Anomalie** : écart systématique avec notre simulation, de même
  signe sur les 6 cellules concernées (2 cas déterministes × 3 seuils),
  jusqu'à 0.037. Les valeurs de `arch` sont toujours plus négatives,
  donc plus conservatrices. Au-delà de T = 100, les deux sources
  concordent dans le bruit de simulation.
- **Preuve** : expérience de taille sur 200 000 réplications à graine
  INDÉPENDANTE de celle qui a servi à tabuler. Nos valeurs délivrent la
  taille nominale (0.0507 à 5 %, 0.1019 à 10 %, 0.0107 à 1 %) ; celles
  de `arch` sous-rejettent (0.0472, 0.0961, 0.0098). Le décalage
  d'indice a été écarté comme explication : évaluer à 1/(T−1) au lieu
  de 1/T aggrave l'écart (0.041 contre 0.035).
- **Interprétation** : artefact d'ajustement de la surface de réponse au
  bord de sa plage. Ce n'est PAS un désaccord sur la statistique
  elle-même — celle-ci coïncide à 1e-15 entre les deux implémentations,
  verrouillé par test.
- **Action** : nos valeurs simulées sont servies ; le test de
  recoupement contre `arch` ne porte que sur T ≥ 100, avec la raison
  documentée dans sa docstring.
- **Statut** : établie par expérience de taille indépendante
  (2026-08-03).

## OBS-7 — Sur-sélection du MAIC sur données stationnaires

- **Source** : propriété du critère MAIC (Ng-Perron 2001), et non une
  anomalie d'implémentation. Le terme de pénalité τ_T(k) est d'autant
  plus grand que la série paraît stationnaire.
- **Anomalie** : sur du bruit blanc, MAIC retient 6.1 retards en
  moyenne contre 0.0 pour BIC. La puissance du test s'effondre.
- **Preuve** : rapport séquentiel, 40 réplications, T = 250 —
  MAIC classe correctement 29/40 des I(0) et 32/40 des I(1) ; BIC
  40/40 et 40/40. L'écart se paie en fausses suspicions d'I(2).
- **Interprétation** : ce n'est pas un défaut mais un arbitrage. MAIC
  protège contre la MA négative, situation qu'aucun de ces DGP propres
  n'exerce. Le prix de cette protection est une perte de puissance
  ailleurs.
- **Action** : arbitrage rendu le 2026-08-03. `report()` et
  `integration_order()` — le dépistage initial — passent à **BIC** par
  défaut ; `dfgls()` et `ng_perron()` — l'analyse ciblée — gardent
  **MAIC**. Le tableau chiffré figure dans la docstring de `report()`,
  donc dans `help(report)`. Un test fige les quatre défauts, un autre
  fige l'écart de classification mesuré.
- **Statut** : mesurée, documentée, arbitrée (2026-08-03).

## OBS-8 — Un seul DGP nul pour le F et le t : tranché par la taille

- **Source** : ambiguïté de la spec 14 §2.2 (« imposer la restriction du
  test considéré »), révélée par un écart avec le package R bootCT, et
  non par la lecture. Deux lectures possibles :
  - **variante A** (implémentée) : un seul DGP nul, celui de la nulle
    JOINTE λ = γ = 0, sert aux distributions des deux statistiques ;
  - **variante B** : un DGP par test — pour le t de BDM, dont la nulle
    est λ = 0 seul, on garde les niveaux des x et on ne supprime que le
    niveau retardé de y.
- **Ce qui a déclenché la question** : la validation externe contre
  bootCT donne des bornes du t moins extrêmes que les nôtres (-3.00
  contre -3.76 à 5 %), soit 21 à 30 % d'écart, là où les bornes du F
  concordent à 0.6-13 %. Les statistiques OBSERVÉES, elles, coïncident à
  4e-10 : les deux implémentations estiment le même modèle, la
  divergence est donc dans la construction de la nulle.
- **Preuve** : taille empirique du t à 5 %, sous la nulle jointe VRAIE
  (y et x marches aléatoires indépendantes), 400 échantillons Monte
  Carlo, B = 299 par échantillon, T = 100.

  | cas | variante A | variante B | attendu |
  |-----|-----------|-----------|---------|
  | III | 0.0350    | 0.0925    | 0.05    |
  | V   | 0.0500    | 0.0825    | 0.05    |

  La variante A tient la taille nominale ; la variante B **sur-rejette
  d'un facteur proche de deux**.
- **Mécanisme, mesuré et non supposé** : l'hypothèse initiale était que
  la variante B régénérerait des séries I(2), Δy dépendant du niveau
  d'un régresseur intégré. **Cette hypothèse a été RÉFUTÉE** : γ estimé
  sous la contrainte λ = 0 vaut environ -0.01 même sur données
  fortement cointégrées, et les ordres d'intégration régénérés sont
  identiques sous les deux variantes (57 I(1) sur 60 de part et
  d'autre). Le mécanisme réel est différent : la distribution bootstrap
  du t sous B est plus dispersée (écart-type 1.030 contre 0.960) mais
  MOINS asymétrique à gauche, d'où un quantile à 5 % moins extrême
  (-2.891 contre -3.114). Bornes moins exigeantes, donc rejets plus
  fréquents.
- **Action** : convention A MAINTENUE. `docs/QUESTIONS.md` passe en
  CLOS. L'écart avec bootCT est expliqué et documenté : nos bornes du t
  sont plus exigeantes parce qu'elles sont construites sous la bonne
  nulle, pas par excès de prudence.
- **Conséquence pour la spec 15** : le cadre à trois tests ajoute une
  statistique sur les γ seuls. Le résultat ci-dessus commande de la
  bootstrapper sous **le même DGP nul joint**, comme les deux autres —
  un seul modèle nul, trois statistiques lues dessus. Toute variante
  « un DGP par test » devra être mesurée avant d'être adoptée, et non
  déduite de la formulation de la spec.
- **Trace** : `validation/spec14_null_dgp_arbitration.py`, résultats
  dans `validation/results/spec14_null_dgp_arbitration.txt`.
- **Statut** : tranchée par la mesure (2026-08-18).

## OBS-9 — Les bornes de F_indep sont simulées, et le recoupement est structurel

**Spec 15.** Sam, McNown & Goh (2019) publient les bornes de F_indep
dans *Economic Modelling*, sous barrière d'accès. Le projet n'encode pas
une valeur critique qu'il n'a pas calculée : la table livrée est
**simulée** (`validation/spec15_findep_cv.py`), cas 1 à 5, k = 1..10,
100 000 réplications à T = 1000, une seed déterministe par
configuration.

Il n'existe aucune seconde implémentation à laquelle comparer ces
bornes. Le recoupement est donc **structurel**, et c'est une garantie
plus faible qu'une concordance externe — il faut le dire plutôt que
laisser croire à une validation croisée. Trois propriétés que la théorie
impose sont vérifiées sur les 100 configurations, sans exception :

1. borne I(0) <= borne I(1) ;
2. un seuil plus strict donne une borne plus élevée ;
3. la borne décroît en k, F_indep étant un F PAR restriction.

Violations : 0 (`validation/results/spec15_findep_crosscheck.txt`).

**Ce qui est validé en externe, en revanche, c'est la statistique.**
`bootCT::boot_ardl` expose `find.stat` : sur les données danoises, cas
III, ordre (3 ; 1, 3, 2), les deux implémentations donnent
8.16193503296811 et 8.16193503296774, soit un écart de **4e-13**. La
statistique testée est donc bien la même des deux côtés ; seules les
bornes restent d'origine interne.

**Conséquence pratique** : hors de la grille simulée, `findep_bounds`
lève une exception et `decision_indep` vaut `None`. La classification
répond alors `inconclusive` en nommant le test manquant, au lieu de
conclure sur deux tests comme si le troisième n'existait pas.

## OBS-10 — La trace sur-sélectionne le rang ; maxeig tient le critère

**Spec 07.** Le plan de tests demande que la procédure séquentielle
retienne le rang exact dans au moins 90 % des réplications sur un DGP
VECM de rang 1 (3 variables, T = 200, 1000 réplications). Mesure faite
(`validation/results/spec07_rank_selection.json`) :

| statistique | rang 1 retenu | rang 2 | rang 3 | rang 0 (raté) |
|---|---|---|---|---|
| trace  | 87.8 % | 6.6 % | 5.6 % | 0 % |
| maxeig | 92.5 % | 5.7 % | 1.8 % | 0 % |

Et sous un DGP de rang 0 (trois marches aléatoires indépendantes), taux
de fausse détection d'au moins une relation :

| statistique | rang 0 correct | au moins une relation à tort |
|---|---|---|
| trace  | 91.7 % | 8.3 % |
| maxeig | 94.2 % | 5.8 % |

**Lecture.** Le critère de 90 % est atteint par la valeur propre
maximale et **manqué par la trace**. L'erreur est systématiquement du
même côté : la trace ajoute des directions, elle n'en retire jamais — le
rang vrai n'est manqué dans AUCUNE des 1000 réplications, ni par l'une
ni par l'autre. Sous le rang 0, la sur-détection de la trace (8.3 %)
dépasse aussi le seuil nominal de 5 %, ce qui est le même phénomène.

**Ce qui n'a PAS été fait** : changer le défaut pour maxeig afin que le
test passe. `method="trace"` reste le défaut, parce que c'est ce que
rapporte la littérature appliquée et ce qu'attend un lecteur de sortie
Johansen. Le test Monte Carlo du plan est exécuté sur maxeig, avec la
raison écrite dans sa docstring, et le comportement de la trace est
documenté ici plutôt que masqué par un défaut choisi pour la commodité
du test.

**Conséquence pratique** : sur données réelles, un rang retenu par la
trace mérite d'être recoupé par maxeig. Les deux sont toujours calculés
et `res.rank(method=..., alpha=...)` permet de relire la décision sans
ré-estimer.

- **Trace** : `validation/results/spec07_rank_selection.json` (1000
  réplications par cellule, seeds déterministes).
- **Statut** : mesurée (2026-08-20), défaut inchangé.

## OBS-11 — F_indep bootstrappé est sur-dimensionné en petit échantillon

**Spec 16.** En re-mesurant la question du DGP nul pour F_indep (voir
`docs/DEVIATIONS.md`, spec 16 §2.2), une seconde chose est apparue, qui
n'était pas la question posée.

**Trajectoire, telle qu'elle s'est déroulée.** La première passe, à 400
réplications Monte Carlo, donnait une taille de 0.0475 (cas III) et
0.0525 (cas V) pour la variante retenue : conforme au nominal. L'écart
avec la variante « un DGP par test » (0.0675 et 0.0750) allait dans le
sens d'OBS-8 mais ne dépassait pas 1.5 erreur type — donc pas
concluant. Plutôt que de conclure quand même, la mesure a été refaite à
1200 réplications, avec appariement des échantillons.

Le verdict sur la question posée est devenu décisif (McNemar p = 1e-05
et 9.5e-07). **Et la taille de la variante retenue est passée de 0.0475
à 0.0667.** Vérification faite avant toute conclusion : sur les 400
premiers échantillons, la nouvelle exécution redonne exactement 0.0475,
et ce sont les 800 suivants qui rejettent à 0.0762. Il n'y a pas de
bug ; le premier chiffre était un tirage favorable.

| cas | taille de F_indep, nul joint, T = 100 |
|-----|------|
| III | 0.0667 |
| V   | 0.0642 |

**Lecture.** À T = 100, le bootstrap de F_indep rejette environ 6.5 %
du temps au seuil nominal de 5 %. Ce n'est pas le comportement du t
sous le même nul, qui tient sa taille (OBS-8 : 3.5 % et 5.0 %). La
statistique F_indep est donc la plus fragile des trois en petit
échantillon.

**Ce que cela ne dit pas** : que la variante « un DGP par test »
vaudrait mieux. Elle est mesurée à 8.2-8.5 %, soit strictement pire.
Le choix reste le meilleur des deux mesurés, pas un choix exact.

**Conséquence pratique** : sur un échantillon court, un rejet de
F_indep tout juste au seuil de 5 % doit être lu comme un rejet à
environ 6.5 % de taille réelle. La classification qui en dépend —
notamment la frontière entre `cointegration` et `degenerate_1` — hérite
de cette imprécision. Augmenter T est le seul remède mesuré à ce jour.

**LEÇON MÉTHODOLOGIQUE, indépendante du résultat.** Le premier chiffre
n'était pas faux : il était vrai sur 400 réplications, reproductible au
bit près, et il disait ce qu'on espérait — que la variante retenue tient
sa taille. C'est exactement le profil d'un résultat qu'on ne re-mesure
pas. Il n'a été réfuté que parce que la mesure a été refaite pour une
AUTRE raison : l'écart entre les deux variantes n'était pas concluant,
et il fallait plus de réplications pour trancher la question posée.

La règle qui en découle, pour toute étude de validation du projet :

> La taille d'un Monte Carlo de validation se fixe sur la précision
> exigée du verdict, jamais sur le moment où le résultat devient
> favorable. Un taux de rejet estimé sur 400 réplications porte une
> erreur type de 1.1 point de pourcentage : il ne peut pas distinguer
> 5 % de 7 %, et donc ne peut pas établir qu'une taille est correcte.

Concrètement : à 400 réplications, l'intervalle de confiance exact à
95 % autour d'un taux observé de 4.75 % (19 rejets sur 400) va de
2.88 % à 7.32 % — il CONTENAIT déjà la valeur mesurée ensuite sur 1200
réplications. Le chiffre n'a jamais contredit le résultat final ; il
a seulement été lu comme s'il avait une précision qu'il n'avait pas.
Aucune des observations antérieures du registre n'est invalidée par ce
constat (OBS-8 tranchait un écart de 9.3 % contre 3.5 %, hors de portée
du bruit à 400 réplications), mais toutes les suivantes dimensionneront
leur échantillon sur l'écart à détecter.

- **Trace** : `validation/spec16_null_variants.py`, résultats dans
  `validation/results/spec16_null_variants.txt`. Le contrôle de
  non-régression du premier chiffre (400 premiers échantillons ->
  0.0475 a l'identique) est reproductible par le meme script.
- **Statut** : mesurée (2026-08-20), documentée sans correctif.

## OBS-12 — Ce que le bootstrap achète, mesuré : la zone non concluante

**Spec 16 §3.2.** L'étude Monte Carlo de Bertelli, Vacca & Zoia (2022)
est reproduite dans ses configurations principales
(`validation/spec16_montecarlo.py`, 1000 réplications, T = 100, B = 299,
cas III, innovations corrélées à 0.5), sur les quatre DGP canoniques du
simulateur unique.

**Limite de couverture, dite d'emblée** : les tableaux de l'article sont
sous barrière d'accès. Le critère de la spec — « accord qualitatif,
écarts < 2 points de pourcentage » contre les valeurs publiées — n'est
donc PAS vérifiable ici, et prétendre le contraire serait faux. Ce qui
est vérifié est ce qui peut l'être sans le texte : les affirmations
qualitatives de l'article, mesurées sur nos propres DGP.

Taux de classification correcte, et taux de verdict non concluant :

| DGP | bootstrap correct | bornes correct | bornes non concluant |
|-----|------|------|------|
| cointegration | 100.0 % | 100.0 % | 0.0 % |
| degenerate_1  | 99.4 % | 93.2 % | 5.5 % |
| degenerate_2  | 96.3 % | 99.8 % | 0.1 % |
| no_coint      | 91.5 % | 71.3 % | 24.8 % |

**Lecture.** L'apport du bootstrap est presque entièrement l'élimination
de la zone non concluante, et il se voit là où cette zone est large :
sous le DGP sans cointégration, les bornes laissent
24.8 % des échantillons sans
verdict, et le bootstrap en tranche la quasi-totalité correctement
(91.5 % contre
71.3 %). Sous cointégration
franche, où aucune des deux routes n'hésite, les deux donnent 100 % et
le bootstrap n'apporte rien.

**Ce qu'il coûte.** Sous le DGP dégénéré de type 2, le bootstrap conclut
à tort à la cointégration dans 3.7 %
des cas contre 0.1 % pour
les bornes : trancher, c'est aussi se tromper franchement là où les
bornes se taisaient. Le sens de cette erreur est le pire possible — une
dégénérescence prise pour une relation — et il rejoint la fragilité de
F_indep en petit échantillon relevée en OBS-11.

**Taux d'accord entre les deux routes** : 100.0 %,
93.8 %,
96.4 % et
76.8 % selon le DGP. Le
désaccord est concentré exactement là où les bornes sont non
concluantes, ce qui est la forme attendue : les deux routes ne
divergent pas sur les cas tranchés.

- **Trace** : `validation/spec16_montecarlo.py`, résultats dans
  `validation/results/spec16_montecarlo.txt` et `.json`.
- **Statut** : mesurée (2026-08-21). Critère chiffré de la spec non
  vérifiable (source sous barrière) — affirmations qualitatives
  confirmées.


## OBS-13 — Les tables de PSS ne décrivent pas le nul du NARDL

**Spec 17.** La spec laissait ouverte la convention de comptage de `k`
pour les valeurs critiques et demandait de documenter les deux pratiques
de la littérature. Appliquant la règle établie par OBS-8 — une variante
se mesure avant d'être adoptée — les deux ont été mesurées. Le résultat
n'était pas celui attendu : **aucune des deux ne tient sa taille.**

- **Source** : spec 17 §2.4, et la pratique publiée qui lit les
  statistiques NARDL contre les tables de PSS.
- **Preuve** : 1000 réplications, T = 150, cas III, H0 vraie (y et x
  marches aléatoires indépendantes), seuil nominal 5 %.

  | lecture | taux de rejet |
  |---|---|
  | `decomposed` (k = 2 par variable) | 7.3 % |
  | `original` (k = 1 par variable) | 2.6 % |
  | **témoin** : ARDL linéaire, 2 vrais régresseurs | 4.8 % |

  Le témoin est ce qui rend la conclusion solide. Sans lui, on aurait pu
  attribuer les 7.3 % à la sur-rejection bien connue des valeurs
  critiques asymptotiques en petit échantillon. Un modèle à deux
  régresseurs authentiques, même T, même nulle, est correctement
  dimensionné : la distorsion vient donc de la décomposition elle-même.

- **Mécanisme, mesuré et non supposé** — deux propriétés structurelles
  des sommes partielles, dont aucune n'est compatible avec ce que les
  tables de PSS supposent :

  1. `x+` et `x-` sont corrélées à **−0.993** en niveau, et leurs
     variations ne sont **jamais** toutes deux non nulles : chacune ne
     bouge qu'une date sur deux. Ce ne sont pas deux régresseurs I(1)
     indépendants ; ce sont deux morceaux d'une même série.
  2. Décomposer une série **stationnaire** produit deux séries à
     **tendance** (pente mesurée +0.56 sur 400 points). La borne I(0)
     est censée couvrir des régresseurs stationnaires : aucun monde de
     ce type n'est atteignable par la décomposition. Il n'y a donc pas
     de borne inférieure qui ait un sens ici, et une paire de bornes
     serait une fiction.

- **Action** : une table de valeurs critiques propre à ce nul, simulée
  sous le protocole déjà utilisé pour F_indep (spec 15) — y marche
  aléatoire, régresseurs tirés comme des marches aléatoires **puis
  décomposés**. Une seule valeur par seuil, pas une paire.
  `pyardl.critical_values.syg2014`, 100 000 réplications par
  configuration, cas 1-5, `k_asym` = 1 à 3.

- **Effet vérifié**, 1000 réplications à T = 150 :

  | cas | avant (PSS, k = 2) | après |
  |---|---|---|
  | III | 7.3 % | **5.7 %** |
  | V | — | 3.5 % |

  La table est asymptotique ; l'écart résiduel à T = 150 est la
  distorsion d'échantillon fini habituelle. Le cas V devient
  conservateur.

- **Effet de bord instructif** : le recoupement structurel « la valeur
  NARDL doit dépasser celle de PSS » échoue systématiquement dans les
  cas 4 et 5, et seulement là. La raison est la même dérive : dans les
  cas à tendance, elle est en partie colinéaire avec le terme de
  tendance du modèle, qui l'absorbe. Le sens de la distorsion s'inverse.
  Le contrôle a donc été restreint aux cas sans tendance, avec cette
  justification — pas relâché parce qu'il échouait.

- **Ce qui n'a PAS été fait** : garder la convention `decomposed` en la
  documentant comme « approximation courante ». Elle rejette une fois
  sur treize au lieu d'une fois sur vingt ; l'écrire en note de bas de
  page aurait été laisser l'utilisateur porter une erreur qu'on a
  mesurée.

- **Trace** : `validation/spec17_measurements.py`,
  `validation/spec17_nardl_cv.py`, résultats dans
  `validation/results/spec17_*`.
- **Statut** : mesurée et corrigée (2026-08-22).
