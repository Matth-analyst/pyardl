# Provenance des valeurs critiques encodées

Règle : aucune valeur critique « de
mémoire » ; chaque table cite sa source exacte et est recoupée par une
seconde source ou par le moteur de simulation interne (spec 12).

## `pss2001.py` — bornes asymptotiques du bounds test

**Source primaire** : Pesaran, M. H., Shin, Y. & Smith, R. J. (2001),
"Bounds Testing Approaches to the Analysis of Level Relationships",
*Journal of Applied Econometrics*, 16(3), 289–326 (DOI 10.1002/jae.616) :

- Statistique F : tables CI(i) à CI(v), pp. 300–301 ;
- Statistique t : tables CII(i), CII(iii), CII(v), pp. 303–304.

**Canal de transcription** : l'article original étant sous barrière
d'accès (Wiley), les valeurs ont été transcrites depuis le code source
public du package R **dynamac** (Jordan & Philips, 2018 — spec 25),
fichier `R/dynamac.R`, fonction `pssbounds()`, branche asymptotique
(commit HEAD du dépôt github.com/andyphilips/dynamac cloné le
2026-07-07). Il s'agit de la transcription des tables PSS 2001, pas de
valeurs propres au package. Seules les *données* publiées ont été
reprises, aucun code.

**Recoupement (seconde source), automatisé dans
`tests/unit/critical_values/test_pss2001.py`** :

1. **F, k = 1..10, tous cas, tous seuils** : comparaison aux valeurs
   critiques asymptotiques de Kripfganz & Schneider (2020) embarquées
   dans `statsmodels.tsa.ardl.pss_critical_values.crit_vals`
   (percentiles 90/95/99). K&S re-simulent les distributions PSS avec
   une précision supérieure : accord attendu à ±0.15 près (tolérance du
   test), les écarts reflétant la précision de simulation de PSS 2001
   (40 000 réplications, T = 1000).
2. **t, colonnes I(0) (tous k) et ligne k = 0 (les deux bornes)** : la
   borne I(0) du t est la valeur critique Dickey-Fuller asymptotique
   (sans constante pour le cas I, avec constante pour le cas III, avec
   tendance pour le cas V) — comparaison aux CV asymptotiques MacKinnon
   de `statsmodels.tsa.adfvalues.mackinnoncrit` à ±0.03.
3. **t, colonnes I(1), k >= 1 et lignes k = 0 du F** : pas de seconde
   source externe — recoupées par le **moteur de simulation interne**
   (spec 12, 2026-07-07) : recoupement intégral des 528 cellules à
   100 000 réplications, 527/528 dans le critère de 3 erreurs types
   combinées (voir section « Critère de recoupement » ci-dessous ;
   l'unique dépassement, à 3.7σ, est documenté en OBS-4 du registre
   docs/VALIDATION_OBSERVATIONS.md). Dette QUESTIONS.md soldée.

**Coquille détectée dans dynamac (en vue d'une issue upstream)** :

- **Position** : `R/dynamac.R`, fonction `pssbounds()`, branche
  `obs <= 35`, cas I (`case == 1`), matrice `fmat`, dernière ligne
  (k = 10), première colonne (seuil 10 %, borne I(0)).
- **Valeur** : `11.60` ; valeur correcte : `1.60`.
- **Démonstration** (trois preuves indépendantes) :
  1. *Monotonie en k* : dans toutes les tables CI de PSS 2001, les
     bornes F décroissent strictement en k (chaque restriction
     supplémentaire dilue la statistique). La colonne 10 %/I(0) du cas I
     décroît de 3.00 (k=0) à 1.63 (k=9) ; une valeur de 11.60 en k=10
     est incompatible (elle dépasserait même le 1 % de k=0).
  2. *Cohérence I(0) <= I(1)* : la borne I(0) doit être inférieure ou
     égale à la borne I(1) du même point, ici 2.72 ; 11.60 > 2.72
     violerait la définition même des bornes.
  3. *Seconde source* : la valeur asymptotique Kripfganz-Schneider 2020
     (statsmodels, clé `(10, 1, False)`, percentile 90) vaut ≈ 1.598,
     cohérente avec 1.60 et pas avec 11.60.
- **Confirmation interne** : la branche asymptotique du même fichier
  (`else # asymtotic`, cas I, même position) porte la valeur correcte
  `1.60` — la coquille est une erreur de duplication propre à la
  branche `obs <= 35` (qui recopie les valeurs asymptotiques pour le
  cas I, non couvert par Narayan 2005).
- Les tables encodées ici utilisent `1.60`.

**Couverture et limitations PSS (exceptions explicites dans le code)** :

- Seuils publiés transcrits : 10 %, 5 %, 1 %. Le seuil **2.5 %**
  (publié dans PSS 2001 mais absent de la transcription dynamac et non
  recoupable via K&S) est fourni par **simulation interne**
  (`pss2001_p025.py`, section dédiée ci-dessous) — provenance
  distincte, needs_review + test d'encadrement 5 %/1 %.
- k = 0..10 (limite des tables PSS) ; au-delà → exception renvoyant
  vers les specs 12/13.
- t : cas I, III, V uniquement (PSS 2001 ne publie pas de bornes t pour
  les cas II et IV, où le t_BDM n'est pas applicable) → exception
  explicite.
- Les tables PSS supposent T = 1000 (asymptotique) : pour les petits
  échantillons, voir Narayan 2005 (spec 12) et les surfaces de réponse
  ajustées en T (spec 13).

## `narayan2005.py` — bornes F petits échantillons (spec 12)

**Source primaire** : Narayan, P. K. (2005), "The saving and investment
nexus for China: evidence from cointegration tests", *Applied
Economics*, 37(17), 1979–1990 — tables « Case II / III / V »
(F seulement ; pas de bornes t), T = 30..80 par pas de 5, k = 0..7,
seuils 10/5/1 %.

**Canal de transcription** : parseur programmatique
(`validation/external/extract_narayan_tables.py`) appliqué au source R
public de dynamac (Jordan & Philips), fichier `R/dynamac.R`, branches
`obs <= 30` à `obs <= 80` de `pssbounds()` (dépôt
github.com/andyphilips/dynamac cloné le 2026-07-07). Le fichier
`narayan2005.py` est GÉNÉRÉ (jamais édité à la main) — l'extraction
programmatique élimine le risque d'erreur de recopie manuelle. Seules
les données publiées sont reprises, aucun code.

**Recoupement (moteur de simulation interne, règle du projet)** :
`tests/unit/critical_values/test_narayan.py` — cellules T = 40 et 60
recoupées par `simulate_bounds` (Narayan a utilisé le même DGP que PSS
2001 avec T fini, 40 000 réplications) : tolérance ±0.1 à n_sims élevé
(version slow), ±0.15 à 20 000 (version fast_mc). Cohérences
structurelles vérifiées sur TOUTES les cellules : I(0) <= I(1),
décroissance en k, décroissance vers l'asymptotique PSS quand T croît.

**Couverture et limitations (exceptions explicites)** : cas I et IV non
publiés par Narayan → erreur orientant vers cv_source="kripfganz"
(spec 13) ; pas de bornes t → erreur ; k <= 7 ; seuil 2.5 % non publié.
Interpolation linéaire entre tailles adjacentes (documentée) ; hors
plage [30, 80] → repli asymptotique PSS + warning.

## Critère de recoupement par simulation (arbitrage du 2026-07-07)

Le recoupement d'une table publiée par le moteur interne compare deux
quantiles empiriques, chacun porteur d'une erreur MC. Critère retenu
(dérivé, pas de seuil ad hoc) : **écart admissible par cellule = 3 x
l'erreur type combinée**

    SE_comb = sqrt( SE(q_p; n_pub)^2 + SE(q_p; n_sim)^2 ),
    SE(q_p; n) = sqrt(p(1-p)/n) / f(q_p),

avec n_pub = 40 000 (PSS 2001 comme Narayan 2005), n_sim = 100 000, et
f(q_p) la densité au quantile estimée par différence finie centrée de
fenêtre 0.005 (en probabilité) sur les tirages simulés. Dérivation et
valeurs représentatives : `validation/spec12_mc_error.py` (ordre de
grandeur des SE combinées : 0.01-0.05 à 10/5 %, 0.03-0.16 à 1 % selon
la dispersion de la cellule — d'où l'intenabilité d'une tolérance
uniforme ±0.05, cf. note de révision de la spec 12). Validation croisée
par la dispersion inter-seeds observée (runs indépendants à 300k :
écarts <= 0.02, cohérents avec les SE calculées).

Lecture des résultats : à 3σ sur 528 cellules, 0 à 3 dépassements
fortuits sont attendus ; seuls les dépassements PERSISTANTS entre seeds
indépendantes sont des anomalies — registre :
`docs/VALIDATION_OBSERVATIONS.md` (OBS-2 : cas I, k=0, 1 %).

## Seuil 2.5 % des tables PSS (spec 12, simulation interne)

Le seuil 2.5 % publié par PSS 2001 n'étant pas transcrit par le canal
dynamac (3 seuils seulement), il est fourni par le moteur interne :
`validation/spec12_montecarlo.py` (T = 1000, n_sims = 100 000, seeds
journalisées dans le script, quantiles 0.975/0.025) — tables
`F_P025`/`T_P025` intégrées à `pss2001.py` avec cette provenance. La
précision MC est ~±0.02 ; ces valeurs sont marquées comme
« simulation interne » et non « PSS 2001 publié ».
