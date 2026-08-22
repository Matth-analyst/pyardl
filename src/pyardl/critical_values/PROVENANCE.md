# Provenance des valeurs critiques encodées

Règle du projet : aucune valeur critique « de
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

**Recoupement (moteur de simulation interne)** :
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

## `ks2020.py` — surfaces de réponse Kripfganz-Schneider (spec 13, voie A1)

**Référence méthodologique** : Kripfganz, S. & Schneider, D. C. (2020),
"Response Surface Regressions for Critical Value Bounds and Approximate
p-values in Equilibrium Correction Models", *OBES* 82(6), 1456-1481
(version ouverte : University of Exeter Discussion Paper 1901).

**Matériel utilisé (voie A1)** : module
`statsmodels.tsa.ardl.pss_critical_values` — qui n'est PAS une
redistribution des coefficients publiés par K&S : statsmodels a
RE-SIMULÉ les distributions (32 000 000 réplications par configuration,
méthodologie PSS/K&S, scripts `pss.py`/`pss-process.py` de statsmodels)
et ajusté ses propres polynômes de p-values asymptotiques. Licence
BSD-3 de statsmodels (dépendance runtime déjà requise) : pyardl ne
redistribue rien — import à l'exécution.

**Validation contre les valeurs publiées**
(`tests/unit/critical_values/test_ks2020.py`) : intercepts theta_{0,0}
des surfaces publiées (WP Exeter 1901, annexe D, Table 12 — CV
asymptotiques), transcrits le 2026-07-10 depuis
https://exetereconomics.github.io/RePEc/dpapers/DP1901.pdf ;
concordance à ±0.03 (10/5 %) et ±0.06 (1 %) — la tolérance 1e-3 de la
spec ne s'applique qu'à la voie A2 (mêmes coefficients), cf.
DEVIATIONS.md. Cohérences internes : CV -> bornes PSS (±0.15, l'erreur
MC des tables publiées), monotonies en k et alpha, aller-retour
p-value/CV à 1e-8, seuil 2.5 % recoupé contre la table interne de la
spec 12 (±0.05 — deux simulations indépendantes).

**OBS-4 confirmée** : la surface donne 3.3084 pour la cellule cas II,
k=2, 10 % I(1) — troisième source concordante contre la valeur publiée
PSS 3.35 (docs/VALIDATION_OBSERVATIONS.md).

**Couverture voie A1 (exceptions explicites)** : F seulement (les
p-values et surfaces du t arriveront avec les coefficients K&S — voie
A2, licence en cours de clarification — ou la voie B) ; k = 1..10 ;
asymptotique (pas d'ajustement fini-T). Les bornes t servies sous
cv_source="kripfganz" restent celles de PSS 2001 (composition
documentée dans bounds_test).

## `ks2020_finite.py` — surfaces K&S complètes, finies-T (spec 13, voie A2)

**STATUT (2026-07-19) : EXPÉRIMENTAL, NON VALIDÉ, BLOQUÉ PAR A3.** Le
code implémente la forme fonctionnelle publiée mais aucune valeur
critique produite par ce module n'est confirmée contre une sortie
Stata de référence légitime. Une comparaison antérieure s'appuyait sur
un exemplaire du fichier de coefficients téléchargé avant la réception
d'une autorisation des auteurs, et sur un exemplaire de l'article trouvé
sur un site tiers dont la légitimité n'a pas été établie ; les deux ont
été supprimés du cache local et leurs valeurs retirées des tests et de
cette page (voir docs/DEVIATIONS.md, CHANGELOG). **Ne pas
retélécharger, ni revalider, avant réception d'une réponse à la
demande de permission (voie A3,
docs/correspondence/2026-07-10_ks_license_draft.md).**

**Matériel visé** (une fois l'autorisation obtenue) : fichier
`ardl_surfreg_coefs.dta` distribué avec le package Stata ardl de
Kripfganz & Schneider (kripfganz.de) — 3 536 lignes de coefficients de
surface (F : cas I-V ; t : cas I/III/V ; 2 bornes ; grille de 221
quantiles en 1/10000e). Aucune licence explicite publiée -> pyardl ne
redistribuera pas ce fichier ; téléchargement prévu au premier usage
explicite (`download_surface_coefs()`), cache local (`PYARDL_CACHE` ou
`~/.pyardl`) avec SHA-256 et journal de provenance.

**Forme fonctionnelle implémentée** (non revalidée) : polynôme en
1/(k+1) (ordres 0..4) avec termes finis-T en 1/n, sr/n, 1/n², sr/n²,
1/n³, sr/n³, où sr = nombre de coefficients de court terme de l'UECM,
régresseurs fixes inclus ; p-values par approximation locale de
MacKinnon (1996, eq. 12). Ces conventions (notamment le sens de la
colonne `p` et le mapping des cas 2/4 vers 3/5 pour le t) ont été lues
dans le source `ardlbounds.ado` des auteurs, mais ne sont PAS validées
empiriquement — traiter comme hypothèses de travail jusqu'à
revalidation.

**Tests** (`tests/unit/critical_values/test_ks2020_finite.py`) :
uniquement des tests de cohérence interne (auto-comparaison, aucune
valeur de référence externe encodée) ; ils sont skip par défaut car ils
requièrent le fichier de coefficients, absent du cache. Le test
d'intégration bout-en-bout est un placeholder explicitement skip,
marqué "bloqué par A3".

---

## Frontières CUSUM et CUSUMSQ (Brown-Durbin-Evans 1975) — `bde1975.py`

### Frontières du CUSUM : coefficient `a`

**Source** : Brown, Durbin & Evans (1975), *JRSS B* 37(2), 149-192, §2.3.
Valeurs `a = 0.850` (10 %), `0.948` (5 %), `1.143` (1 %) — les trois
seuils tabulés par les auteurs, et les seuls servis. Toute autre valeur
d'`alpha` lève une exception explicite : il n'existe pas de quatrième
valeur publiée, et l'interpolation n'aurait pas de sens ici (`a` résout
une équation de probabilité de franchissement, ce n'est pas un
quantile).

**Recoupement** : la suite de frontières produite par `cusum()` est
comparée à celle de `statsmodels.stats.diagnostic.recursive_olsresiduals`
sur 3 configurations (n = 80/150/300, k = 2/3/5). **Écart strictement
nul** après correction d'un décalage d'un pas dans la convention de
départ (voir ci-dessous). Recoupement géométrique complémentaire : les
droites passent bien par `(k, a·sqrt(n))` et `(T, 3a·sqrt(n))`, la
définition de l'article.

#### PIÈGE — décalage d'un pas à l'origine de la récursion

À lire avant toute comparaison de nos frontières CUSUM avec une autre
implémentation. Ce décalage a coûté un faux diagnostic pendant la
validation de la spec 26.

**Le fait** : les `k` premières observations sont ajustées EXACTEMENT
par une régression à `k` régresseurs. Le résidu récursif en `t = k` est
donc identiquement nul, par construction et non par hasard.

**Les deux conventions** :

| | Premier point du chemin | Première demi-largeur | Longueur |
|---|---|---|---|
| Brown-Durbin-Evans 1975 (**pyardl**) | `t = k+1` | `a·sqrt(n) + 2a/sqrt(n)` | `n = T - k` |
| `statsmodels.recursive_olsresiduals` | `t = k` | `a·sqrt(n)` | `n + 1` |

pyardl suit l'article : le chemin commence au premier résidu porteur
d'information. statsmodels inclut le point dégénéré.

**Le symptôme si on l'ignore** : la comparaison terme à terme des deux
suites de frontières donne un écart d'apparence « petite mais non
nulle » (0.156 sur notre cas de test, soit ~1.3 % de la borne). On est
alors tenté de conclure à une divergence de formule, voire d'ajuster un
coefficient pour « rapprocher » les valeurs — ce qui introduirait une
vraie erreur pour corriger une fausse.

**Le diagnostic correct** : décaler d'un pas avant de comparer. L'écart
tombe à **0.00e+00 exactement**, sur 3 configurations (n = 80/150/300,
k = 2/3/5). Un écart strictement nul, et non « petit », est la signature
d'une pure différence d'indexation ; un écart petit mais non nul aurait
signalé un vrai problème de formule.

**Verrouillé par test** :
`tests/unit/diagnostics/test_stability.py::TestBoundaryCrossCheck`, qui
compare après décalage ET vérifie indépendamment la définition
géométrique de l'article (les droites passent par `(k, a·sqrt(n))` et
`(T, 3a·sqrt(n))`).

### Frontières du CUSUMSQ : table `c0`

**Statut** : GÉNÉRÉE par simulation interne
(`validation/spec26_cusumsq_c0.py`), non transcrite. Ce n'est pas un
pis-aller faute de table publiée : la statistique est
**distribution-free**, et la simulation est donc exacte au bruit de
Monte Carlo près, à la précision qu'on veut.

**Justification** : si les résidus récursifs sont i.i.d. N(0, sigma²),
alors `S_t = somme_{s<=t} w_s² / somme_{s<=n} w_s²` ne dépend plus de
sigma² — c'est un rapport de sommes partielles de n variables chi²(1)
i.i.d. La loi de `max_t |S_t - t/n|` ne dépend donc que de `n = T - k`.

**Paramètres** (tous journalisés, exécution reproductible) :
`n_sims = 200 000`, `seed = 20260802 + n` (distincte et déterministe par
point de grille), `chunk = 20 000`, seuils 10/5/1 %. Grille de 100
valeurs de `n`, de 4 à 1000, resserrée en petit échantillon où la
courbure est forte. Interpolation linéaire en `1/sqrt(n)`, échelle sur
laquelle la fonction est quasi affine.

**Recoupement** (`validation/results/spec26_c0_crosscheck.txt`) :
`c0(n)·sqrt(n/2)` doit converger vers le quantile de la loi de
Kolmogorov, puisque `S_t - t/n` se comporte comme un pont brownien
d'échelle `sqrt(2/n)` (la variance d'un chi²(1) vaut 2). Le ratio croît
de façon monotone : 0.63 à n = 4, 0.92 à n = 100, 0.98 à n = 1000. Il
reste inférieur à 1 en échantillon fini — l'approximation asymptotique
élargit donc la bande et rend le test conservateur, jamais l'inverse.
C'est aussi ce repli, avec avertissement explicite, qui sert au-delà de
n = 1000.

**Limite assumée** : aucune comparaison à la table publiée de Durbin
(1969) n'a pu être faite, faute d'accès à une source consultable
librement. Le recoupement est interne (asymptotique) et non externe. Si
une source fiable devient accessible, la comparaison cellule par cellule
reste à faire — voir `docs/QUESTIONS.md`.

---

## Valeurs critiques des pré-tests de racine unitaire (spec 27)

Les deux familles n'ont **pas le même statut de disponibilité**. Elles
sont donc traitées séparément, avec des protocoles de recoupement
différents.

### DF-GLS (Elliott, Rothenberg & Stock 1996) — `ers1996.py`

**Statut** : GÉNÉRÉE par simulation interne
(`validation/spec27_unitroot_cv.py`), avec **recoupement externe
disponible**.

**Paramètres** : 100 000 réplications par point de grille, marche
aléatoire gaussienne sous H0, dé-trending sous l'alternative locale
(c̄ = −7 pour `c`, −13.5 pour `ct`), statistique calculée à retards nuls
— convention de tabulation, la sélection de retards servant à estimer la
variance de long terme sur données réelles, pas à définir la loi limite.
Seed `20260803 + T` (+7919 pour `ct`), chunk 5 000. Grille de 19 valeurs
de T, de 25 à 2000. Interpolation linéaire en 1/T.

**Seconde source** : le package `arch` (Sheppard, BSD-3) expose des
surfaces de réponse elles-mêmes re-simulées selon la méthodologie
MacKinnon. Recoupement dans
`validation/results/spec27_cv_tolerance.txt`, critère dérivé =
3 × erreur type du quantile, obtenue par **bootstrap** (500 rééchan-
tillonnages) et non par la formule asymptotique : celle-ci exige
d'estimer la densité au quantile, et une fenêtre à pas fixe déborde de
(0, 1) quand p = 0.01, produisant une erreur type six fois trop grande —
mesuré, puis corrigé. L'erreur de `arch` étant inconnue, elle n'est PAS
ajoutée : le critère est donc plus strict que la réalité.

**Résultat** : concordance pour T ≥ 100. Divergence systématique à
T = 50 — voir OBS-6, tranchée par une expérience de taille en faveur de
nos valeurs.

### Statistiques M (Ng & Perron 2001) — `ngperron2001.py`

**Statut** : GÉNÉRÉE par la même simulation, **sans aucune seconde
implémentation disponible**.

Ni `arch` ni `statsmodels` ne fournissent les statistiques MZα, MZt, MSB
ou MPT — vérifié par inspection des espaces de noms des deux packages.
Aucun recoupement externe n'est donc possible, et aucune table publiée
librement consultable n'a été trouvée.

**Recoupement interne, non trivial** : MZt partage la loi asymptotique
du DF-GLS (Ng-Perron 2001, §2). Les deux tables étant simulées
séparément dans le même run, leur convergence quand T croît est une
preuve qu'aucune des deux ne repose sur une formule fausse. Mesuré
(`spec27_cv_crosscheck.txt`), cas `ct` à 10 % : écart 0.155 à T = 100,
0.070 à T = 200, 0.026 à T = 500, 0.012 à T = 1000, 0.006 à T = 2000 —
décroissance monotone au rythme attendu.

**Recoupement algébrique** : l'identité MZt = MZα × MSB est vérifiée à
1e-12 sur données réelles (test dédié). Elle lie trois des quatre
statistiques : une erreur dans l'une la romprait.

**Limite assumée** : les quatre tables n'ont pas de seconde source. Si
une devient accessible, la comparaison cellule par cellule reste à
faire — `docs/QUESTIONS.md`.

### PIÈGE — sur quelle série sélectionner les retards

Les tables sont indexées sur la **longueur de la série** (`len(y)`),
l'axe sur lequel elles ont été simulées, et non sur le nombre de lignes
de la régression ADF. Évaluer une surface de réponse externe à
1/(T−1) plutôt qu'à 1/T pour « corriger » un écart aggrave celui-ci —
vérifié, l'écart passe de 0.035 à 0.041 à T = 50.

Second piège, plus coûteux : la sélection de retards opère sur la série
**MCO-détrendée**, pas GLS. Sélectionner sur la série GLS fait
sur-sélectionner tous les critères et détruit la puissance du test
(52 % de rejet sur bruit blanc au lieu de 100 %). Détail et mesures dans
`docs/QUESTIONS.md`.

---

## Valeurs critiques d'Engle-Granger (spec 06) — `mackinnon.py`

**Source primaire** : MacKinnon, J. G. (1994), *JBES* 12(2), 167-176, et
MacKinnon, J. G. (2010), « Critical Values for Cointegration Tests »,
Queen's University WP 1227. Surfaces de réponse
`C(alpha, k, T) = tau_inf + b1/T + b2/T² + b3/T³`, un jeu de
coefficients par combinaison (seuil, nombre de variables, cas
déterministe).

**Canal** : les coefficients sont évalués via
`statsmodels.tsa.adfvalues` (`mackinnoncrit`, `mackinnonp`), qui les
transcrit. `statsmodels` étant DÉJÀ une dépendance runtime obligatoire
de pyardl, aucun matériel n'est dupliqué : il existe exactement une
copie de ces nombres dans l'environnement, et elle n'est pas la nôtre.

Écart assumé à la spec, qui demandait notre propre table de coefficients
— voir `docs/DEVIATIONS.md`. Raison : une seconde transcription des mêmes
nombres publiés ajoute un mode de défaillance (erreur de recopie) sans
ajouter de source. Le précédent inverse (Narayan, spec 12) se justifiait
parce que dynamac n'est PAS une dépendance de pyardl et qu'il fallait
donc embarquer les valeurs.

**Recoupement (seconde source, simulation interne)** :
`validation/spec06_eg_cv.py`, résultats dans
`validation/results/spec06_eg_cv_crosscheck.txt`.

DGP sous H0 : y et les k régresseurs sont des marches aléatoires
gaussiennes INDÉPENDANTES — aucune combinaison stationnaire n'existe.
Étape 1 en MCO, étape 2 en régression ADF sans déterministe ni retard,
quantiles de queue basse. 50 000 réplications par cellule, seed
`20260804 + T + 100k` (+7919 pour `ct`), chunk 2 500. Vectorisation sur
l'axe des réplications par **QR empilée** — chaque réplication a sa
propre matrice de régresseurs, et l'inversion de X'X reste interdite.

Le générateur batché est lui-même vérifié contre l'implémentation
scalaire `engle_granger(..., max_lags=0)` : écart maximal 7.6e-15 sur
les quatre combinaisons (trend, k) testées. Une table générée par un
générateur non vérifié ne prouverait rien.

**Résultat** : critère dérivé = 3 x l'erreur type du quantile simulé,
par **bootstrap** (400 rééchantillonnages, aucune estimation de densité
— même correction que celle apportée en spec 27). L'erreur des surfaces
publiées étant inconnue, elle n'est PAS ajoutée : le critère est plus
strict que la réalité. **54 cellules sur 54 dans le critère**
(trend c/ct x k = 1/2/3 x T = 100/250/500 x seuils 10/5/1 %). Écart
maximal observé 0.0376 pour une tolérance de 0.0397.

**Couverture et limitations (exceptions explicites)** : `trend='n'` n'a
pas de surface publiée par MacKinnon (2010) → NaN + warning, et
`decision()` lève une exception plutôt que de trancher ; jamais de valeur
empruntée à un cas déterministe voisin. Nombre de variables limité à 12
(limite des surfaces) → exception orientant vers le bounds test, qui n'a
pas de régression de première étape.

### PIÈGE — deux conventions d'arrondi qui divergent

1. **Règle de Schwert.** Schwert (1989) définit le nombre maximal de
   retards comme `floor(12 (T/100)^{1/4})` ; `statsmodels` arrondit
   AU-DESSUS (`ceil`). pyardl suit la règle publiée. À T = 200 cela fait
   14 contre 15, et ce retard supplémentaire déplace l'échantillon
   commun de la sélection : sur une quasi-égalité de l'AIC (0.804796
   contre 0.805282, soit 0.06 %), le choix bascule de k = 0 à k = 4 et
   la statistique passe de -14.11 à -7.82. Les tests de concordance
   passent donc `max_lags` explicitement des deux côtés, pour mesurer un
   écart de CALCUL et non un écart de règle d'arrondi. Avec max_lags
   aligné : 18 configurations sur 18 concordent à 1e-13.
2. **Taille d'échantillon d'évaluation de la surface.** `statsmodels`
   évalue les valeurs critiques à `nobs - 1`, avec ce commentaire dans
   son source : « pour coller à egranger de Stata, je ne sais pas
   pourquoi ». pyardl évalue à `nobs`, la taille réellement utilisée par
   la régression de première étape. L'écart est de l'ordre de 0.005 à
   T = 200 et décroît en 1/T ; le test de concordance des valeurs
   critiques le tolère explicitement à 0.02 avec la raison en commentaire.

---

## Valeurs critiques bootstrap (spec 14) — `pyardl.bootstrap`

**Nature** : elles ne sont pas tabulées. Elles sont calculées à chaque
appel, à partir des données de l'utilisateur, en régénérant des séries
sous une hypothèse nulle vraie par construction. Il n'y a donc rien à
transcrire ni à recouper contre une table — la question de provenance se
déplace vers la reproductibilité et l'exactitude de l'algorithme.

**Reproductibilité** : seed obligatoire dans l'objet résultat, tirée de
l'entropie système et JOURNALISÉE quand l'utilisateur n'en fournit pas.
Même seed -> mêmes valeurs critiques au bit près, vérifié par test.
Le nombre de réplications, le schéma de rééchantillonnage, l'ordre du
VAR marginal et le burn-in sont également journalisés.

**Validation externe (§4.5)** : package R **bootCT** (CRAN), exécuté le
2026-08-17 sur les données danoises livrées, B = 2000 des deux côtés.
Script `validation/external/spec14_bootct.R`, sortie brute dans
`validation/results/external_logs/`, comparaison dans
`validation/results/spec14_bootct_comparison.txt`.

- Statistiques OBSERVÉES : concordance à **4e-10** sur le F et **5e-10**
  sur le t. Les deux implémentations estiment bien le même modèle.
- Valeurs critiques du F : écart de 0.6 % à 13 %, cohérent avec deux
  bootstraps à générateurs différents.
- Valeurs critiques du t : écart de 21 % à 30 %, systématiquement dans
  le même sens — les nôtres sont plus exigeantes. Point ouvert
  documenté dans `docs/QUESTIONS.md` (hypothèse : DGP nul du t).
- Décisions à 10, 5 et 1 % : identiques des deux côtés.

### PIÈGE — la convention de `fix.ardl` dans bootCT

`fix.ardl` désigne les ordres des **différences retardées de l'UECM**,
pas les ordres de l'ARDL en niveaux. Pour comparer avec un ordre PSS
(p ; q_1, ..., q_k) il faut passer (p-1 ; q_1-1, ..., q_k-1).

Symptôme si on l'ignore : avec le mapping naïf `c(3,1,3,2)` pour notre
ordre (3 ; 1, 3, 2), bootCT ajuste 17 régresseurs au lieu de 13 et rend
F = 3.93 contre nos 6.21. On aurait conclu à un désaccord de 58 % entre
deux implémentations qui, en réalité, estimaient deux modèles
différents. Diagnostic : lire les NOMS des coefficients du modèle
ajusté (`names(coef(res$ARDL))`), qui montrent immédiatement trois
retards de ΔLRM là où la paramétrisation PSS en compte deux.


## Bornes de F_indep — Sam, McNown & Goh (2019), spec 15

**Statut : SIMULÉES, pas transcrites.** Les bornes publiées sont dans
*Economic Modelling* 80, 130-141, sous barrière d'accès. Le projet
n'encode pas une valeur critique qu'il n'a pas calculée, et la spec
prévoit elle-même le repli par le moteur `simulate_bounds`.

**Générateur** : `validation/spec15_findep_cv.py`. Paramètres du DGP et
seeds en tête de fichier, une seed distincte et déterministe par
configuration `(cas, k, i1)`. Sortie brute :
`validation/results/spec15_findep_table.py`, injectée dans
`smg2019.py` par script avec vérification cellule par cellule contre le
fichier source (règle CLAUDE.md n°9).

**Nul simulé** : exactement celui de PSS — y marche aléatoire,
régresseurs i.i.d. pour la borne I(0), marches aléatoires indépendantes
pour la borne I(1). F_indep est calculé sur les MÊMES réplications que
F_overall et t_BDM : les trois jeux de bornes décrivent un seul monde,
pas trois mondes voisins.

**Recoupements** (aucune seconde source n'existe, ils sont donc
structurels — `validation/results/spec15_findep_crosscheck.txt`) :

1. borne I(0) <= borne I(1) pour toute configuration ;
2. seuil plus strict -> borne plus élevée ;
3. décroissance en k : F_indep est un F PAR restriction, chaque
   régresseur supplémentaire dilue la statistique.

**Couverture** : cas 1 à 5, k = 1..10, niveaux 10 / 5 / 1 %. Hors de
cette grille, `findep_bounds` lève une exception et `decision_indep`
vaut `None` — aucune valeur voisine n'est substituée.

## Valeurs critiques du bounds test NARDL — spec 17

**Statut : SIMULÉES, et il n'existe rien à transcrire.** La littérature
NARDL lit ses statistiques contre les tables de PSS, en comptant soit les
deux sommes partielles, soit la variable d'origine. La mesure a écarté
les deux : ni l'une ni l'autre ne tient sa taille.

| lecture, à 5 % nominal | taux de rejet mesuré |
|---|---|
| `decomposed`, k = 2 par variable | 7.3 % |
| `original`, k = 1 par variable | 2.6 % |
| **témoin** : ARDL linéaire, 2 vrais régresseurs | 4.8 % |

Le témoin est ce qui rend la conclusion solide : un modèle à deux
régresseurs authentiques, même T, même nulle, est correctement
dimensionné. La distorsion ne vient donc pas des valeurs critiques
asymptotiques en petit échantillon — elle vient de la décomposition.

**Mécanisme, mesuré et non supposé** :

1. `x+` et `x-` sont corrélées à **−0.993** en niveau, et leurs
   variations ne sont jamais toutes deux non nulles : chacune bouge une
   date sur deux. Ce ne sont pas deux régresseurs I(1) indépendants, ce
   que les tables de PSS supposent.
2. Décomposer une série **stationnaire** produit deux séries à
   **tendance** (pente mesurée +0.56 sur 400 points). La borne I(0),
   censée couvrir des régresseurs stationnaires, ne décrit donc aucun
   monde atteignable par la décomposition. Une seule valeur critique a
   du sens, pas une paire — et c'est ce que la table livre.

**Générateur** : `validation/spec17_nardl_cv.py`. 100 000 réplications
par configuration, T = 1000, seed déterministe par `(cas, k_asym)`. Nul
simulé : y marche aléatoire, chaque variable asymétrique tirée comme une
marche aléatoire **puis décomposée** — la transformation exacte que subit
la donnée réelle. Injection par `validation/spec17_inject_table.py`, 45
cellules vérifiées une à une contre le fichier source.

**Recoupements structurels**
(`validation/results/spec17_nardl_cv_crosscheck.txt`) :

1. seuil plus strict → valeur plus élevée ;
2. décroissance en `k_asym` (le F est un F par restriction) ;
3. dans les cas **sans tendance** (1, 2, 3), la valeur dépasse la borne
   I(1) de PSS à k = 2·k_asym — c'est exactement ce qui rend la lecture
   usuelle trop permissive.

Le contrôle 3 s'arrête là, et la raison est mesurée : dans les cas 4 et
5, la dérive que porte toute somme partielle est en partie colinéaire
avec le terme de tendance du modèle, qui l'absorbe. La valeur critique
passe alors **sous** celle de PSS — le sens de la distorsion s'inverse,
il ne disparaît pas.

**Effet vérifié à T = 150**, sur 1000 réplications sous H0 :

| cas | avant (PSS, k = 2) | après (table NARDL) |
|---|---|---|
| 3 | 7.3 % | **5.7 %** |
| 5 | — | 3.5 % |

La table est asymptotique (T = 1000) ; l'écart résiduel à T = 150 est la
distorsion d'échantillon fini habituelle, du même ordre que celle qui
justifie les tables de Narayan pour l'ARDL linéaire. Le cas 5 devient
conservateur.

**Couverture** : cas 1 à 5, `k_asym` = 1 à 3, niveaux 10 / 5 / 1 %,
modèles dont **tous** les régresseurs sont décomposés. Hors de cette
grille, `nardl_critical_value` lève une exception et `bounds_test` refuse
— aucune valeur voisine n'est substituée.
