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

**Divergence de convention avec statsmodels** (documentée, assumée) :
statsmodels fait commencer le chemin CUSUM à `t = k`, où le résidu
récursif est identiquement nul par construction (le modèle ajuste
exactement les `k` premières observations). Brown-Durbin-Evans le font
commencer à `t = k+1`, c'est-à-dire au premier résidu porteur
d'information. pyardl suit l'article. Conséquence : nos chemins ont un
point de moins, et la première frontière vaut `a·sqrt(n) + 2a/sqrt(n)`
au lieu de `a·sqrt(n)`.

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
