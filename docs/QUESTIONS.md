# Points en attente / spécifications ambiguës

Chaque entrée : spec concernée, point ambigu ou suspect, interprétation
retenue (la plus standard de la littérature), test marqué
`@pytest.mark.needs_review` correspondant.

## Spec 05 §6.5 — auto_ardl : « mêmes ordres sélectionnés » non exigible

**Constat (validation externe du 2026-07-07, R 4.6.1, ARDL 0.2.5)** :
sur les données danoises (max_order = 5), R `auto_ardl` sélectionne
(1,0,0,0) en BIC et (3,1,3,2) en AIC ; `ARDL.select_order` (échantillon
commun) sélectionne d'autres ordres. Diagnostic complet :

1. **Politique d'échantillon** : auto_ardl évalue chaque candidat via
   `stats::BIC/AIC` sur SON échantillon maximal propre — les IC ne sont
   pas comparables entre candidats (piège spec 02 §4 / spec 05 §3.2,
   que ardlpy interdit par construction). En ÉMULANT cette politique
   avec notre moteur (grille complète, hold_back propre), l'optimum BIC
   coïncide exactement avec le choix d'auto_ardl (1,0,0,0) — ce qui
   valide notre calcul de llf/BIC contre R.
2. **Recherche non exhaustive** : auto_ardl utilise une recherche
   stepwise ; son choix AIC (3,1,3,2) n'est pas l'optimum global de sa
   propre politique (la grille complète donne (5,0,3,5) en AIC
   échantillon-propre). Aucune concordance d'ordre n'est donc exigible
   pour l'AIC.

**Décision (conforme à la note du script spec05_r_ardl.R et à
la correction statistique prime)** : ne PAS aligner ardlpy
sur la politique d'auto_ardl ; l'exigence « mêmes ordres sélectionnés »
de la spec 05 §6.5 est réinterprétée comme « même optimum sous politique
d'échantillon identique », testée dans
`tests/replication/test_spec05.py`. Les coefficients à ordres fixes
concordent à 1e-6 (contrat numérique intact).

## Spec 12 §3.1 — recoupement PSS : 23 cellules hors tolérance ±0.05 (EN ATTENTE D'ARBITRAGE)

**Constat (2026-07-07, `validation/spec12_montecarlo.py`, T=1000,
n_sims=100 000, seeds journalisées)** : 505/528 cellules des tables PSS
2001 encodées sont reproduites à ±0.05 (F) / ±0.04 (t). 23 cellules
dépassent, dont 17 au seuil 1 % (queue de distribution, erreur MC
maximale). Détail : `validation/results/spec12_pss_crosscheck.csv`.

**Diagnostic (triple vérification)** :
1. Sur les 14 cellules F fautives à k >= 1, notre simulation concorde
   avec Kripfganz-Schneider 2020 (la re-simulation la plus précise
   publiée) à ±0.06, alors que les valeurs PSS publiées s'écartent de
   K&S jusqu'à 0.14 — le moteur est validé ; l'écart est l'erreur MC
   des tables PSS elles-mêmes (40 000 réplications en 2001).
2. Cellules k=0 (sans source K&S) : deux runs indépendants à 300 000
   réplications concordent entre eux à ±0.02 et convergent vers PSS
   pour les cas IV/V (15.73 exact) — sauf **cas I, k=0, 1 % : écart
   persistant -0.22** (PSS 7.17 vs ~6.95 convergé), à traiter à part
   (valeur PSS possiblement imprécise ou de source différente ; pas de
   seconde source disponible).
3. La cellule t fautive (cas I, k=10, 1 % I(1)) : écart +0.042 contre
   une tolérance de ±0.04 — marginal.

**Statut : CLOS (arbitrage utilisateur du 2026-07-07, raffiné)** :
critère retenu = tolérance PAR CELLULE de 3 erreurs types combinées,
DÉRIVÉE du calcul d'erreur MC des quantiles (formule et dérivation :
PROVENANCE.md, validation/spec12_mc_error.py) — pas de seuil ad hoc.
Résultat du recoupement officiel (100 000 réplications, critère
appliqué) : **527/528 cellules dans le critère**. La cellule cas I,
k=0, 1 % est requalifiée : sa tolérance dérivée vaut 0.35 (queue
épaisse), l'écart -0.23 n'est qu'à ~2σ — dans l'erreur MC attendue de
la table publiée (VALIDATION_OBSERVATIONS.md, OBS-2, needs_review
maintenu). L'unique dépassement (cas II, k=2, 10 % I(1), 3.7σ,
confirmé par K&S contre la valeur publiée) est documenté en OBS-4 et
tient dans la marge « 0-3 dépassements fortuits » du test slow.
cv_source="pss" continue de servir les valeurs publiées à l'identique
(reproduction de la littérature) ; hiérarchie documentée dans
get_bounds. Note de révision reportée dans la spec 12 §3.

## Spec 10 §4 — DETTE : recoupement des tables PSS 2001 par simulation

**Statut : SOLDÉE (spec 12, 2026-07-07).** Le moteur `simulate_bounds`
a recoupé l'intégralité des 528 cellules encodées (100 000
réplications, T=1000, seeds journalisées) : 527/528 dans le critère de
3 erreurs types combinées (cf. entrée spec 12 §3.1 ci-dessus), Y
COMPRIS les cellules qui n'avaient aucune seconde source (colonnes
I(1) du t pour k >= 1, lignes k=0 du F) — les marques `needs_review`
de monotonie structurelle sont couvertes par ce recoupement. Le seuil
2.5 % est fourni par simulation interne (`pss2001_p025.py`, provenance
distincte, needs_review + test d'encadrement 5 %/1 %). Résultats :
`validation/results/spec12_pss_crosscheck.csv`. Texte d'origine de la
dette ci-dessous, conservé pour l'historique.

*(Historique)* Les bornes asymptotiques PSS 2001
encodées dans `src/ardlpy/critical_values/pss2001.py` sont recoupées par
une seconde source pour : (a) toutes les valeurs F, k = 1..10 (surfaces
asymptotiques Kripfganz-Schneider via statsmodels, ±0.15) ; (b) les
colonnes I(0) du t et la ligne k = 0 (CV Dickey-Fuller MacKinnon,
±0.03). Restent SANS seconde source : les colonnes **I(1) du t pour
k ≥ 1** et les lignes **k = 0 du F** (statsmodels commence à k = 1) —
vérifiées seulement par monotonie structurelle (tests marqués
`needs_review`). Le **seuil 2.5 %** de l'article n'est pas couvert
(exception explicite dans `get_bounds`). À solder à la spec 12 : le
moteur `simulate_bounds` recoupera l'intégralité des tables et ajoutera
le seuil 2.5 %. Détail complet : `src/ardlpy/critical_values/PROVENANCE.md`.

## Spec 05 §4 — GETS : « éliminer le retard le moins significatif »

**Point ambigu.** La spec ne précise pas si l'élimination itérative peut
créer des « trous » dans la structure des retards (ex. garder x_{t} et
x_{t-2} mais pas x_{t-1}), ou si elle doit préserver une structure
ARDL(p, q) contiguë.

**Interprétation retenue (standard)** : réduction contiguë — seul le
retard TERMINAL de chaque variable (y.Lp, x_j.Lq_j) est candidat à
l'élimination, ce qui fait passer de ARDL(p, q) à ARDL(p−1, q) ou
ARDL(p, q_j−1). C'est le comportement des implémentations de référence
(la réduction d'ordre de `gets_ardl_uecm` dans ardl.nardl, la logique de
sélection d'EViews/statsmodels), et cela garantit que le modèle final
reste un ARDL représentable par les conteneurs de la spec 03. Une
élimination avec trous produirait un modèle hors du cadre ARDL(p,q) et
casserait `.to_ecm()`.

**Vérification** : `tests/unit/core/test_ardl.py::TestGETS`, test
principal marqué `@pytest.mark.needs_review` en attente de confirmation
(comparaison avec `ardl.nardl::gets_ardl_uecm` — script external à
prévoir avec la spec 17 qui cible ce package).

## Spec 03 §2.2 — formule de ω_{j,0} et cas limite q_j = 0

> **Statut : CLOS (2026-07-07)** : dérivation de ω et lecture 2 pour
> q_j = 0 validées par revue humaine (formule corrigée reportée dans la
> spec, note de révision 03_sargan_1964.md §2.2), puis **confirmées par
> l'exécution du script R** (`spec03_ardl_uecm.R`, R 4.6.1, ARDL 0.2.5) :
> pour l'ordre c(2,2,2,0), R ARDL::uecm place bien le niveau IDE
> contemporain (colonne `IDE`, pas `L(IDE,1)`), sans terme `d(IDE)`, avec
> résidus ardl/uecm identiques (1.6e-14) et coefficients ECM concordants
> à 1e-6 avec ardlpy (tests/replication/test_spec03.py). Marque
> `needs_review` du test d'équivalence levée.

**Point suspect.** La spec écrit elle-même la formule de ω_{j,i} comme
provisoire et demande explicitement de la dériver et de la tester
(« vérifier ... et TESTER numériquement, cf. §6.1, c'est la source
d'erreurs n°1 »). La formule brouillon donnée (`ω_{j,0} = −(β_{j,1}+...+β_{j,q_j})`)
est mathématiquement incorrecte : elle ne redonne pas des résidus
identiques entre les formes ARDL et ECM (violation de l'identité exacte
x_t = x_{t-1} + Δx_t).

**Dérivation retenue** (sommation par parties, identité exacte, standard
dans la littérature — Pesaran-Shin-Smith 2001, `R::ARDL::uecm()`,
`statsmodels.tsa.ardl.UECM`) :

- ω_{j,0} = β_{j,0} (coefficient du Δx_{j,t} contemporain)
- ω_{j,i} = −Σ_{m=i+1}^{q_j} β_{j,m}, pour i = 1, ..., q_j−1 (formule de
  la spec, valable pour i ≥ 1)

**Cas limite q_j = 0** (non traité explicitement par la spec). Deux
lectures ont été essayées :

1. *Rejetée après test empirique* : forcer un terme ω_{j,0} = β_{j,0}
   même quand q_j = 0, en gardant γ_j sur x_{j,t-1}. Cela ajoute une
   colonne x_{j,t-1} indépendante de Δx_{j,t} au design ECM alors que le
   modèle ARDL(p, q_j=0) n'a qu'un seul degré de liberté pour ce
   régresseur (β_{j,0} sur x_{j,t} seul) : {x_{j,t-1}, Δx_{j,t}} engendre
   un sous-espace de dimension 2 (= {x_{j,t-1}, x_{j,t}}), strictement
   plus grand que {x_{j,t}} (dimension 1). L'OLS sur l'ECM obtient alors
   une SSR strictement inférieure à celle de l'ARDL — les résidus ne
   coïncident pas (test échoué : 3/8 tirages aléatoires, tous ceux avec
   au moins un q_j = 0).
2. **Retenue** : quand q_j = 0, ω_j est vide (aucun terme de court terme
   propre) et γ_j multiplie x_{j,t} (contemporain), et non x_{j,t-1},
   dans la régression ECM — ce qui restaure l'égalité dimensionnelle
   (1 régresseur ARDL <-> 1 régresseur ECM) et donc l'identité exacte des
   résidus. Numériquement γ_j = β_{j,0} reste vrai (formule générale
   inchangée) ; seule la matrice de dessin diffère (spec 05, hors
   périmètre de la spec 03 qui ne manipule que les coefficients).

**Conséquence pour la suite (spec 10, bounds test)** : le test conjoint
sur {λ, γ_1, ..., γ_k} porte sur des régresseurs de niveau ; un
régresseur avec q_j = 0 y entre via x_{j,t} (contemporain) et non
x_{j,t-1}. x_{j,t} reste I(1) sous H0 : par l'identité exacte
x_{j,t} = x_{j,t-1} + Δx_{j,t}, l'écart entre les deux datations est
stationnaire, donc l'asymptotique du test est inchangée — le régresseur
de niveau est toujours intégré d'ordre 1, seule sa datation diffère
d'une période. Point à documenter explicitement dans la spec 10 au
moment de son implémentation. *(Corrigé en revue le 2026-07-07 : la
première rédaction qualifiait à tort x_{j,t} de « stationnaire par
construction ».)*

**Vérification** : `tests/unit/core/test_transforms_equivalence.py`
(résidus identiques à 1e-10 sur 8 tirages aléatoires, dont 3 couvrant
q_j = 0 — tous passent avec la lecture 2), marqué
`@pytest.mark.needs_review` en attendant une confirmation croisée avec
`R::ARDL::uecm()` sur un cas q_j = 0 (spec 03 §6.2, actuellement
`external`, script non encore exécuté).

**Confirmation indépendante obtenue** : `statsmodels.tsa.ardl.UECM`
**refuse explicitement** q_j = 0 à la construction du modèle
(`ValueError: All included exog variables must have a lag length >= 1`,
vérifié interactivement avec statsmodels 0.14.4). C'est une preuve
indépendante forte que la forme UECM standard ne peut pas représenter ce
cas limite avec un unique jeu (γ_j, ω_j) sans sur-paramétrisation —
exactement le diagnostic ci-dessus. `ardlpy` choisit de le supporter quand
même (contrairement à statsmodels), via la convention documentée
(lecture 2). Pour q_j ≥ 1, la concordance `ardl_to_ecm` <->
`UECM.from_ardl` a été vérifiée numériquement à 1e-6
(`tests/unit/core/test_transforms_statsmodels.py`), y compris pour
ω_{j,0} = β_{j,0} (le point que la spec demandait de dériver et tester).
