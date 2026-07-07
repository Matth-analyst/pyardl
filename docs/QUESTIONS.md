# Points en attente / spécifications ambiguës

Chaque entrée : spec concernée, point ambigu ou suspect, interprétation
retenue (la plus standard de la littérature), test marqué
`@pytest.mark.needs_review` correspondant.

## Spec 03 §2.2 — formule de ω_{j,0} et cas limite q_j = 0

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
sur {λ, γ_1, ..., γ_k} suppose des régresseurs de niveau retardés
(I(1) sous H0) ; un régresseur avec q_j = 0 entre dans ce test via x_{j,t}
(contemporain, stationnaire par construction) et non x_{j,t-1} — point à
documenter explicitement dans la spec 10 au moment de son implémentation.

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
