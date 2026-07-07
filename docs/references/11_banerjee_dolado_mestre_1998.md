# Spec 11 — Banerjee, Dolado & Mestre (1998) : le t-test ECM

## Référence
BDM (1998), *Journal of Time Series Analysis*, 19(3), 267–283.
DOI: 10.1111/1467-9892.00091. Clé : `banerjee1998ecm` · Branche : **3**.
Module : `ardlpy.bounds` (intégré au bounds test) · Priorité : v0.1.

## 1. Apport
Test de cointégration en équation unique fondé sur la statistique t du
coefficient de correction d'erreur λ dans l'ECM conditionnel : H₀ : λ = 0
(pas de force de rappel) contre H₁ : λ < 0. Distribution non standard
(fonctionnelle de mouvements browniens, dépend de k et des déterministes).
PSS 2001 reprennent ce test comme second pilier de leur cadre (t-bounds),
et SMG 2019 (spec 15) en font l'un des trois tests obligatoires.

## 2. Implémentation
Presque tout existe déjà via spec 10 ; ce que cette spec ajoute :

1. **Unilatéralité** : test à gauche uniquement — la décision doit exiger
   λ̂ < 0 ET t < CV. Cas piège à gérer : t « significatif » avec λ̂ > 0
   (explosif) → statut `no_cointegration` + warning dédié (erreur
   d'interprétation fréquente dans la littérature appliquée, la
   bibliothèque doit l'empêcher).
2. Tables de CV du t : intégrées au module critical_values (bornes PSS,
   puis surfaces spec 13).
3. **Interprétation jointe F + t** (préparation de la logique SMG) :
   exposer `decision_joint` documenté : la cointégration exige la
   concordance des deux tests ; F rejette mais pas t → suspicion de
   dégénérescence de type 1 (γ seuls significatifs) — message renvoyant
   vers spec 15.
4. Sortie de la vitesse d'ajustement avec IC (attention : l'IC standard
   sur λ n'est valide qu'*sous cointégration établie* — le documenter et
   ne l'afficher qu'après décision positive).

## 3. Tests
1. Unilatéralité : DGP explosif (λ > 0) → jamais « cointegration ».
2. Taille/puissance MC du t seul aux deux bornes, cas III et V.
3. Concordance de t avec R `ARDL::bounds_t_test` et statsmodels sur les
   données danoises (1e-6).

## 4. Liens
Spec 10 (intégration au workflow) ; spec 15 (le t devient l'un des 3 tests
du cadre augmenté) ; spec 14/16 (versions bootstrap du t).
