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
3. **t, colonnes I(1), k >= 1** : PAS de seconde source accessible à ce
   jour — recoupement par simulation interne requis (spec 12).
   **Dette documentée dans docs/QUESTIONS.md.**

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

**Couverture et limitations (exceptions explicites dans le code)** :

- Seuils : 10 %, 5 %, 1 %. Le seuil **2.5 %** (publié dans PSS 2001)
  n'est pas dans la transcription dynamac ni recoupable via K&S
  (percentiles 90/95/99/99.9) → différé à la spec 12 (simulation).
- k = 0..10 (limite des tables PSS) ; au-delà → exception renvoyant
  vers les specs 12/13.
- t : cas I, III, V uniquement (PSS 2001 ne publie pas de bornes t pour
  les cas II et IV, où le t_BDM n'est pas applicable) → exception
  explicite.
- Les tables PSS supposent T = 1000 (asymptotique) : pour les petits
  échantillons, voir Narayan 2005 (spec 12) et les surfaces de réponse
  ajustées en T (spec 13).
