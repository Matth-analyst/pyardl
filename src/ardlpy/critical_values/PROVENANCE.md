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

**Note de transcription** : la branche « petit échantillon » de dynamac
(obs <= 35, cas I) contient une coquille manifeste (10 %, I(0), k = 10 :
`11.60` au lieu de `1.60`) ; la branche asymptotique utilisée ici porte
la valeur correcte `1.60` (cohérente avec la monotonie en k et avec
K&S : 1.598).

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
