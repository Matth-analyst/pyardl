# Spec 28 — Surveys et références de documentation (Nkoro & Uko 2016 ; Greenwood-Nimmo et al. 2013)

## Références
Nkoro & Uko (2016), *Journal of Statistical and Econometric Methods*,
5(4), 63–91 (survey méthodologique ARDL/bounds) ; Greenwood-Nimmo, Shin,
van Treeck & Yu (2013, working paper — asymétries multiples).
Clés : `nkoro2016ardl`, `greenwoodnimmo2013asymmetries` · Branche : **12**.
Module : documentation uniquement (pas de code nouveau) · Priorité :
continue (chaque release).

## 1. Rôle
Ces références ne génèrent pas d'implémentation propre : elles servent de
**check-list de complétude** et de source pour la documentation
pédagogique.

## 2. Livrables de documentation

1. **Vignette « workflow complet »** (structurée sur la séquence
   méthodologique standard que synthétise le survey) :
   pré-tests de racine unitaire (27) → vérification pas d'I(2), pas de
   cointégration entre x (07) → sélection d'ordre (05) → bounds test à
   3 tests + classification (10/11/15) → si cointégration : ECM, θ,
   vitesse (03) → diagnostics et stabilité (26) → robustesse
   (FMOLS/DOLS 08, bootstrap 14/16, NARDL 17, Fourier 19-21) →
   interprétation par simulation (25). Chaque étape : le pourquoi, le
   comment dans pyardl, les erreurs courantes.
2. **Page « erreurs fréquentes de la littérature appliquée »** (forte
   valeur pédagogique et de citation) : conclure sur le seul F global
   (ignorer les dégénérescences), IC sur λ hors cointégration établie,
   mauvaise correspondance des cas I-V entre logiciels, CV asymptotiques
   sur T=35, variables I(2) non détectées, t « significatif » avec λ̂ > 0.
   Montrer comment l'API rend chacune difficile à commettre.
3. **Asymétries multiples** (Greenwood-Nimmo et al.) : documenter
   l'extension à seuils multiples de la décomposition (spec 17 —
   partial_sums à plusieurs seuils est déjà prévu par le paramètre
   threshold ; ajouter l'exemple à seuils {c₁, c₂} dans la doc, code
   utilisateur, sans API dédiée en v1).
4. **Glossaire trilingue** (FR/EN + notation) : bounds, dégénérescence,
   correction d'erreur, etc. — cohérence de toute la doc.

## 3. Tests
Doctests : tous les blocs de code des vignettes s'exécutent en CI
(la doc est testée comme le code) ; la vignette workflow tourne sur le
dataset danois de bout en bout en < 60 s.

## 4. Liens
Clôt la boucle : la généalogie (specs 01-27) devient un manuel. Base de
l'article logiciel (JOSS) : la vignette workflow en est le squelette.
