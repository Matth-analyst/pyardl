# Écarts documentés par rapport aux specs

Chaque entrée : spec concernée, écart, justification, commit.

## Spec 10 §4.1 — format des tables de CV : module Python au lieu de .npz

**Écart** : la spec demande des « fichiers .npz versionnés + provenance
documentée » ; les tables PSS 2001 sont encodées en littéraux Python
dans `src/ardlpy/critical_values/pss2001.py` (+ PROVENANCE.md).

**Justification** : un .npz est binaire — non relisible en revue de code,
non diffable, et impossible à auditer ligne à ligne contre l'article
source, ce qui contredit l'esprit de la règle de provenance (chaque
valeur critique doit être vérifiable). Les littéraux Python offrent la
même performance à cette taille (704 nombres) et rendent chaque valeur
auditable dans le diff git. Le format .npz reste pertinent pour les
grilles volumineuses du moteur de simulation (spec 12) et sera adopté
là-bas.

**Commit** : spec 10 (bounds test PSS 2001).
