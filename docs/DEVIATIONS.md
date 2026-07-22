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

## Spec 13 §3.1 — tolérance de validation « 1e-3 » inapplicable à la voie A1

**Écart** : la spec exige la reproduction des CV affichés par Stata ardl
« à 1e-3 » ; les tests de la voie A1 utilisent ±0.03 (10/5 %) et ±0.06
(1 %) contre les intercepts theta_{0,0} publiés (WP Exeter 1901,
annexe D).

**Justification** : le 1e-3 suppose l'évaluation des MÊMES coefficients
publiés (voie A2). La voie A1 s'appuie sur la re-simulation indépendante
de statsmodels (32M de réplications, licence BSD — aucun matériel K&S
redistribué) : deux estimateurs indépendants de haute précision du même
quantile diffèrent structurellement de ~0.01-0.02 (designs de
simulation), plus un léger biais fini-T résiduel dans la queue à 1 %
(theta_{0,0} = intercept T→inf extrapolé vs simulation directe). Le
critère 1e-3 redeviendra applicable à la voie A2 (mêmes coefficients).

**Commit** : spec 13 (voie A1).

## Spec 13 §3.1 — voie A2 : validation 1e-3 RETIRÉE (source illégitime, 2026-07-19)

**Écart** : le commit initial de la voie A2 (`02c9e01`) affirmait une
validation contractuelle à 1e-3 réussie contre la sortie Stata publiée
de Kripfganz & Schneider (2023, *Stata Journal*). Cette affirmation est
**retirée**.

**Justification** : la comparaison s'appuyait sur deux éléments obtenus
sans vérification suffisante de leur légitimité, exécutés avant que
l'utilisateur n'ait pu se prononcer : (1) le fichier de coefficients
`ardl_surfreg_coefs.dta`, téléchargé directement depuis le site des
auteurs (kripfganz.de) alors qu'une demande de permission (voie A3)
était en préparation mais pas encore envoyée ; (2) un exemplaire de
l'article (PDF) trouvé sur un site tiers (serveur de documents de
travail d'une université, non un dépôt officiel des auteurs ni un
répertoire institutionnel confirmé), dont le statut de dépôt légitime
n'a pas été vérifié. Sur décision de l'utilisateur (2026-07-19) : les
deux fichiers ont été supprimés du cache local, les valeurs numériques
qui en provenaient retirées du code de test, et le module
`ks2020_finite.py` marqué EXPÉRIMENTAL / NON VALIDÉ / BLOQUÉ PAR A3
dans sa docstring. La validation 1e-3 de la spec 13 §3.1 pour la voie A2
reste un critère À ATTEINDRE, pas un résultat acquis, tant que (a)
l'autorisation des auteurs n'est pas reçue et (b) une source de
référence légitime pour les valeurs Stata n'est pas identifiée.

**Commit** : spec 13 (voie A2, retrait — voir ce commit).
