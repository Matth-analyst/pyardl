# Spec 03 §6.2 — validation externe : R::ARDL::uecm() sur le jeu de
# données danois du package (Danish, Johansen & Juselius 1990), tel que
# documenté par la spec (Natsiopoulos & Tzeremes, package "ARDL").
#
# À exécuter manuellement par un humain (R + package ARDL installés) ;
# ardlpy ne doit JAMAIS fabriquer les valeurs de référence produites ici
#.
#
# Sortie attendue : coller les valeurs numériques (lambda, gamma_j,
# theta_j, ecart-types) dans tests/replication/expected/spec03.json
# (avec provenance et tolerance), puis retirer le marqueur
# @pytest.mark.external du test correspondant dans
# tests/replication/test_spec03.py.

# install.packages("ARDL")  # si nécessaire
library(ARDL)

data(denmark)

# ARDL(p, q) choisi arbitrairement ici (p=2, q=2) ; ajuster si la spec
# ou une publication de référence impose un ordre précis à répliquer.
ardl_model <- ardl(LRM ~ LRY + IBO + IDE, data = denmark, order = c(2, 2, 2, 2))
summary(ardl_model)

uecm_model <- uecm(ardl_model)
summary(uecm_model)

cat("Coefficients ECM (lambda, gamma_j) :\n")
print(coef(uecm_model))

cat("Multiplicateurs de long terme (theta_j) :\n")
print(multipliers(ardl_model))
