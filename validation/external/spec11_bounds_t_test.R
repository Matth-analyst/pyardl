# Spec 11 §3.3 — validation externe : R::ARDL::bounds_t_test sur les
# données danoises. À exécuter manuellement (R + package ARDL) ; coller
# les sorties dans tests/replication/expected/spec11.json (valeurs +
# provenance + tolérance 1e-6 sur la statistique t). NE JAMAIS inventer
# les valeurs attendues.

# install.packages("ARDL")
library(ARDL)

data(denmark)

# CORRECTION (journal, 2026-07-07) : le cas 5 exige la tendance dans le
# modèle sous-jacent (API du package) -> deux modèles : sans tendance
# pour le cas 3, avec trend(LRM) pour le cas 5.
ardl_model <- ardl(LRM ~ LRY + IBO + IDE, data = denmark, order = c(3, 1, 3, 2))
uecm_model <- uecm(ardl_model)

cat("=== bounds_t_test, case 3 (constante non restreinte) ===\n")
bt3 <- bounds_t_test(ardl_model, case = 3)
print(bt3)

ardl_trend <- ardl(
  LRM ~ LRY + IBO + IDE + trend(LRM), data = denmark, order = c(3, 1, 3, 2)
)
cat("=== bounds_t_test, case 5 (constante + tendance) ===\n")
bt5 <- bounds_t_test(ardl_trend, case = 5)
print(bt5)

cat("\n=== RÉSUMÉ POUR spec11.json ===\n")
cat("t case 3 :", bt3$statistic, "\n")
cat("t case 5 :", bt5$statistic, "\n")
cat("Coefficient y.L1 de l'UECM (sans tendance) et se :\n")
s <- summary(uecm_model)
print(s$coefficients["L(LRM, 1)", ], digits = 12)
cat("Coefficient y.L1 de l'UECM (avec tendance) et se :\n")
s5 <- summary(uecm(ardl_trend))
print(s5$coefficients["L(LRM, 1)", ], digits = 12)
