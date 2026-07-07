# Spec 10 §6 — JALON DE PHASE 1 : réplication de l'application PSS 2001
# (équation de salaires réels UK) telle que répliquée par Natsiopoulos &
# Tzeremes (2022, J. Applied Econometrics) avec le package R ARDL.
#
# À exécuter manuellement (R + package ARDL >= 0.2.0). NE JAMAIS estimer
# les valeurs attendues : les
# sorties de ce script SONT les valeurs de référence, à coller dans
# tests/replication/expected/spec10_pss2001.json avec provenance.
#
# Le package ARDL embarque le jeu de données PSS2001 (données UK
# trimestrielles utilisées dans l'article : w, Prod, UR, Wedge, Union +
# dummies D7475, D7579). La spécification de référence est celle de
# l'article (ARDL(6, 0, 5, 4, 5) sur w ~ Prod + UR + Wedge + Union),
# répliquée par Natsiopoulos & Tzeremes.

# install.packages("ARDL")
library(ARDL)

data(PSS2001)
cat("=== Données PSS2001 (package ARDL) ===\n")
str(PSS2001)

# --- Modèle de l'article : ARDL(6, 0, 5, 4, 5), cas V (const + trend
# non restreints, cf. PSS 2001 section 6) et cas IV pour le F ----------
# CORRECTION (journal, 2026-07-07) : première version sans trend(w) ->
# erreur "Trying to impose case 4/5 ... doesn't include one or both of
# them" ; les cas IV/V exigent la tendance DANS le modèle sous-jacent
# (API du package : terme trend(w) dans la formule). L'application PSS
# 2001 inclut bien une tendance linéaire (section 6).
ardl_wages <- ardl(
  w ~ Prod + UR + Wedge + Union + trend(w) | D7475 + D7579,
  data = PSS2001, order = c(6, 0, 5, 4, 5)
)
cat("=== ARDL(6,0,5,4,5) ===\n")
print(summary(ardl_wages))

uecm_wages <- uecm(ardl_wages)
cat("=== UECM ===\n")
print(summary(uecm_wages))
cat("Verification residus ardl/uecm identiques (doit être ~0) :\n")
print(max(abs(residuals(ardl_wages) - residuals(uecm_wages))))

# --- Bounds tests : F (cas IV et V) et t (cas V) ----------------------
cat("=== bounds_f_test, case 4 (trend restreinte, k=4) ===\n")
bf4 <- bounds_f_test(ardl_wages, case = 4)
print(bf4)

cat("=== bounds_f_test, case 5 ===\n")
bf5 <- bounds_f_test(ardl_wages, case = 5)
print(bf5)

cat("=== bounds_t_test, case 5 ===\n")
bt5 <- bounds_t_test(ardl_wages, case = 5)
print(bt5)

# --- Valeurs à reporter dans spec10_pss2001.json ----------------------
cat("\n=== RÉSUMÉ POUR spec10_pss2001.json ===\n")
cat("F case 4 :", bf4$statistic, "\n")
cat("F case 5 :", bf5$statistic, "\n")
cat("t case 5 :", bt5$statistic, "\n")
cat("Coefficients UECM :\n")
print(coef(uecm_wages), digits = 12)
cat("SSR UECM :", sum(residuals(uecm_wages)^2), "\n")

# Export CSV du dataset pour intégration dans src/ardlpy/datasets/
# (données publiques de l'article, redistribuées par le package ARDL —
# vérifier la licence du package avant redistribution ; sinon documenter
# la source primaire ONS/Bank of England dans le loader).
write.csv(PSS2001, "pss2001_uk_wages.csv", row.names = FALSE)
cat("Dataset exporté vers pss2001_uk_wages.csv\n")
