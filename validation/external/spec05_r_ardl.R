# Spec 05 §6.5 — validation externe : R::ARDL::ardl() et auto_ardl()
# sur les données danoises (Johansen & Juselius 1990, package "ARDL").
#
# À exécuter manuellement (R + package ARDL). NE JAMAIS inventer les
# valeurs produites ici.
#
# Attendu côté ardlpy :
# 1. ardl(LRM ~ LRY + IBO + IDE, order = c(2,2,2,2)) -> coefficients
#    identiques à 1e-6 avec ARDL(y, x, order=(2, 2)) d'ardlpy sur les
#    mêmes données (à intégrer dans src/ardlpy/datasets/, spec 04).
# 2. auto_ardl(..., max_order = 5, selection = "BIC") -> même ordre
#    sélectionné que ARDL.select_order(..., max_p=5, max_q=5, ic="bic").
#    ATTENTION : vérifier la politique d'échantillon commun d'auto_ardl
#    (documentation du package) ; si auto_ardl compare sur échantillons
#    maximaux propres, documenter l'écart éventuel de sélection dans
#    tests/replication/expected/spec05.json plutôt que d'aligner ardlpy
#    sur un comportement non comparable (la correction
#    statistique prime).
#
# Coller les sorties dans tests/replication/expected/spec05.json
# (valeurs + provenance + tolérance), puis compléter
# tests/replication/test_spec05.py.

# install.packages("ARDL")
library(ARDL)

data(denmark)

# --- 1. Ordres fixes -------------------------------------------------------
model_fixed <- ardl(LRM ~ LRY + IBO + IDE, data = denmark, order = c(2, 2, 2, 2))
cat("=== ardl(order = c(2,2,2,2)) ===\n")
print(summary(model_fixed))
cat("Coefficients (pour spec05.json) :\n")
print(coef(model_fixed), digits = 12)
cat("SSR :", sum(residuals(model_fixed)^2), "\n")
cat("AIC/BIC :", AIC(model_fixed), BIC(model_fixed), "\n")

# --- 2. Sélection automatique ---------------------------------------------
auto_bic <- auto_ardl(LRM ~ LRY + IBO + IDE, data = denmark,
                      max_order = 5, selection = "BIC")
cat("=== auto_ardl BIC ===\n")
cat("Ordre sélectionné :", auto_bic$best_order, "\n")
print(coef(auto_bic$best_model), digits = 12)

auto_aic <- auto_ardl(LRM ~ LRY + IBO + IDE, data = denmark,
                      max_order = 5, selection = "AIC")
cat("=== auto_ardl AIC ===\n")
cat("Ordre sélectionné :", auto_aic$best_order, "\n")
