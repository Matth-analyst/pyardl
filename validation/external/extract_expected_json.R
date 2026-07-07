# Génère tests/replication/expected/*.json à partir du package R ARDL
# (valeurs de référence des specs 03, 05, 10, 11 — précision %.15g).
# Exécuté le 2026-07-07 (R 4.6.1, ARDL 0.2.5, win32).
# Ce script est LA source des valeurs attendues : ne jamais les éditer
# à la main dans les JSON.

library(ARDL)

j <- function(x) {
  if (is.character(x)) return(paste0('"', x, '"'))
  sprintf("%.15g", x)
}
named_obj <- function(v) {
  paste0(
    "{", paste0('"', names(v), '": ', vapply(v, j, ""), collapse = ", "), "}"
  )
}

provenance <- paste0(
  '"provenance": {"r_version": "', R.version.string,
  '", "ardl_package_version": "', as.character(packageVersion("ARDL")),
  '", "execution_date": "', format(Sys.Date()),
  '", "script": "validation/external/extract_expected_json.R"}'
)

dir.create("tests/replication/expected", showWarnings = FALSE, recursive = TRUE)

# ============================ SPEC 03 =====================================
data(denmark)
m1 <- ardl(LRM ~ LRY + IBO + IDE, data = denmark, order = c(3, 1, 3, 2))
u1 <- uecm(m1)
mult1 <- multipliers(m1)
lam1 <- summary(u1)$coefficients["L(LRM, 1)", ]

m0 <- ardl(LRM ~ LRY + IBO + IDE, data = denmark, order = c(2, 2, 2, 0))
u0 <- uecm(m0)
mult0 <- multipliers(m0)
resid_gap_q0 <- max(abs(residuals(m0) - residuals(u0)))

theta1 <- setNames(mult1$Estimate, mult1$Term)
theta0 <- setNames(mult0$Estimate, mult0$Term)
ecm0 <- coef(u0)

spec03 <- paste0(
  "{\n", provenance, ",\n",
  '"tolerance": {"theta": 1e-6, "lambda": 1e-6, "note": "spec 03 §6.2"},\n',
  '"model_318_2": {"order": [3, 1, 3, 2],\n',
  '  "lambda": ', j(lam1[["Estimate"]]), ",\n",
  '  "lambda_se": ', j(lam1[["Std. Error"]]), ",\n",
  '  "theta": ', named_obj(theta1), "},\n",
  '"model_q_ide_0": {"order": [2, 2, 2, 0],\n',
  '  "note": "cas q_j=0 : niveau IDE contemporain dans l ECM (convention ardlpy confirmée)",\n',
  '  "resid_gap_ardl_uecm": ', j(resid_gap_q0), ",\n",
  '  "ecm_coefficients": ', named_obj(ecm0), ",\n",
  '  "theta": ', named_obj(theta0), "}\n}\n"
)
writeLines(spec03, "tests/replication/expected/spec03.json")

# ============================ SPEC 05 =====================================
m5 <- ardl(LRM ~ LRY + IBO + IDE, data = denmark, order = c(2, 2, 2, 2))
a_bic <- auto_ardl(LRM ~ LRY + IBO + IDE, data = denmark,
                   max_order = 5, selection = "BIC")
a_aic <- auto_ardl(LRM ~ LRY + IBO + IDE, data = denmark,
                   max_order = 5, selection = "AIC")

spec05 <- paste0(
  "{\n", provenance, ",\n",
  '"tolerance": {"coefficients": 1e-6, "note": "spec 05 §6.5"},\n',
  '"ardl_2222": {"order": [2, 2, 2, 2],\n',
  '  "coefficients": ', named_obj(coef(m5)), ",\n",
  '  "ssr": ', j(sum(residuals(m5)^2)), "},\n",
  '"auto_ardl_bic": {"order": [',
  paste(a_bic$best_order, collapse = ", "), "],\n",
  '  "coefficients": ', named_obj(coef(a_bic$best_model)), "},\n",
  '"auto_ardl_aic": {"order": [',
  paste(a_aic$best_order, collapse = ", "), "]}\n}\n"
)
writeLines(spec05, "tests/replication/expected/spec05.json")

# ============================ SPEC 10 =====================================
data(PSS2001)
mw <- ardl(w ~ Prod + UR + Wedge + Union + trend(w) | D7475 + D7579,
           data = PSS2001, order = c(6, 0, 5, 4, 5))
uw <- uecm(mw)
bf4 <- bounds_f_test(mw, case = 4)
bf5 <- bounds_f_test(mw, case = 5)
bt5w <- bounds_t_test(mw, case = 5)

spec10 <- paste0(
  "{\n", provenance, ",\n",
  '"tolerance": {"f_stat": 1e-4, "t_stat": 1e-4, "note": "spec 10 §6 : F et t identiques à 1e-4"},\n',
  '"dataset": "PSS2001 (salaires réels UK, package ARDL ; réplication Natsiopoulos & Tzeremes 2022 JAE)",\n',
  '"model": {"formula": "w ~ Prod + UR + Wedge + Union + trend | D7475 + D7579",\n',
  '  "order": [6, 0, 5, 4, 5]},\n',
  '"f_case4": ', j(unname(bf4$statistic)), ",\n",
  '"f_case5": ', j(unname(bf5$statistic)), ",\n",
  '"t_case5": ', j(unname(bt5w$statistic)), ",\n",
  '"uecm_coefficients": ', named_obj(coef(uw)), ",\n",
  '"uecm_ssr": ', j(sum(residuals(uw)^2)), "\n}\n"
)
writeLines(spec10, "tests/replication/expected/spec10_pss2001.json")

# ============================ SPEC 11 =====================================
bt3 <- bounds_t_test(m1, case = 3)
m1t <- ardl(LRM ~ LRY + IBO + IDE + trend(LRM), data = denmark,
            order = c(3, 1, 3, 2))
bt5d <- bounds_t_test(m1t, case = 5)

spec11 <- paste0(
  "{\n", provenance, ",\n",
  '"tolerance": {"t_stat": 1e-6, "note": "spec 11 §3.3"},\n',
  '"order": [3, 1, 3, 2],\n',
  '"t_case3": ', j(unname(bt3$statistic)), ",\n",
  '"t_case5_with_trend": ', j(unname(bt5d$statistic)), "\n}\n"
)
writeLines(spec11, "tests/replication/expected/spec11.json")

cat("4 fichiers JSON écrits.\n")
