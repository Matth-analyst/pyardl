# Spec 09 — Pesaran & Shin (1998) : l'ARDL comme approche de la cointégration

## Référence
Pesaran & Shin (1998), chapitre du volume du centenaire Ragnar Frisch,
Cambridge University Press. Clé : `pesaran1998ardl` · Branche : **3. Noyau**.
Module : `ardlpy.core` (propriétés d'inférence) · Priorité : v0.1-v0.2.

## 1. Apport théorique (ce que la doc doit expliquer)
Résultats qui fondent l'usage de l'ARDL sur séries non stationnaires :
1. L'estimation OLS d'un ARDL correctement spécifié (assez de retards pour
   blanchir les erreurs) donne des estimateurs des coefficients de **long
   terme super-convergents** (taux T), que les régresseurs soient I(0) ou
   I(1), et l'inférence sur θ via la méthode delta est asymptotiquement
   valide (normale) — contrairement à l'OLS statique de spec 06.
2. Le choix des retards par critère d'information (préférence historique
   des auteurs pour le SBC/BIC en petit échantillon) suffit à traiter
   l'endogénéité et l'autocorrélation résiduelle.
3. C'est LA justification de `results.longrun` + se delta (specs 03, 05).

## 2. Implémentations découlant directement de l'article

### 2.1 Inférence de long terme « à deux vitesses »
Documenter et exposer deux jeux de se pour θ :
- delta method standard (déjà spec 03) — valide asymptotiquement ;
- option bootstrap (percentile, réutilisera l'infrastructure spec 14) —
  recommandée en petit échantillon. Paramètre `longrun_inference=
  "delta"|"bootstrap"`.

### 2.2 Garde-fous de spécification
Après tout `.fit()` du cœur : exécuter Ljung-Box automatiquement et si
p < 0.05, warning « erreurs autocorrélées : l'inférence de long terme
n'est pas fiable, augmenter p/q ou revoir la spécification » — c'est la
condition de validité centrale de PS98, elle doit être surveillée par
défaut, pas en option.

### 2.3 Étude Monte Carlo de réplication (docs/validation)
Reproduire l'esprit des expériences de l'article : DGP ARDL(1,1) avec x
I(1), grille de T ∈ {50, 100, 250, 500} → montrer (a) la super-convergence
de θ̂ (variance ∝ 1/T²), (b) la couverture des IC delta vs bootstrap.
Livrable : notebook `validation/ps98_montecarlo.ipynb` + page de doc.

## 3. Tests
1. Super-convergence : pente de log-var(θ̂) sur log T ≈ −2 (MC).
2. Couverture delta ∈ [90, 97] % à T=100 sur DGP de référence ; bootstrap
   ≥ delta en petit T.
3. Le warning §2.2 se déclenche sur DGP à erreurs AR(1) volontairement
   sous-spécifié, pas sur le DGP propre.

## 4. Liens
Fonde l'inférence de `results.longrun` (03/05). Prépare spec 10 : PS98
traite l'estimation, PSS 2001 traite le TEST d'existence de la relation.
L'option bootstrap anticipe l'infrastructure des specs 14-16.
