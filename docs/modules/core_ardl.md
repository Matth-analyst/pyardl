# `ardlpy.core.ardl` — l'estimateur ARDL(p, q)

Référence : Hendry, Pagan & Sargan (1984), "Dynamic Specification",
*Handbook of Econometrics*, vol. 2, ch. 18 — spec
[05_hendry_pagan_sargan_1984](../references/05_hendry_pagan_sargan_1984.md).
Garde-fous d'inférence : Pesaran & Shin (1998) — spec
[09_pesaran_shin_1998](../references/09_pesaran_shin_1998.md).

## Pourquoi ce module existe

L'ARDL(p, q₁,...,q_k) est la forme mère dont dérivent, comme cas
particuliers ou reparamétrisations : le modèle statique, les
différences pures, Koyck, FDL/Almon, l'ECM et l'autorégressif pur. La
bibliothèque encode cette taxonomie : **un seul moteur d'estimation,
plusieurs vues** (`.to_ecm()`, `.longrun`, `.adjustment` — algèbre de la
spec 03, consommée sans conversion manuelle).

## Exemple complet

```python
import numpy as np, pandas as pd
from ardlpy.core.ardl import ARDL

# estimation à ordres fixes
res = ARDL(y, x, order=(2, {"rev": 1, "px": 2}), det="const").fit()
print(res.summary())
print(res.longrun)       # theta_j avec se (delta, PS98)
print(res.adjustment)    # lambda, se, demi-vie
ecm = res.to_ecm()       # vue ECM exacte (spec 03)

# sélection d'ordre (échantillon commun obligatoire)
sel = ARDL.select_order(y, x, max_p=4, max_q=4, ic="bic")
sel.top(5)               # robustesse : vérifier les spécifications proches
best = sel.best_model    # ré-estimé sur l'échantillon maximal

# réduction general-to-specific
g = ARDL.gets(y, x, max_p=4, max_q=4, alpha=0.05)
g.reduction_path         # chemin journalisé (transparence)
```

## Avertissements méthodologiques

- **Garde-fou d'autocorrélation (spec 09 §2.2)** : après chaque
  `.fit()`, un test de Ljung-Box est exécuté automatiquement ; si
  p < 0.05, un `ArdlpyMethodologyWarning` signale que l'inférence de
  long terme n'est pas fiable (condition de validité centrale de
  Pesaran-Shin 1998). Ce n'est pas une option.
- **Échantillon commun dans `select_order` (spec 05 §3.2)** : tous les
  candidats sont estimés sur t = max(max_p, max_q)+1..T ; comparer des
  IC calculés sur des échantillons différents n'a pas de sens. Le
  meilleur modèle est ensuite ré-estimé sur son échantillon maximal.
- **Stabilité dynamique (spec 05 §2.4)** : `.is_stable` vérifie que
  toutes les racines de 1 − Σφᵢ Lⁱ sont hors du cercle unité ; un
  warning est émis sinon (les quantités de long terme n'ont alors pas
  d'interprétation d'équilibre).
- **GETS (spec 05 §4)** : la réduction préserve la structure contiguë
  des retards (seul le retard terminal de chaque variable est candidat)
  — voir [`docs/QUESTIONS.md`](../QUESTIONS.md).

## Différences de conventions vs statsmodels

Les coefficients, se et résidus concordent à 1e-10 avec
`statsmodels.tsa.ardl.ARDL` (testé). En revanche `statsmodels` rapporte
`nobs = T − p` et calcule llf/IC dessus même quand max(qⱼ) > p ; ardlpy
utilise partout la taille de l'échantillon d'estimation réel
(T − hold_back), condition de comparabilité des IC dans `select_order`.
Les deux conventions coïncident dès que p ≥ max(qⱼ).

## API

- `ARDL(y, x, order, det, seasonal, fixed_regressors, hold_back)` —
  `seasonal=True` viendra avec la spec 04 (phase 2).
- `.fit(cov_type)` : `nonrobust`, `HC0`-`HC3`, `HAC` (Newey-West,
  `cov_kwds={"nlags": m}`) — concordance testée avec statsmodels OLS.
- `ARDL.select_order(y, x, max_p, max_q, ic, search)` — `search="grid"`
  ou `"per_variable"` (pour k > 3, l'espace de la grille explose).
- `ARDL.gets(y, x, max_p, max_q, alpha)` — chemin dans
  `.reduction_path`.
- `ARDLResults` : `params`, `bse`, `tvalues`, `pvalues`, `resid`,
  `llf`, `aic/bic/hqic`, `rsquared`, `is_stable`, `ar_roots`,
  `ardl_params` (conteneur spec 03 avec `cov_params`), `to_ecm()`,
  `longrun`, `adjustment`, `diagnostics()`, `summary()`.
