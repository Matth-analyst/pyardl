# `ardlpy.core.transforms` — algèbre ARDL ↔ ECM

Référence : Sargan, J. D. (1964). *Wages and Prices in the United
Kingdom: A Study in Econometric Methodology* — spec
[03_sargan_1964](../references/03_sargan_1964.md).

## Pourquoi ce module existe

Sargan (1964) introduit, dans une équation de salaires, la première
utilisation opérationnelle d'un mécanisme de correction d'erreur : la
croissance des salaires réagit à l'écart de *niveau* entre salaire réel
et productivité à la période précédente. L'idée structurante de toute la
bibliothèque en découle :

> une dynamique en différences + un terme de rappel en niveaux = un
> modèle qui distingue court terme et long terme.

Ce module fournit la reparamétrisation **exacte** (aucune hypothèse
statistique supplémentaire) entre la représentation ARDL(p, q₁,...,q_k)
et sa forme à correction d'erreur (ECM) équivalente.

## Théorie

Pour un ARDL(p, q₁,...,q_k) :

```
y_t = α + δt + Σ_i φ_i y_{t-i} + Σ_j Σ_i β_{j,i} x_{j,t-i} + ε_t
```

la forme ECM exacte est :

```
Δy_t = α + δt + λ y_{t-1} + Σ_j γ_j x_{j,t-1}
       + Σ_i ψ_i Δy_{t-i} + Σ_j Σ_i ω_{j,i} Δx_{j,t-i} + ε_t
```

avec :

| Quantité | Formule | Interprétation |
|---|---|---|
| `lam` (λ) | −(1 − Σφ_i) | vitesse d'ajustement (unilatéral < 0) |
| `gamma_j` (γ_j) | Σ_i β_{j,i} | coefficient de niveau de x_j |
| `theta_j` (θ_j) | γ_j / (−λ) | coefficient de long terme de x_j |
| `psi_i` (ψ_i) | −Σ_{m=i+1}^{p} φ_m | dynamique de court terme de y |
| `omega_{j,0}` | β_{j,0} | effet contemporain de Δx_j |
| `omega_{j,i}`, i≥1 | −Σ_{m=i+1}^{q_j} β_{j,m} | dynamique de court terme de x_j |
| `half_life` | ln(0.5) / ln(1+λ) | demi-vie du retour à l'équilibre |

Ces formules sont vérifiées par équivalence de régressions OLS (résidus
identiques à 1e-10) et par concordance avec
`statsmodels.tsa.ardl.UECM.from_ardl` — voir
`tests/unit/core/test_transforms_equivalence.py` et
`tests/unit/core/test_transforms_statsmodels.py`.

## Avertissement méthodologique : cas limite q_j = 0

Quand un régresseur x_j n'a aucun retard propre (q_j = 0), il n'existe
pas de terme Δx_{j,t} distinct : γ_j multiplie alors x_{j,t}
(contemporain) et non x_{j,t-1}, faute de quoi la régression ECM
obtiendrait un degré de liberté de plus que l'ARDL d'origine (résidus
non identiques). Ce point, non explicité par la spec d'origine (note de
révision reportée dans la spec 03 §2.2), est dérivé et justifié dans
[`docs/QUESTIONS.md`](../QUESTIONS.md).

Comparaison avec les implémentations de référence :

| Implémentation | Comportement pour q_j = 0 |
|---|---|
| `statsmodels.tsa.ardl.UECM` | **Refuse** à la construction (`ValueError: All included exog variables must have a lag length >= 1`) |
| Stata `ardl` | **Accepte** : x_{j,t} contemporain entre dans la partie de niveau de l'EC — même convention qu'ardlpy |
| `ardlpy` | **Accepte** : ω_j vide, γ_j sur x_{j,t} contemporain |

Noter que x_{j,t} reste I(1) : l'identité exacte x_{j,t} = x_{j,t-1} +
Δx_{j,t} (écart stationnaire) garantit que l'asymptotique des tests sur
la partie de niveau est inchangée — le régresseur de niveau est toujours
intégré d'ordre 1, seule sa datation diffère d'une période.

## Cas dégénérés

- `abs(lambda) < tol` (pas de force de rappel) : `longrun_coefs` renvoie
  des `NaN` et émet `ArdlpyMethodologyWarning` (sous-classe
  `DegenerateCaseWarning`). La classification complète de ces cas relève
  des specs 14-15 (bootstrap, cadre à 3 tests).
- `lambda` hors de `]-1, 0[` : `half_life` n'est pas défini (pas de
  convergence géométrique), `NaN` + `DegenerateCaseWarning`.

## Exemple

```python
import numpy as np
from ardlpy.core.transforms import ARDLParams, ardl_to_ecm, longrun_coefs

params = ARDLParams(
    p=1, q=(1,),
    phi=np.array([0.5]),
    beta=(np.array([0.3, 0.2]),),
)
ecm = ardl_to_ecm(params)
print(ecm.lam)      # -0.5
print(ecm.gamma)    # [0.5]
print(longrun_coefs(params))  # theta = 0.5 / 0.5 = 1.0
```

## API

- `ARDLParams`, `ECMParams` : conteneurs immuables des deux
  représentations.
- `ardl_to_ecm`, `ecm_to_ardl` : reparamétrisations exactes (système
  triangulaire, sommes cumulées).
- `longrun_coefs`, `longrun_covariance` : θ_j et sa covariance (méthode
  delta, gradient analytique).
- `speed_of_adjustment`, `half_life` : λ et la demi-vie du retour à
  l'équilibre.
