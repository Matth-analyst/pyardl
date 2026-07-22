# Spec 18 — Cho, Kim & Shin (2015) : le quantile ARDL (QARDL)

## Référence
Cho, Kim & Shin (2015), *Journal of Econometrics*, 188(1), 281–300.
Clé : `cho2015qardl` · Branche : **7** · Module : `pyardl.qardl` ·
Priorité : v0.6 (après le noyau et NARDL) — quasi absent hors GAUSS.

## 1. Le modèle
ARDL estimé par régression quantile : à chaque quantile τ ∈ (0,1),

Q_{Δy_t}(τ | ℱ_{t-1}) = det(τ) + λ(τ) y_{t-1} + Σ γ_j(τ) x_{j,t-1}
                        + Σ ψ_i(τ) Δy_{t-i} + Σ ω_{j,i}(τ) Δx_{j,t-i}

Tous les paramètres — dont la vitesse d'ajustement λ(τ) et les
coefficients de long terme θ_j(τ) = −γ_j(τ)/λ(τ) — deviennent des
**fonctions du quantile** : la relation de long terme peut n'exister que
dans les queues, l'ajustement être plus rapide en bas de distribution, etc.

## 2. Implémentation

1. **Estimation par τ** : régression quantile (réutiliser
   `statsmodels.regression.quantile_regression.QuantReg` — pas de
   réimplémentation du LP) sur l'UECM, pour une grille τ (défaut
   {0.05, 0.1, ..., 0.95}).
2. **Inférence** : covariance par méthode des noyaux (sparsity, bande de
   Hall-Sheather via statsmodels) par τ ; pour l'inférence **jointe entre
   quantiles** (nécessaire aux tests ci-dessous), covariance croisée des
   estimateurs à τ ≠ τ' — implémenter la formule sandwich inter-quantiles
   (bloc essentiel, absent de statsmodels) ; alternative robuste par
   défaut : bootstrap par blocs (moving block, longueur ~ T^{1/3}) qui
   donne directement la loi jointe.
3. **Coefficients de long terme quantile** : θ_j(τ) par delta method à
   chaque τ (helper spec 01) ; garde-fou λ(τ) ≈ 0 → NaN + warning
   (dégénérescence locale au quantile).
4. **Tests à implémenter** :
   a. constance sur τ : H₀ : θ_j(τ₁) = ... = θ_j(τ_m) (Wald joint —
      c'est le test signature du QARDL : la relation de LT est-elle
      homogène sur la distribution ?) ; idem pour λ(τ) ;
   b. cointégration au quantile : t sur λ(τ) — CV non standards →
      fournir par bootstrap (moteur specs 14/16 adapté : régénération
      sous H₀ puis régression quantile) ;
   c. symétrie inter-quantiles : θ(τ) = θ(1−τ).
5. **QNARDL** (jonction avec spec 17) : appliquer la décomposition en
   sommes partielles puis QARDL → θ⁺(τ), θ⁻(τ) ; l'API doit composer
   naturellement : `QARDL(..., asym=[...])`.
6. **Sorties graphiques** : `plot_coefficients()` — θ_j(τ) et λ(τ) en
   fonction de τ avec bandes (le graphique canonique de cette littérature).

## 3. API
```python
res = pyardl.QARDL(y, X, order=..., taus=np.arange(.05,.96,.05),
                   asym=None, case=3).fit(inference="mbb", B=999)
res.longrun(tau=None)      # DataFrame θ_j(τ)
res.wald_constancy("x1"); res.quantile_cointegration_test(tau=0.5)
res.plot_coefficients()
```

## 4. Tests
1. DGP homogène (mêmes coefficients à tous τ) : θ̂(τ) plat, test de
   constance ~ taille nominale ; DGP à hétéroscédasticité conditionnelle
   dépendant de x : θ(τ) pentu, test puissant.
2. τ = 0.5 ≈ estimation LAD ; grille agrégée vs OLS sur DGP gaussien
   symétrique (cohérence qualitative).
3. Bootstrap par blocs : couverture des bandes sur DGP dépendant (MC).
4. Validation externe : code GAUSS de référence de Cho (résultats publiés
   de leur application si reconstructibles) ; sinon concordance avec le
   package R qui implémente QARDL sur données communes — à défaut,
   validation exclusivement par Monte Carlo documenté (l'assumer dans la
   doc : c'est un point où notre implémentation peut devenir LA référence
   open source).

## 5. Liens
Compose avec 17 (QNARDL) ; consomme le bootstrap 14/16 ; complète la
gamme « la relation dépend de l'état » avec Fourier (19-21 : dépendance
au temps) — à présenter ensemble dans la doc.
