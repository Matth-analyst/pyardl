# Spec 23 — Pesaran, Shin & Smith (1999) : le Pooled Mean Group (PMG)

## Référence
PSS (1999), *JASA*, 94(446), 621–634. DOI: 10.1080/01621459.1999.10474156.
Clé : `pesaran1999pmg` · Branche : **9. Panel** · Module : `pyardl.panel` ·
Priorité : v0.7 — l'estimateur panel ARDL le plus utilisé en pratique.

## 1. Le modèle
Compromis entre pooling total (DFE) et hétérogénéité totale (MG, spec 22) :
**les coefficients de long terme θ sont contraints égaux entre individus,
les dynamiques de court terme (λ_i, ψ_i, ω_i, σ²_i) restent libres**.

ECM par individu : Δy_{it} = λ_i (y_{i,t-1} − θ'x_{i,t-1}) 
                   + Σ ψ_{is} Δy_{i,t-s} + Σ ω'_{is} Δx_{i,t-s} + μ_i + ε_{it}

## 2. Estimation par maximum de vraisemblance concentré (l'algorithme clé)

1. Vraisemblance sous normalité, produit sur i (indépendance
   transversale — hypothèse à afficher, levée en spec 24).
2. **Concentration** : à θ fixé, chaque bloc individuel (λ_i, coefficients
   de CT, σ²_i) s'obtient par OLS de Δy_i sur [ξ_i(θ), ΔW_i] où
   ξ_{it}(θ) = y_{i,t-1} − θ'x_{i,t-1} — donc la maximisation ne porte
   que sur θ (dimension k) : boucle Newton/quasi-Newton sur θ avec, à
   chaque évaluation, N OLS individuelles.
3. **Itération de type back-fitting** (implémentation retenue, celle de
   xtpmg) : alterner (a) θ donné → OLS individuelles ; (b) dynamiques
   données → mise à jour de θ par l'équation de score agrégée
   (moindres carrés pondérés par λ_i/σ²_i empilés) ; jusqu'à convergence
   (tol 1e-8, max_iter 200, journal des itérations).
4. **Inférence** : V̂(θ̂) par l'information (hessienne du profil) ;
   V̂(λ̂_i) etc. par bloc. Point de départ : θ̂_MG (spec 22) —
   robustifie la convergence.
5. **Sorties** : θ̂ commun (se), λ̂_i par individu + moyenne de groupe,
   court terme moyenné à la MG.

## 3. Le test d'Hausman MG vs PMG (obligatoire)
H₀ : homogénéité du long terme (PMG convergent ET efficace) vs
H₁ : hétérogénéité (seul MG convergent).
H = (θ̂_MG − θ̂_PMG)' [V̂_MG − V̂_PMG]⁻¹ (θ̂_MG − θ̂_PMG) ~ χ²(k).
Gérer la non-définie-positivité de la différence de variances (fréquente) :
pseudo-inverse + warning, comme la pratique Stata. Exposer
`hausman(res_mg, res_pmg)` et l'inclure dans `summary()` du PMG.

## 4. DFE (troisième estimateur, pour complétude)
Effets fixes dynamiques : tout contraint sauf les intercepts μ_i —
implémentation directe par OLS sur données empilées avec dummies/within ;
utile pour le tableau comparatif MG/PMG/DFE standard des papiers.

## 5. API
```python
res = pyardl.panel.PMG(df, y="lnc", X=["lny","inf"], id="country",
                        time="year", order=(1,1), det="const").fit()
res.longrun; res.adjustment_i; res.hausman_vs_mg(); res.summary()
pyardl.panel.compare(df, ...) -> tableau MG/PMG/DFE + Hausman
```

## 6. Tests
1. DGP à θ commun, dynamiques hétérogènes : PMG retrouve θ (biais < 1 %,
   MC N=30, T=60) et est plus efficace que MG (variance MC) ; Hausman ~
   taille. DGP à θ hétérogènes : Hausman puissant, PMG biaisé (documenté).
2. Convergence : back-fitting vs optimiseur quasi-Newton → même θ̂ (1e-6).
3. **Validation externe (réplication phare)** : reproduire l'application
   consommation-OCDE de l'article via les résultats de Stata xtpmg
   (l'exemple canonique de sa documentation) → θ̂, λ̄ et Hausman
   concordants à 1e-3 ; croiser avec R ardlverse (estimator="pmg").

## 7. Liens
Réutilise le conteneur et la boucle individuelle (22) ; spec 24 lève
l'hypothèse d'indépendance transversale ; le ξ_i(θ) préfigure les termes
de correction communs des modèles à facteurs.
