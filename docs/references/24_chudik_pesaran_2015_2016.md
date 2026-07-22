# Spec 24 — Chudik & Pesaran (2015, 2016) : CS-ARDL et CS-DL (dépendance transversale)

## Références
Chudik & Pesaran (2015), *Journal of Econometrics*, 188(2), 393–420
(CCE dynamique) ; Chudik, Mohaddes, Pesaran & Raissi (2016), *Advances in
Econometrics* 36 (CS-DL). Clés : `chudik2015csardl`, `chudik2016csdl`.
Branche : **9. Panel** · Module : `pyardl.panel` · Priorité : v0.8 —
absent de R comme de Python (référence : Stata xtdcce2) → seconde grande
zone vierge après le noyau.

## 1. Problème et solutions
MG/PMG (22-23) supposent l'indépendance transversale. Avec des chocs
communs (facteurs globaux : cycle mondial, prix des matières premières),
les estimateurs sont biaisés. L'approche CCE : approximer les facteurs
inobservés par les **moyennes transversales** des variables ; en panel
dynamique, il faut en outre des **retards de ces moyennes** (l'y retardé
rend l'approximation statique insuffisante).

1. **CS-ARDL** : ARDL individuel augmenté des moyennes transversales
   courantes et retardées de (y, x) ; long terme reconstruit à partir des
   coefficients de court terme (comme en spec 03).
2. **CS-DL** : régression directe de y sur x courant, ses différences
   retardées, et les moyennes transversales — estime le long terme sans
   passer par la dynamique complète (robuste à la mauvaise spécification
   des retards, valide sous conditions sur λ).

## 2. Implémentation

1. **Moyennes transversales** : helper `cross_section_averages(df, vars,
   lags)` — moyennes par période (pondérations égales par défaut, option
   pondérée), puis retards ; gestion du panel non cylindré (moyenne sur
   les présents, avertir si composition très variable).
2. **CS-ARDL (MG)** :
   a. par individu : ARDL(p, q) + z̄_{t}, z̄_{t-1}, ..., z̄_{t-p_z} où
      z̄ = (ȳ, x̄) ; règle par défaut p_z = floor(T^{1/3}) (celle de la
      pratique), configurable ;
   b. θ̂_i à partir des coefficients dynamiques individuels (formules
      spec 03) ; agrégation MG (spec 22 : moyenne + variance
      inter-individus) ;
   c. exposer aussi la vitesse d'ajustement moyenne.
3. **CS-DL (MG)** : par individu, OLS de y_{it} sur x_{it}, Δx retardés
   (troncature p_x), et moyennes transversales ; θ̂_i = coefficient de
   x_{it} directement ; agrégation MG.
4. **Diagnostics spécifiques** :
   a. test CD de Pesaran (dépendance transversale des résidus) — à
      implémenter comme fonction générale `cd_test(residuals_panel)`
      (corrélations de paires standardisées) : sert avant (motiver
      l'augmentation) et après (vérifier qu'elle a suffi) ;
   b. exposant de dépendance forte/faible : hors périmètre v1 (le noter).
5. **Ordonnancement des colonnes et colinéarité** : avec beaucoup de
   moyennes retardées et T modeste, colinéarité fréquente → détection de
   rang avec retrait automatique journalisé (règle déterministe,
   documentée — sinon résultats non reproductibles entre plateformes).

## 3. API
```python
res = pyardl.panel.CSARDL(df, y=..., X=..., id=..., time=...,
        order=(1,1), cs_lags="auto").fit()
res = pyardl.panel.CSDL(df, ..., trunc_lags="auto").fit()
res.longrun; res.cd_test(); res.summary()
```

## 4. Tests
1. DGP à facteur commun (chargements hétérogènes) : MG naïf biaisé,
   CS-ARDL et CS-DL débiaisés (MC N=40, T=80 — tableau pour la doc).
2. CD : détecte le facteur avant augmentation, silence après.
3. Panel non cylindré, colinéarité provoquée → comportement déterministe.
4. Validation externe : Stata **xtdcce2** sur données publiques (l'exemple
   de sa documentation) → θ̂ CS-ARDL et CS-DL à 1e-3 ; c'est LA référence,
   aucune contrepartie R complète — documenter ce statut dans la doc
   (argument fort de l'article logiciel).

## 5. Liens
Étend 22-23 (mêmes conteneurs et agrégation) ; cd_test sert aussi aux
diagnostics de 23 ; la logique « facteur ≈ moyennes transversales »
est le pont conceptuel vers ton travail FAVAR/nowcasting.
