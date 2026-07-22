# Spec 26 — Brown, Durbin & Evans (1975) : CUSUM et CUSUMSQ (stabilité)

## Référence
Brown, Durbin & Evans (1975), *JRSS B*, 37(2), 149–192.
DOI: 10.1111/j.2517-6161.1975.tb01532.x. Clé : `brown1975cusum`.
Branche : **11. Diagnostics** · Module : `pyardl.diagnostics` ·
Priorité : v0.2 — standard obligatoire de tout papier ARDL appliqué.

## 1. Rôle
Tests de constance des paramètres fondés sur les **résidus récursifs** :
tout papier ARDL/NARDL publie les graphiques CUSUM et CUSUMSQ après le
bounds test. La bibliothèque doit les produire nativement sur l'UECM.

## 2. Algorithme

1. **Résidus récursifs** : pour t = k+1..T (k = nombre de régresseurs),
   estimer le modèle sur 1..t−1, prédire y_t, standardiser l'erreur de
   prédiction : w_t = (y_t − x_t'b_{t-1}) / sqrt(1 + x_t'(X'X)_{t-1}⁻¹x_t).
   Implémentation efficace par mise à jour récursive (formules de
   Sherman-Morrison / filtrage RLS) — pas de ré-estimation complète ;
   réutiliser `statsmodels.stats.diagnostic.recursive_olsresiduals`
   comme référence croisée mais implémenter la version maison vectorisée
   (statsmodels est lent sur longues séries et on veut la maîtrise pour
   le port Rust éventuel).
2. **CUSUM** : W_t = Σ_{s≤t} w_s / σ̂_w ; frontières de rejet à 5 % :
   droites ±a·sqrt(T−k) ± 2a(t−k)/sqrt(T−k) avec a = 0.948 (5 %),
   1.143 (1 %), 0.850 (10 %). Dérive de W hors bandes → instabilité
   de la moyenne des coefficients.
3. **CUSUMSQ** : S_t = Σ_{s≤t} w_s² / Σ_{s≤T} w_s² ; frontières
   S_t ∈ [E(S_t) ± c₀] où E(S_t) = (t−k)/(T−k) et c₀ tabulé (encoder la
   table par T−k et seuil ; interpolation documentée). Sensible aux
   changements de variance.
4. **Sorties** : statistique de dépassement maximal, dates de sortie de
   bande, booléens stables/instables par test, et `plot_cusum()` /
   `plot_cusumsq()` (les deux graphiques canoniques, bandes incluses).
5. Intégration : appelés automatiquement dans les diagnostics du bounds
   test (spec 10 §5) ; disponibles sur tout modèle du package.

## 3. Tests
1. DGP stable → sorties de bande ~ taille nominale (MC) ; rupture de
   coefficient à mi-échantillon → CUSUM détecte ; rupture de variance →
   CUSUMSQ détecte (et CUSUM peu — vérifier la spécialisation).
2. Résidus récursifs maison ≡ statsmodels (1e-10).
3. Concordance des graphiques/décisions avec EViews sur un exemple
   publié (valeurs des bornes vérifiées).

## 4. Liens
Consommé par 10 (diagnostics automatiques) et 17 (stabilité NARDL) ;
complémentaire de la branche Fourier (19-21) : CUSUM détecte l'instabilité,
Fourier la modélise — raconter ce lien dans la doc.
