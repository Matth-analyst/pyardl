# Spec 25 — Jordan & Philips (2018) : simulations dynamiques d'ARDL (dynamac)

## Référence
Jordan & Philips (2018), *The Stata Journal*, 18(4), 902–923 (commande
dynardl + pssbounds ; package R dynamac). Clé : `jordan2018dynamac`.
Branche : **10** · Module : `ardlpy.simulate` (couche interprétation) ·
Priorité : v0.5 — forte valeur d'usage (science politique, papiers
appliqués), coût faible (compose l'existant).

## 1. Apport
Les coefficients d'un ARDL/ECM sont difficiles à interpréter directement
(effets répartis entre niveaux et différences). L'approche : **simuler
stochastiquement la trajectoire de y en réponse à un choc contrefactuel
sur un régresseur**, avec bandes de confiance issues de tirages dans la
distribution des paramètres estimés — transformer les tableaux de
coefficients en graphiques de réponse lisibles.

## 2. Algorithme

Entrées : résultats d'un ARDL/UECM/NARDL estimé (specs 05/10/17), variable
choquée, type de choc (step permanent — défaut — ou impulsion temporaire),
amplitude (défaut : +1 écart-type du régresseur), date du choc t₀ dans la
fenêtre de simulation, horizon H, nombre de tirages R (défaut 1000), seed.

1. **Point de départ** : toutes les x fixées à leur moyenne (option :
   dernières valeurs observées, ou scénario utilisateur complet) ;
   y initialisé à sa valeur d'équilibre implicite ŷ* = θ̂'x̄ (+ det).
2. **Tirages de paramètres** : b_r ~ N(θ̂_full, V̂_full) (vecteur complet
   de l'UECM, r = 1..R) ; option bootstrap des paramètres (réutiliser les
   réplications de spec 14/16 si disponibles — cohérence d'inférence).
3. **Récursion** : pour chaque r, simuler y_t sur t = 1..H via la forme
   ARDL (ecm_to_ardl, spec 03), x suivant le scénario (choc en t₀) ;
   deux variantes exposées :
   a. `stochastic=False` : trajectoires déterministes par tirage
      (incertitude des paramètres seule) ;
   b. `stochastic=True` : + innovations ε_t ~ N(0, σ̂²) (incertitude de
      prévision totale).
4. **Synthèse** : par t, moyenne et quantiles (75/90/95 %) des R
   trajectoires ; sorties absolues et en écart au contrefactuel sans choc
   (différence appariée par tirage — réduit la variance, à faire par
   défaut).
5. **Graphique signature** : éventail de bandes (spike plot / area plot),
   ligne du contrefactuel, marqueur du choc.

## 3. API
```python
sim = res.dynardl_simulate(shock="oil", shock_type="step", size="1sd",
        t0=10, horizon=50, R=1000, stochastic=False, seed=...)
sim.summary_df        # t, mean, q05..q95 (niveau et écart)
sim.plot(bands=(75, 90, 95))
```
Compatibilité NARDL : choc sur x⁺ ou x⁻ séparément (compose avec les
multiplicateurs de spec 17 — vérifier la cohérence des deux sorties).

## 4. Tests
1. Convergence d'équilibre : sans choc, trajectoire moyenne constante à
   ŷ* (1e-6) ; avec step sur x_j, convergence vers ŷ* + θ̂_j·Δx (1e-3 à
   H grand) — test fondamental reliant simulation et algèbre (spec 03).
2. Impulsion : retour à l'équilibre initial ; vitesse cohérente avec
   half_life (spec 03).
3. Largeur des bandes : croît avec stochastic=True ; couverture MC
   (le vrai y simulé sous le DGP tombe dans la bande 95 % ~ 95 %).
4. Validation externe : R dynamac (dynardl) sur son exemple de
   documentation — trajectoires moyennes superposables (tolérance MC),
   mêmes équilibres exacts.

## 5. Liens
Compose 03 (algèbre), 05/10/17 (modèles), 14/16 (option bootstrap) ;
partage l'infrastructure des multiplicateurs (17 §2.5) — factoriser la
récursion en un simulateur unique interne.
