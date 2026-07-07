# Spec 22 — Pesaran & Smith (1995) : panels dynamiques hétérogènes et Mean Group

## Référence
Pesaran & Smith (1995), *Journal of Econometrics*, 68(1), 79–113.
DOI: 10.1016/0304-4076(94)01644-F. Clé : `pesaran1995mg`.
Branche : **9. Panel** · Module : `ardlpy.panel` · Priorité : v0.7.

## 1. Apport
Dans un panel dynamique à coefficients hétérogènes entre individus,
les estimateurs poolés classiques (effets fixes dynamiques, GMM) sont
non convergents même quand N et T → ∞ : forcer l'homogénéité des
dynamiques crée une autocorrélation résiduelle qui contamine les
coefficients. Estimateur convergent à grand T retenu ici :
**Mean Group (MG)** — estimer l'ARDL individu par individu, puis
moyenner les coefficients (et les θ de long terme).

Message architectural : le panel ARDL de la bibliothèque est une
**orchestration de N estimations séries temporelles** (specs 03/05)
+ une couche d'agrégation et d'inférence.

## 2. Implémentation

1. **Conteneur panel** : entrée DataFrame long (colonnes id, time, y, X) ;
   validation : panel non cylindré accepté (T_i variables), T_i min
   configurable (warning si T_i < 30 — le cadre exige T grand).
2. **MG** :
   a. boucle sur i : ARDL_i (ordre commun ou sélectionné par individu —
      les deux modes) → θ̂_i, λ̂_i ;
   b. agrégation : θ̂_MG = moyenne simple des θ̂_i ;
      V̂(θ̂_MG) = variance empirique inter-individus / N (l'inférence
      vient de la dispersion entre individus, pas des V̂_i — le
      documenter, c'est contre-intuitif) ;
   c. option moyennes pondérées et version robuste aux valeurs extrêmes
      (moyenne tronquée / médiane de groupe) pour petits N.
3. **Diagnostics** : distribution des θ̂_i (`plot_heterogeneity()`),
   part d'individus à λ̂_i ≥ 0 (non-ajustement), stockage des N objets
   résultats individuels accessibles (`res.individual["FR"]`).
4. Parallélisation : boucle sur i triviale (joblib ; backend Rust inutile
   ici, l'OLS domine).

## 3. Tests
1. DGP hétérogène (θ_i ~ N(θ̄, σ²)) : θ̂_MG → θ̄ ; couverture de l'IC
   inter-individus correcte (MC sur N ∈ {20, 50}, T ∈ {50, 100}).
2. Biais des effets fixes dynamiques reproduit sur le même DGP
   (démonstration pour la doc — justifie l'existence du module).
3. Panel non cylindré : résultats invariants à l'ordre des individus.
4. Validation externe : Stata xtpmg option mg / R ardlverse (panel_ardl
   estimator="mg") sur données publiques communes → θ̂_MG à 1e-4.

## 4. Liens
Spec 23 (PMG : contraindre le long terme, laisser le court terme libre —
même infrastructure + une étape ML) ; spec 24 (dépendance transversale) ;
le conteneur panel et la boucle par individu servent aux trois.
