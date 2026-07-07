# Spec 14 — McNown, Sam & Goh (2018) : le bootstrap ARDL et les dégénérescences

## Référence
McNown, Sam & Goh (2018), *Applied Economics*, 50(13), 1509–1521.
DOI: 10.1080/00036846.2017.1366643. Clé : `mcnown2018bootstrap`.
Branche : **5** · Module : `ardlpy.bootstrap` · Priorité : **v0.3** —
première grande fonctionnalité absente de Python.

## 1. Apport

Deux contributions :

1. **Formalisation des cas dégénérés** du bounds test :
   - Dégénérescence de type 1 : λ ≠ 0 mais γ₁ = ... = γ_k = 0
     (le F global peut rejeter alors que les x ne portent aucune relation
     de long terme — y s'auto-corrige seul).
   - Dégénérescence de type 2 : γ ≠ 0 mais λ = 0 (pas de force de rappel).
   Conclure à la cointégration exige de rejeter H₀ ET d'écarter les deux
   dégénérescences → il faut tester aussi les γ seuls (d'où le 3e test,
   formalisé en spec 15).
2. **Bootstrap des tests** : au lieu des bornes (et de leur zone non
   concluante), générer la distribution des statistiques **sous H₀,
   conditionnellement aux données** → CV exacts pour l'échantillon, plus
   de zone inconclusive, meilleure taille.

## 2. Algorithme bootstrap (cœur de l'implémentation)

Entrées : y, X, ordres (p, q), cas, B (défaut 2999), seed.

1. **Estimation sans restriction** : UECM complet (spec 10) → statistiques
   observées F_overall, t, (F_indep pour spec 15).
2. **Estimation du DGP sous H₀** : ré-estimer le modèle en imposant la
   restriction du test considéré (λ=γ=0 pour F_overall : le modèle H₀ est
   l'UECM sans les termes de niveau) → coefficients contraints et résidus
   centrés ε̃.
   Les x sont modélisés par leur propre processus (VAR en différences /
   équations marginales estimées sur les données) pour pouvoir régénérer
   un système complet — implémenter l'estimation jointe du bloc marginal.
3. **Boucle b = 1..B** :
   a. rééchantillonner les innovations : iid (défaut) avec option wild
      (Rademacher) pour l'hétéroscédasticité ;
   b. régénérer récursivement (x*_t puis y*_t) sous H₀, avec burn-in et
      valeurs initiales = données observées ;
   c. ré-estimer l'UECM complet sur (y*, X*) et stocker F*_b, t*_b.
4. **CV et p-values bootstrap** : CV_α = quantile (1−α) des F* ;
   p = (1 + #{F* ≥ F_obs}) / (B + 1). Décision binaire (plus de zone
   non concluante) — mais reporter aussi les bornes classiques pour
   comparaison.

Détails critiques :
- t : test unilatéral gauche (quantile α des t*).
- Reproductibilité : seed obligatoire dans l'objet résultat, B, méthode
  de rééchantillonnage et modèle des x journalisés.
- Performance : B×(régénération+OLS) — boucle cible du backend Rust
  (rayon : parallélisme par réplication, RNG par flux indépendants
  contrôlés — attention à la reproductibilité inter-plateformes,
  utiliser un compteur de flux type Philox).

## 3. API
```python
res = ardlpy.bootstrap_bounds_test(y, X, case=3, order=..., B=2999,
        resample="iid"|"wild", seed=..., backend="numpy"|"rust")
# ajoute à BoundsTestResults : boot_cv (df), boot_pvalues,
# decisions bootstrap par test, distribution stockée (option)
```

## 4. Tests
1. **Taille** : DGP sous H₀ (plusieurs configurations, dont erreurs
   hétéroscédastiques pour wild) → rejet à 5 % ∈ [3.5, 6.5] %
   (MC 1000 × B 499 — coût maîtrisé en CI par grille réduite + nightly
   étendu).
2. **Puissance vs bornes** : DGP cointégré T=60 → puissance bootstrap ≥
   test des bornes ; zone non concluante résorbée (compter les cas).
3. Dégénérescence type 1 simulée → F_overall rejette mais le diagnostic
   la signale (précurseur spec 15).
4. Équivalence backend : numpy vs rust, même seed logique → mêmes
   quantiles (tolérance statistique documentée si RNG diffèrent : comparer
   les distributions par KS, p > 0.99).
5. Validation externe : R bootCT sur données communes — mêmes ordres de
   grandeur de CV bootstrap (concordance exacte impossible, RNG
   différents : comparer taille/puissance simulées et décisions).

## 5. Liens
Spec 15 (le cadre à 3 tests utilise ce moteur) ; spec 16 (raffinements
conditionnel/inconditionnel de Bertelli) ; spec 21 (Fourier bootstrap) ;
specs 12-13 (mêmes briques de simulation) ; backend Rust (cible n°1 avec
simulate_bounds).
