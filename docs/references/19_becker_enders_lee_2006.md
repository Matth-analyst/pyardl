# Spec 19 — Becker, Enders & Lee (2006) : les termes de Fourier pour ruptures lisses

## Référence
Becker, Enders & Lee (2006), *Journal of Time Series Analysis*, 27(3),
381–409. DOI: 10.1111/j.1467-9892.2006.00478.x. Clé : `becker2006fourier`.
Branche : **8** · Module : `pyardl.fourier` (briques) · Priorité : v0.5.

## 1. Apport
Idée fondatrice de toute la branche Fourier : plutôt que dater et compter
des ruptures structurelles (Bai-Perron), approximer une composante
déterministe variant dans le temps par une somme de sinusoïdes basse
fréquence :

d(t) ≈ a₀ + Σ_{f=1}^{F} [ a_f sin(2πft/T) + b_f cos(2πft/T) ]

Une seule fréquence (F=1, voire fréquence fractionnaire) capte déjà des
ruptures lisses multiples de forme inconnue, avec 2 paramètres seulement.
L'article développe un test de stationnarité autour de cette composante ;
pour la bibliothèque, l'essentiel est la **brique déterministe Fourier**
et sa méthodologie de sélection, réutilisées par les specs 20-21.

## 2. Implémentation

1. `fourier_terms(T, freqs)` → matrice [sin, cos] par fréquence ; support
   des fréquences entières et fractionnaires (grille pas 0.1) ; intégration
   au conteneur déterministe (specs 03/04/10) : `det="const+fourier"`,
   `fourier_k=1`, `fourier_freq="auto"`.
2. **Sélection de la fréquence** : grille f ∈ {0.1, ..., 5} (ou entiers
   1..5), choisir f minimisant la SSR du modèle — documenter que f
   sélectionnée n'est pas un paramètre « testable » standard (problème de
   paramètre non identifié sous H₀ — Davies) → les CV des tests avals
   doivent être simulés/bootstrappés avec la sélection intégrée à la
   boucle (règle absolue pour les specs 20-21).
3. **Test de non-linéarité déterministe** : F-test de H₀ : a_f = b_f = 0 ;
   CV non standards si f est estimée → simuler via le moteur spec 12
   (option `freq_estimated=True` qui re-sélectionne f à chaque
   réplication).
4. Version du test de stationnarité de l'article (KPSS-avec-Fourier) :
   utile en pré-test (spec 27 s'en servira comme option
   `stationarity="fourier_kpss"`).

## 3. Tests
1. DGP avec rupture lisse (logistique) : la composante Fourier F=1 capte
   la trajectoire (R² de la partie déterministe > 0.9) ; DGP sans rupture :
   test §2.3 ~ taille (CV simulés avec sélection).
2. Fréquence : la grille retrouve la fréquence injectée (MC).
3. fourier_kpss : taille/puissance de base reproduisant qualitativement
   les résultats de l'article.

## 4. Liens
Brique de 20 (Fourier ADL) et 21 (Fourier bounds/bootstrap) ; s'insère
dans le conteneur déterministe créé en 03/04 et consommé par 10 ;
CV simulés via le moteur 12 ; leçon de Davies appliquée partout où f est
estimée.
