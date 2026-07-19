# ardlpy

Bibliothèque Python d'économétrie des séries temporelles couvrant la
généalogie complète des modèles ARDL : estimation ARDL/UECM, bounds
tests de cointégration, valeurs critiques modernes (surfaces de
réponse, p-values), et — sur la feuille de route — inférence bootstrap,
NARDL, QARDL, Fourier ARDL et panels hétérogènes (MG/PMG/CS-ARDL).

**Rigueur méthodologique d'abord** : chaque table de valeurs critiques
cite sa source exacte et est recoupée par une seconde source publiée ou
par le moteur de simulation interne
([PROVENANCE.md](src/ardlpy/critical_values/PROVENANCE.md)) ; chaque
estimateur est validé contre statsmodels, le package R `ARDL` et les
résultats publiés (réplication de l'application salaires UK de Pesaran,
Shin & Smith 2001 incluse en test de non-régression) ; les anomalies
détectées dans les sources de référence elles-mêmes sont consignées
dans un [registre public](docs/VALIDATION_OBSERVATIONS.md).

## Installation

```bash
pip install -e ".[dev]"        # depuis un clone (pas encore sur PyPI)
```

Dépendances : numpy, scipy, pandas, statsmodels (Python ≥ 3.11).

## Exemple : bounds test avec p-values

```python
import pandas as pd
from ardlpy.bounds import bounds_test
from ardlpy.datasets import load_denmark

data = load_denmark()          # données danoises (Johansen & Juselius 1990)
res = bounds_test(
    data["LRM"], data[["LRY", "IBO", "IDE"]],
    case=3,                     # constante non restreinte (PSS 2001)
    order=(3, {"LRY": 1, "IBO": 3, "IDE": 2}),
)
print(res.summary())
```

```text
Bounds test PSS 2001 — cas 3, k=3, UECM(3; LRY:1, IBO:3, IDE:2), cv_source=kripfganz

F_overall = 6.2059   décision (5%) : cointegration
p-values F (K&S 2020) : p_I0 = 0.0005, p_I1 = 0.0039
t_BDM     = -4.5479   décision (5%) : cointegration
décision jointe F+t (spec 11) : cointegration
...
```

- `decision_f` / `decision_t` / `decision_joint` sont à états explicites
  (`cointegration` / `no_cointegration` / `inconclusive` /
  `degenerate_suspicion`) — jamais un booléen ; la zone non concluante
  est lue en continu : « inconclusive, p ∈ [p_I1, p_I0] ».
- `cv_source` : `"kripfganz"` (défaut — surfaces de réponse, p-values),
  `"pss"` (valeurs publiées PSS 2001, reproduction de la littérature),
  `"narayan"` (petits échantillons 30 ≤ T ≤ 80).
- L'estimateur sous-jacent est accessible directement :
  `ardlpy.core.ardl.ARDL` (sélection d'ordre sur échantillon commun,
  GETS, vues ECM/long terme exactes).

## État des phases (feuille de route : [00_INDEX](docs/references/00_INDEX.md))

| Phase | Contenu | État |
|---|---|---|
| 1 (v0.1.0) | Algèbre ARDL↔ECM (spec 03), estimateur ARDL/UECM + sélection + GETS (05), bounds test 5 cas (10), t-test BDM + décision jointe (11), garde-fous PS98 (09) | ✅ close — jalon : réplication PSS 2001 (F/t à 1e-4 vs R ARDL) |
| 2 (en cours) | Moteur de simulation de CV + Narayan 2005 (12) ✅ ; surfaces de réponse K&S, p-values (13) ✅ voie A1 ; CUSUM (26), racines unitaires (27), datasets/saisonnalité (04), Engle-Granger (06) à venir | 🚧 |
| 3 | Bootstrap (14), cadre à 3 tests et dégénérescences (15), Johansen (07) ; premier module Rust | ⬜ |
| 4-8 | NARDL (17), Fourier (19-21), simulations dynamiques (25), QARDL (18), panels (22-24), docs (28) | ⬜ |

La source de vérité du projet est le dossier
[`docs/references/`](docs/references/00_INDEX.md) (28 spécifications
d'implémentation) ; le cycle de travail et les règles numériques non
négociables sont documentées dans le dépôt.

## Tests

```bash
pytest -m "not slow and not external"   # suite CI (~1 min)
pytest -m external                       # réplications R (valeurs pré-générées)
pytest -m slow                           # Monte Carlo complets (nightly)
```

## Licence

MIT — voir [LICENSE](LICENSE). Les valeurs critiques encodées
proviennent de tables publiées (provenance et licences documentées dans
[PROVENANCE.md](src/ardlpy/critical_values/PROVENANCE.md)) ; aucun
matériel tiers non licencié n'est redistribué.
