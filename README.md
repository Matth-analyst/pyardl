# ardlpy

Bibliothèque Python d'économétrie des séries temporelles couvrant la
généalogie complète des modèles ARDL : estimation ARDL/UECM, bounds tests
de cointégration (cadre à 3 tests), valeurs critiques modernes (surfaces
de réponse), inférence bootstrap, NARDL, QARDL, Fourier ARDL, panels
hétérogènes (MG/PMG/CS-ARDL).

La source de vérité du projet est le dossier [`docs/references/`](docs/references/00_INDEX.md)
(spécifications 00 à 28). Voir la documentation de contribution pour le cycle de travail et les
règles de développement.

## État du projet

En développement (v0.1 — cœur ARDL/UECM + bounds test PSS 2001).

## Installation (développement)

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest -m "not slow and not external"
```
