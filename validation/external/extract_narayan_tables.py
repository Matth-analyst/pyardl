"""Extraction programmatique des tables de Narayan (2005) depuis le
source R de dynamac (Jordan & Philips) — spec 12 §2.1.

Transcription des tables « Case II / III / V » de Narayan (2005),
*Applied Economics* 37(17), 1979-1990, telles qu'embarquées dans
``pssbounds()`` (fichier R/dynamac.R, branches ``obs <= 30`` à
``obs <= 80``). Un parseur programmatique élimine le risque d'erreur de
recopie manuelle ; le recoupement statistique est fait par le moteur
interne (tests spec 12).

Usage : python validation/external/extract_narayan_tables.py <dynamac.R>
Écrit src/ardlpy/critical_values/narayan2005.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

T_GRID = (30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80)
CASES = (2, 3, 5)  # Narayan ne couvre ni le cas I ni le cas IV


def parse(source: str) -> dict[int, dict[int, list[list[float]]]]:
    """-> {T: {case: [[I0_10, I1_10, I0_5, I1_5, I0_1, I1_1] x 8 k]}}."""
    # positions des branches obs <= NN (la branche asymptotique est après)
    branch_iter = list(re.finditer(r"(?:else )?if \(obs <= (\d+)\)", source))
    if len(branch_iter) < len(T_GRID):
        raise RuntimeError(f"{len(branch_iter)} branches trouvées, attendu >= 11")

    tables: dict[int, dict[int, list[list[float]]]] = {}
    for i, m in enumerate(branch_iter[: len(T_GRID)]):
        t_value = T_GRID[i]
        assert int(m.group(1)) == t_value + (5 if i > 0 else 0) or int(m.group(1)) in (
            t_value,
            t_value + 5,
        ), m.group(1)
        start = m.end()
        end = branch_iter[i + 1].start() if i + 1 < len(branch_iter) else len(source)
        segment = source[start:end]
        tables[t_value] = {}
        for case in CASES:
            # le cas 5 est la branche « else { # case == 5 » (commentaire)
            case_m = re.search(rf"case == {case}", segment)
            if case_m is None:
                raise RuntimeError(f"case {case} absent de la branche T={t_value}")
            fmat_m = re.search(
                r"fmat <- matrix\(c\((.*?)\),\s*ncol = 6",
                segment[case_m.end() :],
                re.DOTALL,
            )
            if fmat_m is None:
                raise RuntimeError(f"fmat absent : T={t_value}, case {case}")
            body = re.sub(r"#[^\n]*", "", fmat_m.group(1))  # ôte les commentaires
            numbers = [float(v) for v in re.findall(r"-?\d+\.?\d*", body)]
            if len(numbers) != 8 * 6:
                raise RuntimeError(
                    f"T={t_value} case {case} : {len(numbers)} valeurs != 48"
                )
            tables[t_value][case] = [numbers[r * 6 : r * 6 + 6] for r in range(8)]
    return tables


def emit(tables: dict[int, dict[int, list[list[float]]]], out: Path) -> None:
    lines = [
        '"""Tables de Narayan (2005) — bornes F petits échantillons (spec 12 §2.1).',
        "",
        "Source primaire : Narayan, P. K. (2005), 'The saving and investment",
        "nexus for China: evidence from cointegration tests', *Applied",
        "Economics*, 37(17), 1979-1990 — tables « Case II / III / V »,",
        "T = 30..80 (pas de 5), k = 0..7, seuils 10/5/1 %, statistique F",
        "uniquement (Narayan ne publie pas de bornes t).",
        "",
        "Canal de transcription et recoupement : voir PROVENANCE.md (même",
        "protocole que les tables PSS 2001). Fichier GÉNÉRÉ par",
        "validation/external/extract_narayan_tables.py — ne pas éditer à la",
        'main."""',
        "",
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "import numpy.typing as npt",
        "",
        "T_GRID: tuple[int, ...] = " + repr(T_GRID),
        "MAX_K_NARAYAN = 7",
        "",
        "# {T: {case: array (8, 3, 2)}} — (k, seuil 10/5/1 %, borne I0/I1)",
        "_RAW = {",
    ]
    for t_value, cases in tables.items():
        lines.append(f"    {t_value}: {{")
        for case, rows in cases.items():
            lines.append(f"        {case}: [")
            for row in rows:
                pairs = ", ".join(f"({row[i]}, {row[i + 1]})" for i in range(0, 6, 2))
                lines.append(f"            [{pairs}],")
            lines.append("        ],")
        lines.append("    },")
    lines += [
        "}",
        "",
        "F_NARAYAN: dict[int, dict[int, npt.NDArray[np.float64]]] = {",
        "    t: {case: np.asarray(rows, dtype=np.float64)"
        " for case, rows in cases.items()}",
        "    for t, cases in _RAW.items()",
        "}",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"écrit : {out} ({len(tables)} tailles T, cas {CASES})")


if __name__ == "__main__":
    src_path = Path(sys.argv[1])
    out_path = (
        Path(__file__).parents[2]
        / "src"
        / "ardlpy"
        / "critical_values"
        / "narayan2005.py"
    )
    emit(parse(src_path.read_text(encoding="utf-8", errors="replace")), out_path)
