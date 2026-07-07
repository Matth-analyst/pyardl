"""Spec 12 — recoupement intégral des tables PSS 2001 par simulation
interne + génération du seuil 2.5 % (solde la dette QUESTIONS.md
spec 10 §4).

Reproductible : tous les paramètres du DGP sont dans PARAMS ci-dessous ;
les seeds sont dérivées déterministiquement de (case, k, bound). Sorties :

- ``validation/results/spec12_pss_crosscheck.csv`` : chaque cellule
  encodée vs quantile simulé, écart, statut (tolérance : ±0.05 pour F,
  ±0.04 pour t — spec 12 §3.1) ;
- ``validation/results/spec12_p025_table.py`` : tables du seuil 2.5 %
  (F cas I-V, t cas I/III/V, k=0..10) prêtes à intégrer dans
  ``src/ardlpy/critical_values/pss2001.py``, provenance incluse.

Usage : python validation/spec12_montecarlo.py [--fast]
(--fast : n_sims=5000 pour un fumigène rapide, ne pas utiliser pour
générer les tables officielles).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

from ardlpy.critical_values.pss2001 import LEVELS, MAX_K, T_BOUNDS, get_bounds
from ardlpy.critical_values.simulate import simulate_bounds

PARAMS = {
    "t_obs": 1000,  # convention asymptotique PSS 2001
    "n_sims": 100_000,  # > 40 000 de PSS (précision MC ~ ±0.02)
    "chunk": 2_000,
    "seed_base_i0": 910_000,  # seed = base + case*100 + k
    "seed_base_i1": 920_000,
    # Critère de concordance (arbitrage 2026-07-07, cf. PROVENANCE.md) :
    # tolérance PAR CELLULE = 3 x SE combinée, SE_comb =
    # hypot(SE(quantile, n=40k), SE(quantile, n=100k)), la densité au
    # quantile étant estimée par différence finie centrée (h en proba).
    "n_pss": 40_000,
    "se_window": 0.005,
    "n_se": 3.0,
    "alphas": (0.10, 0.05, 0.025, 0.01),
}

RESULTS_DIR = Path(__file__).parent / "results"


def cell_tolerance(stats: np.ndarray, p: float, n_sims: int) -> float:
    """Tolérance par cellule = n_se x SE combinée (PSS 40k + simulation).

    SE(q_p) = sqrt(p(1-p)/n) / f(q_p), densité estimée par différence
    finie centrée de fenêtre ``se_window`` (en probabilité) sur les
    tirages simulés — cf. validation/spec12_mc_error.py et PROVENANCE.md.
    """
    h = PARAMS["se_window"]
    q_hi = np.quantile(stats, min(p + h, 1.0))
    q_lo = np.quantile(stats, max(p - h, 0.0))
    density = 2 * h / (q_hi - q_lo)
    se_pss = np.sqrt(p * (1 - p) / PARAMS["n_pss"]) / density
    se_sim = np.sqrt(p * (1 - p) / n_sims) / density
    return float(PARAMS["n_se"] * np.hypot(se_pss, se_sim))


def main(fast: bool = False) -> int:
    n_sims = 5_000 if fast else PARAMS["n_sims"]
    rows: list[dict[str, object]] = []
    p025_f: dict[int, list[tuple[float, float]]] = {}
    p025_t: dict[int, list[tuple[float, float]]] = {}
    n_fail = 0
    t0 = time.time()

    for case in (1, 2, 3, 4, 5):
        p025_f[case] = []
        if case in T_BOUNDS:
            p025_t[case] = []
        for k in range(MAX_K + 1):
            lo = simulate_bounds(
                case=case,
                k=k,
                t_obs=PARAMS["t_obs"],
                n_sims=n_sims,
                seed=PARAMS["seed_base_i0"] + case * 100 + k,
                i1=False,
                alphas=PARAMS["alphas"],
                chunk=PARAMS["chunk"],
            )
            up = simulate_bounds(
                case=case,
                k=k,
                t_obs=PARAMS["t_obs"],
                n_sims=n_sims,
                seed=PARAMS["seed_base_i1"] + case * 100 + k,
                i1=True,
                alphas=PARAMS["alphas"],
                chunk=PARAMS["chunk"],
            )
            p025_f[case].append((lo.f_cv(0.025), up.f_cv(0.025)))
            if case in T_BOUNDS:
                p025_t[case].append((lo.t_cv(0.025), up.t_cv(0.025)))

            for alpha in LEVELS:
                f_enc = get_bounds("F", case=case, k=k, alpha=alpha)
                for bound, sb, sim, enc in (
                    ("I0", lo, lo.f_cv(alpha), f_enc[0]),
                    ("I1", up, up.f_cv(alpha), f_enc[1]),
                ):
                    gap = sim - enc
                    tol = cell_tolerance(sb.f_stats, 1 - alpha, n_sims)
                    ok = abs(gap) <= tol
                    n_fail += not ok
                    rows.append(
                        dict(
                            stat="F",
                            case=case,
                            k=k,
                            alpha=alpha,
                            bound=bound,
                            encoded=enc,
                            simulated=round(sim, 4),
                            gap=round(gap, 4),
                            tol_3se=round(tol, 4),
                            ok=ok,
                        )
                    )
                if case in T_BOUNDS:
                    t_enc = get_bounds("t", case=case, k=k, alpha=alpha)
                    for bound, sb, sim, enc in (
                        ("I0", lo, lo.t_cv(alpha), t_enc[0]),
                        ("I1", up, up.t_cv(alpha), t_enc[1]),
                    ):
                        gap = sim - enc
                        tol = cell_tolerance(sb.t_stats, alpha, n_sims)
                        ok = abs(gap) <= tol
                        n_fail += not ok
                        rows.append(
                            dict(
                                stat="t",
                                case=case,
                                k=k,
                                alpha=alpha,
                                bound=bound,
                                encoded=enc,
                                simulated=round(sim, 4),
                                gap=round(gap, 4),
                                tol_3se=round(tol, 4),
                                ok=ok,
                            )
                        )
            print(
                f"case {case} k {k:2d} fait "
                f"({time.time() - t0:7.1f}s, {n_fail} hors tolérance)",
                flush=True,
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_fast" if fast else ""  # --fast n'écrase pas les officiels
    csv_path = RESULTS_DIR / f"spec12_pss_crosscheck{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # tables 2.5 % prêtes à intégrer
    py_path = RESULTS_DIR / f"spec12_p025_table{suffix}.py"
    with py_path.open("w", encoding="utf-8") as fh:
        fh.write(
            '"""Seuil 2.5 % simulé (spec 12) — provenance : moteur interne\n'
            f"simulate_bounds, t_obs={PARAMS['t_obs']}, n_sims={n_sims},\n"
            f"seeds={PARAMS['seed_base_i0']}/{PARAMS['seed_base_i1']} + "
            "case*100 + k, chunk="
            f"{PARAMS['chunk']}, généré par validation/spec12_montecarlo.py."
            '"""\n\n'
        )
        fh.write("F_P025 = {\n")
        for case, cells in p025_f.items():
            fh.write(f"    {case}: [\n")
            for lo_v, up_v in cells:
                fh.write(f"        ({lo_v:.4f}, {up_v:.4f}),\n")
            fh.write("    ],\n")
        fh.write("}\n\nT_P025 = {\n")
        for case, cells in p025_t.items():
            fh.write(f"    {case}: [\n")
            for lo_v, up_v in cells:
                fh.write(f"        ({lo_v:.4f}, {up_v:.4f}),\n")
            fh.write("    ],\n")
        fh.write("}\n")

    print(f"\n{len(rows)} cellules recoupées, {n_fail} hors tolérance.")
    print(f"Résultats : {csv_path}\nTables 2.5 % : {py_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(fast="--fast" in sys.argv))
