"""Spec 12 — dérivation des tolérances du recoupement PSS par calcul
d'erreur MC (arbitrage utilisateur du 2026-07-07 : critère « >= 3
erreurs types combinées », pas de seuil ad hoc).

Erreur type asymptotique d'un quantile empirique q_p estimé sur n
tirages iid : SE(q_p) = sqrt(p(1-p)/n) / f(q_p), où f est la densité au
quantile (estimée ici par différence finie centrée des statistiques
d'ordre, fenêtre h=0.005 en probabilité).

Erreur combinée d'une comparaison table publiée (n_pss = 40 000,
PSS 2001) vs simulation interne (n_sim = 100 000) :
SE_comb = sqrt(SE_40k^2 + SE_100k^2). Tolérance par seuil = 3 x le
maximum de SE_comb sur les configurations extrêmes (les distributions
les plus étalées : cas V, et cas I k=0 pour la queue).

Sortie : tableau imprimé + validation croisée avec la dispersion
inter-seeds observée (deux runs indépendants à 300k, cf. QUESTIONS.md).
Les valeurs retenues sont reportées dans PROVENANCE.md et dans le test
slow de tests/unit/critical_values/test_simulate.py.
"""

from __future__ import annotations

import numpy as np

from ardlpy.critical_values.simulate import simulate_bounds

PARAMS = {
    "n_sims": 100_000,
    "t_obs": 1000,
    "seed": 777,
    "h": 0.005,  # fenêtre de la différence finie (en probabilité)
    "n_pss": 40_000,  # réplications de PSS 2001
    "configs": [  # configurations extrêmes (dispersion maximale)
        (1, 0, True),
        (3, 1, False),
        (3, 1, True),
        (5, 0, True),
        (5, 2, True),
        (5, 2, False),
    ],
}


def quantile_se(stats: np.ndarray, p: float, n: int, h: float) -> float:
    """SE(q_p) pour un échantillon de taille n, densité par différence
    finie centrée sur les quantiles empiriques."""
    q_hi = np.quantile(stats, min(p + h, 1.0))
    q_lo = np.quantile(stats, max(p - h, 0.0))
    density = 2 * h / (q_hi - q_lo)
    return float(np.sqrt(p * (1 - p) / n) / density)


def main() -> None:
    rows = []
    for case, k, i1 in PARAMS["configs"]:
        sb = simulate_bounds(
            case=case,
            k=k,
            t_obs=PARAMS["t_obs"],
            n_sims=PARAMS["n_sims"],
            seed=PARAMS["seed"],
            i1=i1,
        )
        for alpha in (0.10, 0.05, 0.025, 0.01):
            p = 1 - alpha  # quantile droit du F
            se40 = quantile_se(sb.f_stats, p, PARAMS["n_pss"], PARAMS["h"])
            se100 = quantile_se(sb.f_stats, p, PARAMS["n_sims"], PARAMS["h"])
            comb = float(np.hypot(se40, se100))
            rows.append((case, k, i1, "F", alpha, se40, se100, comb))
            # t : quantile gauche
            se40t = quantile_se(sb.t_stats, alpha, PARAMS["n_pss"], PARAMS["h"])
            se100t = quantile_se(sb.t_stats, alpha, PARAMS["n_sims"], PARAMS["h"])
            rows.append(
                (case, k, i1, "t", alpha, se40t, se100t, float(np.hypot(se40t, se100t)))
            )

    print(
        f"{'config':16s} {'stat':4s} {'alpha':6s} {'SE_40k':>8s} "
        f"{'SE_100k':>8s} {'SE_comb':>8s} {'3xSE':>7s}"
    )
    for case, k, i1, stat, alpha, se40, se100, comb in rows:
        cfg = f"case {case} k{k} {'I1' if i1 else 'I0'}"
        print(
            f"{cfg:16s} {stat:4s} {alpha:<6g} {se40:8.4f} {se100:8.4f} "
            f"{comb:8.4f} {3 * comb:7.3f}"
        )

    print("\nTolérance par seuil = 3 x max(SE_comb) sur les configs :")
    for stat in ("F", "t"):
        for alpha in (0.10, 0.05, 0.025, 0.01):
            worst = max(r[7] for r in rows if r[3] == stat and r[4] == alpha)
            print(f"  {stat} {alpha:<5g}: 3 x {worst:.4f} = {3 * worst:.3f}")


if __name__ == "__main__":
    main()
