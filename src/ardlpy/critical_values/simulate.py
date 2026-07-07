r"""Moteur de simulation des valeurs critiques du bounds test (spec 12 §2.3).

Simule la distribution sous H0 des statistiques F_overall et t_BDM de
l'UECM (spec 10) pour un cas déterministe, un nombre de régresseurs k et
une taille T donnés :

- H0 : Δy_t = eps_t (lam = 0, gamma = 0) — y est une marche aléatoire ;
- borne inférieure (« tout I(0) ») : x_j iid N(0,1) ;
- borne supérieure (« tout I(1) ») : x_j marches aléatoires
  indépendantes (DGP de PSS 2001, annexe des tables CI/CII).

La régression simulée est Δy_t sur [det, y_{t-1}, x_{1,t-1}, ...,
x_{k,t-1}] — pas de dynamique de court terme, conformément au DGP (les
distributions asymptotiques n'en dépendent pas ; PSS 2001, §5).

Usages (spec 12 §2.3) : (a) reproduire les tables publiées (validation,
solde de la dette QUESTIONS.md spec 10 §4) ; (b) fournir des CV pour des
configurations non tabulées (ex. seuil 2.5 %, k > 10) ; (c) base du
futur backend Rust.

Conventions numériques : moindres carrés par QR batchée (jamais
d'inversion de X'X — règle du projet) ; générateur
``numpy.random.Generator`` à seed explicite, journalisée avec tous les
paramètres dans l'objet résultat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_CASE_DET = {1: "none", 2: "const", 3: "const", 4: "trend", 5: "trend"}
_CASE_RESTRICTED = {2: "const", 4: "trend"}


@dataclass(frozen=True)
class SimulatedBounds:
    """Quantiles simulés des statistiques du bounds test.

    Tous les paramètres de simulation sont journalisés : l'objet suffit à reproduire exactement le calcul.
    """

    case: int
    k: int
    t_obs: int
    n_sims: int
    seed: int
    i1: bool  # True = borne « tout I(1) », False = « tout I(0) »
    chunk: int  # fait partie de la reproductibilité : le flux aléatoire
    # est tiré par lots, donc dépend du découpage (journalisé, règle 2)
    alphas: tuple[float, ...]
    f_quantiles: dict[float, float]
    t_quantiles: dict[float, float]
    f_stats: FloatArray = field(repr=False)
    t_stats: FloatArray = field(repr=False)

    def f_cv(self, alpha: float) -> float:
        """Valeur critique F au seuil ``alpha`` (quantile 1 - alpha)."""
        return self.f_quantiles[alpha]

    def t_cv(self, alpha: float) -> float:
        """Valeur critique t au seuil ``alpha`` (quantile alpha, test
        unilatéral gauche)."""
        return self.t_quantiles[alpha]


def simulate_bounds(
    case: int,
    k: int,
    t_obs: int = 1000,
    n_sims: int = 40_000,
    seed: int = 0,
    i1: bool = True,
    alphas: tuple[float, ...] = (0.10, 0.05, 0.025, 0.01),
    chunk: int = 2_000,
) -> SimulatedBounds:
    """Simule les CV du bounds test sous H0 (spec 12 §2.3).

    Parameters
    ----------
    case : int
        Cas déterministe PSS (1 à 5) ; pour les cas II/IV, le
        déterministe restreint entre dans le vecteur testé (spec 10 §3).
    k : int
        Nombre de régresseurs (k = 0 : test sur y_{t-1} seul).
    t_obs : int
        Taille de l'échantillon simulé (1000 = convention asymptotique
        de PSS 2001 ; 30-80 pour les tables petits échantillons).
    n_sims : int
        Nombre de réplications (40 000 = convention PSS 2001).
    seed : int
        Graine du ``numpy.random.Generator`` (journalisée).
    i1 : bool
        True : x_j marches aléatoires (borne I(1)) ; False : x_j iid
        (borne I(0)).
    alphas : tuple of float
        Seuils des quantiles retournés.
    chunk : int
        Taille des lots de la QR batchée (mémoire ~ chunk × T × (k+3)).
        Les tirages étant faits par lots, ``chunk`` fait partie des
        paramètres de reproductibilité (journalisé dans le résultat) :
        même (seed, n_sims, chunk) -> mêmes statistiques exactes.

    Returns
    -------
    SimulatedBounds
        Quantiles F (droite) et t (gauche) + tirages complets.

    Examples
    --------
    >>> sb = simulate_bounds(case=3, k=1, t_obs=200, n_sims=200, seed=42)
    >>> sb.seed, sb.n_sims, sb.case, sb.i1
    (42, 200, 3, True)
    >>> 0 < sb.f_cv(0.05) < 20
    True
    """
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case doit être dans 1..5, reçu {case}.")
    if k < 0:
        raise ValueError("k >= 0 requis.")
    if t_obs < 20:
        raise ValueError("t_obs >= 20 requis.")

    rng = np.random.default_rng(seed)
    det = _CASE_DET[case]
    n_det = {"none": 0, "const": 1, "trend": 2}[det]
    n_restr = k + 1 + (1 if case in _CASE_RESTRICTED else 0)
    k_par = n_det + 1 + k
    n_eff = t_obs - 1  # Δy_t, t = 2..T
    lam_pos = n_det  # position de y_{t-1} dans le design

    # colonnes déterministes (identiques pour toutes les réplications)
    det_cols = np.empty((n_eff, n_det))
    if n_det >= 1:
        det_cols[:, 0] = 1.0
    if n_det == 2:
        det_cols[:, 1] = np.arange(2, t_obs + 1, dtype=np.float64)

    # colonnes du design restreint (H0 imposée) : déterministes NON testés
    restr_idx = list(range(n_det))
    if case in _CASE_RESTRICTED:
        restr_idx = restr_idx[:-1]  # le dernier déterministe est testé

    f_stats = np.empty(n_sims)
    t_stats = np.empty(n_sims)

    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        eps = rng.standard_normal((m, t_obs))
        y = np.cumsum(eps, axis=1)  # marche aléatoire sous H0
        dy = np.diff(y, axis=1)  # = eps[:, 1:]
        y_lag = y[:, :-1]

        design = np.empty((m, n_eff, k_par))
        design[:, :, :n_det] = det_cols
        design[:, :, lam_pos] = y_lag
        if k > 0:
            x_innov = rng.standard_normal((m, t_obs, k))
            x = np.cumsum(x_innov, axis=1) if i1 else x_innov
            design[:, :, lam_pos + 1 :] = x[:, :-1, :]

        # --- régression non contrainte (QR batchée, règle du projet) ---
        q_u, r_u = np.linalg.qr(design)
        qty = np.einsum("stk,st->sk", q_u, dy)
        coefs = np.linalg.solve(r_u, qty[:, :, None])[:, :, 0]
        ssr_u = np.einsum("st,st->s", dy, dy) - np.einsum("sk,sk->s", qty, qty)

        # --- régression contrainte (H0) ---
        if restr_idx:
            q_r, _ = np.linalg.qr(design[:, :, restr_idx])
            qty_r = np.einsum("stk,st->sk", q_r, dy)
            ssr_r = np.einsum("st,st->s", dy, dy) - np.einsum("sk,sk->s", qty_r, qty_r)
        else:
            ssr_r = np.einsum("st,st->s", dy, dy)

        df = n_eff - k_par
        f_stats[done : done + m] = ((ssr_r - ssr_u) / n_restr) / (ssr_u / df)

        # --- t sur y_{t-1} : se via R^{-1} (diag de (X'X)^{-1}) ---
        r_inv = np.linalg.solve(r_u, np.broadcast_to(np.eye(k_par), r_u.shape))
        xtx_inv_lam = np.einsum("sj,sj->s", r_inv[:, lam_pos, :], r_inv[:, lam_pos, :])
        se_lam = np.sqrt(ssr_u / df * xtx_inv_lam)
        t_stats[done : done + m] = coefs[:, lam_pos] / se_lam

        done += m

    f_q = {a: float(np.quantile(f_stats, 1 - a)) for a in alphas}
    t_q = {a: float(np.quantile(t_stats, a)) for a in alphas}
    return SimulatedBounds(
        case=case,
        k=k,
        t_obs=t_obs,
        n_sims=n_sims,
        seed=seed,
        i1=i1,
        chunk=chunk,
        alphas=tuple(alphas),
        f_quantiles=f_q,
        t_quantiles=t_q,
        f_stats=f_stats,
        t_stats=t_stats,
    )
