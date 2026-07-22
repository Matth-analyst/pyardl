r"""Surfaces de réponse K&S finies-T — voie A2 (spec 13 §2.1).

STATUT (2026-07-19) : **EXPÉRIMENTAL, NON VALIDÉ, BLOQUÉ PAR A3.**
Le code de ce module implémente la forme fonctionnelle publiée, mais
sa concordance avec les valeurs Stata n'a PAS été confirmée par une
comparaison recevable : une comparaison antérieure s'appuyait sur un
exemplaire téléchargé sans autorisation confirmée, et sur un
"preprint" trouvé sur un site tiers dont la légitimité n'a pas été
établie — les deux ont été retirés (voir docs/DEVIATIONS.md,
CHANGELOG). N'UTILISER CE MODULE EN PRODUCTION QU'APRÈS :
(1) autorisation des auteurs (voie A3,
docs/correspondence/2026-07-10_ks_license_draft.md) et
(2) revalidation contre une source de référence légitime.

Valeurs critiques ajustées à la taille d'échantillon et p-values
approchées pour les statistiques F ET t du bounds test, en évaluant les
COEFFICIENTS PUBLIÉS de Kripfganz & Schneider (fichier
``ardl_surfreg_coefs.dta`` distribué avec leur package Stata ardl).

Licence / distribution (voie A2, arbitrage utilisateur) : les
coefficients ne portent pas de licence explicite -> pyardl ne les
REDISTRIBUE PAS. ``download_surface_coefs()`` NE DOIT PAS être appelé
tant que la réponse des auteurs n'est pas reçue (voie A3 en attente).
Une fois l'autorisation obtenue, le téléchargement se fait au premier
usage (appel explicite), mis en cache localement avec empreinte
SHA-256 et provenance journalisée.

Forme fonctionnelle (algorithme publié — Kripfganz & Schneider 2020
§3.2 et Supplementary Appendix ; réimplémentation indépendante, aucun
code Stata copié — règle du projet) : pour chaque quantile tabulé p
(grille de 221 niveaux, en 1/10000e),

    cv(p) = sum_{j=0..4} theta_{j,0,0} / (k+1)^j
          + (1/n)  * sum_{j=0..4} [theta_{j,1,0} + theta_{j,1,1}*sr] / (k+1)^j
          + (1/n^2) * [theta_{0,2,0} + theta_{0,2,1}*sr]
          + (1/n^3) * [theta_{0,3,0} + theta_{0,3,1}*sr]

avec n la taille d'échantillon de l'UECM, k le nombre de régresseurs de
long terme et sr le nombre de coefficients de COURT TERME de l'UECM
(hors déterministes et hors termes de niveau) — pour un UECM(p; q) :
sr = (p - 1) + somme des q_j. Le cas asymptotique s'obtient en
supprimant les termes en 1/n.

p-values : approximation locale de MacKinnon (1996, eq. 12) — les 9
quantiles tabulés les plus proches de la statistique observée sont
projetés sur l'échelle d'une distribution de référence
(F(df1, df2) en fini, chi2/df1 ... cf. code), régression quadratique,
puis probabilité de queue de la valeur ajustée.

Références
----------
Kripfganz & Schneider (2020), *OBES* 82(6) ; Kripfganz & Schneider
(2023), "ardl: Estimating autoregressive distributed lag and
equilibrium correction models", *Stata Journal* 23(4), 983-1019 ;
MacKinnon (1996), *J. Applied Econometrics* 11(6), 601-618.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as sps

__all__ = [
    "download_surface_coefs",
    "crit_value_bounds_finite",
    "pvalue_bounds_finite",
    "cache_dir",
]

_URL = "http://www.kripfganz.de/stata/ardl_surfreg_coefs.dta"
_T_CASES = (1, 3, 5)


def cache_dir() -> Path:
    """Dossier de cache local (``PYARDL_CACHE`` ou ``~/.pyardl``)."""
    base = os.environ.get("PYARDL_CACHE")
    return (Path(base) if base else Path.home() / ".pyardl") / "ks2020"


def _coefs_path() -> Path:
    return cache_dir() / "ardl_surfreg_coefs.dta"


def download_surface_coefs(force: bool = False, url: str = _URL) -> Path:
    """Télécharge les coefficients K&S depuis le site des auteurs
    (premier usage ; aucun matériel redistribué par pyardl).

    Écrit le fichier dans :func:`cache_dir` avec un journal de
    provenance (URL, date, SHA-256). Les usages suivants lisent le
    cache ; ``force=True`` re-télécharge.
    """
    path = _coefs_path()
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        payload = resp.read()
    if not payload[:20].count(b"<") == 0 or len(payload) < 10_000:
        raise RuntimeError(
            f"Téléchargement invalide depuis {url} ({len(payload)} octets) — "
            "vérifier l'URL ou télécharger manuellement dans "
            f"{path}."
        )
    path.write_bytes(payload)
    meta = {
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "downloaded": datetime.date.today().isoformat(),
        "source": "Kripfganz & Schneider, package Stata ardl "
        "(kripfganz.de) — non redistribué par pyardl, cf. PROVENANCE.md",
    }
    (path.parent / "provenance.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return path


_TABLES: dict[tuple[str, int, int], pd.DataFrame] | None = None


def _load_tables() -> dict[tuple[str, int, int], pd.DataFrame]:
    """Charge (et met en cache mémoire) les tables de coefficients."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    path = _coefs_path()
    if not path.exists():
        raise FileNotFoundError(
            "Coefficients K&S finis-T absents du cache local "
            f"({path}). Ils ne sont pas distribués avec pyardl (licence, "
            "cf. PROVENANCE.md) : exécuter une fois\n"
            "    from pyardl.critical_values.ks2020_finite import "
            "download_surface_coefs\n"
            "    download_surface_coefs()\n"
            "pour les télécharger depuis le site des auteurs."
        )
    df = pd.read_stata(path)
    _TABLES = {
        (str(stat), int(case), int(i1)): grp.sort_values("p").reset_index(drop=True)
        for (stat, case, i1), grp in df.groupby(["stat", "c", "I"])
    }
    return _TABLES


def _check(stat: str, case: int, k: int, t_obs: int, sr: int) -> int:
    """Valide les entrées et renvoie le cas EFFECTIF pour la table.

    Pour la statistique t, les cas à déterministe restreint sont servis
    par les surfaces du cas non restreint correspondant (2 -> 3,
    4 -> 5) : la distribution du t n'est pas affectée par la
    restriction des déterministes (elle ne modifie que le vecteur testé
    par le F) — convention lue dans le source ``ardlbounds.ado`` des
    auteurs. NON REVALIDÉE contre une sortie Stata de référence
    légitime (bloqué par A3, cf. docstring du module).
    """
    if stat not in ("F", "t"):
        raise ValueError(f'stat doit être "F" ou "t", reçu {stat!r}.')
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case doit être dans 1..5, reçu {case}.")
    if k < 0:
        raise ValueError("k >= 0 requis.")
    if t_obs < 5:
        raise ValueError("t_obs >= 5 requis.")
    if sr < 0:
        raise ValueError("sr >= 0 requis (nombre de coefficients de court terme).")
    if stat == "t" and case in (2, 4):
        return case + 1
    return case


def _cv_grid(
    stat: str, case: int, i1: bool, k: int, t_obs: int | None, sr: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """(quantiles p en [0,1], CV prédits) pour une borne donnée."""
    tab = _load_tables()[(stat, case, int(i1))]
    kp1 = float(k + 1)
    cv = np.zeros(len(tab))
    for j in range(5):
        cv += tab[f"theta_{j}_0_0"].to_numpy() / kp1**j
        if t_obs is not None:
            cv += (
                tab[f"theta_{j}_1_0"].to_numpy() + tab[f"theta_{j}_1_1"].to_numpy() * sr
            ) / (kp1**j * t_obs)
    if t_obs is not None:
        cv += (
            tab["theta_0_2_0"].to_numpy() + tab["theta_0_2_1"].to_numpy() * sr
        ) / t_obs**2
        cv += (
            tab["theta_0_3_0"].to_numpy() + tab["theta_0_3_1"].to_numpy() * sr
        ) / t_obs**3
    return tab["p"].to_numpy(dtype=np.float64) / 10_000.0, cv


def crit_value_bounds_finite(
    case: int,
    k: int,
    t_obs: int,
    sr: int,
    alpha: float,
    stat: Literal["F", "t"] = "F",
) -> tuple[float, float]:
    """CV finis-T (I0, I1) par les surfaces publiées K&S (voie A2).

    Parameters
    ----------
    case, k : int
        Cas déterministe PSS et nombre de régresseurs de long terme.
    t_obs : int
        Taille de l'échantillon d'estimation de l'UECM.
    sr : int
        Nombre de coefficients de court terme de l'UECM (hors
        déterministes et niveaux) : ``(p - 1) + somme(q_j)``.
    alpha : float
        Seuil ; si alpha*10000 n'est pas un quantile tabulé,
        interpolation linéaire entre les deux quantiles adjacents.
    stat : {"F", "t"}
        Statistique (t : cas I/III/V ; quantile GAUCHE alpha).
    """
    case_eff = _check(stat, case, k, t_obs, sr)
    out = []
    for i1 in (False, True):
        p_grid, cv = _cv_grid(stat, case_eff, i1, k, t_obs, sr)
        # la colonne p du fichier K&S est le NIVEAU DE SIGNIFICATION
        # (probabilité de queue : droite pour F, gauche pour t) — pour
        # les deux statistiques, le CV au seuil alpha est interpolé en
        # p = alpha (vérifié contre les ancres PSS/A1/Narayan).
        if not p_grid[0] <= alpha <= p_grid[-1]:
            raise ValueError(f"alpha={alpha} hors de la grille des quantiles tabulés.")
        out.append(float(np.interp(alpha, p_grid, cv)))
    return out[0], out[1]


def pvalue_bounds_finite(
    stat_value: float,
    case: int,
    k: int,
    t_obs: int,
    sr: int,
    df_resid: int,
    stat: Literal["F", "t"] = "F",
) -> tuple[float, float]:
    """p-values finies-T aux deux bornes (MacKinnon 1996, eq. 12).

    Les 9 quantiles tabulés dont le CV prédit est le plus proche de la
    statistique observée sont projetés sur l'échelle de la distribution
    de référence (F(df1, df_resid) pour F, Student(df_resid) pour t) ;
    une régression quadratique locale y = a + b*cv + c*cv^2 est ajustée
    et la p-value est la probabilité de queue de la valeur prédite en
    ``stat_value``. Hors de la grille : p-value bornée à 0 ou 1 (côté
    approprié).

    ``df_resid`` : degrés de liberté résiduels de l'UECM.
    """
    case_eff = _check(stat, case, k, t_obs, sr)
    df1 = k + 1 + (1 if case in (2, 4) else 0)
    out = []
    for i1 in (False, True):
        p_grid, cv = _cv_grid(stat, case_eff, i1, k, t_obs, sr)
        # p = probabilité de queue : F -> cv DÉCROÎT en p (cv[0] est le
        # quantile extrême droit) ; t -> cv CROÎT en p (cv[0] est le
        # quantile extrême gauche).
        if stat == "F":
            if stat_value >= cv[0]:
                out.append(0.0)
                continue
            if stat_value <= cv[-1]:
                out.append(1.0)
                continue
            invtail = sps.f.isf(p_grid, df1, df_resid)
        else:
            if stat_value <= cv[0]:
                out.append(0.0)
                continue
            if stat_value >= cv[-1]:
                out.append(1.0)
                continue
            invtail = sps.t.ppf(p_grid, df_resid)

        center = int(np.argmin(np.abs(cv - stat_value)))
        lo = max(0, center - 4)
        hi = min(len(cv), center + 5)
        x = cv[lo:hi]
        y = invtail[lo:hi]
        design = np.column_stack([np.ones_like(x), x, x**2])
        coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = float(coefs[0] + coefs[1] * stat_value + coefs[2] * stat_value**2)
        if stat == "F":
            out.append(float(sps.f.sf(fitted, df1, df_resid)))
        else:
            out.append(float(sps.t.cdf(fitted, df_resid)))
    return out[0], out[1]
