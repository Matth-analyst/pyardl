r"""Estimateur ARDL(p, q_1, ..., q_k) général (spec 05 — Hendry, Pagan & Sargan 1984).

Modèle :

    y_t = det_t + sum_{i=1}^{p} phi_i y_{t-i}
          + sum_j sum_{i=0}^{q_j} beta_{j,i} x_{j,t-i} + eps_t

L'ARDL est la forme mère dont dérivent statique, différences, Koyck,
FDL/Almon, ECM et autorégressif pur : un seul moteur d'estimation,
plusieurs vues (``.to_ecm()``, ``.longrun``, ``.adjustment`` — spec 03).

Conventions numériques (spec 05 §6.5, concordance statsmodels) :

- ordre des colonnes du design : const, trend, y.L1..y.Lp,
  x_j.L0..x_j.Lq_j, régresseurs fixes — identique à
  ``statsmodels.tsa.ardl.ARDL`` (concordance des coefficients vérifiée
  à 1e-10 en test) ;
- ``nobs`` = taille de l'échantillon d'estimation réel (T - hold_back).
  Attention : ``statsmodels.tsa.ardl.ARDL`` rapporte ``nobs = T - p``
  et calcule llf/IC dessus, même quand max(q_j) > p ; ardlpy utilise
  l'échantillon d'estimation réel partout (llf, IC, sigma2), condition
  de comparabilité des critères dans ``select_order``. Les deux
  conventions coïncident dès que p >= max(q_j).

Références
----------
Hendry, D. F., Pagan, A. R. & Sargan, J. D. (1984). "Dynamic
Specification", *Handbook of Econometrics*, vol. 2, ch. 18.
Clé BibTeX : ``hendry1984dynamic``.
Pesaran, M. H. & Shin, Y. (1998) — garde-fou d'autocorrélation (§2.2).
Clé BibTeX : ``pesaran1998ardl``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

from ardlpy.core.transforms import (
    ARDLParams,
    ECMParams,
    ardl_to_ecm,
    half_life,
    longrun_coefs,
    longrun_covariance,
    speed_of_adjustment,
)
from ardlpy.exceptions import ArdlpyMethodologyWarning
from ardlpy.utils import check_series, lag_matrix

FloatArray = npt.NDArray[np.float64]

DetType = Literal["none", "const", "trend"]
CovType = Literal["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"]


def _parse_order(
    order: tuple[int, int | dict[str, int]] | int,
    x_names: tuple[str, ...],
) -> tuple[int, dict[str, int]]:
    """Normalise ``order`` en (p, {nom: q_j})."""
    if isinstance(order, int):
        if x_names:
            raise ValueError(
                "order entier réservé au cas sans régresseurs (AR pur) ; "
                "avec x, passer order=(p, q)."
            )
        p, q_spec = order, {}
    else:
        p, q_raw = order
        if isinstance(q_raw, dict):
            unknown = set(q_raw) - set(x_names)
            if unknown:
                raise ValueError(f"Noms inconnus dans order : {sorted(unknown)}")
            q_spec = {name: int(q_raw[name]) for name in x_names}
        else:
            q_spec = {name: int(q_raw) for name in x_names}
    if p < 0:
        raise ValueError("p doit être >= 0.")
    for name, qj in q_spec.items():
        if qj < 0:
            raise ValueError(f"q[{name}] doit être >= 0.")
    return int(p), q_spec


class ARDL:
    """Modèle ARDL(p, q_1, ..., q_k) estimé par OLS (spec 05).

    Parameters
    ----------
    y : array-like, shape (T,)
        Variable dépendante.
    x : array-like, shape (T, k), optional
        Régresseurs à retards distribués (DataFrame recommandé pour les
        noms). ``None`` -> AR(p) pur.
    order : tuple (p, q)
        p : nombre de retards de y (>= 0 ; p=0 -> FDL sans dynamique).
        q : int (même ordre pour tous les x) ou dict {nom: q_j}.
    det : {"none", "const", "trend"}
        Déterministes : rien, constante, ou constante + tendance
        linéaire (« trend » inclut toujours la constante).
    seasonal : bool
        Non implémenté (conventions saisonnières : spec 04, phase 2).
    fixed_regressors : array-like, shape (T, m), optional
        Variables z_t incluses sans retards (ex. dummies).
    hold_back : int, optional
        Nombre d'observations initiales exclues de l'estimation
        (>= max(p, max q_j)). Sert à imposer un échantillon commun
        entre candidats (spec 05 §3.2).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> x = pd.DataFrame({"x": rng.normal(size=100).cumsum()})
    >>> y = pd.Series(rng.normal(size=100), name="y") + 0.5 * x["x"]
    >>> res = ARDL(y, x, order=(1, 1)).fit()
    >>> res.params.index.tolist()
    ['const', 'y.L1', 'x.L0', 'x.L1']
    """

    def __init__(
        self,
        y: npt.ArrayLike,
        x: npt.ArrayLike | None = None,
        order: tuple[int, int | dict[str, int]] | int = (1, 0),
        det: DetType = "const",
        seasonal: bool = False,
        fixed_regressors: npt.ArrayLike | None = None,
        hold_back: int | None = None,
    ) -> None:
        if seasonal:
            raise NotImplementedError(
                "seasonal=True sera implémenté avec les conventions "
                "saisonnières de la spec 04 (phase 2)."
            )
        if det not in ("none", "const", "trend"):
            raise ValueError('det doit être "none", "const" ou "trend".')

        y_arr, x_arr, index, y_name, x_names = check_series(y, x)
        self._y = y_arr
        self._x = x_arr
        self._index = index
        self._y_name = y_name
        self._x_names = x_names
        self.det: DetType = det

        self.p, self._q = _parse_order(order, x_names)
        self.q: tuple[int, ...] = tuple(self._q[name] for name in x_names)

        self._fixed: FloatArray | None = None
        self._fixed_names: tuple[str, ...] = ()
        if fixed_regressors is not None:
            fixed = np.asarray(fixed_regressors, dtype=np.float64)
            if fixed.ndim == 1:
                fixed = fixed[:, None]
            if fixed.shape[0] != y_arr.shape[0]:
                raise ValueError("fixed_regressors : longueur incompatible avec y.")
            if isinstance(fixed_regressors, pd.DataFrame):
                self._fixed_names = tuple(str(c) for c in fixed_regressors.columns)
            else:
                self._fixed_names = tuple(f"z.{j}" for j in range(fixed.shape[1]))
            self._fixed = fixed

        start_required = max([self.p, *self.q]) if self.q else self.p
        if hold_back is None:
            hold_back = start_required
        elif hold_back < start_required:
            raise ValueError(f"hold_back={hold_back} < max(p, max q)={start_required}.")
        self.hold_back = int(hold_back)

        n_est = y_arr.shape[0] - self.hold_back
        n_params = (
            (0 if det == "none" else 1)
            + (1 if det == "trend" else 0)
            + self.p
            + sum(qj + 1 for qj in self.q)
            + len(self._fixed_names)
        )
        if n_est <= n_params:
            raise ValueError(
                f"Pas assez d'observations : n_est={n_est} <= n_params={n_params}."
            )

    # ------------------------------------------------------------------
    # Construction du design (ordre des colonnes = statsmodels, cf. module)
    # ------------------------------------------------------------------
    def _build_design(self) -> tuple[FloatArray, FloatArray, list[str]]:
        y, x = self._y, self._x
        n = y.shape[0]
        hb = self.hold_back
        cols: list[FloatArray] = []
        names: list[str] = []

        if self.det in ("const", "trend"):
            cols.append(np.ones(n - hb))
            names.append("const")
        if self.det == "trend":
            cols.append(np.arange(hb + 1, n + 1, dtype=np.float64))
            names.append("trend")

        if self.p > 0:
            y_lags = lag_matrix(y, hb, first_lag=1)[:, : self.p]
            cols.extend(y_lags.T)
            names.extend(f"{self._y_name}.L{i}" for i in range(1, self.p + 1))

        if x is not None:
            for j, name in enumerate(self._x_names):
                x_lags = lag_matrix(x[:, j], hb, first_lag=0)[:, : self.q[j] + 1]
                cols.extend(x_lags.T)
                names.extend(f"{name}.L{i}" for i in range(self.q[j] + 1))

        if self._fixed is not None:
            cols.extend(self._fixed[hb:].T)
            names.extend(self._fixed_names)

        design = np.column_stack(cols)
        y_dep = y[hb:]
        return design, y_dep, names

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def fit(
        self,
        cov_type: CovType = "nonrobust",
        cov_kwds: dict[str, int] | None = None,
    ) -> ARDLResults:
        """Estime le modèle par OLS et exécute le garde-fou
        d'autocorrélation de Pesaran-Shin 1998 (spec 09 §2.2)."""
        results = self._fit(cov_type=cov_type, cov_kwds=cov_kwds)
        lb_p = results._ljungbox_pvalue()
        if lb_p < 0.05:
            warnings.warn(
                f"Erreurs autocorrélées (Ljung-Box p={lb_p:.4f} < 0.05) : "
                "l'inférence de long terme n'est pas fiable ; augmenter "
                "p/q ou revoir la spécification (Pesaran-Shin 1998, "
                "spec 09 §2.2).",
                ArdlpyMethodologyWarning,
                stacklevel=2,
            )
        return results

    def _fit(
        self,
        cov_type: CovType = "nonrobust",
        cov_kwds: dict[str, int] | None = None,
    ) -> ARDLResults:
        """Estimation sans garde-fou (usage interne : select_order, gets)."""
        design, y_dep, names = self._build_design()
        n_est, k = design.shape

        coefs, _, rank, _ = np.linalg.lstsq(design, y_dep, rcond=None)
        if rank < k:
            warnings.warn(
                "Design singulier (colinéarité parfaite) : coefficients "
                "de norme minimale via lstsq, covariance non fiable.",
                ArdlpyMethodologyWarning,
                stacklevel=3,
            )
        resid = y_dep - design @ coefs
        ssr = float(resid @ resid)

        # inv(X'X) via QR (jamais inv(X.T @ X) — règle du projet)
        q_mat, r_mat = np.linalg.qr(design)
        r_inv = scipy.linalg.solve_triangular(r_mat, np.eye(k))
        xtx_inv = r_inv @ r_inv.T

        df_resid = n_est - k
        scale = ssr / df_resid
        if cov_type == "nonrobust":
            cov = scale * xtx_inv
        elif cov_type in ("HC0", "HC1", "HC2", "HC3"):
            u2 = resid**2
            if cov_type == "HC1":
                u2 = u2 * n_est / df_resid
            elif cov_type in ("HC2", "HC3"):
                leverage = np.sum(q_mat**2, axis=1)
                power = 1 if cov_type == "HC2" else 2
                u2 = u2 / (1.0 - leverage) ** power
            meat = (design * u2[:, None]).T @ design
            cov = xtx_inv @ meat @ xtx_inv
        elif cov_type == "HAC":
            nlags = (cov_kwds or {}).get(
                "nlags", int(np.floor(4 * (n_est / 100.0) ** (2.0 / 9.0)))
            )
            xu = design * resid[:, None]
            meat = (xu.T @ xu).astype(np.float64)
            for lag in range(1, nlags + 1):
                w = 1.0 - lag / (nlags + 1.0)
                gamma = xu[lag:].T @ xu[:-lag]
                meat += w * (gamma + gamma.T)
            cov = xtx_inv @ meat @ xtx_inv
        else:
            raise ValueError(f"cov_type inconnu : {cov_type!r}")

        return ARDLResults(
            model=self,
            _params=coefs.astype(np.float64),
            _cov_params=cov.astype(np.float64),
            _param_names=names,
            _resid=resid.astype(np.float64),
            _ssr=ssr,
            cov_type=cov_type,
        )

    # ------------------------------------------------------------------
    # Sélection d'ordre (spec 05 §3)
    # ------------------------------------------------------------------
    @staticmethod
    def select_order(
        y: npt.ArrayLike,
        x: npt.ArrayLike,
        max_p: int,
        max_q: int,
        ic: Literal["aic", "bic", "hq"] = "aic",
        search: Literal["grid", "per_variable"] = "grid",
        det: DetType = "const",
        min_p: int = 1,
    ) -> ARDLOrderSelection:
        """Sélection d'ordre par critère d'information (spec 05 §3).

        Tous les candidats sont estimés sur l'échantillon COMMUN
        t = max(max_p, max_q)+1..T (hold_back fixé — spec 05 §3.2,
        piège de la spec 02 §4), sinon les critères ne sont pas
        comparables. Le meilleur modèle est ensuite ré-estimé sur
        l'échantillon maximal de son ordre (§3.4).

        Parameters
        ----------
        y, x : array-like
            Données (x obligatoire ici ; l'AR pur a d'autres outils).
        max_p, max_q : int
            Bornes de la grille : p ∈ min_p..max_p, q_j ∈ 0..max_q.
        ic : {"aic", "bic", "hq"}
            Critère de sélection (le tableau rapporte les trois).
        search : {"grid", "per_variable"}
            "grid" : produit cartésien complet. "per_variable" :
            optimisation séquentielle de p puis de chaque q_j (à la
            statsmodels/EViews), pour k > 3 où la grille explose.
        det : déterministes (voir :class:`ARDL`).
        min_p : int
            Borne inférieure de p (défaut 1, cf. spec 05 §3.1).
        """
        if ic not in ("aic", "bic", "hq"):
            raise ValueError('ic doit être "aic", "bic" ou "hq".')
        _, x_arr, _, _, x_names = check_series(y, x)
        if x_arr is None:
            raise ValueError("select_order requiert des régresseurs x.")
        k = x_arr.shape[1]
        hold_back = max(max_p, max_q)

        def eval_candidate(p: int, q_tuple: tuple[int, ...]) -> dict[str, float]:
            q_dict = dict(zip(x_names, q_tuple, strict=True))
            res = ARDL(y, x, order=(p, q_dict), det=det, hold_back=hold_back)._fit()
            row: dict[str, float] = {"p": p}
            for name, qj in q_dict.items():
                row[f"q_{name}"] = qj
            row.update(
                aic=res.aic, bic=res.bic, hq=res.hqic, llf=res.llf, nobs=res.nobs
            )
            return row

        rows: list[dict[str, float]] = []
        if search == "grid":
            for p in range(min_p, max_p + 1):
                for q_tuple in product(range(max_q + 1), repeat=k):
                    rows.append(eval_candidate(p, q_tuple))
        elif search == "per_variable":
            current_p = max_p
            current_q = [max_q] * k
            seen: set[tuple[int, ...]] = set()

            def eval_and_log(p: int, q_t: tuple[int, ...]) -> float:
                key = (p, *q_t)
                row = eval_candidate(p, q_t)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
                return row[ic]

            for _ in range(10):  # itérer jusqu'à stabilité
                changed = False
                best_p = min(
                    range(min_p, max_p + 1),
                    key=lambda p: eval_and_log(p, tuple(current_q)),
                )
                if best_p != current_p:
                    current_p, changed = best_p, True
                for j in range(k):

                    def ic_for(qj: int, j: int = j, p: int = current_p) -> float:
                        trial = current_q.copy()
                        trial[j] = qj
                        return eval_and_log(p, tuple(trial))

                    best_qj = min(range(max_q + 1), key=ic_for)
                    if best_qj != current_q[j]:
                        current_q[j], changed = best_qj, True
                if not changed:
                    break
        else:
            raise ValueError('search doit être "grid" ou "per_variable".')

        table = pd.DataFrame(rows).sort_values(ic, kind="stable").reset_index(drop=True)
        best = table.iloc[0]
        best_p = int(best["p"])
        best_q = {name: int(best[f"q_{name}"]) for name in x_names}

        best_model = ARDL(y, x, order=(best_p, best_q), det=det).fit()
        return ARDLOrderSelection(
            table=table, ic=ic, best_order=(best_p, best_q), best_model=best_model
        )

    # ------------------------------------------------------------------
    # GETS (spec 05 §4)
    # ------------------------------------------------------------------
    @staticmethod
    def gets(
        y: npt.ArrayLike,
        x: npt.ArrayLike,
        max_p: int,
        max_q: int,
        alpha: float = 0.05,
        det: DetType = "const",
    ) -> GETSResults:
        """Réduction general-to-specific (spec 05 §4).

        Part de (max_p, max_q) et réduit itérativement l'ordre du retard
        terminal le moins significatif tant que : (a) sa p-value > alpha,
        (b) les diagnostics restent propres (Ljung-Box,
        Breusch-Pagan > 0.05), (c) le F des restrictions cumulées vs le
        modèle général ne rejette pas (> alpha). Le chemin complet est
        journalisé dans ``.reduction_path``.

        La réduction préserve la structure contiguë des retards (on ne
        supprime que le retard TERMINAL de chaque variable) — cf.
        docs/QUESTIONS.md, entrée spec 05 §4.
        """
        _, x_arr, _, y_name, x_names = check_series(y, x)
        if x_arr is None:
            raise ValueError("gets requiert des régresseurs x.")
        hold_back = max(max_p, max_q)

        def fit_cand(p: int, q_list: list[int]) -> ARDLResults:
            q_dict = dict(zip(x_names, q_list, strict=True))
            return ARDL(y, x, order=(p, q_dict), det=det, hold_back=hold_back)._fit()

        general = fit_cand(max_p, [max_q] * len(x_names))
        current_p, current_q = max_p, [max_q] * len(x_names)
        current = general
        path: list[dict[str, object]] = []

        while True:
            # retards terminaux candidats à l'élimination
            candidates: list[tuple[str, float]] = []
            pvals = current.pvalues
            if current_p >= 1:
                candidates.append(
                    (f"{y_name}.L{current_p}", float(pvals[f"{y_name}.L{current_p}"]))
                )
            for j, name in enumerate(x_names):
                if current_q[j] >= 1:
                    candidates.append(
                        (
                            f"{name}.L{current_q[j]}",
                            float(pvals[f"{name}.L{current_q[j]}"]),
                        )
                    )
            if not candidates:
                break
            drop_name, drop_p = max(candidates, key=lambda c: c[1])
            if drop_p <= alpha:
                break

            # réduction tentée
            trial_p, trial_q = current_p, current_q.copy()
            if drop_name.startswith(f"{y_name}.L"):
                trial_p -= 1
            else:
                var = drop_name.rsplit(".L", 1)[0]
                trial_q[x_names.index(var)] -= 1
            trial = fit_cand(trial_p, trial_q)

            lb_p = trial._ljungbox_pvalue()
            bp_p = trial._breuschpagan_pvalue()
            f_p = _f_test_nested(general, trial)
            ok = lb_p > 0.05 and bp_p > 0.05 and f_p > alpha
            path.append(
                {
                    "dropped": drop_name,
                    "pvalue": drop_p,
                    "ljungbox_p": lb_p,
                    "breuschpagan_p": bp_p,
                    "cumulative_f_p": f_p,
                    "accepted": ok,
                    "aic": trial.aic,
                }
            )
            if not ok:
                break
            current_p, current_q, current = trial_p, trial_q, trial

        final_q = dict(zip(x_names, current_q, strict=True))
        final = ARDL(y, x, order=(current_p, final_q), det=det).fit()
        return GETSResults(
            final_model=final,
            final_order=(current_p, final_q),
            reduction_path=pd.DataFrame(
                path,
                columns=[
                    "dropped",
                    "pvalue",
                    "ljungbox_p",
                    "breuschpagan_p",
                    "cumulative_f_p",
                    "accepted",
                    "aic",
                ],
            ),
            general_model=general,
        )


def _f_test_nested(general: ARDLResults, restricted: ARDLResults) -> float:
    """p-value du F des restrictions cumulées (même échantillon requis)."""
    if restricted.nobs != general.nobs:
        raise ValueError("F imbriqué : échantillons différents.")
    n_restr = len(general.params) - len(restricted.params)
    if n_restr == 0:
        return 1.0
    df2 = general.nobs - len(general.params)
    f_stat = ((restricted.ssr - general.ssr) / n_restr) / (general.ssr / df2)
    from scipy.stats import f as f_dist

    return float(f_dist.sf(f_stat, n_restr, df2))


@dataclass(frozen=True)
class ARDLOrderSelection:
    """Résultat de :meth:`ARDL.select_order` (spec 05 §3.3-3.4)."""

    table: pd.DataFrame
    ic: str
    best_order: tuple[int, dict[str, int]]
    best_model: ARDLResults

    def top(self, n: int = 5) -> pd.DataFrame:
        """Top-N des candidats (robustesse à la Pesaran, spec 05 §3.3)."""
        return self.table.head(n)


@dataclass(frozen=True)
class GETSResults:
    """Résultat de :meth:`ARDL.gets` (spec 05 §4)."""

    final_model: ARDLResults
    final_order: tuple[int, dict[str, int]]
    reduction_path: pd.DataFrame
    general_model: ARDLResults


@dataclass
class ARDLResults:
    """Résultats d'estimation ARDL (immuable ; spec 05 §2.3).

    Toutes les vues de long terme (``to_ecm``, ``longrun``,
    ``adjustment``) consomment l'algèbre de la spec 03 via
    :attr:`ardl_params` — aucune conversion manuelle nécessaire.
    """

    model: ARDL
    _params: FloatArray
    _cov_params: FloatArray
    _param_names: list[str]
    _resid: FloatArray
    _ssr: float
    cov_type: str
    _cache: dict[str, object] = field(default_factory=dict, repr=False)

    # -------------------------- statistiques de base ------------------
    @property
    def params(self) -> pd.Series:
        return pd.Series(self._params, index=self._param_names, name="coef")

    @property
    def cov_params_matrix(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._cov_params, index=self._param_names, columns=self._param_names
        )

    @property
    def bse(self) -> pd.Series:
        return pd.Series(
            np.sqrt(np.diag(self._cov_params)), index=self._param_names, name="se"
        )

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        from scipy.stats import t as t_dist

        df = self.nobs - len(self._params)
        return pd.Series(
            2 * t_dist.sf(np.abs(self.tvalues), df),
            index=self._param_names,
            name="pvalue",
        )

    @property
    def resid(self) -> pd.Series:
        index = (
            self.model._index[self.model.hold_back :]
            if self.model._index is not None
            else pd.RangeIndex(self.model.hold_back, len(self.model._y))
        )
        return pd.Series(self._resid, index=index, name="resid")

    @property
    def fittedvalues(self) -> pd.Series:
        return pd.Series(
            self.model._y[self.model.hold_back :] - self._resid,
            index=self.resid.index,
            name="fitted",
        )

    @property
    def nobs(self) -> int:
        """Taille de l'échantillon d'estimation réel (voir module :
        diffère de statsmodels quand max(q) > p)."""
        return len(self._resid)

    @property
    def ssr(self) -> float:
        return self._ssr

    @property
    def sigma2(self) -> float:
        """Variance ML des erreurs : SSR / nobs."""
        return self._ssr / self.nobs

    @property
    def llf(self) -> float:
        return float(-self.nobs / 2 * (np.log(2 * np.pi * self.sigma2) + 1))

    @property
    def _k_ic(self) -> int:
        return len(self._params) + 1  # + sigma2, convention statsmodels

    @property
    def aic(self) -> float:
        return -2 * self.llf + 2 * self._k_ic

    @property
    def bic(self) -> float:
        return -2 * self.llf + float(np.log(self.nobs)) * self._k_ic

    @property
    def hqic(self) -> float:
        return -2 * self.llf + 2 * self._k_ic * float(np.log(np.log(self.nobs)))

    @property
    def rsquared(self) -> float:
        y_dep = self.model._y[self.model.hold_back :]
        tss = float(np.sum((y_dep - y_dep.mean()) ** 2))
        return 1.0 - self._ssr / tss

    @property
    def rsquared_adj(self) -> float:
        k = len(self._params)
        return 1.0 - (1.0 - self.rsquared) * (self.nobs - 1) / (self.nobs - k)

    # -------------------------- stabilité dynamique -------------------
    @property
    def ar_roots(self) -> npt.NDArray[np.complex128]:
        """Racines du polynôme 1 - phi_1 L - ... - phi_p L^p (spec 05 §2.4)."""
        phi = self._phi_values()
        if phi.shape[0] == 0:
            return np.array([], dtype=np.complex128)
        return np.roots(np.concatenate(([1.0], -phi))[::-1]).astype(np.complex128)

    @property
    def is_stable(self) -> bool:
        """True si toutes les racines AR sont hors du cercle unité."""
        roots = self.ar_roots
        if roots.shape[0] == 0:
            return True
        stable = bool(np.all(np.abs(roots) > 1.0))
        if not stable:
            warnings.warn(
                "Dynamique instable : au moins une racine du polynôme AR "
                "est sur ou dans le cercle unité ; les quantités de long "
                "terme n'ont pas d'interprétation d'équilibre.",
                ArdlpyMethodologyWarning,
                stacklevel=2,
            )
        return stable

    # -------------------------- pont vers la spec 03 ------------------
    def _phi_values(self) -> FloatArray:
        p = self.model.p
        y_name = self.model._y_name
        if p == 0:
            return np.array([], dtype=np.float64)
        return np.array([self.params[f"{y_name}.L{i}"] for i in range(1, p + 1)])

    @property
    def ardl_params(self) -> ARDLParams:
        """Conteneur spec 03, directement consommable par ``ardl_to_ecm``
        et par toutes les fonctions de long terme, ``cov_params`` inclus."""
        model = self.model
        if model.p == 0:
            raise ValueError(
                "p=0 (pas de y retardé) : la forme ECM n'existe pas — modèle FDL pur."
            )
        if model._fixed is not None:
            raise NotImplementedError(
                "ardl_params avec fixed_regressors : mapping non défini "
                "par la spec 03 (les z_t ne sont ni des phi ni des beta)."
            )
        beta = []
        pos = (
            (1 if model.det in ("const", "trend") else 0)
            + (1 if model.det == "trend" else 0)
            + model.p
        )
        for qj in model.q:
            beta.append(self._params[pos : pos + qj + 1])
            pos += qj + 1
        return ARDLParams(
            p=model.p,
            q=model.q,
            phi=self._phi_values(),
            beta=tuple(beta),
            const=(
                float(self.params["const"]) if model.det in ("const", "trend") else 0.0
            ),
            trend=(float(self.params["trend"]) if model.det == "trend" else 0.0),
            has_const=model.det in ("const", "trend"),
            has_trend=model.det == "trend",
            x_names=model._x_names,
            cov_params=self._cov_params,
        )

    def to_ecm(self) -> ECMParams:
        """Vue ECM (reparamétrisation exacte, spec 03) — mêmes données,
        mêmes résidus."""
        return ardl_to_ecm(self.ardl_params)

    @property
    def longrun(self) -> pd.DataFrame:
        """Coefficients de long terme theta_j avec se (delta, spec 03/09)."""
        params = self.ardl_params
        theta = longrun_coefs(params)
        cov_theta = longrun_covariance(params)
        return pd.DataFrame(
            {"theta": theta, "se": np.sqrt(np.diag(cov_theta))}, index=theta.index
        )

    @property
    def adjustment(self) -> pd.Series:
        """Vitesse d'ajustement lambda, se et demi-vie (spec 03 §5)."""
        params = self.ardl_params
        lam = speed_of_adjustment(params)
        # var(lam) = 1' V_phi 1 (lam = -1 + somme des phi)
        p = self.model.p
        n_lead = (1 if self.model.det in ("const", "trend") else 0) + (
            1 if self.model.det == "trend" else 0
        )
        v_phi = self._cov_params[n_lead : n_lead + p, n_lead : n_lead + p]
        se_lam = float(np.sqrt(np.ones(p) @ v_phi @ np.ones(p)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hl = half_life(params)
        return pd.Series(
            {"lambda": lam, "se": se_lam, "half_life": hl}, name="adjustment"
        )

    # -------------------------- diagnostics ---------------------------
    def _ljungbox_lags(self) -> int:
        return max(1, min(10, self.nobs // 5))

    def _ljungbox_pvalue(self) -> float:
        lb = acorr_ljungbox(self._resid, lags=[self._ljungbox_lags()])
        return float(lb["lb_pvalue"].iloc[0])

    def _breuschpagan_pvalue(self) -> float:
        design, _, _ = self.model._build_design()
        if not (design[:, 0] == 1.0).all():
            design = np.column_stack([np.ones(design.shape[0]), design])
        return float(het_breuschpagan(self._resid, design)[1])

    def diagnostics(self) -> pd.DataFrame:
        """Ljung-Box, Jarque-Bera, Breusch-Pagan sur les résidus."""
        lb_lags = self._ljungbox_lags()
        lb = acorr_ljungbox(self._resid, lags=[lb_lags])
        jb_stat, jb_p, _, _ = jarque_bera(self._resid)
        bp_p = self._breuschpagan_pvalue()
        return pd.DataFrame(
            {
                "statistic": [
                    float(lb["lb_stat"].iloc[0]),
                    float(jb_stat),
                    np.nan,
                ],
                "pvalue": [float(lb["lb_pvalue"].iloc[0]), float(jb_p), bp_p],
            },
            index=[f"Ljung-Box({lb_lags})", "Jarque-Bera", "Breusch-Pagan"],
        )

    # -------------------------- présentation --------------------------
    def summary(self) -> str:
        """Tableau de résultats type publication."""
        q_desc = ", ".join(
            f"{name}:{qj}"
            for name, qj in zip(self.model._x_names, self.model.q, strict=True)
        )
        header = (
            f"ARDL({self.model.p}; {q_desc}) — det={self.model.det}, "
            f"cov={self.cov_type}\n"
            f"nobs={self.nobs}, R2={self.rsquared:.4f}, "
            f"R2_adj={self.rsquared_adj:.4f}\n"
            f"llf={self.llf:.4f}, AIC={self.aic:.4f}, BIC={self.bic:.4f}, "
            f"HQIC={self.hqic:.4f}\n"
            f"stable={self.is_stable}\n"
        )
        table = pd.DataFrame(
            {
                "coef": self.params,
                "se": self.bse,
                "t": self.tvalues,
                "P>|t|": self.pvalues,
            }
        )
        return header + str(table.to_string(float_format=lambda v: f"{v: .6f}"))
