r"""Bounds test PSS 2001 (spec 10) — le cœur de la bibliothèque.

UECM (forme conditionnelle) :

    Δy_t = det_t + lam*y_{t-1} + sum_j gamma_j x_{j,t-1}
           + sum_i psi_i Δy_{t-i} + sum_j sum_i omega_{j,i} Δx_{j,t-i} + eps_t

Tests (spec 10 §2) :

1. ``F_overall`` : H0 : lam = gamma_1 = ... = gamma_k = 0 — pour les
   cas II et IV, le déterministe restreint (c0, resp. c1) fait PARTIE du
   vecteur testé (k+2 restrictions au lieu de k+1 — spec 10 §3.1,
   vérifié par le test d'équivalence Wald/régression contrainte).
2. ``t_BDM`` : H0 : lam = 0, test UNILATÉRAL GAUCHE exigeant
   lam_hat < 0.

Décision à trois états :
``"cointegration"`` / ``"no_cointegration"`` / ``"inconclusive"`` —
jamais un booléen ; la zone non concluante est celle que les specs 13-16
viennent résorber.

Cas limite q_j = 0 (note spec 03, docs/QUESTIONS.md) : le régresseur
entre dans le vecteur testé via son niveau CONTEMPORAIN x_{j,t} (pas de
terme Δx_{j,t} distinct). x_{j,t} reste I(1) sous H0 — l'identité exacte
x_{j,t} = x_{j,t-1} + Δx_{j,t} (écart stationnaire) laisse
l'asymptotique du test inchangée ; seule la datation diffère d'une
période.

Références
----------
Pesaran, M. H., Shin, Y. & Smith, R. J. (2001). "Bounds Testing
Approaches to the Analysis of Level Relationships", *J. Applied
Econometrics*, 16(3), 289-326. Clé BibTeX : ``pesaran2001bounds``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.linalg
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

from ardlpy.core.ardl import ARDL
from ardlpy.critical_values import get_bounds
from ardlpy.exceptions import ArdlpyMethodologyWarning, DegenerateCaseWarning
from ardlpy.utils import check_series

FloatArray = npt.NDArray[np.float64]

Decision = Literal["cointegration", "no_cointegration", "inconclusive"]

_CASE_DET = {1: "none", 2: "const", 3: "const", 4: "trend", 5: "trend"}
_CASE_RESTRICTED_DET = {2: "const", 4: "trend"}


@dataclass
class _UECMFit:
    """Estimation OLS de l'UECM d'un cas donné (interne)."""

    params: pd.Series
    cov: FloatArray
    resid: FloatArray
    ssr: float
    design: FloatArray
    names: list[str]
    tested: list[str]  # colonnes du vecteur testé par le F
    lam_name: str

    @property
    def nobs(self) -> int:
        return len(self.resid)

    @property
    def df_resid(self) -> int:
        return self.nobs - len(self.names)


def _estimate_uecm(
    y: FloatArray,
    x: FloatArray,
    x_names: tuple[str, ...],
    y_name: str,
    p: int,
    q: tuple[int, ...],
    case: int,
    fixed: FloatArray | None = None,
    fixed_names: tuple[str, ...] = (),
) -> _UECMFit:
    """Construit et estime l'UECM du cas demandé (design direct)."""
    n, k = x.shape
    start = max([p, *q]) if q else p
    if start < 1:
        raise ValueError("p >= 1 requis pour l'UECM (Δy_t et y_{t-1}).")
    dy = np.diff(y)
    dx = np.diff(x, axis=0)

    cols: list[FloatArray] = []
    names: list[str] = []
    tested: list[str] = []

    det = _CASE_DET[case]
    if det in ("const", "trend"):
        cols.append(np.ones(n - start))
        names.append("const")
    if det == "trend":
        cols.append(np.arange(start + 1, n + 1, dtype=np.float64))
        names.append("trend")
    if case in _CASE_RESTRICTED_DET:
        tested.append(_CASE_RESTRICTED_DET[case])

    lam_name = f"{y_name}.L1"
    cols.append(y[start - 1 : n - 1])
    names.append(lam_name)
    tested.append(lam_name)

    for j, name in enumerate(x_names):
        if q[j] == 0:
            # Niveau contemporain (convention q_j=0, docs/QUESTIONS.md) —
            # reste I(1) sous H0, cf. module docstring.
            cols.append(x[start:n, j])
            names.append(f"{name}.L0")
            tested.append(f"{name}.L0")
        else:
            cols.append(x[start - 1 : n - 1, j])
            names.append(f"{name}.L1")
            tested.append(f"{name}.L1")

    for i in range(1, p):
        cols.append(dy[start - i - 1 : n - i - 1])
        names.append(f"D.{y_name}.L{i}")
    for j, name in enumerate(x_names):
        for i in range(q[j]):
            cols.append(dx[start - i - 1 : n - i - 1, j])
            names.append(f"D.{name}.L{i}")

    if fixed is not None:
        # z_t sans retards (ex. dummies) : hors du vecteur testé.
        cols.extend(fixed[start:].T)
        names.extend(fixed_names)

    design = np.column_stack(cols)
    y_dep = dy[start - 1 :]

    coefs, _, rank, _ = np.linalg.lstsq(design, y_dep, rcond=None)
    if rank < design.shape[1]:
        warnings.warn(
            "Design UECM singulier : covariance non fiable.",
            ArdlpyMethodologyWarning,
            stacklevel=3,
        )
    resid = y_dep - design @ coefs
    ssr = float(resid @ resid)

    n_est, k_par = design.shape
    q_mat, r_mat = np.linalg.qr(design)
    r_inv = scipy.linalg.solve_triangular(r_mat, np.eye(k_par))
    xtx_inv = r_inv @ r_inv.T
    cov = (ssr / (n_est - k_par)) * xtx_inv

    return _UECMFit(
        params=pd.Series(coefs, index=names, name="coef"),
        cov=cov.astype(np.float64),
        resid=resid.astype(np.float64),
        ssr=ssr,
        design=design,
        names=names,
        tested=tested,
        lam_name=lam_name,
    )


def _wald_f(fit: _UECMFit) -> float:
    """F de Wald sur les colonnes ``fit.tested`` (variance nonrobuste —
    algébriquement identique au F par SSR restreint/non restreint)."""
    idx = [fit.names.index(name) for name in fit.tested]
    r_vec = fit.params.to_numpy()[idx]
    v_sub = fit.cov[np.ix_(idx, idx)]
    stat = float(r_vec @ np.linalg.solve(v_sub, r_vec)) / len(idx)
    return stat


JointDecision = Literal[
    "cointegration", "no_cointegration", "inconclusive", "degenerate_suspicion"
]


def _joint_decision(
    decision_f: Decision, decision_t: Decision | None
) -> JointDecision | None:
    """Décision jointe F + t (spec 11 §2.3, préparation du cadre SMG).

    La cointégration exige la concordance des DEUX tests (Banerjee-
    Dolado-Mestre 1998 ; Sam-McNown-Goh 2019, spec 15) :

    - F et t rejettent -> ``"cointegration"`` ;
    - F rejette mais pas t -> ``"degenerate_suspicion"`` (dégénérescence
      de type 1 : les gamma seuls portent la relation, pas de force de
      rappel — classification complète : spec 15) ;
    - aucun ne rejette -> ``"no_cointegration"`` ;
    - toute autre discordance -> ``"inconclusive"`` ;
    - t non tabulé (cas II/IV) -> ``None`` (logique jointe indisponible).
    """
    if decision_t is None:
        return None
    if decision_f == "cointegration":
        if decision_t == "cointegration":
            return "cointegration"
        return "degenerate_suspicion"
    if decision_f == "no_cointegration" and decision_t == "no_cointegration":
        return "no_cointegration"
    return "inconclusive"


def _classify(stat: float, lower: float, upper: float, *, left_tail: bool) -> Decision:
    """Décision à trois états."""
    if left_tail:  # t_BDM : rejet si t < borne I(1) (plus négative)
        if stat < upper:
            return "cointegration"
        if stat > lower:
            return "no_cointegration"
    else:
        if stat > upper:
            return "cointegration"
        if stat < lower:
            return "no_cointegration"
    return "inconclusive"


@dataclass
class BoundsTestResults:
    """Résultat du bounds test PSS 2001 (spec 10 §5)."""

    case: int
    k: int
    order: tuple[int, dict[str, int]]
    f_stat: float
    t_stat: float
    alpha: float
    bounds: pd.DataFrame
    decision_f: Decision
    decision_t: Decision | None
    decision_joint: JointDecision | None
    uecm: pd.DataFrame
    cv_source: str
    p_values: pd.Series | None  # (p_I0, p_I1) du F — None si indisponible
    _fit: _UECMFit = field(repr=False)

    def adjustment(self, alpha: float = 0.05) -> pd.Series:
        """Vitesse d'ajustement lambda avec IC conditionnel (spec 11 §2.4).

        L'IC standard sur lambda n'est valide que SOUS cointégration
        établie (distribution non standard sous H0) : si la décision
        jointe n'est pas ``"cointegration"``, les bornes d'IC sont NaN
        et un warning méthodologique est émis (piège connu —
        « ne jamais afficher d'IC sur la vitesse d'ajustement avant
        cointégration établie »). L'estimée ponctuelle et son se restent
        consultables.
        """
        from scipy.stats import norm

        lam = float(self._fit.params[self._fit.lam_name])
        pos = self._fit.names.index(self._fit.lam_name)
        se = float(np.sqrt(self._fit.cov[pos, pos]))
        if self.decision_joint == "cointegration":
            z = float(norm.ppf(1 - alpha / 2))
            ci_lower, ci_upper = lam - z * se, lam + z * se
        else:
            warnings.warn(
                "IC sur lambda masqué : la cointégration n'est pas établie "
                f"(décision jointe : {self.decision_joint}) — l'IC standard "
                "sur la vitesse d'ajustement n'est valide que sous "
                "cointégration (spec 11 §2.4).",
                ArdlpyMethodologyWarning,
                stacklevel=2,
            )
            ci_lower = ci_upper = np.nan
        return pd.Series(
            {"lambda": lam, "se": se, "ci_lower": ci_lower, "ci_upper": ci_upper},
            name="adjustment",
        )

    def diagnostics(self) -> pd.DataFrame:
        """Ljung-Box, Jarque-Bera, Breusch-Pagan sur les résidus UECM
        (CUSUM/CUSUMSQ arrivent avec la spec 26)."""
        resid = self._fit.resid
        lb_lags = max(1, min(10, len(resid) // 5))
        lb = acorr_ljungbox(resid, lags=[lb_lags])
        jb_stat, jb_p, _, _ = jarque_bera(resid)
        bp_design = self._fit.design
        if not (bp_design[:, 0] == 1.0).all():
            bp_design = np.column_stack([np.ones(bp_design.shape[0]), bp_design])
        bp_p = float(het_breuschpagan(resid, bp_design)[1])
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

    def summary(self) -> str:
        """Présentation type publication : stats, p-values aux deux
        bornes (spec 13), bornes aux 3 seuils, décisions (avec lecture
        continue de la zone non concluante)."""
        p, q = self.order
        q_desc = ", ".join(f"{n}:{v}" for n, v in q.items())

        decision_f_txt: str = self.decision_f
        if self.decision_f == "inconclusive" and self.p_values is not None:
            # lecture continue de la zone non concluante (spec 13 §2.1.4)
            decision_f_txt = (
                f"inconclusive, p ∈ [{self.p_values['p_I1']:.4f}, "
                f"{self.p_values['p_I0']:.4f}]"
            )
        p_line = (
            f"p-values F (K&S 2020) : p_I0 = {self.p_values['p_I0']:.4f}, "
            f"p_I1 = {self.p_values['p_I1']:.4f}"
            if self.p_values is not None
            else "p-values F : indisponibles (k hors couverture des surfaces)"
        )

        lines = [
            f"Bounds test PSS 2001 — cas {self.case}, k={self.k}, "
            f"UECM({p}; {q_desc}), cv_source={self.cv_source}",
            "",
            f"F_overall = {self.f_stat:.4f}   décision ({self.alpha:.0%}) : "
            f"{decision_f_txt}",
            p_line,
            f"t_BDM     = {self.t_stat:.4f}   décision ({self.alpha:.0%}) : "
            + (
                self.decision_t
                if self.decision_t is not None
                else f"non tabulé (cas {self.case})"
            ),
            "décision jointe F+t (spec 11) : "
            + (
                self.decision_joint
                if self.decision_joint is not None
                else f"indisponible (cas {self.case}, t non tabulé)"
            ),
            "",
            self.bounds.to_string(float_format=lambda v: f"{v: .3f}"),
        ]
        return "\n".join(lines)


def bounds_test(
    y: npt.ArrayLike,
    x: npt.ArrayLike,
    case: int = 3,
    order: tuple[int, int | dict[str, int]] | None = None,
    ic: Literal["aic", "bic", "hq"] = "aic",
    max_p: int = 4,
    max_q: int = 4,
    alpha: float = 0.05,
    cv_source: Literal["kripfganz", "pss", "narayan"] = "kripfganz",
    fixed_regressors: npt.ArrayLike | None = None,
) -> BoundsTestResults:
    """Bounds test de cointégration PSS 2001 (spec 10 §5 — fonction phare).

    Parameters
    ----------
    y, x : array-like
        Variable dépendante et régresseurs de niveau.
    case : int
        Cas déterministe PSS (1 à 5, spec 10 §3). Cas III le plus commun.
    order : tuple (p, q), optional
        Ordres de l'UECM. Si None, sélection automatique par
        :meth:`ARDL.select_order` (critère ``ic``, bornes ``max_p``,
        ``max_q``) sur la forme ARDL puis transformation.
    alpha : float
        Seuil des bornes utilisé pour la décision (le tableau
        ``bounds`` rapporte tous les seuils disponibles).
    cv_source : {"kripfganz", "pss", "narayan"}
        Source des valeurs critiques (politique : spec 12 §2.4 ;
        hiérarchie : ardlpy.critical_values). "kripfganz" (DÉFAUT
        depuis la spec 13) : surfaces de réponse via statsmodels — CV F
        asymptotiques précis à tout seuil + p-values aux deux bornes ;
        les bornes t restent celles de PSS 2001 (composition
        documentée : le matériel voie A1 ne couvre pas le t ; k=0 non
        couvert). "pss" : valeurs publiées à l'identique (reproduction
        de la littérature). "narayan" : petits échantillons (T = nobs
        de l'UECM, interpolation ; cas II/III/V, F seulement),
        recommandé si 30 <= T <= 80.
    fixed_regressors : array-like, shape (T, m), optional
        Variables z_t sans retards (ex. dummies), hors du vecteur testé
        (non prises en compte par la sélection d'ordre automatique).

    Returns
    -------
    BoundsTestResults
        Statistiques, bornes, décisions à trois états, UECM estimé,
        diagnostics.

    Notes
    -----
    Hypothèses de validité à vérifier par l'utilisateur (spec 10 §1) :
    x faiblement exogènes, pas de cointégration entre les x (spec 07),
    aucune variable I(2) (spec 27), erreurs non autocorrélées (un
    warning est émis automatiquement si Ljung-Box < 5 %).
    """
    if case not in (1, 2, 3, 4, 5):
        raise ValueError(f"case doit être dans 1..5, reçu {case}.")
    if cv_source not in ("kripfganz", "pss", "narayan"):
        raise ValueError(f"cv_source inconnu : {cv_source!r}.")

    y_arr, x_arr, _, y_name, x_names = check_series(y, x)
    if x_arr is None:
        raise ValueError("bounds_test requiert des régresseurs x.")
    k = x_arr.shape[1]

    if order is None:
        det = _CASE_DET[case]
        sel_det: Literal["const", "trend"] = "trend" if det == "trend" else "const"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ArdlpyMethodologyWarning)
            sel = ARDL.select_order(y, x, max_p=max_p, max_q=max_q, ic=ic, det=sel_det)
        p, q_dict = sel.best_order
    else:
        from ardlpy.core.ardl import _parse_order

        p, q_dict = _parse_order(order, x_names)
    q = tuple(q_dict[name] for name in x_names)

    fixed_arr: FloatArray | None = None
    fixed_names: tuple[str, ...] = ()
    if fixed_regressors is not None:
        fixed_arr = np.asarray(fixed_regressors, dtype=np.float64)
        if fixed_arr.ndim == 1:
            fixed_arr = fixed_arr[:, None]
        if fixed_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("fixed_regressors : longueur incompatible avec y.")
        if isinstance(fixed_regressors, pd.DataFrame):
            fixed_names = tuple(str(c) for c in fixed_regressors.columns)
        else:
            fixed_names = tuple(f"z.{j}" for j in range(fixed_arr.shape[1]))

    fit = _estimate_uecm(
        y_arr, x_arr, x_names, y_name, p, q, case, fixed_arr, fixed_names
    )

    f_stat = _wald_f(fit)
    lam_hat = float(fit.params[fit.lam_name])
    se_lam = float(
        np.sqrt(fit.cov[fit.names.index(fit.lam_name), fit.names.index(fit.lam_name)])
    )
    t_stat = lam_hat / se_lam

    # bornes à tous les seuils disponibles (F : source choisie ; t : tables
    # PSS uniquement — Narayan ne publie pas de bornes t, cf. plus bas)
    rows = []
    for a in (0.10, 0.05, 0.01):
        f_lo, f_up = get_bounds(
            "F", case=case, k=k, alpha=a, cv_source=cv_source, t_obs=fit.nobs
        )
        try:
            t_lo, t_up = get_bounds("t", case=case, k=k, alpha=a)
        except ValueError:
            t_lo = t_up = np.nan
        rows.append(
            {"alpha": a, "F_I0": f_lo, "F_I1": f_up, "t_I0": t_lo, "t_I1": t_up}
        )
    bounds_df = pd.DataFrame(rows).set_index("alpha")

    f_lo, f_up = get_bounds(
        "F", case=case, k=k, alpha=alpha, cv_source=cv_source, t_obs=fit.nobs
    )
    decision_f = _classify(f_stat, f_lo, f_up, left_tail=False)

    decision_t: Decision | None
    if cv_source == "narayan" and case in (3, 5):
        decision_t = None
        warnings.warn(
            "Narayan 2005 ne publie pas de bornes t : décision t "
            'indisponible avec cv_source="narayan" — le t_stat est '
            'rapporté ; utiliser cv_source="pss" pour une décision t '
            "asymptotique (en petit échantillon elle serait trop "
            "libérale, spec 12 §1).",
            ArdlpyMethodologyWarning,
            stacklevel=2,
        )
    elif case in (1, 3, 5):
        t_lo, t_up = get_bounds("t", case=case, k=k, alpha=alpha)
        decision_t = _classify(t_stat, t_lo, t_up, left_tail=True)
        if lam_hat >= 0:
            warnings.warn(
                f"lambda_hat = {lam_hat:.4f} >= 0 : pas de force de rappel "
                "vers l'équilibre ; le t_BDM (unilatéral GAUCHE) n'a pas "
                "d'interprétation de cointégration.",
                DegenerateCaseWarning,
                stacklevel=2,
            )
            decision_t = "no_cointegration"
    else:
        decision_t = None
        warnings.warn(
            f"Cas {case} : PSS 2001 ne tabule pas le t_BDM pour les cas à "
            "déterministes restreints — décision t indisponible, utiliser "
            "le F_overall.",
            ArdlpyMethodologyWarning,
            stacklevel=2,
        )

    decision_joint = _joint_decision(decision_f, decision_t)
    if decision_joint == "degenerate_suspicion":
        warnings.warn(
            "F rejette mais pas t : suspicion de dégénérescence de type 1 "
            "(les gamma seuls portent la relation de niveaux, pas de force "
            "de rappel en y) — la cointégration n'est PAS établie ; le "
            "cadre à 3 tests de Sam-McNown-Goh 2019 (spec 15) classifie "
            "formellement ce cas.",
            ArdlpyMethodologyWarning,
            stacklevel=2,
        )

    # garde-fou d'autocorrélation (spec 09 §2.2)
    lb_lags = max(1, min(10, fit.nobs // 5))
    lb_p = float(acorr_ljungbox(fit.resid, lags=[lb_lags])["lb_pvalue"].iloc[0])
    if lb_p < 0.05:
        warnings.warn(
            f"Erreurs autocorrélées (Ljung-Box p={lb_p:.4f} < 0.05) : le "
            "bounds test n'est pas fiable ; augmenter p/q (Pesaran-Shin "
            "1998, spec 09 §2.2).",
            ArdlpyMethodologyWarning,
            stacklevel=2,
        )

    # p-values approchées du F aux deux bornes (spec 13, surfaces K&S)
    p_values: pd.Series | None
    if 1 <= k <= 10:
        from ardlpy.critical_values import pvalue_bounds

        p_i0, p_i1 = pvalue_bounds(f_stat, case=case, k=k)
        p_values = pd.Series({"p_I0": p_i0, "p_I1": p_i1}, name="F_pvalues")
    else:
        p_values = None  # k hors couverture des surfaces (k = 0)

    se = np.sqrt(np.diag(fit.cov))
    uecm_table = pd.DataFrame({"coef": fit.params, "se": se, "t": fit.params / se})

    return BoundsTestResults(
        case=case,
        k=k,
        order=(p, q_dict),
        f_stat=f_stat,
        t_stat=t_stat,
        alpha=alpha,
        bounds=bounds_df,
        decision_f=decision_f,
        decision_t=decision_t,
        decision_joint=decision_joint,
        uecm=uecm_table,
        cv_source=cv_source,
        p_values=p_values,
        _fit=fit,
    )
