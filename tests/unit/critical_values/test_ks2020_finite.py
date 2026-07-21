"""Spec 13 §3.1 (voie A2) — surfaces de réponse K&S finies-T.

Validation CONTRACTUELLE à 1e-3 (mêmes coefficients publiés) contre la
sortie Stata imprimée dans Kripfganz & Schneider (2023, Stata Journal
23(4) ; preprint ouvert Tohoku TUPD-2022-006, §5, exemple salaires UK
de PSS 2001) : cas 3 et 4, k=4, T=104, sr=26 (coefficients de court
terme, dummies incluses), CV aux 3 seuils + p-values F et t.

Marqués ``external`` : nécessitent les coefficients K&S dans le cache
local (non redistribués par ardlpy — exécuter une fois
``download_surface_coefs()``). Skip explicite sinon.
"""

from __future__ import annotations

import pytest

from ardlpy.critical_values.ks2020_finite import (
    _coefs_path,
    crit_value_bounds_finite,
    pvalue_bounds_finite,
)

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        not _coefs_path().exists(),
        reason=(
            "Coefficients K&S absents du cache local : exécuter "
            "ardlpy.critical_values.ks2020_finite.download_surface_coefs()"
        ),
    ),
]

# Sortie publiée (SJ 2023 / preprint Tohoku, p. 20) — cas 3, k=4, T=104,
# sr=26 ; F=5.421, t=-3.475 ; df_resid = 104 - 32 = 72.
_PUB_F_C3 = {0.10: (2.362, 3.646), 0.05: (2.806, 4.226), 0.01: (3.800, 5.502)}
_PUB_T_C3 = {0.10: (-2.447, -3.499), 0.05: (-2.777, -3.873), 0.01: (-3.421, -4.589)}
# cas 4 (trend restreinte) — F=4.780, t=-2.437 ; t servi par les
# surfaces du cas 5 (mapping K&S).
_PUB_F_C4 = {0.10: (2.576, 3.693), 0.05: (2.988, 4.219), 0.01: (3.906, 5.374)}
_PUB_T_C4 = {0.10: (-2.954, -3.837), 0.05: (-3.281, -4.210), 0.01: (-3.922, -4.926)}

_TOL = 1e-3 + 5e-4  # 1e-3 contractuel + arrondi d'impression (3 décimales)


class TestPublishedStataOutput:
    @pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01])
    def test_f_case3(self, alpha: float) -> None:
        got = crit_value_bounds_finite(case=3, k=4, t_obs=104, sr=26, alpha=alpha)
        exp = _PUB_F_C3[alpha]
        assert got[0] == pytest.approx(exp[0], abs=_TOL)
        assert got[1] == pytest.approx(exp[1], abs=_TOL)

    @pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01])
    def test_t_case3(self, alpha: float) -> None:
        got = crit_value_bounds_finite(
            case=3, k=4, t_obs=104, sr=26, alpha=alpha, stat="t"
        )
        exp = _PUB_T_C3[alpha]
        assert got[0] == pytest.approx(exp[0], abs=_TOL)
        assert got[1] == pytest.approx(exp[1], abs=_TOL)

    @pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01])
    def test_f_case4(self, alpha: float) -> None:
        got = crit_value_bounds_finite(case=4, k=4, t_obs=104, sr=26, alpha=alpha)
        exp = _PUB_F_C4[alpha]
        assert got[0] == pytest.approx(exp[0], abs=_TOL)
        assert got[1] == pytest.approx(exp[1], abs=_TOL)

    @pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01])
    def test_t_case4_served_by_case5_mapping(self, alpha: float) -> None:
        """Cas 4 : bornes t imprimées par Stata = surfaces du cas 5
        (la distribution du t n'est pas affectée par la restriction des
        déterministes — convention K&S encodée dans _check)."""
        got = crit_value_bounds_finite(
            case=4, k=4, t_obs=104, sr=26, alpha=alpha, stat="t"
        )
        exp = _PUB_T_C4[alpha]
        assert got[0] == pytest.approx(exp[0], abs=_TOL)
        assert got[1] == pytest.approx(exp[1], abs=_TOL)

    def test_pvalues_match_printed(self) -> None:
        """p-values imprimées (3 décimales) : F 0.001/0.011, t 0.009/0.104
        (cas 3) ; F 0.002/0.023, t 0.247/0.532 (cas 4)."""
        p_f3 = pvalue_bounds_finite(5.421, 3, 4, 104, 26, df_resid=72)
        assert p_f3[0] == pytest.approx(0.001, abs=5.5e-4)
        assert p_f3[1] == pytest.approx(0.011, abs=5.5e-4)
        p_t3 = pvalue_bounds_finite(-3.475, 3, 4, 104, 26, df_resid=72, stat="t")
        assert p_t3[0] == pytest.approx(0.009, abs=5.5e-4)
        assert p_t3[1] == pytest.approx(0.104, abs=5.5e-4)
        p_f4 = pvalue_bounds_finite(4.780, 4, 4, 104, 26, df_resid=71)
        assert p_f4[0] == pytest.approx(0.002, abs=5.5e-4)
        assert p_f4[1] == pytest.approx(0.023, abs=5.5e-4)
        p_t4 = pvalue_bounds_finite(-2.437, 4, 4, 104, 26, df_resid=71, stat="t")
        assert p_t4[0] == pytest.approx(0.247, abs=5.5e-4)
        assert p_t4[1] == pytest.approx(0.532, abs=5.5e-4)


class TestInternalCoherence:
    def test_asymptotic_limit_matches_a1(self) -> None:
        """T très grand -> CV de la voie A1 (statsmodels) à ±0.05
        (re-simulations indépendantes)."""
        from ardlpy.critical_values.ks2020 import crit_value_bounds

        for case in (1, 3, 5):
            for k in (1, 3):
                fin = crit_value_bounds_finite(
                    case=case, k=k, t_obs=10_000_000, sr=0, alpha=0.05
                )
                a1 = crit_value_bounds(case=case, k=k, alpha=0.05)
                assert fin[0] == pytest.approx(a1[0], abs=0.05)
                assert fin[1] == pytest.approx(a1[1], abs=0.05)

    def test_t_asymptotic_limit_matches_pss(self) -> None:
        got = crit_value_bounds_finite(
            case=3, k=1, t_obs=10_000_000, sr=0, alpha=0.05, stat="t"
        )
        assert got[0] == pytest.approx(-2.86, abs=0.02)
        assert got[1] == pytest.approx(-3.22, abs=0.02)

    def test_cv_decrease_toward_asymptotic_in_t_obs(self) -> None:
        """Bornes plus conservatrices en petit échantillon, décroissantes
        vers l'asymptotique (motivation de Narayan/K&S)."""
        values = [
            crit_value_bounds_finite(case=3, k=2, t_obs=t, sr=3, alpha=0.05)[1]
            for t in (30, 50, 80, 200, 1000)
        ]
        assert all(a >= b - 1e-9 for a, b in zip(values[:-1], values[1:], strict=True))

    def test_sr_increases_cv_in_small_samples(self) -> None:
        """Plus de coefficients de court terme -> bornes plus élevées à
        T petit (consommation de degrés de liberté)."""
        low = crit_value_bounds_finite(case=3, k=2, t_obs=40, sr=0, alpha=0.05)[1]
        high = crit_value_bounds_finite(case=3, k=2, t_obs=40, sr=10, alpha=0.05)[1]
        assert high > low

    def test_pvalue_roundtrip_at_cv(self) -> None:
        for stat in ("F", "t"):
            cv = crit_value_bounds_finite(3, 2, 90, 4, 0.05, stat=stat)  # type: ignore[arg-type]
            p = pvalue_bounds_finite(cv[0], 3, 2, 90, 4, df_resid=80, stat=stat)  # type: ignore[arg-type]
            assert p[0] == pytest.approx(0.05, abs=2e-3)


class TestCoverageAndErrors:
    def test_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="case"):
            crit_value_bounds_finite(0, 1, 100, 2, 0.05)
        with pytest.raises(ValueError, match="sr"):
            crit_value_bounds_finite(3, 1, 100, -1, 0.05)
        with pytest.raises(ValueError, match="stat"):
            crit_value_bounds_finite(3, 1, 100, 2, 0.05, stat="W")  # type: ignore[arg-type]


def test_missing_cache_raises_with_instructions(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Sans cache : erreur explicite avec la marche à suivre (pas de
    téléchargement silencieux). Ce test ne dépend PAS du cache réel."""
    import ardlpy.critical_values.ks2020_finite as mod

    monkeypatch.setenv("ARDLPY_CACHE", str(tmp_path))
    monkeypatch.setattr(mod, "_TABLES", None)
    with pytest.raises(FileNotFoundError, match="download_surface_coefs"):
        mod._load_tables()


class TestBoundsTestIntegration:
    """Reproduction bout-en-bout de la sortie publiée du Stata Journal
    2023 (§5) via bounds_test(finite_t=True) sur nos données PSS2001."""

    def test_full_sj2023_case3_reproduction(self) -> None:
        import warnings

        from ardlpy.bounds import bounds_test
        from ardlpy.datasets import load_pss2001

        data = load_pss2001()
        # échantillon d'estimation Stata (smpl) : 1972q1-1997q4 = 104 obs
        sub = data.iloc[2:].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(
                sub["w"],
                sub[["Prod", "UR", "Wedge", "Union"]],
                case=3,
                order=(6, {"Prod": 1, "UR": 6, "Wedge": 6, "Union": 6}),
                finite_t=True,
                fixed_regressors=sub[["D7475", "D7579"]],
            )
        assert res._fit.nobs == 104
        assert res.f_stat == pytest.approx(5.421, abs=1e-3)
        assert res.t_stat == pytest.approx(-3.475, abs=1e-3)
        assert res.p_values is not None
        assert res.p_values["p_I0"] == pytest.approx(0.001, abs=5.5e-4)
        assert res.p_values["p_I1"] == pytest.approx(0.011, abs=5.5e-4)
        assert res.p_values["t_p_I0"] == pytest.approx(0.009, abs=5.5e-4)
        assert res.p_values["t_p_I1"] == pytest.approx(0.104, abs=5.5e-4)
        # bornes 5 % = colonnes imprimées
        assert res.bounds.loc[0.05, "F_I0"] == pytest.approx(2.806, abs=_TOL)
        assert res.bounds.loc[0.05, "F_I1"] == pytest.approx(4.226, abs=_TOL)
        assert res.bounds.loc[0.05, "t_I1"] == pytest.approx(-3.873, abs=_TOL)
        # décisions : Stata affiche « inconclusive » (règle jointe) ; notre
        # taxonomie détaille : F rejette, t entre les bornes -> suspicion
        # de dégénérescence de type 1 (même substance, label plus riche)
        assert res.decision_f == "cointegration"
        assert res.decision_t == "inconclusive"
        assert res.decision_joint == "degenerate_suspicion"
        # summary affiche aussi les p-values du t
        assert "p-values t" in res.summary()

    def test_finite_t_requires_kripfganz(self) -> None:
        from ardlpy.bounds import bounds_test
        from ardlpy.datasets import load_denmark

        data = load_denmark()
        with pytest.raises(ValueError, match="finite_t"):
            bounds_test(
                data["LRM"],
                data[["LRY", "IBO", "IDE"]],
                case=3,
                order=(1, 1),
                cv_source="pss",
                finite_t=True,
            )
