"""Spec 24 §2.2/§2.3/§2.5 — CS-ARDL et CS-DL.

Le verrou de ce module est `TestDeterministicCollinearity`. Avec k+1
moyennes, leurs retards et un T modeste, le design individuel est
souvent deficient en rang : il FAUT retirer des colonnes. Le faire selon
ce que l'algebre lineaire prefere ce jour-la, en revanche, produirait
des coefficients de long terme differents d'une plateforme a l'autre a
partir des memes donnees — un resultat non reproductible qui aurait
toutes les apparences d'un resultat.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.panel import CSARDL, CSDL
from pyardl.panel.csardl import _select_independent


def factor_panel(
    n_units: int = 20,
    n_obs: int = 70,
    seed: int = 0,
    theta: float = 0.80,
    gamma_bar: float = 0.6,
    gamma_sd: float = 0.25,
    dynamic: bool = True,
) -> pd.DataFrame:
    """Facteur commun dans y ET dans x : le monde du CS-ARDL."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.normal(size=n_obs))
    rows = []
    for i in range(n_units):
        gamma = gamma_bar + gamma_sd * rng.normal()
        x = np.cumsum(rng.normal(size=n_obs)) + gamma * factor
        if dynamic:
            lam = min(-0.4 + 0.08 * rng.normal(), -0.05)
            y = np.zeros(n_obs)
            for t in range(1, n_obs):
                y[t] = (
                    y[t - 1]
                    + lam * (y[t - 1] - theta * x[t - 1])
                    + gamma * (factor[t] - factor[t - 1])
                    + rng.normal(scale=0.4)
                )
        else:
            y = theta * x + gamma * factor + rng.normal(scale=0.4, size=n_obs)
        rows.append(
            pd.DataFrame({"id": f"u{i:02d}", "t": np.arange(n_obs), "y": y, "x": x})
        )
    return pd.concat(rows, ignore_index=True)


_KW = {"y": "y", "X": ["x"], "id": "id", "time": "t"}


def _fit(cls, df: pd.DataFrame, **kw: object) -> object:  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(df, **{**_KW, **kw}).fit()


class TestDeterministicCollinearity:
    """§2.5 — LE verrou : le retrait de colonnes doit etre une regle,
    pas un accident du solveur."""

    def test_columns_are_examined_left_to_right(self) -> None:
        """La troisieme colonne est la somme des deux premieres. C'est
        elle qui doit tomber — pas l'une des deux qui la precedent."""
        base = np.arange(1.0, 21.0)
        design = np.column_stack([np.ones(20), base, np.ones(20) + base])
        kept_design, kept, dropped = _select_independent(
            design, ["const", "x", "somme"]
        )
        assert kept == ["const", "x"]
        assert dropped == ["somme"]
        assert kept_design.shape == (20, 2)

    def test_the_order_decides_which_column_survives(self) -> None:
        """Meme information, ordre inverse : c'est desormais `const` qui
        tombe. Ce n'est pas un defaut — c'est ce qui rend le resultat
        PREVISIBLE, et pourquoi l'appelant place le modele avant
        l'approximation."""
        base = np.arange(1.0, 21.0)
        design = np.column_stack([np.ones(20) + base, base, np.ones(20)])
        _, kept, dropped = _select_independent(design, ["somme", "x", "const"])
        assert kept == ["somme", "x"]
        assert dropped == ["const"]

    def test_a_duplicated_column_is_dropped(self) -> None:
        base = np.arange(1.0, 16.0)
        design = np.column_stack([base, base])
        _, kept, dropped = _select_independent(design, ["x", "x_copie"])
        assert kept == ["x"]
        assert dropped == ["x_copie"]

    def test_repeated_runs_give_the_same_design(self) -> None:
        """Le meme panel doit produire exactement les memes colonnes
        retenues a chaque execution."""
        df = factor_panel(n_units=12, n_obs=40, seed=1)
        a = _fit(CSARDL, df, order=(1, 1), cs_lags=3)
        b = _fit(CSARDL, df, order=(1, 1), cs_lags=3)
        assert a.dropped_columns == b.dropped_columns
        assert a.longrun.loc["x", "theta"] == pytest.approx(
            b.longrun.loc["x", "theta"], rel=1e-14
        )

    @staticmethod
    def _panel_with_identical_averages(
        n_units: int = 12, n_obs: int = 60, seed: int = 2
    ) -> pd.DataFrame:
        """Un panel ou cs_y est EXACTEMENT egal a cs_x, sans qu'aucun
        individu n'ait y = x.

        Construction : y_i est une permutation circulaire des x a travers
        les individus. La moyenne transversale porte alors sur les memes
        nombres a chaque date — donc cs_y == cs_x — tandis que chaque
        individu garde des series distinctes et un design sain.

        C'est le seul moyen propre de declencher la regle de colinearite
        sur une MOYENNE plutot que sur une colonne du modele, et donc de
        verifier que c'est bien l'approximation qui tombe.
        """
        rng = np.random.default_rng(seed)
        xs = [np.cumsum(rng.normal(size=n_obs)) for _ in range(n_units)]
        rows = []
        for i in range(n_units):
            rows.append(
                pd.DataFrame(
                    {
                        "id": f"u{i:02d}",
                        "t": np.arange(n_obs),
                        "y": xs[(i + 1) % n_units],
                        "x": xs[i],
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def test_drops_are_recorded_not_silent(self) -> None:
        """Un retrait doit apparaitre dans le resultat et dans le resume,
        jamais se faire en silence."""
        df = self._panel_with_identical_averages()
        res = _fit(CSARDL, df, order=(1, 1), cs_lags=2)
        assert res.dropped_columns, "aucun retrait declenche par ce panel"
        assert all(isinstance(v, list) and v for v in res.dropped_columns.values())
        assert "collinear columns removed" in res.summary()

    def test_the_averages_are_dropped_before_the_model(self) -> None:
        """Quand quelque chose doit tomber, ce doit etre l'approximation
        et non le modele. Ici cs_x duplique cs_y : c'est cs_x qui part,
        et aucune colonne du modele."""
        df = self._panel_with_identical_averages(seed=3)
        res = _fit(CSARDL, df, order=(1, 1), cs_lags=2)
        assert res.dropped_columns
        for dropped in res.dropped_columns.values():
            assert all(name.startswith("cs_") for name in dropped), dropped
        for fit in res.individual.values():
            for kept in ("const", "y.L1", "x.L0", "x.L1"):
                assert kept in fit.params.index


class TestCSARDL:
    def test_recovers_theta_under_a_common_factor(self) -> None:
        res = _fit(CSARDL, factor_panel(n_units=30, n_obs=90, seed=4), order=(1, 1))
        assert abs(res.longrun.loc["x", "theta"] - 0.80) < 0.10

    def test_adjustment_is_negative(self) -> None:
        res = _fit(CSARDL, factor_panel(n_units=20, seed=5), order=(1, 1))
        assert res.adjustment["lambda"] < 0
        assert res.adjustment["se"] > 0

    def test_between_individual_standard_error(self) -> None:
        """L'agregation est celle de la spec 22 : la variance vient de la
        dispersion INTER-individus."""
        res = _fit(CSARDL, factor_panel(n_units=25, seed=6), order=(1, 1))
        theta_i = res.theta_i["x"].to_numpy()
        n = theta_i.size
        expected = np.sqrt(np.sum((theta_i - theta_i.mean()) ** 2) / (n * (n - 1)))
        assert res.longrun.loc["x", "se"] == pytest.approx(expected, rel=1e-12)

    def test_auto_cs_lags_uses_the_cube_root_rule(self) -> None:
        df = factor_panel(n_units=12, n_obs=64, seed=7)
        res = _fit(CSARDL, df, order=(1, 1))
        assert res.cs_lags == 4

    def test_cd_test_after_augmentation(self) -> None:
        res = _fit(CSARDL, factor_panel(n_units=25, n_obs=80, seed=8), order=(1, 1))
        cd = res.cd_test()
        assert cd.n_units >= 2
        assert "absorbed" in cd.summary(context="after") or "SURVIVES" in cd.summary(
            context="after"
        )

    def test_summary_reports_the_construction(self) -> None:
        res = _fit(CSARDL, factor_panel(n_units=15, seed=9), order=(1, 1))
        text = res.summary()
        assert "CS-ARDL (Chudik & Pesaran 2015)" in text
        assert "BETWEEN-individual" in text
        assert "cross-sectional averages" in text

    def test_unit_root_individual_is_refused_not_averaged(self) -> None:
        """Si les coefficients autoregressifs somment a un, le long terme
        est un rapport a denominateur nul : cet individu n'a pas de long
        terme, et l'inclure avec un theta infini contaminerait la
        moyenne."""
        df = factor_panel(n_units=12, n_obs=60, seed=10)
        res = _fit(CSARDL, df, order=(1, 1))
        assert np.all(np.isfinite(res.theta_i.to_numpy()))

    def test_heterogeneity_table(self) -> None:
        res = _fit(CSARDL, factor_panel(n_units=20, seed=11), order=(1, 1))
        table = res.heterogeneity()
        assert list(table.columns) == ["mean", "sd", "min", "median", "max", "cv"]


class TestCSDL:
    def test_recovers_theta_on_a_static_dgp(self) -> None:
        df = factor_panel(n_units=30, n_obs=80, seed=12, dynamic=False)
        res = _fit(CSDL, df)
        assert abs(res.longrun.loc["x", "theta"] - 0.80) < 0.05

    def test_theta_is_read_off_x_directly(self) -> None:
        """Le coefficient de x EST le long terme : aucun rapport n'est
        forme, et c'est ce qui rend CS-DL robuste a un ordre de retards
        mal choisi."""
        df = factor_panel(n_units=15, n_obs=70, seed=13, dynamic=False)
        res = _fit(CSDL, df, trunc_lags=0, cs_lags=0)
        for key, fit in res.individual.items():
            assert res.theta_i.loc[key, "x"] == pytest.approx(
                float(fit.params["x"]), rel=1e-12
            )

    def test_robust_to_the_truncation_choice(self) -> None:
        """Changer le nombre de differences retardees ne doit pas
        bouleverser le long terme sur un DGP net."""
        df = factor_panel(n_units=25, n_obs=80, seed=14, dynamic=False)
        a = _fit(CSDL, df, trunc_lags=1)
        b = _fit(CSDL, df, trunc_lags=3)
        assert abs(a.longrun.loc["x", "theta"] - b.longrun.loc["x", "theta"]) < 0.05

    def test_summary_says_there_is_no_adjustment_speed(self) -> None:
        """CS-DL n'estime pas la dynamique : il ne peut pas dire a quelle
        vitesse l'ajustement se fait, et le resume doit le dire plutot
        que de laisser un vide."""
        res = _fit(CSDL, factor_panel(n_units=15, seed=15, dynamic=False))
        text = res.summary()
        assert "No adjustment speed" in text
        assert not hasattr(res, "adjustment")

    def test_between_individual_standard_error(self) -> None:
        res = _fit(CSDL, factor_panel(n_units=25, seed=16, dynamic=False))
        theta_i = res.theta_i["x"].to_numpy()
        n = theta_i.size
        expected = np.sqrt(np.sum((theta_i - theta_i.mean()) ** 2) / (n * (n - 1)))
        assert res.longrun.loc["x", "se"] == pytest.approx(expected, rel=1e-12)


class TestCSDLFailurePaths:
    """Ce qui arrive quand le NIVEAU de x ne survit pas au filtre.

    CS-DL lit le long terme directement sur le coefficient de x. Si
    cette colonne est retiree comme colineaire, il n'y a plus rien a
    lire — et rendre un theta pris ailleurs serait pire que de refuser.
    """

    def test_asking_for_a_perfectly_collinear_regressor_is_refused(self) -> None:
        """`dup` duplique `x`. L'utilisateur demande donc un theta sur
        une colonne qui ne peut pas survivre au filtre : CS-DL lisant le
        long terme DIRECTEMENT sur ce coefficient, il n'y a rien a lire,
        et aucune reponse n'existe. Le refus nomme la cause plutot que
        de rendre un theta pris ailleurs."""
        rng = np.random.default_rng(30)
        n_obs, n_units = 60, 10
        rows = []
        for i in range(n_units):
            x = np.cumsum(rng.normal(size=n_obs))
            y = 0.8 * x + rng.normal(scale=0.3, size=n_obs)
            rows.append(
                pd.DataFrame({"id": f"u{i:02d}", "t": np.arange(n_obs), "y": y, "x": x})
            )
        df = pd.concat(rows, ignore_index=True)
        df["dup"] = df["x"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="dropped as collinear"):
                CSDL(
                    df,
                    y="y",
                    X=["x", "dup"],
                    id="id",
                    time="t",
                    trunc_lags=0,
                    cs_lags=0,
                ).fit()

    def test_a_single_collinear_regressor_still_works(self) -> None:
        """Contraste : demander UNIQUEMENT x, sur les memes donnees,
        doit aboutir. Le refus ci-dessus porte sur la demande, pas sur
        les donnees."""
        rng = np.random.default_rng(32)
        n_obs = 60
        rows = []
        for i in range(10):
            x = np.cumsum(rng.normal(size=n_obs))
            y = 0.8 * x + rng.normal(scale=0.3, size=n_obs)
            rows.append(
                pd.DataFrame({"id": f"u{i:02d}", "t": np.arange(n_obs), "y": y, "x": x})
            )
        df = pd.concat(rows, ignore_index=True)
        df["dup"] = df["x"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CSDL(
                df, y="y", X=["x"], id="id", time="t", trunc_lags=0, cs_lags=0
            ).fit()
        assert res.n_units == 10
        assert abs(res.longrun.loc["x", "theta"] - 0.8) < 0.05

    def test_warns_when_individuals_are_absent_from_the_average(self) -> None:
        """Un individu ecarte doit etre nomme : le N d'un tableau de
        resultats doit toujours pouvoir se justifier."""
        rng = np.random.default_rng(31)
        n_obs = 60
        rows = []
        for i in range(8):
            x = np.cumsum(rng.normal(size=n_obs))
            y = 0.8 * x + rng.normal(scale=0.3, size=n_obs)
            rows.append(
                pd.DataFrame({"id": f"u{i:02d}", "t": np.arange(n_obs), "y": y, "x": x})
            )
        # Un individu dont x est constant : le conteneur l'exclut deja,
        # et le resume doit le dire.
        df = pd.concat(rows, ignore_index=True)
        df.loc[df["id"] == "u00", "x"] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CSDL(df, y="y", X=["x"], id="id", time="t").fit()
        assert res.n_units == 7
        assert "u00" in res.panel.excluded


class TestRobustness:
    def test_row_order_does_not_matter(self) -> None:
        df = factor_panel(n_units=15, seed=17)
        shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
        a = _fit(CSARDL, df, order=(1, 1))
        b = _fit(CSARDL, shuffled, order=(1, 1))
        assert a.longrun.loc["x", "theta"] == pytest.approx(
            b.longrun.loc["x", "theta"], rel=1e-10
        )

    def test_unbalanced_panel(self) -> None:
        """§4.3 — panel non cylindre : les moyennes se prennent sur les
        presents, et l'estimation aboutit."""
        df = factor_panel(n_units=18, n_obs=70, seed=18)
        df = df[~((df["id"] == "u03") & (df["t"] >= 45))]
        res = _fit(CSARDL, df, order=(1, 1))
        assert res.n_units == 18
        assert res.panel.unbalanced

    @pytest.mark.parametrize("det", ["none", "const", "trend"])
    def test_every_deterministic_case_builds(self, det: str) -> None:
        res = _fit(CSARDL, factor_panel(n_units=12, seed=19), order=(1, 1), det=det)
        assert np.isfinite(res.longrun.loc["x", "theta"])

    def test_two_regressors(self) -> None:
        rng = np.random.default_rng(20)
        n_obs, n_units = 80, 20
        factor = np.cumsum(rng.normal(size=n_obs))
        rows = []
        for i in range(n_units):
            g = 0.5 + 0.2 * rng.normal()
            a = np.cumsum(rng.normal(size=n_obs)) + g * factor
            b = np.cumsum(rng.normal(size=n_obs))
            y = 0.8 * a - 0.4 * b + g * factor + rng.normal(scale=0.4, size=n_obs)
            rows.append(
                pd.DataFrame(
                    {"id": f"u{i:02d}", "t": np.arange(n_obs), "y": y, "a": a, "b": b}
                )
            )
        df = pd.concat(rows, ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CSDL(df, y="y", X=["a", "b"], id="id", time="t").fit()
        assert abs(res.longrun.loc["a", "theta"] - 0.8) < 0.1
        assert abs(res.longrun.loc["b", "theta"] + 0.4) < 0.1


class TestValidation:
    @pytest.mark.parametrize(
        ("cls", "kwargs", "match"),
        [
            (CSARDL, {"det": "quadratic"}, "det must be"),
            (CSARDL, {"order": (0, 1)}, "p must be at least 1"),
            (CSARDL, {"order": (1, -1)}, "q must be non-negative"),
            (CSARDL, {"cs_lags": -1}, "cs_lags must be non-negative"),
            (CSDL, {"det": "quadratic"}, "det must be"),
            (CSDL, {"trunc_lags": -2}, "trunc_lags must be non-negative"),
        ],
    )
    def test_refusals(self, cls, kwargs: dict, match: str) -> None:  # type: ignore[no-untyped-def]
        df = factor_panel(n_units=8, n_obs=40, seed=21)
        with pytest.raises(ValueError, match=match):
            cls(df, **{**_KW, **kwargs})

    def test_needs_two_individuals(self) -> None:
        df = factor_panel(n_units=1, n_obs=40, seed=22)
        with pytest.raises(ValueError, match="at least two|dispersion ACROSS"):
            CSARDL(df, order=(1, 1), **_KW).fit()  # type: ignore[arg-type]

    def test_too_many_columns_for_the_sample_is_reported(self) -> None:
        """Beaucoup de retards sur un T court : les individus qui n'ont
        plus assez d'observations doivent etre nommes, pas disparaitre."""
        df = factor_panel(n_units=6, n_obs=22, seed=23)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                res = CSARDL(df, order=(1, 1), cs_lags=9, **_KW).fit()  # type: ignore[arg-type]
            except ValueError as exc:
                assert "Failures" in str(exc) or "at least two" in str(exc)
                return
        assert isinstance(res.failed, dict)

    def test_results_are_immutable(self) -> None:
        res = _fit(CSARDL, factor_panel(n_units=10, seed=24), order=(1, 1))
        with pytest.raises(AttributeError):
            res.longrun = pd.DataFrame()  # type: ignore[misc]


class TestWhyTheModuleExists:
    def test_the_augmentation_beats_a_naive_mean_group(self) -> None:
        """La comparaison qui justifie le module, sur les MEMES donnees.
        L'etude dimensionnee est dans validation/spec24_montecarlo.py ;
        ceci est le verrou permanent."""
        from pyardl.panel import MeanGroup

        df = factor_panel(n_units=30, n_obs=90, seed=25, gamma_bar=0.8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            naive = MeanGroup(df, order=(1, 1), **_KW).fit()  # type: ignore[arg-type]
        augmented = _fit(CSARDL, df, order=(1, 1))
        true_theta = 0.80
        assert abs(augmented.longrun.loc["x", "theta"] - true_theta) < abs(
            naive.longrun.loc["x", "theta"] - true_theta
        )

    def test_cd_detects_the_factor_before_augmentation(self) -> None:
        """§4.2 — le CD doit crier AVANT. Sans lui, rien ne dit a
        l'utilisateur qu'il a besoin de ce module."""
        from pyardl.panel import MeanGroup, cd_test

        df = factor_panel(n_units=25, n_obs=80, seed=26, gamma_bar=0.8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            naive = MeanGroup(df, order=(1, 1), **_KW).fit()  # type: ignore[arg-type]
        resid = pd.DataFrame({k: f.resid for k, f in naive.individual.items()})
        assert cd_test(resid).rejects
