"""Spec 22 §2.2/§3 — l'estimateur Mean Group.

Le module n'estime rien lui-meme : il orchestre N ARDL (spec 05, deja
validee) et agrege. Le plan de tests porte donc sur l'agregation, et
d'abord sur la chose contre-intuitive que la spec signale — la variance
vient de la dispersion INTER-individus, jamais des V_i individuelles.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.panel import MeanGroup


def heterogeneous_panel(
    n_units: int = 20,
    n_obs: int = 60,
    seed: int = 0,
    theta_bar: float = 0.75,
    theta_sd: float = 0.15,
    lambda_bar: float = -0.4,
    lambda_sd: float = 0.08,
    noise: float = 0.4,
) -> pd.DataFrame:
    """DGP de Pesaran-Smith : theta_i et lambda_i propres a chaque individu."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        theta = theta_bar + theta_sd * rng.normal()
        lam = lambda_bar + lambda_sd * rng.normal()
        x = np.cumsum(rng.normal(size=n_obs))
        y = np.zeros(n_obs)
        for t in range(1, n_obs):
            y[t] = (
                y[t - 1] + lam * (y[t - 1] - theta * x[t - 1]) + rng.normal(scale=noise)
            )
        rows.append(
            pd.DataFrame({"id": f"u{i:02d}", "t": np.arange(n_obs), "y": y, "x": x})
        )
    return pd.concat(rows, ignore_index=True)


def _fit(df: pd.DataFrame, **kw: object) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MeanGroup(
            df, y="y", X=["x"], id="id", time="t", order=kw.pop("order", (1, 1)), **kw
        ).fit()  # type: ignore[arg-type]


class TestConsistency:
    """§3.1 — theta_MG doit retrouver theta_bar."""

    def test_group_mean_recovers_the_population_mean(self) -> None:
        res = _fit(heterogeneous_panel(n_units=40, n_obs=80, seed=1))
        theta = res.longrun.loc["x", "theta"]
        se = res.longrun.loc["x", "se"]
        assert abs(theta - 0.75) < 3 * se

    def test_adjustment_is_negative_on_a_converging_dgp(self) -> None:
        res = _fit(heterogeneous_panel(n_units=30, seed=2))
        assert res.adjustment["lambda"] < 0
        assert res.adjustment["se"] > 0


class TestBetweenIndividualVariance:
    """§2.2b — LE point de la spec, et le piege nomme dans CLAUDE.md."""

    def test_se_is_the_dispersion_of_the_theta_i(self) -> None:
        """Formule explicite : sum((theta_i - theta_MG)^2) / (N(N-1)).
        Recalculee ici a la main, pour que le test echoue si quelqu'un
        remplace la dispersion inter-individus par une combinaison des
        erreurs types individuelles."""
        res = _fit(heterogeneous_panel(n_units=25, seed=3))
        theta_i = res.theta_i["x"].to_numpy()
        n = theta_i.size
        expected = np.sqrt(np.sum((theta_i - theta_i.mean()) ** 2) / (n * (n - 1)))
        assert res.longrun.loc["x", "se"] == pytest.approx(expected, rel=1e-12)

    def test_se_ignores_the_individual_standard_errors(self) -> None:
        """Verification par contraste : les erreurs types individuelles
        existent et sont bonnes, mais elles ne doivent PAS entrer dans
        celle du groupe. Sur un panel heterogene la version poolee est
        beaucoup plus etroite — c'est exactement l'erreur a ne pas
        commettre."""
        res = _fit(heterogeneous_panel(n_units=25, n_obs=80, seed=4))
        pooled = np.sqrt(
            np.mean([f.longrun.loc["x", "se"] ** 2 for f in res.individual.values()])
            / res.n_units
        )
        between = res.longrun.loc["x", "se"]
        assert between > 1.5 * pooled

    def test_point_estimate_is_the_plain_mean(self) -> None:
        res = _fit(heterogeneous_panel(n_units=20, seed=5))
        assert res.longrun.loc["x", "theta"] == pytest.approx(
            res.theta_i["x"].mean(), rel=1e-12
        )

    def test_reference_distribution_is_t_with_n_minus_one(self) -> None:
        """La variance est estimee sur N estimations individuelles. A
        N = 20 la difference entre la loi normale et la t(19) n'est pas
        cosmetique sur les bornes d'intervalle."""
        from scipy import stats

        res = _fit(heterogeneous_panel(n_units=20, seed=6))
        row = res.longrun.loc["x"]
        crit = stats.t.ppf(0.975, res.n_units - 1)
        assert row["ci_lower"] == pytest.approx(
            row["theta"] - crit * row["se"], rel=1e-12
        )
        assert row["pvalue"] == pytest.approx(
            2 * stats.t.sf(abs(row["t"]), res.n_units - 1), rel=1e-12
        )


class TestOrderModes:
    def test_common_order_is_applied_to_everyone(self) -> None:
        res = _fit(heterogeneous_panel(n_units=10, seed=7), order=(2, 2))
        assert (res.orders["p"] == 2).all()
        assert (res.orders["q[x]"] == 2).all()

    def test_auto_selects_per_individual(self) -> None:
        """Les deux modes existent parce qu'ils repondent a deux
        questions : un ordre commun rend les coefficients comparables,
        la selection par individu laisse chaque dynamique etre ce que
        ses donnees disent."""
        res = _fit(
            heterogeneous_panel(n_units=12, n_obs=70, seed=8),
            order="auto",
            max_p=3,
            max_q=3,
        )
        assert res.orders.shape[0] == 12
        assert res.n_units == 12

    def test_auto_leaves_raw_coefficients_unaveraged(self) -> None:
        """Des coefficients issus de specifications differentes ne sont
        pas la meme quantite : les moyenner produirait un nombre qui ne
        veut rien dire. Le long terme, lui, reste comparable."""
        res = _fit(
            heterogeneous_panel(n_units=12, n_obs=70, seed=9),
            order="auto",
            max_p=3,
            max_q=3,
        )
        if res.orders.nunique().gt(1).any():
            assert res.coefficients.empty
            assert not res.longrun.empty

    def test_common_order_does_average_raw_coefficients(self) -> None:
        res = _fit(heterogeneous_panel(n_units=10, seed=10))
        assert list(res.coefficients.index) == ["const", "y.L1", "x.L0", "x.L1"]
        assert res.coefficients["se"].gt(0).all()


class TestAggregators:
    def test_median_resists_one_extreme_individual(self) -> None:
        """A petit N, un individu a la dynamique explosive deplace une
        moyenne de vingt. L'estimation de groupe decrirait alors cet
        individu-la plutot que le groupe."""
        df = heterogeneous_panel(n_units=15, seed=11)
        rng = np.random.default_rng(0)
        mask = df["id"] == "u00"
        df.loc[mask, "y"] = np.cumsum(rng.normal(size=int(mask.sum()))) * 50
        mean_res = _fit(df, aggregator="mean")
        median_res = _fit(df, aggregator="median")
        spread = df.groupby("id").size().size
        assert spread == 15
        assert abs(median_res.longrun.loc["x", "theta"] - 0.75) <= abs(
            mean_res.longrun.loc["x", "theta"] - 0.75
        )

    def test_trimmed_drops_both_tails(self) -> None:
        res = _fit(heterogeneous_panel(n_units=20, seed=12), aggregator="trimmed")
        assert res.n_effective == 16
        assert res.n_units == 20

    def test_median_variance_carries_the_efficiency_factor(self) -> None:
        """La variance d'echantillonnage de la mediane vaut pi/2 fois
        celle de la moyenne sous normalite. Utiliser la formule de la
        moyenne la sous-estimerait de 25 %."""
        df = heterogeneous_panel(n_units=21, seed=13)
        mean_res = _fit(df, aggregator="mean")
        med_res = _fit(df, aggregator="median")
        theta_i = mean_res.theta_i["x"].to_numpy()
        n = theta_i.size
        med = np.median(theta_i)
        expected = np.sqrt(np.sum((theta_i - med) ** 2) / (n * (n - 1)) * (np.pi / 2))
        assert med_res.longrun.loc["x", "se"] == pytest.approx(expected, rel=1e-12)

    def test_constructor_caps_trim_below_a_half(self) -> None:
        """La borne du constructeur est ce qui protege l'utilisateur :
        avec trim < 0.5 il reste toujours au moins un individu."""
        df = heterogeneous_panel(n_units=6, seed=14)
        with pytest.raises(ValueError, match="trim must be in"):
            MeanGroup(
                df,
                y="y",
                X=["x"],
                id="id",
                time="t",
                aggregator="trimmed",
                trim=0.5,
            )

    def test_helper_guards_against_trimming_everything(self) -> None:
        """Garde de defense en profondeur sur l'agregateur lui-meme : il
        est appele par le constructeur, mais rien ne l'empeche d'etre
        appele directement, et un trim de 0.5 ne laisserait rien."""
        from pyardl.panel.mg import _aggregate

        with pytest.raises(ValueError, match="leaving nothing to average"):
            _aggregate(np.arange(6.0).reshape(6, 1), "trimmed", 0.5)


class TestDiagnostics:
    def test_individual_fits_are_kept(self) -> None:
        """Une moyenne de groupe doit toujours pouvoir etre remontee a ce
        qui l'a produite."""
        res = _fit(heterogeneous_panel(n_units=8, seed=15))
        assert set(res.individual) == set(res.theta_i.index)
        one = res.individual["u03"]
        assert one.longrun.loc["x", "theta"] == pytest.approx(
            res.theta_i.loc["u03", "x"], rel=1e-12
        )

    def test_non_adjusting_units_are_named_not_dropped(self) -> None:
        """lambda_i >= 0 : l'individu ne revient vers aucun equilibre, et
        son theta_i n'est pas un coefficient de long terme au sens
        moyenne. Le retirer serait selectionner sur le resultat ; il est
        garde et nomme."""
        df = pd.concat(
            [
                heterogeneous_panel(n_units=8, seed=16),
                heterogeneous_panel(
                    n_units=2, seed=17, lambda_bar=0.02, lambda_sd=0.005
                ).assign(id=lambda d: d["id"].str.replace("u", "b")),
            ],
            ignore_index=True,
        )
        res = _fit(df)
        if len(res.non_adjusting):
            assert res.adjustment["share_non_adjusting"] > 0
            assert "lambda_i >= 0" in res.summary()
            assert set(res.non_adjusting) <= set(res.theta_i.index)

    def test_non_adjusting_warns(self) -> None:
        df = heterogeneous_panel(
            n_units=6, seed=18, lambda_bar=0.05, lambda_sd=0.01, noise=0.2
        )
        with pytest.warns(PyardlMethodologyWarning, match="do not error-correct"):
            MeanGroup(df, y="y", X=["x"], id="id", time="t", order=(1, 1)).fit()

    def test_heterogeneity_table(self) -> None:
        res = _fit(heterogeneous_panel(n_units=20, seed=19))
        table = res.heterogeneity()
        assert list(table.columns) == ["mean", "sd", "min", "median", "max", "cv"]
        assert table.loc["x", "sd"] > 0

    def test_summary_names_the_variance_construction(self) -> None:
        """Un lecteur doit savoir d'ou vient l'erreur type sans lire le
        code."""
        text = _fit(heterogeneous_panel(n_units=12, seed=20)).summary()
        assert "BETWEEN-individual" in text
        assert "not pooled within-individual" in text
        assert "Mean Group estimation (Pesaran & Smith 1995)" in text


class TestRobustness:
    def test_row_order_does_not_change_the_estimates(self) -> None:
        """§3.3 — invariance a l'ordre des individus et des lignes."""
        df = heterogeneous_panel(n_units=10, seed=21)
        shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
        a, b = _fit(df), _fit(shuffled)
        assert a.longrun.loc["x", "theta"] == pytest.approx(
            b.longrun.loc["x", "theta"], rel=1e-12
        )
        assert a.longrun.loc["x", "se"] == pytest.approx(
            b.longrun.loc["x", "se"], rel=1e-12
        )

    def test_unbalanced_panel_is_estimated(self) -> None:
        df = heterogeneous_panel(n_units=10, n_obs=60, seed=22)
        df = df[~((df["id"] == "u03") & (df["t"] >= 35))]
        res = _fit(df)
        assert res.n_units == 10
        assert res.panel.unbalanced

    def test_failed_individual_is_reported_not_hidden(self) -> None:
        df = heterogeneous_panel(n_units=8, seed=23)
        res = _fit(df)
        assert isinstance(res.failed, dict)
        assert res.n_units + len(res.failed) == res.panel.n_units


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"aggregator": "geometric"}, "aggregator must be"),
            ({"trim": 0.6}, "trim must be in"),
            ({"det": "quadratic"}, "det must be"),
            ({"order": (0, 1)}, "p must be at least 1"),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        df = heterogeneous_panel(n_units=6, seed=24)
        with pytest.raises(ValueError, match=match):
            MeanGroup(df, y="y", X=["x"], id="id", time="t", **kwargs)

    def test_single_individual_has_no_standard_error(self) -> None:
        """Avec un seul individu il n'y a aucune dispersion
        inter-individus, donc aucune erreur type. Rendre un point sans
        intervalle inviterait a le lire comme s'il en avait un."""
        df = heterogeneous_panel(n_units=1, seed=25)
        with pytest.raises(ValueError, match="dispersion ACROSS individuals"):
            MeanGroup(df, y="y", X=["x"], id="id", time="t", order=(1, 1)).fit()

    def test_results_are_immutable(self) -> None:
        res = _fit(heterogeneous_panel(n_units=6, seed=26))
        with pytest.raises(AttributeError):
            res.longrun = pd.DataFrame()  # type: ignore[misc]


class TestPlot:
    def test_plot_requires_matplotlib_or_draws(self) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        res = _fit(heterogeneous_panel(n_units=10, seed=27))
        axes = res.plot_heterogeneity()
        assert len(np.atleast_1d(axes)) == 1
