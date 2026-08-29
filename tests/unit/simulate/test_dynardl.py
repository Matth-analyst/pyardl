"""Spec 25 §4 — simulations dynamiques d'ARDL (Jordan & Philips 2018).

Une simulation ne se teste pas sur l'allure de sa figure. Elle se teste
contre l'algèbre qu'elle est censée rendre lisible : un choc permanent
doit conduire exactement au coefficient de long terme (spec 03), une
impulsion doit revenir à l'équilibre à la vitesse de la demi-vie, et la
réponse d'un NARDL doit reproduire au bit près les multiplicateurs
dynamiques déjà implémentés (spec 17). Deux chemins vers le même objet
qui divergent, c'est l'un des deux qui est faux.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.core.ardl import ARDL
from pyardl.core.transforms import half_life
from pyardl.exceptions import PyardlMethodologyWarning
from pyardl.nardl import NARDL
from pyardl.simulate import DynardlSimulation, dynardl_simulate


def _stationary_ardl(n: int = 250, seed: int = 0, phi: float = 0.6):
    """ARDL(1, 1) stationnaire, avec un vrai long terme à retrouver."""
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(size=n).cumsum() + 5.0, name="x")
    e = rng.normal(scale=0.5, size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 1.0 + phi * y[t - 1] + 0.8 * x.iloc[t] - 0.3 * x.iloc[t - 1] + e[t]
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


@pytest.fixture(scope="module")
def fitted():
    y, x = _stationary_ardl()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARDL(y, x, order=(1, 1), det="const").fit()


class TestEquilibrium:
    """Spec 25 §4.1 — le test fondamental : simulation contre algèbre."""

    def test_counterfactual_branch_is_exactly_flat(self, fitted) -> None:
        """Sans choc, rien ne bouge — et pas seulement à peu près.

        Chaque tirage part de SON propre équilibre, donc la branche
        sans choc est plate exactement, tirage par tirage. Une tolérance
        de 1e-6 passerait aussi avec un départ approximatif ; 1e-12 ne
        passe que si le point de départ est le bon.
        """
        sim = fitted.dynardl_simulate("x", size=1.0, t0=10, horizon=40, r=200, seed=1)
        before = sim.summary_df["response"].iloc[:10]
        assert np.max(np.abs(before.to_numpy())) == 0.0
        level = sim.summary_df[("level", "point")].iloc[:10].to_numpy()
        assert np.allclose(level, sim.equilibrium, atol=1e-12)

    def test_step_converges_to_the_longrun_coefficient(self, fitted) -> None:
        """Spec 25 §4.1 : un pas permanent conduit à theta * dx."""
        theta = float(fitted.longrun.loc["x", "theta"])
        sim = fitted.dynardl_simulate("x", size=2.5, t0=5, horizon=200, r=50, seed=2)
        final = float(sim.summary_df[("response", "point")].iloc[-1])
        assert final == pytest.approx(theta * 2.5, abs=1e-6)
        assert sim.longrun_target == pytest.approx(theta * 2.5, rel=1e-12)

    def test_impulse_returns_to_the_initial_equilibrium(self, fitted) -> None:
        """Spec 25 §4.2 : une impulsion ne déplace pas l'équilibre."""
        sim = fitted.dynardl_simulate(
            "x", shock_type="impulse", size=1.0, t0=5, horizon=200, r=50, seed=3
        )
        assert float(sim.summary_df[("response", "point")].iloc[-1]) == pytest.approx(
            0.0, abs=1e-8
        )
        assert np.isnan(sim.longrun_target)

    def test_impulse_decay_matches_the_half_life(self) -> None:
        """Spec 25 §4.2 : la vitesse est celle qu'annonce ``half_life``.

        Sur un ARDL(1, 0) la réponse à une impulsion vaut beta * phi^h,
        donc à h = demi-vie elle vaut exactement la moitié du pic. Le
        test relie la figure au chiffre que la table de résultats
        affiche déjà, ce qui interdit qu'ils se contredisent.
        """
        y, x = _stationary_ardl(n=300, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARDL(y, x, order=(1, 0), det="const").fit()
        hl = half_life(res.ardl_params)
        sim = res.dynardl_simulate(
            "x", shock_type="impulse", size=1.0, t0=0, horizon=80, r=10, seed=4
        )
        path = sim.summary_df[("response", "point")].to_numpy()
        phi = float(res.params["y.L1"])
        assert path[0] * phi**hl == pytest.approx(path[0] / 2.0, rel=1e-10)
        # ...et la trajectoire simulée suit bien cette géométrique.
        expected = path[0] * phi ** np.arange(81)
        assert np.allclose(path, expected, atol=1e-10)


class TestInnovations:
    """Ce que ``stochastic=True`` change, et ce qu'il ne change pas."""

    def test_response_is_invariant_to_the_innovations(self, fitted) -> None:
        """Le modèle est linéaire en y : la différence appariée les efface.

        Ce n'est pas une approximation Monte Carlo, c'est une identité.
        La tolérance est donc celle de l'arithmétique flottante, pas
        celle d'un tirage.
        """
        kwargs = dict(size=1.0, t0=5, horizon=40, r=300, seed=5)
        quiet = fitted.dynardl_simulate("x", stochastic=False, **kwargs)
        noisy = fitted.dynardl_simulate("x", stochastic=True, **kwargs)
        assert np.allclose(
            quiet.summary_df["response"].to_numpy(),
            noisy.summary_df["response"].to_numpy(),
            atol=1e-12,
        )

    def test_innovations_widen_the_band_on_the_level(self, fitted) -> None:
        """L'incertitude de prévision se voit sur le niveau, seul endroit
        où elle a un sens."""
        kwargs = dict(size=1.0, t0=5, horizon=40, r=500, seed=6)
        quiet = fitted.dynardl_simulate("x", stochastic=False, **kwargs)
        noisy = fitted.dynardl_simulate("x", stochastic=True, **kwargs)

        def width(sim: DynardlSimulation) -> float:
            block = sim.summary_df["level"]
            return float((block["hi_95"] - block["lo_95"]).iloc[-1])

        assert width(noisy) > width(quiet)


class TestAgreementWithMultipliers:
    """Spec 25 §3 — cohérence avec les multiplicateurs de la spec 17."""

    def test_nardl_step_reproduces_the_dynamic_multipliers(self) -> None:
        rng = np.random.default_rng(3)
        n = 250
        e = rng.normal(size=n)
        x = np.cumsum(rng.normal(size=n))
        y = np.zeros(n)
        for t in range(1, n):
            dx = x[t] - x[t - 1]
            y[t] = 0.6 * y[t - 1] + 0.8 * max(dx, 0.0) + 0.3 * min(dx, 0.0) + e[t]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = NARDL(
                pd.Series(y, name="y"),
                pd.DataFrame({"x": x}),
                asym="x",
                order=(1, 1),
            ).fit()
        mult = res.dynamic_multipliers(h=30, r=20, seed=1)["m_pos"].to_numpy()
        sim = res.dynardl_simulate("x_pos", size=1.0, t0=0, horizon=30, r=20, seed=1)
        assert np.allclose(
            sim.summary_df[("response", "point")].to_numpy(), mult, atol=1e-10
        )


class TestScenario:
    """Le point de départ, l'amplitude, et les tirages fournis."""

    def test_size_1sd_uses_the_sample_standard_deviation(self, fitted) -> None:
        sd = float(np.std(fitted.model._x[:, 0], ddof=1))
        sim = fitted.dynardl_simulate("x", size="1sd", horizon=20, r=10, seed=8)
        assert sim.shock_size == pytest.approx(sd, rel=1e-12)

    def test_start_last_moves_the_baseline(self, fitted) -> None:
        at_mean = fitted.dynardl_simulate("x", size=1.0, horizon=20, r=10, seed=9)
        at_last = fitted.dynardl_simulate(
            "x", size=1.0, horizon=20, r=10, seed=9, start="last"
        )
        assert at_mean.equilibrium != pytest.approx(at_last.equilibrium)
        # Le niveau de départ change, la réponse non : elle ne dépend
        # que des coefficients, pas de l'endroit d'où l'on part.
        assert np.allclose(
            at_mean.summary_df["response"].to_numpy(),
            at_last.summary_df["response"].to_numpy(),
            atol=1e-10,
        )

    def test_scenario_overrides_one_regressor(self, fitted) -> None:
        base = fitted.dynardl_simulate("x", size=1.0, horizon=20, r=10, seed=10)
        moved = fitted.dynardl_simulate(
            "x", size=1.0, horizon=20, r=10, seed=10, scenario={"x": 0.0}
        )
        assert base.equilibrium != pytest.approx(moved.equilibrium)

    def test_param_draws_are_used_verbatim(self, fitted) -> None:
        """Le crochet bootstrap : des tirages fournis remplacent la normale."""
        draws = np.tile(fitted._params, (7, 1))
        sim = fitted.dynardl_simulate(
            "x", size=1.0, horizon=30, r=1000, seed=11, param_draws=draws
        )
        assert sim.n_draws == 7
        block = sim.summary_df["response"]
        # Tous les tirages identiques au point : la bande est un trait.
        assert np.allclose(block["lo_95"], block["hi_95"], atol=1e-12)
        assert np.allclose(block["mean"], block["point"], atol=1e-12)


class TestDeterministics:
    """Tendance, saisonnalité et régresseurs fixes traversent la récursion."""

    def test_trend_drifts_the_level_but_not_the_response(self) -> None:
        y, x = _stationary_ardl(n=250, seed=12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARDL(y, x, order=(1, 1), det="trend").fit()
        theta = float(res.longrun.loc["x", "theta"])
        sim = res.dynardl_simulate("x", size=1.0, t0=5, horizon=200, r=20, seed=13)
        level = sim.summary_df[("level", "point")].to_numpy()
        assert abs(level[-1] - level[0]) > 1e-6  # la tendance avance
        assert float(sim.summary_df[("response", "point")].iloc[-1]) == pytest.approx(
            theta, abs=1e-6
        )

    def test_seasonal_dummies_cancel_from_the_response(self) -> None:
        """Le test qui a trouve OBS-25 : la reponse simulee et le theta
        algebrique doivent coincider meme quand le design porte des
        dummies saisonnieres entre la constante et les retards."""
        y, x = _stationary_ardl(n=240, seed=14)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARDL(y, x, order=(1, 1), det="const", seasonal=True).fit()
        theta = float(res.longrun.loc["x", "theta"])
        sim = res.dynardl_simulate("x", size=1.0, t0=4, horizon=300, r=20, seed=15)
        # Les dummies font osciller le niveau ; la réponse est une
        # différence appariée et les efface.
        assert float(sim.summary_df[("response", "point")].iloc[-1]) == pytest.approx(
            theta, abs=1e-6
        )

    def test_fixed_regressors_simulate_without_a_longrun_target(self) -> None:
        """Un régresseur fixe n'a pas de vue ECM — la simulation, si.

        ``longrun`` refuse de répondre dans ce cas (ce n'est ni un phi ni
        un beta), et la bonne réaction est de laisser la trajectoire se
        calculer en annonçant qu'il n'y a pas de cible à tracer, pas
        d'inventer une valeur de repli.
        """
        y, x = _stationary_ardl(n=240, seed=14)
        z = pd.DataFrame({"dummy": np.arange(240) % 17 == 0}, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARDL(y, x, order=(1, 1), det="const", fixed_regressors=z).fit()
        sim = res.dynardl_simulate("x", size=1.0, t0=4, horizon=300, r=20, seed=15)
        assert np.isnan(sim.longrun_target)
        path = sim.summary_df[("response", "point")].to_numpy()
        assert path[-1] == pytest.approx(path[-2], abs=1e-9)  # elle a convergé
        assert abs(path[-1]) > 1e-6


class TestGuards:
    """Ce qui doit refuser de tourner plutôt que de rendre un chiffre."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"shock_type": "ramp"}, "shock_type"),
            ({"horizon": 0}, "horizon must be at least 1"),
            ({"t0": 99}, "must lie in"),
            ({"r": 1}, "at least 2"),
            ({"bands": (0,)}, "percentages"),
            ({"bands": ()}, "percentages"),
            ({"size": "2sd"}, "1sd"),
            ({"start": "middle"}, "mean"),
        ],
    )
    def test_argument_validation(self, fitted, kwargs, message) -> None:
        call = {"size": 1.0, "horizon": 20, "r": 10, "seed": 0}
        call.update(kwargs)
        with pytest.raises(ValueError, match=message):
            fitted.dynardl_simulate("x", **call)

    def test_unknown_shock_variable(self, fitted) -> None:
        with pytest.raises(KeyError, match="not a distributed-lag regressor"):
            fitted.dynardl_simulate("zzz", size=1.0, horizon=20, r=10)

    def test_unknown_scenario_key(self, fitted) -> None:
        with pytest.raises(KeyError, match="not a regressor"):
            fitted.dynardl_simulate(
                "x", size=1.0, horizon=20, r=10, scenario={"zzz": 1.0}
            )

    def test_param_draws_wrong_width(self, fitted) -> None:
        with pytest.raises(ValueError, match="columns for"):
            fitted.dynardl_simulate(
                "x", size=1.0, horizon=20, r=10, param_draws=np.zeros((5, 2))
            )

    def test_unit_root_has_no_equilibrium(self, fitted) -> None:
        """Sum(phi) = 1 : la récursion ne converge vers rien.

        Une division par 1 - sum(phi) rendrait un infini sans rien dire ;
        le message doit nommer la raison.
        """
        params = fitted._params.copy()
        names = list(fitted._param_names)
        params[names.index("y.L1")] = 1.0
        broken = dataclasses.replace(fitted, _params=params)
        with pytest.raises(ValueError, match="no equilibrium"):
            broken.dynardl_simulate("x", size=1.0, horizon=20, r=10)

    def test_unstable_model_warns(self, fitted) -> None:
        params = fitted._params.copy()
        names = list(fitted._param_names)
        params[names.index("y.L1")] = 1.05
        broken = dataclasses.replace(fitted, _params=params)
        with pytest.warns(PyardlMethodologyWarning, match="not stable"):
            broken.dynardl_simulate("x", size=1.0, horizon=10, r=10, seed=0)


class TestPresentation:
    """La sortie lisible, et la figure."""

    def test_summary_reports_the_setup_and_the_target(self, fitted) -> None:
        sim = fitted.dynardl_simulate("x", size=1.0, horizon=60, r=100, seed=16)
        text = sim.summary()
        assert "step shock" in text
        assert "seed = 16" in text
        assert "long-run target" in text
        assert "cancel out of the response" in text

    def test_seed_is_recorded_when_drawn(self, fitted) -> None:
        sim = fitted.dynardl_simulate("x", size=1.0, horizon=10, r=10)
        assert isinstance(sim.seed, int)
        again = fitted.dynardl_simulate("x", size=1.0, horizon=10, r=10, seed=sim.seed)
        assert np.allclose(
            sim.summary_df.to_numpy(), again.summary_df.to_numpy(), atol=1e-12
        )

    def test_plot(self, fitted) -> None:
        plt = pytest.importorskip("matplotlib.pyplot")
        sim = fitted.dynardl_simulate("x", size=1.0, horizon=30, r=50, seed=17)
        for block in ("response", "level"):
            fig = sim.plot(block=block)
            assert fig is not None
            plt.close(fig)
        fig = sim.plot(bands=(95,))
        plt.close(fig)
        impulse = fitted.dynardl_simulate(
            "x", shock_type="impulse", size=1.0, horizon=30, r=50, seed=18
        )
        fig = impulse.plot()
        plt.close(fig)
        with pytest.raises(ValueError, match="block must be"):
            sim.plot(block="nope")
        with pytest.raises(ValueError, match="were not computed"):
            sim.plot(bands=(99,))


def test_module_level_function_matches_the_method(fitted) -> None:
    """La méthode n'est qu'un raccourci : mêmes chiffres, même graine."""
    a = dynardl_simulate(fitted, "x", size=1.0, horizon=20, r=25, seed=19)
    b = fitted.dynardl_simulate("x", size=1.0, horizon=20, r=25, seed=19)
    assert np.allclose(a.summary_df.to_numpy(), b.summary_df.to_numpy(), atol=1e-12)


def _coverage_study(n_mc: int, seed: int) -> float:
    """Part des replications ou la VRAIE reponse tombe dans la bande 95 %.

    Le DGP a des coefficients connus, donc la vraie trajectoire se
    calcule par la meme recursion appliquee aux vrais parametres. C'est
    la seule facon de verifier une bande : la comparer a ce qu'elle est
    censee contenir, pas a une autre bande.
    """
    phi, b0, b1, t0, horizon = 0.6, 0.8, -0.3, 5, 40
    truth = 0.0
    for t in range(t0, horizon + 1):
        truth = phi * truth + b0 + (b1 if t > t0 else 0.0)
    rng = np.random.default_rng(seed)
    hits = 0
    for m in range(n_mc):
        y, x = _stationary_ardl(n=200, seed=int(rng.integers(1 << 31)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARDL(y, x, order=(1, 1), det="const").fit()
            sim = res.dynardl_simulate(
                "x", size=1.0, t0=t0, horizon=horizon, r=400, seed=seed + m
            )
        block = sim.summary_df["response"]
        lo = float(block["lo_95"].iloc[horizon])
        hi = float(block["hi_95"].iloc[horizon])
        hits += lo <= truth <= hi
    return hits / n_mc


@pytest.mark.fast_mc
def test_band_coverage_fast() -> None:
    """Spec 25 §4.3 — version CI.

    Dimensionnement (regle 10) : a 200 replications l'erreur type d'un
    taux voisin de 95 % vaut 1.5 point. La borne inferieure est donc
    posee a 90 %, soit trois erreurs types sous le nominal : ce test
    detecte un effondrement de la couverture, PAS un ecart de deux
    points. C'est l'etude complete de ``validation/spec25_montecarlo.py``
    (1000 replications, erreur type 0.69 point) qui tranche la question
    fine, et elle mesure 95.0 % a l'horizon de long terme.
    """
    assert _coverage_study(n_mc=200, seed=20260829) >= 0.90


@pytest.mark.slow
def test_band_coverage_full() -> None:
    """Version complete : la bande doit tenir [93, 97] %."""
    coverage = _coverage_study(n_mc=1000, seed=20260829)
    assert 0.93 <= coverage <= 0.97
