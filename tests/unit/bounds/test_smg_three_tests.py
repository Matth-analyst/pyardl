"""Spec 15 (Sam, McNown & Goh 2019) — plan de tests §3.

Le cadre à trois tests se juge sur une seule question : sait-il
distinguer une vraie cointégration des deux dégénérescences ? Les quatre
DGP canoniques ci-dessous en fabriquent une chacun, et la classification
doit les nommer correctement.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pyardl.bounds import bounds_test
from pyardl.bounds.pss import _wald_f, _wald_f_indep
from pyardl.critical_values.smg2019 import MAX_K_FINDEP, findep_bounds

# --- Les quatre DGP canoniques (spec 15 §3.1) --------------------------


def _dgp_cointegration(seed: int, n: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    """lambda != 0 ET gamma != 0 : la vraie cointégration."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] - 0.4 * (y[t - 1] - (1.0 + x[t - 1])) + rng.normal(scale=0.5)
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


def _dgp_degenerate_1(seed: int, n: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    """lambda != 0, gamma = 0 : y revient vers son propre passé.

    y est stationnaire autour d'une constante et x est une marche
    aléatoire indépendante. Le F global rejette (lambda porte tout), le t
    rejette, mais les niveaux de x ne portent rien.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * y[t - 1] + rng.normal(scale=0.5)
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


def _dgp_degenerate_2(seed: int, n: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    """gamma != 0, lambda = 0 : aucun rappel vers l'équilibre.

    Delta y_t = 0.5 x_{t-1} + eps : le niveau de x est significatif, mais
    rien ne ramène y.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] + 0.5 * x[t - 1] + rng.normal(scale=0.5)
    return pd.Series(y, name="y"), pd.DataFrame({"x": x})


def _dgp_no_cointegration(seed: int, n: int = 300) -> tuple[pd.Series, pd.DataFrame]:
    """lambda = gamma = 0 : deux marches aléatoires indépendantes."""
    rng = np.random.default_rng(seed)
    return (
        pd.Series(np.cumsum(rng.normal(size=n)), name="y"),
        pd.DataFrame({"x": np.cumsum(rng.normal(size=n))}),
    )


def _classify_dgp(dgp, seed: int) -> str:  # type: ignore[no-untyped-def]
    y, x = dgp(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = bounds_test(y, x, case=3, order=(1, 1))
    return res.classification()[0]


# --- La statistique F_indep --------------------------------------------


class TestFIndepStatistic:
    """§2.1 — F_indep porte sur les gamma seuls."""

    def test_excludes_lambda_from_the_restriction(self) -> None:
        """La restriction testée compte une contrainte de moins que le F
        global : celle sur lambda."""
        y, x = _dgp_cointegration(seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(y, x, case=3, order=(1, 1))
        assert res.f_indep_stat == pytest.approx(_wald_f_indep(res._fit))
        assert len(res._fit.tested) == 2  # lambda + un gamma
        assert res.f_indep_stat != pytest.approx(res.f_stat)

    def test_matches_the_overall_f_when_lambda_is_the_only_extra(self) -> None:
        """Contrôle algébrique : le F global sur les mêmes deux
        restrictions n'est PAS le F_indep, mais les deux sortent de la
        même matrice de covariance — leur rapport est fini et positif."""
        y, x = _dgp_cointegration(seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(y, x, case=3, order=(1, 1))
        assert _wald_f(res._fit) > 0
        assert res.f_indep_stat > 0

    def test_multiple_regressors_average_over_k_restrictions(self) -> None:
        """Avec k régresseurs, F_indep est un F par restriction : il
        divise par k, pas par k+1."""
        rng = np.random.default_rng(7)
        n = 300
        x1 = np.cumsum(rng.normal(size=n))
        x2 = np.cumsum(rng.normal(size=n))
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = (
                y[t - 1]
                - 0.4 * (y[t - 1] - (1.0 + x1[t - 1] - 0.5 * x2[t - 1]))
                + rng.normal(scale=0.5)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bounds_test(
                pd.Series(y, name="y"),
                pd.DataFrame({"x1": x1, "x2": x2}),
                case=3,
                order=(1, 1),
            )
        fit = res._fit
        names = [n_ for n_ in fit.tested if n_ != fit.lam_name]
        assert len(names) == 2
        idx = [fit.names.index(n_) for n_ in names]
        theta = fit.params.iloc[idx].to_numpy()
        vcv = fit.cov[np.ix_(idx, idx)]
        expected = float(theta @ np.linalg.solve(vcv, theta)) / 2
        assert res.f_indep_stat == pytest.approx(expected, rel=1e-12)


# --- Les bornes simulées ------------------------------------------------


class TestFIndepBounds:
    """§2.2 — la table simulée, et ses refus explicites."""

    def test_bounds_are_ordered(self) -> None:
        lo, up = findep_bounds(3, 2, 0.05)
        assert lo <= up

    def test_stricter_level_gives_a_higher_bound(self) -> None:
        assert findep_bounds(3, 2, 0.01)[1] > findep_bounds(3, 2, 0.10)[1]

    def test_out_of_grid_raises_rather_than_substituting(self) -> None:
        """Aucune valeur voisine n'est substituée en silence."""
        with pytest.raises(ValueError, match="simulated for k"):
            findep_bounds(3, MAX_K_FINDEP + 1, 0.05)
        with pytest.raises(ValueError, match="case must be"):
            findep_bounds(6, 2, 0.05)

    def test_unavailable_bound_gives_no_decision(self) -> None:
        """Hors couverture, decision_indep vaut None et la
        classification refuse de conclure — elle ne devine pas."""
        from pyardl.bounds.pss import _findep_decision

        assert _findep_decision(3.0, 3, MAX_K_FINDEP + 1, 0.05) is None


# --- La classification sur les quatre DGP -------------------------------


class TestCanonicalDGPs:
    """§3.1 — chaque DGP doit recevoir son nom."""

    @pytest.mark.parametrize("seed", range(3))
    def test_cointegration(self, seed: int) -> None:
        assert _classify_dgp(_dgp_cointegration, 300 + seed) == "cointegration"

    @pytest.mark.parametrize("seed", range(3))
    def test_degenerate_1(self, seed: int) -> None:
        assert _classify_dgp(_dgp_degenerate_1, 400 + seed) == "degenerate_1"

    @pytest.mark.parametrize("seed", range(3))
    def test_no_cointegration(self, seed: int) -> None:
        assert _classify_dgp(_dgp_no_cointegration, 500 + seed) in (
            "no_cointegration",
            "inconclusive",
        )

    def test_degenerate_2_is_never_called_cointegration(self) -> None:
        """Le point de la spec : sans le troisième test, ce DGP passait
        pour une cointégration dès que le F rejetait."""
        for seed in range(5):
            assert _classify_dgp(_dgp_degenerate_2, 600 + seed) != "cointegration"


@pytest.mark.fast_mc
class TestClassificationRates:
    """§3.2 — taux de classification correcte sur réplications."""

    @pytest.mark.parametrize(
        ("dgp", "expected"),
        [
            (_dgp_cointegration, "cointegration"),
            (_dgp_degenerate_1, "degenerate_1"),
        ],
    )
    def test_rate_above_90_percent(self, dgp, expected: str) -> None:  # type: ignore[no-untyped-def]
        hits = sum(_classify_dgp(dgp, 1000 + s) == expected for s in range(40))
        assert hits / 40 >= 0.90
