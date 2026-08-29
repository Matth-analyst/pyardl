"""Le backend natif contre le chemin NumPy — equivalence, pas confiance.

CE QUE CE FICHIER VERROUILLE
----------------------------
Le noyau Rust ne remplace pas NumPy : il doit produire **exactement** ce
que NumPy produit. Les innovations sont tirees cote Python et passees au
noyau, qui ne tire rien lui-meme ; les deux implementations voient donc
les memes nombres et n'ont aucune excuse pour diverger au-dela de
l'ordre de sommation.

C'est pour cela que le verrou principal est une EGALITE a 1e-12, et non
le test de Kolmogorov-Smirnov que l'architecture prevoyait. Le KS est
present aussi, applique aux distributions bootstrap de bout en bout,
parce qu'il repond a une autre question : que la substitution ne deplace
pas la loi des decisions. Mais il ne pouvait pas etre le verrou — un KS
sur 2000 points ne distingue pas deux lois qui different de 1e-9, et
aurait laisse passer une faute de signe sur un coefficient rarement
actif.

Ces tests SAUTENT quand le noyau n'est pas compile, ce qui est le cas
par defaut : `pip install pyardl` n'installe aucune chaine Rust. Ils ne
sautent pas en silence pour autant — la raison de l'import manque est
rapportee.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyardl import backend
from pyardl.bootstrap import bootstrap_bounds_test
from pyardl.bootstrap.dgp import estimate_null_dgp, simulate_paths
from pyardl.bootstrap.resample import resample_residuals
from pyardl.simulate import degenerate_system, vecm_ardl

needs_rust = pytest.mark.skipif(
    not backend.rust_available(),
    reason=f"native kernel not built ({backend.why_unavailable()}); "
    "build it with `python rust/build.py`",
)


def _null_dgp(n_obs: int, k: int, case: int, p: int, q: tuple[int, ...], seed: int):
    alpha, beta = degenerate_system(None, k=k, speed=-0.4)
    sim = vecm_ardl(n_obs, alpha=alpha, beta=beta, seed=seed)
    y = sim.y.to_numpy()
    x = sim.x.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dgp = estimate_null_dgp(y, x, p=p, q=q, case=case)
    return dgp, y, x


def _innovations(dgp, n_periods: int, n_rep: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [resample_residuals(dgp.residuals, n_periods, rng) for _ in range(n_rep)]
    )


class TestResolution:
    """Quel backend repond, et que dit-il quand il ne peut pas."""

    def test_numpy_is_always_available(self) -> None:
        assert backend.resolve("numpy") == "numpy"

    def test_auto_never_fails(self) -> None:
        assert backend.resolve("auto") in ("numpy", "rust")

    def test_unknown_backend_is_refused(self) -> None:
        with pytest.raises(ValueError, match="backend must be one of"):
            backend.resolve("cuda")

    def test_explicit_rust_does_not_fall_back_silently(self) -> None:
        """Demander rust et obtenir numpy sans le savoir fausse une mesure.

        Quelqu'un qui chronometre une acceleration doit savoir laquelle
        des deux implementations il vient de chronometrer. `auto` est la
        pour qui veut le repli.
        """
        if backend.rust_available():
            assert backend.resolve("rust") == "rust"
        else:
            with pytest.raises(ImportError, match="rust/build.py"):
                backend.resolve("rust")

    def test_why_unavailable_is_informative(self) -> None:
        reason = backend.why_unavailable()
        if backend.rust_available():
            assert reason is None
        else:
            assert reason and "_rust" in reason

    @needs_rust
    def test_thread_count(self) -> None:
        assert backend.thread_count() >= 1


@needs_rust
class TestExactEquivalence:
    """Memes innovations, memes trajectoires — a l'arrondi pres."""

    @pytest.mark.parametrize(
        ("k", "case", "p", "q"),
        [
            (1, 3, 2, (2,)),
            (3, 3, 2, (2, 2, 2)),
            (2, 5, 3, (1, 3)),
            (1, 1, 1, (1,)),
            (2, 2, 2, (2, 2)),
            (1, 4, 3, (3,)),
        ],
    )
    def test_trajectories_agree(self, k: int, case: int, p: int, q) -> None:
        """Les cinq cas deterministes, des q ragged, et plusieurs k.

        Le cas 5 porte une tendance et le cas 4 une tendance restreinte :
        ce sont les deux ou un decalage d'indice sur `t - burn_in + 1`
        passerait inapercu sur les autres.
        """
        dgp, y, x = _null_dgp(200, k, case, p, q, seed=11)
        inn = _innovations(dgp, 250, 40, seed=5)
        common = dict(y0=float(y[0]), x0=x[0], burn_in=50)
        y_np, x_np = simulate_paths(dgp, inn, backend="numpy", **common)
        y_rs, x_rs = simulate_paths(dgp, inn, backend="rust", **common)
        assert y_np.shape == y_rs.shape
        assert x_np.shape == x_rs.shape
        assert np.max(np.abs(y_np - y_rs)) < 1e-12
        assert np.max(np.abs(x_np - x_rs)) < 1e-12

    def test_the_gap_is_rounding_and_not_drift(self) -> None:
        """Un ecart qui s'accumulerait signalerait une formule differente.

        NumPy somme par paires, le noyau sequentiellement : les derniers
        bits divergent, et c'est tout ce qui doit diverger. Si l'ecart
        croissait avec l'horizon, ce ne serait plus de l'arrondi.
        """
        dgp, y, x = _null_dgp(400, 2, 3, 2, (2, 2), seed=13)
        inn = _innovations(dgp, 450, 20, seed=6)
        common = dict(y0=float(y[0]), x0=x[0], burn_in=50)
        y_np, _ = simulate_paths(dgp, inn, backend="numpy", **common)
        y_rs, _ = simulate_paths(dgp, inn, backend="rust", **common)
        gap = np.abs(y_np - y_rs)
        scale = np.maximum(np.abs(y_np), 1.0)
        assert np.max(gap / scale) < 1e-13

    def test_auto_matches_the_explicit_choice(self) -> None:
        dgp, y, x = _null_dgp(150, 1, 3, 2, (2,), seed=17)
        inn = _innovations(dgp, 200, 10, seed=8)
        common = dict(y0=float(y[0]), x0=x[0], burn_in=50)
        chosen = backend.resolve("auto")
        a = simulate_paths(dgp, inn, backend="auto", **common)
        b = simulate_paths(dgp, inn, backend=chosen, **common)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])


@needs_rust
class TestEndToEnd:
    """Le test complet, pas seulement le noyau."""

    @staticmethod
    def _run(be: str, n_boot: int = 1999):
        alpha, beta = degenerate_system(None, k=2, speed=-0.4)
        sim = vecm_ardl(200, alpha=alpha, beta=beta, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bootstrap_bounds_test(
                sim.y,
                sim.x,
                case=3,
                order=(2, 2),
                n_boot=n_boot,
                seed=42,
                backend=be,
                store_distribution=True,
            )

    def test_critical_values_and_pvalues_agree(self) -> None:
        """Meme graine, memes innovations, donc memes quantiles.

        C'est plus exigeant qu'une concordance distributionnelle : les
        deux runs doivent tomber sur le meme nombre, pas seulement sur
        la meme loi.
        """
        a, b = self._run("numpy"), self._run("rust")
        assert a.f_stat == pytest.approx(b.f_stat, rel=1e-12)
        for level in a.f_critical:
            assert a.f_critical[level] == pytest.approx(b.f_critical[level], rel=1e-10)
            assert a.t_critical[level] == pytest.approx(b.t_critical[level], rel=1e-10)
        assert a.f_pvalue == pytest.approx(b.f_pvalue, abs=1e-12)
        assert a.t_pvalue == pytest.approx(b.t_pvalue, abs=1e-12)
        assert a.n_failed == b.n_failed

    def test_kolmogorov_smirnov_on_the_bootstrap_distributions(self) -> None:
        """Le controle que l'architecture demande, p > 0.99.

        Il est ici pour ce qu'il verifie vraiment — que substituer le
        noyau ne deplace pas la loi des statistiques — et non comme
        verrou principal : sur des echantillons identiques il ne peut
        que renvoyer 1.0, ce qui est justement l'enonce voulu.
        """
        from scipy import stats

        a, b = self._run("numpy"), self._run("rust")
        assert a.distribution is not None and b.distribution is not None
        assert list(a.distribution.columns) == ["F", "t", "F_indep"]
        for key in a.distribution.columns:
            left = np.asarray(a.distribution[key], dtype=float)
            right = np.asarray(b.distribution[key], dtype=float)
            assert stats.ks_2samp(left, right).pvalue > 0.99

    def test_the_decision_is_the_same(self) -> None:
        a, b = self._run("numpy", n_boot=999), self._run("rust", n_boot=999)
        assert a.classification(0.05) == b.classification(0.05)


@needs_rust
class TestFallback:
    """Ce que le noyau ne couvre pas doit retomber sur NumPy, sans bruit."""

    def test_a_python_callback_falls_back(self) -> None:
        """La decomposition NARDL passe `expand`, appele a chaque periode.

        Le faire traverser la frontiere mille fois couterait plus que la
        boucle n'economise. Le repli est une decision de performance,
        pas un avertissement methodologique : il est donc silencieux, et
        le resultat doit rester celui de NumPy.
        """
        dgp, y, x = _null_dgp(150, 1, 3, 2, (2,), seed=19)
        inn = _innovations(dgp, 200, 8, seed=3)
        common = dict(y0=float(y[0]), x0=x[0], burn_in=50)

        def identity(block: np.ndarray) -> np.ndarray:
            return block

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            asked = simulate_paths(dgp, inn, expand=identity, backend="rust", **common)
        reference = simulate_paths(dgp, inn, expand=identity, backend="numpy", **common)
        assert np.array_equal(asked[0], reference[0])
        assert np.array_equal(asked[1], reference[1])

    def test_single_path_wrapper_accepts_both(self) -> None:
        from pyardl.bootstrap.dgp import simulate_path

        dgp, y, x = _null_dgp(150, 1, 3, 2, (2,), seed=21)
        rng = np.random.default_rng(4)
        inn = resample_residuals(dgp.residuals, 200, rng)
        out = simulate_path(dgp, inn, y0=float(y[0]), x0=x[0], burn_in=50)
        assert out[0].shape == (150,)


def test_numpy_stays_the_default() -> None:
    """Le defaut ne doit pas changer parce que le noyau existe.

    L'architecture le dit : `backend='numpy'` tant que l'equivalence
    n'est pas etablie, et meme apres, c'est NumPy qui est la reference
    contre laquelle le noyau est verifie. Un defaut qui basculerait tout
    seul ferait de cet accord une tautologie.
    """
    import inspect

    for func in (simulate_paths, bootstrap_bounds_test):
        assert inspect.signature(func).parameters["backend"].default == "numpy"
