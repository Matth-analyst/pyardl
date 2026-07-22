"""Spec 03 §6.1 — plan de tests algébriques (hors verrou n°1, cf.
test_transforms_equivalence.py) : aller-retour, long terme par
simulation, covariance par méthode delta.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyardl.core.transforms import (
    ARDLParams,
    ardl_to_ecm,
    ecm_to_ardl,
    half_life,
    longrun_coefs,
    longrun_covariance,
    speed_of_adjustment,
)
from pyardl.exceptions import DegenerateCaseWarning
from pyardl.utils import _delta_method


@pytest.mark.fast_mc
def test_round_trip_ardl_ecm_ardl() -> None:
    """Spec 03 §6.1.1 : ardl_to_ecm . ecm_to_ardl = identité (1e-12).

    1000 tirages aléatoires, p, q_j in {1..5}, k in {1..4} (conformément
    au plan de tests ; q_j = 0 est couvert séparément, cf. docs/QUESTIONS.md).
    """
    rng = np.random.default_rng(20260707)
    for _ in range(1000):
        p = int(rng.integers(1, 6))
        k = int(rng.integers(1, 5))
        q = tuple(int(v) for v in rng.integers(1, 6, size=k))
        phi = rng.uniform(-2.0, 2.0, size=p)
        beta = tuple(rng.uniform(-2.0, 2.0, size=qj + 1) for qj in q)
        const = float(rng.uniform(-5.0, 5.0))
        trend = float(rng.uniform(-1.0, 1.0))
        has_trend = bool(rng.integers(0, 2))

        params = ARDLParams(
            p=p,
            q=q,
            phi=phi,
            beta=beta,
            const=const,
            trend=trend,
            has_const=True,
            has_trend=has_trend,
        )
        roundtrip = ecm_to_ardl(ardl_to_ecm(params))

        np.testing.assert_allclose(roundtrip.phi, params.phi, atol=1e-12)
        for b0, b1 in zip(roundtrip.beta, params.beta, strict=True):
            np.testing.assert_allclose(b0, b1, atol=1e-12)


def _simulate_step_response(
    params: ARDLParams, *, shock_index: int, horizon: int
) -> float:
    """Réponse de long terme de y à un step permanent de x_{shock_index} (test only)."""
    p, q = params.p, params.q
    k = params.k
    burn = 50
    n = burn + horizon
    start = max(p, max(q) + 1)

    xs = [np.zeros(n) for _ in range(k)]
    xs[shock_index][burn:] = 1.0
    y = np.zeros(n)
    for t in range(start, n):
        val = 0.0
        for i in range(p):
            val += params.phi[i] * y[t - i - 1]
        for j in range(k):
            for i in range(q[j] + 1):
                val += params.beta[j][i] * xs[j][t - i]
        y[t] = val
    return float(y[-1])


@pytest.mark.parametrize("seed", range(5))
def test_longrun_coefs_matches_step_response_simulation(seed: int) -> None:
    """Spec 03 §6.1.3 : theta par transformation = theta par simulation (1e-6)."""
    rng = np.random.default_rng(1000 + seed)
    p = int(rng.integers(1, 3))
    k = int(rng.integers(1, 3))
    q = tuple(int(v) for v in rng.integers(1, 3, size=k))

    # phi petits : garantit la stabilité (convergence de la réponse au step).
    phi = rng.uniform(-0.15, 0.15, size=p)
    beta = tuple(rng.uniform(-1.0, 1.0, size=qj + 1) for qj in q)

    params = ARDLParams(p=p, q=q, phi=phi, beta=beta, has_const=False, has_trend=False)
    theta = longrun_coefs(params)

    for j in range(k):
        shifted = _simulate_step_response(params, shock_index=j, horizon=500)
        assert shifted == pytest.approx(theta.iloc[j], abs=1e-6)


def test_longrun_covariance_matches_numerical_delta_method() -> None:
    """Spec 03 §6.1.4 : gradient analytique de longrun_covariance = gradient
    numérique du helper générique _delta_method (1e-6)."""
    rng = np.random.default_rng(42)
    p, q = 2, (2, 1)
    phi = rng.uniform(-0.2, 0.2, size=p)
    beta = tuple(rng.uniform(-1.0, 1.0, size=qj + 1) for qj in q)
    params = ARDLParams(p=p, q=q, phi=phi, beta=beta, has_const=False, has_trend=False)

    theta_vec = params.param_vector()
    n_params = theta_vec.shape[0]
    a = rng.normal(size=(n_params, n_params))
    v_hat = a @ a.T * 1e-4  # matrice SPD arbitraire

    cov_analytic = longrun_covariance(params, v_hat)

    def g(vec: np.ndarray) -> np.ndarray:
        phi_ = vec[:p]
        idx = p
        thetas = []
        for qj in q:
            b = vec[idx : idx + qj + 1]
            idx += qj + 1
            thetas.append(np.sum(b) / (1.0 - np.sum(phi_)))
        return np.array(thetas)

    _, cov_numeric = _delta_method(g, theta_vec, v_hat)

    np.testing.assert_allclose(cov_analytic, cov_numeric, atol=1e-6, rtol=1e-4)


def test_degenerate_lambda_near_zero_warns_and_nans() -> None:
    """Spec 03 §3, point 4 : |lambda| < tol -> NaN + DegenerateCaseWarning."""
    params = ARDLParams(
        p=1,
        q=(1,),
        phi=np.array([1.0]),  # sum(phi) = 1 => lambda = 0
        beta=(np.array([0.3, 0.2]),),
    )
    with pytest.warns(DegenerateCaseWarning):
        theta = longrun_coefs(params)
    assert np.isnan(theta.iloc[0])


def test_half_life_invalid_outside_domain_warns_and_nans() -> None:
    """Spec 03 §3, point 3 : half_life valide seulement si -1 < lambda < 0."""
    params = ARDLParams(p=1, q=(1,), phi=np.array([1.2]), beta=(np.array([0.3, 0.2]),))
    with pytest.warns(DegenerateCaseWarning):
        hl = half_life(params)
    assert np.isnan(hl)


def test_half_life_valid_domain() -> None:
    params = ARDLParams(p=1, q=(1,), phi=np.array([0.5]), beta=(np.array([0.3, 0.2]),))
    lam = speed_of_adjustment(params)
    assert lam == pytest.approx(-0.5)
    hl = half_life(params)
    assert hl == pytest.approx(np.log(0.5) / np.log(1 + lam))


def test_round_trip_covers_q_zero() -> None:
    """Cas limite q_j = 0 (docs/QUESTIONS.md) : round-trip toujours exact."""
    params = ARDLParams(
        p=2,
        q=(0, 2),
        phi=np.array([0.1, -0.05]),
        beta=(np.array([0.7]), np.array([0.3, -0.2, 0.1])),
    )
    roundtrip = ecm_to_ardl(ardl_to_ecm(params))
    np.testing.assert_allclose(roundtrip.phi, params.phi, atol=1e-12)
    for b0, b1 in zip(roundtrip.beta, params.beta, strict=True):
        np.testing.assert_allclose(b0, b1, atol=1e-12)


def test_longrun_covariance_requires_cov_params() -> None:
    params = ARDLParams(p=1, q=(1,), phi=np.array([0.5]), beta=(np.array([0.3, 0.2]),))
    with pytest.raises(ValueError, match="cov_params"):
        longrun_covariance(params)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"p": 0}, "p doit"),
        ({"phi": np.array([0.1, 0.2])}, "phi doit"),
        ({"q": (1, 1)}, "même longueur"),
    ],
)
def test_ardlparams_validation(kwargs: dict, match: str) -> None:
    base = {
        "p": 1,
        "q": (1,),
        "phi": np.array([0.5]),
        "beta": (np.array([0.3, 0.2]),),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ARDLParams(**base)


def test_ardlparams_negative_q_rejected() -> None:
    with pytest.raises(ValueError, match=r"q\[0\] doit"):
        ARDLParams(p=1, q=(-1,), phi=np.array([0.5]), beta=(np.array([0.3, 0.2]),))


def test_ardlparams_x_names_length_mismatch() -> None:
    with pytest.raises(ValueError, match="x_names"):
        ARDLParams(
            p=1,
            q=(1,),
            phi=np.array([0.5]),
            beta=(np.array([0.3, 0.2]),),
            x_names=("a", "b"),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"p": 0}, "p doit"),
        ({"psi": np.array([1.0])}, "psi doit"),
        ({"gamma": np.array([0.5, 0.5])}, "même longueur"),
    ],
)
def test_ecmparams_validation(kwargs: dict, match: str) -> None:
    from pyardl.core.transforms import ECMParams

    base = {
        "p": 1,
        "q": (1,),
        "lam": -0.5,
        "gamma": np.array([0.5]),
        "psi": np.array([]),
        "omega": (np.array([0.3]),),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ECMParams(**base)


def test_ardlparams_k_property() -> None:
    params = ARDLParams(
        p=1,
        q=(1, 2),
        phi=np.array([0.5]),
        beta=(np.array([0.3, 0.2]), np.array([0.1, 0.2, 0.3])),
    )
    assert params.k == 2
