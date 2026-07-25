"""Finite-T response surfaces — EXPERIMENTAL, NOT VALIDATED.

Validation of this module against published reference output is pending
permission from the authors of the underlying material. Until that
permission is received:

- no comparison against published values is encoded here;
- ``download_surface_coefs()`` must not be executed, and no test may be
  added that depends on it;
- only tests that need no external file are run.
"""

from __future__ import annotations

import pytest

from pyardl.critical_values.ks2020_finite import (
    _coefs_path,
    crit_value_bounds_finite,
    pvalue_bounds_finite,
)

# Any test that would evaluate the response-surface coefficients needs
# the locally cached file, which is no longer downloaded (pending
# permission). The tests below are therefore either purely internal or
# skipped.
_HAS_CACHE = _coefs_path().exists()


class TestInternalCoherence:
    """No external dependency: the surface is compared with itself
    (asymptotic limits, monotonicity)."""

    @pytest.mark.skipif(not _HAS_CACHE, reason="pending permission (see module header)")
    def test_asymptotic_limit_matches_a1(self) -> None:
        """Very large T should reproduce the asymptotic critical values
        within 0.05."""
        from pyardl.critical_values.ks2020 import crit_value_bounds

        for case in (1, 3, 5):
            for k in (1, 3):
                fin = crit_value_bounds_finite(
                    case=case, k=k, t_obs=10_000_000, sr=0, alpha=0.05
                )
                a1 = crit_value_bounds(case=case, k=k, alpha=0.05)
                assert fin[0] == pytest.approx(a1[0], abs=0.05)
                assert fin[1] == pytest.approx(a1[1], abs=0.05)

    @pytest.mark.skipif(not _HAS_CACHE, reason="pending permission (see module header)")
    def test_t_asymptotic_limit_matches_pss(self) -> None:
        got = crit_value_bounds_finite(
            case=3, k=1, t_obs=10_000_000, sr=0, alpha=0.05, stat="t"
        )
        assert got[0] == pytest.approx(-2.86, abs=0.02)
        assert got[1] == pytest.approx(-3.22, abs=0.02)

    @pytest.mark.skipif(not _HAS_CACHE, reason="pending permission (see module header)")
    def test_cv_decrease_toward_asymptotic_in_t_obs(self) -> None:
        """Bounds are more conservative in small samples and decrease
        towards their asymptotic values."""
        values = [
            crit_value_bounds_finite(case=3, k=2, t_obs=t, sr=3, alpha=0.05)[1]
            for t in (30, 50, 80, 200, 1000)
        ]
        assert all(a >= b - 1e-9 for a, b in zip(values[:-1], values[1:], strict=True))

    @pytest.mark.skipif(not _HAS_CACHE, reason="pending permission (see module header)")
    def test_sr_increases_cv_in_small_samples(self) -> None:
        """More short-run coefficients raise the bounds at small T, as
        degrees of freedom are consumed."""
        low = crit_value_bounds_finite(case=3, k=2, t_obs=40, sr=0, alpha=0.05)[1]
        high = crit_value_bounds_finite(case=3, k=2, t_obs=40, sr=10, alpha=0.05)[1]
        assert high > low

    @pytest.mark.skipif(not _HAS_CACHE, reason="pending permission (see module header)")
    def test_pvalue_roundtrip_at_cv(self) -> None:
        for stat in ("F", "t"):
            cv = crit_value_bounds_finite(3, 2, 90, 4, 0.05, stat=stat)  # type: ignore[arg-type]
            p = pvalue_bounds_finite(cv[0], 3, 2, 90, 4, df_resid=80, stat=stat)  # type: ignore[arg-type]
            assert p[0] == pytest.approx(0.05, abs=2e-3)


class TestCoverageAndErrors:
    """Input validation: no dependency on the external file."""

    def test_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="case"):
            crit_value_bounds_finite(0, 1, 100, 2, 0.05)
        with pytest.raises(ValueError, match="sr"):
            crit_value_bounds_finite(3, 1, 100, -1, 0.05)
        with pytest.raises(ValueError, match="stat"):
            crit_value_bounds_finite(3, 1, 100, 2, 0.05, stat="W")  # type: ignore[arg-type]


def test_missing_cache_raises_with_instructions(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Without the cache, the error must be explicit and state what to
    do. This test neither depends on the real cache nor downloads
    anything: it only checks the error message."""
    import pyardl.critical_values.ks2020_finite as mod

    monkeypatch.setenv("PYARDL_CACHE", str(tmp_path))
    monkeypatch.setattr(mod, "_TABLES", None)
    with pytest.raises(FileNotFoundError, match="download_surface_coefs"):
        mod._load_tables()


@pytest.mark.needs_review
@pytest.mark.external
class TestBoundsTestIntegrationPendingPermission:
    """End-to-end reproduction against published reference output.

    No reference value is encoded until permission is received from the
    authors of the material. This is a placeholder documenting the
    intent, not an active validation.
    """

    @pytest.mark.skip(
        reason=(
            "pending permission: requires (a) authorisation to call "
            "download_surface_coefs(), (b) a legitimate source of "
            "reference values. Do not re-enable before the reply."
        )
    )
    def test_full_reproduction_placeholder(self) -> None:
        raise NotImplementedError("Pending permission.")
