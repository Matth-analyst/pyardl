"""Fixtures partagées (seed explicite, jamais l'état global numpy)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260707)
