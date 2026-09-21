"""ARDL-GARCH: joint mean/variance estimation (Engle 1982, Bollerslev 1986)."""

from pyardl.volatility.model import ARDLGarch, ARDLGarchResults

__all__ = ["ARDLGarch", "ARDLGarchResults"]
