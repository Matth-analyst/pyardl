"""System ARDL: SUR-ECM estimation of several linked equations (Zellner 1962)."""

from pyardl.system.model import SystemARDL, SystemARDLResults, WaldResult

__all__ = ["SystemARDL", "SystemARDLResults", "WaldResult"]
