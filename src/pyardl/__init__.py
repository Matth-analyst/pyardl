"""pyardl: ARDL models and bounds tests for cointegration.

The headline entry points are re-exported here, so that

    from pyardl import ARDL, bounds_test

works alongside the fully-qualified ``from pyardl.core import ARDL``.
The submodule paths remain the canonical ones and every documentation
page uses them; this is a convenience layer, not a second API.

**The re-exports are lazy, deliberately.** Importing the ARDL machinery
pulls in statsmodels, which costs about 3.5 seconds; binding these names
eagerly would charge that to every ``import pyardl``, including one that
only wants ``__version__``. PEP 562's module ``__getattr__`` defers the
cost to the first attribute access, so ``import pyardl`` stays at about
a millisecond and ``pyardl.ARDL`` pays for what it uses.

One consequence worth knowing: ``from pyardl import *`` and tab
completion work (``__dir__`` is defined), but a typo raises
``AttributeError`` at first use rather than at import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.5.0"

# name -> module it lives in. Kept explicit rather than derived, so that
# what is public here is a decision and not a side effect of a scan.
_EXPORTS: dict[str, str] = {
    # core
    "ARDL": "pyardl.core",
    "ARDLResults": "pyardl.core",
    "ardl_to_ecm": "pyardl.core",
    "ecm_to_ardl": "pyardl.core",
    "longrun_restriction": "pyardl.core",
    # cointegration testing
    "bounds_test": "pyardl.bounds",
    "bootstrap_bounds_test": "pyardl.bootstrap",
    "engle_granger": "pyardl.cointegration",
    "johansen": "pyardl.cointegration",
    "cointegration_analysis": "pyardl.unified",
    # efficient long-run estimators
    "dols": "pyardl.cointegration",
    "fmols": "pyardl.cointegration",
    "ccr": "pyardl.cointegration",
    "compare_longrun": "pyardl.cointegration",
    # beyond the linear ARDL
    "NARDL": "pyardl.nardl",
    "QARDL": "pyardl.qardl",
    "fourier_bounds_test": "pyardl.fourier",
    # panels
    "MeanGroup": "pyardl.panel",
    "PMG": "pyardl.panel",
    "DFE": "pyardl.panel",
    "CSARDL": "pyardl.panel",
    "CSDL": "pyardl.panel",
    "panel_from_frame": "pyardl.panel",
    # roots of the genealogy
    "KoyckModel": "pyardl.distributed_lags",
    "AlmonModel": "pyardl.distributed_lags",
    # interpretation, pre-tests, data
    "dynardl_simulate": "pyardl.simulate",
    "integration_order": "pyardl.unitroot",
    "load_denmark": "pyardl.datasets",
    "load_pss2001": "pyardl.datasets",
}

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from pyardl.bootstrap import bootstrap_bounds_test as bootstrap_bounds_test
    from pyardl.bounds import bounds_test as bounds_test
    from pyardl.cointegration import (
        ccr as ccr,
    )
    from pyardl.cointegration import (
        compare_longrun as compare_longrun,
    )
    from pyardl.cointegration import (
        dols as dols,
    )
    from pyardl.cointegration import (
        engle_granger as engle_granger,
    )
    from pyardl.cointegration import (
        fmols as fmols,
    )
    from pyardl.cointegration import (
        johansen as johansen,
    )
    from pyardl.core import (
        ARDL as ARDL,
    )
    from pyardl.core import (
        ARDLResults as ARDLResults,
    )
    from pyardl.core import (
        ardl_to_ecm as ardl_to_ecm,
    )
    from pyardl.core import (
        ecm_to_ardl as ecm_to_ardl,
    )
    from pyardl.core import (
        longrun_restriction as longrun_restriction,
    )
    from pyardl.datasets import load_denmark as load_denmark
    from pyardl.datasets import load_pss2001 as load_pss2001
    from pyardl.distributed_lags import AlmonModel as AlmonModel
    from pyardl.distributed_lags import KoyckModel as KoyckModel
    from pyardl.fourier import fourier_bounds_test as fourier_bounds_test
    from pyardl.nardl import NARDL as NARDL
    from pyardl.panel import CSARDL as CSARDL
    from pyardl.panel import CSDL as CSDL
    from pyardl.panel import DFE as DFE
    from pyardl.panel import PMG as PMG
    from pyardl.panel import MeanGroup as MeanGroup
    from pyardl.panel import panel_from_frame as panel_from_frame
    from pyardl.qardl import QARDL as QARDL
    from pyardl.simulate import dynardl_simulate as dynardl_simulate
    from pyardl.unified import cointegration_analysis as cointegration_analysis
    from pyardl.unitroot import integration_order as integration_order

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Resolve a re-exported name on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'pyardl' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # bind it, so the lookup happens once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
