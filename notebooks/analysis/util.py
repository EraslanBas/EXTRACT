"""Compatibility shim -- keeps the pre-refactor flat import working.

    from util import *          # still works, resolves here

The real implementation now lives in ``src/perturbseq_sim/utils.py``.
New code should import it directly:

    from perturbseq_sim.utils import *

This file exists only so the simulator notebooks run unedited. Delete it once
the notebooks have been converted to package imports.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from perturbseq_sim.utils import *  # noqa: F401,F403,E402
