"""Model components.

``B`` is the only factor->gene map. There is no encoder: activations are the
masked least-squares projection of ``x`` onto ``B``'s row space, derived from
``B`` rather than fitted alongside it.
"""

from .heads import BASIS_FUNCTIONS, DEFAULT_BASIS, PerComponentHead, UnconstrainedHead
from .label_net import FactorizedLabelNet
from .loadings import NO_MASK, GlobalLoadings

__all__ = [
    "GlobalLoadings",
    "NO_MASK",
    "PerComponentHead",
    "UnconstrainedHead",
    "FactorizedLabelNet",
    "BASIS_FUNCTIONS",
    "DEFAULT_BASIS",
]
