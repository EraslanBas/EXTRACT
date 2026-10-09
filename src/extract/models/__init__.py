"""Model components.

``B`` is the only factor->gene map. There is no encoder: activations are the
masked least-squares projection of ``x`` onto ``B``'s row space, derived from
``B`` rather than fitted alongside it.
"""

from .heads import BASIS_FUNCTIONS, DEFAULT_BASIS, DistanceHead, PerComponentHead, UnconstrainedHead
from .label_net import AMMILabelNet, FactorizedLabelNet, ProductLabelNet
from .loadings import initial_axes, NO_MASK, FixedBasisLoadings, GlobalLoadings

__all__ = [
    "GlobalLoadings",
    "FixedBasisLoadings",
    "NO_MASK",
    "PerComponentHead",
    "DistanceHead",
    "UnconstrainedHead",
    "FactorizedLabelNet",
    "ProductLabelNet",
    "AMMILabelNet",
    "BASIS_FUNCTIONS",
    "DEFAULT_BASIS",
]
