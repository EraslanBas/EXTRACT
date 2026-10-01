"""Training objectives: contrastive discrimination + reconstruction anchor.

``L_disc`` picks the rotation; ``L_recon`` picks the subspace. The optional
total-correlation term is in :mod:`~extract.objectives.total_correlation`
and is off by default.
"""

from .contrastive import (
    DEFAULT_WEIGHTS,
    STRATEGIES,
    StratifiedNegativeSampler,
    contrastive_loss,
    discrimination_accuracy,
)
from .reconstruction import precision_weights, weighted_squared_error

__all__ = [
    "StratifiedNegativeSampler",
    "contrastive_loss",
    "discrimination_accuracy",
    "precision_weights",
    "weighted_squared_error",
    "STRATEGIES",
    "DEFAULT_WEIGHTS",
]
