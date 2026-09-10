"""Training objectives: contrastive discrimination + reconstruction anchor."""

from .contrastive import (
    STRATEGIES,
    NegativeSampler,
    contrastive_loss,
    discrimination_accuracy,
)
from .reconstruction import weighted_mse

__all__ = [
    "NegativeSampler",
    "contrastive_loss",
    "discrimination_accuracy",
    "weighted_mse",
    "STRATEGIES",
]
