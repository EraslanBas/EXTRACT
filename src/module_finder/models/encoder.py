"""Linear encoder: genes -> factors."""

from __future__ import annotations

import torch
from torch import nn


class LinearEncoder(nn.Module):
    """``z = x @ W`` (+ optional bias).

    Linear by design, not for convenience. Requiring the factor->gene map to be
    global and consistent across all perturbations and contexts forces the
    decoder to be linear, and an invertible linear decoder implies a linear
    encoder. Any nonlinearity in the model therefore lives in the
    ``(perturbation, context) -> z`` map (``models.label_net``), which is where
    the drug x perturbation interaction belongs anyway.

    Parameters
    ----------
    n_genes
        Input dimension.
    n_factors
        Number of latent factors. Keep this near the effective rank of the data
        rather than large: on the ChemoGenetic posterior-mean matrices the
        participation-ratio effective rank is ~16, so 20-30 is the sensible
        range and 100+ is over-parameterised.
    """

    def __init__(self, n_genes: int, n_factors: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(n_genes, n_factors, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        # Fixes the per-component scale/location that the per-component head
        # leaves free; without it the optimiser drifts along that direction.
        # PCL uses a BatchNorm here for exactly this reason.
        self.norm = nn.BatchNorm1d(n_factors, affine=False)

    @property
    def weight(self) -> torch.Tensor:
        """``W`` as [n_factors, n_genes]. NOT a loading vector -- see
        ``interpret.loadings`` for why encoder weights must not be read as
        gene loadings."""
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))
