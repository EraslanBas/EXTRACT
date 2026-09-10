"""Global linear decoder: factors -> genes."""

from __future__ import annotations

import torch
from torch import nn


class GlobalLinearDecoder(nn.Module):
    """``x_hat = z @ B``, with ``B`` [n_factors, n_genes] shared globally.

    Row ``k`` of ``B`` is factor ``k``'s gene loading vector. Because ``B`` does
    not depend on perturbation or context, a factor means the same thing in
    every drug context -- that is the modelling assumption the design buys, and
    it is testable by holding out ``(p, c)`` combinations
    (``evaluation.heldout``) and looking for context-specific residual
    structure.
    """

    def __init__(self, n_factors: int, n_genes: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_factors, n_genes, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)

    @property
    def loadings(self) -> torch.Tensor:
        """``B`` as [n_factors, n_genes]."""
        return self.linear.weight.T

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)
