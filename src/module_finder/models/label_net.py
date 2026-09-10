"""(perturbation, context) -> per-component coefficients lambda.

Deliberately expressive. The identifiability constraint in
``models.heads.PerComponentHead`` restricts only *how* lambda meets the
features -- never how lambda itself is computed. All of the perturbation x
context interaction lives here and is unconstrained.

Factorised rather than one free embedding row per observed ``(p, c)`` tuple.
That distinction decides whether the model learns or memorises: a free lookup
table has one row per pair, so with a single observation per pair the optimal
discriminator is ``answer real iff x == x_u`` -- a hash table, reachable
without any notion of components. Sharing ``e_p`` across contexts and ``e_c``
across perturbations drops the parameter count from ``n_pairs * d`` to
``(n_perturbations + n_contexts) * d`` and makes memorisation impossible.
"""

from __future__ import annotations

import torch
from torch import nn


class FactorizedLabelNet(nn.Module):
    """Map ``(perturbation_idx, context_idx)`` to ``lambda`` [d, n_basis].

    Parameters
    ----------
    embedding_dim
        Width of ``e_p`` and ``e_c``.
    hidden
        Width of the interaction MLP. Set ``hidden=0`` for a purely bilinear
        (additive-in-embeddings) map, which is the low-rank interaction model.
    """

    def __init__(
        self,
        n_perturbations: int,
        n_contexts: int,
        n_factors: int,
        n_basis: int,
        embedding_dim: int = 32,
        hidden: int = 128,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.n_basis = n_basis

        self.perturbation_embedding = nn.Embedding(n_perturbations, embedding_dim)
        self.context_embedding = nn.Embedding(n_contexts, embedding_dim)
        nn.init.normal_(self.perturbation_embedding.weight, std=0.02)
        nn.init.normal_(self.context_embedding.weight, std=0.02)

        out_dim = n_factors * n_basis
        if hidden:
            self.head = nn.Sequential(
                nn.Linear(2 * embedding_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.head = nn.Linear(2 * embedding_dim, out_dim)

    def forward(
        self, perturbation_idx: torch.Tensor, context_idx: torch.Tensor
    ) -> torch.Tensor:
        e_p = self.perturbation_embedding(perturbation_idx)
        e_c = self.context_embedding(context_idx)
        lam = self.head(torch.cat([e_p, e_c], dim=-1))
        return lam.view(-1, self.n_factors, self.n_basis)
