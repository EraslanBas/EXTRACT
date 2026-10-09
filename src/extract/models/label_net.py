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
        Width of ``e_p`` and ``e_c``, unless ``context_embedding_dim`` is given.
    context_embedding_dim
        Width of ``e_c`` when it differs from ``e_p``'s.
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
        dropout: float = 0.0,
        context_embedding_dim: int | None = None,
    ):
        super().__init__()
        ctx_dim = embedding_dim if context_embedding_dim is None else context_embedding_dim
        self.n_factors = n_factors
        self.n_basis = n_basis

        self.perturbation_embedding = nn.Embedding(n_perturbations, embedding_dim)
        self.context_embedding = nn.Embedding(n_contexts, ctx_dim)
        nn.init.normal_(self.perturbation_embedding.weight, std=0.02)
        nn.init.normal_(self.context_embedding.weight, std=0.02)

        out_dim = n_factors * n_basis
        # dropout on the concatenated embeddings and the hidden layer:
        # regularises against memorising the training pairs
        if hidden:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_dim + ctx_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim + ctx_dim, out_dim))

    def forward(
        self, perturbation_idx: torch.Tensor, context_idx: torch.Tensor
    ) -> torch.Tensor:
        e_p = self.perturbation_embedding(perturbation_idx)
        e_c = self.context_embedding(context_idx)
        lam = self.head(torch.cat([e_p, e_c], dim=-1))
        return lam.view(-1, self.n_factors, self.n_basis)


class ProductLabelNet(nn.Module):
    """Structured label model: ``lambda_kj(p, c) = f_kj(p) * g_kj(c)``.

    Each factor's coefficients are a product of a perturbation term and a
    context term, with no interaction beyond that product. The label-dependent
    part of the response is then a CP (tensor) decomposition over
    perturbations x contexts x genes. A CP decomposition is unique up to the
    order and scale of its components, so the axes are fixed by the labels:
    rotating ``z`` by ``W`` would need ``W (f(p) * g(c))``, which is not of
    product form. The unstructured :class:`FactorizedLabelNet` can absorb any
    rotation and leaves the axes free.

    ``f`` and ``g`` are free tables (one ``[d, n_basis]`` entry per
    perturbation and per context). ``g`` starts near 1 and ``f`` near 0, so
    the product starts small without a vanishing gradient on either side.
    """

    def __init__(self, n_perturbations: int, n_contexts: int, n_factors: int, n_basis: int):
        super().__init__()
        self.n_factors = n_factors
        self.n_basis = n_basis
        self.f = nn.Embedding(n_perturbations, n_factors * n_basis)
        self.g = nn.Embedding(n_contexts, n_factors * n_basis)
        nn.init.normal_(self.f.weight, std=0.02)
        nn.init.normal_(self.g.weight, mean=1.0, std=0.02)

    def forward(
        self, perturbation_idx: torch.Tensor, context_idx: torch.Tensor
    ) -> torch.Tensor:
        lam = self.f(perturbation_idx) * self.g(context_idx)
        return lam.view(-1, self.n_factors, self.n_basis)


class AMMILabelNet(ProductLabelNet):
    """Main effects plus one multiplicative interaction per factor:
    ``lambda_kj(p, c) = a_kj(p) + b_kj(c) + f_kj(p) * g_kj(c)``.

    The AMMI model of genotype x environment analysis. The additive part is
    preserved by any mixing of the factors, so it carries no information about
    the axes; the interaction ``f * g`` does, exactly as in
    :class:`ProductLabelNet`. Removing each factor's perturbation and context
    averages leaves a rank-one table on the true axes and a higher-rank one on
    mixed axes. The axes are therefore fixed only as strongly as the
    interactions are.
    """

    def __init__(self, n_perturbations: int, n_contexts: int, n_factors: int, n_basis: int):
        super().__init__(n_perturbations, n_contexts, n_factors, n_basis)
        self.a = nn.Embedding(n_perturbations, n_factors * n_basis)
        self.b = nn.Embedding(n_contexts, n_factors * n_basis)
        nn.init.normal_(self.a.weight, std=0.02)
        nn.init.normal_(self.b.weight, std=0.02)

    def forward(
        self, perturbation_idx: torch.Tensor, context_idx: torch.Tensor
    ) -> torch.Tensor:
        lam = (self.a(perturbation_idx) + self.b(context_idx)
               + self.f(perturbation_idx) * self.g(context_idx))
        return lam.view(-1, self.n_factors, self.n_basis)
