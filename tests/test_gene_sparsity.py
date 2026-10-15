"""Gene-side prior: each program concentrates its weight in few genes."""

from __future__ import annotations

import numpy as np
import torch

from extract.models.loadings import gene_concentration
from extract.train import TrainConfig, fit


def test_concentration_is_the_top_k_share():
    B = torch.tensor([[3.0, 4.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(gene_concentration(B, 1), torch.tensor([16 / 25, 0.25]))
    assert torch.allclose(gene_concentration(B, 2), torch.tensor([1.0, 0.5]))
    assert torch.allclose(gene_concentration(5 * B, 2), gene_concentration(B, 2))   # scale-free


def test_gradient_moves_weight_into_the_top_genes():
    B = torch.tensor([[2.0, 1.0, 1.0, 1.0]], requires_grad=True)
    (1 - gene_concentration(B, 1)).sum().backward()
    assert B.grad[0, 0] < 0 and (B.grad[0, 1:] > 0).all()      # descent grows the top gene, shrinks the rest


def _data(seed=0):
    rng = np.random.default_rng(seed)
    G, d = 60, 3
    B = np.zeros((d, G)); B[0, :8] = 1; B[1, 8:16] = 1; B[2, 16:24] = 1   # programs on disjoint gene sets
    p = np.repeat(np.arange(45) // 3, 4); c = np.repeat(np.arange(45) % 3, 4)
    Z = rng.standard_normal((45, d)) * (rng.random((45, d)) < 0.5)
    X = (Z[np.repeat(np.arange(45), 4)] @ B + 0.05 * rng.standard_normal((len(p), G))).astype(np.float32)
    return X, p, c


def test_penalty_raises_gene_concentration_in_fit():
    X, p, c = _data()
    shares = {}
    for g in (0.0, 5.0):
        cfg = TrainConfig(n_factors=3, epochs=40, batch_size=64, log_every=0, seed=0, head="statistics",
                          subspace="fixed", subspace_method="pca", axes_init="random", noise_rank=2,
                          gene_sparsity=g, gene_topk=8, lr=1e-2)
        model, hist = fit(X, p, c, config=cfg)
        shares[g] = float(gene_concentration(model.loadings.B.detach(), 8).mean())
        assert "gene_top_share" in hist[-1]
    assert shares[5.0] > shares[0.0] + 0.05 and shares[5.0] > 0.9
