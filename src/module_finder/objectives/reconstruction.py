"""Reconstruction term -- anchors factors to directions that explain variance.

This term is load-bearing, not decoration. A contrastive objective has no
preference for high-variance directions: a factor explaining 0.01% of variance
that perfectly separates two contexts will be favoured, and its loading vector
is then uninterpretable as a gene program. The reconstruction term is what
keeps factors large enough to mean something.

Tune the weight so it dominates *direction selection* without collapsing the
model onto plain PCA -- if the factors stop responding to the label, the
contrastive term is too weak; if held-out reconstruction is poor while
discrimination is perfect, it is too strong.
"""

from __future__ import annotations

import torch


def weighted_mse(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error, optionally weighted per observation.

    ``row_weights`` should carry the precision of each LFC estimate. Unweighted,
    a poorly powered perturbation's noisy vector counts as much as a
    well-powered one; per-perturbation SD spans ~341x on these matrices, part
    real effect size and part estimation noise.
    """
    err = (x - x_hat).pow(2).mean(dim=1)
    if row_weights is None:
        return err.mean()
    w = row_weights / row_weights.sum()
    return (err * w).sum() * len(err)
