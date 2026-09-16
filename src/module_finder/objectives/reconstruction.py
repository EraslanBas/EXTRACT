"""Precision-weighted, on-target-masked reconstruction -- eq. (10).

    L_recon = sum_i w_i * || m_i * (x_i - z_i B) ||^2 / n_kept,   w_i ~ n_cells

This term is load-bearing in a specific way. A tied linear autoencoder trained
on reconstruction alone recovers the top-d **principal subspace** and nothing
more -- the span is determined, the orientation inside it is not (Baldi &
Hornik, 1989). So ``L_recon`` picks *which subspace* the factors live in and
``L_disc`` picks *their orientation inside it*. Drop the discriminator and this
is PCA with extra steps; drop this term and the contrastive objective will
happily choose a direction explaining 0.01% of the variance because it separates
two contexts cleanly, whose loading vector is then not a gene program.

The precision weight is the payoff of building eleven rows per perturbation.
Unweighted, a 50-cell row counts as much as a 3,275-cell one, while the spread
of those estimates falls from 0.042 to 0.0049 across that range.
"""

from __future__ import annotations

import numpy as np
import torch


def precision_weights(
    n_cells: np.ndarray, scheme: str = "linear", clip_quantile: float = 0.99
) -> np.ndarray:
    """Row weights from cell counts, normalised to mean 1.

    Parameters
    ----------
    scheme
        ``"linear"`` gives ``w ~ n_cells``, which is what the spec states and
        what the variance of a mean implies. ``"sqrt"`` is the gentler
        alternative for when a handful of very deep perturbations dominate the
        gradient; ``"uniform"`` disables weighting, for ablation.
    clip_quantile
        Cap before normalising, so one outlier perturbation cannot own the
        batch. Set to 1.0 to disable.
    """
    n = np.asarray(n_cells, dtype=np.float64)
    if np.any(n <= 0):
        raise ValueError("n_cells must be positive")
    if scheme == "linear":
        w = n
    elif scheme == "sqrt":
        w = np.sqrt(n)
    elif scheme == "uniform":
        w = np.ones_like(n)
    else:
        raise ValueError(f"unknown scheme: {scheme!r}")
    if clip_quantile < 1.0:
        w = np.minimum(w, np.quantile(w, clip_quantile))
    return w / w.mean()


def weighted_squared_error(
    per_row_error: torch.Tensor, row_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Weighted mean of per-row errors.

    ``per_row_error`` comes from
    :meth:`module_finder.models.loadings.GlobalLoadings.squared_error`, which
    has already applied the on-target mask and divided by the retained gene
    count. Weights are renormalised within the batch so ``alpha`` means the
    same thing at any batch size.
    """
    if row_weights is None:
        return per_row_error.mean()
    w = row_weights / row_weights.sum().clamp_min(1e-12)
    return (per_row_error * w).sum()
