"""Optional total-correlation term on ``z`` -- eq. (11). Off by default.

Shuffling each coordinate of ``z`` independently across the batch turns the
joint into the product of its marginals, so a discriminator trained to tell the
two apart estimates the total correlation; penalising it discourages dependence
*between* factors. This is FactorVAE's term (Kim & Mnih, 2018).

**Independence is not identifiability.** Darmois' construction yields exactly
independent components that are still nonlinear mixtures of the truth -- the
central negative result of the nonlinear ICA literature. This term encourages a
property the true sources happen to have; it is *not* a substitute for the
per-component head, which is the only thing in this model that breaks the
rotation symmetry. Turn it on if factors come out correlated. Do not expect it
to do the head's job.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TotalCorrelationDiscriminator(nn.Module):
    """MLP that separates ``z`` from its coordinate-wise shuffle."""

    def __init__(self, n_factors: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_factors
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.LeakyReLU(0.2)]
            width = hidden
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def shuffle_coordinates(z: torch.Tensor, generator=None) -> torch.Tensor:
    """Permute each coordinate of ``z`` independently across the batch."""
    out = torch.empty_like(z)
    for k in range(z.shape[1]):
        perm = torch.randperm(z.shape[0], device=z.device, generator=generator)
        out[:, k] = z[perm, k]
    return out


def tc_penalty(discriminator: TotalCorrelationDiscriminator, z: torch.Tensor):
    """Penalty added to the generator objective: ``logit_joint - logit_marginal``."""
    logits = discriminator(z)
    return (logits[:, 0] - logits[:, 1]).mean()


def tc_discriminator_loss(
    discriminator: TotalCorrelationDiscriminator,
    z: torch.Tensor,
    z_shuffled: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for the TC discriminator's own update step."""
    d_joint = discriminator(z.detach())
    d_marginal = discriminator(z_shuffled.detach())
    zeros = torch.zeros(len(z), dtype=torch.long, device=z.device)
    ones = torch.ones(len(z_shuffled), dtype=torch.long, device=z.device)
    return 0.5 * (F.cross_entropy(d_joint, zeros) + F.cross_entropy(d_marginal, ones))
