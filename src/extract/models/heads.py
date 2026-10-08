"""The per-component contrastive head -- the identifiability constraint.

    score(x, u)  =  sum_k sum_j  lambda_{k,j}(u) * q_j(z_k)

The sum runs over components and **nothing multiplies z_k by z_j for k != j**.
That restriction is the whole mechanism, and it is worth stating why:

The optimal discriminator for the real-vs-shuffled-label task is the log density
ratio ``log p(x|u) / p(x)``. Under the model (components independent given u,
and marginally) that ratio equals

    sum_i [ log p_i(s_i|u) - log p_i(s_i) ]

i.e. the *true* answer is already additively separable in the true source
coordinates. Additive separability is not preserved under mixing, so a head
restricted to separable functions can only reach the optimum if its coordinates
are the source coordinates. With an unrestricted head (an MLP over the whole
feature vector) the substitution ``z -> M z`` can be absorbed by the head's
first layer, the loss cannot see ``M``, and the axes float.

One caveat on the basis: a purely quadratic head identifies nothing, because
``sum_k z_k^2`` is rotation invariant -- the same reason Gaussian sources are
unidentifiable in linear ICA. Keep at least one non-quadratic statistic
(``abs`` or ``tanh``) in the basis.

This is TCL's exponential family (Hyvarinen & Morioka 2016, eq. 8 of the 2023
Patterns review) with ``q_j`` as the sufficient statistics and ``lambda(u)`` as
the label-dependent natural parameters.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

#: Scalar statistics applied coordinate-wise to ``z``.
#: ``linear`` + ``square`` alone detect per-component mean and variance shifts
#: (a Gaussian family, and rotation-invariant in the square term); ``abs`` and
#: ``tanh`` add sensitivity to non-Gaussian shape, which is extra identifying
#: information.
BASIS_FUNCTIONS = {
    "linear": lambda z: z,
    "square": lambda z: z**2,
    "abs": torch.abs,
    "tanh": torch.tanh,
}

DEFAULT_BASIS = ("linear", "square", "abs", "tanh")


class PerComponentHead(nn.Module):
    """Score real vs shuffled ``(x, u)`` pairs, separably in the components.

    Parameters
    ----------
    basis
        Names from :data:`BASIS_FUNCTIONS`. Must include at least one
        non-quadratic statistic or the head is rotation invariant.
    """

    def __init__(self, basis: Sequence[str] = DEFAULT_BASIS):
        super().__init__()
        unknown = set(basis) - set(BASIS_FUNCTIONS)
        if unknown:
            raise ValueError(f"unknown basis functions: {sorted(unknown)}")
        if set(basis) <= {"square"}:
            raise ValueError(
                "a purely quadratic basis is rotation invariant and identifies "
                "nothing; include 'abs' or 'tanh'"
            )
        self.basis = tuple(basis)
        self.bias = nn.Parameter(torch.zeros(1))

    @property
    def n_basis(self) -> int:
        return len(self.basis)

    def statistics(self, z: torch.Tensor) -> torch.Tensor:
        """[batch, n_factors] -> [batch, n_factors, n_basis]."""
        return torch.stack([BASIS_FUNCTIONS[name](z) for name in self.basis], dim=-1)

    def forward(self, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Logits for the real/fake decision.

        Parameters
        ----------
        z
            [batch, n_factors] encoder output.
        lam
            [batch, n_factors, n_basis] label-derived coefficients from
            :class:`~extract.models.label_net.FactorizedLabelNet`.

        Returns
        -------
        [batch] logits. Each component contributes a scalar of *evidence*; the
        single real/fake decision is taken on the total. No component votes
        alone, and no pair of components is ever multiplied together.
        """
        if lam.shape != (*z.shape, self.n_basis):
            raise ValueError(
                f"lam has shape {tuple(lam.shape)}, expected "
                f"{(*z.shape, self.n_basis)}"
            )
        return (lam * self.statistics(z)).sum(dim=(1, 2)) + self.bias


class UnconstrainedHead(nn.Module):
    """An MLP over ``[z ; lambda]`` -- for ablation only.

    Present so the per-component constraint can be ablated and shown to matter.
    This head is a fine *classifier* and gives no identifiability whatsoever:
    it can form ``z_k + z_j`` in its first layer, so mixed and unmixed features
    score identically and nothing pushes the encoder to unmix. Do not use it to
    produce factors that will be interpreted.
    """

    def __init__(self, n_factors: int, n_basis: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_factors * (1 + n_basis), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([z, lam.flatten(start_dim=1)], dim=1)
        return self.net(flat).squeeze(-1)


class DistanceHead(nn.Module):
    """Score a row by how far its activities are from the label's prediction.

        logit = b - sum_k (z_k - zhat_k(u))^2 / (2 sigma_k^2)

    ``lam`` here is the predicted activity ``zhat`` [batch, n_factors, 1], in
    the same units as ``z``: the label model says how active each program
    should be, and a large ``z_k`` on a program predicted to be untouched
    counts against the label. This makes the discriminator consistent with
    reconstruction (``x ~ z B``, ``z ~ zhat(u)``, so ``x ~ zhat(u) B``). Up to
    a constant it is the Gaussian log-likelihood of ``z`` given the label,
    with one learned noise level ``sigma_k`` per factor. Separable in the
    components, like :class:`PerComponentHead`. It uses nothing but ``z`` and
    the label -- in particular not the row's cell count, which a model applied
    to new data cannot rely on.
    """

    n_basis = 1

    def __init__(self, n_factors: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_factors))
        self.bias = nn.Parameter(torch.zeros(1))

    def evidence(self, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """``-e_k`` per factor, [batch, n_factors]."""
        if lam.shape != (*z.shape, 1):
            raise ValueError(f"lam has shape {tuple(lam.shape)}, expected {(*z.shape, 1)}")
        return -(z - lam[..., 0]) ** 2 / (2 * torch.exp(2 * self.log_sigma))

    def forward(self, z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return self.evidence(z, lam).sum(dim=1) + self.bias
