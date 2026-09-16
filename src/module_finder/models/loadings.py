"""The one factor->gene map, and the projection derived from it.

    B      [d, G]   the only interpretable output, the only learned map
    B^+    [G, d]   B^T (B B^T)^{-1}  -- computed, never fitted
    z      [d]      argmin_z ||m * (x - z B)||^2

There is deliberately no encoder. With a separate ``W`` there are two maps
between factors and genes, they are free to disagree about what factor ``k``
means, and only one of them is a loading vector: encoder weights are *filters*,
not *patterns*, and a filter can place large weight on a gene precisely to
cancel it as a nuisance (Haufe et al. 2014). Tying removes the question --
``B`` is a pattern by construction.

``x`` is therefore never on a generative path. It is projected so the
discriminator can judge it, and it is the regression target.

See ``docs/paper/modulefinder.pdf`` sections 3.1-3.2 for the derivation; the
equation numbers in the comments below refer to it.
"""

from __future__ import annotations

import torch
from torch import nn

#: Sentinel in a ``target_col`` array: this row masks nothing (the perturbation
#: targets no measured gene, e.g. a non-targeting control).
NO_MASK = -1


class GlobalLoadings(nn.Module):
    """``B`` [n_factors, n_genes], plus the projection and reconstruction.

    Parameters
    ----------
    ridge
        Added to the diagonal of ``B B^T`` before inversion. Small but not
        optional: early in training rows of ``B`` are near-collinear and the
        Gram matrix is ill-conditioned.
    """

    def __init__(self, n_factors: int, n_genes: int, ridge: float = 1e-4):
        super().__init__()
        self.n_factors = n_factors
        self.n_genes = n_genes
        self.ridge = ridge
        self.B = nn.Parameter(torch.empty(n_factors, n_genes))
        nn.init.xavier_uniform_(self.B)

    # ---- the Gram matrix and its inverse (one d x d factorisation per step) --

    def gram_inverse(self) -> torch.Tensor:
        """``(B B^T + ridge I)^{-1}`` -- [d, d]. Cheap: ``d`` is 20-30."""
        G = self.B @ self.B.T
        G = G + self.ridge * torch.eye(
            self.n_factors, device=G.device, dtype=G.dtype
        )
        return torch.linalg.inv(G)

    @property
    def pinv(self) -> torch.Tensor:
        """``B^+ = B^T (B B^T)^{-1}`` -- [n_genes, n_factors]."""
        return self.B.T @ self.gram_inverse()

    # ---- projection -------------------------------------------------------

    def project(
        self,
        x: torch.Tensor,
        target_col: torch.Tensor | None = None,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Masked least-squares coordinates of ``x`` in ``B``'s row space.

        Unmasked this is eq. (2), ``z = x B^+``. With one masked entry per row
        it is eq. (4), a rank-one downdate of the shared Gram matrix, applied
        via Sherman-Morrison (eq. 5) so that one ``d x d`` inverse serves every
        row and each row costs ``O(d^2)`` however many distinct genes the batch
        masks.

        Parameters
        ----------
        x
            [batch, n_genes].
        target_col
            [batch] int64 column to exclude per row, :data:`NO_MASK` for none.
            ``None`` masks nothing at all.
        tol
            Rows whose Sherman-Morrison denominator ``1 - b^T G^-1 b`` falls
            below this are re-solved explicitly. The identity is exact but
            loses precision as the denominator approaches zero, which happens
            when a masked gene's loading column nearly spans a factor
            direction on its own.
        """
        Ginv = self.gram_inverse()                      # [d, d], symmetric
        if target_col is None:
            return (x @ self.B.T) @ Ginv                # eq. (2)

        masked = target_col != NO_MASK
        x_tilde = x
        if masked.any():
            x_tilde = x.clone()
            rows = torch.nonzero(masked, as_tuple=True)[0]
            x_tilde[rows, target_col[rows]] = 0.0

        Bx = x_tilde @ self.B.T                         # [n, d]
        z = Bx @ Ginv
        if not masked.any():
            return z

        # Sherman-Morrison correction on the masked rows only.
        b = self.B[:, target_col.clamp(min=0)].T        # [n, d]; junk where unmasked
        Ginv_b = b @ Ginv                               # [n, d]
        denom = 1.0 - (Ginv_b * b).sum(-1)              # [n]
        num = (Ginv_b * Bx).sum(-1)                     # [n]

        stable = masked & (denom.abs() > tol)
        correction = torch.zeros_like(z)
        if stable.any():
            correction[stable] = (
                Ginv_b[stable] * (num[stable] / denom[stable]).unsqueeze(-1)
            )
        z = z + correction                              # eq. (4) via eq. (5)

        # Explicit solve for the few rows where the identity is ill-conditioned.
        unstable = masked & ~stable
        if unstable.any():
            idx = torch.nonzero(unstable, as_tuple=True)[0]
            bb = b[idx]                                 # [k, d]
            G = torch.linalg.inv(Ginv).unsqueeze(0) - bb.unsqueeze(2) * bb.unsqueeze(1)
            z = z.clone()
            z[idx] = torch.linalg.solve(G, Bx[idx].unsqueeze(-1)).squeeze(-1)
        return z

    # ---- reconstruction ---------------------------------------------------

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """``x_hat = z B`` -- [batch, n_genes]. No bias: ``x`` is already a
        difference against within-context controls, so zero is the reference."""
        return z @ self.B

    def squared_error(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        target_col: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-row ``|| m * (x - z B) ||^2 / n_retained`` -- [batch].

        Divided by the number of retained genes so the scale does not depend on
        ``G`` or on whether a row masks anything.
        """
        resid = x - self.reconstruct(z)
        sq = resid.pow(2).sum(dim=1)
        n_kept = torch.full_like(sq, float(self.n_genes))
        if target_col is not None:
            masked = target_col != NO_MASK
            if masked.any():
                rows = torch.nonzero(masked, as_tuple=True)[0]
                dropped = resid[rows, target_col[rows]].pow(2)
                sq = sq.index_add(0, rows, -dropped)
                n_kept = n_kept.index_add(0, rows, -torch.ones_like(dropped))
        return sq / n_kept

    # ---- diagnostics ------------------------------------------------------

    @torch.no_grad()
    def span_residual(self, x: torch.Tensor) -> torch.Tensor:
        """``||x - x B^+ B|| / ||x||`` per row -- eq. (14).

        The one measurement that can invalidate the architecture rather than
        call for a tuning pass: a held-out drug whose rows carry a large
        orthogonal residual introduced a program the training contexts never
        showed, and no amount of nonlinearity elsewhere can absorb it.
        """
        z = self.project(x)
        resid = torch.linalg.norm(x - self.reconstruct(z), dim=1)
        return resid / torch.linalg.norm(x, dim=1).clamp_min(1e-12)
