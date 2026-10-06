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

See the method paper sections 3.1-3.2 for the derivation; the
equation numbers in the comments below refer to it.
"""

from __future__ import annotations

import numpy as np
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
            # Rebuild the Gram matrix directly rather than inverting Ginv: this
            # is the path chosen precisely because the system is ill-conditioned,
            # so inverting an inverse is the last thing to do here.
            G_full = self.B @ self.B.T + self.ridge * torch.eye(
                self.n_factors, device=self.B.device, dtype=self.B.dtype
            )
            G = G_full.unsqueeze(0) - bb.unsqueeze(2) * bb.unsqueeze(1)
            z = z.clone()
            z[idx] = torch.linalg.solve(G, Bx[idx].unsqueeze(-1)).squeeze(-1)
        return z

    # ---- reconstruction ---------------------------------------------------

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """``x_hat = z B`` -- [batch, n_genes]. No bias: ``x`` is already a
        difference against within-context controls, so zero is the reference."""
        return z @ self.B

    # ---- optional noise metric ------------------------------------------

    def set_noise_metric(self, U: np.ndarray, s: np.ndarray, c: float) -> None:
        """Measure reconstruction error in ``M~ = I + U diag(s/c) U^T``.

        The noise metric ``M = c I + U diag(s) U^T`` rescaled so directions with
        no correlated noise keep weight 1 (so ``alpha`` keeps its meaning), and
        each shared-noise direction ``u_k`` is down-weighted by ``c / (c + s_k)``.
        No gene's own scale changes.
        """
        self.register_buffer("metric_U", torch.as_tensor(U, dtype=torch.float32))
        self.register_buffer("metric_w", torch.as_tensor(
            np.asarray(s) / (float(c) + np.asarray(s)), dtype=torch.float32))

    def squared_error(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        target_col: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-row ``|| m * (x - z B) ||^2 / n_retained`` -- [batch].

        Divided by the number of retained genes so the scale does not depend on
        ``G`` or on whether a row masks anything. With a noise metric set
        (:meth:`set_noise_metric`) the squared norm is ``r M~^{-1} r^T`` of the
        masked residual instead.
        """
        resid = x - self.reconstruct(z)
        if getattr(self, "metric_U", None) is not None:
            r = resid
            if target_col is not None:
                masked = target_col != NO_MASK
                if masked.any():
                    rows = torch.nonzero(masked, as_tuple=True)[0]
                    r = resid.clone()
                    r[rows, target_col[rows]] = 0.0
            proj = r @ self.metric_U
            sq = r.pow(2).sum(1) - (proj.pow(2) * self.metric_w).sum(1)
            n_kept = torch.full_like(sq, float(self.n_genes))
            if target_col is not None:
                n_kept = n_kept - (target_col != NO_MASK).to(sq.dtype)
            return sq / n_kept
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
    def span_residual(
        self, x: torch.Tensor, target_col: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``||m * (x - x B^+ B)|| / ||m * x||`` per row -- eq. (14).

        The one measurement that can invalidate the architecture rather than
        call for a tuning pass: a held-out drug whose rows carry a large
        orthogonal residual introduced a program the training contexts never
        showed, and no amount of nonlinearity elsewhere can absorb it.

        Pass ``target_col`` so the on-target entry is excluded from both the
        projection and the norms, exactly as in training. ``B`` is fitted never
        to reconstruct that entry, and it carries a median ~25% of a row's
        squared norm, so leaving it in charges ``B`` for mass it was told to
        ignore and inflates ``r``. Omitting ``target_col`` reproduces the
        earlier unmasked behaviour.
        """
        z = self.project(x, target_col)
        resid = x - self.reconstruct(z)
        if target_col is not None:
            masked = target_col != NO_MASK
            if masked.any():
                rows = torch.nonzero(masked, as_tuple=True)[0]
                resid = resid.clone()
                x = x.clone()
                resid[rows, target_col[rows]] = 0.0
                x[rows, target_col[rows]] = 0.0
        num = torch.linalg.norm(resid, dim=1)
        return num / torch.linalg.norm(x, dim=1).clamp_min(1e-12)


class FixedBasisLoadings(GlobalLoadings):
    """``B = A V`` with ``V`` [d, G] fixed and only ``A`` [d, d] learned.

    The span of ``B`` is the span of ``V`` and cannot move, so reconstruction is
    fixed by ``V`` and the discriminator chooses only the axes inside it --
    oblique ones, since ``A`` is any invertible matrix: the structure of ICA
    (whiten, then unmix) with the separable head as the unmixing criterion.
    Every method of :class:`GlobalLoadings` works unchanged through ``B``.
    """

    def __init__(self, V: np.ndarray, ridge: float = 1e-4):
        nn.Module.__init__(self)
        V = torch.as_tensor(np.asarray(V), dtype=torch.float32)
        self.n_factors, self.n_genes = V.shape
        self.ridge = ridge
        self.register_buffer("V", V)
        self.A = nn.Parameter(torch.eye(self.n_factors))

    @property
    def B(self) -> torch.Tensor:  # type: ignore[override]
        return self.A @ self.V
