"""Estimation noise from the subsample rows, and the label-driven subspace.

A subsample row and its pair's full-data row estimate the same response; the
subsample uses a random subset of the same cells, against the same fixed
control set. Their difference is therefore pure estimation noise, with

    Cov(x_sub - x_full) = (1/n_sub - 1/n_full) * Sigma_cell

(the control term is shared and cancels). Scaling each difference by
``1 / sqrt(1/n_sub - 1/n_full)`` gives draws with covariance ``Sigma_cell``,
the per-cell noise covariance across genes, and a row with ``n`` cells has
noise covariance ``Sigma_cell / n``. ``ashr`` shrinks rows of different
precision differently, so this is an approximation.

``Sigma_cell`` is G x G; it is kept as diagonal + rank-r. The **metric** used
downstream is isotropic + the same rank-r part,

    M = c I + U diag(s) U^T,     c = mean of the per-gene noise variances,

so it discounts noise directions *shared across genes* but leaves every gene's
own scale alone: no per-gene reweighting, in keeping with using the shrunken
logFC exactly as ``ashr`` produced it.

The label-driven subspace solves ``Sigma_S v = lambda M v``: the directions with
the most perturbation-driven variation per unit of correlated noise.
``Sigma_S`` is the second moment of the full-data rows minus their expected
noise. The eigenvectors ``v`` are filters; the loading subspace is spanned by
the patterns ``a = M v``, returned as the rows of ``V`` in logFC units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NoiseModel:
    """``Sigma_cell ~ diag(diag) + U diag(s) U^T``, and the metric built on it."""
    diag: np.ndarray        # [G] per-gene noise variance (per cell)
    U: np.ndarray           # [G, r] orthonormal correlated-noise directions
    s: np.ndarray           # [r] their variances (per cell)

    @property
    def c(self) -> float:
        """Isotropic level of the metric: the mean per-gene noise variance."""
        return float(self.diag.mean())

    def metric_inv_quadratic(self, R: np.ndarray) -> np.ndarray:
        """Per-row ``r M^{-1} r^T`` for rows of ``R``, by Woodbury."""
        c = self.c
        proj = R @ self.U
        return ((R * R).sum(1) - (proj * proj * (self.s / (c + self.s))).sum(1)) / c

    def metric_inv_sqrt(self, X: np.ndarray) -> np.ndarray:
        """``X M^{-1/2}``."""
        c = self.c
        k = 1.0 - np.sqrt(c / (c + self.s))
        return (X - (X @ self.U) * k @ self.U.T) / np.sqrt(c)

    def metric_sqrt(self, X: np.ndarray) -> np.ndarray:
        """``X M^{1/2}``."""
        c = self.c
        k = np.sqrt((c + self.s) / c) - 1.0
        return (X + (X @ self.U) * k @ self.U.T) * np.sqrt(c)

    def cell_cov_apply(self, X: np.ndarray) -> np.ndarray:
        """``X Sigma_cell``."""
        return X * self.diag + (X @ self.U) * self.s @ self.U.T


def subsample_residuals(
    X: np.ndarray, meta: pd.DataFrame, rows: np.ndarray, max_rows: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Scaled (subsample - full-data) differences for the pairs of ``rows``.

    ``rows`` [n] boolean or index over ``X``: only measured rows are used, and
    only pairs that have their full-data row among them. Each difference is
    divided by ``sqrt(1/n_sub - 1/n_full)``, so every returned row has
    covariance ``Sigma_cell``.
    """
    real = (meta.is_real.to_numpy(dtype=bool) if "is_real" in meta
            else np.ones(len(meta), bool))
    pair = (meta.perturbation.astype(str) + "|" + meta.context.astype(str)).to_numpy()
    return residuals_from_arrays(X, pair, (meta.variant == "main").to_numpy(),
                                 meta.n_cells.to_numpy(), rows,
                                 max_rows=max_rows, seed=seed, is_real=real)


def residuals_from_arrays(
    X: np.ndarray, pair: np.ndarray, is_full: np.ndarray, n_cells: np.ndarray,
    rows: np.ndarray, max_rows: int | None = None, seed: int = 0,
    is_real: np.ndarray | None = None,
) -> np.ndarray:
    """:func:`subsample_residuals` on plain arrays: ``pair`` [n] any hashable
    pair key, ``is_full`` [n] marks full-data rows, ``rows`` boolean or index."""
    rows = np.nonzero(rows)[0] if np.asarray(rows).dtype == bool else np.asarray(rows)
    if is_real is not None:
        rows = rows[np.asarray(is_real, dtype=bool)[rows]]
    pr, full = np.asarray(pair)[rows], np.asarray(is_full, dtype=bool)[rows]
    nc = np.asarray(n_cells, dtype=float)[rows]
    full_of = dict(zip(pr[full], rows[full]))
    n_full = dict(zip(pr[full], nc[full]))
    sub = np.nonzero(~full & np.isin(pr, list(full_of)))[0]
    if max_rows is not None and len(sub) > max_rows:
        sub = np.sort(np.random.default_rng(seed).choice(sub, max_rows, replace=False))
    if not len(sub):
        raise ValueError("no subsample rows with their full-data row among the given rows; "
                         "the noise model needs subsampled re-estimates")
    i_sub = rows[sub]
    i_full = np.array([full_of[p] for p in pr[sub]])
    nf = np.array([n_full[p] for p in pr[sub]], dtype=float)
    scale = 1.0 / np.sqrt(np.clip(1.0 / nc[sub] - 1.0 / nf, 1e-12, None))
    return ((X[i_sub] - X[i_full]) * scale[:, None]).astype(np.float32)


def noise_covariance(R: np.ndarray, rank: int = 50, seed: int = 0) -> NoiseModel:
    """Diagonal + rank-``rank`` fit of the covariance of the residuals ``R``.

    The residuals have mean zero by construction, so second moments are used.
    The low-rank part is the top principal directions of ``R``; the diagonal is
    what those leave of each gene's variance (floored at a small fraction of the
    mean, so no gene's noise is zero).
    """
    import torch

    Rt = torch.from_numpy(np.asarray(R, dtype=np.float32))
    n = Rt.shape[0]
    total = (Rt * Rt).sum(0).double().numpy() / n
    torch.manual_seed(seed)
    _, S, V = torch.svd_lowrank(Rt, q=rank + 10, niter=4)
    U = V[:, :rank].double().numpy()
    lam = (S[:rank].double().numpy() ** 2) / n     # variance along each u_k
    # Each leading direction's variance also contains the per-gene noise along
    # it, u_k^T diag u_k; separate the two by a few factor-analysis-style
    # alternations (otherwise s is biased up by about the mean gene variance).
    floor = 1e-3 * total.mean()
    diag = np.maximum(total - (U * U * lam).sum(1), floor)
    for _ in range(5):
        s = np.maximum(lam - (U * U * diag[:, None]).sum(0), 0.0)
        diag = np.maximum(total - (U * U * s).sum(1), floor)
    return NoiseModel(diag=diag, U=U, s=s)


def label_subspace(
    X_full: np.ndarray, n_full: np.ndarray, noise: NoiseModel, d: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(V, eigenvalues)``: the top-``d`` label-driven loading directions.

    ``X_full`` [pairs, G] are full-data rows of the training pairs and
    ``n_full`` their cell counts. ``Sigma_S = E[x^T x] - Sigma_cell E[1/n]``;
    the generalised eigenproblem ``Sigma_S v = lambda M v`` is solved in the
    ``M``-whitened coordinates, and the patterns ``a = M v`` are returned as the
    rows of ``V`` [d, G], each scaled to unit norm.
    """
    Xw = noise.metric_inv_sqrt(np.asarray(X_full, dtype=np.float64))
    S_w = Xw.T @ Xw / len(Xw)
    # expected noise in whitened coordinates: M^-1/2 Sigma_cell M^-1/2 * E[1/n]
    I = np.eye(Xw.shape[1])
    Nw = noise.metric_inv_sqrt(noise.cell_cov_apply(noise.metric_inv_sqrt(I)))
    S_w -= Nw * float(np.mean(1.0 / np.asarray(n_full, dtype=float)))
    S_w = 0.5 * (S_w + S_w.T)
    evals, evecs = np.linalg.eigh(S_w)
    order = np.argsort(evals)[::-1][:d]
    W = evecs[:, order].T                      # [d, G], whitened eigenvectors
    V = noise.metric_sqrt(W)                   # patterns a = M v = M^{1/2} w
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V.astype(np.float32), evals[order]


def label_subspace_from_rows(
    X: np.ndarray, perturbation_idx: np.ndarray, context_idx: np.ndarray,
    stratum: np.ndarray, n_cells: np.ndarray, rows: np.ndarray, d: int,
    rank: int = 50, max_rows: int | None = 150_000, seed: int = 0,
    is_real: np.ndarray | None = None,
) -> tuple[np.ndarray, NoiseModel]:
    """Noise model and label-driven subspace ``V`` [d, G] from the measured
    rows among ``rows`` (the training rows): subsample residuals for the noise,
    full-data rows (``stratum == 0``) for the signal. What :func:`extract.train.fit`
    uses when ``subspace="fixed"`` and no basis is given."""
    pair = np.asarray(perturbation_idx, dtype=np.int64) * (int(np.max(context_idx)) + 1) \
        + np.asarray(context_idx, dtype=np.int64)
    is_full = np.asarray(stratum) == 0
    R = residuals_from_arrays(X, pair, is_full, n_cells, rows, max_rows, seed, is_real)
    noise = noise_covariance(R, rank=min(rank, R.shape[0] - 1, R.shape[1] - 1), seed=seed)
    del R
    sel = np.zeros(len(X), dtype=bool)
    r = np.nonzero(rows)[0] if np.asarray(rows).dtype == bool else np.asarray(rows)
    sel[r] = True
    if is_real is not None:
        sel &= np.asarray(is_real, dtype=bool)
    sel &= is_full
    V, _ = label_subspace(X[sel], np.asarray(n_cells)[sel], noise, d)
    return V, noise
