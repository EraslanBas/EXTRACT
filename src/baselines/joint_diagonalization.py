"""Orthogonal joint diagonalization of a set of symmetric matrices.

Jacobi-angle algorithm of Cardoso & Souloumiac (1996), the same core used by
JADE. Its use here is the nonstationarity route to identifiability: with a
linear mixing, a set of per-context covariance matrices that share one
diagonalizing basis pins down the mixing without needing non-Gaussianity
(Matsuoka et al. 1995; Pham & Cardoso 2001 -- refs 67/68 of the 2023 Patterns
review, cited there as the linear predecessors of TCL).

A 16-context screen supplies 16 matrices, which is a well-conditioned problem.
"""

from __future__ import annotations

import numpy as np


def joint_diagonalize(
    matrices: np.ndarray, tol: float = 1e-8, max_sweeps: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Find orthogonal ``V`` making every ``matrices[k]`` as diagonal as possible.

    Parameters
    ----------
    matrices
        [n_matrices, d, d], each symmetric.

    Returns
    -------
    V
        [d, d] orthogonal, such that ``V.T @ matrices[k] @ V`` is approximately
        diagonal for every ``k``.
    diagonals
        [n_matrices, d] the resulting diagonals.
    """
    M = np.array(matrices, dtype=np.float64, copy=True)
    if M.ndim != 3 or M.shape[1] != M.shape[2]:
        raise ValueError(f"expected [k, d, d], got {M.shape}")
    n_mat, d, _ = M.shape
    V = np.eye(d)

    for _ in range(max_sweeps):
        max_rotation = 0.0
        for p in range(d - 1):
            for q in range(p + 1, d):
                h1 = M[:, p, p] - M[:, q, q]
                h2 = M[:, p, q] + M[:, q, p]
                ton = float(h1 @ h1 - h2 @ h2)
                toff = float(2.0 * (h1 @ h2))
                theta = 0.5 * np.arctan2(
                    toff, ton + np.sqrt(ton * ton + toff * toff)
                )
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) <= tol:
                    continue
                max_rotation = max(max_rotation, abs(s))

                # Givens rotation applied on both sides of every matrix.
                cols_p, cols_q = M[:, :, p].copy(), M[:, :, q].copy()
                M[:, :, p] = c * cols_p + s * cols_q
                M[:, :, q] = -s * cols_p + c * cols_q
                rows_p, rows_q = M[:, p, :].copy(), M[:, q, :].copy()
                M[:, p, :] = c * rows_p + s * rows_q
                M[:, q, :] = -s * rows_p + c * rows_q

                v_p, v_q = V[:, p].copy(), V[:, q].copy()
                V[:, p] = c * v_p + s * v_q
                V[:, q] = -s * v_p + c * v_q

        if max_rotation <= tol:
            break

    diagonals = np.stack([np.diag(M[k]) for k in range(n_mat)])
    return V, diagonals


def off_diagonal_energy(matrices: np.ndarray, V: np.ndarray) -> float:
    """Fraction of total energy left off-diagonal after applying ``V``.

    0 means perfectly jointly diagonalizable; use it to check whether the
    per-context covariances actually share a basis before trusting the result.
    """
    M = np.asarray(matrices, dtype=np.float64)
    total = off = 0.0
    for k in range(M.shape[0]):
        T = V.T @ M[k] @ V
        total += float((T**2).sum())
        off += float((T**2).sum() - (np.diag(T) ** 2).sum())
    return off / total if total > 0 else 0.0
