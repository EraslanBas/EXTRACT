"""Cross-seed factor stability.

Unmixedness is not directly testable -- you never observe the true sources, and
you cannot test for it by testing independence, since Darmois' construction
yields exactly-independent components that are still nonlinear mixtures of the
truth. What you can test is **reproducibility**: identified axes should recur
across seeds, arbitrary ones should not.

Factors are recovered only up to permutation and sign, so any comparison must
match them first (Hungarian assignment on |correlation|) before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FactorMatch:
    permutation: np.ndarray  # column j of B2 matched to factor j of B1
    signs: np.ndarray
    correlations: np.ndarray  # matched |correlation| per factor
    mcc: float  # mean |correlation| = MCC


def match_factors(A: np.ndarray, B: np.ndarray) -> FactorMatch:
    """Match rows of ``B`` to rows of ``A`` by |correlation|.

    Both are [n_factors, n_features] -- loading matrices, or factor activations
    transposed. Returns the permutation, the recovered signs, and the mean
    absolute correlation (MCC), the standard permutation-invariant score used
    in the nonlinear ICA literature.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"feature dimension differs: {A.shape[1]} vs {B.shape[1]}"
        )
    corr = _cross_correlation(A, B)
    rows, cols = linear_sum_assignment(-np.abs(corr))
    matched = corr[rows, cols]
    return FactorMatch(
        permutation=cols,
        signs=np.where(matched < 0, -1.0, 1.0),
        correlations=np.abs(matched),
        mcc=float(np.abs(matched).mean()),
    )


def _cross_correlation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Ac = A - A.mean(axis=1, keepdims=True)
    Bc = B - B.mean(axis=1, keepdims=True)
    Ac /= np.linalg.norm(Ac, axis=1, keepdims=True) + 1e-12
    Bc /= np.linalg.norm(Bc, axis=1, keepdims=True) + 1e-12
    return Ac @ Bc.T


def stability_across_seeds(
    loadings: list[np.ndarray], reference: int = 0
) -> dict[str, np.ndarray | float]:
    """Match every run to one reference run and summarise per-factor recurrence.

    Returns ``per_factor`` (mean matched |corr| for each reference factor across
    the other runs) and ``mcc`` (the overall mean). Report only factors whose
    ``per_factor`` score is high; the rest are run-specific artefacts.
    """
    if len(loadings) < 2:
        raise ValueError("need at least two runs to assess stability")
    ref = loadings[reference]
    scores = [
        match_factors(ref, other).correlations
        for i, other in enumerate(loadings)
        if i != reference
    ]
    stacked = np.vstack(scores)
    return {
        "per_factor": stacked.mean(axis=0),
        "per_factor_min": stacked.min(axis=0),
        "mcc": float(stacked.mean()),
        "n_runs": len(loadings),
    }
