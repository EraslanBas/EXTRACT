"""Evaluation: cross-seed stability and held-out (p, c) generalisation."""

from .heldout import (
    PairSplit,
    reconstruction_score,
    split_context_holdout,
    split_pair_holdout,
)
from .stability import FactorMatch, match_factors, stability_across_seeds

__all__ = [
    "match_factors",
    "FactorMatch",
    "stability_across_seeds",
    "split_pair_holdout",
    "split_context_holdout",
    "PairSplit",
    "reconstruction_score",
]
