"""Negative sampling must never emit a false negative."""
import numpy as np
import pytest

from module_finder.objectives import NegativeSampler


@pytest.mark.parametrize("seed", range(5))
def test_no_false_negatives(seed):
    rng = np.random.default_rng(seed)
    s = NegativeSampler(n_perturbations=50, n_contexts=16, control_context_idx=0)
    p = rng.integers(0, 50, 5000)
    c = rng.integers(0, 16, 5000)
    neg_p, neg_c = s.sample(p, c, rng)
    assert not ((neg_p == p) & (neg_c == c)).any()


def test_against_control_requires_control_index():
    with pytest.raises(ValueError, match="control_context_idx"):
        NegativeSampler(
            n_perturbations=10, n_contexts=4, weights={"against_control": 1.0}
        )


def test_same_perturbation_keeps_perturbation():
    rng = np.random.default_rng(0)
    s = NegativeSampler(
        n_perturbations=20, n_contexts=8, weights={"same_perturbation": 1.0}
    )
    p = rng.integers(0, 20, 500)
    c = rng.integers(0, 8, 500)
    neg_p, neg_c = s.sample(p, c, rng)
    assert (neg_p == p).all()
    assert (neg_c != c).all()


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="unknown strategies"):
        NegativeSampler(n_perturbations=5, n_contexts=2, weights={"nonsense": 1.0})
