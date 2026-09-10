"""Real-vs-shuffled-label discrimination, and the negative samplers.

Negatives replace the ``(perturbation, context)`` label of a real observation.
The *observation vector itself is never altered*: whole rows move as units, so
the marginal distribution of ``x`` is preserved exactly and the only way to win
the discrimination is to model the dependence between ``x`` and ``u``.

Do not shuffle values *inside* the expression vector. That destroys the gene
marginals too, and on the ChemoGenetic posterior means the per-gene SD spans
~5.4e4x -- a single threshold then separates real from shuffled and the model
learns nothing. Preserving marginals is the whole point of the permutation.

Similarly, ``random`` negatives alone are largely solved by effect-size
matching (per-perturbation SD spans ~341x). Mix in the hard strategies below or
the model will plateau on main effects and never reach the interactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

STRATEGIES = ("random", "same_perturbation", "same_context", "against_control")


@dataclass
class NegativeSampler:
    """Draw shuffled labels according to a weighted mixture of strategies.

    Strategies, and what each one forces the model to learn:

    ``random``
        Both label fields resampled. Easy; learnable from effect size alone.
    ``same_perturbation``
        Keep ``p``, resample ``c``. Forces the perturbation x context
        interaction (``delta_pc``).
    ``same_context``
        Keep ``c``, resample ``p``. Forces perturbation specificity within a
        context.
    ``against_control``
        Keep ``p``, set ``c`` to the control context. Forces drug-specific
        modulation relative to vehicle. Requires ``control_context_idx``.
    """

    n_perturbations: int
    n_contexts: int
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "random": 0.25,
            "same_perturbation": 0.3,
            "same_context": 0.3,
            "against_control": 0.15,
        }
    )
    control_context_idx: int | None = None

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"unknown strategies: {sorted(unknown)}")
        if self.weights.get("against_control", 0) and self.control_context_idx is None:
            raise ValueError(
                "'against_control' requires control_context_idx (e.g. the "
                "index of DMSO_round2)"
            )
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("strategy weights must sum to a positive number")
        self._names = list(self.weights)
        self._probs = np.array([self.weights[n] for n in self._names]) / total

    def sample(
        self,
        perturbation_idx: np.ndarray,
        context_idx: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(negative_perturbation_idx, negative_context_idx)``."""
        n = len(perturbation_idx)
        neg_p = perturbation_idx.copy()
        neg_c = context_idx.copy()
        choice = rng.choice(len(self._names), size=n, p=self._probs)

        for i, name in enumerate(self._names):
            m = choice == i
            k = int(m.sum())
            if not k:
                continue
            if name == "random":
                neg_p[m] = rng.integers(0, self.n_perturbations, k)
                neg_c[m] = rng.integers(0, self.n_contexts, k)
            elif name == "same_perturbation":
                neg_c[m] = _resample_excluding(context_idx[m], self.n_contexts, rng)
            elif name == "same_context":
                neg_p[m] = _resample_excluding(
                    perturbation_idx[m], self.n_perturbations, rng
                )
            elif name == "against_control":
                neg_c[m] = self.control_context_idx

        # A negative that happens to equal its own label is a mislabelled
        # training example. This bites 'against_control' hardest -- for an
        # observation already in the control context it is a no-op, i.e. ~1/16
        # of those draws on a 16-context screen. Repair by resampling the
        # perturbation away from its own value, which is always a true negative.
        collided = (neg_p == perturbation_idx) & (neg_c == context_idx)
        if collided.any():
            neg_p[collided] = _resample_excluding(
                perturbation_idx[collided], self.n_perturbations, rng
            )
        return neg_p, neg_c


def _resample_excluding(
    current: np.ndarray, n_values: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw uniformly from ``range(n_values)`` excluding each element's own value."""
    if n_values < 2:
        raise ValueError("cannot resample a label with fewer than 2 possible values")
    offset = rng.integers(1, n_values, len(current))
    return (current + offset) % n_values


def contrastive_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy with real=1, fake=0."""
    logits = torch.cat([real_logits, fake_logits])
    targets = torch.cat(
        [torch.ones_like(real_logits), torch.zeros_like(fake_logits)]
    )
    return F.binary_cross_entropy_with_logits(logits, targets)


def discrimination_accuracy(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> float:
    """Diagnostic. Near-perfect accuracy early in training is a warning sign,
    not a success: it usually means a shortcut (noise scale, effect size) is
    solving the task. Compare against held-out ``(p, c)`` performance."""
    correct = (real_logits > 0).sum() + (fake_logits <= 0).sum()
    return float(correct) / (len(real_logits) + len(fake_logits))
