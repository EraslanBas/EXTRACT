"""Real-vs-permuted-label discrimination, and the stratified negative sampler.

A negative replaces the ``(perturbation, context)`` label of a real observation.
**The observation vector itself is never altered** -- whole rows move as units,
so the marginal law of ``x`` is preserved exactly and the only way to win the
discrimination is to model the dependence between ``x`` and ``u``.

Why stratify
------------
Naive permutation is solved by precision rather than biology. The noise scale of
``x`` encodes ``n_cells``, and ``n_cells`` is tied to perturbation identity, so
a discriminator can separate real from permuted pairs by reading off effect-size
scale and never look at structure. Drawing the negative from the same subsample
stratum holds precision roughly fixed across the swap: measured on the built
matrices the median ``|log2(n_real / n_neg)|`` falls from **1.748** unstratified
to **0.617** stratified.

Do not shuffle values *inside* the expression vector. That destroys the gene
marginals, and per-gene SD spans ~5.4e4x on these matrices, so a single
threshold then separates real from shuffled: separability is d = 0.59 for
within-row shuffling against d = 0.00 for column shuffling.

See ``docs/paper/modulefinder.pdf`` sections 4.1 and 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

#: The three strategies of the spec. Names match
#: :data:`module_finder.data.augment.STRATEGIES` so a sampled negative and a
#: pre-built augmented table mean the same thing.
STRATEGIES = ("same_s_other_pert", "same_s_other_context", "against_vehicle")

DEFAULT_WEIGHTS = {
    "same_s_other_pert": 0.45,
    "same_s_other_context": 0.40,
    "against_vehicle": 0.15,
}


@dataclass
class StratifiedNegativeSampler:
    """Draw permuted labels within a subsample stratum.

    Built from the full label arrays so the pools can be precomputed once; the
    per-batch draw is then a handful of fancy-index operations.

    Parameters
    ----------
    perturbation_idx, context_idx, stratum
        [n_rows] each. ``stratum`` is the subsample index ``s`` in ``0..10``
        (0 = the full-data row), i.e. the sample's precision tier.
    vehicle_context_idx
        Context indices that count as vehicle (DMSO). Required if
        ``against_vehicle`` has nonzero weight.
    weights
        Mixture over :data:`STRATEGIES`.

    Strategies, and what each one forces the model to learn:

    ``same_s_other_pert``
        ``A1BG__sub03 -> BRCA1__sub03``. Same context, same stratum, different
        perturbation. Forces perturbation identity.
    ``same_s_other_context``
        ``A1BG__sub03`` under a different drug. Forces the drug interaction
        ``delta_pc``.
    ``against_vehicle``
        Pair with a DMSO context label. Forces drug-specific modulation.
    """

    perturbation_idx: np.ndarray
    context_idx: np.ndarray
    stratum: np.ndarray
    n_perturbations: int
    n_contexts: int
    vehicle_context_idx: tuple[int, ...] = ()
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"unknown strategies: {sorted(unknown)}")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("strategy weights must sum to a positive number")
        if self.weights.get("against_vehicle", 0) and not self.vehicle_context_idx:
            raise ValueError(
                "'against_vehicle' requires vehicle_context_idx (the indices of "
                "the DMSO contexts)"
            )
        self._names = list(self.weights)
        self._probs = np.array([self.weights[n] for n in self._names]) / total

        self.perturbation_idx = np.asarray(self.perturbation_idx, dtype=np.int64)
        self.context_idx = np.asarray(self.context_idx, dtype=np.int64)
        self.stratum = np.asarray(self.stratum, dtype=np.int64)

        # Pool 1: perturbations observed in each (context, stratum) cell.
        self._by_cs: dict[tuple[int, int], np.ndarray] = {}
        for (c, s), grp in _groupby(self.context_idx, self.stratum):
            self._by_cs[(c, s)] = np.unique(self.perturbation_idx[grp])

        # Pool 2: contexts in which each (perturbation, stratum) was observed.
        self._by_ps: dict[tuple[int, int], np.ndarray] = {}
        for (p, s), grp in _groupby(self.perturbation_idx, self.stratum):
            self._by_ps[(p, s)] = np.unique(self.context_idx[grp])

        self._vehicles = np.asarray(self.vehicle_context_idx, dtype=np.int64)
        self._n_strata = int(self.stratum.max()) + 1
        self._depth_cache = None

    # ------------------------------------------------------------------

    def sample(
        self, rows: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Negatives for the given row positions.

        Returns ``(neg_perturbation_idx, neg_context_idx)``.
        """
        rows = np.asarray(rows, dtype=np.int64)
        p = self.perturbation_idx[rows]
        c = self.context_idx[rows]
        s = self.stratum[rows]

        neg_p = p.copy()
        neg_c = c.copy()
        choice = rng.choice(len(self._names), size=len(rows), p=self._probs)

        for i, name in enumerate(self._names):
            sel = np.nonzero(choice == i)[0]
            if not len(sel):
                continue
            if name == "same_s_other_pert":
                for j in sel:
                    pool = self._by_cs.get((int(c[j]), int(s[j])))
                    neg_p[j] = _draw_excluding(pool, int(p[j]), self.n_perturbations, rng)
            elif name == "same_s_other_context":
                for j in sel:
                    pool = self._by_ps.get((int(p[j]), int(s[j])))
                    neg_c[j] = _draw_excluding(pool, int(c[j]), self.n_contexts, rng)
            elif name == "against_vehicle":
                neg_c[sel] = self._vehicles[rng.integers(0, len(self._vehicles), len(sel))]

        # A negative that coincides with its own label is a mislabelled training
        # example. This bites 'against_vehicle' hardest: for a row already in a
        # vehicle context the swap is a no-op. Repair by moving the perturbation
        # away from its own value, which is always a true negative. Before the
        # repair this affected 13/2000 draws in testing.
        collided = (neg_p == p) & (neg_c == c)
        if collided.any():
            for j in np.nonzero(collided)[0]:
                pool = self._by_cs.get((int(neg_c[j]), int(s[j])))
                neg_p[j] = _draw_excluding(pool, int(p[j]), self.n_perturbations, rng)
        return neg_p, neg_c

    def precision_mismatch(
        self,
        rows: np.ndarray,
        neg_p: np.ndarray,
        neg_c: np.ndarray,
        n_cells: np.ndarray,
        neg_stratum: np.ndarray | None = None,
    ) -> np.ndarray:
        """``|log2(n_real / n_neg)|`` per draw -- the diagnostic that justifies
        stratification.

        ``n_neg`` is the cell count of the row the *negative label* would name
        at the drawn row's own stratum, i.e. ``(neg_p, neg_c, s)``. Comparing
        instead against a perturbation's depth pooled over strata erases the
        very thing stratification fixes, and reports ~1.0 for draws that are in
        fact well matched.

        Median should sit near 0.6 on the real matrices; near 1.7 means the
        draws are effectively unstratified and the discriminator has a shortcut
        through noise scale. ``NaN`` marks a negative naming a
        ``(p, c, s)`` combination that was never observed.

        Pass ``neg_stratum`` to score a hypothetical sampler that ignores the
        stratum -- that is how the unstratified baseline is computed, and it is
        the only way to reproduce the 1.748 figure from these matrices.
        """
        rows = np.asarray(rows, dtype=np.int64)
        n_cells = np.asarray(n_cells, dtype=np.float64)
        table = self._depth_table(n_cells)

        s = self.stratum[rows] if neg_stratum is None else np.asarray(neg_stratum)
        real = n_cells[rows]
        neg = table[self._key(np.asarray(neg_p), np.asarray(neg_c), s)]

        out = np.full(len(rows), np.nan)
        ok = (real > 0) & (neg > 0)
        out[ok] = np.abs(np.log2(real[ok] / neg[ok]))
        return out

    # ---- internals for the diagnostic above ---------------------------

    def _key(self, p: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
        return (p * self.n_contexts + c) * self._n_strata + s

    def _depth_table(self, n_cells: np.ndarray) -> np.ndarray:
        """``(p, c, s) -> n_cells``, zero where unobserved. Cached."""
        if getattr(self, "_depth_cache", None) is None:
            size = self.n_perturbations * self.n_contexts * self._n_strata
            table = np.zeros(size)
            keys = self._key(self.perturbation_idx, self.context_idx, self.stratum)
            table[keys] = n_cells
            self._depth_cache = table
        return self._depth_cache


# ---------------------------------------------------------------------------


def _groupby(a: np.ndarray, b: np.ndarray):
    """Yield ``((a_val, b_val), row_positions)`` for each distinct pair."""
    keys = np.stack([a, b], axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    ks = keys[order]
    boundaries = np.nonzero(np.any(ks[1:] != ks[:-1], axis=1))[0] + 1
    for grp in np.split(order, boundaries):
        yield (int(a[grp[0]]), int(b[grp[0]])), grp


def _draw_excluding(
    pool: np.ndarray | None, own: int, n_values: int, rng: np.random.Generator
) -> int:
    """Draw from ``pool`` excluding ``own``; fall back to the full range.

    The fallback matters: a perturbation observed in only one context has no
    same-stratum alternative, and refusing to emit a negative there would
    silently drop those rows from the discrimination task.
    """
    if pool is not None and len(pool) > 1:
        candidates = pool[pool != own]
        if len(candidates):
            return int(candidates[rng.integers(0, len(candidates))])
    if n_values < 2:
        raise ValueError("cannot resample a label with fewer than 2 values")
    return int((own + rng.integers(1, n_values)) % n_values)



# ---------------------------------------------------------------------------


def contrastive_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """``BCE(s(x,u), 1) + BCE(s(x,u*), 0)`` -- eq. (9)."""
    return F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    ) + F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )


def discrimination_accuracy(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> float:
    """Diagnostic. Near-perfect accuracy early in training is a warning, not a
    success: it usually means a shortcut (noise scale, effect size) is solving
    the task. Check ``precision_mismatch`` and held-out accuracy before
    believing it."""
    correct = (real_logits > 0).sum() + (fake_logits <= 0).sum()
    return float(correct) / (len(real_logits) + len(fake_logits))
