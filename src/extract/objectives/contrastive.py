"""True-vs-swapped-label discrimination, and the negative sampler.

A negative replaces the ``(perturbation, context)`` label of a measured row.
**The observation vector itself is never altered** -- whole rows move as units,
so the distribution of ``x`` is the same for true and false examples and the
only way to win the discrimination is to model the dependence between ``x``
and ``u``. With label swaps as the only negatives the optimal classifier is the
density ratio ``log p(x, u) / (p(x) p(u))``, which is the form the
per-component head is built for (``models.heads``).

Every drawn label is a pair observed among the sampler's own rows, so a sampler
built from training rows never names a validation or test pair. All rows of a
pair are exchangeable: ``fit`` passes a single stratum, so no row type (full
data or pseudobulk) is matched to another and no cell counts are used.

Do not shuffle values *inside* the expression vector: that destroys the gene
marginals and asks a different question (is this a plausible response at
all?), whose optimal classifier is not of the head's form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

#: The two swap strategies, half of the negatives each by default. No strategy
#: assumes a reference context (vehicle, untreated): contexts are symmetric.
STRATEGIES = ("same_s_other_pert", "same_s_other_context")

DEFAULT_WEIGHTS = {
    "same_s_other_pert": 0.5,
    "same_s_other_context": 0.5,
}


@dataclass
class StratifiedNegativeSampler:
    """Draw swapped labels, each naming a pair observed among the given rows.

    Built from the label arrays of the rows it serves, so the pools can be
    precomputed once; the per-batch draw is then a handful of fancy-index
    operations.

    Parameters
    ----------
    perturbation_idx, context_idx, stratum
        [n_rows] each. Draws prefer a label from the same ``stratum``; ``fit``
        passes all zeros (the rows of a pair are exchangeable), so the
        stratum has no effect there.
    weights
        Mixture over :data:`STRATEGIES`.

    Strategies, and what each one forces the model to learn:

    ``same_s_other_pert``
        Same context, different perturbation (observed in that context).
        Forces perturbation identity.
    ``same_s_other_context``
        Same perturbation, different context (in which it was observed).
        Forces the interaction. Unavailable with a single context; then the
        other axis is swapped.
    """

    perturbation_idx: np.ndarray
    context_idx: np.ndarray
    stratum: np.ndarray
    n_perturbations: int
    n_contexts: int
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(STRATEGIES)
        if unknown:
            raise ValueError(f"unknown strategies: {sorted(unknown)}")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("strategy weights must sum to a positive number")
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

        # Stratum-free fallbacks. Used only when a (cell, stratum) pool cannot
        # supply an alternative; drawing from these keeps the emitted negative
        # an *observed* (perturbation, context) pair, which is the invariant
        # that makes held-out evaluation meaningful.
        self._perts_by_c: dict[int, np.ndarray] = {}
        for ctx in np.unique(self.context_idx):
            self._perts_by_c[int(ctx)] = np.unique(
                self.perturbation_idx[self.context_idx == ctx]
            )
        self._ctxs_by_p: dict[int, np.ndarray] = {}
        for prt in np.unique(self.perturbation_idx):
            self._ctxs_by_p[int(prt)] = np.unique(
                self.context_idx[self.perturbation_idx == prt]
            )

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

        # Every emitted negative must name a (p, c) pair observed among the
        # sampler's own rows -- the invariant that keeps held-out pairs from
        # ever being named. When the chosen axis has no observed alternative
        # (a perturbation with a single training context cannot swap its
        # context), swap the other axis instead. Drawing from the full label
        # range, as an earlier version did, named held-out and never-measured
        # pairs: ~50 per epoch on the 16-context screen.
        for i, name in enumerate(self._names):
            for j in np.nonzero(choice == i)[0]:
                pj, cj, sj = int(p[j]), int(c[j]), int(s[j])
                swaps = (
                    (self._swap_pert, self._swap_context)
                    if name == "same_s_other_pert"
                    else (self._swap_context, self._swap_pert)
                )
                for swap in swaps:
                    drawn = swap(pj, cj, sj, rng)
                    if drawn is not None:
                        neg_p[j], neg_c[j] = drawn
                        break
                else:
                    raise ValueError(
                        f"label (perturbation {pj}, context {cj}) has no observed "
                        "alternative on either axis; it cannot be given a "
                        "negative without naming an unobserved pair"
                    )
        return neg_p, neg_c

    def _swap_pert(self, p: int, c: int, s: int, rng) -> tuple[int, int] | None:
        """``(p', c)`` with ``p'`` observed in ``c``, same stratum preferred."""
        drawn = _draw_excluding(
            self._by_cs.get((c, s)), p, rng, fallback_pool=self._perts_by_c.get(c)
        )
        return None if drawn is None else (drawn, c)

    def _swap_context(self, p: int, c: int, s: int, rng) -> tuple[int, int] | None:
        """``(p, c')`` with ``p`` observed in ``c'``, same stratum preferred."""
        drawn = _draw_excluding(
            self._by_ps.get((p, s)), c, rng, fallback_pool=self._ctxs_by_p.get(p)
        )
        return None if drawn is None else (p, drawn)

    def precision_mismatch(
        self,
        rows: np.ndarray,
        neg_p: np.ndarray,
        neg_c: np.ndarray,
        n_cells: np.ndarray,
        neg_stratum: np.ndarray | None = None,
    ) -> np.ndarray:
        """``|log2(n_real / n_neg)|`` per draw: how different the cell counts of
        the true and the swapped label are (a diagnostic; not used in training).

        ``n_neg`` is the cell count of the row the *negative label* would name
        at the drawn row's own stratum, i.e. ``(neg_p, neg_c, s)``. ``NaN``
        marks a negative naming a ``(p, c, s)`` combination that was never
        observed. Pass ``neg_stratum`` to score a hypothetical sampler that
        ignores the stratum.
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
    pool: np.ndarray | None,
    own: int,
    rng: np.random.Generator,
    fallback_pool: np.ndarray | None = None,
) -> int | None:
    """Draw from ``pool`` excluding ``own``, widening to ``fallback_pool``.

    ``fallback_pool`` is the same quantity without the stratum constraint, so
    the widened draw is still an *observed* label. Returns ``None`` when
    neither offers an alternative; the caller then swaps the other axis.
    """
    for candidates in (pool, fallback_pool):
        if candidates is not None and len(candidates) > 1:
            c = candidates[candidates != own]
            if len(c):
                return int(c[rng.integers(0, len(c))])
    return None



# ---------------------------------------------------------------------------


def contrastive_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """``BCE(s(x,u), 1) + BCE(s(x,u*), 0)``, each a mean over its rows."""
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
