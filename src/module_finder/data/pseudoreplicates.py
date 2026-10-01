"""Fixed-n LFC pseudo-replicates from cell-level perturb-seq data.

Why this exists: the contrastive objective needs several observations sharing
each ``u = (perturbation, context)`` label. Posterior-mean matrices give exactly
one row per pair, which makes ``p(x|u)`` a point mass and the discrimination
task solvable by memorisation. Subsampling cells restores that spread.

Two rules, both load-bearing:

**Fix the cell count.** ``n_cells`` is identical for every pair. Per-pair counts
otherwise leak the label through noise magnitude -- 800 cells versus 60 differ
~3.7x in noise scale, and a discriminator will identify ``u`` from that alone
without learning any biology. This is the same class of shortcut as per-gene
scale heterogeneity. The cost is a power floor that defines which pairs are
usable.

**Do not re-shrink.** Compute plain mean differences against context-matched
controls. Re-running shrinkage per subsample would cost n_drugs x n_perts x
n_replicates fits, induce dependence between replicates through the pooled
prior, and reintroduce the count leak (shrinkage strength depends on n). The
low-rank model *is* the shrinkage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

try:  # anndata is optional at import time so the rest of the package stays light
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None


@dataclass
class PseudoreplicateSet:
    """LFC vectors with their labels and replicate ids."""

    X: np.ndarray  # [n_replicates_total, n_genes]
    perturbations: np.ndarray  # label per row
    contexts: np.ndarray  # label per row
    replicate: np.ndarray  # 0..n_replicates-1 within each (p, c)
    genes: np.ndarray
    n_cells: int

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]


def build_pseudoreplicates(
    adata,
    perturbation_key: str,
    context_key: str | None,
    control_label: str,
    n_cells: int = 100,
    n_control_cells: int | None = None,
    n_replicates: int = 5,
    layer: str | None = None,
    disjoint: bool = True,
    seed: int = 0,
) -> PseudoreplicateSet:
    """Build fixed-``n`` LFC pseudo-replicates per ``(perturbation, context)``.

    Parameters
    ----------
    adata
        Cell x gene AnnData. Values are assumed already log-transformed
        (``log1p``); the LFC is a difference of means on that scale.
    context_key
        Column in ``.obs`` holding the context. ``None`` treats the whole object
        as a single context.
    control_label
        Value of ``perturbation_key`` marking control cells (e.g.
        ``"non-targeting"``). Controls are matched *within* context.
    n_cells, n_control_cells
        Cells per replicate, identical for every pair. ``n_control_cells``
        defaults to ``n_cells``.
    n_replicates
        Replicates per pair. With ``disjoint=True`` a pair needs
        ``n_cells * n_replicates`` cells to qualify; pairs with fewer are
        skipped and reported in ``skipped_``.
    disjoint
        Disjoint cell splits (recommended) rather than bootstrap resampling.
        Bootstrap replicates are correlated, which inflates the apparent sample
        size and leaks across any split not taken at the ``(p, c)`` level.

    Notes
    -----
    Hold out ``(perturbation, context)`` *combinations*, never replicates --
    replicates of the same pair share cells' underlying state, so a
    replicate-level split leaks.
    """
    if ad is None:  # pragma: no cover
        raise ImportError("anndata is required for build_pseudoreplicates")
    if n_control_cells is None:
        n_control_cells = n_cells

    rng = np.random.default_rng(seed)
    obs = adata.obs
    genes = np.asarray(adata.var_names)

    if context_key is None:
        context_values = np.array(["all"] * adata.n_obs)
    else:
        context_values = obs[context_key].to_numpy().astype(str)
    pert_values = obs[perturbation_key].to_numpy().astype(str)

    rows, row_pert, row_ctx, row_rep = [], [], [], []
    skipped: list[tuple[str, str, int]] = []

    for context in np.unique(context_values):
        in_context = context_values == context
        control_cells = np.flatnonzero(in_context & (pert_values == control_label))
        if len(control_cells) < n_control_cells:
            skipped.append((control_label, context, len(control_cells)))
            continue

        for perturbation in np.unique(pert_values[in_context]):
            if perturbation == control_label:
                continue
            cells = np.flatnonzero(in_context & (pert_values == perturbation))
            needed = n_cells * n_replicates if disjoint else n_cells
            if len(cells) < needed:
                skipped.append((perturbation, context, len(cells)))
                continue

            splits = _replicate_splits(cells, n_cells, n_replicates, disjoint, rng)
            for r, cell_idx in enumerate(splits):
                ctrl_idx = rng.choice(control_cells, n_control_cells, replace=False)
                lfc = _mean_expression(adata, cell_idx, layer) - _mean_expression(
                    adata, ctrl_idx, layer
                )
                rows.append(lfc)
                row_pert.append(perturbation)
                row_ctx.append(context)
                row_rep.append(r)

    if not rows:
        raise ValueError(
            "no (perturbation, context) pair met the cell-count floor; lower "
            f"n_cells (currently {n_cells}) or n_replicates ({n_replicates})"
        )

    result = PseudoreplicateSet(
        X=np.vstack(rows),
        perturbations=np.array(row_pert),
        contexts=np.array(row_ctx),
        replicate=np.array(row_rep),
        genes=genes,
        n_cells=n_cells,
    )
    result.skipped_ = skipped  # type: ignore[attr-defined]
    return result


def _replicate_splits(
    cells: np.ndarray,
    n_cells: int,
    n_replicates: int,
    disjoint: bool,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if disjoint:
        shuffled = rng.permutation(cells)
        return [shuffled[i * n_cells : (i + 1) * n_cells] for i in range(n_replicates)]
    return [rng.choice(cells, n_cells, replace=False) for _ in range(n_replicates)]


def _mean_expression(adata, cell_idx: np.ndarray, layer: str | None) -> np.ndarray:
    X = adata.layers[layer] if layer else adata.X
    block = X[cell_idx]
    if sp.issparse(block):
        return np.asarray(block.mean(axis=0)).ravel()
    return np.asarray(block).mean(axis=0)
