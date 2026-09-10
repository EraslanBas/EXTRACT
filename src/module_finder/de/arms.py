"""Build the augmented input that the ashr pipeline consumes.

No differential expression happens here. ``ComputeSE.py`` takes one ``.h5ad``
per context and reads its grouping from a single ``adata.obs`` column, so to get
**full and subsampled estimates into the same matrix** the input has to carry
one label per intended output row:

===========================  ==========================================
label                        rows it produces
===========================  ==========================================
``<pert>``                   the full-data estimate for that perturbation
``<pert>__sub<k>``           subsample ``k`` of that perturbation
``non-targeting``            the shared control baseline
===========================  ==========================================

A cell can only carry one label, so the subsampled cells are *duplicated* under
their ``__sub<k>`` label. That is what lets the whole thing run in one
unmodified ``ComputeSE.py`` pass and land in one ashr fit -- which is the right
place for it, since ashr conditions on ``se`` and so handles the full and
subsampled rows' different precisions in a single prior.

Subsampling rule: a perturbation is subsampled only if it has more than
``min_cells_to_subsample`` (50) cells in that context, and then ``n`` is drawn
uniformly at random from ``[50, n_cells]`` and the cells themselves are drawn
uniformly without replacement. Perturbations at or below the floor contribute
their full row only.

Control rule: a **fixed** set of ``n_control_cells`` (default 10,000) control
cells is drawn once per context and shared by every row -- the full estimates
and the subsampled ones alike. ``ComputeSE.py`` computes the control mean and
variance once before its per-label loop, so one control group necessarily
serves all labels; fixing its size makes the ``var_c / n_ctrl`` term of the
standard error identical across rows and comparable across contexts. Control
cells beyond that number are dropped from the augmented object.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

#: Separator between a perturbation name and its subsample tag.
SUBSAMPLE_SEP = "__sub"

#: Perturbations with more cells than this are eligible for subsampling.
MIN_CELLS_TO_SUBSAMPLE = 50

#: Control cells used per context, fixed and shared by every row.
N_CONTROL_CELLS = 10_000


def plan_augmentation(
    perturbations: np.ndarray,
    control_label: str,
    n_subsamples: int = 1,
    min_cells_to_subsample: int = MIN_CELLS_TO_SUBSAMPLE,
    n_control_cells: int = N_CONTROL_CELLS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Plan the augmented cell list for one context.

    Parameters
    ----------
    n_control_cells
        Size of the fixed control set. Drawn once, uniformly without
        replacement, and shared by every row. If the context has fewer control
        cells than this, all of them are used and a warning is issued -- the
        control-side precision is then lower than requested and not comparable
        with contexts that met the target.

    Returns
    -------
    cell_index
        Indices into the original object: the fixed control set, every
        perturbed cell (for the full rows), and the subsampled cells repeated.
        Control cells outside the fixed set do not appear.
    labels
        The ``adata.obs`` label for each entry of ``cell_index``.
    plan
        One row per output row of the eventual matrix: ``label, perturbation,
        variant, subsample, n_cells_planned``. ``n_cells_planned`` is the ``n``
        drawn for a subsample, the full count for a full row, and the fixed
        control-set size for the control row.
    """
    perturbations = np.asarray(perturbations).astype(str)
    rng = np.random.default_rng(seed)

    cell_index: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rows = []

    # ---- the fixed control set, drawn once and shared by every row ----
    control_idx = np.flatnonzero(perturbations == control_label)
    if len(control_idx) == 0:
        raise ValueError(f"no cells labelled {control_label!r}")
    if len(control_idx) <= n_control_cells:
        if len(control_idx) < n_control_cells:
            warnings.warn(
                f"only {len(control_idx)} control cells available, fewer than the "
                f"requested {n_control_cells}; using all of them. Control-side "
                "precision is lower than requested and not comparable with "
                "contexts that met the target.",
                stacklevel=2,
            )
        chosen_control = control_idx
    else:
        chosen_control = rng.choice(control_idx, size=n_control_cells, replace=False)
    cell_index.append(chosen_control)
    labels.append(np.full(len(chosen_control), control_label))
    rows.append(
        {
            "label": control_label,
            "perturbation": control_label,
            "variant": "control",
            "subsample": -1,
            "n_cells_planned": len(chosen_control),
        }
    )

    for pert in np.unique(perturbations):
        if pert == control_label:
            continue
        idx = np.flatnonzero(perturbations == pert)

        # full row: every cell this perturbation has in this context
        cell_index.append(idx)
        labels.append(np.full(len(idx), pert))
        rows.append(
            {
                "label": pert,
                "perturbation": pert,
                "variant": "full",
                "subsample": -1,
                "n_cells_planned": len(idx),
            }
        )
        if len(idx) <= min_cells_to_subsample:
            continue

        for k in range(n_subsamples):
            # inclusive of both ends: n in [50, n_cells]
            n = int(rng.integers(min_cells_to_subsample, len(idx) + 1))
            chosen = rng.choice(idx, size=n, replace=False)
            label = f"{pert}{SUBSAMPLE_SEP}{k:02d}"
            cell_index.append(chosen)
            labels.append(np.full(n, label))
            rows.append(
                {
                    "label": label,
                    "perturbation": pert,
                    "variant": "subsample",
                    "subsample": k,
                    "n_cells_planned": n,
                }
            )

    return (
        np.concatenate(cell_index),
        np.concatenate(labels),
        pd.DataFrame(rows),
    )


def build_augmented_adata(
    adata,
    perturbation_key: str,
    control_label: str,
    label_key: str = "mf_label",
    n_subsamples: int = 1,
    min_cells_to_subsample: int = MIN_CELLS_TO_SUBSAMPLE,
    n_control_cells: int = N_CONTROL_CELLS,
    seed: int = 0,
    permute: bool = False,
    permute_seed: int = 0,
):
    """Materialise the augmented AnnData plus its plan.

    Parameters
    ----------
    permute
        Shuffle perturbation labels among the perturbed cells *before*
        planning, giving the label-permuted null. Control cells keep their
        label so the baseline stays a real baseline, and every label keeps its
        cell count.

    Returns
    -------
    (adata_augmented, plan)
        ``adata_augmented.obs[label_key]`` is what to pass to ``ComputeSE.py``
        as ``--group-key``. The control set is fixed at ``n_control_cells``,
        drawn once and shared by every row; controls are never duplicated and
        any beyond that number are dropped.
    """
    perturbations = adata.obs[perturbation_key].astype(str).to_numpy()

    if permute:
        rng = np.random.default_rng(permute_seed)
        movable = np.flatnonzero(perturbations != control_label)
        perturbations = perturbations.copy()
        perturbations[movable] = perturbations[rng.permutation(movable)]

    cell_index, labels, plan = plan_augmentation(
        perturbations,
        control_label=control_label,
        n_subsamples=n_subsamples,
        min_cells_to_subsample=min_cells_to_subsample,
        n_control_cells=n_control_cells,
        seed=seed,
    )

    augmented = adata[cell_index].to_memory() if adata.isbacked else adata[cell_index].copy()
    augmented.obs[label_key] = pd.Categorical(labels)
    # duplicated cells make obs_names non-unique, which AnnData warns about
    augmented.obs_names = [f"cell{i:09d}" for i in range(augmented.n_obs)]
    return augmented, plan
