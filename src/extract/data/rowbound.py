"""Load the (perturbation x context) x gene log-fold-change matrix.

The ChemoGenetic screen already ships in exactly the shape this model wants:

    ChemoGeneticScreens/TextFiles/PosteriorMeanMatrices_rowbound_anyDrugSig_geneMed40.csv
        30,704 x 14,639  =  (1,919 perturbations x 16 contexts) x 14,639 genes

so rows *are* ``(p, c)`` pairs. Per-drug matrices in
``ChemoGeneticScreens/PosteriorMeanMatrices/PosteriorMean_matrix_<drug>.csv``
can be stacked to the same layout with :func:`load_per_drug_matrices`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class RowboundDataset:
    """A ``(p, c) x gene`` LFC matrix with its label indices."""

    X: np.ndarray  # [n_pairs, n_genes]
    perturbation_idx: np.ndarray  # [n_pairs] into `perturbations`
    context_idx: np.ndarray  # [n_pairs] into `contexts`
    perturbations: np.ndarray
    contexts: np.ndarray
    genes: np.ndarray
    row_weights: np.ndarray | None = None

    @property
    def n_pairs(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"RowboundDataset({self.n_pairs} pairs = "
            f"{len(self.perturbations)} perturbations x {len(self.contexts)} contexts, "
            f"{self.n_genes} genes)"
        )


def load_rowbound(
    path: str | Path,
    context_column: str = "drug",
    perturbation_column: str = "perturbation",
    usecols: Sequence[str] | None = None,
) -> RowboundDataset:
    """Read a rowbound CSV whose rows are ``(perturbation, context)`` pairs.

    The shrunken logFC values are used **as they are**. No per-gene centring or
    scaling is applied anywhere in this package -- ``ashr`` has already put every
    entry on a common, interpretable log-fold-change scale, and rescaling by
    per-gene SD would divide that out and inflate low-variance genes.
    """
    df = pd.read_csv(path, usecols=usecols)
    for col in (perturbation_column, context_column):
        if col not in df.columns:
            raise KeyError(
                f"column {col!r} not in {Path(path).name}; found "
                f"{list(df.columns[:6])}..."
            )
    labels = df[[perturbation_column, context_column]]
    genes = np.array([c for c in df.columns if c not in labels.columns])
    X = df[genes].to_numpy(dtype=np.float64)

    perturbations, perturbation_idx = np.unique(
        labels[perturbation_column].to_numpy(), return_inverse=True
    )
    contexts, context_idx = np.unique(
        labels[context_column].to_numpy(), return_inverse=True
    )

    return RowboundDataset(
        X=X,
        perturbation_idx=perturbation_idx,
        context_idx=context_idx,
        perturbations=perturbations,
        contexts=contexts,
        genes=genes,
    )


def load_per_drug_matrices(
    directory: str | Path,
    pattern: str = "PosteriorMean_matrix_*.csv",
    perturbation_column: str = "perturbation",
    drugs: Iterable[str] | None = None,
) -> RowboundDataset:
    """Stack per-drug matrices into one rowbound dataset.

    Intersects perturbations and genes across drugs so the result is a complete
    ``perturbations x contexts`` design. Note the FDR matrices use ``target``
    rather than ``perturbation`` as the first column, and carry slightly more
    rows -- intersect by ID before joining them to anything here.
    """
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    if drugs is not None:
        wanted = set(drugs)
        paths = [p for p in paths if _drug_from_path(p, pattern) in wanted]
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {directory}")

    frames = {}
    for path in paths:
        df = pd.read_csv(path, index_col=perturbation_column)
        frames[_drug_from_path(path, pattern)] = df

    common_perts = sorted(set.intersection(*(set(d.index) for d in frames.values())))
    common_genes = sorted(set.intersection(*(set(d.columns) for d in frames.values())))
    if not common_perts or not common_genes:
        raise ValueError("no perturbations or genes shared across all drugs")

    blocks, pert_labels, ctx_labels = [], [], []
    for drug in sorted(frames):
        block = frames[drug].loc[common_perts, common_genes]
        blocks.append(block.to_numpy(dtype=np.float64))
        pert_labels.extend(common_perts)
        ctx_labels.extend([drug] * len(common_perts))

    X = np.vstack(blocks)
    perturbations, perturbation_idx = np.unique(pert_labels, return_inverse=True)
    contexts, context_idx = np.unique(ctx_labels, return_inverse=True)

    return RowboundDataset(
        X=X,
        perturbation_idx=perturbation_idx,
        context_idx=context_idx,
        perturbations=perturbations,
        contexts=contexts,
        genes=np.array(common_genes),
    )


def _drug_from_path(path: Path, pattern: str) -> str:
    prefix, _, suffix = pattern.partition("*")
    name = path.name
    return name[len(prefix) : len(name) - len(suffix)]

