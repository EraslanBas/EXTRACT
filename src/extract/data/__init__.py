"""Dataset construction, label augmentation and leak-free splits."""

from .augment import (
    STRATEGIES,
    build_augmented,
    shuffle_matrix,
    Split,
    make_split,
    apply_split,
    load_shuffled,
    permute_labels,
    sample_index,
)
from .pseudoreplicates import PseudoreplicateSet, build_pseudoreplicates
from .gene_filter import (
    affected_counts,
    apply_gene_list,
    compute_gene_list,
    filter_summary,
    load_gene_list,
    select_genes,
)
from .ontarget import mask_coverage, on_target_index
from .splits import (
    PARTITIONS,
    SHUFFLE_LEVELS,
    add_shuffle_levels,
    assign_shuffle_levels,
    build_split,
    draw_test_pairs,
    load_partition,
    split_val_per_perturbation,
    subsample_mask,
    thin_synthetic,
)
from .rowbound import (
    RowboundDataset,
    load_per_drug_matrices,
    load_rowbound,
)

__all__ = [
    "PARTITIONS",
    "SHUFFLE_LEVELS",
    "add_shuffle_levels",
    "assign_shuffle_levels",
    "build_split",
    "draw_test_pairs",
    "load_partition",
    "split_val_per_perturbation",
    "subsample_mask",
    "thin_synthetic",
    "affected_counts",
    "compute_gene_list",
    "load_gene_list",
    "apply_gene_list",
    "select_genes",
    "filter_summary",
    "on_target_index",
    "mask_coverage",
    "build_augmented",
    "permute_labels",
    "shuffle_matrix",
    "make_split",
    "apply_split",
    "load_shuffled",
    "Split",
    "sample_index",
    "STRATEGIES",
    "RowboundDataset",
    "load_rowbound",
    "load_per_drug_matrices",
    "PseudoreplicateSet",
    "build_pseudoreplicates",
]
