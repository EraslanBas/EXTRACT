"""One call: augmented AnnData -> shrunken logFC matrix + per-row cell counts.

Chains input construction (``arms.py``) through the unmodified
ChemoGeneticScreens pipeline (``RunAshrPipeline.sh``: ``ComputeSE.py`` ->
``run_ashr_on_chunk.R`` -> ``MergeAshrRes.py`` -> ``AshrGenerateMatrix.py``).

The resulting matrix carries **both** the full-data and the subsampled
estimates as separate rows, indexed by label (``<pert>`` and
``<pert>__sub<k>``), and the companion ``row_metadata.csv`` records how many
cells went into each row.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .arms import (
    MIN_CELLS_TO_SUBSAMPLE,
    N_CONTROL_CELLS,
    SUBSAMPLE_SEP,
    build_augmented_adata,
)

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "RunAshrPipeline.sh"


@dataclass
class PosteriorRun:
    """Everything one run produced."""

    name: str
    out_dir: Path
    matrix_path: Path
    row_metadata_path: Path
    augmented_h5ad: Path | None
    matrix: pd.DataFrame
    row_metadata: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - display only
        n_full = int((self.row_metadata["variant"] == "full").sum())
        n_sub = int((self.row_metadata["variant"] == "subsample").sum())
        return (
            f"PosteriorRun({self.name!r}: {self.matrix.shape[0]} rows "
            f"({n_full} full + {n_sub} subsampled) x {self.matrix.shape[1]} genes)"
        )


def generate_posterior_matrices(
    adata,
    context: str,
    out_dir: str | Path,
    perturbation_key: str = "target_gene",
    control_label: str = "non-targeting",
    n_subsamples: int = 1,
    min_cells_to_subsample: int = MIN_CELLS_TO_SUBSAMPLE,
    n_control_cells: int = N_CONTROL_CELLS,
    seed: int = 0,
    permute: bool = False,
    permute_seed: int = 0,
    layer: str | None = None,
    min_cells: int = 20,
    chunk_perts: int = 100,
    n_ashr_jobs: int = 6,
    matrix_format: str = "csv",
    keep_augmented_h5ad: bool = False,
    python: str | None = None,
    rscript: str = "Rscript",
) -> PosteriorRun:
    """Generate a shrunken-logFC matrix holding full **and** subsampled rows.

    Parameters
    ----------
    adata
        AnnData (in-memory or backed) or a path to one ``.h5ad``. One context
        per call -- the ChemoGenetic per-drug files carry no drug column, so
        the context name is supplied here and used to name the outputs.
    context
        Name for this context; the matrix lands at
        ``<out_dir>/<name>/PosteriorMean_matrix_<name>.<fmt>`` where ``name``
        is ``context`` (plus ``_permuted_seed<k>`` for the null arm).
    n_subsamples
        Subsamples per eligible perturbation. Perturbations with more than
        ``min_cells_to_subsample`` cells get ``n`` drawn uniformly at random
        from ``[min_cells_to_subsample, n_cells]``, with the cells themselves
        drawn uniformly without replacement; the rest contribute only their
        full row.
    n_control_cells
        Size of the fixed control set for this context, shared by every row
        (full and subsampled alike). Defaults to 10,000. Controls beyond this
        are dropped, so the ``var_c / n_ctrl`` term of the standard error is
        identical across rows and comparable across contexts.
    permute
        Build the label-permuted null instead: perturbation labels are shuffled
        among perturbed cells, controls untouched, cell counts preserved.
    min_cells, chunk_perts, n_ashr_jobs, layer, matrix_format
        Passed through to ``RunAshrPipeline.sh``. ``chunk_perts`` reproduces
        the original per-100-perturbation ashr grouping; full and subsampled
        rows of the same perturbation may land in different chunks, so raise it
        (or set it above the row count) if you want one shared prior.

    Returns
    -------
    PosteriorRun
        With ``matrix`` (rows = labels) and ``row_metadata`` (``label,
        perturbation, variant, subsample, n_cells_used, n_ctrl``).

    Notes
    -----
    Requires R with ``ashr``. Subsampled cells are duplicated in the augmented
    ``.h5ad`` so that one ``ComputeSE.py`` pass yields both variants, which
    means the temporary file is larger than the input -- roughly
    ``1 + n_subsamples/2`` times the perturbed cells.
    """
    import anndata as ad

    if not PIPELINE.exists():
        raise FileNotFoundError(f"pipeline driver missing: {PIPELINE}")
    python = python or sys.executable

    if isinstance(adata, (str, Path)):
        adata = ad.read_h5ad(str(adata), backed="r")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = context if not permute else f"{context}_permuted_seed{permute_seed}"
    label_key = "mf_label"

    started = time.time()
    augmented, plan = build_augmented_adata(
        adata,
        perturbation_key=perturbation_key,
        control_label=control_label,
        label_key=label_key,
        n_subsamples=n_subsamples,
        min_cells_to_subsample=min_cells_to_subsample,
        n_control_cells=n_control_cells,
        seed=seed,
        permute=permute,
        permute_seed=permute_seed,
    )
    h5ad_path = out_dir / f"{name}.h5ad"
    augmented.write_h5ad(h5ad_path)
    build_seconds = time.time() - started
    n_augmented_cells = int(augmented.n_obs)
    del augmented

    env = {
        **os.environ,
        "OUT_BASE": str(out_dir),
        "PYTHON": python,
        "RSCRIPT": rscript,
        "GROUP_KEY": label_key,
        "CONTROL_LABEL": control_label,
        "MIN_CELLS": str(min_cells),
        "CHUNK_PERTS": str(chunk_perts),
        "N_ASHR_JOBS": str(n_ashr_jobs),
        "MATRIX_FORMAT": matrix_format,
        "LAYER": layer or "",
    }
    started = time.time()
    proc = subprocess.run(
        ["bash", str(PIPELINE), str(h5ad_path)],
        env=env,
        capture_output=True,
        text=True,
    )
    pipeline_seconds = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(
            "RunAshrPipeline.sh failed "
            f"(exit {proc.returncode}).\n--- stdout ---\n{proc.stdout[-3000:]}"
            f"\n--- stderr ---\n{proc.stderr[-3000:]}"
        )

    run_dir = out_dir / name
    matrix_path = run_dir / f"PosteriorMean_matrix_{name}.{matrix_format}"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"pipeline reported success but {matrix_path} is missing.\n"
            f"stdout tail:\n{proc.stdout[-2000:]}"
        )
    matrix = (
        pd.read_parquet(matrix_path)
        if matrix_format == "parquet"
        else pd.read_csv(matrix_path, index_col=0)
    )

    row_metadata = _collect_row_metadata(run_dir, plan, context)
    row_metadata_path = run_dir / "row_metadata.csv"
    row_metadata.to_csv(row_metadata_path, index=False)

    meta = {
        "name": name,
        "context": context,
        "permuted": permute,
        "permute_seed": permute_seed if permute else None,
        "perturbation_key": perturbation_key,
        "control_label": control_label,
        "n_subsamples": n_subsamples,
        "min_cells_to_subsample": min_cells_to_subsample,
        "n_control_cells": n_control_cells,
        "seed": seed,
        "min_cells": min_cells,
        "chunk_perts": chunk_perts,
        "layer": layer,
        "n_input_cells": int(adata.n_obs),
        "n_augmented_cells": n_augmented_cells,
        "n_genes": int(adata.n_vars),
        "n_rows": int(matrix.shape[0]),
        "build_seconds": round(build_seconds, 1),
        "pipeline_seconds": round(pipeline_seconds, 1),
        "ashr_version": _ashr_version(rscript),
    }
    (run_dir / "module_finder_manifest.json").write_text(json.dumps(meta, indent=2))

    if not keep_augmented_h5ad:
        h5ad_path.unlink(missing_ok=True)

    return PosteriorRun(
        name=name,
        out_dir=run_dir,
        matrix_path=matrix_path,
        row_metadata_path=row_metadata_path,
        augmented_h5ad=h5ad_path if keep_augmented_h5ad else None,
        matrix=matrix,
        row_metadata=row_metadata,
        meta=meta,
    )


def _collect_row_metadata(
    run_dir: Path, plan: pd.DataFrame, context: str
) -> pd.DataFrame:
    """Per-row cell counts, taken from what the pipeline itself recorded.

    ``ComputeSE.py`` writes ``n_pert``/``n_ctrl`` into every chunk and
    ``MergeAshrRes.py`` merges them into the ashr output, so the counts used
    for each estimate come straight from the pipeline rather than being
    recomputed here.
    """
    frames = []
    for path in sorted(run_dir.glob("AshrResult_chunk_*.csv")):
        frames.append(
            pd.read_csv(path, usecols=["perturbation", "n_pert", "n_ctrl"])
            .drop_duplicates()
        )
    if not frames:
        raise FileNotFoundError(f"no AshrResult_chunk_*.csv under {run_dir}")

    counts = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .rename(columns={"perturbation": "label", "n_pert": "n_cells_used"})
    )
    merged = plan.merge(counts, on="label", how="right")
    merged["context"] = context
    missing = merged["variant"].isna()
    if missing.any():  # label present in output but not in the plan
        merged.loc[missing, "variant"] = "unknown"
    return merged[
        [
            "label",
            "perturbation",
            "context",
            "variant",
            "subsample",
            "n_cells_used",
            "n_cells_planned",
            "n_ctrl",
        ]
    ].sort_values(["perturbation", "variant", "subsample"], na_position="first")


def _ashr_version(rscript: str) -> str | None:
    if shutil.which(rscript) is None:
        return None
    probe = subprocess.run(
        [rscript, "-e", 'cat(as.character(packageVersion("ashr")))'],
        capture_output=True,
        text=True,
    )
    return probe.stdout.strip() or None
