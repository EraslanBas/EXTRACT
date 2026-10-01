# Where the data lives

The repository holds **code only**. Every data artifact lives under one root on
large_storage:

```
/large_storage/ctc/beraslan/ModuleFinder/
```

Override with `EXTRACT_ROOT`. Resolve it in Python via
`extract.paths`; never hard-code a path in a script or notebook.

```python
from extract import paths
paths.root()       # /large_storage/ctc/beraslan/ModuleFinder
paths.matrices()   # .../matrices
```

## Layout

| dir | size | contents |
|---|---|---|
| `computese/` | 990 GB | `ComputeSE.py` + `ashr` intermediates, one dir per context. **Regenerable** — kept only so a re-run skips the SE pass. |
| `matrices/` | 14 GB | `<context>_PosteriorMean.parquet` + `<context>_row_metadata.csv`. What the model consumes. |
| `augmented/` | 15 GB | permuted-label tables (pooled, small) and column-shuffled matrices (one per context). |
| `simulator/` | 40 GB | perturb-seq simulator output — `perturbseq_{A,B,C}.h5ad`, `control_data_*.h5ad`, `grn.pkl`, `gt_*.pkl`. |
| `results/` | 568 MB | baseline fits — `mofa/`, `de_chunks/`. |
| `logs/` | small | driver, matrix-builder and transfer logs. |

## Loading

```python
from extract.de import load_matrices

X, meta = load_matrices()                      # all contexts, all 11 rows each
X, meta = load_matrices(variant="main")         # one row per (perturbation, context)
X, meta = load_matrices(contexts=["Stattic"])
```

`X` is `[rows x genes]` float32 indexed by `row_id = "<context>|<label>"`;
`meta` carries `row_id, label, context, perturbation, variant, n_cells, n_ctrl,
shard` in the same row order. `label` alone is **not** unique — every context
has its own `A1BG__sub00`.

## Scripts

All output paths default to the root, so none needs to be passed:

```bash
python scripts/compute_se_shards.py --screen-dir /processed_datasets/VCI/ChemoGenetic_H1_Basak
python scripts/build_context_matrices.py
python scripts/build_augmented_data.py
```

## History

`data/` and `results/` used to sit inside the repo, which meant two directories
named `EXTRACT` holding unrelated things — the code repo with a `data/` of
simulator artifacts, and the large_storage tree with the screen outputs. They
were consolidated here on 2026-09-16.
