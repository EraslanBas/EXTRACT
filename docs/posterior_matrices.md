# Vignette: generating shrunken logFC (posterior-mean) matrices

Produces one matrix per context in which **each perturbation appears several
times** — once from all its cells, plus one row per subsample — with the cell
count behind every row recorded.

This wraps the ChemoGeneticScreens pipeline, whose four scripts are vendored
unchanged in `src/extract/de/`:

| step | script | what it does |
|---|---|---|
| 1 | `ComputeSE.py` | cells → per-label `mean_diff` + Welch `se`, chunked |
| 2 | `run_ashr_on_chunk.R` | `ash(mean_diff, se)` per chunk |
| 3 | `MergeAshrRes.py` | merge label/gene metadata back in |
| 4 | `AshrGenerateMatrix.py` | pivot to wide label × gene `PosteriorMean` |

The only added code builds the *input*: one label per intended output row
(`<pert>` for the full estimate, `<pert>__sub<k>` for subsamples), with
subsampled cells duplicated so a single `ComputeSE.py` pass yields both
variants and one `ashr` fit covers them — which is where their differing
precisions belong, since `ash` conditions on `se`.

**Requires R with the `ashr` package** (`R -e 'install.packages("ashr")'`).

## Run it from the command line, not a notebook

These objects are hundreds of GB resident. A notebook kernel keeps them
reachable, Python does not reliably return freed memory to the OS, and one
re-run of a cell doubles peak — the kernel then dies and takes the whole run
with it. The CLI gives each context its own subprocess, so the OS reclaims
everything on exit and a crash costs one context instead of sixteen.

```bash
python scripts/build_posterior_matrices.py \
    --screen-dir /processed_datasets/VCI/ChemoGenetic_H1_Basak \
    --out-dir    /large_storage/ctc/<user>/ModuleFinder/posterior_matrices \
    --n-control-cells 100000 \
    --n-subsamples 6 --spacing log \
    --n-ashr-jobs 16 \
    --summary summary.csv
```

Defaults to the 16 chemogenetic contexts. It is **resumable** — contexts whose
matrix already exists are skipped — and a failure in one does not stop the
rest. `--help` lists every flag.

One context at a time, to measure before committing:

```bash
python scripts/build_posterior_matrices.py ... --contexts Stattic
```

## Parameters that matter

| flag | default | what it controls |
|---|---|---|
| `--n-subsamples` | 6 | subsamples per eligible perturbation |
| `--spacing` | `linear` | `linear` spreads sizes evenly in `n`; `log` spreads them geometrically |
| `--min-cells-to-subsample` | 50 | a perturbation is subsampled only above this |
| `--n-control-cells` | 10000 | fixed control set, shared by every row |
| `--min-cells` | 20 | `ComputeSE.py` drops perturbations below this entirely |
| `--chunk-perts` | 100 | labels per `ashr` fit |
| `--permute` | off | label-permuted null instead of the real arm |

**Subsample sizes span `[min_cells_to_subsample, n_cells)`.** The full row
already sits at `n_cells`, so the subsamples stop short of it; together they
cover the range end to end. `n_subsamples + 1` points are spread across the
closed range and the top one dropped, so the requested count is honoured.
*Which* cells go into each subsample is random (uniform, no replacement).

Prefer `--spacing log` when the point is to characterise how the estimate
degrades as cells are removed: `se ∝ 1/√n`, so linear spacing under-samples the
low-`n` end where the estimate moves fastest. It is also about half the memory
(below).

**Controls are fixed and shared.** `ComputeSE.py` computes the control
mean/variance once before its per-label loop, so one control group necessarily
serves every label. Fixing its size makes the `var_c / n_ctrl` term of the
standard error identical across rows and comparable across contexts. Note this
also means **subsampling reduces precision only on the perturbation side** — a
50-cell subsample is still compared against the full control set.

To reproduce the published `PosteriorMeanMatrices/`, which used every control
cell, pass `--n-control-cells 1000000`.

## Sizing the memory, which is the binding constraint

Measured on `Stattic` (2.70M cells × 18,154 genes, 241 GB on disk):

* `X` stores **int64 indices**, so it is **12 bytes per nonzero** — the source
  object is **241 GB** in memory, not the ~160 GB an int32 assumption gives.
* The augmented object is larger by the duplicated cells. At `--n-subsamples 6`
  the six sizes sum to **2.64×** each perturbation's own cells with `linear`
  spacing, **1.32×** with `log`.

| `--n-subsamples` | spacing | augmented | peak (source + augmented) |
|---|---|---|---|
| 6 | linear | 777 GB | **1018 GB** |
| 6 | log | 498 GB | **739 GB** |
| 4 | linear | 558 GB | 799 GB |
| 4 | log | 380 GB | 621 GB |
| 3 | log | 323 GB | 563 GB |

**Check your real budget, not `free`.** Inside a Slurm allocation `free`
reports the host while the cgroup caps you far lower:

```bash
CG=/sys/fs/cgroup$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
awk '{printf "limit   %.0f GB\n", $1/1e9}' $CG/memory.max
awk '{printf "current %.0f GB\n", $1/1e9}' $CG/memory.current
cat $CG/memory.events        # oom_kill > 0 means something here was killed
```

A 966 GB allocation cannot run `--n-subsamples 6 --spacing linear` (1018 GB) —
it gets SIGKILLed during `read_h5ad`, which surfaces as `exit -9`. Anything
else sharing the allocation counts against you: a Jupyter kernel holding a big
AnnData can easily be 100+ GB. Find what is holding memory with

```bash
for pid in $(cat $CG/cgroup.procs); do
  awk -v p=$pid '/^VmRSS/{printf "%s %.1f GB\n", p, $2/1048576}' /proc/$pid/status
done | sort -k2 -rn | head
```

and map a kernel back to its notebook via the Jupyter server's
`/api/sessions`.

**Parallelism.** One context at a time per allocation. Real parallelism comes
from separate Slurm jobs with disjoint context lists, not from backgrounding
inside one:

```bash
sbatch --mem=800G --cpus-per-task=20 --wrap "python scripts/build_posterior_matrices.py \
    ... --contexts LDN-193189 VX-11e DG-172 DMSO_round2"
```

Because the script skips finished contexts, splitting the 16 across jobs — or
re-running after a failure — is safe.

## Outputs

```
<out-dir>/<context>/
    PosteriorMean_matrix_<context>.parquet   labels × genes; <pert> rows are
                                             full-data, <pert>__sub<k> subsampled
    row_metadata.csv                         label → perturbation, variant,
                                             subsample, n_cells_used, n_ctrl
    module_finder_manifest.json              parameters, timings, ashr version
    chunk_*.csv                              ComputeSE.py output (mean_diff, se)
    AshrResult_chunk_*.csv                   ashr output, metadata merged back
```

`n_cells_used` comes from the `n_pert` column the pipeline itself writes, not
from a recomputation. Intermediates are kept, so a re-run skips
`ComputeSE.py`; they are ~26 GB per context at full width.

## Python API

For programmatic use — same mechanism, same subprocess isolation:

```python
from extract.de import run_screen

summary = run_screen(
    screen_dir="/processed_datasets/VCI/ChemoGenetic_H1_Basak",
    contexts=["Stattic"],
    out_dir="/large_storage/ctc/<user>/ModuleFinder/posterior_matrices",
    n_control_cells=100_000,
    n_subsamples=6,
    spacing="log",
)
```

`generate_posterior_matrices(adata, context=..., out_dir=...)` runs a single
context from an AnnData already in hand and returns the matrix plus row
metadata; `run_screen(..., isolate=False)` runs in-process, which is useful
only for small data or for seeing a traceback.

The label-permuted null is the same call with `permute=True`. It destroys which
perturbation does what, but the average perturbation effect still reaches every
row because the control baseline is left intact — so it is a null for
**specificity**, not for overall effect magnitude.
