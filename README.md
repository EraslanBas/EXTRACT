# ModuleFinder

Identifiable latent factors from perturbation x context screens, plus the
perturb-seq simulator used to generate ground truth and the baselines to beat.

## Layout

```
src/module_finder/      the model (see docs/DESIGN.md)
    data/               (p,c) x gene matrices; fixed-n cell pseudo-replicates
    models/             linear encoder, per-component head, global linear decoder
    objectives/         contrastive discrimination + reconstruction anchor
    interpret/          gene loadings, mu_p / gamma_c / delta_pc, enrichment export
    evaluation/         cross-seed stability, held-out (p,c) generalisation
    train.py            training loop
src/perturbseq_sim/     the simulator, promoted out of the notebooks
src/baselines/          pca, linear_ica, context_jd + mofa/muvi/svae_plus adapters

notebooks/simulator/    01-09, the data-generation pipeline
notebooks/analysis/     context distance sweep
notebooks/baselines/    MOFA on simulated perturb-seq
scripts/de/             ashr / glmGamPoi / DESeq2 pipeline scripts
de_benchmark/           DE-method comparison outputs (53 GB, gitignored)
data/                   simulator artifacts + h5ads (40 GB, gitignored)
results/                de_chunks/, mofa/ (gitignored)
external/               vendored MuVI, ashr (own .git)
reference/go_data/      NCBI/GO downloads (gitignored)
docs/                   DESIGN.md, SIMULATION_PIPELINE.md, notes
tests/                  36 tests, `pytest tests`
```

**The import name is `module_finder`, not `modulefinder`** — the latter is a
Python standard-library module and would be shadowed.

## Install

```bash
pip install -e .                      # or run with PYTHONPATH=src
pytest tests                          # 36 tests, ~3s, no data needed
```

## Quickstart

```python
from module_finder.data import load_rowbound
from module_finder.train import TrainConfig, fit, encode
from module_finder.interpret import decompose_effects, loadings_from_decoder
from module_finder.evaluation import split_pair_holdout, reconstruction_score

ds = load_rowbound(
    "/home/beraslan/Projects/ChemoGeneticScreens/TextFiles/"
    "PosteriorMeanMatrices_rowbound_anyDrugSig_geneMed40.csv",
    context_column="drug",
)                                     # rows are already (perturbation, context)

split = split_pair_holdout(ds.perturbations[ds.perturbation_idx],
                           ds.contexts[ds.context_idx], frac=0.1)

model, history = fit(ds.X[split.train],
                     ds.perturbation_idx[split.train],
                     ds.context_idx[split.train],
                     TrainConfig(n_factors=20))

B = loadings_from_decoder(model.loadings, ds.genes, ds.gene_scale)   # LFC units
Z = encode(model, ds.X)
effects = decompose_effects(Z, ds.perturbations[ds.perturbation_idx],
                            ds.contexts[ds.context_idx])
print(effects.variance_shares())      # perturbation / context / interaction
```

## Run the cheap baselines first

With a global linear decoder this model *is* linear ICA with auxiliary
variables, so the baselines estimate the same model class in minutes. Hold the
contrastive model to beating them on held-out `(p,c)` reconstruction.

```python
import baselines
m = baselines.get("linear_ica")(n_factors=20).fit(ds.X, contexts=...)
m = baselines.get("context_jd")(n_factors=20).fit(ds.X, contexts=...)  # 16-context route
```

If they win, that is a real result about the data, not a failed experiment.

## Notebooks

The simulator notebooks run **unedited** from `notebooks/simulator/`. Each
directory carries thin compatibility shims (`libraries.py`, `GetGOPrograms.py`,
...) that re-export from `src/perturbseq_sim/`, plus a `DATA -> ../../data`
symlink so `Path('./DATA')` still resolves. Delete a shim once its notebook has
been converted to `from perturbseq_sim... import ...`.

## Status

Implemented and tested: the model, both objectives, interpretation, evaluation,
and the `pca` / `linear_ica` / `context_jd` baselines.
Adapters that raise a "not wired up" message: `mofa`, `muvi`, `svae_plus`.
Never run on real screen data yet — every number in `docs/DESIGN.md` marked
synthetic came from generated data.
