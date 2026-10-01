# ModuleFinder

Gene programs from perturbation × context screens.

ModuleFinder fits one global matrix of gene programs `B` (factors × genes) to a
screen in which every perturbation is measured in several contexts (drug
backgrounds, cell lines, timepoints, donors). Factor activations are the masked
least-squares projection of each response onto `B`, so there is one map between
factors and genes. `B` is fitted by

    L = L_disc + alpha * L_recon

where reconstruction fixes the subspace of the programs and a per-component
contrastive head (does this response belong to this perturbation in this
context?) fixes their orientation inside it.

**Documentation:** https://eraslanbas.github.io/ModuleFinder/ — the method,
page by page.

## Layout

```
src/module_finder/      the method
    de/                 shrunken-logFC pipeline (Welch + ashr), per context
    data/               gene filter, on-target mask, leak-free splits, synthetic rows
    models/             global loadings B, per-component head, label network
    objectives/         contrastive discrimination, precision-weighted reconstruction
    interpret/          loadings, sign anchoring, effect decomposition, enrichment export
    evaluation/         span test, cross-seed stability, held-out evaluation
    train.py            training loop and evaluation
src/perturbseq_sim/     perturb-seq simulator
src/baselines/          PCA, linear ICA, joint diagonalisation, adapters
scripts/                CLIs: logFC matrices, split, fit, sweep, summaries, test read
docs/source/            documentation site (Sphinx)
notebooks/              simulator and analysis notebooks
tests/                  unit tests, no screen data needed
```

The import name is **`module_finder`**, not `modulefinder` (a Python
standard-library module).

## Install

```bash
pip install -e .                 # or run with PYTHONPATH=src
python -m pytest -q tests/
```

Building logFC matrices from single-cell data additionally needs R with `ashr`.

## Data

The repository holds code only. Data artifacts (matrices, splits, sweep
outputs) live under `$MODULEFINDER_ROOT`; resolve paths with
`module_finder.paths`. See [`docs/data_layout.md`](docs/data_layout.md).

## Documentation

```bash
pip install -r docs/requirements.txt
make -C docs html                # -> docs/_build/html/index.html
```

The site is rebuilt and published to GitHub Pages on every push to `main`
(`.github/workflows/docs.yml`).
