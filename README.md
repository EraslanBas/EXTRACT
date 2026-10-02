# EXTRACT

**EX**pression programs from **T**reatment **R**esponses **A**cross **C**ontexts
via **T**ensors.

**Documentation: https://eraslanbas.github.io/EXTRACT/**

## What it is for

Perturbation screens are increasingly run in more than one context: the same
CRISPR library under a panel of drugs, across cell lines or genotypes, over a
time course, or in cells from different donors. The result is a three-way array,
perturbation × context × gene, in which every entry of a response vector says
how one perturbation moved one gene in one context.

Each of those response vectors is a messy, high-dimensional readout. It spans
thousands of genes, it carries estimation noise from a finite number of cells,
and underneath it is a mixture of a small number of gene programs that the
perturbation switched on or off. EXTRACT recovers that hidden structure:

- a **shared set of gene programs** (which genes move together), the same for
  every perturbation and every context;
- for every perturbation in every context, **how strongly it engages each
  program**;
- and from those, which part of a perturbation's effect is the **same
  everywhere**, which part is the **context's own**, and which part is an
  **interaction**: where a perturbation's role is rewired by the context it acts
  in.

## Why it matters

**The context dependence is the point of the experiment.** A multi-context
screen is run to find out how a gene's function changes under a drug, in another
genetic background or at another time. Analysing each context separately gives
programs that cannot be compared across contexts; pooling all responses into one
matrix forgets which perturbation and which context produced each row. EXTRACT
keeps the design: one program space for the whole screen, so a perturbation's
effect in one context and in another are read on the same axes, and their
difference is the interaction.

**Programs should be reproducible, not an artefact of the run.** Standard
factorisations of a response matrix (PCA, and the many methods built on it)
pin down the subspace the programs live in but not the individual axes inside
it: any rotation of the factors fits the data equally well, so the "programs"
you read off are one arbitrary choice among many. EXTRACT uses the
perturbation and context labels as extra information. A discriminator that has
to tell whether a response belongs to its (perturbation, context) label, and is
only allowed to score each program separately, can succeed only when the axes
are the underlying programs rather than mixtures of them. That is what makes the
programs identifiable, and the evidence for it is that they recur across
independent fits.

**The output is directly interpretable.** There is exactly one map between
programs and genes, a loading matrix in log-fold-change units, so a program's
gene list is read straight off it, and each program's activity has a clear sign:
up, down, or untouched relative to unperturbed cells of the same context.

**It is tested on combinations it has never seen.** Models are evaluated on
held-out (perturbation, context) pairs whose perturbation and context were each
seen in training but never together. A model that memorised the training pairs
scores at chance there; one that learned the programs, and how contexts reweight
them, does not.

## How it works

EXTRACT fits one global matrix of gene programs `B` (factors × genes). Factor
activations are the masked least-squares projection of each response onto `B`,
so there is one map between factors and genes. `B` is fitted by

    L = L_disc + alpha * L_recon

where reconstruction fixes the subspace of the programs and a per-component
contrastive head (does this response belong to this perturbation in this
context?) fixes their orientation inside it.

## Layout

```
src/extract/      the method
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

The distribution is `extract-screens`; the import name is **`extract`**.

## Install

```bash
pip install -e .                 # or run with PYTHONPATH=src
python -m pytest -q tests/
```

Building logFC matrices from single-cell data additionally needs R with `ashr`.

## Data

The repository holds code only. Data artifacts (matrices, splits, sweep
outputs) live under `$EXTRACT_ROOT`; resolve paths with
`extract.paths`. See [`docs/data_layout.md`](docs/data_layout.md).

## Documentation

```bash
pip install -r docs/requirements.txt
make -C docs html                # -> docs/_build/html/index.html
```

The site is rebuilt and published to GitHub Pages on every push to `main`
(`.github/workflows/docs.yml`).
