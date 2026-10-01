---
sd_hide_title: true
---

# ModuleFinder

::::{grid} 1
:::{grid-item}
:class: sd-text-center sd-fs-1 sd-font-weight-bold

ModuleFinder
:::
:::{grid-item}
:class: sd-text-center sd-fs-5 sd-text-secondary

GENE PROGRAMS FROM PERTURBATION × CONTEXT SCREENS
:::
::::

---

A perturbation screen measured in several contexts (drug backgrounds, cell
lines, timepoints, donors) records how each perturbation moves the
transcriptome *in each context*. ModuleFinder turns those measurements into a
small set of **gene programs** shared across the whole screen, and asks how
each perturbation engages each program and how that engagement changes with
context.

It is one linear model with one interpretable output:

```{math}
:class: mf-equation
\mathbf{x} \;\approx\; \mathbf{z}\,\mathbf{B},
\qquad
\mathbf{z} \;=\; \arg\min_{\mathbf{z}} \big\| \mathbf{m} \odot (\mathbf{x} - \mathbf{z}\mathbf{B}) \big\|^2 .
```

`B` (factors × genes) is a **global** loading matrix, the same for every
perturbation and every context. The factor activations `z` are not produced by
an encoder; they are the least-squares projection of a response onto `B`, so
there is exactly one map between factors and genes. `B` is fitted by two terms
with different jobs:

```{math}
:class: mf-equation
\mathcal{L} \;=\; \mathcal{L}_{\text{disc}} \;+\; \alpha\,\mathcal{L}_{\text{recon}}
```

- **Reconstruction** fixes the *subspace* the programs live in. On its own, a
  tied linear autoencoder recovers the principal subspace and leaves the axes
  inside it free.
- **Discrimination** fixes their *orientation*. A head asks whether a response
  belongs to a given perturbation in a given context, and it is restricted to a
  sum over factors, so its optimum is reachable only in unmixed coordinates.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} What a row is
:link: method/data
:link-type: doc

Within-context logFC, subsampled re-estimates, `ashr` shrinkage, the gene list.
:::

:::{grid-item-card} The model
:link: method/model
:link-type: doc

One factor→gene map, the masked projection, the per-component head, the label network.
:::

:::{grid-item-card} The objective
:link: method/objective
:link-type: doc

Discrimination and reconstruction, measured vs synthetic rows, the `β` term.
:::

:::{grid-item-card} Negative sampling
:link: method/negatives
:link-type: doc

Stratified label permutation, column-shuffled vectors, balancing.
:::

:::{grid-item-card} Why the head is constrained
:link: method/identifiability
:link-type: doc

Why a sum over components picks the rotation, and what breaks it.
:::

:::{grid-item-card} Reading the factors
:link: method/interpretation
:link-type: doc

Gene programs, sign anchoring, perturbation / context / interaction effects.
:::

:::{grid-item-card} Training and model selection
:link: method/training
:link-type: doc

Splits, the hyperparameters, choosing an epoch and a configuration, the test read.
:::

:::{grid-item-card} Evaluation
:link: method/evaluation
:link-type: doc

The span test, cross-seed stability, reconstruction against the truth.
:::

:::{grid-item-card} Glossary
:link: method/glossary
:link-type: doc

Symbols and terms used throughout.
:::

::::

:::{note}
ModuleFinder is under active development. The full specification, with every
equation, is the {download}`method paper (PDF) <../paper/modulefinder.pdf>`. Where the paper
and the current code differ, these pages follow the code.
:::

```{toctree}
:hidden:
:maxdepth: 2

method/index
```
