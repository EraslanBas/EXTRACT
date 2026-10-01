# What a row is

The method needs one thing from a screen: that each perturbation's effect can
be written as a differential-expression vector against controls **matched to
its own context**. Everything else is free.

## Perturbations and contexts are labels

Nothing in the model inspects what a perturbation `p` or a context `c` is.
Both are opaque labels that enter only through learned embeddings. The
perturbation can be a CRISPR knockout, knockdown or activation, a compound at
one or several doses, a cytokine, or a defined combination (which is simply its
own label). The context can be a drug background, a cell line or genotype, a
timepoint, a donor or a culture condition. The only structural requirement is
that each context has its **own** controls, because `x` is a within-context
contrast: that is what differences the context's own main effect out of `x` and
leaves the perturbation's effect *in* that context.

With one context the label is `u = p`, and the question is which programs the
perturbation set moves. With several, `u = (p, c)`, and two more questions open
up: how a perturbation's engagement of a program changes with context, and
which programs are responsive in each context.

## Estimation

For perturbation `p` in context `c`, let $\bar y^{\,p}_g$ and
$\bar y^{\,\text{ctrl}}_g$ be per-gene means of $\log(1+\text{CPM})$ over the
perturbed cells and over a **fixed** control set for that context. The estimate
and its standard error are Welch's:

```{math}
\hat\Delta_g = \bar y^{\,p}_g - \bar y^{\,\text{ctrl}}_g,
\qquad
\widehat{\mathrm{se}}_g = \sqrt{\frac{s^{2,p}_g}{n_p} + \frac{s^{2,\text{ctrl}}_g}{n_{\text{ctrl}}}} .
```

Each $(\hat\Delta_g, \widehat{\mathrm{se}}_g)$ pair is passed through adaptive
shrinkage ([`ashr`](https://doi.org/10.1093/biostatistics/kxw041)), which fits
a unimodal prior at zero over all entries of a shard and returns a posterior
mean. The row `x` is the vector of posterior means. The control set is sampled
once per context ($n_{\text{ctrl}} = 10^5$ cells) and reused, so the control
term of the standard error is the same for every row of a context, and
differences in precision between rows come only from $n_p$.

## Eleven rows per pair

A perturbation with at least 50 cells contributes its **full-data** estimate
plus **ten subsampled** re-estimates, with subsample sizes spread over
$[50, n_p)$. The same quantity therefore appears at eleven known precisions.
The subsamples are unbiased re-estimates; only their precision changes. They
are used in three places:

- the **precision weights** of the reconstruction term, $w \propto n$;
- the **stratified negatives**, which swap labels only between rows of the same
  subsample stratum, so precision cannot give a negative away;
- the hyperparameter **`K`**, which keeps the full-data row plus a nested
  random subset of `K` subsamples per pair in training
  (see {doc}`training`).

## No standardisation

`x` is used **unscaled**, exactly as `ashr` produced it. `ashr` has already put
every entry on a common log-fold-change scale; dividing by per-gene standard
deviation would remove that scale, promote low-variance genes to equal footing,
and destroy the units in which `B` is read. It also destroys the spectral gap:
standardisation raises the participation ratio of the data from 8.6 to 81.8, so
no rank-`d` subspace is determined.

## The gene list

Gene-axis dispersion is handled by **dropping** genes no perturbation moves. A
gene that no perturbation shifted in any context belongs to no active program
and contributes only noise. A gene is kept if

```{math}
|x_g| > \ln 1.2 \;\; (\text{a } 20\% \text{ change}) \quad \text{in more than } N \text{ (perturbation, context) pairs},
```

counted on the **full-data rows of the training pairs only**. Subsample rows
re-estimate the same pair and would multiply every count by about eleven, and
held-out pairs must not influence anything that is fitted. The same gene list
is then applied to every partition.

## The on-target entry

A genetic perturbation of gene `p` removes `p`'s own transcript, so that entry
measures how well the reagent worked, not regulation. These entries are the
strongest and most nearly deterministic in the matrix: tens of standard errors
from zero, almost untouched by shrinkage. Left in, they would claim a factor,
and that factor would be knockdown efficiency. Each row therefore carries a
mask $\mathbf m$ that removes its on-target entry from both the projection and
the reconstruction (see {doc}`model`). A compound screen has no such entry and
the mask is empty.
