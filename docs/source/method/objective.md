# The objective

```{math}
:class: mf-equation
\mathcal L \;=\; \mathcal L_{\text{disc}} \;+\; \alpha\,\mathcal L_{\text{recon}}
```

The two terms share one parameter set and do different jobs.
$\mathcal L_{\text{recon}}$ selects **which subspace** the factors live in;
$\mathcal L_{\text{disc}}$ selects **their orientation** inside it. A tied
linear autoencoder trained on reconstruction alone recovers the top-$d$
principal subspace and nothing more: the span is determined, the axes inside
it are not ([Baldi & Hornik 1989](https://doi.org/10.1016/0893-6080(89)90014-2)).
Drop the discriminator and the model is PCA with extra steps. Drop the
reconstruction term and the contrastive objective is free to pick a direction
explaining a tiny share of the variance because it happens to separate two
contexts, and that direction's loading vector is not a gene program.

## Three kinds of row

A training row carries two independent bits: whether $\mathbf x$ is a
**measured** response vector, and whether the label attached to it is **its
own**. The two terms read them differently.

| $\mathbf x$ | label | $\mathcal L_{\text{disc}}$ target | $\mathcal L_{\text{recon}}$ weight |
|---|---|---|---|
| measured | own | 1 (real) | $+w_i$ |
| measured | permuted | 0 (fake) | $+w_i$ |
| synthetic (column-shuffled) | own | 0 (fake) | $-\beta\, w_j$ |

A **synthetic** row is a column-shuffled copy of a measured row: each gene's
values are permuted independently across the rows of one context, so per-gene
marginals are preserved exactly and gene–gene structure is destroyed. It keeps
the $(p, c)$ label it was generated for. The label is right; only the vector is
not. That makes it the harder negative: the head cannot reject it on a label
mismatch and must judge whether the vector is a plausible response at all. The
two negatives probe different things through the same head: a permuted label
asks *is this the right response for this label?*, a shuffled vector asks
*is this a response at all?*

## Discrimination

```{math}
\mathcal L_{\text{disc}} = \mathrm{BCE}\big(s(\mathbf x, u),\,1\big) + \mathrm{BCE}\big(s(\mathbf x^{-}, u^{-}),\,0\big),
```

where the first term averages over positives and the second over negatives
(permuted-label and synthetic rows together). A measured vector is never
altered to make a negative; whole rows move as units, so the marginal law of
$\mathbf x$ is preserved and the only way to win is to model the dependence
between $\mathbf x$ and $u$. How negatives are drawn and balanced is on
{doc}`negatives`.

## Reconstruction

```{math}
\mathcal L_{\text{recon}} =
\sum_{i\,\in\,\text{measured}} w_i\,\big\|\mathbf m_i \odot (\mathbf x_i - \mathbf z_i\mathbf B)\big\|^2
\;-\; \beta \sum_{j\,\in\,\text{synthetic}} w_j\,\big\|\mathbf m_j \odot (\mathbf x_j - \mathbf z_j\mathbf B)\big\|^2 ,
```

with $\mathbf z_i$ the masked projection, each squared error divided by the
number of retained genes, and the weights normalised within the batch so
$\alpha$ means the same at any batch size.

- **Measured rows enter with a plus sign whatever label they carry.** A
  label-permuted negative is still a real response vector, so $\mathbf B$
  should reconstruct it.
- **Synthetic rows never enter with a plus sign.** At $\beta = 0$ they are
  dropped from the term; at $\beta > 0$ the subspace is actively pushed away
  from the shuffled cloud, which is contrastive PCA against a marginal-matched
  background. Minimising their error instead would train $\mathbf B$ to span a
  product-of-marginals cloud whose covariance is diagonal by construction and
  whose leading directions are just the highest-variance genes.

**Precision weights.** $w_i \propto n_i$, the cells behind row $i$, which is
what the variance of a mean implies (capped at the 99th percentile so one very
deep perturbation cannot own a batch). Unweighted, a 50-cell subsample would
count as much as a 3,000-cell full-data row whose estimate is several times
more precise. This is the payoff of building eleven rows per pair.

## The weights $\alpha$ and $\beta$

$\alpha$ is, with $d$, the main knob. Its scale is set by the data: the
discrimination term is of order 1, while the reconstruction error per gene is
of order the mean squared logFC, about $10^{-3}$ on the example screen. So
$\alpha$ in the hundreds to thousands is needed for the two terms to be
comparable. Too low and reconstruction is inert, the contrastive term picks
low-variance directions; too high and the model collapses towards PCA and stops
responding to the label. $\beta$ sets how strongly the subspace is pushed off
the shuffled background; both are tuned on validation ({doc}`training`).

## An optional independence term

A total-correlation penalty on $\mathbf z$ (a FactorVAE-style discriminator
between $\mathbf z$ and $\mathbf z$ with each coordinate permuted across the
batch) is available and **off by default**. Independence is not
identifiability: Darmois' construction gives exactly independent components
that are still mixtures of the truth, so the term encourages a property true
sources happen to have and cannot replace the per-component head, which is the
only element that breaks the rotation symmetry.
