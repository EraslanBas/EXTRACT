# EXTRACT

**EX**pression programs from **T**reatment **R**esponses **A**cross **C**ontexts
via **T**ensors.

EXTRACT is a contrastive, identifiable factor model for perturbation × context
screens. The screen is a three-way array (perturbation × context × gene), and
EXTRACT recovers the gene programs behind it: one matrix $\mathbf B$ that maps
$d$ latent programs to genes, shared by every perturbation and every context.
It works in two stages. First it chooses the **subspace** the programs live in:
the directions with the most perturbation-driven signal relative to measurement
noise, measured from the screen's own subsampled replicates. Then it chooses the
**axes** inside that subspace, the individual programs, by asking a deliberately
restricted discriminator whether a response vector belongs to the
(perturbation, context) label attached to it.

## What the model is

Each row is a vector $\mathbf x \in \mathbb R^G$ of `ashr`-shrunken log-fold
changes for one perturbation $p$ in one context $c$ (for example a drug
background), measured against controls from the same context. The model assumes
the rows are linear mixtures of $d$ latent programs, $\mathbf x \approx
\mathbf z\mathbf B$, and wants the axes of $\mathbf z$ to be the true programs
rather than an arbitrary rotation of them.

Finding them is two separate problems: *which subspace* the programs span, and
*which axes* inside it are the programs. Variance alone cannot answer either.
The top-variance subspace mixes real perturbation effects with estimation noise
from a finite number of cells, and any rotation of axes inside a subspace fits
the data equally well. EXTRACT answers the two questions with two different
pieces of information.

1. **The subspace, from signal relative to noise.** Each perturbation is
   measured on its full set of cells and on random subsets of them, so the
   difference between a subsample and its full-data row is pure measurement
   noise. From these differences EXTRACT estimates the noise, and keeps the $d$
   directions with the most perturbation-driven variation per unit of noise:
   the **label-driven subspace** $\mathbf V$ ($d \times G$), computed once from
   the training rows before training.
2. **The axes, from the labels.** The programs are $\mathbf B = \mathbf A\mathbf V$,
   where only the $d \times d$ matrix $\mathbf A$ is learned. They are chosen by
   a discriminator that must separate real $(\mathbf x, u)$ pairs, $u = (p, c)$,
   from label-permuted ones and may only score each factor separately. That
   restriction is what breaks the rotation symmetry (an idea from nonlinear ICA
   with auxiliary variables: time-contrastive learning, Hyvärinen & Morioka 2016).

$$
\mathbf B = \mathbf A\,\mathbf V, \qquad \mathcal L = \mathcal L_{\text{disc}}
\quad\big[\,+\;\lambda_{\text{tc}}\,\mathcal L_{\text{tc}}\,\big]
$$

This is the structure of ICA (choose a subspace, then unmix inside it), with the
two criteria replaced: signal-to-noise from the subsamples instead of variance,
and label-based discrimination instead of non-Gaussianity. Because the subspace
cannot move during training, the programs are pinned to directions that carry
perturbation-driven signal, and the discriminator cannot drag them toward
directions that separate training labels but are mostly noise.

The earlier, one-stage version learns $\mathbf B$ directly with
$\mathcal L = \mathcal L_{\text{disc}} + \alpha\,\mathcal L_{\text{recon}}$,
reconstruction choosing the subspace and discrimination the axes; it remains
available as `subspace="free"`.

## What a row is

The model never sees counts. The `de/` package turns cell-level single-cell
data into one matrix per context:

1. **Plan the rows.** Each perturbation gets a full-data label `<pert>` plus
   subsample labels `<pert>__subNN`. A perturbation with at least 50 cells gets
   ten subsample sizes spanning $[50, n_p)$, with random cells inside each
   size.
2. **Fix the control set.** One control subset per context ($10^5$ cells) is
   drawn once, saved, and shared by every row, so the control term of the
   standard error is identical across rows and differences in precision come
   only from $n_p$.
3. **Welch statistics.** On the $\log(1+\text{CPM})$ scale, per gene,
   $\hat\Delta_g = \bar y^{\,p}_g - \bar y^{\,\text{ctrl}}_g$ and
   $\widehat{\mathrm{se}}_g = \sqrt{s^{2,p}_g/n_p + s^{2,\text{ctrl}}_g/n_{\text{ctrl}}}$.
4. **Shrink.** `ashr` runs on $(\hat\Delta_g, \widehat{\mathrm{se}}_g)$. Full
   and subsampled rows share one prior, and `ashr` conditions on the standard
   error, so their different precisions are handled in one fit. The row
   $\mathbf x$ is the vector of posterior means.
5. **Assemble.** Each context's posterior means become one labels × genes
   matrix, with row metadata carrying `row_id = "<context>|<label>"`,
   `n_cells` and `variant` (full-data or subsample).

Two row-level facts are carried into training. The **stratum** $s$ (0 for the
full-data row, $k+1$ for `__sub{k}`) is the row's precision tier. The **cell
count** $n$ drives the reconstruction weight.

### The gene axis

No centring or per-gene scaling is applied anywhere. `ashr` has already put
every entry on a common logFC scale; dividing by per-gene standard deviation
would promote low-variance genes and erase the spectral gap (it raises the
participation ratio of the data from 8.6 to 81.8). Instead, genes that no
perturbation moves are dropped: a gene is kept if $|x| > \ln 1.2$ (a 20%
change) in more than 150 (perturbation, context) pairs, counted on the
full-data rows of the training pairs only. The same list is used for every
partition and every fit.

### The on-target mask

Knocking down gene X drops X's own transcript by roughly its control mean.
These entries sit 40 to 50 standard errors from zero and carry a median 25% of
a row's squared norm. Left in, they would claim a factor that encodes knockdown
efficiency. Each row is mapped to its own gene's column (`NO_MASK = −1` for
controls or unmatched names), and that column is excluded from the projection,
the reconstruction error and the span residual alike. A compound screen simply
has no masked entry.

## The label-driven subspace

$\mathbf V$ is the space the programs are allowed to live in. It is computed
once, before training, from the measured rows of the training pairs. The aim is
simple: keep the directions where responses differ **because of the
perturbation**, not because of measurement noise.

Nothing is done to the data values: no row is shrunk or denoised, and the
rows enter the model exactly as `ashr` produced them. The noise estimate only
decides **which directions** of gene space count as important. It works with
the variance of the rows along each direction.

**What a full-data row contains.** Each full-data row is the true response plus
measurement noise. Along any direction (a weighted combination of genes) the
two add, because they are independent:

$$
\text{variance along a direction} \;=\; \text{signal variance} \;+\; \text{noise variance}.
$$

The left side is measured directly from the full-data rows. PCA ranks
directions by it, mixing the two.

**Measuring the noise in each direction.** Every perturbation is also measured
on random subsets of its cells (the subsample rows). A subsample and its
full-data row share the same true response, so their difference contains no
signal, only noise. How much these differences vary along a direction gives the
noise variance in that direction. Noise variance scales as one over the number
of cells, so the differences are rescaled to the noise level of a full-data
row.

**Subtracting it.** For every direction,
signal variance = total variance (full-data rows) − noise variance (subsample
differences). This is done for all directions at once with covariance
matrices. $\mathbf V$ is the $d$ directions with the largest ratio of signal
variance to noise variance. Noise shared across many genes (for example a
cell-quality axis) counts as noise; each gene's own scale is left untouched.

For example, if direction A has total variance 10 of which 8 is noise, and
direction B has total variance 6 of which 1 is noise:

| direction | total variance | noise | signal | signal ÷ noise |
|---|---|---|---|---|
| A | 10 | 8 | 2 | 0.25 |
| B | 6 | 1 | 5 | 5 |

PCA prefers A, because it varies more; $\mathbf V$ prefers B, because its
variation is mostly perturbation-driven.

| | picks directions by | uses the subsamples |
|---|---|---|
| PCA | total variation (signal + noise) | no |
| ICA | PCA's directions, then the most independent axes inside them | no |
| $\mathbf V$ | perturbation-driven variation relative to measured noise | yes |

### Computing $\mathbf V$ in closed form

$\mathbf V$ is not learned by gradient descent and does not change during
training. It is the solution of an eigenvalue problem, computed once from the
measured rows of the training pairs (`extract.data.noise`).

**The measurement model.** Write every row as a vector in $\mathbb R^G$. For a
training pair $i$ (one perturbation in one context), let $\mathbf x_i^{(0)}$ be
its full-data row, estimated from $n_i$ cells, and $\mathbf x_i^{(s)}$ a
subsample row, estimated from $n_{is} < n_i$ of the same cells. Each is the
pair's true response $\boldsymbol\mu_i$ plus estimation noise,

$$
\mathbf x_i^{(0)} = \boldsymbol\mu_i + \boldsymbol\varepsilon_i^{(0)},
\qquad
\mathbf x_i^{(s)} = \boldsymbol\mu_i + \boldsymbol\varepsilon_i^{(s)},
\qquad
\operatorname{Cov}\big(\boldsymbol\varepsilon\big) = \frac{\boldsymbol\Sigma_{\text{cell}}}{n},
$$

where $n$ is the number of cells behind the row and $\boldsymbol\Sigma_{\text{cell}}$
($G \times G$) is the per-cell noise covariance across genes. The control mean
is estimated once per context from a fixed set of cells, so its error is the
same in every row of a context and cancels in the differences below.

**Step 1: the noise, from subsample differences.** A subsample uses a subset of
the full row's cells, so the two noise terms are correlated:
$\operatorname{Cov}(\boldsymbol\varepsilon^{(s)}, \boldsymbol\varepsilon^{(0)}) = \boldsymbol\Sigma_{\text{cell}}/n_i$
(the mean of a set and the mean of its subset share the subset's cells). Their
difference has no signal, and

$$
\operatorname{Cov}\big(\mathbf x_i^{(s)} - \mathbf x_i^{(0)}\big)
= \frac{\boldsymbol\Sigma_{\text{cell}}}{n_{is}} - 2\,\frac{\boldsymbol\Sigma_{\text{cell}}}{n_i} + \frac{\boldsymbol\Sigma_{\text{cell}}}{n_i}
= \Big(\frac{1}{n_{is}} - \frac{1}{n_i}\Big)\boldsymbol\Sigma_{\text{cell}} .
$$

So every scaled difference
$\mathbf r = (\mathbf x_i^{(s)} - \mathbf x_i^{(0)}) \big/ \sqrt{1/n_{is} - 1/n_i}$
has covariance exactly $\boldsymbol\Sigma_{\text{cell}}$, whatever the cell counts.
Stacking up to 150,000 of them gives a matrix $\mathbf R$ ($N \times G$).

**Step 2: a noise model with shared directions.** $\boldsymbol\Sigma_{\text{cell}}$
has $G^2 \approx 2 \times 10^7$ entries, too many to estimate freely, so it is
modelled as per-gene noise plus $r = 50$ directions of noise shared across genes:

$$
\widehat{\boldsymbol\Sigma}_{\text{cell}} = \mathbf D + \mathbf U \operatorname{diag}(\mathbf s)\,\mathbf U^\top,
$$

with $\mathbf D$ diagonal, $\mathbf U$ ($G \times r$) orthonormal and
$\mathbf s \ge 0$. $\mathbf U$ is the top-$r$ right singular vectors of $\mathbf R$,
with $\lambda_k = \sigma_k^2 / N$ the variance along each. That variance also
contains the per-gene noise along $\mathbf u_k$, so $\mathbf D$ and $\mathbf s$ are
separated by a few factor-analysis alternations, starting from
$t_g = \tfrac1N \sum_n R_{ng}^2$ (each gene's total noise variance):

$$
s_k \leftarrow \lambda_k - \textstyle\sum_g U_{gk}^2 D_g ,
\qquad
D_g \leftarrow t_g - \textstyle\sum_k U_{gk}^2 s_k .
$$

**Step 3: the signal.** The full-data rows of the $P$ training pairs give the
second moment (uncentred, because rows are differences from controls and zero
is the reference):

$$
\frac{1}{P}\sum_i \mathbf x_i^{(0)\top}\mathbf x_i^{(0)}
= \underbrace{\frac{1}{P}\sum_i \boldsymbol\mu_i^\top\boldsymbol\mu_i}_{\boldsymbol\Sigma_S}
\;+\; \overline{1/n}\;\boldsymbol\Sigma_{\text{cell}},
\qquad
\widehat{\boldsymbol\Sigma}_S = \frac{1}{P}\sum_i \mathbf x_i^{(0)\top}\mathbf x_i^{(0)} - \overline{1/n}\;\widehat{\boldsymbol\Sigma}_{\text{cell}},
$$

where $\overline{1/n}$ is the mean of $1/n_i$. $\boldsymbol\Sigma_S$ is the covariance
of the true responses: the perturbation-driven signal with the estimation noise
removed.

**Step 4: a noise metric that keeps gene scales.** The comparison in step 5 is
made against

$$
\mathbf M = c\,\mathbf I + \mathbf U \operatorname{diag}(\mathbf s)\,\mathbf U^\top,
\qquad c = \text{mean of } \mathbf D .
$$

$\mathbf M$ is the noise model with its per-gene part $\mathbf D$ replaced by its
average $c$. Shared noise directions are counted as noise, but no gene is up- or
down-weighted for having a small or large noise of its own, so the gene axis is
never rescaled.

**Step 5: maximise signal per unit of noise.** For a direction $\mathbf v$ in gene
space, $\mathbf v^\top\boldsymbol\Sigma_S\mathbf v$ is the signal variance along it and
$\mathbf v^\top\mathbf M\mathbf v$ the noise variance. $\mathbf V$ is built from the $d$
directions that maximise their ratio (a generalised Rayleigh quotient), which
are the leading solutions of the generalised eigenproblem

$$
\boldsymbol\Sigma_S\,\mathbf v_k = \lambda_k\,\mathbf M\,\mathbf v_k,
\qquad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d ,
$$

where $\lambda_k$ is the signal-to-noise ratio along $\mathbf v_k$. It is solved by
whitening with $\mathbf M^{-1/2}$, which has a closed form because $\mathbf U$ is
orthonormal,

$$
\mathbf M^{-1/2} = c^{-1/2}\Big(\mathbf I - \mathbf U \operatorname{diag}\big(1 - \sqrt{c/(c + s_k)}\big)\,\mathbf U^\top\Big),
$$

and taking the top-$d$ eigenvectors $\mathbf w_k$ of the symmetric matrix
$\mathbf M^{-1/2}\boldsymbol\Sigma_S\mathbf M^{-1/2}$; then $\mathbf v_k = \mathbf M^{-1/2}\mathbf w_k$.

**Step 6: patterns, not filters.** The eigenvectors $\mathbf v_k$ are *filters*:
the weights that best read the signal out of a noisy row. The gene programs
must be *patterns*, the directions the signal itself points along. From the
eigenproblem, $\mathbf M\mathbf v_k = \boldsymbol\Sigma_S\mathbf v_k / \lambda_k$, so the
pattern $\mathbf a_k = \mathbf M\mathbf v_k = \mathbf M^{1/2}\mathbf w_k$ lies in the span of the
signal covariance $\boldsymbol\Sigma_S$. The rows of $\mathbf V$ are these patterns,
scaled to unit length:

$$
\mathbf V = \big[\,\mathbf a_1/\lVert\mathbf a_1\rVert;\; \dots;\; \mathbf a_d/\lVert\mathbf a_d\rVert\,\big] \in \mathbb R^{d \times G},
$$

in logFC units, like the data.

**Special case.** If the noise were the same in every direction
($\mathbf M \propto \mathbf I$), the eigenproblem would reduce to the ordinary
eigenvectors of $\boldsymbol\Sigma_S$: PCA of the responses after the noise has
been subtracted. $\mathbf V$ differs from plain PCA in two ways: noise variance is
subtracted from the variation before ranking directions, and directions of
shared noise are discounted.

**Caveat.** `ashr` shrinks rows of different precision by different amounts, so
the noise of a shrunken row is only approximately $\boldsymbol\Sigma_{\text{cell}}/n$;
the construction is exact for unshrunken estimates.

$\mathbf V$ fixes only the space. Which programs sit inside it is decided during
training, by the labels (next section). `fit()` computes $\mathbf V$
automatically (`extract.data.noise`); a precomputed basis can also be passed in.

## Architecture

There are three learned parts and no encoder: the axes $\mathbf A$ that make the
loadings $\mathbf B = \mathbf A\mathbf V$, a label network, and a per-component head
(`extract.train.Extract`).

```{raw} html
<div class="ex-figure">
<svg viewBox="0 0 780 330" width="780" height="330" role="img" aria-label="Forward pass: x is masked and projected through the pseudo-inverse of B to z; basis statistics of z are multiplied by label coefficients lambda and summed into a logit; z times B reconstructs x.">
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ah"/></marker>
    <marker id="aB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahB"/></marker>
  </defs>
  <rect class="box" x="16" y="36" width="104" height="52" rx="4"/>
  <text x="68" y="59" text-anchor="middle" font-size="15" font-weight="600">x</text>
  <text x="68" y="77" text-anchor="middle" class="sub">logFC · G genes</text>
  <rect class="box" x="150" y="36" width="130" height="52" rx="4"/>
  <text x="215" y="59" text-anchor="middle" font-size="13.5">mask on-target</text>
  <text x="215" y="77" text-anchor="middle" class="sub">x̃ = m ⊙ x</text>
  <rect class="box" x="310" y="36" width="140" height="52" rx="4"/>
  <text x="380" y="59" text-anchor="middle" font-size="13.5">z = x̃ B⁺</text>
  <text x="380" y="77" text-anchor="middle" class="sub">Sherman–Morrison · d</text>
  <rect class="box" x="480" y="36" width="140" height="52" rx="4"/>
  <text x="550" y="59" text-anchor="middle" font-size="13.5">q<tspan font-size="10" dy="3">j</tspan><tspan dy="-3">(z</tspan><tspan font-size="10" dy="3">k</tspan><tspan dy="-3">)</tspan></text>
  <text x="550" y="77" text-anchor="middle" class="sub">4 statistics · d×4</text>
  <circle class="box" cx="668" cy="62" r="20"/>
  <text x="668" y="67" text-anchor="middle" font-size="15">Σ</text>
  <rect class="box" x="706" y="36" width="62" height="52" rx="4"/>
  <text x="737" y="59" text-anchor="middle" font-size="13.5">s(x,u)</text>
  <text x="737" y="77" text-anchor="middle" class="sub">logit</text>
  <path class="ln" d="M120 62 H148" marker-end="url(#a)"/>
  <path class="ln" d="M280 62 H308" marker-end="url(#a)"/>
  <path class="ln" d="M450 62 H478" marker-end="url(#a)"/>
  <path class="ln" d="M620 62 H646" marker-end="url(#a)"/>
  <path class="ln" d="M688 62 H704" marker-end="url(#a)"/>
  <rect class="boxB" x="310" y="150" width="140" height="52" rx="4"/>
  <text x="380" y="173" text-anchor="middle" font-size="14" font-weight="600">B</text>
  <text x="380" y="191" text-anchor="middle" class="sub">B = A V · A learned</text>
  <path class="lnB" d="M380 150 V90" marker-end="url(#aB)"/>
  <text x="388" y="124" class="sub">B⁺ = Bᵀ(BBᵀ+εI)⁻¹</text>
  <rect class="box" x="150" y="150" width="130" height="52" rx="4"/>
  <text x="215" y="173" text-anchor="middle" font-size="13.5">x̂ = z B</text>
  <text x="215" y="191" text-anchor="middle" class="sub">no bias term</text>
  <path class="lnB" d="M310 176 H282" marker-end="url(#aB)"/>
  <path class="ln" d="M322 88 L262 148" marker-end="url(#a)"/>
  <rect class="loss" x="16" y="150" width="104" height="52" rx="4"/>
  <text x="68" y="173" text-anchor="middle" font-size="13.5">L_recon</text>
  <text x="68" y="191" text-anchor="middle" class="sub">masked · w ∝ n</text>
  <path class="ln" d="M150 176 H122" marker-end="url(#a)"/>
  <path class="ln" d="M68 88 V148" stroke-dasharray="3 3" marker-end="url(#a)"/>
  <rect class="box" x="310" y="256" width="140" height="52" rx="4"/>
  <text x="380" y="279" text-anchor="middle" font-size="13.5">u = (p, c)</text>
  <text x="380" y="297" text-anchor="middle" class="sub">own or permuted</text>
  <rect class="box" x="480" y="256" width="140" height="52" rx="4"/>
  <text x="550" y="279" text-anchor="middle" font-size="13.5">[e_p ; e_c] → MLP</text>
  <text x="550" y="297" text-anchor="middle" class="sub">λ(u) · d×4</text>
  <path class="ln" d="M450 282 H478" marker-end="url(#a)"/>
  <path class="ln" d="M620 282 H668 V84" marker-end="url(#a)"/>
  <text x="660" y="230" text-anchor="end" class="sub">λ ⊙ q(z)</text>
  <rect class="loss" x="706" y="150" width="62" height="52" rx="4"/>
  <text x="737" y="173" text-anchor="middle" font-size="13.5">L_disc</text>
  <text x="737" y="191" text-anchor="middle" class="sub">BCE</text>
  <path class="ln" d="M737 88 V148" marker-end="url(#a)"/>
</svg>
</div>
<p class="ex-caption">One forward pass. Blue marks B = A V and the two maps derived from it; red marks the loss terms. V is fixed, so the reconstruction branch is constant during training and only reported; the label u reaches the score only through λ, which multiplies per-factor statistics before the sum.</p>
```

### Loadings: one map, no encoder

$\mathbf B = \mathbf A\mathbf V$ has shape $d \times G$: $\mathbf V$ is fixed and only
the $d \times d$ matrix $\mathbf A$ is learned, starting from the identity.
$\mathbf A$ is any invertible matrix, not just a rotation, so the programs can be
oblique: correlated with each other and overlapping in genes. Factor activations
are not predicted by a network; they are the least-squares coordinates of
$\mathbf x$ in $\mathbf B$'s row space:

$$
\mathbf z = \mathbf x\,\mathbf B^\top(\mathbf B\mathbf B^\top + \varepsilon\mathbf I)^{-1} .
$$

The reason is interpretive. A separate encoder would give two factor → gene
maps that can disagree about what factor $k$ means, and encoder weights are
filters, not patterns (Haufe et al. 2014): a gene can get a large weight
precisely to cancel it. Deriving the projection from $\mathbf B$ makes
$\mathbf B$ a pattern by construction. The ridge $\varepsilon = 10^{-4}$ keeps
the Gram matrix well conditioned.

**Masked projection.** With one excluded column per row, that row's Gram matrix
is $\mathbf G - \mathbf b\mathbf b^\top$, where $\mathbf b$ is $\mathbf B$'s
column for the masked gene. One $d \times d$ inverse is computed per step and a
Sherman–Morrison correction applied per row,

$$
(\mathbf G - \mathbf b\mathbf b^\top)^{-1}
= \mathbf G^{-1} + \frac{\mathbf G^{-1}\mathbf b\,\mathbf b^\top\mathbf G^{-1}}{1 - \mathbf b^\top\mathbf G^{-1}\mathbf b},
$$

so each row costs $O(d^2)$ however many distinct genes a batch masks. Rows whose
denominator falls below $10^{-6}$ are re-solved directly against a freshly
built Gram matrix.

The reconstruction is $\hat{\mathbf x} = \mathbf z\mathbf B$ with no bias,
since $\mathbf x$ is already a difference from controls. The squared error
skips the masked entry and is divided by the number of retained genes, so its
scale does not depend on $G$.

### Label network: where all the interaction lives

The label network maps $(p, c)$ to coefficients
$\boldsymbol\lambda \in \mathbb R^{d \times J}$. It concatenates a perturbation
embedding $e_p$ and a context embedding $e_c$ (32 dimensions each) and passes
them through `Linear(64, 128) → ReLU → Linear(128, d·J)`.

The factorisation matters. A free embedding per observed pair would let the
optimal discriminator become a lookup table ("answer real if and only if
$\mathbf x$ is the vector stored for $u$"), reachable without any notion of
components. Sharing $e_p$ across contexts and $e_c$ across perturbations cuts
the parameters from $n_{\text{pairs}} \cdot d$ to $(n_p + n_c) \cdot d$ and
makes memorisation impossible. It is also what lets a held-out pair, whose
perturbation and context were both seen but never together, be scored at all.
The network itself is unconstrained: the identifiability restriction applies
only to how $\boldsymbol\lambda$ meets $\mathbf z$.

### Per-component head: the identifiability constraint

$$
s(\mathbf x, u) \;=\; \sum_{k=1}^{d}\sum_{j=1}^{J} \lambda_{kj}(u)\, q_j(z_k) \;+\; b,
\qquad
\Pr(\text{real} \mid \mathbf x, u) = \sigma\big(s(\mathbf x, u)\big)
$$

The basis $q$ is (linear, square, abs, tanh), applied coordinate-wise, so
$J = 4$. No term ever multiplies $z_k$ by $z_{k'}$. The head's only learned
parameter is the bias $b$; all flexibility sits in $\boldsymbol\lambda(u)$,
and in $\mathbf z$ through $\mathbf B$.

**Why this identifies the axes.** The Bayes-optimal discriminator for real
versus permuted labels is $\log p(\mathbf x \mid u) / p(\mathbf x)$. If the
sources are independent given $u$, that ratio is a sum of per-source terms,
$\sum_i \big[\log p_i(s_i \mid u) - \log p_i(s_i)\big]$: additively separable
in the true coordinates. Separability is not preserved under mixing, so a
separable head can reach the optimum only when its coordinates are the sources,
up to permutation and element-wise transformation. An MLP head could absorb any
$\mathbf z \to \mathbf M\mathbf z$ in its first layer, and the axes would float.

Two guards follow. A basis of only `square` is refused, because
$\sum_k z_k^2$ is rotation invariant (the same reason Gaussian sources are
unidentifiable in linear ICA). And the per-factor sums
$\psi_k = \sum_j \lambda_{kj} q_j(z_k)$ are evidence added before a single
decision, not per-factor votes.

The identifiability theorem of generalised contrastive learning needs
$p(\mathbf s \mid u)$ to vary with $u$ in a non-degenerate way; here the spread
of rows within a pair is subsampling noise, which the stratified negatives
deliberately hide from the discriminator. The theorem therefore does not
transfer exactly, and only the structural argument above is claimed.

### Shapes

| Tensor | Shape | Where | Note |
|---|---|---|---|
| `x` | [batch, G] | input | unscaled shrunken logFC |
| `target_col` | [batch] | input | column to mask, −1 for none |
| `B` | [d, G] | loadings | the only interpretable output |
| `(BBᵀ + εI)⁻¹` | [d, d] | loadings | one inverse per step |
| `z` | [batch, d] | projection | derived, not fitted |
| `q(z)` | [batch, d, J] | head | J = 4 |
| `e_p`, `e_c` | [batch, 32] | label network | shared embeddings |
| `λ` | [batch, d, J] | label network | label-dependent coefficients |
| `s` | [batch] | head | real/fake logit |

## Objective

### Discrimination fixes the orientation

$$
\mathcal L_{\text{disc}} = \mathrm{BCE}\big(s(\mathbf x, u), 1\big) + \mathrm{BCE}\big(s(\mathbf x^{-}, u^{-}), 0\big)
$$

Each term is averaged over its own rows. A label-permuted negative keeps the
observed vector and swaps the label; whole rows move as units, so the marginal
law of $\mathbf x$ is preserved exactly and the only way to win is to model how
$\mathbf x$ depends on $u$.

### Reconstruction

With $\mathbf B = \mathbf A\mathbf V$ the span cannot move, and the projection onto a
span does not depend on the axes chosen inside it, so every reconstruction
error below is **constant during training**: the subspace was already chosen,
by signal-to-noise, when $\mathbf V$ was computed. The term, and the weights
$\alpha$ and $\beta$, act only in the one-stage `free` model, where
reconstruction is what fixes the subspace. Reconstruction errors are still
recorded for every model, as a diagnostic.

A row carries two bits: whether $\mathbf x$ is a **measured** response and
whether its label is **its own**. A **synthetic** row is a column-shuffled copy
of a measured row (each gene's values permuted across the rows of one context)
that keeps the label it was made for. The two loss terms treat the three kinds
of row as follows:

| Row | Label | $\mathcal L_{\text{disc}}$ | $\mathcal L_{\text{recon}}$ |
|---|---|---|---|
| measured | own | positive | $+w$ |
| measured | permuted | negative | $+w$ (same vector) |
| synthetic | own | negative | $-\beta w$ |

$$
\mathcal L_{\text{recon}} =
\sum_{i \in \text{measured}} \tilde w_i\,\frac{\|\mathbf m_i \odot (\mathbf x_i - \mathbf z_i\mathbf B)\|^2}{n_{\text{kept},i}}
\;-\;\beta \sum_{j \in \text{synthetic}} \tilde w_j\,\frac{\|\mathbf m_j \odot (\mathbf x_j - \mathbf z_j\mathbf B)\|^2}{n_{\text{kept},j}}
$$

Measured rows enter with a plus sign whatever label they carry: a permuted
negative is still a real response vector, so $\mathbf B$ should reconstruct it.
Synthetic rows never enter with a plus sign. At $\beta = 0$ they are dropped
from the term; at $\beta > 0$ the subspace is pushed away from the shuffled
cloud, which is contrastive PCA against a marginal-matched background.
Minimising their error would instead train $\mathbf B$ to span a cloud whose
covariance is diagonal by construction.

**Precision weights.** $w_i \propto n_i$, since the variance of a mean scales as
$1/n$, capped at the 99th percentile so one deep perturbation cannot own a
batch, and renormalised within each batch so $\alpha$ means the same at any
batch size. Without weighting, a 50-cell row would count as much as a
3,000-cell one.

**The weight $\alpha$.** The discrimination term is of order 1, while the
reconstruction error per gene is of order the mean squared logFC, about
$10^{-3}$. So $\alpha$ in the hundreds to thousands is needed for the terms to
interact. Too low and $\mathcal L_{\text{disc}}$ picks tiny-variance directions
that separate contexts but carry no gene program; too high and the model
collapses towards PCA and stops responding to the label.

### Independence term: optional, off by default

A total-correlation penalty (FactorVAE style: an MLP learns to separate
$\mathbf z$ from a coordinate-wise batch shuffle) is available. Independence is
not identifiability: Darmois' construction gives independent components that are
still mixtures, so this term cannot do the head's job.

## Negatives

Naive label permutation is solved by precision rather than biology: the noise
scale of $\mathbf x$ encodes $n$, and $n$ is tied to perturbation identity. The
negative sampler therefore draws the replacement label from the **same
stratum** $s$, which cuts the median $|\log_2(n_{\text{real}}/n_{\text{neg}})|$
from 1.75 to 0.62.

| Strategy | Weight | Swap | Forces |
|---|---|---|---|
| same stratum, other perturbation | 0.5 | $p$, within $(c, s)$ | perturbation identity |
| same stratum, other context | 0.5 | $c$, within $(p, s)$ | the interaction $\delta_{pc}$ |

Contexts are symmetric: no strategy assumes a reference arm such as vehicle.
With one context only the first strategy exists, and it suffices.

- **Only observed pairs.** The sampler is built from training rows only, and
  every drawn label is a pair observed among them, so a validation or test pair
  is never named, not even as a negative. If one axis has no observed
  alternative the other axis is swapped; if neither has, it is an error rather
  than a silent leak.
- **Synthetic rows** are negatives under their own label, so they need no
  sampler. They must be shuffled across rows within a gene, never across genes
  within a row: per-gene SD spans about $5 \times 10^4$, so a misplaced value
  would be detectable by magnitude alone (Cohen's $d = 0.59$ against $0.00$
  for column shuffling).
- **Balancing.** Fakes are balanced against real rows: permuted negatives are
  kept with probability $1 - n_{\text{synthetic}}/n_{\text{real}}$, computed from
  global counts, so $|\text{permuted}| + |\text{synthetic}| = |\text{positives}|$.
  Synthetic rows are loaded at half the number of measured rows, so they make up
  half the negatives.

## Training loop

`fit()` takes the rows, their integer-encoded perturbation and context, stratum,
cell count, on-target column, an `is_real` flag and train/validation masks.
First, unless a basis is supplied, it computes the label-driven subspace
$\mathbf V$ from the measured training rows and initialises $\mathbf A$ to the
identity. Then, per batch:

1. Draw training rows; the sampler returns a permuted label $(p^*, c^*)$ for
   each measured row.
2. Project with the on-target mask: $\mathbf z = $ `project(x, target_col)`.
3. Score positives $s(\mathbf z, p, c)$ and negatives: $s(\mathbf z, p^*, c^*)$
   for (thinned) measured rows and $s(\mathbf z, p, c)$ for synthetic rows. The
   same $\mathbf z$ serves both.
4. Compute $\mathcal L_{\text{disc}}$ and the signed, precision-weighted masked
   reconstruction.
5. Step on $\mathcal L_{\text{disc}}$ (plus $\alpha\,\mathcal L_{\text{recon}}$ in the
   `free` model) with Adam (learning rate $10^{-3}$, gradient-norm clip 5),
   updating $\mathbf A$, the label network and the head bias.
6. If the independence term is on, step its discriminator on the detached
   $\mathbf z$.

At regular intervals the validation rows are scored with a sampler built on
validation rows, and train and validation discrimination and reconstruction are
recorded per epoch. $\mathbf B$ can be snapshotted at every evaluation. With only
$\mathbf A$ and the label network learning, validation accuracy typically peaks
within the first few dozen epochs.

## Splits and model selection

**Three partitions, split by pair.** All rows of a (perturbation, context) pair,
full-data and subsamples, share cells, so they always move together.

| Partition | Drawn | Used for |
|---|---|---|
| test | 10% of pairs, first, from metadata alone; every perturbation keeps a train/val pair | read once, at the end |
| validation | about 10% of each perturbation's train/val pairs, at least one kept in train | choosing the epoch and the hyperparameters |
| train | the rest | fitting |

A held-out pair is an unseen combination of a seen perturbation and a seen
context: the model has to supply the interaction. Synthetic rows are made by
shuffling each context's rows before the split, and each goes to the partition
of the label it carries.

**Hyperparameters.** $d$ (number of programs), $K$ (the number of subsample
rows per pair kept in training besides the full-data row) and the rank $r$ of
the shared noise behind $\mathbf V$. In the `free` model, $\alpha$
(reconstruction weight) and $\beta$ (negative weight on synthetic rows) are
tuned as well. $K$ picks a
nested random subset fixed per pair ($K = 1 \subset K = 2 \subset \dots$), and
thins training rows only, so every $K$ is scored on the same validation rows.

**Choosing the epoch.** Every model trains for the same fixed number of epochs,
and the kept state is the epoch with the highest **validation discrimination
accuracy** (each validation row under its own label and under a permuted one;
chance is 0.5). Validation cross-entropy is not used: it rises early, as the
head grows confident, even while validation accuracy keeps improving.

**Choosing a configuration.** Configurations are compared on validation only,
at each one's selected epoch: validation accuracy first, validation
reconstruction error on measured rows second. The total loss is not comparable
across configurations, because its $-\alpha\beta$ term moves with $\alpha\beta$
whatever the model does. Differences smaller than the spread across seeds are
not differences.

**The test read.** The chosen configuration's test metrics are read once, and
the read is logged.

## Evaluation

| Metric | Tests |
|---|---|
| held-out accuracy | Can the model tell an unseen $(p, c)$ from a permuted label? A lookup table scores 0.5. |
| real vs synthetic accuracy | Can it tell a real response from a marginal-matched fake under the same label? |
| reconstruction | Fraction of held-out signal $\mathbf z\mathbf B$ recovers, against PCA of the same rank fitted on training rows. |
| span residual $r$ | $\|\mathbf m \odot (\mathbf x - \mathbf x\mathbf B^{+}\mathbf B)\| / \|\mathbf m \odot \mathbf x\|$: do held-out rows need programs outside $\operatorname{span}(\mathbf B)$? |
| rank-matched PCA $r$ | The noise floor. Only a gap above it is evidence against $\mathbf B$. |
| cross-seed MCC | Hungarian matching on $\lvert\text{corr}\rvert$ of $\mathbf B$ across seeds, against shuffled and random nulls. Identified axes recur; arbitrary ones do not. |

The span test is the one result that could invalidate the architecture. A
global $\mathbf B$ allows non-additive interactions, but only as re-weightings
of the same $d$ programs. With a few contexts and thousands of perturbations,
contexts are the scarce resource for deciding which programs exist.

## Reading the factors

**Loadings.** Row $\mathbf B_k$ is program $k$, already in logFC units; no
back-transform is needed.

**Sign and scale.** Both are free per factor: $(-z_k)(-\mathbf B_k) =
z_k\mathbf B_k$. Signs are anchored by a reference perturbation whose direction
is known or, failing that, by making the largest-magnitude loading positive.
Gene lists for enrichment report the two tails separately.

**Activations.** $z_k$ is the change in program $k$ relative to unperturbed
cells of the same context: positive is up-regulation, negative is
down-regulation, and "does not regulate" is $|z_k| \approx 0$, not $z_k < 0$.

**Effect decomposition.**
$z_{pc} = \bar z + \mu_p + \gamma_c + \delta_{pc}$ separates the perturbation
main effect, the context main effect and the interaction. $\delta_{pc}$ is
epistasis on the log scale, identified by the multiple rows per pair.

**Responsiveness.** $a_{kc} = \operatorname{Var}_p[z_k(p, c)]$ says how
modulable program $k$ is in context $c$: large when the panel engages it, near
zero when the context has locked the program (with $\gamma_{kc}$ far from zero)
or when nothing engages it.

**Global versus per-context.** Refitting loadings within each context and
comparing them with the global $\mathbf B$ tests the single-$\mathbf B$
assumption directly.

## Configuration

Defaults of `TrainConfig`:

| Field | Default | Role |
|---|---|---|
| `subspace` | fixed | `fixed`: $\mathbf B = \mathbf A\mathbf V$, only $\mathbf A$ learned; `free`: $\mathbf B$ learned directly (the one-stage model) |
| `n_factors` | 24 | $d$ |
| `noise_rank` / `noise_max_rows` | 50 / 150,000 | shared-noise rank behind $\mathbf V$, and the subsample residuals used to fit it |
| `alpha` | 1.0 | reconstruction weight; `free` only |
| `recon_fake_weight` | 0.0 | $\beta$ for synthetic rows; `free` only |
| `basis` | lin, sq, abs, tanh | must keep a non-quadratic statistic |
| `embedding_dim` / `label_hidden` | 32 / 128 | label network width |
| `ridge` | 1e-4 | Gram matrix conditioning |
| `weight_scheme` | linear | $w \propto n$, capped at the 99th percentile |
| `balance_negatives` | True | fakes equal positives |
| `tc_weight` | 0.0 | independence penalty, off |
| `batch_size` / `epochs` / `lr` | 256 / 50 / 1e-3 | Adam, gradient clip 5.0 |
| `eval_every` | 5 | validation interval, in epochs |
| `select_on` | total | metric that picks the epoch; `accuracy` for the selection rule above |
| `patience` / `min_epochs` | 4 / 0 | early stopping; patience 0 trains every epoch |
| `unconstrained_head` | False | ablation: removes identifiability |
| `no_mask` | False | ablation: knockdown claims a factor |

## Module map

| Path | Contents |
|---|---|
| `extract/de/` | Welch statistics → `ashr` → merge → pivot; row planning; matrix assembly and `load_matrices` |
| `extract/data/` | `noise.py` noise model and label-driven subspace $\mathbf V$; `splits.py` test pairs, validation, `K` thinning and synthetic rows (including partially shuffled ones); `gene_filter.py`; `ontarget.py`; `augment.py` label permutation and column shuffle |
| `extract/models/` | `loadings.py` $\mathbf B$, $\mathbf B = \mathbf A\mathbf V$ and the projection; `heads.py`; `label_net.py` |
| `extract/objectives/` | `contrastive.py` sampler and BCE; `reconstruction.py`; `total_correlation.py` |
| `extract/train.py` | `TrainConfig`, `Extract`, `fit`, `evaluate`, `prepare`, `encode` |
| `extract/interpret/` | loadings, sign anchoring, effect decomposition, enrichment export |
| `extract/evaluation/` | held-out reconstruction (the same function for EXTRACT and every baseline), span residual, cross-seed stability |
| `baselines/` | PCA, linear ICA, context joint diagonalisation; `scripts/baseline_ica.py` scores ICA on the same split |
| `scripts/` | build the split, fit one model, run a sweep, summarise it, read test once |
