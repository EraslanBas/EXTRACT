# Appendix

## Computing $\mathbf V$ in closed form

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
