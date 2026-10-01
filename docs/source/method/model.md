# The model

```{list-table} Notation and shapes. Only B, the label network and the head bias are learned; B⁺, z and λ are computed.
:header-rows: 1

* - symbol
  - shape
  - meaning
  - status
* - $\mathbf x$
  - $G$
  - shrunken logFC for one (perturbation, context, subsample) row
  - data
* - $n$
  - 1
  - cells behind the row
  - data
* - $u$
  - $p$ or $(p, c)$
  - the row's label
  - data
* - $s$
  - $0 \ldots 10$
  - subsample stratum (0 = full-data row)
  - data
* - $\mathbf B$
  - $d \times G$
  - global factor → gene map
  - **learned**
* - $\mathbf B^{+}$
  - $G \times d$
  - $\mathbf B^\top(\mathbf B\mathbf B^\top)^{-1}$
  - derived
* - $\mathbf z$
  - $d$
  - factor activations
  - derived
* - $e_p,\ e_c$
  - $32$ each
  - perturbation and context embeddings
  - **learned**
* - $\boldsymbol\lambda(u)$
  - $d \times J$
  - label-derived head coefficients, $J = 4$
  - derived
* - $b$
  - 1
  - head bias
  - **learned**
```

## One factor → gene map

The only interpretable output is $\mathbf B \in \mathbb R^{d \times G}$, whose
row $\mathbf B_k$ is factor $k$'s gene loading vector: a **gene program**.
$\mathbf B$ does not depend on $p$ or $c$; a factor means the same thing in
every context. That global commitment is the modelling assumption the design
buys, and the {doc}`span test <evaluation>` makes it testable.

Factor activations are **not** produced by a learned encoder. They are the
least-squares coordinates of $\mathbf x$ in $\mathbf B$'s row space,

```{math}
\mathbf z(\mathbf x) = \arg\min_{\mathbf z} \|\mathbf x - \mathbf z\mathbf B\|^2
= \mathbf x\,\mathbf B^{+},
\qquad \mathbf B^{+} = \mathbf B^\top(\mathbf B\mathbf B^\top)^{-1},
```

so the projection is *derived* from $\mathbf B$ rather than fitted alongside
it. With a separate encoder there would be two maps between factors and genes,
free to disagree about what factor $k$ is, and only one of them would be a
loading vector: encoder weights are *filters*, not *patterns*, and a filter can
put large weight on a gene precisely to cancel it as a nuisance
([Haufe et al. 2014](https://doi.org/10.1016/j.neuroimage.2013.10.067)). With
one map, $\mathbf B$ is a pattern by construction. $\mathbf x$ is never on a
generative path: it is projected so the discriminator can judge it, and it is
the reconstruction target.

## The masked projection

With the on-target mask $\mathbf m_i$ (zero at row $i$'s own gene, one
elsewhere), the projection is a masked least-squares problem,

```{math}
\mathbf z_i = \arg\min_{\mathbf z} \big\|\mathbf m_i \odot (\mathbf x_i - \mathbf z\mathbf B)\big\|^2 .
```

Because exactly one entry is masked, this is a rank-one downdate of one shared
Gram matrix. With $\mathbf G = \mathbf B\mathbf B^\top + \varepsilon\mathbf I$
and $\mathbf b_g$ the $g$-th column of $\mathbf B$,

```{math}
\mathbf z_i = \big(\mathbf G - \mathbf b_{g_i}\mathbf b_{g_i}^\top\big)^{-1}\mathbf B\,\tilde{\mathbf x}_i,
\qquad \tilde{\mathbf x}_i = \mathbf m_i \odot \mathbf x_i,
```

and by the Sherman–Morrison identity

```{math}
\big(\mathbf G - \mathbf b\mathbf b^\top\big)^{-1}
= \mathbf G^{-1} + \frac{\mathbf G^{-1}\mathbf b\,\mathbf b^\top\mathbf G^{-1}}{1 - \mathbf b^\top\mathbf G^{-1}\mathbf b},
```

so one $d \times d$ inverse per step serves every row, and each row costs
$O(d^2)$ however many distinct genes a batch masks. Rows whose denominator
falls below a tolerance are re-solved explicitly. With no mask the projection
reduces exactly to $\mathbf x\mathbf B^{+}$.

## The per-component head

The discriminator scores whether a row is paired with its true label. With
scalar statistics $q_1, \dots, q_J$ applied coordinate-wise and label-derived
coefficients $\boldsymbol\lambda(u) \in \mathbb R^{d \times J}$,

```{math}
\psi_k(z_k, u) = \sum_{j=1}^{J} \lambda_{kj}(u)\, q_j(z_k),
\qquad
s(\mathbf x, u) = \sum_{k=1}^{d} \psi_k(z_k, u) + b,
\qquad
\Pr(\text{real} \mid \mathbf x, u) = \sigma\big(s(\mathbf x, u)\big),
```

with $J = 4$ and $q = (z,\ z^2,\ |z|,\ \tanh z)$. There is one term per factor,
and **no term multiplies $z_k$ by $z_{k'}$ for $k \neq k'$**. That restriction
is the identifiability mechanism ({doc}`identifiability`).

Two properties are easy to misread:

- **Each $\psi_k$ is evidence, not a vote.** The terms are summed and one
  sigmoid turns the total into one real/fake decision. No per-factor decision
  is ever taken.
- **$\psi_k$ does not test whether $z_k \neq 0$.** It asks whether the observed
  $z_k$ looks like a draw from the distribution this label implies. With
  $q = (z, z^2)$, $\psi_k = a_k(u)z_k + b_k(u)z_k^2$ is a Gaussian log-density
  whose mean and spread depend on $u$. Adding the terms is evidence
  accumulation over factors: log-odds add.

The head itself learns only the bias $b$. All flexibility sits in
$\boldsymbol\lambda(u)$, and in $\mathbf z$ through $\mathbf B$. That rigidity
is the constraint.

## The label network

$\boldsymbol\lambda$ is deliberately unconstrained. The restriction above
governs only *how* $\boldsymbol\lambda$ meets the statistics, never how
$\boldsymbol\lambda$ is computed, so all of the perturbation × context
interaction lives here:

```{math}
\boldsymbol\lambda(u) = \mathrm{MLP}\big([\,e_p\,;\,e_c\,]\big) \in \mathbb R^{d \times J},
```

with $e_p, e_c \in \mathbb R^{32}$ and one hidden layer of width 128. With one
context the map is $\boldsymbol\lambda(p) = \mathrm{MLP}(e_p)$.

The factorisation into $e_p$ and $e_c$, rather than one free row per observed
$(p, c)$ pair, is what separates learning from memorisation. A free lookup
table has one row per pair, and with a single observation per pair the optimal
discriminator is "answer real iff $\mathbf x$ equals the vector stored for
$u$", a hash table that needs no notion of components. Sharing $e_p$ across
contexts and $e_c$ across perturbations reduces the parameter count from
$O(n_{\text{pairs}} \cdot d)$ to $O((n_p + n_c) \cdot d)$ and makes that
solution unreachable. It is also why a held-out pair, whose perturbation and
context were both seen in training but never together, can be scored at all.

## Deliberately absent

- **A learned encoder.** Replaced by $\mathbf x\mathbf B^{+}$: one
  factor → gene map instead of two that can disagree.
- **A label prior $f(e_p, e_c) \to \mathbf z$.** It exists only to predict a
  response without seeing $\mathbf x$. Prediction is not the objective; the
  objective is to recover the factors that generated the data.
- **Control-cell covariance as an input.** At $G^2$ entries it is badly
  conditioned, and its dominant axes are cell cycle and technical variation
  rather than regulation. Co-response to perturbation is stronger evidence
  about regulation than co-expression in controls.
