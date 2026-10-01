# Why the head is constrained

## The mechanism

For the real-versus-permuted-label task with balanced classes, the
Bayes-optimal score is the log density ratio

```{math}
s^{*}(\mathbf x, u) = \log \frac{p(\mathbf x \mid u)}{p(\mathbf x)} .
```

Suppose the data are generated as $\mathbf x = f(\mathbf s)$ with $f$
injective and the sources independent given $u$,
$p(\mathbf s \mid u) = \prod_i p_i(s_i \mid u)$, and marginally independent.
Substituting, the Jacobian terms of the change of variables cancel between
numerator and denominator, leaving

```{math}
s^{*}(\mathbf x, u) = \sum_{i=1}^{d} \Big[\log p_i(s_i \mid u) - \log p_i(s_i)\Big].
```

The optimum is **already additively separable in the true source
coordinates**. Additive separability is not preserved under mixing: if
$\mathbf z = \mathbf M\mathbf s$ for a non-diagonal, non-permutation
$\mathbf M$, a function $\sum_k \psi_k(z_k, u)$ generically cannot equal the
expression above, because expanding $\psi_k(\sum_i M_{ki}s_i, u)$ produces
cross terms in $s_i s_{i'}$ that no sum of univariate functions can cancel. A
head of the form $\sum_k \psi_k(z_k, u)$ can therefore reach the optimum only
when its coordinates are the source coordinates, up to permutation and
element-wise transformation.

With an unrestricted head, an MLP over the whole feature vector, the
substitution $\mathbf z \mapsto \mathbf M\mathbf z$ is absorbed by its first
layer, the loss cannot see $\mathbf M$, and the axes float. The model keeps an
unconstrained head only as an ablation, to show the constraint matters; its
factors are not to be interpreted.

This is the exponential family of time-contrastive learning
([Hyvärinen & Morioka 2016](https://arxiv.org/abs/1605.06336)), with the $q_j$
as sufficient statistics and $\boldsymbol\lambda(u)$ as label-dependent natural
parameters.

## A worked reading

An illustrative row with $d = 4$ and $\mathbf z = (1.8, -0.2, 0.05, 2.4)$.
Under its true label, factors 1 and 4 are where $\boldsymbol\lambda$ expects
activity, and they contribute strong positive evidence. Under a permuted label
$\boldsymbol\lambda$ expects activity elsewhere, the same $\mathbf z$ now
looks wrong, and those terms reverse.

| | true $u$ | permuted $u^{*}$ |
|---|---|---|
| $\psi_1$ | +2.1 | −1.4 |
| $\psi_2$ | +0.3 | +0.2 |
| $\psi_3$ | −0.1 | 0.0 |
| $\psi_4$ | +1.7 | −2.2 |
| $s$ (the sum) | **+4.0** | **−3.4** |
| $\sigma(s)$ | 0.982 | 0.032 |

## One caveat on the basis

A purely quadratic head identifies nothing: $\sum_k z_k^2$ is invariant to
rotation, which is the same reason Gaussian sources are unidentifiable in linear
ICA. The basis must keep at least one non-quadratic statistic, which is why
$q$ includes $|z|$ and $\tanh z$ and not only $(z, z^2)$. The code refuses a
purely quadratic basis.

## What is not claimed

The identifiability theorem of generalised contrastive learning
([Hyvärinen et al. 2019](https://arxiv.org/abs/1805.08651)) requires
$p(\mathbf s \mid u)$ to vary with $u$ in a non-degenerate way. Here the spread
of rows within a fixed $(p, c)$ is subsampling noise governed by $n$, which is
exactly the variation the stratified negatives remove from the discriminator's
reach. The theorem does not transfer cleanly, and it is not invoked. What
survives is the structural argument: the head can only represent separable
functions, the true ratio is separable in the source coordinates, and mixing
destroys separability.

Because unmixedness cannot be observed directly, the practical evidence is
**reproducibility**: identified axes should recur across independent fits and
arbitrary ones should not ({doc}`evaluation`).
