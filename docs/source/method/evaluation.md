# Evaluation

Unmixedness is not directly testable. The true sources are never observed, and
independence tests cannot stand in for it, since exactly independent
components can still be mixtures. What *is* testable is below.

## Discrimination accuracy on held-out pairs

The central number. Both the perturbation and the context of a held-out pair
were seen in training, never together, so a model that memorised training pairs
scores at chance (0.5) while a model that recovered structure supplying the
interaction scores above it. The train–test gap separates the two failure
directions: a large gap with high test accuracy is benign overfitting of the
head; test accuracy near 0.5 is the failure the architecture is designed to
avoid.

A second accuracy, real against synthetic rows each under its own label, asks
whether the model can tell a real response from a marginal-matched fake one.

## Reconstruction against the truth

On held-out rows, with the on-target entry excluded:

- the fraction of the rows' squared norm that $\mathbf z\mathbf B$
  reconstructs, globally, per row and per gene;
- the correlation between true and reconstructed entries, and the slope of
  reconstructed on true (projection onto a $d$-dimensional subspace shrinks
  magnitudes);
- for strongly moved entries ($|x| > \ln 1.2$), how often the reconstruction
  has the right sign and stays above the threshold.

Each is reported against **PCA of the same rank fitted on the training rows**,
the best a rank-$d$ linear reconstruction can do, so the numbers say how much
of the reachable signal $\mathbf B$ keeps.

## The span test

The decoder is linear, but linearity is not what limits interactions; a
non-additive $\delta_{pc}$ is fully expressible. What a global $\mathbf B$
forbids is narrower: **an interaction may only re-weight the same $d$
programs**. A combination that switches on a program appearing in no training
condition falls outside $\operatorname{span}(\mathbf B)$ and cannot be
represented. For a held-out row,

```{math}
r(\mathbf x) = \frac{\big\|\mathbf m \odot (\mathbf x - \mathbf x\mathbf B^{+}\mathbf B)\big\|}{\|\mathbf m \odot \mathbf x\|} .
```

The mask matters: $\mathbf B$ is trained never to reconstruct the on-target
entry, which holds a large share of a row's squared norm, so scoring against it
would charge $\mathbf B$ for mass it was told to ignore. Small $r$ means
re-weighting the existing programs is enough. A context whose rows carry large
$r$ introduced a program the training contexts never showed. That is the one
failure a global $\mathbf B$ cannot absorb, and the only outcome that would
invalidate the architecture rather than call for tuning. Leaving out one
context at a time makes it a direct measurement.

Most rows are subsampled estimates whose norm is largely estimation noise, and
noise is orthogonal to any $d$-dimensional subspace, so $r$ sits near 1 even for
a good $\mathbf B$. It is interpretable only against the rank-matched PCA
baseline, and per subsample stratum, since the noise floor falls as $n$ rises.
Only a gap above the baseline is evidence against the span.

Which programs can appear in $\mathbf B$ at all depends on the design. With one
context, it is the perturbation panel: a program no perturbation moves leaves
no trace. With several, contexts are the scarce resource: thousands of
perturbations pin down each program's loadings, but the handful of contexts are
all there is to establish which programs exist.

## Cross-seed stability

Identified axes should recur across independent fits; arbitrary ones should not.
Factors are recovered only up to permutation and sign, so fits are matched by
Hungarian assignment on $|\text{correlation}|$ between rows of $\mathbf B$ and
scored by the mean matched $|\text{correlation}|$ (MCC). Fits compared this way
must share one split, with only the initialisation seed varying.

A bare MCC is not interpretable: Hungarian matching over $d \times d$ picks
maxima, so even unrelated matrices score above zero. It is reported against two
nulls: $\mathbf B$ with its gene columns permuted (destroys which genes go
together, keeps every marginal), and Gaussian matrices of the same shape. The
verdict is the gap.
