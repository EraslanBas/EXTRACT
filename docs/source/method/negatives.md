# Negative sampling

## Stratify, or precision gives the answer away

Naive label permutation is solved by precision rather than biology. The noise
scale of $\mathbf x$ encodes the cell count $n$, and $n$ is tied to perturbation
identity, so a discriminator can separate real from permuted pairs by reading
off effect-size scale and never looking at structure. Negatives are therefore
drawn **within a subsample stratum** $s$, which holds precision roughly fixed
across the swap. On the example screen, stratification cuts the median
$|\log_2(n_{\text{real}}/n_{\text{neg}})|$ from 1.75 to 0.62.

## Label-permuted negatives

Each measured row in a batch is paired with a permuted label, drawn by one of
two strategies with equal probability:

| strategy | example | forces the model to learn |
|---|---|---|
| same stratum, **other perturbation** | `A1BG__sub03` → `BRCA1__sub03`, same context | perturbation identity |
| same stratum, same perturbation, **other context** | `A1BG__sub03` under a different drug | the perturbation × context interaction |

The second strategy is unavailable with a single context; there, the first
alone is sufficient, since the question becomes "does this response belong to
*this* perturbation", whose optimum is still separable in the sources. Contexts
are symmetric: no strategy assumes a reference arm (vehicle, untreated).

**Every permuted label is an observed pair of the sampler's own rows.** The
sampler is built on the training rows alone, so a validation or test pair is
never named, not even as a negative. If one axis has no observed alternative,
the other axis is swapped; a label with no observed alternative on either axis
is an error rather than a silent leak.

## Synthetic negatives

Every synthetic row is a negative by construction, under its own label, so it
needs no sampler. It is the only negative that does not depend on an
alternative label existing.

Shuffling must be **across rows, within a gene**, never across genes within a
row. Per-gene standard deviation spans more than four orders of magnitude, so a
value landing in the wrong gene column is detectable by magnitude alone, and
the discriminator would win on scale. Measured separability between real and
column-shuffled rows is Cohen's $d = 0.00$, against $0.59$ for within-row
shuffling.

## Balancing

Fakes are balanced against real rows: label-permuted negatives are thinned so
that

```{math}
|\text{permuted}| + |\text{synthetic}| = |\text{positives}| .
```

The thinning probability comes from the global counts, not the batch, so it is
stable from batch to batch. Since the synthetic rows are a fixed fraction of
the loaded data (one per two measured rows by default), this also fixes the
share of synthetic rows among the negatives without a separate weight.
Mean-reduced binary cross-entropy already gives the positive and negative terms
equal weight whatever their counts; balancing the counts also makes the mix
*between* the two kinds of negative a property of the data loaded rather than
an accident of how many synthetic rows exist.
