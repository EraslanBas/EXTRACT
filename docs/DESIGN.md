# module_finder: design and assumptions

## The model

    x_{p,c}  =  z_{p,c} @ B  +  noise

* `B` [n_factors, n_genes] — **global** loading matrix, shared across every
  perturbation and context. Row `k` is factor `k`'s gene loading vector, and it
  is a model parameter rather than a post-hoc estimate.
* `z_{p,c}` — how perturbation `p` in context `c` activates each factor,
  decomposed by `interpret.effects` into

      z_{p,c} = grand_mean + mu_p + gamma_c + delta_pc

  where `delta_pc` (the interaction) is the scientific payload of a
  chemogenetic screen.

Requiring `B` to be global and consistent forces the decoder to be linear, which
in turn makes the encoder linear. **This model is therefore linear ICA with
auxiliary variables, not nonlinear ICA.** That is a simplification, not a loss:
identifiability becomes available through several weaker routes, and the
baselines in `src/baselines/` estimate the same model class much more cheaply.
Any nonlinearity lives in the `(p,c) -> z` map (`models.label_net`), which is
where the drug x perturbation interaction belongs.

## Why the per-component head

The optimal discriminator for the real-vs-shuffled-label task is
`log p(x|u) / p(x)`. Under the model this equals

    sum_i [ log p_i(s_i|u) - log p_i(s_i) ]

so the true answer is already additively separable in the true source
coordinates. Separability is not preserved under mixing, so a head restricted to
`sum_k psi_k(z_k, u)` can only reach the optimum if its coordinates *are* the
source coordinates. An unrestricted MLP head can absorb `z -> M z` in its first
layer, the loss cannot see `M`, and the axes float. `models.heads` enforces the
restriction; `UnconstrainedHead` exists to ablate it.

Caveat: a purely quadratic basis is rotation invariant (`sum_k z_k^2`), the same
reason Gaussian sources are unidentifiable in linear ICA. Keep `abs` or `tanh`.

The constraint binds **only** on how the label-derived coefficients meet the
features. `label_net` is fully expressive and carries all of the interaction.

## Two ingredients, both required

1. **The label must modulate the factors** (sufficient variability). A property
   of the data and design, not the code. Verified empirically: with one row per
   `(p,c)` the estimator reaches 0.96 discrimination accuracy but only ~0.80 MCC
   against the true loadings; with 5 and 20 replicates per pair MCC rises to
   0.93 and 0.94 while accuracy *falls* to 0.88 and 0.83. High accuracy with
   thin replication is memorisation, not learning. (Synthetic data, 40
   perturbations x 6 contexts x 5 factors x 100 genes.)
2. **The per-component head** (above).

## Known shortcuts to defend against

| Shortcut | Magnitude in the ChemoGenetic matrices | Defence |
|---|---|---|
| Per-gene scale heterogeneity | per-gene SD spans ~5.4e4x | `data.rowbound.standardize_genes` |
| Per-perturbation effect size | per-perturbation SD spans ~341x | hard negatives; row weighting |
| Cell-count leak via noise scale | counts vary several-fold per pair | fixed `n_cells` in `data.pseudoreplicates` |
| Shuffling values *within* a vector | destroys gene marginals | never do it; permute whole rows only |

Measured on a 2,292-perturbation x first-1,000-gene slice of
`PosteriorMean_matrix_Stattic.csv`; participation-ratio effective rank of that
slice is 16.0, which is why `n_factors` should be ~20-30, not 100+.

## What is identified, and what is not

Recovery is up to **permutation and sign** (and, in the nonlinear reading, an
element-wise transformation). So factor ordering is arbitrary across runs, sign
and scale carry no meaning until anchored, and magnitudes are not comparable
across factors. What *is* pinned down is the partition: which genes and which
perturbations belong to each factor.

Unmixedness cannot be tested directly, and **cannot be tested by testing
independence** — Darmois' construction yields exactly-independent components
that are still mixtures of the truth. The available evidence is:

* cross-seed reproducibility (`evaluation.stability`),
* held-out `(p,c)` reconstruction (`evaluation.heldout`),
* beating `baselines.linear_ica` and `baselines.context_jd`, which estimate the
  same model class cheaply. If they win, that is a real result about the data.

## Assumption being bought

A global linear `B` assumes factor effects **add** in LFC space and that a
factor means the same thing in DMSO as under Romidepsin. Defensible — LFCs are
already differential — but testable: hold out `(p,c)` combinations and look for
systematic per-context residual structure, or compare
`interpret.loadings.loadings_per_context` via `loading_agreement`.
