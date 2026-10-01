# Glossary

$\mathbf x$
: One row: the shrunken log-fold-change vector of one (perturbation, context,
  subsample) estimate against controls from the same context. Used unscaled.

pair
: One (perturbation, context) combination. All of its rows move together in
  every split.

full-data row, subsample row
: A pair's estimate from all its cells, and its re-estimates from random
  subsets of them (stratum $s = 1 \ldots 10$; the full-data row is $s = 0$).

$\mathbf B$
: The global loading matrix, $d \times G$. Row $k$ is factor $k$'s gene
  program. The only interpretable output.

$\mathbf z$
: Factor activations of a row: its masked least-squares projection onto
  $\mathbf B$. Derived, not learned.

on-target mask $\mathbf m$
: Zero at the perturbed gene's own transcript, one elsewhere. Removes knockdown
  efficiency from projection and reconstruction.

$\psi_k$, head
: The per-component evidence of factor $k$ for a (row, label) pairing; the head
  sums them into one score.

$\boldsymbol\lambda(u)$, label network
: Maps the label's embeddings $e_p$, $e_c$ to the head's per-factor
  coefficients. Carries all the perturbation × context interaction.

positive, negative
: A measured row under its own label (positive); a measured row under a
  permuted label, or a synthetic row under its own label (negatives).

synthetic row
: A column-shuffled copy of a measured row within one context: per-gene
  marginals kept, gene–gene structure destroyed. Keeps its twin's label.

stratum
: A row's subsample index. Negatives are drawn within a stratum so precision
  cannot give them away.

$\alpha$, $\beta$, $d$, $K$
: Reconstruction weight; weight of the negative reconstruction term on
  synthetic rows; number of factors; subsample rows per pair kept in training.

$\mu_p$, $\gamma_c$, $\delta_{pc}$
: Perturbation main effect, context main effect, and interaction in the
  decomposition $z_{pc} = \bar z + \mu_p + \gamma_c + \delta_{pc}$.

span residual $r$
: The part of a held-out row that no re-weighting of the $d$ programs can
  reconstruct, relative to the row's norm.

MCC
: Mean matched absolute correlation between the factors of two fits after
  Hungarian matching; the cross-seed stability score.
