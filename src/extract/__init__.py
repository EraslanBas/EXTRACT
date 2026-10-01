"""extract -- identifiable gene programs from perturbation x context screens.

The model, in one line:

    x  ~  z B ,      z = argmin_z || m * (x - z B) ||^2

* ``B`` [n_factors, n_genes] is a **global** loading matrix, shared across every
  perturbation and every context, and it is the only interpretable output. Row
  ``k`` is factor ``k``'s gene loading vector.
* ``z`` is the masked least-squares projection of ``x`` onto ``B``'s row space
  -- *derived* from ``B``, not produced by a separate encoder. One
  factor->gene map, so no two modules can disagree about what factor ``k``
  means. ``m`` masks the perturbation's own transcript.

Fitting combines two terms (``objectives/``):

1. **L_disc** -- a contrastive term discriminating real ``(x, u)`` pairs from
   pairs carrying a permuted label ``u = (perturbation, context)``, scored by a
   head restricted to a *sum over components*, ``sum_k psi_k(z_k, u)``, with no
   cross-component terms (``models.heads``). This **picks the rotation**:
   additive separability is not preserved under mixing, so the sum-form optimum
   is reachable only in unmixed coordinates.
2. **L_recon** -- a precision-weighted reconstruction term. This **picks the
   subspace**: a tied linear autoencoder on reconstruction alone recovers the
   principal subspace and leaves orientation free (Baldi & Hornik, 1989).

    L = L_disc + alpha * L_recon

Both are needed. Drop the discriminator and this is PCA with extra steps; drop
the reconstruction term and the contrastive objective selects tiny-variance
directions that discriminate well but carry no gene program.

The objective is to recover the factors that *generated* the data, not to
predict responses of unseen perturbations.

Full specification, with all mathematical and architectural detail:
the method paper.
"""

__version__ = "0.1.0.dev0"

__all__ = [
    "data",
    "models",
    "objectives",
    "interpret",
    "evaluation",
    "train",
    "paths",
]
