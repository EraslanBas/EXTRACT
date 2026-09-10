"""module_finder -- identifiable latent factors from perturbation x context screens.

The model, in one line:

    x_{p,c}  =  z_{p,c} @ B  +  noise

* ``B`` [n_factors, n_genes] is a **global** loading matrix, shared across every
  perturbation and every context. Row ``k`` is factor ``k``'s gene loading
  vector, and it is a model parameter rather than a post-hoc estimate.
* ``z_{p,c}`` is how perturbation ``p`` in context ``c`` activates each factor,
  decomposable into ``mu_p + gamma_c + delta_pc`` (``interpret.effects``).

Fitting combines two objectives (``objectives/``):

1. a **contrastive** term discriminating real ``(x, u)`` pairs from pairs with a
   shuffled ``u = (perturbation, context)`` label, scored by a head that is a
   *sum over components* with no cross-component terms (``models.heads``);
2. a **reconstruction** term anchoring the factors to directions that explain
   real variance.

Both are needed. The contrastive term supplies the identifiability structure;
the reconstruction term stops it from selecting tiny-variance directions that
discriminate well but carry no interpretable gene program.

See ``docs/DESIGN.md`` for the derivation and the assumptions being bought.
"""

__version__ = "0.1.0.dev0"

__all__ = ["data", "models", "objectives", "interpret", "evaluation"]
