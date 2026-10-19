# Current EXTRACT configuration (defaults on `main`, 2026-10-19)

Chosen from the 90-model comparison on the no-MT data (`sweep_jdstats_d20_noMT/`). Running
`scripts/sweep_grid.py` with no model flags reproduces it exactly.

| Part | Setting | Flag / config field |
|---|---|---|
| Data | disjoint split, full row + 100-cell pseudobulks, pairs < 100 cells removed, mitochondrial genes removed (5,504 genes); no gene centring or scaling | `--split-dir .../pair_seed0_ctxshuffle_pgenes_disjoint_min100_noMT` |
| Programs | d = 20 | `--d 20` / `n_factors` |
| Subspace | fixed V = PCA of the training pair means; B = A V, A learned | `--subspace fixed --subspace-method pca` |
| Starting axes | joint diagonalisation of the per-context covariances of the pair means | `--axes-init jd` / `axes_init` |
| Activities | group lasso per (perturbation, context) pair, lambda = 0.0272, then least-squares refit on the selected programs | `--coding group_lasso --lasso-lambda 0.0272` |
| Head | four statistics (z, z^2, \|z\|, tanh z) with a learned label-only normaliser c_k(p, c) | `--head statistics --head-normaliser` |
| Discriminator | binary; false examples = label swaps only, one per measured row (no shuffled data) | `--shuffled-frac 0` |
| Label network | MLP on [perturbation 64, context 8] embeddings, dropout 0.2, weight decay 0.01 | `--label-dropout 0.2 --label-weight-decay 0.01` |
| Embedding starts | perturbations: target-gene responses in the training pairs; contexts: PCA of control-cell means | `--pert-embedding-init response --context-embedding-init control-means` |
| Loss | L_disc + alpha * L_lasso, alpha = 700 | `--alpha 700` |
| Training | 1000 epochs, batch 1,024, evaluated every 3 epochs, no early stopping, best epoch by validation ROC AUC; 3 seeds | `--epochs 1000 --select-on auc` |
| Saved per model | best and final weights, config, label levels, history (every loss term), validation logits | |

Not part of the defaults yet: the gene-axis prior (L1/L2 sparseness + collinearity hinge; branch
`gene-prior-l1`, untested). Old variants (least squares, distance head, shuffled negatives, InfoNCE on
its own branch) remain available through flags. See `group_lasso/group_lasso.pdf` for the group lasso.

Open next: gene-axis sparsity, random starts per seed, training on halves of the perturbations.
