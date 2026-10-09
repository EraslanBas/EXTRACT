"""EXTRACT: the model and its training loop.

    L = L_disc + alpha * L_recon   [ + tc_weight * L_tc ]

``L_disc`` picks the rotation, ``L_recon`` picks the subspace. Implements
the method paper exactly; equation numbers in comments refer to
it.
"""

from __future__ import annotations

from typing import Callable

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data.ontarget import on_target_index
from .models import (initial_axes, AMMILabelNet, FactorizedLabelNet, ProductLabelNet, FixedBasisLoadings, GlobalLoadings,
                     PerComponentHead, UnconstrainedHead, DistanceHead)
from .models.heads import DEFAULT_BASIS
from .models.loadings import NO_MASK
from .objectives import (
    DEFAULT_WEIGHTS,
    StratifiedNegativeSampler,
    contrastive_loss,
    discrimination_accuracy,
    weighted_squared_error,
)
from .objectives.total_correlation import (
    TotalCorrelationDiscriminator,
    shuffle_coordinates,
    tc_discriminator_loss,
    tc_penalty,
)


@dataclass
class TrainConfig:
    #: Keep near the effective rank of the data. Participation-ratio effective
    #: rank on these matrices is 16.0, so 20-30 is the sensible range.
    n_factors: int = 24

    #: Must retain a non-quadratic statistic: a purely quadratic head is
    #: rotation invariant and identifies nothing.
    basis: tuple[str, ...] = DEFAULT_BASIS
    embedding_dim: int = 32
    label_hidden: int = 128
    ridge: float = 1e-4

    #: The reconstruction weight of eq. (12). With ``d``, the only real knob.
    #: Too low and the contrastive term selects tiny-variance directions that
    #: discriminate well but carry no interpretable gene program; too high and
    #: the model collapses to PCA and stops responding to the label.
    alpha: float = 1.0

    #: Thin the label-permuted negatives so that
    #: ``|permuted| + |synthetic| == |positives|``, i.e. one negative per
    #: positive overall. Without it every real row emits a permuted negative
    #: AND every synthetic row is a negative, so negatives outnumber positives.
    #: Mean-reduced BCE already equalises the two *terms*, but balancing the
    #: counts also makes the synthetic share of the negatives equal the
    #: fraction of synthetic rows loaded -- so the mixture is set by
    #: ``--shuffled-frac`` and needs no separate weight. No effect without
    #: synthetic rows.
    balance_negatives: bool = True

    #: Weight on the reconstruction error of SYNTHETIC rows (``is_real=False``),
    #: entered with a MINUS sign. ``0.0`` drops them from ``L_recon`` entirely;
    #: ``> 0`` actively pushes ``B`` away from spanning them, which is
    #: contrastive PCA against a marginal-matched background. Either way ``B``
    #: is fitted only to genuinely observed perturbation responses.
    recon_fake_weight: float = 0.0

    #: Off by default. Independence is not identifiability -- see
    #: objectives.total_correlation.
    tc_weight: float = 0.0
    tc_hidden: int = 256
    tc_lr: float = 1e-4

    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 5.0
    seed: int = 0
    device: str = "cpu"

    #: How the span of B is set.
    #:
    #: ``"fixed"`` (the model): B = A V. ``V`` [d, G] is the label-driven
    #: subspace -- the d directions with the most perturbation-driven variation
    #: per unit of estimation noise, computed once from the training rows
    #: (:mod:`extract.data.noise`) -- and only the d x d matrix ``A`` is learned,
    #: so the discriminator chooses oblique axes inside span(V). The span, and
    #: with it the reconstruction, cannot move, so ``alpha`` and
    #: ``recon_fake_weight`` have no effect.
    #:
    #: ``"free"``: B learned directly by both terms (the earlier model; an
    #: ablation). Experimental: ``"anchored"`` (B initialised at V,
    #: reconstruction in the correlated-noise metric) and ``"frozen"`` (B = V
    #: exactly, a probe of a given set of axes).
    subspace: str = "fixed"

    #: How V is computed (fixed mode). ``"pca"``: PCA of the pair means (every
    #: row of a pair averaged; uncentred). ``"rca"``: reliable components analysis
    #: -- the directions whose values agree most between different rows of the
    #: same pair, relative to their total variance (Dmochowski et al.); R_W
    #: truncated to its top ``rca_rank`` eigen-directions. ``"snr"``: between-
    #: pair signal over the within-pair noise model (the previous method; also
    #: what ``"anchored"`` uses). See extract.data.noise.
    subspace_method: str = "rca"
    rca_rank: int = 100

    #: Free mode only. ``"random"``: B starts at random (the earlier free
    #: model). ``"basis"``: B starts where the fixed model starts, ``A0 V``
    #: (``A0`` from ``axes_init`` and the seed, V from ``subspace_method``), and
    #: is then learned freely -- the same start as the fixed model, without the
    #: constraint.
    free_init: str = "random"

    #: Rank of the correlated part of the noise model (``"snr"`` and
    #: ``"anchored"``), and the number of within-pair residuals used to fit it.
    noise_rank: int = 50
    noise_max_rows: int = 150_000

    #: Starting axes A in the fixed model: ``"identity"`` (B starts at V) or
    #: ``"random"`` (a random rotation of V's axes, drawn from ``seed``).
    axes_init: str = "identity"

    #: How labels map to the head's coefficients lambda. ``"mlp"``: embeddings
    #: of p and c through an MLP, unconstrained (does not fix the axes).
    #: ``"product"``: lambda_k(p, c) = f_k(p) * g_k(c), a CP structure that
    #: does. ``"ammi"``: lambda_k(p, c) = a_k(p) + b_k(c) + f_k(p) * g_k(c),
    #: main effects plus that interaction; the axes are fixed by the
    #: interaction alone. See models.label_net.
    label_model: str = "mlp"

    #: Discriminator head. ``"statistics"``: sum_k sum_j lambda_kj(u) q_j(z_k)
    #: over the statistics in ``basis``. ``"distance"``: b - sum_k (z_k -
    #: zhat_k(u))^2 / (2 sigma_k^2), with the label model predicting the
    #: activities zhat (``basis`` is then unused). See models.heads.
    head: str = "statistics"

    #: Regularisation of the label network against memorising training pairs:
    #: decoupled (AdamW) weight decay on its parameters only, and dropout on
    #: its embeddings and hidden layer (MLP label model). 0 = off.
    label_weight_decay: float = 0.0
    label_dropout: float = 0.0

    #: Starting perturbation embeddings (MLP label model). ``"random"``.
    #: ``"response"``: from how each perturbation's target gene responds, as a
    #: response gene, across the training pairs -- see
    #: :func:`response_embeddings`. Perturbations whose target is not a
    #: response gene keep a random start. Trained further either way.
    pert_embedding_init: str = "random"

    #: Weight rho of the activity-sparsity penalty,
    #: rho * sum_k mean_rows |z_k| / sd(z_k), over the batch's measured
    #: rows. Each perturbation should engage few programs; mixing the
    #: axes spreads every response over many factors, so the penalty prefers
    #: the true axes. Dividing by each factor's spread (kept in the graph)
    #: stops A from lowering it by rescaling a factor. 0 = off.
    sparsity: float = 0.0

    #: Ablation only: replaces the per-component head with an MLP, which
    #: destroys identifiability. See models.heads.UnconstrainedHead.
    unconstrained_head: bool = False

    #: Ablation only: skips the on-target mask, so knockdown efficiency is free
    #: to claim a factor.
    no_mask: bool = False

    #: Evaluate on the validation rows every N epochs (0 = never).
    eval_every: int = 5

    #: Stop when validation accuracy has not improved for this many
    #: evaluations, and restore the best-scoring parameters. Set to 0 to run
    #: all epochs. This is the reason a validation set exists: on the real
    #: matrices held-out accuracy peaks near epoch 20 and then declines while
    #: training accuracy keeps climbing, so the last epoch is not the model you
    #: want -- and choosing the epoch on the *test* set would make its numbers
    #: no longer held out.
    patience: int = 4

    #: Never stop before this many epochs, however early validation bottomed
    #: out. Early stopping still RESTORES the best epoch -- this only decides
    #: when training stops, not which state is kept. Keeping it running past
    #: the onset records the full shape of the train and val curves, which is
    #: the evidence that the onset is an onset rather than noise.
    min_epochs: int = 0

    #: Validation metric to select on. "accuracy" maximises; "total", "disc"
    #: and "recon" minimise. "total" is eq. (12) itself, so selecting on it
    #: stops at the overfitting onset -- the epoch where validation turns up
    #: while training keeps falling. "accuracy" and "total" can disagree
    #: sharply: BCE worsens as the head grows confident even while the ranking
    #: it induces still improves.
    select_on: str = "total"

    negative_weights: dict[str, float] | None = None
    log_every: int = 10


class Extract(nn.Module):
    """One global loading matrix, a per-component head, and a label network.

    There is no encoder and no label prior. ``B`` is the only factor->gene map;
    ``z`` is the masked least-squares projection of ``x`` onto its row space.
    """

    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        n_contexts: int,
        config: TrainConfig,
        subspace_basis: np.ndarray | None = None,
        noise_model=None,
    ):
        super().__init__()
        self.config = config
        if config.subspace not in ("free", "fixed", "anchored", "frozen"):
            raise ValueError(f"unknown subspace mode {config.subspace!r}")
        if config.subspace != "free" and subspace_basis is None:
            raise ValueError(f"subspace={config.subspace!r} needs subspace_basis")
        if config.subspace == "free" and config.free_init == "random":
            subspace_basis = None
        if subspace_basis is not None and np.asarray(subspace_basis).shape != (config.n_factors, n_genes):
            raise ValueError(f"subspace_basis must be [{config.n_factors}, {n_genes}]")
        if config.subspace in ("fixed", "frozen"):
            self.loadings = FixedBasisLoadings(subspace_basis, ridge=config.ridge,
                                               learn_axes=config.subspace == "fixed",
                                               axes_init=config.axes_init, seed=config.seed)
        else:
            self.loadings = GlobalLoadings(config.n_factors, n_genes, ridge=config.ridge)
            if config.subspace == "free" and config.free_init == "basis":
                if subspace_basis is None:
                    raise ValueError("free_init='basis' needs subspace_basis")
                A0 = initial_axes(config.n_factors, config.axes_init, config.seed)
                with torch.no_grad():
                    self.loadings.B.copy_(A0 @ torch.as_tensor(np.asarray(subspace_basis), dtype=torch.float32))
            elif config.subspace == "free" and config.free_init != "random":
                raise ValueError(f"free_init must be 'random' or 'basis', got {config.free_init!r}")
            if config.subspace == "anchored":
                if noise_model is None:
                    raise ValueError("subspace='anchored' needs noise_model")
                with torch.no_grad():
                    self.loadings.B.copy_(torch.as_tensor(np.asarray(subspace_basis), dtype=torch.float32))
                self.loadings.set_noise_metric(noise_model.U, noise_model.s, noise_model.c)
        if config.head not in ("statistics", "distance"):
            raise ValueError(f"head must be 'statistics' or 'distance', got {config.head!r}")
        if config.head == "distance":
            if config.unconstrained_head:
                raise ValueError("unconstrained_head is an ablation of the statistics head")
            self.head = DistanceHead(config.n_factors)
        else:
            self.head = (
                UnconstrainedHead(config.n_factors, len(config.basis))
                if config.unconstrained_head
                else PerComponentHead(config.basis)
            )
        n_basis = self.head.n_basis if config.head == "distance" else len(config.basis)
        if config.label_model in ("product", "ammi"):
            net = ProductLabelNet if config.label_model == "product" else AMMILabelNet
            self.label_net = net(n_perturbations, n_contexts,
                                             config.n_factors, n_basis)
        elif config.label_model == "mlp":
            self.label_net = FactorizedLabelNet(
                n_perturbations=n_perturbations,
                n_contexts=n_contexts,
                n_factors=config.n_factors,
                n_basis=n_basis,
                embedding_dim=config.embedding_dim,
                hidden=config.label_hidden,
                dropout=config.label_dropout,
            )
        else:
            raise ValueError(f"label_model must be 'mlp', 'product' or 'ammi', got {config.label_model!r}")

    # ---- forward pieces ---------------------------------------------------

    def project(
        self, x: torch.Tensor, target_col: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``z`` -- eq. (2)/(4). Masked when ``target_col`` is given."""
        return self.loadings.project(x, target_col)

    def score(
        self,
        z: torch.Tensor,
        perturbation_idx: torch.Tensor,
        context_idx: torch.Tensor,
    ) -> torch.Tensor:
        """``s(x, u) = sum_k psi_k(z_k, u) + b`` -- eq. (7)."""
        return self.head(z, self.label_net(perturbation_idx, context_idx))

    def component_evidence(
        self,
        z: torch.Tensor,
        perturbation_idx: torch.Tensor,
        context_idx: torch.Tensor,
    ) -> torch.Tensor:
        """``psi_k`` per factor -- [batch, n_factors]. The per-factor evidence
        of table 2, not a per-factor decision: these are summed before any
        decision is taken."""
        if isinstance(self.head, UnconstrainedHead):
            raise TypeError("the unconstrained head has no per-component terms")
        lam = self.label_net(perturbation_idx, context_idx)
        if isinstance(self.head, DistanceHead):
            return self.head.evidence(z, lam)
        return (lam * self.head.statistics(z)).sum(dim=-1)

    @property
    def B(self) -> torch.Tensor:
        return self.loadings.B

    @torch.no_grad()
    def loading_matrix(self) -> np.ndarray:
        """``B`` as [n_factors, n_genes]. The interpretable output -- see
        ``interpret.loadings`` for sign anchoring, which is required before any
        directional claim."""
        return self.loadings.B.detach().cpu().numpy()


def response_embeddings(
    X: np.ndarray, perturbation_idx: np.ndarray, context_idx: np.ndarray,
    target_col: np.ndarray, rows: np.ndarray, n_perturbations: int, dim: int,
    is_real: np.ndarray | None = None, seed: int = 0, scale: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturbation embeddings from how each target gene responds.

    Transpose the training data: one row per target gene, one column per
    training pair, holding that gene's response (the pair's mean over its
    measured rows) in that pair. Entries where the pair's own perturbation
    targets the gene are left out (set to 0) -- the knockdown would dominate.
    A truncated SVD of these rows gives ``dim`` numbers per target gene,
    rescaled so the entries have standard deviation ``scale`` (the size of the
    random start). Uses training pairs only, so validation and test pairs never
    shape an embedding. Returns ``(embeddings [n_perturbations, dim], has [n])``;
    ``has`` is False for perturbations whose target is not a response gene.
    """
    from sklearn.utils.extmath import randomized_svd
    from .data.noise import pair_means
    rows = np.nonzero(rows)[0] if np.asarray(rows).dtype == bool else np.asarray(rows)
    pair = perturbation_idx * (int(context_idx.max()) + 1) + context_idx
    keys, means, _ = pair_means(X, pair, rows, is_real)
    pair_pert = keys // (int(context_idx.max()) + 1)
    col_of = np.full(n_perturbations, NO_MASK, dtype=np.int64)
    ok = target_col != NO_MASK
    col_of[perturbation_idx[ok]] = target_col[ok]
    has = col_of != NO_MASK
    T = means[:, col_of[has]].T.copy()                       # [targets, training pairs]
    T[pair_pert[None, :] == np.nonzero(has)[0][:, None]] = 0.0   # own knockdown entries
    U, S, _ = randomized_svd(T.astype(np.float32), min(dim, *T.shape), n_iter=7, random_state=seed)
    E = np.zeros((n_perturbations, dim), dtype=np.float32)
    Z = U * S
    E[has, :Z.shape[1]] = (Z / Z.std() * scale).astype(np.float32)
    return E, has


def fit(
    X: np.ndarray,
    perturbation_idx: np.ndarray,
    context_idx: np.ndarray,
    target_col: np.ndarray | None = None,
    config: TrainConfig | None = None,
    train_rows: np.ndarray | None = None,
    val_rows: np.ndarray | None = None,
    is_real: np.ndarray | None = None,
    on_eval: Callable[[int, Extract], None] | None = None,
    synth_level: np.ndarray | None = None,
    subspace_basis: np.ndarray | None = None,
    noise_model=None,
) -> tuple[Extract, list[dict]]:
    """Fit the model. Returns ``(model, history)``.

    Parameters
    ----------
    X
        [n_rows, n_genes] shrunken logFC, used **as it is**. Do not centre or
        scale the gene axis: ``ashr`` has already placed every entry on a
        common log-fold-change scale, and dividing by per-gene SD would remove
        exactly that and promote low-variance genes to equal footing.

        The model sees only ``X`` and the labels. All measured rows of a pair
        are exchangeable estimates of the same response: no cell counts, no
        subsample index, and no row is treated as the most precise.
    is_real
        [n_rows] bool: is this row an actual *measured* response vector?
        Defaults to all-True.
    synth_level
        [n_rows] fraction of genes shuffled in each synthetic row (NaN for
        measured rows). Only reported, per level, in validation; training
        treats every synthetic row alike.
    on_eval
        Called as ``on_eval(epoch, model)`` after every validation pass, before
        the best state is restored. Used to snapshot ``B`` at each evaluation
        so the epoch can be re-chosen afterwards under a different rule.

        A training row carries two independent bits: whether ``x`` was
        measured (``is_real``), and whether the label attached to it is its
        own.

        ``L_disc`` uses **every** row, through one head asking one question:
        "is this response the result of this perturbation in this context?"
        Measured rows with their own label are the positives. Both a measured
        row under a permuted label and a synthetic row under its **assigned**
        ``(p, c)`` are negatives. The synthetic row keeps the label it was
        generated for, so the head cannot reject it on a label mismatch -- it
        has to decide whether the vector is a plausible response for that
        label, which is the harder and more informative negative.

        ``L_recon`` is where the bits differ. Measured rows enter with a plus
        sign whatever label they carry: a label-permuted negative is still a
        real response vector, so ``B`` should reconstruct it. Synthetic rows
        enter with a **minus** sign scaled by ``TrainConfig.recon_fake_weight``
        (``0`` drops them entirely), so ``B`` is fitted only to genuinely
        observed perturbation responses. Minimising a synthetic row's
        reconstruction error would instead train ``B`` to span a
        product-of-marginals cloud whose covariance is diagonal by
        construction.
    train_rows, val_rows
        Boolean masks over rows, from
        :func:`extract.data.augment.make_split` at ``level="pair"``. All
        eleven samples of a held-out ``(perturbation, context)`` move together,
        so no pair appears in more than one half. ``None`` trains on
        everything, which leaves no way to tell a model that found structure
        from one that memorised -- pass a split.

        **Validation, not test.** ``fit`` looks at ``val_rows`` repeatedly, for
        early stopping. The test rows must not be passed here; score them once
        with :func:`evaluate` after fitting.

        The negative sampler is built from the **training rows only**, so a
        held-out pair is never named even as a negative.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)

    X = np.asarray(X, dtype=np.float32)
    perturbation_idx = np.asarray(perturbation_idx, dtype=np.int64)
    context_idx = np.asarray(context_idx, dtype=np.int64)
    n_rows, n_genes = X.shape

    for name, arr in (
        ("perturbation_idx", perturbation_idx),
        ("context_idx", context_idx),
    ):
        if len(arr) != n_rows:
            raise ValueError(f"{name} has {len(arr)} entries, X has {n_rows} rows")

    n_perturbations = int(perturbation_idx.max()) + 1
    n_contexts = int(context_idx.max()) + 1

    train_idx = (
        np.arange(n_rows)
        if train_rows is None
        else np.nonzero(np.asarray(train_rows, dtype=bool))[0]
    )
    val_idx = (
        None
        if val_rows is None
        else np.nonzero(np.asarray(val_rows, dtype=bool))[0]
    )
    if val_idx is not None and np.intersect1d(train_idx, val_idx).size:
        raise ValueError("train_rows and val_rows overlap")
    if len(train_idx) < config.batch_size:
        raise ValueError(
            f"only {len(train_idx)} training rows for batch_size "
            f"{config.batch_size}"
        )

    needs_basis = config.subspace in ("fixed", "anchored") or (
        config.subspace == "free" and config.free_init == "basis")
    if needs_basis and subspace_basis is None:
        # the label-driven subspace, from the measured TRAINING rows only.
        # is_real is validated here as well, since it is read before the
        # general check further down.
        if is_real is not None:
            if len(np.asarray(is_real)) != n_rows:
                raise ValueError(f"is_real has {len(np.asarray(is_real))} entries, X has {n_rows} rows")
            if not np.asarray(is_real, dtype=bool).any():
                raise ValueError("no real rows: L_recon would have nothing to fit")
        from .data.noise import (label_subspace_from_rows, pca_subspace_from_rows,
                                 rca_subspace_from_rows)
        if config.subspace_method not in ("rca", "snr", "pca"):
            raise ValueError(f"subspace_method must be 'rca', 'snr' or 'pca', got {config.subspace_method!r}")
        if config.subspace_method == "pca" and config.subspace != "anchored":
            subspace_basis = pca_subspace_from_rows(
                X, perturbation_idx, context_idx, train_idx, d=config.n_factors,
                seed=config.seed, is_real=is_real)
            what = "PCA of the pair means"
        elif config.subspace_method == "rca" and config.subspace != "anchored":
            subspace_basis = rca_subspace_from_rows(
                X, perturbation_idx, context_idx, train_idx, d=config.n_factors,
                rank=config.rca_rank, seed=config.seed, is_real=is_real)
            what = f"RCA, R_W rank {config.rca_rank}"
        else:
            subspace_basis, nm = label_subspace_from_rows(
                X, perturbation_idx, context_idx, train_idx,
                d=config.n_factors, rank=config.noise_rank,
                max_rows=config.noise_max_rows, seed=config.seed, is_real=is_real)
            noise_model = noise_model if noise_model is not None else nm
            what = f"signal/noise, noise rank {len(nm.s)}"
        if config.log_every:
            print(f"label-driven subspace: d={config.n_factors} ({what}), from the "
                  f"training rows", flush=True)

    model = Extract(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        n_contexts=n_contexts,
        config=config,
        subspace_basis=subspace_basis,
        noise_model=noise_model,
    ).to(device)
    if config.pert_embedding_init == "response":
        if not isinstance(model.label_net, FactorizedLabelNet):
            raise ValueError("pert_embedding_init='response' needs the MLP label model")
        if target_col is None:
            raise ValueError("pert_embedding_init='response' needs target_col (the target gene's column)")
        emb, has = response_embeddings(
            X, perturbation_idx, context_idx, np.asarray(target_col), train_idx,
            n_perturbations, config.embedding_dim, is_real=is_real, seed=config.seed)
        with torch.no_grad():
            w = model.label_net.perturbation_embedding.weight
            w[torch.from_numpy(has)] = torch.from_numpy(emb[has]).to(w)
        if config.log_every:
            print(f"perturbation embeddings: {int(has.sum())} of {n_perturbations} from "
                  f"target-gene responses", flush=True)
    elif config.pert_embedding_init != "random":
        raise ValueError(f"pert_embedding_init must be 'random' or 'response', "
                         f"got {config.pert_embedding_init!r}")
    model.subspace_basis_ = subspace_basis
    model.noise_model_ = noise_model

    weights = dict(config.negative_weights or DEFAULT_WEIGHTS)

    def _sampler(rows: np.ndarray) -> StratifiedNegativeSampler:
        return StratifiedNegativeSampler(
            perturbation_idx=perturbation_idx[rows],
            context_idx=context_idx[rows],
            stratum=np.zeros(len(rows), dtype=np.int64),   # rows are exchangeable
            n_perturbations=n_perturbations,
            n_contexts=n_contexts,
            weights=weights,
        )

    # Built on the training rows only: a held-out pair is never named, even as
    # a negative. The sampler indexes the rows it was given, so its positions
    # are offsets into `train_idx` rather than into X.
    sampler = _sampler(train_idx)
    eval_sampler = _sampler(val_idx) if val_idx is not None else None

    w = np.ones(n_rows)                     # every row weighs the same
    if is_real is None:
        is_real = np.ones(n_rows, dtype=bool)
    else:
        is_real = np.asarray(is_real, dtype=bool)
        if len(is_real) != n_rows:
            raise ValueError(
                f"is_real has {len(is_real)} entries, X has {n_rows} rows"
            )
        if not is_real.any():
            raise ValueError("no real rows: L_recon would have nothing to fit")

    X_t = torch.from_numpy(X).to(device)
    w_t = torch.from_numpy(w.astype(np.float32)).to(device)
    real_t = torch.from_numpy(is_real).to(device)
    p_all = torch.from_numpy(perturbation_idx).to(device)
    c_all = torch.from_numpy(context_idx).to(device)
    if target_col is None or config.no_mask:
        col_t = None
    else:
        col_t = torch.from_numpy(np.asarray(target_col, dtype=np.int64)).to(device)

    if config.label_weight_decay:
        label_params = set(map(id, model.label_net.parameters()))
        optimiser = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) not in label_params],
             "weight_decay": 0.0},
            {"params": list(model.label_net.parameters()),
             "weight_decay": config.label_weight_decay},
        ], lr=config.lr)
        if config.weight_decay:
            raise ValueError("weight_decay and label_weight_decay together are not supported")
    else:
        optimiser = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    tc_disc = tc_opt = None
    if config.tc_weight > 0:
        tc_disc = TotalCorrelationDiscriminator(
            config.n_factors, hidden=config.tc_hidden
        ).to(device)
        tc_opt = torch.optim.Adam(tc_disc.parameters(), lr=config.tc_lr)

    # One negative per positive: |permuted| + |synthetic| == |positives|.
    n_real_rows, n_synth_rows = int(is_real.sum()), int((~is_real).sum())
    p_keep_perm = 1.0
    if config.balance_negatives and n_synth_rows:
        p_keep_perm = max(0.0, 1.0 - n_synth_rows / max(n_real_rows, 1))
        print(
            f"negative budget: {n_real_rows:,} positives | "
            f"{n_synth_rows:,} synthetic + "
            f"~{round(p_keep_perm * n_real_rows):,} permuted "
            f"(keeping {p_keep_perm:.0%} of permuted draws)",
            flush=True,
        )

    history: list[dict] = []
    best = {"score": None, "epoch": -1, "state": None}
    since_best = 0
    higher_is_better = config.select_on == "accuracy"

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        totals = {"disc": 0.0, "recon": 0.0, "recon_measured": 0.0,
                  "recon_synth": 0.0, "tc": 0.0, "sparsity": 0.0, "accuracy": 0.0}
        n_batches = 0

        for start in range(0, len(train_idx), config.batch_size):
            local = order[start : start + config.batch_size]
            if len(local) < 2:
                continue
            rows = train_idx[local]          # positions in X
            rows_t = torch.from_numpy(rows).to(device)

            x = X_t[rows_t]
            cols = col_t[rows_t] if col_t is not None else None
            neg_p, neg_c = sampler.sample(local, rng)

            z = model.project(x, cols)                       # eq. (4)
            neg_p_t = torch.from_numpy(neg_p).to(device)
            neg_c_t = torch.from_numpy(neg_c).to(device)

            keep = real_t[rows_t]                 # measured rows
            synth = ~keep                         # column-shuffled rows
            all_real = bool(keep.all())

            # ---- L_disc, eq. (9) -------------------------------------
            # One head, one question: "is this response the result of this
            # perturbation in this context?" Three kinds of row answer it:
            #   measured x  + its own label       -> yes  (positive)
            #   measured x  + a permuted label    -> no   (negative)
            #   synthetic x + its ASSIGNED label  -> no   (negative)
            # The synthetic row keeps the (p, c) it was generated for -- the
            # label is real and correct, only the vector is not. That makes it
            # the harder negative of the two: the head cannot reject it on a
            # label mismatch, it has to judge whether the vector is a plausible
            # response for that label at all.
            if all_real:
                pos = model.score(z, p_all[rows_t], c_all[rows_t])
                neg = model.score(z, neg_p_t, neg_c_t)
            else:
                pos_l, neg_l = [], []
                if bool(keep.any()):
                    pos_l.append(model.score(z[keep], p_all[rows_t][keep],
                                             c_all[rows_t][keep]))
                    # Thin the permuted negatives so the two kinds together
                    # match the positives one for one. `p_keep_perm` is derived
                    # from the GLOBAL counts, not this batch, so the ratio is
                    # stable batch to batch.
                    kidx = torch.nonzero(keep, as_tuple=True)[0]
                    if p_keep_perm < 1.0:
                        take = torch.from_numpy(
                            rng.random(len(kidx)) < p_keep_perm).to(device)
                        kidx = kidx[take]
                    if len(kidx):
                        neg_l.append(model.score(z[kidx], neg_p_t[kidx],
                                                 neg_c_t[kidx]))
                if bool(synth.any()):
                    neg_l.append(model.score(z[synth], p_all[rows_t][synth],
                                             c_all[rows_t][synth]))
                pos = torch.cat(pos_l) if pos_l else torch.zeros(0, device=device)
                neg = torch.cat(neg_l) if neg_l else torch.zeros(0, device=device)
            real, fake = pos, neg

            loss_d = (
                contrastive_loss(pos, neg)
                if len(pos) and len(neg)
                else torch.zeros((), device=device)
            )

            # ---- L_recon, eq. (10) -----------------------------------
            # Measured rows enter with a plus sign, whatever label they carry.
            # Synthetic rows enter with a MINUS sign scaled by
            # `recon_fake_weight` (0 = not at all), so B is never fitted to a
            # product-of-marginals cloud whose covariance is diagonal by
            # construction.
            def _sub_cols(m):
                return cols[m] if cols is not None else None

            if all_real:
                per_row = model.loadings.squared_error(x, z, cols)
                loss_r = weighted_squared_error(per_row, w_t[rows_t])
            elif bool(keep.any()):
                per_row = model.loadings.squared_error(
                    x[keep], z[keep], _sub_cols(keep))
                loss_r = weighted_squared_error(per_row, w_t[rows_t][keep])
            else:
                loss_r = torch.zeros((), device=device)

            # Keep the two halves as well as the combined term. The combined
            # value is what is minimised and is the right single number to
            # compare against val -- but it is NOT comparable across beta: at
            # beta=0 it IS the measured error, at beta>0 it is a difference, so
            # a lower value cannot distinguish "fits the real data better" from
            # "fits the fakes worse". The guard on beta needs the measured half.
            loss_r_measured = loss_r
            loss_r_synth = torch.zeros((), device=device)
            if bool(synth.any()):
                per_row_f = model.loadings.squared_error(
                    x[synth], z[synth], _sub_cols(synth))
                loss_r_synth = weighted_squared_error(
                    per_row_f, w_t[rows_t][synth])
                if config.recon_fake_weight:
                    loss_r = loss_r - config.recon_fake_weight * loss_r_synth

            loss = loss_d + config.alpha * loss_r             # eq. (12)

            loss_s = torch.zeros((), device=device)
            if config.sparsity:
                zf = z[keep]
                if len(zf) > 1:
                    loss_s = (zf.abs() / (zf.std(0) + 1e-8)).mean(0).sum()
                    loss = loss + config.sparsity * loss_s

            loss_tc = torch.zeros((), device=device)
            if tc_disc is not None:
                loss_tc = tc_penalty(tc_disc, z)
                loss = loss + config.tc_weight * loss_tc

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()

            if tc_disc is not None:
                z_det = z.detach()
                tc_loss = tc_discriminator_loss(
                    tc_disc, z_det, shuffle_coordinates(z_det)
                )
                tc_opt.zero_grad(set_to_none=True)
                tc_loss.backward()
                tc_opt.step()

            totals["disc"] += float(loss_d)
            totals["recon"] += float(loss_r)
            totals["recon_measured"] += float(loss_r_measured)
            totals["recon_synth"] += float(loss_r_synth)
            totals["tc"] += float(loss_tc)
            totals["sparsity"] += float(loss_s)
            totals["accuracy"] += discrimination_accuracy(real, fake)
            n_batches += 1

        record = {
            "epoch": epoch,
            **{k: v / max(n_batches, 1) for k, v in totals.items()},
        }
        # the quantity actually minimised, eq. (12). Logged so the epoch where
        # val turns up while train keeps falling -- the overfitting onset -- is
        # readable from one column rather than reconstructed from two.
        record["total"] = record["disc"] + config.alpha * record["recon"]
        is_last = epoch == config.epochs - 1
        evaluated = False
        if val_idx is not None and config.eval_every and (
            epoch % config.eval_every == 0 or is_last
        ):
            evaluated = True
            record.update(
                {
                    f"val_{k}": v
                    for k, v in evaluate(
                        model,
                        X_t,
                        perturbation_idx,
                        context_idx,
                        val_idx,
                        eval_sampler,
                        np.random.default_rng(config.seed + 10_000 + epoch),
                        col_t,
                        w_t,
                        is_real=is_real,
                        recon_fake_weight=config.recon_fake_weight,
                        synth_level=synth_level,
                    ).items()
                }
            )
            record["val_total"] = (record["val_disc"]
                                   + config.alpha * record["val_recon"])
            score = record[f"val_{config.select_on}"]
            improved = (
                best["score"] is None
                or (score > best["score"] if higher_is_better else score < best["score"])
            )
            if improved:
                best = {
                    "score": score,
                    "epoch": epoch,
                    "state": {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    },
                }
                since_best = 0
            else:
                since_best += 1
            record["val_best"] = improved
            if on_eval is not None:
                on_eval(epoch, model)

        history.append(record)
        if config.log_every and (epoch % config.log_every == 0 or is_last or evaluated):
            line = (
                f"epoch {epoch:4d}  disc {record['disc']:.4f}  "
                f"recon {record['recon']:.4f}  acc {record['accuracy']:.3f}"
            )
            if "val_accuracy" in record:
                line += (
                    f"   | val acc {record['val_accuracy']:.3f}  "
                    f"recon {record['val_recon']:.4f}  "
                    f"r {record['val_span_residual_median']:.3f}"
                    + (f"  synth acc {record['val_synth_accuracy']:.3f}"
                       if record.get("val_n_synthetic") else "")
                    + f"{'  *best' if record.get('val_best') else ''}"
                )
            print(line, flush=True)

        if (config.patience and since_best >= config.patience
                and epoch + 1 >= config.min_epochs):
            print(
                f"early stop at epoch {epoch}: no val {config.select_on} "
                f"improvement in {since_best} evaluations "
                f"(best {best['score']:.4f} at epoch {best['epoch']})",
                flush=True,
            )
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(
            f"restored parameters from epoch {best['epoch']} "
            f"(val {config.select_on} {best['score']:.4f})",
            flush=True,
        )
        for rec in history:
            rec["selected_epoch"] = best["epoch"]

    return model, history


@torch.no_grad()
def evaluate(
    model: Extract,
    X_t: torch.Tensor,
    perturbation_idx: np.ndarray,
    context_idx: np.ndarray,
    rows: np.ndarray,
    sampler: StratifiedNegativeSampler,
    rng: np.random.Generator,
    target_col: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
    is_real: np.ndarray | None = None,
    recon_fake_weight: float = 0.0,
    batch_size: int = 2048,
    synth_level: np.ndarray | None = None,
) -> dict:
    """Metrics on held-out rows, scored exactly as training scores them.

    ``is_real`` [n_rows of X] marks measured rows; ``None`` means all measured.
    Measured rows under their own label are the positives. Measured rows under
    a permuted label, and synthetic rows under their **own** label, are the
    two kinds of negative. Synthetic rows are never positives and never enter
    ``recon`` or the span residual.

    What each number is testing:

    ``accuracy`` / ``disc``
        Measured rows under their own label vs a permuted one. Both ``p`` and
        ``c`` were seen in training but *never together*, so this is the
        interaction test. **A lookup table scores at chance here** -- this is
        the number that separates a model which found structure from one that
        memorised the training pairs. Synthetic rows do not enter it, so it is
        comparable between runs with and without them.
    ``synth_accuracy`` / ``disc_synth``
        Measured rows vs synthetic rows, each under its own label: can the
        model tell a real response from a marginal-matched fake one? Balanced,
        i.e. the mean of positive and synthetic-negative accuracy. ``NaN``
        when no synthetic rows are present.
    ``recon``
        Precision-weighted masked squared error on measured rows, comparable
        to the training ``recon`` directly.
    ``span_residual_median`` / ``_q90``
        ``||x - x B^+ B|| / ||x||`` on measured held-out rows -- eq. (14).
        Large values mean the held-out pairs need programs outside
        ``span(B)``, the one failure mode the architecture cannot absorb.
    ``synth_accuracy_f<x>`` / ``recon_synth_f<x>``
        With ``synth_level`` [n_rows of X] (the fraction of genes shuffled in
        each synthetic row), the two synthetic metrics again for the synthetic
        rows of each level: a difficulty curve, from barely altered rows (small
        fraction) to fully shuffled ones (1.0).
    ``real_score_mean`` / ``fake_score_mean`` / ``synth_score_mean``
        The raw logits. A collapsed head shows up here as both near zero even
        when accuracy looks acceptable.
    """
    was_training = model.training
    model.eval()
    # Take the device from the MODEL, not from X_t. A caller may keep the full
    # matrix on the CPU on purpose (it is ~11 GB here) and let batches move; if
    # we read the device off X_t instead, a CUDA model gets CPU batches and the
    # first matmul raises.
    device = next(model.parameters()).device
    src = X_t.device
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and (rows.max() >= len(X_t) or rows.min() < 0):
        raise IndexError(
            f"rows span [{rows.min()}, {rows.max()}] but X has {len(X_t)} rows"
        )
    if len(rows) != len(sampler.perturbation_idx):
        raise ValueError(
            f"sampler was built on {len(sampler.perturbation_idx)} rows but "
            f"{len(rows)} were passed; sample() indexes the sampler's own "
            "rows, so the two must correspond one-to-one"
        )

    real_rows = (
        np.ones(len(rows), dtype=bool)
        if is_real is None
        else np.asarray(is_real, dtype=bool)[rows]
    )
    if not real_rows.any():
        raise ValueError("no measured rows to evaluate")


    def _bce_sum(logits: torch.Tensor, target: float) -> float:
        return float(F.binary_cross_entropy_with_logits(
            logits, torch.full_like(logits, target), reduction="sum"))

    # Accumulated as sums and counts, so batches with few measured rows weigh
    # exactly as much as their rows do.
    t = dict.fromkeys(
        ["pos_ok", "pos_bce", "pos_sum", "perm_ok", "perm_bce", "perm_sum",
         "syn_ok", "syn_bce", "syn_sum", "wsq", "w",
         "wsq_syn", "w_syn"], 0.0)
    n_pos = n_syn = 0
    residuals = []
    lv_all = (np.asarray(synth_level, dtype=float) if synth_level is not None else None)
    per_level: dict[float, list[float]] = {}     # level -> [ok, n, wsq, w]

    for start in range(0, len(rows), batch_size):
        local = np.arange(start, min(start + batch_size, len(rows)))
        keep = real_rows[local]
        local_real, local_syn = local[keep], local[~keep]

        if len(local_real):
            idx = rows[local_real]
            idx_t = torch.from_numpy(idx).to(src)
            x = X_t[idx_t].to(device)
            cols = (target_col[idx_t].to(device)
                    if target_col is not None else None)
            z = model.project(x, cols)
            pos = model.score(
                z,
                torch.from_numpy(perturbation_idx[idx]).to(device),
                torch.from_numpy(context_idx[idx]).to(device),
            )
            neg_p, neg_c = sampler.sample(local_real, rng)
            perm = model.score(
                z,
                torch.from_numpy(neg_p).to(device),
                torch.from_numpy(neg_c).to(device),
            )
            t["pos_ok"] += float((pos > 0).sum())
            t["pos_bce"] += _bce_sum(pos, 1.0)
            t["pos_sum"] += float(pos.sum())
            t["perm_ok"] += float((perm <= 0).sum())
            t["perm_bce"] += _bce_sum(perm, 0.0)
            t["perm_sum"] += float(perm.sum())

            per_row = model.loadings.squared_error(x, z, cols)
            w = (row_weights[idx_t].to(device) if row_weights is not None
                 else torch.ones_like(per_row))
            t["wsq"] += float((per_row * w).sum())
            t["w"] += float(w.sum())
            residuals.append(model.loadings.span_residual(x, cols).cpu().numpy())
            n_pos += len(idx)

        if len(local_syn):
            idx = rows[local_syn]
            idx_t = torch.from_numpy(idx).to(src)
            cols = (target_col[idx_t].to(device)
                    if target_col is not None else None)
            z = model.project(X_t[idx_t].to(device), cols)
            syn = model.score(                     # its OWN label: a negative
                z,
                torch.from_numpy(perturbation_idx[idx]).to(device),
                torch.from_numpy(context_idx[idx]).to(device),
            )
            t["syn_ok"] += float((syn <= 0).sum())
            t["syn_bce"] += _bce_sum(syn, 0.0)
            t["syn_sum"] += float(syn.sum())
            per_row_s = model.loadings.squared_error(
                X_t[idx_t].to(device), z, cols)
            w_s = (row_weights[idx_t].to(device) if row_weights is not None
                   else torch.ones_like(per_row_s))
            t["wsq_syn"] += float((per_row_s * w_s).sum())
            t["w_syn"] += float(w_s.sum())
            n_syn += len(idx)
            if lv_all is not None:
                lv = lv_all[idx]
                ok = (syn <= 0).cpu().numpy(); sq = (per_row_s * w_s).cpu().numpy()
                ww = w_s.cpu().numpy()
                for f in np.unique(lv[~np.isnan(lv)]):
                    at = lv == f
                    acc = per_level.setdefault(float(f), [0.0, 0.0, 0.0, 0.0])
                    acc[0] += float(ok[at].sum()); acc[1] += float(at.sum())
                    acc[2] += float(sq[at].sum()); acc[3] += float(ww[at].sum())

    model.train(was_training)
    r = np.concatenate(residuals)
    nan = float("nan")
    return {
        "n_rows": int(n_pos),
        "n_synthetic": int(n_syn),
        "accuracy": (t["pos_ok"] + t["perm_ok"]) / (2 * n_pos),
        "disc": (t["pos_bce"] + t["perm_bce"]) / n_pos,
        "synth_accuracy": (
            0.5 * (t["pos_ok"] / n_pos + t["syn_ok"] / n_syn) if n_syn else nan
        ),
        "disc_synth": (
            t["pos_bce"] / n_pos + t["syn_bce"] / n_syn if n_syn else nan
        ),
        "recon_measured": t["wsq"] / max(t["w"], 1e-12),
        "recon_synth": (t["wsq_syn"] / t["w_syn"]) if t["w_syn"] else nan,
        # the combined objective, the same quantity training minimises
        "recon": (t["wsq"] / max(t["w"], 1e-12)
                  - recon_fake_weight * (t["wsq_syn"] / t["w_syn"]
                                         if t["w_syn"] else 0.0)),
        "real_score_mean": t["pos_sum"] / n_pos,
        "fake_score_mean": t["perm_sum"] / n_pos,
        "synth_score_mean": t["syn_sum"] / n_syn if n_syn else nan,
        "span_residual_median": float(np.median(r)),
        "span_residual_q90": float(np.quantile(r, 0.90)),
        **{k: v for f, (ok, n, wsq, w) in sorted(per_level.items()) for k, v in (
            (f"synth_accuracy_f{f:g}", 0.5 * (t["pos_ok"] / n_pos + ok / n)),
            (f"recon_synth_f{f:g}", wsq / w if w else nan))},
    }


# ---------------------------------------------------------------------------


def prepare(
    X: np.ndarray,
    meta,
    genes: np.ndarray,
    mask_on_target: bool = True,
    perturbation_levels: np.ndarray | None = None,
    context_levels: np.ndarray | None = None,
) -> dict:
    """Turn ``(X, meta)`` from :func:`extract.de.load_matrices` into
    :func:`fit` arguments.

    ``meta`` needs ``perturbation`` and ``context`` (and ``label`` only through
    the caller's choice of rows). Cell counts and subsample indices are not
    passed on: the model sees the matrix and the labels only.

    Pass ``perturbation_levels`` / ``context_levels`` from a previous call to
    encode a second row set (the test partition) with the **same** indices, so
    its labels address the embeddings the model was trained with. A label
    absent from the given levels is an error, not a new index.
    """
    import pandas as pd

    perturbations = meta["perturbation"].to_numpy()
    contexts = meta["context"].to_numpy()
    labels = meta["label"].astype(str).to_numpy()

    def _encode(values, levels, what):
        if levels is None:
            return pd.factorize(values, sort=True)
        codes = pd.Categorical(values, categories=levels).codes
        if (codes < 0).any():
            unknown = sorted(set(values[codes < 0]))
            raise ValueError(
                f"{len(unknown)} {what} not in the given levels, e.g. {unknown[:3]}"
            )
        return codes, np.asarray(levels)

    pert_codes, pert_levels = _encode(perturbations, perturbation_levels, "perturbations")
    ctx_codes, ctx_levels = _encode(contexts, context_levels, "contexts")

    out = {
        "X": np.asarray(X, dtype=np.float32),
        "perturbation_idx": pert_codes.astype(np.int64),
        "context_idx": ctx_codes.astype(np.int64),
        "target_col": (
            on_target_index(perturbations, genes) if mask_on_target else None
        ),
    }
    out["perturbation_levels"] = np.asarray(pert_levels)
    out["context_levels"] = np.asarray(ctx_levels)
    return out


@torch.no_grad()
def encode(
    model: Extract, X: np.ndarray, target_col: np.ndarray | None = None
) -> np.ndarray:
    """Factor activations ``Z`` for all rows of ``X``."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    cols = (
        torch.from_numpy(np.asarray(target_col, dtype=np.int64)).to(device)
        if target_col is not None
        else None
    )
    return model.project(x, cols).cpu().numpy()


@torch.no_grad()
def reconstruct(
    model: Extract, X: np.ndarray, target_col: np.ndarray | None = None
) -> np.ndarray:
    """``X_hat = z B`` with ``z`` the masked projection of ``X``."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    cols = (
        torch.from_numpy(np.asarray(target_col, dtype=np.int64)).to(device)
        if target_col is not None
        else None
    )
    return model.loadings.reconstruct(model.project(x, cols)).cpu().numpy()


__all__ = [
    "Extract",
    "evaluate",
    "TrainConfig",
    "fit",
    "prepare",
    "encode",
    "reconstruct",
    "NO_MASK",
]
