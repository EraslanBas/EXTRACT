"""Training loop: contrastive discrimination + reconstruction anchor."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .models import (
    FactorizedLabelNet,
    GlobalLinearDecoder,
    LinearEncoder,
    PerComponentHead,
    UnconstrainedHead,
)
from .models.heads import DEFAULT_BASIS
from .objectives import (
    NegativeSampler,
    contrastive_loss,
    discrimination_accuracy,
    weighted_mse,
)


@dataclass
class TrainConfig:
    n_factors: int = 20
    basis: tuple[str, ...] = DEFAULT_BASIS
    embedding_dim: int = 32
    label_hidden: int = 128

    #: Weight on the reconstruction term. Load-bearing: too low and the
    #: contrastive term selects tiny-variance directions that discriminate well
    #: but carry no interpretable gene program; too high and the model collapses
    #: to PCA and stops responding to the label. Watch both diagnostics.
    reconstruction_weight: float = 1.0

    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"

    #: Set True to ablate the identifiability constraint (see
    #: models.heads.UnconstrainedHead). For comparison only.
    unconstrained_head: bool = False

    negative_weights: dict[str, float] | None = None
    control_context_idx: int | None = None
    log_every: int = 10
    _: bool = field(default=False, repr=False)


class ModuleFinder(nn.Module):
    """Linear encoder + per-component contrastive head + global linear decoder."""

    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        n_contexts: int,
        config: TrainConfig,
    ):
        super().__init__()
        self.config = config
        self.encoder = LinearEncoder(n_genes, config.n_factors)
        self.decoder = GlobalLinearDecoder(config.n_factors, n_genes)
        self.head = (
            UnconstrainedHead(config.n_factors, len(config.basis))
            if config.unconstrained_head
            else PerComponentHead(config.basis)
        )
        self.label_net = FactorizedLabelNet(
            n_perturbations=n_perturbations,
            n_contexts=n_contexts,
            n_factors=config.n_factors,
            n_basis=len(config.basis),
            embedding_dim=config.embedding_dim,
            hidden=config.label_hidden,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def score(
        self, z: torch.Tensor, perturbation_idx: torch.Tensor, context_idx: torch.Tensor
    ) -> torch.Tensor:
        return self.head(z, self.label_net(perturbation_idx, context_idx))

    @property
    def loadings(self) -> np.ndarray:
        """``B`` as [n_factors, n_genes]. See interpret.loadings for how to put
        these back into log-fold-change units."""
        return self.decoder.loadings.detach().cpu().numpy()


def fit(
    X: np.ndarray,
    perturbation_idx: np.ndarray,
    context_idx: np.ndarray,
    config: TrainConfig | None = None,
    row_weights: np.ndarray | None = None,
) -> tuple[ModuleFinder, list[dict]]:
    """Fit the model. Returns ``(model, history)``.

    ``X`` should already be gene-standardised
    (:func:`module_finder.data.rowbound.standardize_genes`) -- per-gene SD spans
    ~5.4e4x on these matrices and without standardising the leading factors are
    a handful of high-variance genes.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)

    X = np.asarray(X, dtype=np.float32)
    perturbation_idx = np.asarray(perturbation_idx, dtype=np.int64)
    context_idx = np.asarray(context_idx, dtype=np.int64)
    n_samples = len(X)

    model = ModuleFinder(
        n_genes=X.shape[1],
        n_perturbations=int(perturbation_idx.max()) + 1,
        n_contexts=int(context_idx.max()) + 1,
        config=config,
    ).to(device)

    sampler = NegativeSampler(
        n_perturbations=int(perturbation_idx.max()) + 1,
        n_contexts=int(context_idx.max()) + 1,
        control_context_idx=config.control_context_idx,
        **({"weights": config.negative_weights} if config.negative_weights else {}),
    )

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    X_t = torch.from_numpy(X).to(device)
    w_t = (
        torch.from_numpy(np.asarray(row_weights, dtype=np.float32)).to(device)
        if row_weights is not None
        else None
    )

    history: list[dict] = []
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(n_samples)
        totals = {"contrastive": 0.0, "reconstruction": 0.0, "accuracy": 0.0}
        n_batches = 0

        for start in range(0, n_samples, config.batch_size):
            idx = order[start : start + config.batch_size]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            x = X_t[idx]
            p = perturbation_idx[idx]
            c = context_idx[idx]
            neg_p, neg_c = sampler.sample(p, c, rng)

            p_t = torch.from_numpy(p).to(device)
            c_t = torch.from_numpy(c).to(device)
            neg_p_t = torch.from_numpy(neg_p).to(device)
            neg_c_t = torch.from_numpy(neg_c).to(device)

            z = model.encode(x)
            real = model.score(z, p_t, c_t)
            fake = model.score(z, neg_p_t, neg_c_t)

            loss_c = contrastive_loss(real, fake)
            loss_r = weighted_mse(x, model.decoder(z), w_t[idx] if w_t is not None else None)
            loss = loss_c + config.reconstruction_weight * loss_r

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            totals["contrastive"] += float(loss_c)
            totals["reconstruction"] += float(loss_r)
            totals["accuracy"] += discrimination_accuracy(real, fake)
            n_batches += 1

        record = {"epoch": epoch, **{k: v / max(n_batches, 1) for k, v in totals.items()}}
        history.append(record)
        if config.log_every and epoch % config.log_every == 0:
            print(
                f"epoch {epoch:4d}  contrastive {record['contrastive']:.4f}  "
                f"recon {record['reconstruction']:.4f}  acc {record['accuracy']:.3f}"
            )

    return model, history


@torch.no_grad()
def encode(model: ModuleFinder, X: np.ndarray) -> np.ndarray:
    """Factor activations ``Z`` for all rows of ``X``."""
    model.eval()
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(
        next(model.parameters()).device
    )
    return model.encode(x).cpu().numpy()


@torch.no_grad()
def reconstruct(model: ModuleFinder, X: np.ndarray) -> np.ndarray:
    """``X_hat = encode(X) @ B``."""
    model.eval()
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(
        next(model.parameters()).device
    )
    return model.decoder(model.encode(x)).cpu().numpy()
