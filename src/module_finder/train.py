"""ModuleFinder: the model and its training loop.

    L = L_disc + alpha * L_recon   [ + tc_weight * L_tc ]

``L_disc`` picks the rotation, ``L_recon`` picks the subspace. Implements
``docs/paper/modulefinder.pdf`` exactly; equation numbers in comments refer to
it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .data.ontarget import on_target_index
from .models import FactorizedLabelNet, GlobalLoadings, PerComponentHead, UnconstrainedHead
from .models.heads import DEFAULT_BASIS
from .models.loadings import NO_MASK
from .objectives import (
    DEFAULT_WEIGHTS,
    StratifiedNegativeSampler,
    contrastive_loss,
    discrimination_accuracy,
    precision_weights,
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

    #: Precision weighting of the reconstruction term: "linear" is w ~ n_cells.
    weight_scheme: str = "linear"

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

    #: Validation metric to select on. "accuracy" maximises; "disc" and
    #: "recon" minimise.
    select_on: str = "accuracy"

    negative_weights: dict[str, float] | None = None
    vehicle_context_idx: tuple[int, ...] = ()
    log_every: int = 10
    _: bool = field(default=False, repr=False)


class ModuleFinder(nn.Module):
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
    ):
        super().__init__()
        self.config = config
        self.loadings = GlobalLoadings(config.n_factors, n_genes, ridge=config.ridge)
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


def fit(
    X: np.ndarray,
    perturbation_idx: np.ndarray,
    context_idx: np.ndarray,
    stratum: np.ndarray,
    n_cells: np.ndarray,
    target_col: np.ndarray | None = None,
    config: TrainConfig | None = None,
    train_rows: np.ndarray | None = None,
    val_rows: np.ndarray | None = None,
) -> tuple[ModuleFinder, list[dict]]:
    """Fit the model. Returns ``(model, history)``.

    Parameters
    ----------
    X
        [n_rows, n_genes] shrunken logFC, **already gene-standardised**
        (:func:`module_finder.data.rowbound.standardize_genes`). Per-gene SD
        spans ~5.4e4x on these matrices; without standardising, the leading
        factors are a handful of high-variance genes.
    stratum
        [n_rows] subsample index ``s`` in ``0..10`` (0 = the full-data row).
        Required: the negative sampler stratifies on it, and without that the
        discriminator wins on noise scale.
    n_cells
        [n_rows] cells behind each estimate. Drives the precision weight.
    target_col
        [n_rows] on-target column per row from
        :func:`module_finder.data.ontarget.on_target_index`, or ``None`` to
        mask nothing (which lets knockdown efficiency claim a factor).
    train_rows, val_rows
        Boolean masks over rows, from
        :func:`module_finder.data.augment.make_split` at ``level="pair"``. All
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
    stratum = np.asarray(stratum, dtype=np.int64)
    n_cells = np.asarray(n_cells, dtype=np.float64)
    n_rows, n_genes = X.shape

    for name, arr in (
        ("perturbation_idx", perturbation_idx),
        ("context_idx", context_idx),
        ("stratum", stratum),
        ("n_cells", n_cells),
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

    model = ModuleFinder(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        n_contexts=n_contexts,
        config=config,
    ).to(device)

    weights = dict(config.negative_weights or DEFAULT_WEIGHTS)
    if not config.vehicle_context_idx and weights.pop("against_vehicle", 0):
        # No vehicle context was identified, so that strategy has nothing to
        # draw. Renormalise over the remaining two rather than failing: the
        # against-vehicle negatives sharpen drug-specific modulation, they are
        # not required for the discrimination to work.
        warnings.warn(
            "no vehicle_context_idx given; dropping the 'against_vehicle' "
            "negatives and renormalising over the remaining strategies",
            stacklevel=2,
        )
    def _sampler(rows: np.ndarray) -> StratifiedNegativeSampler:
        return StratifiedNegativeSampler(
            perturbation_idx=perturbation_idx[rows],
            context_idx=context_idx[rows],
            stratum=stratum[rows],
            n_perturbations=n_perturbations,
            n_contexts=n_contexts,
            vehicle_context_idx=tuple(config.vehicle_context_idx),
            weights=weights,
        )

    # Built on the training rows only: a held-out pair is never named, even as
    # a negative. The sampler indexes the rows it was given, so its positions
    # are offsets into `train_idx` rather than into X.
    sampler = _sampler(train_idx)
    eval_sampler = _sampler(val_idx) if val_idx is not None else None

    w = precision_weights(n_cells, scheme=config.weight_scheme)

    X_t = torch.from_numpy(X).to(device)
    w_t = torch.from_numpy(w.astype(np.float32)).to(device)
    p_all = torch.from_numpy(perturbation_idx).to(device)
    c_all = torch.from_numpy(context_idx).to(device)
    if target_col is None or config.no_mask:
        col_t = None
    else:
        col_t = torch.from_numpy(np.asarray(target_col, dtype=np.int64)).to(device)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    tc_disc = tc_opt = None
    if config.tc_weight > 0:
        tc_disc = TotalCorrelationDiscriminator(
            config.n_factors, hidden=config.tc_hidden
        ).to(device)
        tc_opt = torch.optim.Adam(tc_disc.parameters(), lr=config.tc_lr)

    history: list[dict] = []
    best = {"score": None, "epoch": -1, "state": None}
    since_best = 0
    higher_is_better = config.select_on == "accuracy"

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        totals = {"disc": 0.0, "recon": 0.0, "tc": 0.0, "accuracy": 0.0}
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
            real = model.score(z, p_all[rows_t], c_all[rows_t])
            fake = model.score(
                z,
                torch.from_numpy(neg_p).to(device),
                torch.from_numpy(neg_c).to(device),
            )

            loss_d = contrastive_loss(real, fake)            # eq. (9)
            per_row = model.loadings.squared_error(x, z, cols)
            loss_r = weighted_squared_error(per_row, w_t[rows_t])  # eq. (10)
            loss = loss_d + config.alpha * loss_r             # eq. (12)

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
            totals["tc"] += float(loss_tc)
            totals["accuracy"] += discrimination_accuracy(real, fake)
            n_batches += 1

        record = {
            "epoch": epoch,
            **{k: v / max(n_batches, 1) for k, v in totals.items()},
        }
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
                    ).items()
                }
            )
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
                    f"{'  *best' if record.get('val_best') else ''}"
                )
            print(line, flush=True)

        if config.patience and since_best >= config.patience:
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
    model: ModuleFinder,
    X_t: torch.Tensor,
    perturbation_idx: np.ndarray,
    context_idx: np.ndarray,
    rows: np.ndarray,
    sampler: StratifiedNegativeSampler,
    rng: np.random.Generator,
    target_col: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
    batch_size: int = 2048,
) -> dict:
    """Metrics on held-out rows.

    What each number is testing:

    ``accuracy`` / ``disc``
        Can the model tell a held-out ``(p, c)`` pair from a permuted one? Both
        ``p`` and ``c`` were seen in training but *never together*, so this is
        the interaction test. **A lookup table scores at chance here** -- this
        is the number that separates a model which found structure from one
        that memorised the training pairs.
    ``recon``
        Precision-weighted masked squared error, comparable to the training
        ``recon`` directly.
    ``span_residual_median`` / ``_q90``
        ``||x - x B^+ B|| / ||x||`` on held-out rows -- eq. (14). Large values
        mean the held-out pairs need programs outside ``span(B)``, the one
        failure mode the architecture cannot absorb.
    ``real_score_mean`` / ``fake_score_mean``
        The raw logits. A collapsed head shows up here as both near zero even
        when accuracy looks acceptable.
    """
    model.eval()
    device = X_t.device
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

    n = 0
    acc = disc = recon = real_sum = fake_sum = 0.0
    residuals = []

    for start in range(0, len(rows), batch_size):
        local = np.arange(start, min(start + batch_size, len(rows)))
        if len(local) < 2:
            continue
        idx = rows[local]
        idx_t = torch.from_numpy(idx).to(device)
        x = X_t[idx_t]
        cols = target_col[idx_t] if target_col is not None else None

        neg_p, neg_c = sampler.sample(local, rng)
        z = model.project(x, cols)
        real = model.score(
            z,
            torch.from_numpy(perturbation_idx[idx]).to(device),
            torch.from_numpy(context_idx[idx]).to(device),
        )
        fake = model.score(
            z,
            torch.from_numpy(neg_p).to(device),
            torch.from_numpy(neg_c).to(device),
        )

        k = len(local)
        acc += discrimination_accuracy(real, fake) * k
        disc += float(contrastive_loss(real, fake)) * k
        per_row = model.loadings.squared_error(x, z, cols)
        w = row_weights[idx_t] if row_weights is not None else None
        recon += float(weighted_squared_error(per_row, w)) * k
        real_sum += float(real.sum())
        fake_sum += float(fake.sum())
        residuals.append(model.loadings.span_residual(x).cpu().numpy())
        n += k

    model.train()
    r = np.concatenate(residuals) if residuals else np.array([np.nan])
    return {
        "n_rows": int(n),
        "accuracy": acc / max(n, 1),
        "disc": disc / max(n, 1),
        "recon": recon / max(n, 1),
        "real_score_mean": real_sum / max(n, 1),
        "fake_score_mean": fake_sum / max(n, 1),
        "span_residual_median": float(np.median(r)),
        "span_residual_q90": float(np.quantile(r, 0.90)),
    }


# ---------------------------------------------------------------------------


def prepare(
    X: np.ndarray,
    meta,
    genes: np.ndarray,
    mask_on_target: bool = True,
) -> dict:
    """Turn ``(X, meta)`` from :func:`module_finder.de.load_matrices` into
    :func:`fit` arguments.

    ``meta`` needs ``perturbation``, ``context``, ``label`` and ``n_cells``.
    The stratum is read off the ``__subNN`` suffix of ``label``: the full-data
    row has no suffix and becomes stratum 0.
    """
    import pandas as pd

    perturbations = meta["perturbation"].to_numpy()
    contexts = meta["context"].to_numpy()
    labels = meta["label"].astype(str).to_numpy()

    pert_codes, pert_levels = pd.factorize(perturbations, sort=True)
    ctx_codes, ctx_levels = pd.factorize(contexts, sort=True)

    stratum = np.zeros(len(labels), dtype=np.int64)
    for i, lab in enumerate(labels):
        if "__sub" in lab:
            stratum[i] = int(lab.split("__sub")[-1]) + 1

    out = {
        "X": np.asarray(X, dtype=np.float32),
        "perturbation_idx": pert_codes.astype(np.int64),
        "context_idx": ctx_codes.astype(np.int64),
        "stratum": stratum,
        "n_cells": meta["n_cells"].to_numpy(),
        "target_col": (
            on_target_index(perturbations, genes) if mask_on_target else None
        ),
    }
    out["perturbation_levels"] = np.asarray(pert_levels)
    out["context_levels"] = np.asarray(ctx_levels)
    return out


@torch.no_grad()
def encode(
    model: ModuleFinder, X: np.ndarray, target_col: np.ndarray | None = None
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
    model: ModuleFinder, X: np.ndarray, target_col: np.ndarray | None = None
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
    "ModuleFinder",
    "evaluate",
    "TrainConfig",
    "fit",
    "prepare",
    "encode",
    "reconstruct",
    "NO_MASK",
]
