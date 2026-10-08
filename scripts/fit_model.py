#!/usr/bin/env python3
"""Fit EXTRACT on a saved split (scripts/build_split.py).

Implements the method paper.

    python scripts/fit_model.py --n-factors 24 --alpha 1.0 --beta 0 --n-subsamples 10

Data discipline:

* fitting reads only ``<split>/trainval``; ``<split>/test`` is opened **once**,
  after fitting, for the final score on the parameters validation chose;
* validation is drawn inside trainval, grouped per perturbation, from
  ``--val-seed`` alone, so every config shares it;
* ``--n-subsamples`` (K) thins the TRAINING rows to a nested random subset of
  each pair's subsamples; validation and test keep all of theirs, so every K
  is scored on the same rows;
* synthetic (column-shuffled) rows are always used; ``--shuffled-frac`` (rho)
  sets their share of the negatives.

Run more than one ``--seed``: cross-seed reproducibility of B under Hungarian
matching (extract.evaluation.stability) is the acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract import paths
from extract.data import mask_coverage
from extract.data.splits import (
    load_partition,
    split_val_per_perturbation,
    subsample_mask,
    thin_synthetic,
)
from extract.objectives import DEFAULT_WEIGHTS, StratifiedNegativeSampler
from extract.train import TrainConfig, encode, evaluate, fit, prepare


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-dir", type=Path,
                   default=paths.root() / "splits" / "pair_seed0")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: $EXTRACT_ROOT/models/<split name>")
    p.add_argument("--contexts", nargs="+", default=None,
                   help="default: every context in the split")

    g = p.add_argument_group("model")
    g.add_argument("--sparsity", type=float, default=0.0,
                   help="weight of the activity-sparsity penalty (0 = off); adds _sparse<w> to the tag")
    g.add_argument("--head", default="statistics", choices=["statistics", "distance"],
                   help="discriminator head: per-factor statistics (default) or the distance "
                        "between measured and predicted activity; distance adds _dist to the tag")
    g.add_argument("--label-model", default="mlp", choices=["mlp", "product", "ammi"],
                   help="label -> lambda map: unconstrained MLP (default) or the structured "
                        "product lambda_k(p, c) = f_k(p) g_k(c), or ammi a_k(p) + b_k(c) + f_k(p) g_k(c)")
    g.add_argument("--subspace", default="fixed", choices=["fixed", "free", "anchored"],
                   help="fixed (default): B = A V, V the label-driven subspace computed "
                        "from the training rows, only A learned; free: B learned "
                        "directly (the earlier model)")
    g.add_argument("--noise-rank", type=int, default=50,
                   help="rank of the correlated noise behind V")
    g = p.add_argument_group("tuned hyperparameters")
    g.add_argument("--n-factors", "-d", type=int, default=24)
    g.add_argument("--alpha", type=float, default=1.0,
                   help="reconstruction weight")
    g.add_argument("--beta", "--recon-fake-weight", dest="beta", type=float,
                   default=0.0,
                   help="synthetic rows' reconstruction error, entered with a "
                        "MINUS sign, relative to alpha. 0 keeps them out of "
                        "L_recon; >0 is the contrastive-PCA-like setting")
    g.add_argument("--n-subsamples", "-K", type=int, default=10,
                   help="subsamples kept per training pair (nested random "
                        "subset of the 10 available); val/test keep all")

    g = p.add_argument_group("draws (fixed across a sweep)")
    g.add_argument("--val-frac", type=float, default=0.1)
    g.add_argument("--val-seed", type=int, default=0)
    g.add_argument("--draw-seed", type=int, default=0,
                   help="which subsamples K keeps, and which synthetic rows "
                        "rho keeps")
    g.add_argument("--shuffled-seed", type=int, default=0,
                   help="which <ctx>_shuffled_seed<k>.parquet to load")
    g.add_argument("--shuffled-frac", type=float, default=0.5,
                   help="rho: synthetic rows kept, as a fraction of measured "
                        "rows. With balancing this IS the synthetic share of "
                        "the negatives")

    g = p.add_argument_group("optimisation")
    g.add_argument("--seed", type=int, default=0,
                   help="model init / batching / negatives")
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--batch-size", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--device", default="cpu")
    g.add_argument("--eval-every", type=int, default=5)
    g.add_argument("--patience", type=int, default=4,
                   help="stop after this many evaluations without a val "
                        "improvement, and restore the best parameters")
    g.add_argument("--select-on", default="total",
                   choices=["accuracy", "total", "disc", "recon"])
    g.add_argument("--no-balance-negatives", dest="balance_negatives",
                   action="store_false")

    g = p.add_argument_group("ablations")
    g.add_argument("--tc-weight", type=float, default=0.0)
    g.add_argument("--no-mask", action="store_true",
                   help="leave the on-target diagonal in")
    g.add_argument("--unconstrained-head", action="store_true",
                   help="destroys identifiability, for comparison only")
    return p


def _pair_key(meta: pd.DataFrame) -> pd.Series:
    return meta.perturbation.astype(str) + "|" + meta.context.astype(str)


def main() -> None:
    args = build_argparser().parse_args()
    split_dir = args.split_dir
    manifest = json.loads((split_dir / "manifest.json").read_text())
    out_dir = args.out_dir or paths.root() / "models" / split_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"d{args.n_factors}_a{args.alpha:g}_b{args.beta:g}"
           f"_K{args.n_subsamples}_seed{args.seed}_{args.subspace}"
           + ("" if args.label_model == "mlp" else f"_{args.label_model}")
           + ("_dist" if args.head == "distance" else "")
           + (f"_sparse{args.sparsity:g}" if args.sparsity else ""))

    # ---- train/val partition only ---------------------------------------
    t0 = time.time()
    X, meta, genes = load_partition(split_dir, "trainval", args.contexts,
                                    shuffled=True, shuffle_seed=args.shuffled_seed)
    print(f"trainval: {len(meta):,} rows ({int(meta.is_real.sum()):,} measured) x "
          f"{len(genes):,} genes in {time.time() - t0:.1f}s", flush=True)

    real = meta.is_real.to_numpy()
    val_real = split_val_per_perturbation(meta[real], args.val_frac, args.val_seed)
    val_pairs = set(_pair_key(meta[real])[val_real])
    val = _pair_key(meta).isin(val_pairs).to_numpy()

    keep = subsample_mask(meta, args.n_subsamples, seed=args.draw_seed) | val
    keep &= thin_synthetic(meta, args.shuffled_frac, seed=args.draw_seed)
    X, meta, val = X[keep], meta[keep].reset_index(drop=True), val[keep]
    is_real = meta.is_real.to_numpy()
    train = ~val
    print(f"K={args.n_subsamples} (train only), rho={args.shuffled_frac}: "
          f"{int((train & is_real).sum()):,} train + {int((val & is_real).sum()):,} "
          f"val measured rows, {int((~is_real).sum()):,} synthetic; "
          f"{len(val_pairs):,} val pairs", flush=True)

    fit_args = prepare(X, meta, genes, mask_on_target=not args.no_mask)
    pert_levels = fit_args.pop("perturbation_levels")
    ctx_levels = fit_args.pop("context_levels")
    del X
    if fit_args["target_col"] is not None:
        cov = mask_coverage(fit_args["target_col"])
        print(f"on-target mask: {cov['fraction_masked']:.1%} of rows, "
              f"{int(cov['n_distinct_genes']):,} distinct genes", flush=True)

    config = TrainConfig(
        n_factors=args.n_factors, alpha=args.alpha, recon_fake_weight=args.beta,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, device=args.device, tc_weight=args.tc_weight,
        no_mask=args.no_mask, unconstrained_head=args.unconstrained_head,
        balance_negatives=args.balance_negatives,
        eval_every=args.eval_every, patience=args.patience,
        select_on=args.select_on, log_every=max(1, args.epochs // 25),
        subspace=args.subspace, noise_rank=args.noise_rank,
        label_model=args.label_model, head=args.head, sparsity=args.sparsity,
    )
    t0 = time.time()
    model, history = fit(config=config, train_rows=train, val_rows=val,
                         is_real=is_real, **fit_args)
    print(f"fit in {time.time() - t0:.1f}s", flush=True)
    if getattr(model, "subspace_basis_", None) is not None:
        np.save(out_dir / f"V_{tag}.npy", model.subspace_basis_)

    import torch

    def _metrics(fa: dict, rows: np.ndarray, real_flags: np.ndarray, name: str) -> dict:
        """Score ``rows`` of ``fa`` with a sampler built on those rows only."""
        sampler = StratifiedNegativeSampler(
            perturbation_idx=fa["perturbation_idx"][rows],
            context_idx=fa["context_idx"][rows],
            stratum=np.zeros(len(rows), dtype=np.int64),
            n_perturbations=len(pert_levels), n_contexts=len(ctx_levels),
            weights=dict(DEFAULT_WEIGHTS),
        )
        col_t = (torch.from_numpy(fa["target_col"])
                 if fa["target_col"] is not None else None)
        w_t = None                                    # every row weighs the same
        m = evaluate(model, torch.from_numpy(fa["X"]), fa["perturbation_idx"],
                     fa["context_idx"], rows, sampler, np.random.default_rng(12345),
                     col_t, w_t, is_real=real_flags,
                     recon_fake_weight=args.beta)
        print(f"\n{name:<6} n={m['n_rows']:>7,}  acc {m['accuracy']:.4f}  "
              f"disc {m['disc']:.4f}  recon {m['recon']:.4f}  "
              f"span r median {m['span_residual_median']:.4f} "
              f"q90 {m['span_residual_q90']:.4f}")
        print(f"       logits: real {m['real_score_mean']:+.2f}  "
              f"fake {m['fake_score_mean']:+.2f}")
        if m["n_synthetic"]:
            print(f"       vs synthetic (n={m['n_synthetic']:,}): "
                  f"acc {m['synth_accuracy']:.4f}  disc {m['disc_synth']:.4f}  "
                  f"logit {m['synth_score_mean']:+.2f}")
        return m

    metrics = {
        "train": _metrics(fit_args, np.nonzero(train)[0], is_real, "TRAIN"),
        "val": _metrics(fit_args, np.nonzero(val)[0], is_real, "VAL"),
    }

    B = model.loading_matrix()
    np.save(out_dir / f"B_{tag}.npy", B)
    np.save(out_dir / f"Z_{tag}.npy", encode(model, fit_args["X"], fit_args["target_col"]))
    meta.assign(val=val).to_csv(out_dir / f"rows_{tag}.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / f"history_{tag}.csv", index=False)
    del fit_args

    # ---- the single read of the test partition --------------------------
    Xte, mte, genes_te = load_partition(split_dir, "test", args.contexts,
                                        shuffled=True, shuffle_seed=args.shuffled_seed)
    if list(genes_te) != list(genes):
        raise ValueError("test partition genes differ from trainval")
    keep_te = thin_synthetic(mte, args.shuffled_frac, seed=args.draw_seed)
    Xte, mte = Xte[keep_te], mte[keep_te].reset_index(drop=True)
    fa_te = prepare(Xte, mte, genes, mask_on_target=not args.no_mask,
                    perturbation_levels=pert_levels, context_levels=ctx_levels)
    del Xte
    metrics["test"] = _metrics(fa_te, np.arange(len(mte)),
                               mte.is_real.to_numpy(), "TEST")
    np.save(out_dir / f"Z_test_{tag}.npy", encode(model, fa_te["X"], fa_te["target_col"]))

    print(f"\n  accuracy: train {metrics['train']['accuracy']:.4f}  "
          f"val {metrics['val']['accuracy']:.4f}  test {metrics['test']['accuracy']:.4f}"
          "   (0.5 = chance on held-out pairs)")
    print(f"  |val - test| = {abs(metrics['val']['accuracy'] - metrics['test']['accuracy']):.4f}"
          "  (large means the selection overfitted validation)")

    # the epoch whose parameters were restored, not simply the last one run
    sel = history[-1].get("selected_epoch", history[-1]["epoch"])
    selected = next(r for r in history if r["epoch"] == sel)
    summary = {
        "tag": tag,
        "split": {"dir": str(split_dir), **{k: manifest[k] for k in
                  ("split_seed", "test_frac", "gene_filter", "shuffle_seeds")}},
        "hyperparameters": {"d": args.n_factors, "alpha": args.alpha,
                            "beta": args.beta, "K": args.n_subsamples,
                            "rho": args.shuffled_frac},
        "draws": {"val_frac": args.val_frac, "val_seed": args.val_seed,
                  "draw_seed": args.draw_seed, "shuffled_seed": args.shuffled_seed,
                  "n_val_pairs": len(val_pairs)},
        "metrics": metrics,
        "selected_epoch": selected,
        "epochs_run": len(history),
        "n_genes": int(B.shape[1]),
        "n_perturbations": int(len(pert_levels)),
        "contexts": [str(c) for c in ctx_levels],
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in vars(config).items() if not k.startswith("_")},
    }
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {out_dir}/*_{tag}.*", flush=True)


if __name__ == "__main__":
    main()
