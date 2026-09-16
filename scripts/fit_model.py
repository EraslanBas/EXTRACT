#!/usr/bin/env python3
"""Fit ModuleFinder on the built posterior-mean matrices.

Implements docs/paper/modulefinder.pdf. Defaults read from
$MODULEFINDER_ROOT/matrices and write to $MODULEFINDER_ROOT/models.

    python scripts/fit_model.py --contexts Stattic DG-172 --epochs 50
    python scripts/fit_model.py --n-factors 24 --alpha 1.0 --seed 0

Run it more than once with different seeds: cross-seed reproducibility of B
under Hungarian matching (module_finder.evaluation.stability) is the only
acceptance evidence available, since unmixedness is not directly testable and
independence tests cannot substitute for it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from module_finder import paths
from module_finder.data import mask_coverage, on_target_index
from module_finder.data.augment import make_split
from module_finder.data.rowbound import standardize_genes
from module_finder.de import load_matrices
from module_finder.train import TrainConfig, encode, fit, prepare


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrices", type=Path, default=paths.matrices())
    p.add_argument("--out-dir", type=Path, default=paths.root() / "models")
    p.add_argument("--contexts", nargs="+", default=None,
                   help="default: every context present")
    p.add_argument("--variant", choices=["main", "subsample"], default=None,
                   help="default: all 11 rows per perturbation, which is what "
                        "the precision weight and the stratified negatives need")

    p.add_argument("--n-factors", type=int, default=24,
                   help="participation-ratio effective rank is 16.0, so 20-30")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="reconstruction weight; with d, the only real knob")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0,
                   help="model init / batching / negatives")
    p.add_argument("--device", default="cpu")

    p.add_argument("--test-frac", type=float, default=0.1,
                   help="looked at ONCE, after fitting")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="looked at repeatedly: early stopping and "
                        "hyperparameters. 0 disables, which means the stopping "
                        "epoch would have to come off the test set")
    p.add_argument("--split-level", choices=["pair", "perturbation", "context"],
                   default="pair",
                   help="pair holds out (perturbation, context) combinations, "
                        "both sides seen elsewhere, so the model must supply "
                        "the interaction. All 11 samples move together.")
    p.add_argument("--split-seed", type=int, default=0,
                   help="kept separate from --seed on purpose: two model seeds "
                        "must share one held-out set for their B matrices to "
                        "be comparable")
    p.add_argument("--held-out-contexts", nargs="+", default=None,
                   help="for --split-level context")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=4,
                   help="stop after this many evaluations without a val "
                        "improvement, and restore the best parameters")
    p.add_argument("--select-on", default="accuracy",
                   choices=["accuracy", "disc", "recon"])
    p.add_argument("--no-split", action="store_true",
                   help="train on every row; leaves no way to tell a model "
                        "that found structure from one that memorised")

    p.add_argument("--vehicle-contexts", nargs="+", default=["DMSO_round2",
                                                             "DMSO_round2_batch2"])
    p.add_argument("--tc-weight", type=float, default=0.0,
                   help="optional total-correlation term; off by default, and "
                        "not a substitute for the per-component head")
    p.add_argument("--no-mask", action="store_true",
                   help="ablation: leave the on-target diagonal in, which lets "
                        "knockdown efficiency claim a factor")
    p.add_argument("--unconstrained-head", action="store_true",
                   help="ablation: destroys identifiability, for comparison only")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"d{args.n_factors}_a{args.alpha:g}_seed{args.seed}"

    t0 = time.time()
    X, meta = load_matrices(args.matrices, contexts=args.contexts, variant=args.variant)
    genes = np.asarray(X.columns)
    print(f"loaded {X.shape[0]:,} rows x {X.shape[1]:,} genes in {time.time()-t0:.1f}s",
          flush=True)

    Xs, gene_scale = standardize_genes(X.to_numpy())
    del X

    fit_args = prepare(Xs, meta, genes, mask_on_target=not args.no_mask)
    pert_levels = fit_args.pop("perturbation_levels")
    ctx_levels = fit_args.pop("context_levels")

    if fit_args["target_col"] is not None:
        cov = mask_coverage(fit_args["target_col"])
        print(f"on-target mask: {cov['fraction_masked']:.1%} of rows, "
              f"{int(cov['n_distinct_genes']):,} distinct genes", flush=True)
        if cov["fraction_masked"] < 0.5:
            print("  WARNING: under half the rows matched a gene column. The "
                  "diagonal is still in the matrix and will claim a factor. "
                  "Check symbol vs Ensembl-ID naming.", flush=True)

    vehicle_idx = tuple(
        int(i) for i, name in enumerate(ctx_levels) if name in set(args.vehicle_contexts)
    )
    print(f"contexts: {list(ctx_levels)}")
    print(f"vehicle contexts: {[ctx_levels[i] for i in vehicle_idx] or 'none present'}",
          flush=True)

    # ---- the split ---------------------------------------------------
    split = None
    if not args.no_split:
        split = make_split(
            meta,
            level=args.split_level,
            test_frac=args.test_frac,
            val_frac=args.val_frac,
            seed=args.split_seed,
            held_out_contexts=args.held_out_contexts,
        )
        info = split.summary(meta)
        print(f"split ({args.split_level}, seed {args.split_seed}): "
              f"{info['train_rows']:,} train / "
              f"{info.get('val_rows', 0):,} val / "
              f"{info['test_rows']:,} test rows "
              f"({len(split.held_out_val):,} val + {len(split.held_out):,} test "
              f"held-out {args.split_level}s)", flush=True)
        if not split.three_way:
            print("  WARNING: no validation set. Early stopping is disabled, "
                  "because selecting the epoch on test would stop test from "
                  "being held out.", flush=True)

        # An accidental leak would silently invert the meaning of every test
        # number, so check rather than trust.
        pairs = list(zip(meta.perturbation.astype(str), meta.context.astype(str)))
        halves = {
            "train": {pc for pc, m in zip(pairs, split.train) if m},
            "test": {pc for pc, m in zip(pairs, split.test) if m},
        }
        if split.val is not None:
            halves["val"] = {pc for pc, m in zip(pairs, split.val) if m}
        names = list(halves)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = halves[a] & halves[b]
                if overlap:
                    raise SystemExit(
                        f"LEAK: {len(overlap)} pairs in both {a} and {b}, "
                        f"e.g. {sorted(overlap)[:3]}"
                    )
        print("  leak check: no pair in more than one half  ("
              + " / ".join(f"{len(halves[n]):,} {n}" for n in names)
              + " pairs)", flush=True)

    config = TrainConfig(
        n_factors=args.n_factors, alpha=args.alpha, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=args.device,
        vehicle_context_idx=vehicle_idx, tc_weight=args.tc_weight,
        no_mask=args.no_mask, unconstrained_head=args.unconstrained_head,
        eval_every=args.eval_every, patience=args.patience,
        select_on=args.select_on,
        log_every=max(1, args.epochs // 25),
    )

    t0 = time.time()
    model, history = fit(
        config=config,
        train_rows=split.train if split else None,
        val_rows=split.val if split else None,
        **fit_args,
    )
    print(f"fit in {time.time()-t0:.1f}s", flush=True)

    B = model.loading_matrix()
    np.save(args.out_dir / f"B_{tag}.npy", B)
    np.save(args.out_dir / f"gene_scale_{tag}.npy", gene_scale)
    Z = encode(model, fit_args["X"], fit_args["target_col"])
    np.save(args.out_dir / f"Z_{tag}.npy", Z)
    meta.to_csv(args.out_dir / f"rows_{tag}.csv", index=False)
    np.save(args.out_dir / f"genes_{tag}.npy", genes)

    import torch

    from module_finder.objectives import StratifiedNegativeSampler
    from module_finder.objectives.reconstruction import precision_weights
    from module_finder.train import evaluate

    X_t = torch.from_numpy(fit_args["X"])
    col_t = (
        torch.from_numpy(fit_args["target_col"])
        if fit_args["target_col"] is not None
        else None
    )
    w_t = torch.from_numpy(
        precision_weights(fit_args["n_cells"], scheme=config.weight_scheme).astype(
            np.float32
        )
    )

    def _metrics(rows: np.ndarray, tag: str) -> dict:
        sampler = StratifiedNegativeSampler(
            perturbation_idx=fit_args["perturbation_idx"][rows],
            context_idx=fit_args["context_idx"][rows],
            stratum=fit_args["stratum"][rows],
            n_perturbations=int(fit_args["perturbation_idx"].max()) + 1,
            n_contexts=int(fit_args["context_idx"].max()) + 1,
            vehicle_context_idx=vehicle_idx,
            weights={"same_s_other_pert": 0.5, "same_s_other_context": 0.5},
        )
        m = evaluate(
            model, X_t, fit_args["perturbation_idx"], fit_args["context_idx"],
            rows, sampler, np.random.default_rng(12345), col_t, w_t,
        )
        print(f"\n{tag:<6} n={m['n_rows']:>7,}  acc {m['accuracy']:.4f}  "
              f"disc {m['disc']:.4f}  recon {m['recon']:.4f}  "
              f"span r median {m['span_residual_median']:.4f} "
              f"q90 {m['span_residual_q90']:.4f}")
        print(f"       logits: real {m['real_score_mean']:+.2f}  "
              f"fake {m['fake_score_mean']:+.2f}")
        return m

    all_rows = np.arange(len(Xs))
    train_rows = np.nonzero(split.train)[0] if split else all_rows
    metrics = {"train": _metrics(train_rows, "TRAIN")}
    if split is not None:
        if split.val is not None:
            metrics["val"] = _metrics(np.nonzero(split.val)[0], "VAL")
            np.save(args.out_dir / f"val_mask_{tag}.npy", split.val)
        # The single read of the test set, on the parameters validation chose.
        metrics["test"] = _metrics(np.nonzero(split.test)[0], "TEST")
        np.save(args.out_dir / f"test_mask_{tag}.npy", split.test)
        print(f"\n  accuracy: train {metrics['train']['accuracy']:.4f}"
              + (f"  val {metrics['val']['accuracy']:.4f}" if "val" in metrics else "")
              + f"  test {metrics['test']['accuracy']:.4f}"
              f"   (0.5 = chance on held-out pairs)")
        gap = metrics["train"]["accuracy"] - metrics["test"]["accuracy"]
        print(f"  train - test = {gap:+.4f}")
        if "val" in metrics:
            vt = abs(metrics["val"]["accuracy"] - metrics["test"]["accuracy"])
            print(f"  |val - test| = {vt:.4f}  "
                  "(large means the selection overfitted validation)")

    r = model.loadings.span_residual(X_t[: min(5000, len(Xs))])
    summary = {
        "metrics": metrics,
        "split": (
            {
                "level": args.split_level,
                "seed": args.split_seed,
                "test_frac": args.test_frac,
                "n_held_out": len(split.held_out),
                **{k: v for k, v in split.summary(meta).items()},
            }
            if split
            else None
        ),
        "tag": tag,
        "n_rows": int(len(Xs)),
        "n_genes": int(B.shape[1]),
        "n_factors": int(B.shape[0]),
        "n_perturbations": int(len(pert_levels)),
        "contexts": [str(c) for c in ctx_levels],
        "final": history[-1],
        "span_residual_median": float(r.median()),
        "span_residual_q90": float(r.quantile(0.9)),
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in vars(config).items() if not k.startswith("_")},
    }
    (args.out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["final"], indent=2))
    print(f"span residual: median {summary['span_residual_median']:.3f} "
          f"q90 {summary['span_residual_q90']:.3f}")
    print(f"wrote {args.out_dir}/*_{tag}.*", flush=True)


if __name__ == "__main__":
    main()
