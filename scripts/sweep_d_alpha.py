#!/usr/bin/env python3
"""Sweep d (and alpha) in ONE process, recording both loss terms everywhere.

Loads the matrices, gene list, synthetic rows and split exactly once, then
fits every (d, alpha, seed) against them. Calling fit_model.py per cell would
re-read ~44 GB of parquet each time; here it is read once.

Written per model:
  B_<tag>.npy            the loading matrix
  history_<tag>.csv      PER EPOCH: train disc/recon/accuracy, and val
                         disc/recon/accuracy/span at every eval point
Written once:
  sweep_metrics.csv      one row per model x split, with disc AND recon for
                         train / val / test, plus accuracy, span residual and
                         the logit means

Both loss terms are kept for every split because they diagnose different
failures: recon flat across epochs means alpha is too small to anchor the
subspace; disc falling while val disc rises is the head overfitting; recon
low with disc at chance means the model reconstructs but has not oriented.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from module_finder import paths
from module_finder.data import apply_split, load_shuffled, mask_coverage
from module_finder.data import apply_gene_list, load_gene_list
from module_finder.data.augment import make_split
from module_finder.de import load_matrices
from module_finder.objectives import StratifiedNegativeSampler
from module_finder.objectives.reconstruction import precision_weights
from module_finder.train import TrainConfig, evaluate, fit, prepare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene-list", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--d", nargs="+", type=int,
                    default=[2, 4, 6, 8, 10, 12, 16, 20])
    ap.add_argument("--alpha", nargs="+", type=float, default=[1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--shuffled", action="store_true")
    ap.add_argument("--shuffled-frac", type=float, default=0.5)
    ap.add_argument("--recon-fake-weight", type=float, default=0.0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- load once -------------------------------------
    t0 = time.time()
    X, meta = load_matrices()
    genes_all = np.asarray(X.columns)
    Xs = X.to_numpy(); del X
    Xs, genes = apply_gene_list(Xs, load_gene_list(args.gene_list), genes=genes_all)
    print(f"real: {Xs.shape[0]:,} x {Xs.shape[1]:,}  ({time.time()-t0:.0f}s)", flush=True)

    is_real = None
    if args.shuffled:
        Xf, meta_f = load_shuffled(sorted(meta.context.unique()), meta,
                                   genes=genes, frac=args.shuffled_frac)
        print(f"synthetic: {len(meta_f):,} rows "
              f"({len(meta_f)/len(meta):.0%} of real)", flush=True)
        Xs = np.vstack([Xs, Xf]); del Xf
        is_real = np.concatenate([np.ones(len(meta), bool),
                                  np.zeros(len(meta_f), bool)])
        meta = pd.concat([meta, meta_f], ignore_index=True)

    fit_args = prepare(Xs, meta, genes)
    ctx_levels = fit_args.pop("context_levels"); fit_args.pop("perturbation_levels")
    cov = mask_coverage(fit_args["target_col"])
    print(f"on-target mask: {cov['fraction_masked']:.1%} of rows", flush=True)

    meta_real = meta[is_real] if is_real is not None else meta
    split = make_split(meta_real, level="pair", test_frac=args.test_frac,
                       val_frac=args.val_frac, seed=args.split_seed)
    if is_real is not None:
        split = apply_split(split, meta)
    print(f"split: {int(split.train.sum()):,} train / {int(split.val.sum()):,} val "
          f"/ {int(split.test.sum()):,} test rows", flush=True)

    X_t = torch.from_numpy(fit_args["X"])
    col_t = torch.from_numpy(fit_args["target_col"])
    w_t = torch.from_numpy(precision_weights(fit_args["n_cells"]).astype(np.float32))
    rows_of = {"train": np.nonzero(split.train)[0],
               "val": np.nonzero(split.val)[0],
               "test": np.nonzero(split.test)[0]}

    def score(model, rows):
        s = StratifiedNegativeSampler(
            perturbation_idx=fit_args["perturbation_idx"][rows],
            context_idx=fit_args["context_idx"][rows],
            stratum=fit_args["stratum"][rows],
            n_perturbations=int(fit_args["perturbation_idx"].max()) + 1,
            n_contexts=int(fit_args["context_idx"].max()) + 1,
            weights={"same_s_other_pert": 0.5, "same_s_other_context": 0.5})
        return evaluate(model, X_t, fit_args["perturbation_idx"],
                        fit_args["context_idx"], rows, s,
                        np.random.default_rng(12345), col_t, w_t,
                        is_real=is_real)

    # ---------------- the sweep -------------------------------------
    records = []
    total = len(args.d) * len(args.alpha) * len(args.seeds)
    k = 0
    for d in args.d:
        for a in args.alpha:
            for seed in args.seeds:
                k += 1
                tag = f"d{d}_a{a:g}_seed{seed}"
                t = time.time()
                cfg = TrainConfig(n_factors=d, alpha=a, seed=seed,
                                  epochs=args.epochs, log_every=0,
                                  eval_every=2, patience=6,
                                  recon_fake_weight=args.recon_fake_weight)
                model, hist = fit(config=cfg, train_rows=split.train,
                                  val_rows=split.val, is_real=is_real, **fit_args)
                pd.DataFrame(hist).to_csv(args.out_dir / f"history_{tag}.csv",
                                          index=False)
                np.save(args.out_dir / f"B_{tag}.npy", model.loading_matrix())
                row = {"d": d, "alpha": a, "seed": seed, "tag": tag,
                       "epochs_run": len(hist), "seconds": round(time.time() - t, 1)}
                for sp, rr in rows_of.items():
                    m = score(model, rr)
                    for key in ("disc", "recon", "accuracy",
                                "span_residual_median", "real_score_mean",
                                "fake_score_mean"):
                        row[f"{sp}_{key}"] = m[key]
                records.append(row)
                pd.DataFrame(records).to_csv(args.out_dir / "sweep_metrics.csv",
                                             index=False)
                print(f"[{k}/{total}] {tag:<18} "
                      f"train disc {row['train_disc']:.4f} recon {row['train_recon']:.5f} | "
                      f"val disc {row['val_disc']:.4f} recon {row['val_recon']:.5f} "
                      f"acc {row['val_accuracy']:.4f} | {row['seconds']:.0f}s", flush=True)

    (args.out_dir / "sweep_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(f"\nwrote {args.out_dir}/sweep_metrics.csv  ({len(records)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
