#!/usr/bin/env python3
"""Grid-search the number of latents on three criteria at once.

    discrimination   val accuracy on held-out (perturbation, context) pairs
    reconstruction   val precision-weighted masked squared error
    reproducibility  agreement of B between two seeds on the same split

The three pull in opposite directions, which is why all three are needed.
Discrimination and reconstruction improve monotonically with d, because more
latents is more capacity; reproducibility degrades, because latents beyond the
number the data determines are unconstrained and land wherever initialisation
puts them. The optimum is therefore interior, and a grid on any one criterion
alone would run to the edge of it.

Selection is on VALIDATION. The test set is read once, afterwards, for the
chosen d only -- choosing d on test would make its numbers no longer held out.

Reproducibility is measured two ways, because they answer different questions:

  per-factor MCC     Hungarian matching on |correlation| over gene loadings.
                     Asks whether the individual axes recur.
  subspace overlap   mean cos^2 of the principal angles between the two row
                     spaces. Asks whether the d-dimensional subspace recurs,
                     ignoring the basis inside it. A high overlap with a low
                     MCC is the signature of a determined subspace whose
                     rotation is not pinned; both low means the subspace itself
                     is undetermined.

Both are reported against their own null, which depends on d: Hungarian
matching over d x d selects maxima, and two random rank-d subspaces of R^G
overlap by roughly d/G.

    python scripts/grid_search_d.py --d 4 8 12 16 24 32 48 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from module_finder import paths
from module_finder.data.augment import make_split
from module_finder.de import load_matrices
from module_finder.evaluation.stability import match_factors
from module_finder.objectives import StratifiedNegativeSampler
from module_finder.objectives.reconstruction import precision_weights
from module_finder.train import TrainConfig, evaluate, fit, prepare


def subspace_overlap(B0: np.ndarray, B1: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cos^2 of principal angles between the two row spaces, and the angles."""
    Q0 = np.linalg.qr(np.asarray(B0, dtype=np.float64).T)[0]
    Q1 = np.linalg.qr(np.asarray(B1, dtype=np.float64).T)[0]
    cos = np.clip(np.linalg.svd(Q0.T @ Q1, compute_uv=False), 0.0, 1.0)
    return float(np.mean(cos**2)), np.degrees(np.arccos(cos))


def nulls_for_d(d: int, G: int, rng: np.random.Generator, n: int = 8) -> dict:
    """Both nulls at this d. Needed because both scale with d."""
    mcc, ov = [], []
    for _ in range(n):
        A = rng.normal(size=(d, G))
        B = rng.normal(size=(d, G))
        mcc.append(match_factors(A, B).mcc)
        ov.append(subspace_overlap(A, B)[0])
    return {"mcc_null": float(np.mean(mcc)), "overlap_null": float(np.mean(ov))}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrices", type=Path, default=paths.matrices())
    p.add_argument("--out-dir", type=Path, default=paths.root() / "grid_d")
    p.add_argument("--contexts", nargs="+", default=None)
    p.add_argument("--d", nargs="+", type=int, default=[4, 8, 12, 16, 24, 32, 48])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--min-gene-sd", type=float, default=0.0,
                   help="drop genes whose raw logFC SD is at or below this "
                        "before standardising. 0 keeps every gene, which is "
                        "what the first runs did")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---- load and prepare ONCE -------------------------------------
    t0 = time.time()
    X, meta = load_matrices(args.matrices, contexts=args.contexts)
    genes = np.asarray(X.columns)
    Xr = X.to_numpy()
    del X
    print(f"loaded {Xr.shape[0]:,} x {Xr.shape[1]:,} in {time.time()-t0:.0f}s", flush=True)

    if args.min_gene_sd > 0:
        keep = Xr.std(axis=0) > args.min_gene_sd
        print(f"gene filter SD > {args.min_gene_sd}: keeping {keep.sum():,} of "
              f"{len(genes):,} genes", flush=True)
        Xr = Xr[:, keep]
        genes = genes[keep]

    Xs = Xr                      # used as-is: no per-gene centring or scaling
    fit_args = prepare(Xs, meta, genes)
    pert_levels = fit_args.pop("perturbation_levels")
    ctx_levels = fit_args.pop("context_levels")
    G = Xs.shape[1]

    split = make_split(meta, level="pair", test_frac=args.test_frac,
                       val_frac=args.val_frac, seed=args.split_seed)
    print(f"split: {int(split.train.sum()):,} train / {int(split.val.sum()):,} val"
          f" / {int(split.test.sum()):,} test rows", flush=True)

    X_t = torch.from_numpy(fit_args["X"])
    col_t = (torch.from_numpy(fit_args["target_col"])
             if fit_args["target_col"] is not None else None)
    w_t = torch.from_numpy(
        precision_weights(fit_args["n_cells"]).astype(np.float32))

    def metrics_on(model, rows):
        sampler = StratifiedNegativeSampler(
            perturbation_idx=fit_args["perturbation_idx"][rows],
            context_idx=fit_args["context_idx"][rows],
            stratum=fit_args["stratum"][rows],
            n_perturbations=int(fit_args["perturbation_idx"].max()) + 1,
            n_contexts=int(fit_args["context_idx"].max()) + 1,
            weights={"same_s_other_pert": 0.5, "same_s_other_context": 0.5},
        )
        return evaluate(model, X_t, fit_args["perturbation_idx"],
                        fit_args["context_idx"], rows, sampler,
                        np.random.default_rng(12345), col_t, w_t)

    val_rows = np.nonzero(split.val)[0]
    test_rows = np.nonzero(split.test)[0]

    # ---- the grid ---------------------------------------------------
    rows_out, Bs = [], {}
    for d in args.d:
        per_seed = []
        for seed in args.seeds:
            cfg = TrainConfig(
                n_factors=d, alpha=args.alpha, epochs=args.epochs,
                batch_size=args.batch_size, seed=seed, log_every=0,
                eval_every=args.eval_every, patience=args.patience,
            )
            t = time.time()
            model, hist = fit(config=cfg, train_rows=split.train,
                              val_rows=split.val, **fit_args)
            m = metrics_on(model, val_rows)
            B = model.loading_matrix()
            Bs[(d, seed)] = B
            per_seed.append(m)
            print(f"d={d:>3} seed={seed}  val acc {m['accuracy']:.4f}  "
                  f"recon {m['recon']:.4f}  r {m['span_residual_median']:.4f}  "
                  f"epochs {len(hist)}  {time.time()-t:.0f}s", flush=True)

        pair = match_factors(Bs[(d, args.seeds[0])], Bs[(d, args.seeds[1])])
        ov, angles = subspace_overlap(Bs[(d, args.seeds[0])], Bs[(d, args.seeds[1])])
        nl = nulls_for_d(d, G, rng)
        rec = {
            "d": d,
            "val_accuracy": float(np.mean([m["accuracy"] for m in per_seed])),
            "val_recon": float(np.mean([m["recon"] for m in per_seed])),
            "val_span_r": float(np.mean([m["span_residual_median"] for m in per_seed])),
            "mcc": pair.mcc,
            "mcc_null": nl["mcc_null"],
            "overlap": ov,
            "overlap_null": nl["overlap_null"],
            "factors_above_0.5": int((pair.correlations > 0.5).sum()),
            "median_angle_deg": float(np.median(angles)),
        }
        rec["mcc_over_null"] = rec["mcc"] / max(rec["mcc_null"], 1e-9)
        rec["overlap_over_null"] = rec["overlap"] / max(rec["overlap_null"], 1e-9)
        rows_out.append(rec)
        print(f"d={d:>3}          MCC {pair.mcc:.4f} (null {nl['mcc_null']:.4f})  "
              f"overlap {ov:.4f} (null {nl['overlap_null']:.4f})  "
              f"{rec['factors_above_0.5']}/{d} factors > 0.5", flush=True)
        np.save(args.out_dir / f"B_d{d}_seed{args.seeds[0]}.npy", Bs[(d, args.seeds[0])])
        np.save(args.out_dir / f"B_d{d}_seed{args.seeds[1]}.npy", Bs[(d, args.seeds[1])])

    table = pd.DataFrame(rows_out)

    # ---- rank-sum over the three criteria ---------------------------
    # No weighted sum: the three are on incomparable scales, and any weighting
    # would be an invented preference. Rank each criterion, sum the ranks.
    table["rank_acc"] = table.val_accuracy.rank(ascending=False)
    table["rank_recon"] = table.val_recon.rank(ascending=True)
    table["rank_repro"] = table.overlap.rank(ascending=False)
    table["rank_sum"] = table[["rank_acc", "rank_recon", "rank_repro"]].sum(axis=1)
    table = table.sort_values("d")

    print("\n" + "=" * 78)
    cols = ["d", "val_accuracy", "val_recon", "overlap", "overlap_over_null",
            "mcc", "factors_above_0.5", "median_angle_deg", "rank_sum"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    table.to_csv(args.out_dir / "grid_d.csv", index=False)

    best_d = int(table.loc[table.rank_sum.idxmin(), "d"])
    print(f"\nrank-sum favours d = {best_d}")
    print("Read the table, not just the rank-sum: accuracy and reconstruction "
          "rise with d by construction, so if reproducibility is flat and poor "
          "across the whole grid, d is not the binding constraint and the "
          "rank-sum is just picking the largest d.")

    # ---- the single test read, for the selected d only ---------------
    cfg = TrainConfig(n_factors=best_d, alpha=args.alpha, epochs=args.epochs,
                      batch_size=args.batch_size, seed=args.seeds[0], log_every=0,
                      eval_every=args.eval_every, patience=args.patience)
    model, _ = fit(config=cfg, train_rows=split.train, val_rows=split.val, **fit_args)
    test_m = metrics_on(model, test_rows)
    print(f"\nTEST at d={best_d}: acc {test_m['accuracy']:.4f}  "
          f"recon {test_m['recon']:.4f}  "
          f"span r {test_m['span_residual_median']:.4f}  n={test_m['n_rows']:,}")

    (args.out_dir / "grid_d_summary.json").write_text(json.dumps(
        {"grid": rows_out, "selected_d": best_d, "test": test_m,
         "alpha": args.alpha, "min_gene_sd": args.min_gene_sd,
         "n_genes": int(G), "contexts": [str(c) for c in ctx_levels]}, indent=2))
    print(f"\nwrote {args.out_dir}/grid_d.csv and grid_d_summary.json")


if __name__ == "__main__":
    main()
