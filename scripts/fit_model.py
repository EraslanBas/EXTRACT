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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")

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

    config = TrainConfig(
        n_factors=args.n_factors, alpha=args.alpha, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=args.device,
        vehicle_context_idx=vehicle_idx, tc_weight=args.tc_weight,
        no_mask=args.no_mask, unconstrained_head=args.unconstrained_head,
        log_every=max(1, args.epochs // 25),
    )

    t0 = time.time()
    model, history = fit(config=config, **fit_args)
    print(f"fit in {time.time()-t0:.1f}s", flush=True)

    B = model.loading_matrix()
    np.save(args.out_dir / f"B_{tag}.npy", B)
    np.save(args.out_dir / f"gene_scale_{tag}.npy", gene_scale)
    Z = encode(model, fit_args["X"], fit_args["target_col"])
    np.save(args.out_dir / f"Z_{tag}.npy", Z)
    meta.to_csv(args.out_dir / f"rows_{tag}.csv", index=False)
    np.save(args.out_dir / f"genes_{tag}.npy", genes)

    r = model.loadings.span_residual(
        __import__("torch").from_numpy(fit_args["X"][: min(5000, len(Xs))])
    )
    summary = {
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
