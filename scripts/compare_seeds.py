#!/usr/bin/env python3
"""Compare B matrices from independent fits -- the acceptance evidence of Q4.

Factors are recovered only up to permutation and sign, so runs are matched by
Hungarian assignment on |correlation| before scoring (Kuhn-Munkres), and the
score is the mean matched |correlation| (MCC), the standard permutation-
invariant measure in the nonlinear ICA literature.

A bare MCC is not interpretable on its own: Hungarian matching over d x d
picks maxima, so even unrelated matrices score above zero. This script
therefore reports two nulls alongside it --

  shuffled   each run's B with its gene columns permuted, matched as usual.
             Destroys which genes go together, keeps every marginal.
  random     Gaussian matrices of the same shape.

-- and the verdict is the gap between the real MCC and those, not the MCC.

    python scripts/compare_seeds.py --tags d24_a1_seed0 d24_a1_seed1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract import paths
from extract.evaluation.stability import match_factors, stability_across_seeds


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", type=Path, default=paths.root() / "models")
    p.add_argument("--tags", nargs="+", required=True,
                   help="tags as written by fit_model.py, e.g. d24_a1_seed0")
    p.add_argument("--n-null", type=int, default=20,
                   help="null replicates for the two baselines")
    p.add_argument("--top-genes", type=int, default=12,
                   help="genes to print for the most stable factors")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rng = np.random.default_rng(args.seed)

    Bs, Zs = [], []
    for tag in args.tags:
        Bs.append(np.load(args.models_dir / f"B_{tag}.npy"))
        zp = args.models_dir / f"Z_{tag}.npy"
        Zs.append(np.load(zp) if zp.exists() else None)
    d, G = Bs[0].shape
    for tag, B in zip(args.tags, Bs):
        if B.shape != (d, G):
            raise SystemExit(f"{tag}: shape {B.shape} != {(d, G)}")
    print(f"{len(Bs)} runs, d={d}, G={G:,}\n")

    # ---- loadings -------------------------------------------------------
    stab = stability_across_seeds(Bs)
    print(f"B  (loadings)   MCC = {stab['mcc']:.4f}")

    pair = match_factors(Bs[0], Bs[1])
    order = np.argsort(-pair.correlations)
    print("\nper-factor matched |corr|, reference run = "
          f"{args.tags[0]}, sorted:")
    line = "  "
    for rank, k in enumerate(order):
        line += f"{pair.correlations[k]:.3f} "
        if rank % 8 == 7:
            print(line); line = "  "
    if line.strip():
        print(line)
    print(f"  median {np.median(pair.correlations):.4f}   "
          f"min {pair.correlations.min():.4f}   "
          f"max {pair.correlations.max():.4f}")
    print(f"  factors above 0.9: {int((pair.correlations > 0.9).sum())}/{d}   "
          f"above 0.5: {int((pair.correlations > 0.5).sum())}/{d}")

    # ---- activations ----------------------------------------------------
    if all(z is not None for z in Zs):
        z_stab = stability_across_seeds([z.T for z in Zs])
        print(f"\nZ  (activations) MCC = {z_stab['mcc']:.4f}   "
              f"(rows x factors -> factors x rows)")

    # ---- nulls ----------------------------------------------------------
    shuffled, random_null = [], []
    for _ in range(args.n_null):
        perm = rng.permutation(G)
        shuffled.append(match_factors(Bs[0], Bs[1][:, perm]).mcc)
        random_null.append(
            match_factors(rng.normal(size=(d, G)), rng.normal(size=(d, G))).mcc
        )
    print(f"\nnulls over {args.n_null} replicates")
    print(f"  gene-shuffled   {np.mean(shuffled):.4f} "
          f"+/- {np.std(shuffled):.4f}")
    print(f"  random Gaussian {np.mean(random_null):.4f} "
          f"+/- {np.std(random_null):.4f}")

    gap = stab["mcc"] - np.mean(shuffled)
    sd = np.std(shuffled) if np.std(shuffled) > 0 else 1e-9
    print(f"\n  real - shuffled = {gap:+.4f}  ({gap/sd:.1f} null SD)")

    # ---- what the stable factors load on --------------------------------
    genes_path = args.models_dir / f"genes_{args.tags[0]}.npy"
    if genes_path.exists() and args.top_genes:
        genes = np.load(genes_path, allow_pickle=True)
        print(f"\ntop {args.top_genes} genes of the 3 most stable factors "
              "(reference run, sign-anchored on the largest loading):")
        for k in order[:3]:
            v = Bs[0][k]
            if v[np.argmax(np.abs(v))] < 0:
                v = -v
            top = np.argsort(-np.abs(v))[: args.top_genes]
            names = ", ".join(
                f"{genes[i]}{'-' if v[i] < 0 else '+'}" for i in top
            )
            print(f"  factor {k:>2}  |corr|={pair.correlations[k]:.3f}  {names}")


if __name__ == "__main__":
    main()
