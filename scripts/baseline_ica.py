#!/usr/bin/env python3
"""Linear ICA baseline on the EXTRACT split, scored on the test pairs.

    python scripts/baseline_ica.py --d 4 6 8 ... 30 --seeds 0 1 2

Fits ``baselines.linear_ica`` (PCA whitening to max(d, 50) dimensions, then
FastICA to d components) on the **measured** rows of ``<split>/trainval`` --
train and validation pairs, full-data rows and every subsample, i.e. the rows a
K = 10 EXTRACT model sees. Synthetic rows are not used. Each row's on-target
entry is set to zero for the fit, since ICA cannot mask it the way EXTRACT does.

Every fit is then scored on the measured rows of ``<split>/test`` with
``extract.evaluation.reconstruction.reconstruction_metrics`` -- the same
function used for EXTRACT -- and the loadings are saved, so factor-level
comparisons (cross-seed stability, matching to EXTRACT) can follow.
The split is the one EXTRACT is fitted on; nothing is redrawn.
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
from extract.data.splits import load_partition
from extract.evaluation.reconstruction import reconstruction_metrics
from extract.models.loadings import NO_MASK
from extract.train import prepare
from sklearn.decomposition import PCA, FastICA


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", type=Path,
                    default=paths.root() / "splits" / "pair_seed0_ctxshuffle")
    ap.add_argument("--out-dir", type=Path, default=paths.root() / "baselines" / "ica")
    ap.add_argument("--d", type=int, nargs="+", default=list(range(4, 31, 2)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-whiten", type=int, default=None,
                    help="PCA dimension before ICA; default max(d, 50)")
    ap.add_argument("--max-iter", type=int, default=1000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    Xtv, mtv, genes = load_partition(args.split_dir, "trainval", shuffled=False)
    Xte, mte, _ = load_partition(args.split_dir, "test", shuffled=False)
    ftv = prepare(Xtv, mtv, genes)
    levels = dict(perturbation_levels=ftv["perturbation_levels"],
                  context_levels=ftv["context_levels"])
    fte = prepare(Xte, mte, genes, **levels)
    Xfit = Xtv.astype(np.float64)
    tc = ftv["target_col"]
    r = np.nonzero(tc != NO_MASK)[0]
    Xfit[r, tc[r]] = 0.0                       # on-target entries out of the fit
    print(f"fit on {len(Xfit):,} trainval measured rows; test {len(Xte):,} rows "
          f"({time.time() - t0:.0f}s)", flush=True)

    # The PCA whitening step of baselines.linear_ica is the same for every
    # d <= n_whiten, so it is fitted once and reused; each (d, seed) then runs
    # FastICA on the whitened scores, exactly as FastICABaseline does.
    n_whiten = args.n_whiten or max(max(args.d), 50)
    t = time.time()
    pca = PCA(n_components=n_whiten, random_state=0).fit(Xfit)
    scores = pca.transform(Xfit)
    np.savez(args.out_dir / "pca_whitening.npz", components=pca.components_,
             mean=pca.mean_, explained_variance=pca.explained_variance_)
    print(f"PCA whitening to {n_whiten} dims ({time.time() - t:.0f}s)", flush=True)

    out_csv = args.out_dir / "ica_test_metrics.csv"
    done = set(pd.read_csv(out_csv).tag) if out_csv.exists() else set()
    for d in args.d:
        for seed in args.seeds:
            tag = f"ica_d{d}_seed{seed}"
            if tag in done:
                continue
            t = time.time()
            ica = FastICA(n_components=d, max_iter=args.max_iter, random_state=seed,
                          whiten="unit-variance").fit(scores)
            B = (ica.mixing_.T @ pca.components_).astype(np.float32)   # factor -> gene
            np.save(args.out_dir / f"B_{tag}.npy", B)
            row = {"tag": tag, "method": "ica", "d": d, "seed": seed,
                   "n_whiten": n_whiten, "n_iter": int(ica.n_iter_),
                   "seconds": round(time.time() - t, 1),
                   **reconstruction_metrics(B, Xte, fte["target_col"], fte["stratum"])}
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(),
                                       index=False)
            print(f"{tag:16s} test R2 {row['R2_global']:.4f}  row-median {row['R2_row_median']:.4f}  "
                  f"gene-median {row['R2_gene_median']:.4f}  ({row['seconds']:.0f}s, "
                  f"{row['n_iter']} iters)", flush=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    with open(args.split_dir / "TEST_READS.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tbaseline ICA, d={args.d}, "
                 f"seeds={args.seeds}\t{out_csv}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
