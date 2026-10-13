#!/usr/bin/env python3
"""Validation sparsity term across training, for sweeps that did not log it.

The sparsity penalty depends only on the activities z, and z only on B, so it
can be recomputed from the loadings saved at every evaluation (B_evals_<tag>.npz)
on the validation rows -- the same split the sweep used. Writes
<sweep>/val_sparsity.parquet with columns tag, epoch, val_sparsity.

    python scripts/compute_val_sparsity.py --sweep $R/sweep_main_e1000 \
        --split-dir $R/splits/pair_seed0_ctxshuffle_pgenes
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract.data.splits import _pair_key, load_partition, split_val_per_perturbation
from extract.models import GlobalLoadings
from extract.train import prepare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--n-rows", type=int, default=5000, help="validation rows used (random subset)")
    ap.add_argument("--every", type=int, default=2, help="use every N-th evaluation")
    a = ap.parse_args()
    out = a.sweep / "val_sparsity.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["tag", "epoch", "val_sparsity"])
    done = set(have.tag)
    X, m, genes = load_partition(a.split_dir, "trainval", None, shuffled=False)
    val = _pair_key(m).isin(set(_pair_key(m)[split_val_per_perturbation(m, a.val_frac, a.val_seed)])).to_numpy()
    fa = prepare(X, m, genes)
    rows = np.random.default_rng(0).choice(np.nonzero(val)[0], min(a.n_rows, int(val.sum())), replace=False)
    x = torch.from_numpy(fa["X"][rows]); col = torch.from_numpy(fa["target_col"][rows])
    frames = [have]
    tags = sorted(p.stem[len("B_evals_"):] for p in a.sweep.glob("B_evals_*.npz"))
    for i, tag in enumerate(t for t in tags if t not in done):
        ev = np.load(a.sweep / f"B_evals_{tag}.npz")
        recs = []
        for k in range(0, len(ev["epoch"]), a.every):
            B = ev["B"][k]
            gl = GlobalLoadings(*B.shape, ridge=1e-4)
            with torch.no_grad():
                gl.B.copy_(torch.from_numpy(B))
                z = gl.project(x, col)
                s = float((z.abs() / (z.std(0) + 1e-8)).mean(0).sum())
            recs.append((tag, int(ev["epoch"][k]), s))
        frames.append(pd.DataFrame(recs, columns=["tag", "epoch", "val_sparsity"]))
        if i % 20 == 0:
            print(f"{i + 1} models", flush=True)
            pd.concat(frames, ignore_index=True).to_parquet(out)
    pd.concat(frames, ignore_index=True).to_parquet(out)
    print("done", out)


if __name__ == "__main__":
    main()
