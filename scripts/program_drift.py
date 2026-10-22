#!/usr/bin/env python
"""Diagnostic only: how much the programs and validation AUC still move late in training.

For each model in a sweep folder, compares the loadings B at earlier evaluations with the final B
(matched |cos|, Hungarian) and reports the validation AUC plateau. Training always runs the full
number of epochs; nothing here is used to stop or shorten a fit.

    python scripts/program_drift.py SWEEP_DIR [--match PATTERN] [--epochs 300 500 700 900]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def matched_cos(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    C = np.abs(A @ B.T)
    r, c = linear_sum_assignment(-C)
    return C[r, c]


def drift(path, epochs):
    z = np.load(path)
    ep, Bs = z["epoch"], z["B"]
    out = {}
    for e in epochs:
        if e >= ep[-1]:
            continue
        m = matched_cos(Bs[np.argmin(np.abs(ep - e))], Bs[-1])
        out[f"cos_mean@{e}"] = m.mean()
        out[f"cos_min@{e}"] = m.min()
    return out


def auc_plateau(path):
    v = pd.read_csv(path).dropna(subset=["val_auc"])
    last = int(v.epoch.max())
    best = v.loc[v.val_auc.idxmax()]
    late = v[v.epoch > last - 100].val_auc.mean()
    early = v[(v.epoch > last - 300) & (v.epoch <= last - 200)].val_auc.mean()
    return {"last_epoch": last, "best_epoch": int(best.epoch), "best_auc": best.val_auc,
            "auc_gain_last200": late - early}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweep_dir")
    ap.add_argument("--match", default="", help="only tags containing this substring")
    ap.add_argument("--epochs", type=int, nargs="+", default=[300, 500, 700, 900])
    ap.add_argument("--out", help="optional CSV path")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.sweep_dir, "B_evals_*.npz"))):
        tag = os.path.basename(f)[len("B_evals_"):-len(".npz")]
        if args.match not in tag:
            continue
        row = {"tag": tag, **drift(f, args.epochs)}
        h = os.path.join(args.sweep_dir, f"history_{tag}.csv")
        if os.path.exists(h):
            row.update(auc_plateau(h))
        rows.append(row)
    if not rows:
        raise SystemExit("no B_evals_*.npz found")
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.precision", 4):
        print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
