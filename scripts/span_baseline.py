#!/usr/bin/env python3
"""Is the span residual large because B is wrong, or because the rows are noisy?

``r = ||x - x B^+ B|| / ||x||`` near 1 looks like failure and usually isn't. Most
rows are subsampled estimates whose norm is mostly estimation noise, and noise is
orthogonal to any d-dimensional subspace, so no rank-d basis can do better than
the noise floor. The number is only interpretable against baselines:

  PCA(d) on train   the best possible rank-d subspace for these rows. If B
                    matches it, B is at the floor and the residual is the data's,
                    not the model's.
  PCA(d) on test    an optimistic bound that has seen the test rows.
  random rank-d     the other end of the scale.

Reported per stratum as well, because the floor itself moves with precision:
full-data rows should reconstruct far better than 50-cell subsamples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract import paths
from extract.de import load_matrices
from extract.evaluation.span import _pca_basis, span_residual


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=paths.root() / "models")
    ap.add_argument("--tag", default="d24_a1_seed0")
    ap.add_argument("--matrices", type=Path, default=paths.matrices())
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="subsample for the PCA baselines; the SVD is the cost")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    B = np.load(args.models_dir / f"B_{args.tag}.npy")
    test_mask = np.load(args.models_dir / f"test_mask_{args.tag}.npy")
    d = B.shape[0]

    X, meta = load_matrices(args.matrices)
    Xs = X.to_numpy()            # used as-is: no per-gene centring or scaling
    del X

    test_idx = np.nonzero(test_mask)[0]
    train_idx = np.nonzero(~test_mask)[0]
    if len(test_idx) > args.max_rows:
        test_idx = rng.choice(test_idx, args.max_rows, replace=False)
    train_sub = rng.choice(train_idx, min(args.max_rows, len(train_idx)),
                           replace=False)

    Xtest, Xtrain = Xs[test_idx], Xs[train_sub]
    print(f"d={d}  test rows={len(test_idx):,}  train rows for PCA={len(train_sub):,}\n")

    bases = {
        "B (fitted)": B,
        f"PCA({d}) on train": _pca_basis(Xtrain, d),
        f"PCA({d}) on test": _pca_basis(Xtest, d),
        "random": rng.normal(size=(d, B.shape[1])),
    }
    residuals = {}
    print(f"{'basis':<22} {'median r':>9} {'q10':>7} {'q90':>7}   explained")
    for name, basis in bases.items():
        r = span_residual(basis, Xtest)
        residuals[name] = r
        print(f"{name:<22} {np.median(r):>9.4f} {np.quantile(r,.1):>7.4f} "
              f"{np.quantile(r,.9):>7.4f}   {1-np.median(r)**2:>6.1%}")

    gap = np.median(residuals["B (fitted)"]) - np.median(residuals[f"PCA({d}) on train"])
    print(f"\nB - PCA(train) = {gap:+.4f} median r")
    print("  <= 0 means B is at or below the best rank-d subspace for these rows,")
    print("  so the residual is the data's noise floor and not a defect in B.")

    # ---- the floor moves with precision -------------------------------
    strat = np.zeros(len(meta), dtype=int)
    labels = meta["label"].astype(str).to_numpy()
    for i, lab in enumerate(labels):
        if "__sub" in lab:
            strat[i] = int(lab.split("__sub")[-1]) + 1
    s_test = strat[test_idx]
    n_test = meta["n_cells"].to_numpy()[test_idx]

    print(f"\nby stratum (test rows, basis = B):")
    print(f"{'s':>3} {'rows':>7} {'median n_cells':>15} {'median r':>9}")
    rB = residuals["B (fitted)"]
    for s in np.unique(s_test):
        m = s_test == s
        print(f"{s:>3} {int(m.sum()):>7,} {np.median(n_test[m]):>15,.0f} "
              f"{np.median(rB[m]):>9.4f}")


if __name__ == "__main__":
    main()
