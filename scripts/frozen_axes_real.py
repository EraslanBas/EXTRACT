#!/usr/bin/env python3
"""Frozen-axes test on the real screen: does EXTRACT's objective prefer the
reproducible ICA axes over rotations of them?

B is frozen; only the label network and head are trained. Arms (24, run one
per job-array task with --arm i): data {nested, disjoint} x head variance
{shared, label} x frozen basis {V (PCA of pair means), ICA axes in span(V),
ICA axes from another ICA seed, ICA axes mixed by expm(t S) with t = 0.5, 1, 2}.
ICA is computed on the training pair means only. Everything else as in the
current sweep (distance head, MLP label network 64x8 with data-based starts,
label regularisation). Saves history, config, full model and a summary row.

    python scripts/frozen_axes_real.py --arm 0 --out-dir $R/frozen_axes_real
"""
from __future__ import annotations
import argparse, dataclasses, itertools, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.linalg import expm
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract import paths
from extract.data.noise import pair_means, pca_subspace
from extract.data.splits import (_pair_key, load_partition, split_val_per_perturbation,
                                 subsample_mask, thin_synthetic)
from extract.train import TrainConfig, control_mean_embeddings, fit, prepare

R = Path("/large_storage/ctc/beraslan/ModuleFinder")
SPLITS = {"nested": R / "splits/pair_seed0_ctxshuffle_pgenes",
          "disjoint": R / "splits/pair_seed0_ctxshuffle_pgenes_disjoint"}
BASES = ["pca", "ica", "ica_seed1", "mix0.5", "mix1", "mix2"]
ARMS = list(itertools.product(SPLITS, ["shared", "label"], BASES))


def ica_axes(means, V, seed):
    from sklearn.decomposition import FastICA
    Q = np.linalg.qr(V.T.astype(np.float64))[0]
    f = FastICA(V.shape[0], whiten="unit-variance", max_iter=3000, random_state=seed).fit(means @ Q)
    A = f.mixing_.T @ Q.T
    return (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=R / "frozen_axes_real")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    data, hvar, basis = ARMS[a.arm]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{data}_{hvar}_{basis}"
    warnings.filterwarnings("ignore")
    t0 = time.time()
    levels = (0.2, 0.4, 0.6, 0.8, 1.0)
    X, m, genes = load_partition(SPLITS[data], "trainval", None, shuffled=True, shuffle_seed=0, shuffle_levels=levels)
    real = m.is_real.to_numpy()
    vp = set(_pair_key(m[real])[split_val_per_perturbation(m[real], 0.1, 0)])
    val_all = _pair_key(m).isin(vp).to_numpy()
    keep = (subsample_mask(m, 10, seed=0) | val_all) & thin_synthetic(m, 0.5, seed=0)
    X, m, val = X[keep], m[keep].reset_index(drop=True), val_all[keep]
    is_real = m.is_real.to_numpy()
    pair = (m.perturbation.astype(str) + "|" + m.context.astype(str)).to_numpy()
    train_real = is_real & ~val
    V, _ = pca_subspace(X, pair, train_real, 26, seed=0)
    if basis == "pca":
        B = V
    else:
        _, means, _ = pair_means(X, pair, train_real)
        B = ica_axes(means, V, seed=1 if basis == "ica_seed1" else 0)
        if basis.startswith("mix"):
            S = np.random.default_rng(7).normal(size=(26, 26)); S = (S - S.T) / 2; S /= np.linalg.norm(S, 2)
            B = (expm(float(basis[3:]) * S) @ B).astype(np.float32)
    fa = prepare(X, m, genes)
    plev, clev = fa.pop("perturbation_levels"), fa.pop("context_levels")
    ctx = control_mean_embeddings(pd.read_parquet(R / "context_control_means.parquet"), clev, 8)
    cfg = TrainConfig(n_factors=26, seed=0, epochs=a.epochs, batch_size=1024, device=a.device, log_every=0,
                      eval_every=3, patience=0, select_on="accuracy", subspace="frozen",
                      head="distance", head_variance=hvar, label_model="mlp",
                      label_weight_decay=0.01, label_dropout=0.2, embedding_dim=64,
                      context_embedding_dim=8, pert_embedding_init="response")
    model, hist = fit(config=cfg, train_rows=~val, val_rows=val, is_real=is_real,
                      synth_level=m.shuffle_frac.to_numpy(dtype=float), subspace_basis=B,
                      context_embedding_init=ctx, **fa)
    H = pd.DataFrame(hist); H.to_csv(a.out_dir / f"history_{tag}.csv", index=False)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, a.out_dir / f"model_best_{tag}.pt")
    torch.save(model.final_state_, a.out_dir / f"model_final_{tag}.pt")
    np.save(a.out_dir / f"B_{tag}.npy", B)
    (a.out_dir / f"config_{tag}.json").write_text(json.dumps(
        {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(cfg).items()},
         "selected_epoch": int(model.selected_epoch_), "n_genes": int(fa["X"].shape[1]),
         "n_perturbations": int(fa["perturbation_idx"].max()) + 1, "n_contexts": int(fa["context_idx"].max()) + 1,
         "data": data, "basis": basis, "split_dir": str(SPLITS[data])}, indent=1, default=str))
    np.save(a.out_dir / f"levels_{tag}.npy", np.array([plev, clev], dtype=object), allow_pickle=True)
    v = H.dropna(subset=["val_accuracy"])
    best = v.loc[v.val_accuracy.idxmax()]
    row = {"tag": tag, "data": data, "head_variance": hvar, "basis": basis,
           "best_val_accuracy": float(best.val_accuracy), "val_auc_at_best": float(best.val_auc),
           "min_val_disc": float(v.val_disc.min()), "selected_epoch": int(best.epoch),
           "minutes": round((time.time() - t0) / 60, 1)}
    pd.DataFrame([row]).to_csv(a.out_dir / f"summary_{tag}.csv", index=False)
    print(row, flush=True)


if __name__ == "__main__":
    main()
