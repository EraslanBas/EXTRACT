#!/usr/bin/env python3
"""Small probe: what are L_disc and L_recon actually worth, and at what alpha
do they balance?

Deliberately cheap -- a few contexts, small K, few epochs, one seed -- because
the question is about SCALES, not about which model wins. It loads the split
once and reports, per (d, alpha):

  disc, recon         at epoch 0 and at the end, train and val
  alpha * recon       what the reconstruction term is actually worth
  ratio               (alpha * recon) / disc -- 1.0 means balanced
  recon_drop          how far recon fell; ~0 means the term is inert
  unexplained         recon / mean-squared-logFC -- fraction of signal left

The last column is the scale-free reading: recon is per-entry MSE, so dividing
by the mean squared logFC of the same rows turns it into "fraction of signal
not captured", which is comparable across gene sets, d and K.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from module_finder import paths
from module_finder.data.splits import (_pair_key, load_partition,
                                       split_val_per_perturbation,
                                       subsample_mask, thin_synthetic)
from module_finder.models.loadings import NO_MASK
from module_finder.train import TrainConfig, fit, prepare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", type=Path,
                    default=paths.root() / "splits" / "pair_seed0")
    ap.add_argument("--out-dir", type=Path,
                    default=paths.root() / "probe_loss_scale")
    ap.add_argument("--contexts", nargs="+",
                    default=["DMSO_round2", "Stattic", "CHIR-98014", "Romidepsin"])
    ap.add_argument("--d", nargs="+", type=int, default=[4, 8, 14])
    ap.add_argument("--alpha", nargs="+", type=float,
                    default=[1, 10, 100, 1000, 10000])
    ap.add_argument("--n-subsamples", "-K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--shuffled-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X, meta, genes = load_partition(args.split_dir, "trainval", args.contexts,
                                    shuffled=True, shuffle_seed=0)
    real = meta.is_real.to_numpy()
    val_real = split_val_per_perturbation(meta[real], 0.1, 0)
    val_pairs = set(_pair_key(meta[real])[val_real])
    val = _pair_key(meta).isin(val_pairs).to_numpy()
    keep = subsample_mask(meta, args.n_subsamples, seed=0) | val
    keep &= thin_synthetic(meta, args.shuffled_frac, seed=0)
    X, meta, val = X[keep], meta[keep].reset_index(drop=True), val[keep]
    is_real = meta.is_real.to_numpy(); train = ~val
    print(f"{len(args.contexts)} contexts, K={args.n_subsamples}: "
          f"{int((train & is_real).sum()):,} train + {int((val & is_real).sum()):,} val "
          f"measured, {int((~is_real).sum()):,} synthetic  ({time.time()-t0:.0f}s)",
          flush=True)

    fit_args = prepare(X, meta, genes)
    fit_args.pop("perturbation_levels"); fit_args.pop("context_levels")
    del X

    # scale-free reference: mean squared logFC over the MEASURED rows, with the
    # on-target entry excluded exactly as squared_error excludes it
    Xm = fit_args["X"][is_real]; col = fit_args["target_col"][is_real]
    sq = (Xm ** 2).sum(); n = Xm.size
    m = col != NO_MASK
    sq -= (Xm[np.nonzero(m)[0], col[m]] ** 2).sum(); n -= int(m.sum())
    ms = float(sq / n)
    print(f"mean squared logFC (masked, measured rows): {ms:.6f}\n", flush=True)

    rows = []
    for d in args.d:
        for a in args.alpha:
            cfg = TrainConfig(n_factors=d, alpha=a, seed=args.seed,
                              epochs=args.epochs, batch_size=args.batch_size,
                              log_every=0, eval_every=max(1, args.epochs // 4),
                              patience=0)
            t = time.time()
            model, hist = fit(config=cfg, train_rows=train, val_rows=val,
                              is_real=is_real, **fit_args)
            h = pd.DataFrame(hist)
            h.to_csv(args.out_dir / f"history_d{d}_a{a:g}.csv", index=False)
            first, last = h.iloc[0], h.iloc[-1]
            v = h.dropna(subset=["val_accuracy"]).iloc[-1] if "val_accuracy" in h else last
            rows.append({
                "d": d, "alpha": a,
                "disc_0": first.disc, "disc_end": last.disc,
                "recon_0": first.recon, "recon_end": last.recon,
                "alpha_recon_end": a * last.recon,
                "ratio_end": a * last.recon / max(last.disc, 1e-12),
                "recon_drop_pct": 100 * (first.recon - last.recon) / max(first.recon, 1e-12),
                "unexplained_end": last.recon / ms,
                "train_acc": last.accuracy,
                "val_acc": float(v.get("val_accuracy", np.nan)),
                "seconds": round(time.time() - t, 1),
            })
            r = rows[-1]
            print(f"  d={d:<3} a={a:<7g} disc {r['disc_end']:.4f}  recon {r['recon_end']:.6f}  "
                  f"a*recon {r['alpha_recon_end']:.4f}  ratio {r['ratio_end']:.3f}  "
                  f"drop {r['recon_drop_pct']:+.1f}%  unexpl {r['unexplained_end']:.3f}  "
                  f"val {r['val_acc']:.3f}  {r['seconds']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(args.out_dir / "probe.csv", index=False)
    print(f"\nwrote {args.out_dir}/probe.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
