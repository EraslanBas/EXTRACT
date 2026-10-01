#!/usr/bin/env python3
"""Build the per-model table from a sweep's per-epoch histories.

Two outputs:

``epochs_long.csv``   every model x epoch: the settings, then train and val
                      total / disc / recon / recon_measured / recon_synth /
                      accuracy. This is the raw curve table.

``model_summary.csv`` one row per model: the settings, the SELECTED epoch
                      (max val accuracy, ``*_sel``) and the metrics there,
                      the val-loss OVERFITTING ONSET (diagnostic), plus the
                      final-epoch values.

The onset is the epoch minimising validation total loss -- the point past
which val rises while train keeps falling. It is read off the recorded curve
rather than assumed, and the row carries the evidence:

  onset_epoch        argmin of val_total
  val_total_onset    its value
  train_total_onset  train total at the same epoch
  train_still_falling   train total at the LAST epoch < at the onset, i.e. the
                     model really did keep fitting after val turned
  val_rise           val_total(last) - val_total(onset); how far it overfit
  n_after_onset      epochs observed past the onset (0 = the curve never
                     turned and the onset is just the last point, so the run
                     was stopped too early to see it)
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd

TAG = re.compile(r"history_d(?P<d>\d+)_a(?P<alpha>[\d.eE+-]+)_b(?P<beta>[\d.eE+-]+)"
                 r"_K(?P<K>\d+)_seed(?P<seed>\d+)\.csv$")
CURVE = ["total", "disc", "recon", "recon_measured", "recon_synth", "accuracy"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path,
                    help="a sweep directory, or a parent of shard directories "
                         "(history_*.csv is searched one level down too)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or args.sweep_dir

    files = sorted(list(args.sweep_dir.glob("history_*.csv"))
                   + list(args.sweep_dir.glob("*/history_*.csv")))
    if not files:
        raise SystemExit(f"no history_*.csv under {args.sweep_dir}")
    names = [f.name for f in files]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"{len(dup)} tags appear in more than one shard, e.g. {dup[0]}")
    print(f"{len(files)} models")

    long_rows, summary, stale = [], [], []
    for f in files:
        m = TAG.search(f.name)
        if not m:
            print(f"  skip (unparsed name): {f.name}")
            continue
        cfg = {"d": int(m["d"]), "alpha": float(m["alpha"]),
               "beta": float(m["beta"]), "K": int(m["K"]),
               "seed": int(m["seed"]), "tag": f.name[8:-4]}
        h = pd.read_csv(f)
        if "val_total" not in h:
            # written by code that predates total-loss selection -- not a
            # result of this protocol; count it, never report it
            stale.append(cfg["tag"])
            continue

        keep = ["epoch"] + [c for c in CURVE if c in h]
        keep += [f"val_{c}" for c in CURVE if f"val_{c}" in h]
        cur = h[keep].copy()
        for k, v in cfg.items():
            cur[k] = v
        long_rows.append(cur)

        ev = h.dropna(subset=["val_total"]) if "val_total" in h else pd.DataFrame()
        row = dict(cfg, epochs_run=len(h))
        if len(ev):
            i = int(ev.val_total.idxmin())
            onset = h.loc[i]
            last = h.iloc[-1]
            last_ev = ev.iloc[-1]
            row.update(
                onset_epoch=int(onset.epoch),
                val_total_onset=float(onset.val_total),
                train_total_onset=float(onset.total),
                val_total_last=float(last_ev.val_total),
                train_total_last=float(last.total),
                val_rise=float(last_ev.val_total - onset.val_total),
                train_fall_after=float(onset.total - last.total),
                train_still_falling=bool(last.total < onset.total),
                n_after_onset=int(len(ev[ev.epoch > onset.epoch])),
                val_accuracy_onset=float(onset.get("val_accuracy", np.nan)),
                val_recon_measured_onset=float(onset.get("val_recon_measured", np.nan)),
                val_recon_synth_onset=float(onset.get("val_recon_synth", np.nan)),
                val_disc_onset=float(onset.get("val_disc", np.nan)),
            )
            # the SELECTED epoch: max val accuracy (what fit restored and what
            # B_<tag>.npy holds). The onset above is kept as a diagnostic only.
            j = int(ev.val_accuracy.idxmax())
            sel = h.loc[j]
            row.update(
                sel_epoch=int(sel.epoch),
                val_accuracy_sel=float(sel.val_accuracy),
                val_disc_sel=float(sel.get("val_disc", np.nan)),
                val_recon_measured_sel=float(sel.get("val_recon_measured", np.nan)),
                val_recon_synth_sel=float(sel.get("val_recon_synth", np.nan)),
                train_accuracy_sel=float(sel.get("accuracy", np.nan)),
                n_after_sel=int(len(ev[ev.epoch > sel.epoch])),
            )
        summary.append(row)

    long = pd.concat(long_rows, ignore_index=True)
    cols = ["tag", "d", "alpha", "beta", "K", "seed", "epoch"] + \
           [c for c in long.columns if c not in
            ("tag", "d", "alpha", "beta", "K", "seed", "epoch")]
    long[cols].to_csv(out / "epochs_long.csv", index=False)
    s = pd.DataFrame(summary).sort_values(["d", "alpha", "beta", "K", "seed"])
    s.to_csv(out / "model_summary.csv", index=False)

    # ---- choosing a config ACROSS runs ------------------------------------
    # val_total is right for picking the epoch WITHIN a run (alpha and beta are
    # fixed there) and wrong for comparing runs: its -alpha*beta*recon_synth
    # term makes it fall with alpha*beta whatever the model does, so the
    # "best" config by val_total is just the largest alpha*beta. Compare runs
    # on scale-free quantities read at each run's own SELECTED epoch instead.
    # Ranking: val accuracy FIRST (higher is better), val measured-row recon
    # second (lower is better) as the tie-breaker. val_disc is not used: it
    # rises with over-confidence even while accuracy improves.
    A, B = "val_accuracy_sel", "val_recon_measured_sel"
    if A in s and B in s:
        c = s.dropna(subset=[A, B]).copy()
        a, b = -c[A].to_numpy(), c[B].to_numpy()   # both "lower is better"
        # Pareto front: no other config is at least as good on both and
        # strictly better on one
        dominated = [bool(((a <= a[i]) & (b <= b[i]) & ((a < a[i]) | (b < b[i]))).any())
                     for i in range(len(c))]
        c["pareto"] = ~np.array(dominated)
        c["rank_accuracy"] = c[A].rank(method="min", ascending=False)
        c["rank_recon"] = c[B].rank(method="min")
        c = c.sort_values([A, B], ascending=[False, True])
        c.to_csv(out / "best_configs.csv", index=False)
        print(f"  wrote {out}/best_configs.csv  ({int(c.pareto.sum())} on the Pareto front)")
        print("  (selection is on validation only; read test ONCE for the chosen "
              "config with scripts/read_test.py)")

    print(f"  wrote {out}/epochs_long.csv    {len(long):,} rows")
    print(f"  wrote {out}/model_summary.csv  {len(s):,} models")
    if stale:
        print(f"  SKIPPED {len(stale)} stale histories (no val_total), e.g. {stale[:2]}")
    if "train_still_falling" in s:
        ok = s.train_still_falling.sum()
        never = (s.n_after_onset == 0).sum()
        print(f"\n  {ok}/{len(s)} models overfit by the definition "
              f"(val turned up while train kept falling)")
        print(f"  {never}/{len(s)} never turned -- stopped too early to see it")
        best = s.loc[s.val_accuracy_sel.idxmax()]
        print(f"\n  highest val accuracy: {best.tag}  "
              f"epoch {best.sel_epoch}  val acc {best.val_accuracy_sel:.4f}")
        print(f"  {(s.n_after_sel == 0).sum()}/{len(s)} peaked at the last "
              f"evaluation -- still improving when stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
