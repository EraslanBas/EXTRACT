#!/usr/bin/env python3
"""Frozen-axes diagnostic: does the discriminator objective prefer the true axes?

On a simulated screen (scripts/benchmark_identifiability_sim.py), B is frozen
at the true programs and at mixtures of them that span the same subspace;
only the label model and head are trained. If the objective identifies the
axes, validation accuracy should be highest at the truth and fall as the
mixture moves away from it (lower MCC to the truth).

Mixtures are M B_true with M = expm(t S), S a random skew-symmetric matrix:
t = 0 is the truth, larger t rotates further. The truth is also fitted with
several seeds to measure the seed-to-seed spread of accuracy.

    PYTHONPATH=src python scripts/diagnose_frozen_axes.py --structure multiplicative \
        --label-model product --out frozen_diag
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import benchmark_identifiability_sim as bench  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="frozen_diag")
    ap.add_argument("--family", default="laplace")
    ap.add_argument("--structure", default="multiplicative")
    ap.add_argument("--label-model", default="mlp", choices=["mlp", "product", "ammi"])
    ap.add_argument("--head", default="statistics", choices=["statistics", "distance"])
    ap.add_argument("--noise", default="isotropic", choices=["isotropic", "realistic"])
    ap.add_argument("--amp", type=float, default=3.0)
    ap.add_argument("--interaction", type=float, default=1.0)
    ap.add_argument("--angles", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--n-directions", type=int, default=3)
    ap.add_argument("--truth-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    import torch
    from extract.evaluation.stability import match_factors
    from extract.train import TrainConfig, fit
    bench.match_factors = match_factors
    torch.set_num_threads(a.threads)
    warnings.filterwarnings("ignore", category=UserWarning)

    sim = types.SimpleNamespace(genes=400, programs=6, perts=200, contexts=6, subsamples=4,
                                program_corr=0.0, active_frac=0.5, amp=a.amp, noise=a.noise,
                                interaction=a.interaction, gene_sd_spread=1.0, shared_noise=3,
                                shared_var=20.0)
    B, X, pi, ci, st, nc = bench.simulate(a.family, a.structure, 0, sim)
    tr, va = bench.split(pi, ci, sim.perts, sim.contexts, np.random.default_rng(1))
    D = sim.programs

    def run(basis, seed):
        cfg = TrainConfig(n_factors=D, epochs=a.epochs, eval_every=a.eval_every, patience=0,
                          select_on="accuracy", seed=seed, log_every=0, subspace="frozen",
                          label_model=a.label_model, head=a.head)
        with contextlib.redirect_stdout(io.StringIO()):
            _, hist = fit(X, pi, ci, st, nc, config=cfg, train_rows=tr, val_rows=va,
                          subspace_basis=basis.astype(np.float32))
        return max(h["val_accuracy"] for h in hist if "val_accuracy" in h)

    t0 = time.time()
    rows = [{"kind": "truth", "t": 0.0, "direction": -1, "seed": s, "mcc": 1.0,
             "val_acc": run(B, s)} for s in a.truth_seeds]
    rng = np.random.default_rng(7)
    for k in range(a.n_directions):
        S = rng.normal(size=(D, D)); S = (S - S.T) / 2
        S /= np.linalg.norm(S, 2)
        for t in a.angles:
            Bm = expm(t * S) @ B
            rows.append({"kind": "mixed", "t": t, "direction": k, "seed": 0,
                         "mcc": round(bench.mcc(B, Bm), 3), "val_acc": run(Bm, 0)})
            print(f"dir {k} t={t:<5g} mcc={rows[-1]['mcc']:.3f} val_acc={rows[-1]['val_acc']:.4f}", flush=True)

    truth = [r["val_acc"] for r in rows if r["kind"] == "truth"]
    mixed = [r for r in rows if r["kind"] == "mixed"]
    corr = float(np.corrcoef([r["mcc"] for r in mixed], [r["val_acc"] for r in mixed])[0, 1])
    far = [r["val_acc"] for r in mixed if r["mcc"] < 0.8]
    summary = {"truth_mean": float(np.mean(truth)), "truth_sd": float(np.std(truth)),
               "far_mean": float(np.mean(far)) if far else None,
               "gap_truth_minus_far": float(np.mean(truth) - np.mean(far)) if far else None,
               "corr_mcc_valacc": corr, "seconds": round(time.time() - t0)}
    print(f"\n{a.family}/{a.structure} noise={a.noise} label_model={a.label_model} head={a.head}")
    print(f"  truth val acc {summary['truth_mean']:.4f} +- {summary['truth_sd']:.4f} (seeds)")
    if far:
        print(f"  mixtures with MCC < 0.8: {summary['far_mean']:.4f}  gap {summary['gap_truth_minus_far']:+.4f}")
    print(f"  corr(MCC to truth, val acc) over mixtures: {corr:+.2f}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = (f"{a.noise}_amp{a.amp:g}_{a.family}_{a.structure}_lm-{a.label_model}"
           + ("" if a.head == "statistics" else f"_head-{a.head}"))
    (out / f"frozen_{tag}.json").write_text(json.dumps({"settings": vars(a), "summary": summary,
                                                        "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
