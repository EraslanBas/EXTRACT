#!/usr/bin/env python3
"""Full-factorial sweep over (d, alpha, beta, K, seed) on a saved split.

Loads ``<split>/trainval`` (and ``test``) ONCE and reuses them for every cell;
calling fit_model.py per cell would re-read ~11 GB each time. Resumable: a cell
whose row is already in ``sweep_metrics.csv`` is skipped, so the job can be
killed and restarted.

Per cell it records BOTH loss terms on all three partitions, because they
diagnose different failures:
  recon flat across epochs        alpha too small to anchor the subspace
  disc falling, val disc rising   the head is overfitting
  recon low, disc at chance       reconstructs but never oriented

TEST DISCIPLINE. Test metrics are recorded for every cell, but selection must
use validation only -- see the selection rule in docs/HANDOFF.md. Choosing a
cell on its test column turns test into a second validation set and its numbers
stop being held out.
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract import paths
from extract.data.splits import (_pair_key, load_partition,
                                       split_val_per_perturbation,
                                       subsample_mask, thin_synthetic)
from extract.objectives import StratifiedNegativeSampler
from extract.train import TrainConfig, evaluate, fit, prepare

METRICS = ("disc", "recon", "recon_measured", "recon_synth", "accuracy",
           "synth_accuracy", "disc_synth", "span_residual_median",
           "real_score_mean", "fake_score_mean")


class _Queue:
    """A work queue that lives in the sweep directory itself.

    claims/<tag>.claim   created with O_CREAT|O_EXCL, so exactly one worker
                         gets a cell; holds host, pid and Slurm step
    cells/<tag>.csv      the finished cell's metrics row (written atomically);
                         a cell is DONE iff this file exists
    sweep_metrics.csv    rebuilt from cells/ under .metrics.lock after each
                         cell, so readers always see a consistent table

    A worker that dies leaves its claim behind. ``clear_stale`` (run at each
    worker's startup, serialised by the same lock) removes claims whose
    process is gone on this host or whose Slurm step no longer exists.
    """

    def __init__(self, root: Path):
        import os, socket
        self.root = root
        self.claims = root / "claims"; self.claims.mkdir(exist_ok=True)
        self.cells = root / "cells"; self.cells.mkdir(exist_ok=True)
        self.lock = root / ".metrics.lock"
        job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
        self.me = {"host": socket.gethostname(), "pid": os.getpid(),
                   "step": f"{job}.{step}" if job and step else None}

    def _locked(self):
        import contextlib, fcntl

        @contextlib.contextmanager
        def cm():
            with open(self.lock, "w") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                yield
        return cm()

    def done(self) -> set[str]:
        return {p.stem for p in self.cells.glob("*.csv")}

    def claim(self, tag: str) -> bool:
        import os
        try:
            fd = os.open(self.claims / f"{tag}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, json.dumps(self.me).encode()); os.close(fd)
        return True

    def finish(self, tag: str, row: dict) -> None:
        tmp = self.cells / f".{tag}.tmp"
        pd.DataFrame([row]).to_csv(tmp, index=False)
        tmp.rename(self.cells / f"{tag}.csv")
        with self._locked():
            rows = [pd.read_csv(p) for p in sorted(self.cells.glob("*.csv"))]
            out = self.root / "sweep_metrics.csv"
            tmpm = self.root / ".sweep_metrics.tmp"
            pd.concat(rows, ignore_index=True).to_csv(tmpm, index=False)
            tmpm.rename(out)
        (self.claims / f"{tag}.claim").unlink(missing_ok=True)

    def _alive(self, c: dict) -> bool:
        import os, subprocess
        if c.get("host") == self.me["host"]:
            try:
                os.kill(int(c["pid"]), 0)
                return True
            except (ProcessLookupError, PermissionError, ValueError):
                return False
        if c.get("step"):
            job = c["step"].split(".")[0]
            r = subprocess.run(["squeue", "-h", "-s", "-j", job, "-o", "%i"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return True        # cannot tell: keep the claim
            return c["step"] in r.stdout.split()
        return True                # other host, no step: cannot tell

    def clear_stale(self) -> None:
        with self._locked():
            done = self.done()
            for p in self.claims.glob("*.claim"):
                try:
                    c = json.loads(p.read_text())
                except (OSError, ValueError):
                    c = {}
                if p.stem in done or not self._alive(c):
                    print(f"queue: clearing stale claim {p.stem} ({c})", flush=True)
                    p.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", type=Path, default=paths.root()/"splits"/"pair_seed0")
    ap.add_argument("--out-dir", type=Path, default=paths.root()/"sweep_grid")
    ap.add_argument("--d", nargs="+", type=int, default=[2,4,6,8,10,12,14])
    ap.add_argument("--alpha", nargs="+", type=float,
                    default=[500,700,900,1000,1500,2000,3000])
    ap.add_argument("--beta", nargs="+", type=float, default=[0.0,0.25,0.5,1.0])
    ap.add_argument("-K","--n-subsamples", nargs="+", type=int, default=[1,2,3,5,7,10])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-epochs", type=int, default=100,
                    help="never stop before this; the best epoch is still restored")
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--shuffled-frac", type=float, default=0.5)
    ap.add_argument("--shuffle-levels", type=float, nargs="+", default=None,
                    help="graded synthetic rows: take each row's twin from one of "
                         "these gene-shuffle fractions (e.g. 0.2 0.4 0.6 0.8 1.0); "
                         "the number of synthetic rows is unchanged, validation "
                         "and test report accuracy per level. Default: 1.0 only")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--draw-seed", type=int, default=0)
    ap.add_argument("--select-on", default="accuracy",
                    choices=["total", "accuracy", "disc", "recon"],
                    help="validation metric early stopping selects on. "
                         "accuracy (default): val BCE rises from epoch ~0-3 "
                         "as the head grows over-confident while val "
                         "accuracy keeps climbing for 100 epochs, so "
                         "'total' picks near-untrained models. B is saved "
                         "at every evaluation (B_evals_<tag>.npz) so the "
                         "epoch can be re-chosen under another rule")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--queue", action="store_true",
                    help="shared work queue: any number of workers (GPU or "
                         "CPU, any host) may point at the SAME --out-dir; each "
                         "cell is claimed atomically (claims/<tag>.claim) and "
                         "its metrics row written to cells/<tag>.csv. Cells "
                         "run in descending K (longest first); a worker only "
                         "takes the K values it was given")
    ap.add_argument("--subspace", nargs="+", default=["fixed"],
                    choices=["free", "fixed", "anchored", "frozen"],
                    help="how span(B) is set; a grid dimension. fixed / anchored use "
                         "the label-driven subspace from the subsample noise "
                         "(extract.data.noise); tags get a _<mode> suffix")
    ap.add_argument("--basis", nargs="+", default=None, metavar="NAME=PATH",
                    help="external axes for fixed / frozen, e.g. ica=B_ica_d26.npy "
                         "(a [d, G] .npy); a grid dimension. Default: the "
                         "label-driven subspace V. Tags get _<mode>_<name>")
    ap.add_argument("--head-basis", nargs="+", default=["linear,square,abs,tanh"],
                    help="statistics the head applies to each factor, one comma-"
                         "separated set per grid value, e.g. abs  linear,abs. "
                         "Non-default sets add _head-<names> to the tag")
    ap.add_argument("--axes-init", default="identity", choices=["identity", "random"],
                    help="fixed mode: start A at the identity or at a random rotation "
                         "(per seed); random adds _Arandom to the tag")
    ap.add_argument("--label-weight-decay", type=float, default=0.0,
                    help="AdamW weight decay on the label network only; adds _lwd<x>")
    ap.add_argument("--label-dropout", type=float, default=0.0,
                    help="dropout in the MLP label network; adds _ldo<x>")
    ap.add_argument("--embedding-dim", type=int, default=64, help="perturbation embedding size")
    ap.add_argument("--context-embedding-dim", type=int, default=16, help="context embedding size")
    ap.add_argument("--pert-embedding-init", choices=["random", "response"], default="random",
                    help="start perturbation embeddings at random or from how each target "
                         "gene responds across the training pairs; response adds _embresp")
    ap.add_argument("--sparsity", type=float, default=0.0,
                   help="weight of the activity-sparsity penalty (0 = off); adds _sparse<w> to the tag")
    ap.add_argument("--head", default="statistics", choices=["statistics", "distance"],
                   help="discriminator head: per-factor statistics (default) or the distance "
                        "between measured and predicted activity; distance adds _dist to the tag")
    ap.add_argument("--label-model", default="mlp", choices=["mlp", "product", "ammi"],
                    help="label -> lambda map: unconstrained MLP, the structured "
                         "product f(p) * g(c), or ammi a(p) + b(c) + f(p) * g(c); "
                         "non-mlp adds _<label-model> to the tag")
    ap.add_argument("--subspace-method", choices=["rca", "snr", "pca"], default="rca",
                    help="how V is computed: reliable components analysis (default), "
                         "signal over the within-pair noise model, or PCA of the pair means; "
                         "adds _rca<rank> / _snr / _pca to the tag")
    ap.add_argument("--free-init", choices=["random", "basis"], default="random",
                    help="free mode: B starts at random, or where the fixed model starts "
                         "(A0 V, with A0 from --axes-init); basis adds _initV<method> to the tag")
    ap.add_argument("--rca-rank", type=int, default=100,
                    help="RCA: R_W truncated to its top RANK eigen-directions")
    ap.add_argument("--noise-rank", type=int, default=50,
                    help="rank of the correlated-noise part of the noise model")
    ap.add_argument("--noise-max-rows", type=int, default=150_000,
                    help="subsample residuals used to fit the noise model")
    ap.add_argument("--no-test", action="store_true",
                    help="skip the test read entirely (strictest discipline)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv = args.out_dir / "sweep_metrics.csv"

    # ONE writer per directory. Two sweeps writing the same directory each
    # read the done-set once, never see each other, rerun the same cells and
    # overwrite each other's history files. That happened: a step that
    # survived a "kill" of its local srun client kept writing for 12 h into the
    # restarted run's directory, clobbering 72 cells. Refuse to start instead.
    import fcntl, os
    if args.queue:
        queue = _Queue(args.out_dir)
        queue.clear_stale()
    else:
        queue = None
        lock_path = args.out_dir / ".sweep.lock"
        lock_fh = open(lock_path, "w")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                f"{args.out_dir} is locked by another sweep ({lock_path}); "
                "stop it first -- with `scancel <jobid>.<step>`, since killing "
                "the local srun client does not stop an --overlap step")
        lock_fh.write(f"{os.getpid()} {os.uname().nodename}\n"); lock_fh.flush()
    done = set()
    if queue is not None:
        done = queue.done()
        print(f"queue: {len(done)} cells already done in {args.out_dir}", flush=True)
    elif csv.exists():
        prev = pd.read_csv(csv)
        done = set(prev.tag)
        print(f"resuming: {len(done)} cells already done", flush=True)

    # ---------------- load once --------------------------------------
    t0 = time.time()
    levels = tuple(args.shuffle_levels) if args.shuffle_levels else None
    Xtv, mtv, genes = load_partition(args.split_dir, "trainval", None,
                                     shuffled=True, shuffle_seed=0,
                                     shuffle_levels=levels)
    real = mtv.is_real.to_numpy()
    val_real = split_val_per_perturbation(mtv[real], args.val_frac, args.val_seed)
    val_pairs = set(_pair_key(mtv[real])[val_real])
    val_all = _pair_key(mtv).isin(val_pairs).to_numpy()
    print(f"trainval {len(mtv):,} rows x {len(genes):,} genes, "
          f"{len(val_pairs):,} val pairs  ({time.time()-t0:.0f}s)", flush=True)

    Xte = mte = None
    if not args.no_test:
        Xte, mte, _ = load_partition(args.split_dir, "test", None,
                                     shuffled=True, shuffle_seed=0,
                                     shuffle_levels=levels)
        kte = thin_synthetic(mte, args.shuffled_frac, seed=args.draw_seed)
        Xte, mte = Xte[kte], mte[kte].reset_index(drop=True)
        print(f"test {len(mte):,} rows", flush=True)

    # ---- noise model and label-driven subspaces (train pairs only) -----
    noise_model, bases = None, {}
    if set(args.subspace) - {"free"} or args.free_init == "basis":
        from extract.data.noise import (label_subspace, noise_covariance, pair_means,
                                        pca_subspace, rca_subspace, within_pair_residuals)
        t1 = time.time()
        train_real = real & ~val_all
        pair = (mtv.perturbation.astype(str) + "|" + mtv.context.astype(str)).to_numpy()
        if args.subspace_method == "snr" or "anchored" in args.subspace:
            R = within_pair_residuals(Xtv, pair, train_real, max_rows=args.noise_max_rows,
                                      seed=args.draw_seed)
            noise_model = noise_covariance(R, rank=args.noise_rank, seed=args.draw_seed)
            del R
            np.savez(args.out_dir / "noise_model.npz", diag=noise_model.diag,
                     U=noise_model.U, s=noise_model.s)
        if args.subspace_method == "snr":
            _, means, counts = pair_means(Xtv, pair, train_real)
        for d_ in sorted(set(args.d)):
            if args.subspace_method == "pca":
                bases[d_], ev = pca_subspace(Xtv, pair, train_real, d_, seed=args.draw_seed)
                np.save(args.out_dir / f"subspace_V_pca_d{d_}.npy", bases[d_])
            elif args.subspace_method == "rca":
                bases[d_], ev = rca_subspace(Xtv, pair, train_real, d_, rank=args.rca_rank,
                                             seed=args.draw_seed)
                np.save(args.out_dir / f"subspace_V_rca{args.rca_rank}_d{d_}.npy", bases[d_])
                np.save(args.out_dir / f"rca_reliability_rca{args.rca_rank}_d{d_}.npy", ev)
            else:
                bases[d_], ev = label_subspace(means, counts, noise_model, d_)
                np.save(args.out_dir / f"subspace_V_snr_d{d_}.npy", bases[d_])
        print(f"label subspaces ({args.subspace_method}) for d={sorted(bases)} from the "
              f"train pairs  ({time.time()-t1:.0f}s)", flush=True)

    # NAME=PATH loads external axes; a bare "V" means the label-driven subspace
    ext_bases = {}
    for kv in (args.basis or []):
        name, _, path = kv.partition("=")
        ext_bases[name] = np.load(path).astype(np.float32) if path else None
    basis_names = list(ext_bases) or [None]
    cells = [c for c in itertools.product(args.d, args.alpha, args.beta,
                                          args.n_subsamples, args.seeds, args.subspace,
                                          basis_names, args.head_basis)
             if c[5] in ("fixed", "frozen") or c[6] == basis_names[0]]
    if queue is not None:
        # longest first (cost grows with K): the expensive cells start early
        # and the cheap ones fill the tail, which shortens the makespan
        cells.sort(key=lambda c: -c[3])
    print(f"{len(cells)} cells, {len(cells)-len(done)} to run\n", flush=True)

    records = []
    for i, (d, a, b, K, seed, sub, bname, hbasis) in enumerate(cells, 1):
        head = tuple(hbasis.split(","))
        named = bname is not None and sub in ("fixed", "frozen")
        use_ext = named and ext_bases[bname] is not None
        tag = (f"d{d}_a{a:g}_b{b:g}_K{K}_seed{seed}" + ("" if sub == "free" else f"_{sub}")
               + (f"_{bname}" if named else "")
               + ("" if head == ("linear", "square", "abs", "tanh") else f"_head-{'-'.join(head)}")
               + ("_Arandom" if args.axes_init == "random" and sub == "fixed" else "")
               + ("" if args.label_model == "mlp" else f"_{args.label_model}")
               + ((f"_initV{args.subspace_method}" + ("_Arandom" if args.axes_init == "random" else "")
                   if args.free_init == "basis" else "") if sub == "free"
                  else {"snr": "_snr", "pca": "_pca"}.get(args.subspace_method, f"_rca{args.rca_rank}"))
               + ("_dist" if args.head == "distance" else "")
               + (f"_lwd{args.label_weight_decay:g}" if args.label_weight_decay else "")
               + (f"_ldo{args.label_dropout:g}" if args.label_dropout else "")
               + ("_embresp" if args.pert_embedding_init == "response" else "")
               + (f"_sparse{args.sparsity:g}" if args.sparsity else ""))
        # re-read each time: cheap, and means a cell finished by any earlier
        # or concurrent process is never repeated
        if queue is not None:
            if tag in queue.done() or not queue.claim(tag):
                continue
        else:
            if csv.exists():
                done = set(pd.read_csv(csv, usecols=["tag"]).tag)
            if tag in done:
                continue
        t = time.time()
        keep = subsample_mask(mtv, K, seed=args.draw_seed) | val_all
        keep &= thin_synthetic(mtv, args.shuffled_frac, seed=args.draw_seed)
        X, meta, val = Xtv[keep], mtv[keep].reset_index(drop=True), val_all[keep]
        is_real = meta.is_real.to_numpy()
        synth_level = meta.shuffle_frac.to_numpy(dtype=float)
        fa = prepare(X, meta, genes)
        plev, clev = fa.pop("perturbation_levels"), fa.pop("context_levels")
        cfg = TrainConfig(n_factors=d, alpha=a, recon_fake_weight=b, seed=seed,
                          epochs=args.epochs, batch_size=args.batch_size,
                          device=args.device, log_every=0,
                          eval_every=args.eval_every, patience=args.patience,
                          min_epochs=args.min_epochs,
                          select_on=args.select_on, subspace=sub, basis=head,
                          axes_init=args.axes_init, label_model=args.label_model, head=args.head,
                          sparsity=args.sparsity,
                          subspace_method=args.subspace_method, rca_rank=args.rca_rank,
                          free_init=args.free_init, pert_embedding_init=args.pert_embedding_init,
                          embedding_dim=args.embedding_dim,
                          context_embedding_dim=args.context_embedding_dim,
                          label_weight_decay=args.label_weight_decay,
                          label_dropout=args.label_dropout)
        snaps: dict[int, np.ndarray] = {}
        model, hist = fit(config=cfg, train_rows=~val, val_rows=val,
                          is_real=is_real, synth_level=synth_level,
                          subspace_basis=(ext_bases[bname] if use_ext else bases.get(d)),
                          noise_model=noise_model,
                          on_eval=lambda ep, m: snaps.__setitem__(ep, m.loading_matrix().copy()),
                          **fa)
        pd.DataFrame(hist).to_csv(args.out_dir/f"history_{tag}.csv", index=False)
        np.save(args.out_dir/f"B_{tag}.npy", model.loading_matrix())
        ep = np.array(sorted(snaps))
        np.savez(args.out_dir/f"B_evals_{tag}.npz", epoch=ep,
                 B=np.stack([snaps[e] for e in ep]).astype(np.float32))

        X_t = torch.from_numpy(fa["X"]); col_t = torch.from_numpy(fa["target_col"])
        w_t = None                                    # every row weighs the same

        def score(fa_, rows, Xt_, ct_, wt_):
            s = StratifiedNegativeSampler(
                perturbation_idx=fa_["perturbation_idx"][rows],
                context_idx=fa_["context_idx"][rows],
                stratum=np.zeros(len(rows), dtype=np.int64),
                n_perturbations=int(fa_["perturbation_idx"].max())+1,
                n_contexts=int(fa_["context_idx"].max())+1,
                weights={"same_s_other_pert":0.5,"same_s_other_context":0.5})
            # evaluate() indexes is_real by `rows` itself -- pass the FULL array
            return evaluate(model, Xt_, fa_["perturbation_idx"], fa_["context_idx"],
                            rows, s, np.random.default_rng(12345), ct_, wt_,
                            is_real=fa_["is_real"], recon_fake_weight=b,
                            synth_level=fa_.get("synth_level"))

        fa["is_real"] = is_real
        fa["synth_level"] = synth_level
        row = {"tag": tag, "d": d, "alpha": a, "beta": b, "K": K, "seed": seed,
               "subspace": sub, "head_basis": ",".join(head), "basis": (bname if named else ("V" if sub != "free" else "")),
               "label_model": args.label_model, "head": args.head, "sparsity": args.sparsity,
               "subspace_method": args.subspace_method, "rca_rank": args.rca_rank,
               "free_init": args.free_init, "pert_embedding_init": args.pert_embedding_init,
               "embedding_dim": args.embedding_dim, "context_embedding_dim": args.context_embedding_dim,
               "label_weight_decay": args.label_weight_decay, "label_dropout": args.label_dropout,
               "epochs_run": len(hist), "seconds": round(time.time()-t, 1)}
        for name, rows_ in (("train", np.nonzero(~val)[0]), ("val", np.nonzero(val)[0])):
            m = score(fa, rows_, X_t, col_t, w_t)
            for k in METRICS + tuple(k for k in m if k.startswith(("synth_accuracy_f", "recon_synth_f"))):
                row[f"{name}_{k}"] = m.get(k)
        if Xte is not None:
            fte = prepare(Xte, mte, genes, perturbation_levels=plev,
                          context_levels=clev)
            fte.pop("perturbation_levels", None); fte.pop("context_levels", None)
            fte["is_real"] = mte.is_real.to_numpy()
            fte["synth_level"] = mte.shuffle_frac.to_numpy(dtype=float)
            m = score(fte, np.arange(len(mte)),
                      torch.from_numpy(fte["X"]),
                      torch.from_numpy(fte["target_col"]),
                      None)
            for k in METRICS + tuple(k for k in m if k.startswith(("synth_accuracy_f", "recon_synth_f"))):
                row[f"test_{k}"] = m.get(k)
        records.append(row)
        if queue is not None:
            queue.finish(tag, row)
        else:
            # append safely: re-read + concat keeps one consistent header
            allr = pd.concat([pd.read_csv(csv), pd.DataFrame([row])], ignore_index=True) \
                   if csv.exists() else pd.DataFrame([row])
            allr.to_csv(csv, index=False)
        print(f"[{i}/{len(cells)}] {tag:<26} "
              f"disc {row['train_disc']:.4f}/{row['val_disc']:.4f}  "
              f"recon {row['train_recon']:.6f}/{row['val_recon']:.6f}  "
              f"val acc {row['val_accuracy']:.4f}  {row['seconds']:.0f}s", flush=True)
    (args.out_dir/"sweep_config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print("\nSWEEP DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
