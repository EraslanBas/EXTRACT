"""Split-first data layout: draws, nesting, and partition isolation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from extract.data.splits import (
    build_split,
    draw_test_pairs,
    load_partition,
    split_val_per_perturbation,
    subsample_mask,
    thin_synthetic,
)


def _meta(n_pert=30, n_ctx=5, n_sub=10, seed=0, ragged=True):
    """Rows like the real metadata: a full row plus n_sub subsamples per pair.
    With ``ragged`` some perturbations are in few contexts, some have few
    subsamples."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pert):
        p = f"P{i:02d}"
        ctxs = [f"C{j}" for j in range(n_ctx)]
        if ragged and i % 7 == 0:
            ctxs = ctxs[:1]                          # single-context perturbation
        elif ragged and i % 5 == 0:
            ctxs = ctxs[:2]
        for c in ctxs:
            k = n_sub if not (ragged and i % 4 == 0) else int(rng.integers(0, 4))
            rows.append({"perturbation": p, "context": c, "label": p,
                         "variant": "main", "n_cells": 500})
            for s in range(k):
                rows.append({"perturbation": p, "context": c,
                             "label": f"{p}__sub{s:02d}", "variant": "subsample",
                             "n_cells": 50 + 40 * s})
    m = pd.DataFrame(rows)
    m.insert(0, "row_id", m.context + "|" + m.label)
    return m


def _pairs(m):
    return set(zip(m.perturbation, m.context))


def test_test_pairs_leave_every_perturbation_a_trainval_pair():
    m = _meta()
    for seed in range(5):
        test = set(draw_test_pairs(m, test_frac=0.3, seed=seed))
        left = _pairs(m) - test
        assert {p for p, _ in left} == set(m.perturbation)
        n = len(_pairs(m))
        assert len(test) == round(0.3 * n)


def test_val_split_keeps_a_train_pair_per_perturbation_and_hits_the_fraction():
    m = _meta(n_pert=200, n_ctx=8, ragged=True)
    val = split_val_per_perturbation(m, val_frac=0.1, seed=0)
    train_pairs = _pairs(m[~val])
    val_pairs = _pairs(m[val])
    assert not train_pairs & val_pairs
    assert {p for p, _ in val_pairs} <= {p for p, _ in train_pairs}
    # all rows of a pair move together
    key = m.perturbation + "|" + m.context
    assert (pd.Series(val).groupby(key.to_numpy()).nunique() == 1).all()
    frac = len(val_pairs) / len(_pairs(m))
    assert 0.07 < frac < 0.13, frac


def test_subsample_mask_is_nested_and_keeps_full_rows():
    m = _meta(ragged=True)
    masks = {k: subsample_mask(m, k, seed=3) for k in (1, 2, 3, 5, 7, 10)}
    full = (m.variant == "main").to_numpy()
    for k, mk in masks.items():
        assert mk[full].all(), "full-data rows are always kept"
        per_pair = (pd.Series(mk & ~full)
                    .groupby((m.perturbation + "|" + m.context).to_numpy()).sum())
        avail = (pd.Series(~full)
                 .groupby((m.perturbation + "|" + m.context).to_numpy()).sum())
        assert (per_pair == np.minimum(avail, k)).all(), k
    ks = sorted(masks)
    for a, b in zip(ks, ks[1:]):
        assert not (masks[a] & ~masks[b]).any(), f"K={a} not inside K={b}"


def test_subsample_mask_is_a_function_of_the_pair_not_the_row_set():
    """Same pair, same seed, same pick, whatever other rows are present --
    so train/val and test (or real and synthetic) agree."""
    m = _meta(ragged=False)
    full = subsample_mask(m, 3, seed=1)
    part = m[m.context == "C2"]
    assert (subsample_mask(part, 3, seed=1) == full[part.index]).all()
    twins = pd.concat([m, m], ignore_index=True)
    both = subsample_mask(twins, 3, seed=1)
    assert (both[: len(m)] == both[len(m):]).all()


def test_thin_synthetic_keeps_all_real_rows():
    m = _meta(ragged=False)
    both = pd.concat([m.assign(is_real=True),
                      m.assign(is_real=False, row_id=m.row_id + "|shuffled")],
                     ignore_index=True)
    keep = thin_synthetic(both, 0.5, seed=0)
    assert keep[both.is_real].all()
    n_syn = int((keep & ~both.is_real).sum())
    assert abs(n_syn - 0.5 * len(m)) < 0.05 * len(m)
    # nested in frac, and independent of which other rows are loaded
    assert not (thin_synthetic(both, 0.3, seed=0) & ~keep).any()
    half = both.iloc[len(m) // 2:]
    assert (thin_synthetic(half, 0.5, seed=0) == keep[len(m) // 2:]).all()


def _write_matrices(d, meta, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"G{j}" for j in range(n_genes)]
    for ctx, m in meta.groupby("context", sort=True):
        m = m.reset_index(drop=True)
        vals = rng.normal(scale=0.3, size=(len(m), n_genes)).astype(np.float32)
        vals[:, 0] *= 0.01                           # a gene nothing moves
        frame = pd.DataFrame(vals, columns=genes)
        frame.insert(0, "label", m.label.to_numpy())
        frame.to_parquet(d / f"{ctx}_PosteriorMean.parquet", index=False)
        m.to_csv(d / f"{ctx}_row_metadata.csv", index=False)
    return genes


def test_build_split_isolates_the_partitions(tmp_path):
    meta = _meta(n_pert=40, n_ctx=4, ragged=True)
    mats = tmp_path / "matrices"; mats.mkdir()
    genes = _write_matrices(mats, meta)
    out = tmp_path / "split"
    man = build_split(mats, out, test_frac=0.2, split_seed=0,
                      min_affected=5, threshold=0.2, shuffle_within="partition")

    test_pairs = set(map(tuple, pd.read_csv(out / "test_pairs.tsv", sep="\t")
                         .astype(str).to_numpy()))
    Xtv, mtv, g = load_partition(out, "trainval")
    Xte, mte, g2 = load_partition(out, "test")

    assert list(g) == list(g2) and "G0" not in set(g)
    assert set(zip(mtv.perturbation, mtv.context)).isdisjoint(test_pairs)
    assert set(zip(mte.perturbation, mte.context)) == test_pairs
    assert len(mtv[mtv.is_real]) + len(mte[mte.is_real]) == len(meta)
    assert man["partitions"]["test"]["pairs"] == len(test_pairs)

    # synthetic rows are column permutations WITHIN their own partition and
    # context: each gene's multiset of values is preserved there
    for m, X in ((mtv, Xtv), (mte, Xte)):
        for ctx in m.context.unique():
            r = ((m.context == ctx) & m.is_real).to_numpy()
            f = ((m.context == ctx) & ~m.is_real).to_numpy()
            assert np.allclose(np.sort(X[r], 0), np.sort(X[f], 0))

    # the gene list was decided on train/val full-data rows only
    header = (out / "gene_list.tsv").read_text().splitlines()[1]
    n_tv_main = int(((meta.variant == "main")
                     & ~pd.Series(list(zip(meta.perturbation, meta.context)))
                     .isin(test_pairs)).sum())
    assert f"n_pairs={n_tv_main}" in header

    with pytest.raises(FileExistsError):
        build_split(mats, out, test_frac=0.2)
    assert json.loads((out / "manifest.json").read_text())["split_seed"] == 0


def test_build_split_shuffles_over_the_whole_context(tmp_path):
    """shuffle_within='context': one column shuffle over ALL of a context's
    rows, before the split. Per gene the multiset of values is preserved over
    the context (not per partition), each synthetic row keeps its twin's label
    and partition, and the real rows and test pairs are unchanged."""
    meta = _meta(n_pert=40, n_ctx=4, ragged=True)
    mats = tmp_path / "matrices"; mats.mkdir()
    _write_matrices(mats, meta)
    ctx_out, part_out = tmp_path / "ctx", tmp_path / "part"
    man = build_split(mats, ctx_out, test_frac=0.2, split_seed=0,
                      min_affected=5, threshold=0.2, shuffle_within="context")
    build_split(mats, part_out, test_frac=0.2, split_seed=0,
                min_affected=5, threshold=0.2, shuffle_within="partition")
    assert man["shuffled_within"].startswith("context")
    assert ((ctx_out / "test_pairs.tsv").read_text()
            == (part_out / "test_pairs.tsv").read_text())
    assert ((ctx_out / "gene_list.tsv").read_text()
            == (part_out / "gene_list.tsv").read_text())

    parts = {p: load_partition(ctx_out, p) for p in ("trainval", "test")}
    crossed = False
    for ctx in meta.context.astype(str).unique():
        real, fake = [], []
        for p, (X, m, _) in parts.items():
            here = (m.context.astype(str) == ctx).to_numpy()
            r, f = here & m.is_real.to_numpy(), here & ~m.is_real.to_numpy()
            # twins: same labels, same order, same partition
            assert (m.label[r].to_numpy() == m.label[f].to_numpy()).all()
            real.append(X[r]); fake.append(X[f])
            if not np.allclose(np.sort(X[r], 0), np.sort(X[f], 0)):
                crossed = True
        assert np.allclose(np.sort(np.vstack(real), 0), np.sort(np.vstack(fake), 0))
    assert crossed, "values never crossed partitions: shuffle was not context-wide"

    Xa, ma, _ = load_partition(ctx_out, "trainval", shuffled=False)
    Xb, mb, _ = load_partition(part_out, "trainval", shuffled=False)
    assert np.array_equal(Xa, Xb) and ma.row_id.equals(mb.row_id)


def test_fit_model_script_runs_on_a_split(tmp_path):
    """End to end: build a split, fit on trainval, score test once."""
    import subprocess
    import sys
    from pathlib import Path

    meta = _meta(n_pert=40, n_ctx=4, ragged=False)
    mats = tmp_path / "matrices"; mats.mkdir()
    _write_matrices(mats, meta, n_genes=16)
    split = tmp_path / "split"
    build_split(mats, split, test_frac=0.15, min_affected=5, threshold=0.2)

    script = Path(__file__).resolve().parents[1] / "scripts" / "fit_model.py"
    out = tmp_path / "models"
    proc = subprocess.run(
        [sys.executable, str(script), "--split-dir", str(split), "--out-dir", str(out),
         "-d", "3", "-K", "3", "--beta", "0.1", "--epochs", "3", "--batch-size", "64",
         "--eval-every", "1", "--patience", "0"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    summary = json.loads((out / "summary_d3_a1_b0.1_K3_seed0_fixed.json").read_text())
    for part in ("train", "val", "test"):
        assert 0.0 <= summary["metrics"][part]["accuracy"] <= 1.0
        assert summary["metrics"][part]["n_synthetic"] > 0
    assert summary["hyperparameters"] == {"d": 3, "alpha": 1.0, "beta": 0.1,
                                          "K": 3, "rho": 0.5}


def test_graded_shuffle_levels(tmp_path):
    """add_shuffle_levels writes partially shuffled twins; loading with levels
    keeps the same rows and the same thinning, changes only which file each
    twin comes from, and a fraction-f twin differs from its real row in about
    a fraction f of genes."""
    from extract.data.splits import (SHUFFLE_LEVELS, add_shuffle_levels,
                                     assign_shuffle_levels, thin_synthetic)
    meta = _meta(n_pert=60, n_ctx=8, ragged=False)
    mats = tmp_path / "matrices"; mats.mkdir()
    _write_matrices(mats, meta)
    out = tmp_path / "split"
    build_split(mats, out, test_frac=0.1, min_affected=5, threshold=0.2)
    man = add_shuffle_levels(out, (0.2, 0.4, 0.6, 0.8), matrices_dir=mats)
    assert man["shuffle_levels"] == [0.2, 0.4, 0.6, 0.8, 1.0]

    X1, m1, _ = load_partition(out, "trainval")
    Xg, mg, _ = load_partition(out, "trainval", shuffle_levels=SHUFFLE_LEVELS)
    assert len(m1) == len(mg) and (m1.row_id == mg.row_id).all()
    assert (thin_synthetic(m1, 0.5) == thin_synthetic(mg, 0.5)).all()
    real = mg.is_real.to_numpy()
    assert np.allclose(X1[real], Xg[real])
    lev = mg.shuffle_frac.to_numpy()[~real]
    assert set(np.unique(lev)) == set(SHUFFLE_LEVELS)
    # roughly uniform over the five levels
    counts = np.array([(lev == f).mean() for f in SHUFFLE_LEVELS])
    assert np.all(np.abs(counts - 0.2) < 0.06), counts
    # twins at fraction f leave about 1 - f of the genes at their real values
    Xr, Xs = Xg[real], Xg[~real]
    for f in (0.2, 0.6):
        at = lev == f
        same = np.isclose(Xr[at], Xs[at]).mean()
        assert abs(same - (1 - f)) < 0.12, (f, same)
    # assignment is fixed by row id alone
    ids = mg.row_id.to_numpy()[real][:50]
    assert (assign_shuffle_levels(ids) == assign_shuffle_levels(ids[::-1])[::-1]).all()


def test_evaluate_reports_each_shuffle_level(tmp_path):
    from extract.data.splits import SHUFFLE_LEVELS, add_shuffle_levels
    from extract.train import TrainConfig, fit, prepare
    meta = _meta(n_pert=60, n_ctx=8, ragged=False)
    mats = tmp_path / "matrices"; mats.mkdir()
    _write_matrices(mats, meta)
    out = tmp_path / "split"
    build_split(mats, out, test_frac=0.1, min_affected=5, threshold=0.2)
    add_shuffle_levels(out, (0.2, 0.4, 0.6, 0.8), matrices_dir=mats)
    X, m, genes = load_partition(out, "trainval", shuffle_levels=SHUFFLE_LEVELS)
    from extract.data.splits import split_val_per_perturbation, _pair_key
    real = m.is_real.to_numpy()
    vp = set(_pair_key(m[real])[split_val_per_perturbation(m[real], 0.15, 0)])
    val = _pair_key(m).isin(vp).to_numpy()
    fa = prepare(X, m, genes); fa.pop("perturbation_levels"); fa.pop("context_levels")
    cfg = TrainConfig(n_factors=3, epochs=2, batch_size=64, eval_every=1,
                      patience=0, log_every=0, select_on="accuracy")
    _, hist = fit(config=cfg, train_rows=~val, val_rows=val, is_real=real,
                  synth_level=m.shuffle_frac.to_numpy(dtype=float), **fa)
    keys = {k for k in hist[-1] if k.startswith("val_synth_accuracy_f")}
    assert keys == {f"val_synth_accuracy_f{f:g}" for f in SHUFFLE_LEVELS}
    assert all(0.0 <= hist[-1][k] <= 1.0 for k in keys)
