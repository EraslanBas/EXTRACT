"""Split first, then build every data artifact separately per partition.

The held-out test pairs are drawn **once**, from the metadata alone, before any
operation touches the values. Everything derived from values is then built
inside a partition and never across one:

* the gene list is computed from **train/val** full-data rows only, and the
  same list is applied to both partitions;
* synthetic rows are column-shuffled within each context. ``shuffle_within``
  chooses the pool: ``"context"`` (the default since 2026-09-29) shuffles all
  of a context's rows before they are divided, so a synthetic row's values are
  draws from the context's full per-gene marginals and may come from any
  partition -- test values included; ``"partition"`` shuffles inside
  (partition, context) only, so no test value lands in a train/val row. Either
  way each synthetic row keeps its real twin's label and goes to that label's
  partition;
* negatives are drawn live, by a sampler built on one partition's rows.

Layout written by :func:`build_split`::

    <split_dir>/
      manifest.json          seeds, fractions, gene filter, row/pair counts
      test_pairs.tsv         the held-out (perturbation, context) pairs
      gene_list.tsv          from train/val full-data rows only
      trainval/  test/
        <ctx>_real.parquet                label + genes, measured rows
        <ctx>_shuffled_seed<k>.parquet    label + genes, same row order
        <ctx>_row_metadata.csv

Two partitions only. Validation is drawn inside ``trainval`` at fit time with
:func:`split_val_per_perturbation`; the number of subsamples ``K`` is chosen at
fit time with :func:`subsample_mask`. Both are functions of the pair and a
seed, so every model config sees the same draw.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from .augment import SUB_RE, shuffle_matrix
from .gene_filter import gene_list_table, load_gene_list

PARTITIONS = ("trainval", "test")


def _pair_key(meta: pd.DataFrame) -> pd.Series:
    return meta.perturbation.astype(str) + "|" + meta.context.astype(str)


def _stable_uniform(keys, seed: int) -> np.ndarray:
    """Uniform [0, 1) per key, fixed by key and seed alone (not by row order
    or by which other rows are present). ``hash()`` is salted per process."""
    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        digest = hashlib.md5(f"{k}:{seed}".encode()).digest()
        out[i] = int.from_bytes(digest[:8], "little") / 2**64
    return out


# ---- the three draws --------------------------------------------------------


def draw_test_pairs(
    meta: pd.DataFrame, test_frac: float = 0.1, seed: int = 0
) -> list[tuple[str, str]]:
    """Hold out ``test_frac`` of all ``(perturbation, context)`` pairs.

    Every perturbation keeps **at least one pair** outside test: a perturbation
    with all its pairs held out has an embedding ``e_p`` that is never trained,
    so its test rows are unpredictable by construction. Pairs are visited in a
    random order and a pair is skipped if taking it would remove its
    perturbation's last remaining pair.
    """
    pairs = (
        meta[["perturbation", "context"]].astype(str).drop_duplicates()
        .sort_values(["perturbation", "context"]).to_numpy()
    )
    n_test = int(round(test_frac * len(pairs)))
    if not 0 < n_test < len(pairs):
        raise ValueError(f"test_frac={test_frac} gives {n_test} of {len(pairs)} pairs")

    remaining = pd.Series(pairs[:, 0]).value_counts().to_dict()
    chosen = []
    for i in np.random.default_rng(seed).permutation(len(pairs)):
        p = pairs[i, 0]
        if remaining[p] <= 1:
            continue
        chosen.append((pairs[i, 0], pairs[i, 1]))
        remaining[p] -= 1
        if len(chosen) == n_test:
            break
    if len(chosen) < n_test:
        raise ValueError(
            f"only {len(chosen)} pairs can be held out while keeping one pair "
            f"per perturbation; {n_test} requested"
        )
    return sorted(chosen)


def split_val_per_perturbation(
    meta: pd.DataFrame, val_frac: float = 0.1, seed: int = 0
) -> np.ndarray:
    """Boolean mask over rows: ``True`` for validation pairs.

    Grouped per perturbation: each perturbation holds out about ``val_frac`` of
    its pairs and **always keeps at least one** in train, so every validation
    pair's perturbation and context were both trained and validation measures
    the interaction. ``val_frac * n_pairs`` is rounded stochastically, so
    perturbations with few pairs still contribute and the total comes out near
    ``val_frac``. All rows of a pair move together.
    """
    pairs = (
        meta[["perturbation", "context"]].astype(str).drop_duplicates()
        .sort_values(["perturbation", "context"])
    )
    rng = np.random.default_rng(seed)
    held = []
    for pert, grp in pairs.groupby("perturbation", sort=True):
        n = len(grp)
        target = val_frac * n
        k = int(np.floor(target)) + int(rng.random() < target - np.floor(target))
        k = min(k, n - 1)
        if k:
            ctxs = grp.context.to_numpy()[rng.choice(n, k, replace=False)]
            held += [f"{pert}|{c}" for c in ctxs]
    return _pair_key(meta).isin(set(held)).to_numpy()


def subsample_mask(
    meta: pd.DataFrame, n_subsamples: int, seed: int = 0
) -> np.ndarray:
    """Keep the full-data row and ``n_subsamples`` of each pair's subsamples.

    The chosen subsamples are the first ``K`` of a random ordering fixed per
    ``(perturbation, context)`` and ``seed``, so the sets are **nested**
    (K=1 inside K=2 inside ... K=10) and a difference between two K values is
    due to K, not to which subsamples were drawn. A synthetic row has the same
    pair and label as its real twin, so the two are kept or dropped together.
    A pair with fewer than ``K`` subsamples keeps all it has.
    """
    labels = meta.label.astype(str)
    sub = labels.str.extract(SUB_RE.pattern, expand=False)
    is_sub = sub.notna().to_numpy()
    keep = ~is_sub
    if n_subsamples <= 0 or not is_sub.any():
        return keep
    frame = pd.DataFrame({
        "pair": _pair_key(meta).to_numpy()[is_sub],
        "label": labels.to_numpy()[is_sub],
    })
    # rank distinct labels within a pair (a synthetic twin shares the label)
    uniq = frame.drop_duplicates()
    uniq = uniq.assign(u=_stable_uniform((uniq.pair + "|" + uniq.label).to_numpy(), seed))
    uniq["rank"] = uniq.groupby("pair").u.rank(method="first") - 1
    rank = frame.merge(uniq, on=["pair", "label"], how="left")["rank"].to_numpy()
    keep[np.nonzero(is_sub)[0]] = rank < n_subsamples
    return keep


def thin_synthetic(
    meta: pd.DataFrame, frac: float, seed: int = 0
) -> np.ndarray:
    """Keep every measured row and about ``frac`` of the synthetic rows.

    ``frac`` is relative to the synthetic rows present, which equal the
    measured rows in number, so it is the synthetic share ``rho`` of the
    negative budget once permuted negatives are balanced against it.

    Each synthetic row is kept iff a hash of its ``row_id`` and ``seed`` falls
    below ``frac``: the decision depends on the row alone, so it is the same
    whichever other rows are loaded and nested across ``K`` and ``frac``.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    real = meta.is_real.to_numpy(dtype=bool)
    keep = real.copy()
    synth = np.nonzero(~real)[0]
    u = _stable_uniform(meta.row_id.astype(str).to_numpy()[synth], seed)
    keep[synth] = u < frac
    return keep


# ---- graded synthetic rows ----------------------------------------------------

#: Fractions of gene columns shuffled in a synthetic row. 1.0 is the original
#: fully shuffled matrix; smaller fractions leave the rest of the row's genes at
#: their real values, so the negative is harder to reject.
SHUFFLE_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)

#: Salt for the level assignment, so it is independent of the keep/drop draw
#: of :func:`thin_synthetic` (which hashes the same row ids with ``seed``).
_LEVEL_SALT = 104_729


def _shuffled_file(name: str, frac: float, seed: int) -> str:
    """File name of a context's synthetic rows at one shuffle fraction. The
    fully shuffled level keeps its original name, so older splits still load."""
    if frac >= 1.0:
        return f"{name}_shuffled_seed{seed}.parquet"
    return f"{name}_shuffled_frac{frac:g}_seed{seed}.parquet"


def assign_shuffle_levels(
    row_ids, levels: tuple[float, ...] = SHUFFLE_LEVELS, seed: int = 0
) -> np.ndarray:
    """The shuffle fraction each measured row's synthetic twin is taken from.

    Uniform over ``levels``, fixed by the row id alone, so it is the same in
    every fit and nested across ``K``; independent of :func:`thin_synthetic`,
    which therefore keeps exactly the synthetic rows it kept before levels
    existed. The number of synthetic negatives is unchanged; only their
    difficulty varies.
    """
    u = _stable_uniform(np.asarray(row_ids, dtype=str), seed + _LEVEL_SALT)
    idx = np.minimum((u * len(levels)).astype(int), len(levels) - 1)
    return np.asarray(levels, dtype=float)[idx]


def add_shuffle_levels(
    split_dir: str | Path,
    fracs: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    matrices_dir: str | Path | None = None,
    shuffle_seeds: tuple[int, ...] | None = None,
) -> dict:
    """Write partially shuffled synthetic rows into an existing split.

    Uses exactly the construction of the fully shuffled rows in
    :func:`build_split` with ``shuffle_within="context"``: each context's rows
    are shuffled together, before the split, and every synthetic row goes to
    the partition of the label it carries. A fraction ``f`` permutes a random
    ``f`` of the gene columns (chosen per context) across rows; the other
    columns keep the row's own values. Real rows, test pairs and the gene list
    are untouched. Returns the updated manifest.
    """
    from ..de.shards import context_seed
    import pyarrow.parquet as pq
    from .augment import shuffle_matrix

    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    if not str(manifest.get("shuffled_within", "")).startswith("context"):
        raise ValueError("graded levels are defined for context-wide shuffling only")
    matrices_dir = Path(matrices_dir or manifest["source_matrices"])
    seeds = tuple(shuffle_seeds or manifest.get("shuffle_seeds", [0]))
    genes = [str(g) for g in load_gene_list(split_dir / "gene_list.tsv")]
    held = {f"{p}|{c}" for p, c in pd.read_csv(split_dir / "test_pairs.tsv", sep="\t")
            .astype(str).itertuples(index=False)}
    for n in manifest["contexts"]:
        m = pd.read_csv(matrices_dir / f"{n}_row_metadata.csv")
        table = pq.read_table(matrices_dir / f"{n}_PosteriorMean.parquet",
                              columns=["label"] + genes).to_pandas()
        if not (table.label.to_numpy() == m.label.to_numpy()).all():
            raise ValueError(f"{n}: matrix and metadata row order disagree")
        in_test = _pair_key(m).isin(held).to_numpy()
        values = table[genes].to_numpy(dtype=np.float32)
        for k in seeds:
            for f in fracs:
                fake_all = shuffle_matrix(values, seed=context_seed(f"{n}|frac{f:g}", k), frac=f)
                for part, mask in (("trainval", ~in_test), ("test", in_test)):
                    rows = np.nonzero(mask)[0]
                    saved = pd.read_csv(split_dir / part / f"{n}_row_metadata.csv",
                                        usecols=["label"])
                    if not (saved.label.to_numpy() == table.label.to_numpy()[rows]).all():
                        raise ValueError(f"{part}/{n}: split rows do not match the matrix")
                    fake = pd.DataFrame(fake_all[rows], columns=genes)
                    fake.insert(0, "label", table.label.to_numpy()[rows])
                    _write_atomic_parquet(fake, split_dir / part / _shuffled_file(n, f, k))
        print(f"[levels] {n}: {', '.join(f'{f:g}' for f in fracs)}", flush=True)
        del table, values
    manifest["shuffle_levels"] = sorted(set(manifest.get("shuffle_levels", [1.0])) | set(fracs))
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ---- building and loading ---------------------------------------------------


def _write_atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def build_split(
    matrices_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    test_frac: float = 0.1,
    split_seed: int = 0,
    min_affected: int = 151,
    threshold: float = float(np.log(1.2)),
    shuffle_seeds: tuple[int, ...] = (0,),
    contexts: list[str] | None = None,
    shuffle_within: str = "context",
) -> dict:
    """Draw the test pairs, then write both partitions. Returns the manifest.

    Two passes over the matrices: the first counts affected genes on train/val
    full-data rows, the second writes each context's rows, gene-filtered, into
    its partition. Synthetic rows are shuffled over the whole context
    (``shuffle_within="context"``) or inside each partition (``"partition"``);
    see the module docstring.

    Refuses to write into a directory that already holds a manifest: a split
    is fixed once made, and every fit that used it depends on that.
    """
    if shuffle_within not in ("context", "partition"):
        raise ValueError(f"shuffle_within must be 'context' or 'partition', got {shuffle_within!r}")
    from ..de.shards import context_seed
    from .gene_filter import affected_counts
    import pyarrow.parquet as pq

    matrices_dir = Path(matrices_dir or paths.matrices())
    out_dir = Path(out_dir or paths.root() / "splits" / f"pair_seed{split_seed}")
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"{out_dir} already holds a split; use a new directory")
    for part in PARTITIONS:
        (out_dir / part).mkdir(parents=True, exist_ok=True)
    started = time.time()

    found = sorted(matrices_dir.glob("*_PosteriorMean.parquet"))
    names = [f.name.replace("_PosteriorMean.parquet", "") for f in found]
    if contexts:
        names = [n for n in names if n in set(contexts)]
    metas = {n: pd.read_csv(matrices_dir / f"{n}_row_metadata.csv") for n in names}
    meta_all = pd.concat(metas.values(), ignore_index=True)

    # ---- 1. the split, from metadata alone --------------------------------
    test_pairs = draw_test_pairs(meta_all, test_frac=test_frac, seed=split_seed)
    pd.DataFrame(test_pairs, columns=["perturbation", "context"]).to_csv(
        out_dir / "test_pairs.tsv", sep="\t", index=False)
    held = {f"{p}|{c}" for p, c in test_pairs}
    print(f"[split] {len(test_pairs):,} test pairs of "
          f"{meta_all.groupby(['perturbation', 'context']).ngroups:,}", flush=True)

    # ---- 2. gene list from train/val full-data rows -----------------------
    counts, genes_all = None, None
    for n in names:
        m = metas[n]
        rows = np.nonzero(((m.variant == "main") & ~_pair_key(m).isin(held)).to_numpy())[0]
        table = pq.read_table(matrices_dir / f"{n}_PosteriorMean.parquet").take(rows)
        genes = [c for c in table.column_names if c != "label"]
        if genes_all is None:
            genes_all = genes
        elif genes != genes_all:
            raise ValueError(f"{n}: gene columns differ from {names[0]}")
        values = table.select(genes).to_pandas().to_numpy(dtype=np.float32)
        c = affected_counts(values, m.iloc[rows], threshold=threshold, variant="main")
        counts = c.to_numpy() if counts is None else counts + c.to_numpy()
        del table, values
    counts = pd.Series(counts, index=genes_all, name="n_affected")
    tv_main = meta_all[(meta_all.variant == "main") & ~_pair_key(meta_all).isin(held)]
    gl = gene_list_table(counts, tv_main, min_affected, threshold,
                         out_dir / "gene_list.tsv")
    genes = gl.loc[gl.keep, "gene"].tolist()
    print(f"[genes] {len(genes):,} of {len(genes_all):,} kept "
          f"(|logFC| > {threshold:.5f} in >= {min_affected} train/val pairs)", flush=True)

    # ---- 3. per context, per partition ------------------------------------
    per_context = []
    for n in names:
        m = metas[n].copy()
        table = pq.read_table(matrices_dir / f"{n}_PosteriorMean.parquet",
                              columns=["label"] + genes).to_pandas()
        if not (table.label.to_numpy() == m.label.to_numpy()).all():
            raise ValueError(f"{n}: matrix and metadata row order disagree")
        in_test = _pair_key(m).isin(held).to_numpy()
        if shuffle_within == "context":
            # one shuffle over every row of the context, BEFORE the split;
            # each partition then takes the synthetic twins of its own rows
            all_values = table[genes].to_numpy(dtype=np.float32)
            fakes_all = {k: shuffle_matrix(all_values, seed=context_seed(n, k))
                         for k in shuffle_seeds}
            del all_values
        for part, mask in (("trainval", ~in_test), ("test", in_test)):
            rows = np.nonzero(mask)[0]
            block = table.iloc[rows].reset_index(drop=True)
            _write_atomic_parquet(block, out_dir / part / f"{n}_real.parquet")
            m.iloc[rows].assign(partition=part).to_csv(
                out_dir / part / f"{n}_row_metadata.csv", index=False)
            values = block[genes].to_numpy(dtype=np.float32)
            for k in shuffle_seeds:
                fake = pd.DataFrame(
                    fakes_all[k][rows] if shuffle_within == "context"
                    else shuffle_matrix(values, seed=context_seed(f"{n}|{part}", k)),
                    columns=genes,
                )
                fake.insert(0, "label", block.label.to_numpy())
                _write_atomic_parquet(fake, out_dir / part / f"{n}_shuffled_seed{k}.parquet")
            per_context.append({"context": n, "partition": part, "rows": len(rows),
                                "pairs": int(_pair_key(m.iloc[rows]).nunique())})
            del block, values
        if shuffle_within == "context":
            del fakes_all
        print(f"[write] {n}: {int((~in_test).sum()):,} trainval / "
              f"{int(in_test.sum()):,} test rows", flush=True)
        del table

    pc = pd.DataFrame(per_context)
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_matrices": str(matrices_dir),
        "split_level": "pair",
        "test_frac": test_frac,
        "split_seed": split_seed,
        "rule": "every perturbation keeps >= 1 pair in trainval",
        "gene_filter": {"threshold": threshold, "min_affected": min_affected,
                        "computed_on": "trainval full-data rows",
                        "n_genes": len(genes), "n_genes_before": len(genes_all)},
        "shuffle_seeds": list(shuffle_seeds),
        "shuffled_within": ("context, before the split" if shuffle_within == "context"
                            else "partition x context"),
        "contexts": names,
        "partitions": {
            part: {"rows": int(pc[pc.partition == part].rows.sum()),
                   "pairs": int(pc[pc.partition == part].pairs.sum())}
            for part in PARTITIONS
        },
        "minutes": round((time.time() - started) / 60, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_partition(
    split_dir: str | Path,
    partition: str,
    contexts: list[str] | None = None,
    shuffled: bool = True,
    shuffle_seed: int = 0,
    shuffle_levels: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """``(X, meta, genes)`` for one partition, measured rows first.

    ``meta`` carries ``is_real``; with ``shuffled=True`` the synthetic rows
    follow the measured ones, one per measured row, with the same label, and
    ``meta.shuffle_frac`` says what fraction of their genes was shuffled.
    ``shuffle_levels=None`` loads the fully shuffled rows only; a tuple such as
    :data:`SHUFFLE_LEVELS` takes each measured row's twin from the level
    :func:`assign_shuffle_levels` gives it.
    """
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    names = manifest["contexts"]
    if contexts:
        missing = set(contexts) - set(names)
        if missing:
            raise FileNotFoundError(f"not in this split: {sorted(missing)}")
        names = [n for n in names if n in set(contexts)]
    genes = np.asarray(load_gene_list(split_dir / "gene_list.tsv"))
    cols = [str(g) for g in genes]

    real_x, real_m, fake_x, fake_m = [], [], [], []
    for n in names:
        d = split_dir / partition
        m = pd.read_csv(d / f"{n}_row_metadata.csv")
        block = pd.read_parquet(d / f"{n}_real.parquet")
        if not (block.label.to_numpy() == m.label.to_numpy()).all():
            raise ValueError(f"{partition}/{n}: matrix and metadata disagree")
        real_x.append(block[cols].to_numpy(dtype=np.float32))
        real_m.append(m.assign(is_real=True))
        if shuffled:
            levels = tuple(shuffle_levels) if shuffle_levels else (1.0,)
            lev = (assign_shuffle_levels(m.row_id, levels, shuffle_seed)
                   if len(levels) > 1 else np.full(len(m), levels[0]))
            fx = np.empty((len(m), len(cols)), dtype=np.float32)
            for f in levels:
                fb = pd.read_parquet(d / _shuffled_file(n, f, shuffle_seed))
                if not (fb.label.to_numpy() == m.label.to_numpy()).all():
                    raise ValueError(f"{partition}/{n}: shuffled rows out of order")
                at = lev == f
                fx[at] = fb.loc[at, cols].to_numpy(dtype=np.float32)
                del fb
            fake_x.append(fx)
            fake_m.append(m.assign(is_real=False, shuffle_frac=lev,
                                   row_id=m.row_id + "|shuffled"))
        del block

    X = np.vstack(real_x + fake_x)
    meta = pd.concat(real_m + fake_m, ignore_index=True)
    return X, meta, genes
