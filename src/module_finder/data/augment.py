"""Augmented (label-permuted) data and leak-free train/test splits.

**The augmentation does not copy the matrix.** A negative is the *same* expression
row paired with a *different* label, so augmenting 170,800 rows costs a permuted
label column, not another 14 GB of parquet.

Two rules both matter, and both are about what must travel together:

* **Splits group by ``(perturbation, context)``.** All 11 samples of a held-out
  pair go to test. A sample-level split would leak: ``A1BG__sub03`` and
  ``A1BG__sub07`` are subsets of the same cells, so one in train tells you most
  of the other.
* **Negatives are stratified by sample index.** The noise scale of a row encodes
  its ``n_cells``, and ``n_cells`` is tied to perturbation identity, so an
  unstratified swap lets the discriminator win on precision rather than biology.
  Swapping only within a sample index holds precision roughly fixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths

SUB_RE = re.compile(r"__sub(\d+)$")

#: Negative-sampling strategies, all stratified by sample index.
STRATEGIES = ("same_s_other_pert", "same_s_other_context")


def sample_index(labels) -> np.ndarray:
    """``-1`` for a main row, otherwise the subsample number from the label."""
    out = np.full(len(labels), -1, dtype=np.int64)
    for i, lab in enumerate(np.asarray(labels).astype(str)):
        m = SUB_RE.search(lab)
        if m:
            out[i] = int(m.group(1))
    return out


def permute_labels(
    meta: pd.DataFrame,
    strategy: str = "same_s_other_pert",
    seed: int = 0,
) -> pd.DataFrame:
    """Return ``meta`` with a permuted ``(perturbation, context)`` per row.

    Adds ``perturbation_neg`` and ``context_neg``; the expression matrix is
    untouched. Permutation happens **within a sample-index stratum**, so a
    50-cell row is only ever paired with another 50-cell row's label.

    Strategies
    ----------
    ``same_s_other_pert``
        Keep context and sample index, swap the perturbation. Forces
        perturbation identity.
    ``same_s_other_context``
        Keep perturbation and sample index, swap the context. Forces the
        context interaction.

    Every emitted negative names an **observed** ``(perturbation, context)``
    pair. That is what keeps a held-out pair from being named, and it must hold
    here as well as in the live sampler -- these tables are built from the full
    metadata, so a blind pick would name test pairs.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    for col in ("perturbation", "context", "label"):
        if col not in meta.columns:
            raise KeyError(f"meta is missing {col!r}")

    out = meta.copy()
    out["s"] = sample_index(out.label)
    rng = np.random.default_rng(seed)
    # Which (perturbation, context) pairs actually exist, so a negative never
    # names a combination that was never measured.
    observed_by_pert = (
        out.groupby("perturbation", observed=True)["context"]
        .agg(lambda v: frozenset(v)).to_dict()
    )
    observed_by_ctx = (
        out.groupby("context", observed=True)["perturbation"]
        .agg(lambda v: tuple(pd.unique(v))).to_dict()
    )
    neg_p = out.perturbation.to_numpy().copy()
    neg_c = out.context.to_numpy().copy()

    ctx_arr = out.context.to_numpy()
    pert_arr = out.perturbation.to_numpy()
    s_arr = out["s"].to_numpy()

    def _pick(options, own):
        alt = [o for o in options if o != own]
        return alt[rng.integers(0, len(alt))] if alt else None

    if strategy == "same_s_other_pert":
        # Swap within the row's OWN (context, stratum) cell, so the emitted
        # (neg_p, c) is a pair that was measured. Permuting across the whole
        # stratum -- as an earlier version did -- lets a perturbation land on a
        # context it was never observed in, and on full metadata that means
        # naming held-out pairs.
        for (c, st), gi in out.groupby(["context", "s"], sort=True).indices.items():
            pool = pd.unique(pert_arr[gi])
            for j in gi:
                pick = _pick(pool, pert_arr[j])
                if pick is None:                      # cell has one pert only
                    pick = _pick(observed_by_ctx.get(c, ()), pert_arr[j])
                if pick is not None:
                    neg_p[j] = pick
    else:  # same_s_other_context
        for (pt, st), gi in out.groupby(["perturbation", "s"], sort=True).indices.items():
            pool = pd.unique(ctx_arr[gi])
            for j in gi:
                pick = _pick(pool, ctx_arr[j])
                if pick is None:
                    pick = _pick(sorted(observed_by_pert.get(pt, ())), ctx_arr[j])
                if pick is not None:
                    neg_c[j] = pick

    # Repair pass. A row whose strategy could not move it -- a perturbation
    # observed in exactly one context cannot have its context swapped -- would
    # otherwise be a mislabelled example. Swap the other axis instead, staying
    # inside observed pairs. Mirrors the collision repair in
    # objectives.contrastive.StratifiedNegativeSampler.
    stuck = np.nonzero((neg_p == pert_arr) & (neg_c == ctx_arr))[0]
    for j in stuck:
        pick = _pick(observed_by_ctx.get(neg_c[j], ()), pert_arr[j])
        if pick is not None:
            neg_p[j] = pick
            continue
        pick = _pick(sorted(observed_by_pert.get(pert_arr[j], ())), ctx_arr[j])
        if pick is not None:
            neg_c[j] = pick

    out["perturbation_neg"], out["context_neg"] = neg_p, neg_c
    same = (out.perturbation_neg == out.perturbation) & (out.context_neg == out.context)
    if same.any():
        n_bad = int(same.sum())
        raise AssertionError(
            f"{n_bad} negatives kept their own (perturbation, context) -- "
            "these would be mislabelled training examples. Every one is a "
            "label with no observed alternative on either axis; drop those "
            "rows or widen the pool."
        )
    return out


def _shuffle_away(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute so no element keeps its own value, when the stratum allows it."""
    unique = pd.unique(values)
    if len(unique) < 2:
        raise ValueError(
            "cannot build a false pair from a stratum with one distinct label; "
            "widen the stratum or drop it"
        )
    out = values.copy()
    for _ in range(20):
        stuck = out == values
        if not stuck.any():
            return out
        out[stuck] = rng.choice(unique, size=int(stuck.sum()))
    # deterministic fallback: rotate within the stratum
    order = rng.permutation(len(values))
    rotated = values[np.roll(order, 1)]
    out = np.where(rotated == values, values[np.roll(order, 2)], rotated)
    return out


@dataclass
class Split:
    """Boolean masks over rows, plus what was held out.

    ``val`` is ``None`` for a two-way split. Prefer three ways: the validation
    half is what you are allowed to look at repeatedly -- early stopping,
    ``alpha``, ``d`` -- and the test half is what you look at once, at the end.
    Choosing the stopping epoch on the test half turns it into a validation set
    and the number reported from it is no longer held out.
    """

    train: np.ndarray
    test: np.ndarray
    level: str
    held_out: list
    val: np.ndarray | None = None
    held_out_val: list = field(default_factory=list)

    @property
    def three_way(self) -> bool:
        return self.val is not None

    def summary(self, meta: pd.DataFrame) -> dict:
        out = {
            "level": self.level,
            "train_rows": int(self.train.sum()),
            "test_rows": int(self.test.sum()),
            "held_out": len(self.held_out),
            "train_perts": int(meta[self.train].perturbation.nunique()),
            "test_perts": int(meta[self.test].perturbation.nunique()),
            "train_contexts": int(meta[self.train].context.nunique()),
            "test_contexts": int(meta[self.test].context.nunique()),
        }
        if self.val is not None:
            out.update(
                val_rows=int(self.val.sum()),
                held_out_val=len(self.held_out_val),
                val_perts=int(meta[self.val].perturbation.nunique()),
                val_contexts=int(meta[self.val].context.nunique()),
            )
        return out


def make_split(
    meta: pd.DataFrame,
    level: str = "pair",
    test_frac: float = 0.1,
    seed: int = 0,
    held_out_contexts: list[str] | None = None,
    val_frac: float = 0.0,
    protect_single_context: bool = True,
) -> Split:
    """Leak-free split at the requested granularity.

    ``level``
        ``"pair"`` holds out ``(perturbation, context)`` combinations -- both
        sides seen elsewhere, so the model must supply the interaction.
        ``"perturbation"`` holds out whole perturbations across every context.
        ``"context"`` holds out whole contexts (pass ``held_out_contexts``).

    ``val_frac``
        Non-zero makes it a three-way split, drawing the validation units
        disjointly from the test units. Use validation for early stopping and
        hyperparameters and keep test for a single final read; otherwise the
        stopping epoch is chosen on the test set and its numbers are no longer
        held out.

    ``protect_single_context``
        At ``level="pair"``, keep every pair of a perturbation observed in only
        ONE context out of the holdout draw, so it always trains. Such a
        perturbation has a single pair; if that pair is held out, ``e_p`` is
        never trained and the row is unpredictable by construction -- a noise
        floor on the held-out score rather than a test of anything. On the
        12-context screen this affected 3 perturbations at seed 0. Fractions
        stay relative to *all* pairs, so the held-out count is unchanged; only
        the pool drawn from shrinks.

    In every case all 11 samples of an affected ``(perturbation, context)`` move
    together, which is checked before returning.
    """
    rng = np.random.default_rng(seed)
    pert = meta.perturbation.to_numpy().astype(str)
    ctx = meta.context.to_numpy().astype(str)
    val = held_val = None

    if level == "pair":
        pairs = pd.MultiIndex.from_arrays([pert, ctx])
        unique = pairs.unique()
        n_test = max(1, int(round(test_frac * len(unique))))
        n_val = max(1, int(round(val_frac * len(unique)))) if val_frac else 0
        if n_test + n_val > len(unique):
            raise ValueError(
                f"test_frac + val_frac would hold out {n_test + n_val} of "
                f"{len(unique)} pairs"
            )

        eligible = np.arange(len(unique))
        if protect_single_context:
            # A perturbation with one pair has nothing left to train e_p on if
            # that pair is held out. Keep those pairs in train.
            n_by_pert = pd.Series(unique.get_level_values(0)).value_counts()
            singles = set(n_by_pert.index[n_by_pert == 1])
            if singles:
                keep_in_train = np.array(
                    [p in singles for p in unique.get_level_values(0)]
                )
                eligible = np.nonzero(~keep_in_train)[0]
                if n_test + n_val > len(eligible):
                    raise ValueError(
                        f"after protecting {len(singles)} single-context "
                        f"perturbations only {len(eligible)} pairs remain "
                        f"drawable, need {n_test + n_val}"
                    )
        drawn = eligible[rng.choice(len(eligible), n_test + n_val, replace=False)]
        chosen = set(unique[drawn[:n_test]])
        test = np.array([pc in chosen for pc in pairs])
        held = sorted(chosen)
        if n_val:
            chosen_val = set(unique[drawn[n_test:]])
            val = np.array([pc in chosen_val for pc in pairs])
            held_val = sorted(chosen_val)
    elif level == "perturbation":
        unique = np.unique(pert)
        n_test = max(1, int(round(test_frac * len(unique))))
        n_val = max(1, int(round(val_frac * len(unique)))) if val_frac else 0
        if n_test + n_val > len(unique):
            raise ValueError(
                f"test_frac + val_frac would hold out {n_test + n_val} of "
                f"{len(unique)} perturbations"
            )
        drawn = rng.choice(unique, n_test + n_val, replace=False)
        chosen = set(drawn[:n_test])
        test = np.isin(pert, list(chosen))
        held = sorted(chosen)
        if n_val:
            chosen_val = set(drawn[n_test:])
            val = np.isin(pert, list(chosen_val))
            held_val = sorted(chosen_val)
    elif level == "context":
        if not held_out_contexts:
            raise ValueError("level='context' needs held_out_contexts")
        missing = set(held_out_contexts) - set(np.unique(ctx))
        if missing:
            raise ValueError(f"context(s) not present: {sorted(missing)}")
        test = np.isin(ctx, held_out_contexts)
        held = sorted(held_out_contexts)
        if val_frac:
            raise ValueError(
                "level='context' does not draw a validation set; name the "
                "validation contexts explicitly with a second call"
            )
    else:
        raise ValueError(f"level must be pair, perturbation or context, got {level!r}")

    train = ~test if val is None else ~(test | val)
    split = Split(
        train=train, test=test, level=level, held_out=held,
        val=val, held_out_val=held_val or [],
    )
    _assert_no_pair_leak(meta, split)
    return split


def _assert_no_pair_leak(meta: pd.DataFrame, split: Split) -> None:
    """No (perturbation, context) may appear in more than one half."""
    key = meta.perturbation.astype(str) + "|" + meta.context.astype(str)
    halves = {"train": set(key[split.train]), "test": set(key[split.test])}
    if split.val is not None:
        halves["val"] = set(key[split.val])

    names = list(halves)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = halves[a] & halves[b]
            if both:
                raise AssertionError(
                    f"{len(both)} (perturbation, context) pairs appear in BOTH "
                    f"{a} and {b}, e.g. {sorted(both)[:3]} -- a sample-level leak"
                )

    covered = sum(m.sum() for m in
                  [split.train, split.test] + ([split.val] if split.val is not None else []))
    if int(covered) != len(meta):
        raise AssertionError(
            f"the halves cover {covered} rows but meta has {len(meta)}"
        )


def shuffle_matrix(
    X: np.ndarray | pd.DataFrame, seed: int = 0, frac: float = 1.0
) -> np.ndarray:
    """Generate fake expression rows by permuting each gene column independently.

    A different kind of negative from :func:`permute_labels`. A label-permuted
    negative asks *"does this profile match this label?"*; a shuffled-matrix
    negative asks *"is this a real expression profile at all?"*

    Each gene column is permuted across rows, so per-gene marginals are
    preserved **exactly** and only gene-gene coherence is destroyed -- a draw
    from the product of the marginals. Measured on a 4,000 x 3,000 block of the
    Stattic matrix: separability from per-gene scale alone is Cohen's d = 0.00,
    while row-to-row correlation spread drops from 0.164 to 0.049.

    Shuffling *within* a row (across genes) is deliberately not offered. It
    destroys the per-gene marginals, and per-gene SD spans ~5.4e4x here, so a
    value landing in the wrong gene column is detectable from magnitude alone
    (d = 0.59 on the same block). A discriminator would win on scale and learn
    nothing about biology.

    Note this negative is much easier to detect than a permuted label. Sharing
    one loss between the two lets the easy task dominate; give it a low weight
    or its own head.

    Parameters
    ----------
    frac
        Fraction of gene columns to shuffle, chosen uniformly at random.
        ``1.0`` shuffles every column, so no output row corresponds to any real
        perturbation. Below 1.0 the remaining columns keep their real values,
        which makes the negative harder: the row is genuine over
        ``1 - frac`` of the transcriptome and the discriminator has to notice
        incoherence in the rest. This is the knob for difficulty; ``frac``
        near 0 approaches a real row.
    """
    values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    rng = np.random.default_rng(seed)
    n_rows, n_cols = values.shape

    if frac >= 1.0:
        # Every column, each with its own permutation. The RNG is not touched
        # before the loop, so this reproduces files generated before `frac`
        # existed, byte for byte.
        columns = np.arange(n_cols)
        out = np.empty_like(values)
    else:
        k = max(1, int(round(frac * n_cols)))
        columns = rng.choice(n_cols, k, replace=False)
        out = values.copy()          # unshuffled columns keep their real values

    for j in columns:
        out[:, j] = values[rng.permutation(n_rows), j]
    return out


def build_augmented(
    matrices_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    label_strategies: tuple[str, ...] = STRATEGIES,
    label_seeds: tuple[int, ...] = (0,),
    shuffle_seeds: tuple[int, ...] = (0,),
    contexts: list[str] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Materialise both kinds of augmented data next to the real matrices.

    Two kinds, saved differently because they differ in size by three orders of
    magnitude:

    ``permuted_<strategy>_seed<k>.parquet``
        One pooled table of ~170,800 rows carrying ``perturbation_neg`` /
        ``context_neg``. A few MB -- the expression matrix is *not* copied,
        because a label-permuted negative is the same row with a wrong label.

    ``<context>_shuffled_seed<k>.parquet``
        Fake expression rows, one file per context, ~1.8 GB each. Columns are
        permuted **within a context**, never across the pooled matrix: a fake
        row must carry its own context's per-gene marginals, or the context
        label becomes recoverable from the values and the negative is trivial.

    Returns a manifest of what was written.
    """
    if matrices_dir is None:
        matrices_dir = paths.matrices()
    if out_dir is None:
        out_dir = paths.augmented()
    matrices_dir, out_dir = Path(matrices_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    found = sorted(matrices_dir.glob("*_PosteriorMean.parquet"))
    names = [f.name.replace("_PosteriorMean.parquet", "") for f in found]
    if contexts:
        names = [n for n in names if n in set(contexts)]
    if not names:
        raise FileNotFoundError(f"no matrices under {matrices_dir}")

    meta = pd.concat(
        [pd.read_csv(matrices_dir / f"{n}_row_metadata.csv") for n in names],
        ignore_index=True,
    )
    written = []

    # ---- label permutations: small, pooled across contexts ----
    for strategy in label_strategies:
        for seed in label_seeds:
            path = out_dir / f"permuted_{strategy}_seed{seed}.parquet"
            if path.exists() and not overwrite:
                print(f"[skip] {path.name}", flush=True)
                continue
            aug = permute_labels(meta, strategy=strategy, seed=seed)
            aug.to_parquet(path)
            written.append({"kind": "permuted_labels", "file": path.name,
                            "rows": len(aug), "MB": round(path.stat().st_size / 1e6, 1)})
            print(f"[ok] {path.name}  {len(aug):,} rows", flush=True)

    # ---- shuffled matrices: large, one file per context ----
    for seed in shuffle_seeds:
        for name in names:
            path = out_dir / f"{name}_shuffled_seed{seed}.parquet"
            if path.exists() and not overwrite:
                print(f"[skip] {path.name}", flush=True)
                continue
            block = pd.read_parquet(matrices_dir / f"{name}_PosteriorMean.parquet")
            labels = block.pop("label") if "label" in block.columns else block.index
            fake = pd.DataFrame(
                shuffle_matrix(block, seed=seed),
                columns=block.columns,
            )
            fake.insert(0, "label", np.asarray(labels))
            fake.to_parquet(path)
            written.append({"kind": "shuffled_matrix", "file": path.name,
                            "rows": len(fake), "MB": round(path.stat().st_size / 1e6, 1)})
            print(f"[ok] {path.name}  {fake.shape}", flush=True)
            del block, fake

    return pd.DataFrame(written)


def apply_split(split: Split, meta: pd.DataFrame) -> Split:
    """Re-derive a split's masks for a different row set.

    A synthetic row inherits the ``(perturbation, context)`` of the real row it
    was generated from, so it must land in the **same half** as that pair. If it
    does not, a held-out pair's synthetic twin appears in training and the
    held-out numbers stop meaning anything -- the same class of leak as a
    negative that names a held-out pair, and one :func:`make_split` cannot
    catch, because it splits on ``(perturbation, context)`` and knows nothing
    about ``is_real``.

    Rather than re-drawing (which would put the twins in different halves),
    this re-applies the *units* the original split held out.
    """
    pert = meta.perturbation.astype(str).to_numpy()
    ctx = meta.context.astype(str).to_numpy()
    val = None

    if split.level == "pair":
        held = {(str(a), str(b)) for a, b in split.held_out}
        held_v = {(str(a), str(b)) for a, b in split.held_out_val}
        key = list(zip(pert, ctx))
        test = np.array([k in held for k in key])
        if held_v:
            val = np.array([k in held_v for k in key])
    elif split.level == "perturbation":
        test = np.isin(pert, [str(x) for x in split.held_out])
        if split.held_out_val:
            val = np.isin(pert, [str(x) for x in split.held_out_val])
    elif split.level == "context":
        test = np.isin(ctx, [str(x) for x in split.held_out])
    else:  # pragma: no cover - make_split validates this
        raise ValueError(f"unknown split level {split.level!r}")

    train = ~test if val is None else ~(test | val)
    out = Split(train=train, test=test, level=split.level,
                held_out=split.held_out, val=val,
                held_out_val=split.held_out_val)
    _assert_no_pair_leak(meta, out)
    return out


def load_shuffled(
    contexts,
    meta_real: pd.DataFrame,
    augmented_dir=None,
    seed: int = 0,
    genes=None,
    frac: float = 1.0,
    subsample_seed: int = 0,
):
    """Load ``<context>_shuffled_seed<k>.parquet`` as synthetic training rows.

    These are the column-shuffled matrices from :func:`build_augmented`: per-gene
    marginals preserved exactly, gene-gene covariance destroyed. Each row keeps
    the ``label`` of the real row it was generated from, so it carries a real,
    **assigned** ``(perturbation, context)`` -- the label is correct and only the
    vector is not. That is what makes it a hard negative for ``L_disc``: the head
    cannot reject it on a label mismatch.

    ``frac`` thins the synthetic rows, sampled per context so coverage stays
    even. It is how the negative budget is set: with the two kinds of negative
    pooled into one mean-reduced term, their shares in the loss *are* their
    counts, so loading ``frac`` of the real row count makes the synthetic share
    of the negatives equal ``frac`` without any separate weight. ``frac=0.5``
    with the permuted negatives thinned to match gives one negative per
    positive overall.

    Returns ``(X_fake, meta_fake)`` with ``meta_fake`` carrying the same columns
    as ``meta_real`` so the two can be concatenated. Contexts with no shuffled
    file are skipped and named in the returned frame's ``attrs['missing']``.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    rng = np.random.default_rng(subsample_seed)
    if augmented_dir is None:
        augmented_dir = paths.augmented()
    augmented_dir = Path(augmented_dir)

    ref = meta_real.set_index(["context", "label"])
    blocks, metas, missing = [], [], []
    for ctx in contexts:
        path = augmented_dir / f"{ctx}_shuffled_seed{seed}.parquet"
        if not path.exists():
            missing.append(ctx)
            continue
        block = pd.read_parquet(path)
        labels = block.pop("label").to_numpy()
        if genes is not None:
            block = block[[str(g) for g in genes]]
        idx = pd.MultiIndex.from_arrays(
            [np.full(len(labels), ctx), labels], names=["context", "label"]
        )
        known = idx.isin(ref.index)
        if not known.all():
            block, labels, idx = block[known], labels[known], idx[known]
        rows = ref.loc[idx].reset_index()
        rows["row_id"] = ctx + "|shuffled|" + rows["label"].astype(str)
        values = block.to_numpy(dtype=np.float32)
        if frac < 1.0:
            k = max(1, int(round(frac * len(values))))
            take = np.sort(rng.choice(len(values), k, replace=False))
            values, rows = values[take], rows.iloc[take]
        blocks.append(values)
        metas.append(rows[list(meta_real.columns)])
        del block, values

    if not blocks:
        raise FileNotFoundError(
            f"no *_shuffled_seed{seed}.parquet under {augmented_dir} for "
            f"{list(contexts)}"
        )
    X_fake = np.vstack(blocks)
    meta_fake = pd.concat(metas, ignore_index=True)
    meta_fake.attrs["missing"] = missing
    return X_fake, meta_fake
