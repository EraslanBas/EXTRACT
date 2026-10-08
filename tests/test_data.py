"""Pseudo-replicates, gene filtering and label augmentation."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from extract.data import build_pseudoreplicates


def _adata(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        "target_gene": rng.choice(["non-targeting", "A", "B"], n, p=[0.5, 0.25, 0.25]),
        "drug": rng.choice(["DMSO", "Stattic"], n),
    })
    return ad.AnnData(
        X=rng.normal(size=(n, 40)),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(40)]),
    )


def test_pseudoreplicates_are_fixed_n_and_disjoint():
    ps = build_pseudoreplicates(
        _adata(), "target_gene", "drug", "non-targeting", n_cells=40, n_replicates=3
    )
    assert ps.n_cells == 40
    pairs = set(zip(ps.perturbations, ps.contexts))
    assert len(pairs) == 4                      # {A,B} x {DMSO,Stattic}
    assert ps.n_rows == len(pairs) * 3
    assert set(ps.replicate) == {0, 1, 2}


def test_pairs_below_the_power_floor_are_skipped_not_shrunk():
    """An under-powered pair is dropped, never given fewer/smaller replicates --
    varying n between pairs would leak the label through noise magnitude."""
    rng = np.random.default_rng(0)
    labels = ["non-targeting"] * 600 + ["abundant"] * 600 + ["rare"] * 20
    obs = pd.DataFrame({
        "target_gene": labels,
        # assign context at random, not by row position -- positional
        # assignment can leave a context with no control cells at all
        "drug": rng.choice(["DMSO", "Stattic"], len(labels)),
    })
    adata = ad.AnnData(
        X=rng.normal(size=(len(labels), 40)),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(40)]),
    )
    ps = build_pseudoreplicates(
        adata, "target_gene", "drug", "non-targeting", n_cells=50, n_replicates=3
    )
    assert "rare" not in set(ps.perturbations)          # dropped
    assert "abundant" in set(ps.perturbations)          # kept
    assert any(p == "rare" for p, _, _ in ps.skipped_)  # and reported
    # every surviving pair has the full replicate count
    counts = pd.Series(list(zip(ps.perturbations, ps.contexts))).value_counts()
    assert (counts == 3).all()


def test_impossible_floor_raises():
    with pytest.raises(ValueError, match="cell-count floor"):
        build_pseudoreplicates(
            _adata(n=200), "target_gene", "drug", "non-targeting",
            n_cells=500, n_replicates=5,
        )


# ---------------------------------------------- subset shuffling, gene filter


def test_shuffle_matrix_frac_leaves_the_rest_real():
    import numpy as np
    from extract.data.augment import shuffle_matrix

    X = np.arange(200, dtype=float).reshape(20, 10)
    out = shuffle_matrix(X, seed=0, frac=0.3)
    kept = [j for j in range(10) if np.array_equal(out[:, j], X[:, j])]
    shuffled = [j for j in range(10) if not np.array_equal(out[:, j], X[:, j])]
    assert len(shuffled) == 3, f"expected 3 shuffled columns, got {len(shuffled)}"
    assert len(kept) == 7
    # every column is still a permutation of its own real values
    for j in range(10):
        assert np.array_equal(np.sort(out[:, j]), np.sort(X[:, j]))


def test_shuffle_matrix_frac_one_is_unchanged_behaviour():
    """frac=1.0 must not consume RNG differently, or previously generated
    augmented files stop being reproducible."""
    import numpy as np
    from extract.data.augment import shuffle_matrix

    X = np.arange(120, dtype=float).reshape(12, 10)
    a = shuffle_matrix(X, seed=7)
    b = shuffle_matrix(X, seed=7, frac=1.0)
    assert np.array_equal(a, b)
    for j in range(10):
        assert not np.array_equal(a[:, j], X[:, j])


def test_shuffle_matrix_rejects_bad_frac():
    import numpy as np
    import pytest
    from extract.data.augment import shuffle_matrix

    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="frac"):
            shuffle_matrix(np.zeros((4, 4)), frac=bad)


def test_affected_counts_uses_main_rows_only():
    """Subsample rows are re-estimates of the same pair, so counting them would
    multiply every gene's count without adding information."""
    import numpy as np
    import pandas as pd
    from extract.data import affected_counts

    X = pd.DataFrame(
        [[0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [0.0, 0.4]],
        columns=["A", "B"],
    )
    meta = pd.DataFrame({"variant": ["main", "subsample", "subsample", "main"]})
    counts = affected_counts(X, meta, threshold=0.1)
    assert counts["A"] == 1          # one main row, not three
    assert counts["B"] == 1
    all_rows = affected_counts(X, meta, threshold=0.1, variant=None)
    assert all_rows["A"] == 3


def test_select_genes_applies_the_minimum():
    import numpy as np
    import pandas as pd
    from extract.data import select_genes

    X = pd.DataFrame(
        [[0.5, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.05]],
        columns=["keep", "borderline", "drop"],
    )
    meta = pd.DataFrame({"variant": ["main"] * 3})
    assert list(select_genes(X, meta, 3, threshold=0.1)) == [True, False, False]
    assert list(select_genes(X, meta, 1, threshold=0.1)) == [True, True, False]


def test_affected_counts_rejects_misaligned_meta():
    import numpy as np
    import pandas as pd
    import pytest
    from extract.data import affected_counts

    X = pd.DataFrame([[0.5], [0.5]], columns=["A"])
    with pytest.raises(ValueError, match="align"):
        affected_counts(X, pd.DataFrame({"variant": ["main"]}), threshold=0.1)


def test_permute_labels_only_names_observed_pairs():
    """A negative must never name a (p, c) pair that was not measured, or
    tables built from full metadata would name held-out pairs."""
    from extract.data.augment import permute_labels

    drugs = ["DrugX", "DrugY"]
    rows = []
    for i in range(40):
        p = f"P{i:02d}"
        # deliberately uneven: only some perturbations appear in each context
        ctxs = list(drugs)
        if i % 3 == 0:
            ctxs.append("DrugZ")
        if i % 5 == 0:
            ctxs.append("DrugW")
        for c in ctxs:
            for s in range(2):
                rows.append({"perturbation": p, "context": c,
                             "label": f"{p}__sub{s:02d}" if s else p})
    meta = pd.DataFrame(rows)
    observed = set(zip(meta.perturbation, meta.context))

    for strategy in ("same_s_other_pert", "same_s_other_context"):
        out = permute_labels(meta, strategy=strategy, seed=0)
        emitted = set(zip(out.perturbation_neg, out.context_neg))
        unseen = emitted - observed
        assert not unseen, f"{strategy} emitted unobserved pairs: {sorted(unseen)[:3]}"
        same = (out.perturbation_neg == out.perturbation) & \
               (out.context_neg == out.context)
        assert not same.any(), f"{strategy} kept {int(same.sum())} own labels"


def test_disjoint_scheme_gives_non_overlapping_equal_size_pseudobulks():
    from extract.de.shards import plan_shard_rows
    p = np.array(["A"] * 350 + ["B"] * 1500 + ["C"] * 90)
    cell_index, labels, plan = plan_shard_rows(p, 10, scheme="disjoint", replicate_cells=100, seed=1)
    sub = plan[plan.variant == "subsample"]
    assert set(sub.n_cells_planned) == {100}
    assert sub.groupby("perturbation").size().to_dict() == {"A": 3, "B": 10}
    for pert in ("A", "B"):
        groups = [set(cell_index[labels == lab]) for lab in sub.label[sub.perturbation == pert]]
        assert sum(len(a & b) for i, a in enumerate(groups) for b in groups[i + 1:]) == 0
        assert all(g <= set(np.flatnonzero(p == pert)) for g in groups)
