"""Pseudo-replicates and gene standardisation."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from module_finder.data import build_pseudoreplicates, standardize_genes


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


def test_standardize_handles_huge_scale_spread():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 20)) * np.logspace(-6, 1, 20)
    Xs, scale = standardize_genes(X)
    assert scale.max() / scale.min() > 1e3     # original spread was enormous
    assert Xs.std(0).max() <= 1.0 + 1e-9       # and is gone afterwards
    assert np.isfinite(Xs).all()


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
