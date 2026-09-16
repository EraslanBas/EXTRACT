"""The implementation must match docs/paper/modulefinder.pdf, equation by equation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from module_finder.data.ontarget import mask_coverage, on_target_index
from module_finder.evaluation import span_residual
from module_finder.models import NO_MASK, GlobalLoadings, PerComponentHead
from module_finder.models.label_net import FactorizedLabelNet
from module_finder.objectives import (
    StratifiedNegativeSampler,
    contrastive_loss,
    precision_weights,
)
from module_finder.train import TrainConfig, fit


# ---------------------------------------------------------------- eq. (2)/(4)


def test_unmasked_projection_is_x_times_pinv():
    torch.manual_seed(0)
    L = GlobalLoadings(5, 40, ridge=0.0)
    x = torch.randn(7, 40, dtype=torch.float32)
    z = L.project(x)
    assert torch.allclose(z, x @ L.pinv, atol=1e-4)


def test_unmasked_projection_is_the_least_squares_solution():
    torch.manual_seed(1)
    L = GlobalLoadings(4, 30, ridge=0.0)
    x = torch.randn(3, 30, dtype=torch.float64)
    B = L.B.detach().double()
    z = L.project(x.float()).double()
    expected = torch.linalg.lstsq(B.T, x.T).solution.T
    assert torch.allclose(z, expected, atol=1e-4)


def test_masked_projection_matches_explicit_masked_least_squares():
    """eq. (4) via Sherman-Morrison must equal dropping the column and solving."""
    torch.manual_seed(2)
    L = GlobalLoadings(6, 50, ridge=0.0)
    B = L.B.detach().double()
    x = torch.randn(9, 50, dtype=torch.float64)
    cols = torch.tensor([3, 17, 49, 0, 25, 25, NO_MASK, 8, 41])

    z = L.project(x.float(), cols).double()

    for i in range(len(x)):
        g = int(cols[i])
        keep = torch.ones(50, dtype=torch.bool)
        if g != NO_MASK:
            keep[g] = False
        expected = torch.linalg.lstsq(B[:, keep].T, x[i, keep].unsqueeze(-1)).solution
        assert torch.allclose(z[i], expected.squeeze(-1), atol=1e-4), f"row {i}"


def test_masking_changes_the_projection():
    """If masking were a no-op the on-target diagonal would still be in play."""
    torch.manual_seed(3)
    L = GlobalLoadings(4, 20)
    x = torch.randn(5, 20)
    x[:, 7] = -50.0  # a dominant on-target-like entry
    cols = torch.full((5,), 7)
    assert not torch.allclose(L.project(x), L.project(x, cols), atol=1e-3)


def test_no_mask_sentinel_rows_are_unmasked():
    torch.manual_seed(4)
    L = GlobalLoadings(3, 15)
    x = torch.randn(4, 15)
    cols = torch.full((4,), NO_MASK)
    assert torch.allclose(L.project(x, cols), L.project(x), atol=1e-5)


def test_projection_is_differentiable_through_B():
    L = GlobalLoadings(3, 12)
    x = torch.randn(6, 12)
    cols = torch.tensor([1, 2, NO_MASK, 4, 4, 11])
    L.project(x, cols).sum().backward()
    assert L.B.grad is not None and torch.isfinite(L.B.grad).all()


# -------------------------------------------------------------------- eq. (10)


def test_squared_error_excludes_the_masked_entry():
    torch.manual_seed(5)
    L = GlobalLoadings(3, 10)
    x = torch.randn(4, 10)
    z = L.project(x)
    cols = torch.tensor([2, 2, NO_MASK, 9])

    err = L.squared_error(x, z, cols)
    resid = (x - L.reconstruct(z)).detach()

    for i in range(4):
        g = int(cols[i])
        keep = torch.ones(10, dtype=torch.bool)
        if g != NO_MASK:
            keep[g] = False
        expected = resid[i, keep].pow(2).sum() / keep.sum()
        assert torch.allclose(err[i], expected, atol=1e-5), f"row {i}"


def test_precision_weights_are_proportional_to_cells():
    n = np.array([50.0, 100.0, 200.0, 400.0])
    w = precision_weights(n, clip_quantile=1.0)
    assert np.allclose(w / w[0], n / n[0])
    assert np.isclose(w.mean(), 1.0)


def test_precision_weights_reject_nonpositive_counts():
    with pytest.raises(ValueError):
        precision_weights(np.array([10.0, 0.0]))


# --------------------------------------------------------------- eq. (6)/(7)


def test_head_is_additively_separable():
    """Changing z_k must move the score only through psi_k -- no cross terms."""
    torch.manual_seed(6)
    head = PerComponentHead()
    d, J = 5, head.n_basis
    lam = torch.randn(1, d, J)
    z = torch.randn(1, d)

    base = head(z, lam)
    total_delta = 0.0
    for k in range(d):
        z_k = z.clone()
        z_k[0, k] += 0.5
        total_delta += float(head(z_k, lam) - base)

    z_all = z.clone() + 0.0
    z_all[0, :] += 0.5
    joint_delta = float(head(z_all, lam) - base)
    assert abs(joint_delta - total_delta) < 1e-4


def test_purely_quadratic_basis_is_rejected():
    with pytest.raises(ValueError, match="rotation invariant"):
        PerComponentHead(basis=("square",))


def test_default_basis_has_four_statistics():
    assert PerComponentHead().n_basis == 4


def test_component_evidence_sums_to_the_score():
    torch.manual_seed(7)
    d = 4
    head = PerComponentHead()
    net = FactorizedLabelNet(6, 3, d, head.n_basis, embedding_dim=8, hidden=0)
    z = torch.randn(5, d)
    p = torch.tensor([0, 1, 2, 3, 4])
    c = torch.tensor([0, 1, 2, 0, 1])
    lam = net(p, c)
    psi = (lam * head.statistics(z)).sum(dim=-1)
    assert torch.allclose(psi.sum(dim=1) + head.bias, head(z, lam), atol=1e-5)


def test_label_net_shares_embeddings_rather_than_memorising_pairs():
    net = FactorizedLabelNet(100, 8, 10, 4, embedding_dim=16, hidden=0)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params < 100 * 8 * 10  # far below one row per observed pair


# ----------------------------------------------------------------- negatives


def _toy_labels(n_pert=12, n_ctx=4, n_strata=11):
    p, c, s = [], [], []
    for pi in range(n_pert):
        for ci in range(n_ctx):
            for si in range(n_strata):
                p.append(pi), c.append(ci), s.append(si)
    return np.array(p), np.array(c), np.array(s)


def _sampler(weights=None, vehicles=(0,)):
    p, c, s = _toy_labels()
    return (
        StratifiedNegativeSampler(
            perturbation_idx=p,
            context_idx=c,
            stratum=s,
            n_perturbations=12,
            n_contexts=4,
            vehicle_context_idx=vehicles,
            **({"weights": weights} if weights else {}),
        ),
        p,
        c,
        s,
    )


def test_negatives_never_equal_their_own_label():
    sampler, p, c, _ = _sampler()
    rng = np.random.default_rng(0)
    rows = rng.permutation(len(p))[:2000]
    neg_p, neg_c = sampler.sample(rows, rng)
    collisions = (neg_p == p[rows]) & (neg_c == c[rows])
    assert collisions.sum() == 0, f"{collisions.sum()} false negatives"


def test_other_pert_strategy_keeps_the_context():
    sampler, p, c, _ = _sampler(weights={"same_s_other_pert": 1.0}, vehicles=())
    rng = np.random.default_rng(1)
    rows = np.arange(500)
    neg_p, neg_c = sampler.sample(rows, rng)
    assert np.array_equal(neg_c, c[rows])
    assert (neg_p != p[rows]).all()


def test_other_context_strategy_keeps_the_perturbation():
    sampler, p, c, _ = _sampler(weights={"same_s_other_context": 1.0}, vehicles=())
    rng = np.random.default_rng(2)
    rows = np.arange(500)
    neg_p, neg_c = sampler.sample(rows, rng)
    assert np.array_equal(neg_p, p[rows])
    assert (neg_c != c[rows]).all()


def test_against_vehicle_uses_a_vehicle_context():
    sampler, _, _, _ = _sampler(weights={"against_vehicle": 1.0}, vehicles=(0, 1))
    rng = np.random.default_rng(3)
    rows = np.arange(400)
    _, neg_c = sampler.sample(rows, rng)
    assert set(np.unique(neg_c)).issubset({0, 1})


def test_against_vehicle_requires_vehicle_indices():
    p, c, s = _toy_labels()
    with pytest.raises(ValueError, match="vehicle_context_idx"):
        StratifiedNegativeSampler(
            perturbation_idx=p,
            context_idx=c,
            stratum=s,
            n_perturbations=12,
            n_contexts=4,
            weights={"against_vehicle": 1.0},
        )


def test_stratification_narrows_the_precision_gap():
    """The measurement that justifies stratifying: median |log2 ratio| should
    be well below the unstratified figure."""
    rng = np.random.default_rng(4)
    n_pert, n_ctx, n_strata = 40, 3, 11
    p, c, s, n_cells = [], [], [], []
    depth = rng.integers(60, 4000, n_pert)  # per-perturbation cell counts
    for pi in range(n_pert):
        for ci in range(n_ctx):
            for si in range(n_strata):
                p.append(pi), c.append(ci), s.append(si)
                frac = 1.0 if si == 0 else 0.1 + 0.08 * (si - 1)
                n_cells.append(max(50, int(depth[pi] * frac)))
    p, c, s = np.array(p), np.array(c), np.array(s)
    n_cells = np.array(n_cells, dtype=float)

    stratified = StratifiedNegativeSampler(
        perturbation_idx=p, context_idx=c, stratum=s,
        n_perturbations=n_pert, n_contexts=n_ctx,
        weights={"same_s_other_pert": 1.0},
    )
    rows = rng.permutation(len(p))[:3000]
    neg_p, neg_c = stratified.sample(rows, rng)
    strat_gap = np.nanmedian(
        stratified.precision_mismatch(rows, neg_p, neg_c, n_cells)
    )

    # Unstratified: a perturbation drawn without regard to stratum.
    # Unstratified baseline: same negative perturbations, but the negative
    # names a row at a *random* stratum -- what a sampler that ignores the
    # stratum effectively does.
    naive_s = rng.integers(0, n_strata, len(rows))
    naive_gap = np.nanmedian(
        stratified.precision_mismatch(
            rows, neg_p, neg_c, n_cells, neg_stratum=naive_s
        )
    )
    assert strat_gap < naive_gap, f"stratified {strat_gap:.3f} vs naive {naive_gap:.3f}"


# ----------------------------------------------------------------- on-target


def test_on_target_index_matches_bare_and_subsampled_labels():
    genes = np.array(["A1BG", "BRCA1", "TP53"])
    perts = np.array(["A1BG", "A1BG__sub03", "TP53__sub10", "non-targeting", "XYZ"])
    cols = on_target_index(perts, genes)
    assert cols.tolist() == [0, 0, 2, NO_MASK, NO_MASK]


def test_mask_coverage_reports_the_matched_fraction():
    cov = mask_coverage(np.array([0, 0, 2, NO_MASK]))
    assert cov["n_masked"] == 3
    assert cov["n_distinct_genes"] == 2


# --------------------------------------------------------------- eq. (14)


def test_span_residual_is_zero_inside_the_span():
    rng = np.random.default_rng(5)
    B = rng.normal(size=(4, 30))
    Z = rng.normal(size=(10, 4))
    assert np.allclose(span_residual(B, Z @ B), 0.0, atol=1e-8)


def test_span_residual_is_positive_outside_the_span():
    rng = np.random.default_rng(6)
    B = rng.normal(size=(3, 40))
    X = rng.normal(size=(12, 40))
    assert (span_residual(B, X) > 0.5).all()


# ------------------------------------------------------------------ eq. (12)


def _toy_dataset(n_pert=20, n_ctx=3, n_strata=4, n_genes=60, d_true=5, seed=0):
    rng = np.random.default_rng(seed)
    B_true = rng.normal(size=(d_true, n_genes))
    rows, p, c, s, n_cells = [], [], [], [], []
    mu = rng.normal(size=(n_pert, d_true))
    gamma = rng.normal(size=(n_ctx, d_true)) * 0.5
    for pi in range(n_pert):
        for ci in range(n_ctx):
            for si in range(n_strata):
                n = 3000 if si == 0 else int(80 * (si + 1))
                z = mu[pi] + gamma[ci]
                noise = rng.normal(size=n_genes) / np.sqrt(n)
                rows.append(z @ B_true + noise)
                p.append(pi), c.append(ci), s.append(si), n_cells.append(n)
    return (
        np.asarray(rows, dtype=np.float32),
        np.array(p), np.array(c), np.array(s), np.array(n_cells, dtype=float),
    )


def test_fit_runs_and_reduces_both_terms():
    X, p, c, s, n = _toy_dataset()
    cfg = TrainConfig(n_factors=5, epochs=12, batch_size=64, log_every=0, seed=0)
    model, history = fit(X, p, c, s, n, target_col=None, config=cfg)

    assert len(history) == 12
    assert history[-1]["recon"] < history[0]["recon"]
    assert history[-1]["disc"] < history[0]["disc"]
    assert model.loading_matrix().shape == (5, X.shape[1])


def test_fit_accepts_a_mask_and_a_vehicle_context():
    X, p, c, s, n = _toy_dataset(n_pert=12, n_genes=40)
    target_col = (p % 40).astype(np.int64)
    cfg = TrainConfig(
        n_factors=4, epochs=4, batch_size=48, log_every=0,
        vehicle_context_idx=(0,),
        negative_weights={"same_s_other_pert": 0.5, "against_vehicle": 0.5},
    )
    model, history = fit(X, p, c, s, n, target_col=target_col, config=cfg)
    assert np.isfinite(history[-1]["recon"])
    assert model.loading_matrix().shape == (4, 40)


def test_fit_rejects_mismatched_array_lengths():
    X, p, c, s, n = _toy_dataset(n_pert=6, n_genes=20)
    with pytest.raises(ValueError, match="stratum"):
        fit(X, p, c, s[:-1], n, config=TrainConfig(epochs=1, log_every=0))


def test_optional_tc_term_runs():
    X, p, c, s, n = _toy_dataset(n_pert=10, n_genes=30)
    cfg = TrainConfig(
        n_factors=4, epochs=3, batch_size=32, log_every=0, tc_weight=0.1
    )
    _, history = fit(X, p, c, s, n, config=cfg)
    assert history[-1]["tc"] != 0.0


def test_contrastive_loss_is_minimised_by_confident_correct_logits():
    good = contrastive_loss(torch.tensor([6.0, 6.0]), torch.tensor([-6.0, -6.0]))
    bad = contrastive_loss(torch.tensor([-6.0, -6.0]), torch.tensor([6.0, 6.0]))
    assert float(good) < 0.01 < float(bad)


def test_unknown_strategy_rejected():
    p, c, s = _toy_labels()
    with pytest.raises(ValueError, match="unknown strategies"):
        StratifiedNegativeSampler(
            perturbation_idx=p, context_idx=c, stratum=s,
            n_perturbations=12, n_contexts=4, weights={"nonsense": 1.0},
        )


def test_zero_weights_rejected():
    p, c, s = _toy_labels()
    with pytest.raises(ValueError, match="positive"):
        StratifiedNegativeSampler(
            perturbation_idx=p, context_idx=c, stratum=s,
            n_perturbations=12, n_contexts=4,
            weights={"same_s_other_pert": 0.0},
        )


# -------------------------------------------------------------- train/test


def _toy_meta(n_pert=12, n_ctx=4, n_strata=3):
    import pandas as pd
    rows = []
    for pi in range(n_pert):
        for ci in range(n_ctx):
            for si in range(n_strata):
                lab = f"P{pi}" if si == 0 else f"P{pi}__sub{si-1:02d}"
                rows.append(
                    {"perturbation": f"P{pi}", "context": f"C{ci}",
                     "label": lab, "n_cells": 3000 if si == 0 else 100 * si}
                )
    return pd.DataFrame(rows)


def test_fit_trains_only_on_train_rows():
    """A held-out pair must never appear in a training batch."""
    from module_finder.data.augment import make_split

    meta = _toy_meta()
    X, p, c, s, n = _toy_dataset(n_pert=12, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.2, seed=0)

    pairs = list(zip(meta.perturbation, meta.context))
    train_pairs = {pc for pc, m in zip(pairs, split.train) if m}
    test_pairs = {pc for pc, m in zip(pairs, split.test) if m}
    assert not (train_pairs & test_pairs)

    cfg = TrainConfig(n_factors=4, epochs=4, batch_size=32, log_every=0,
                      eval_every=2, patience=0)
    model, history = fit(
        X, p, c, s, n, config=cfg,
        train_rows=split.train, val_rows=split.test,
    )
    assert "val_accuracy" in history[-1]
    assert history[-1]["val_n_rows"] == int(split.test.sum())


def test_all_eleven_samples_of_a_held_out_pair_move_together():
    from module_finder.data.augment import make_split

    meta = _toy_meta(n_pert=20, n_ctx=3, n_strata=11)
    split = make_split(meta, level="pair", test_frac=0.15, seed=1)
    grouped = meta.assign(test=split.test).groupby(
        ["perturbation", "context"]
    )["test"].nunique()
    assert (grouped == 1).all(), "a pair was split across train and test"


def test_fit_rejects_a_split_smaller_than_one_batch():
    X, p, c, s, n = _toy_dataset(n_pert=6, n_ctx=2, n_strata=2, n_genes=20)
    train = np.zeros(len(X), dtype=bool)
    train[:10] = True
    with pytest.raises(ValueError, match="training rows"):
        fit(X, p, c, s, n, config=TrainConfig(batch_size=256, epochs=1,
                                              log_every=0),
            train_rows=train)


def test_evaluate_returns_the_documented_keys():
    from module_finder.objectives import StratifiedNegativeSampler
    from module_finder.train import evaluate

    X, p, c, s, n = _toy_dataset(n_pert=10, n_ctx=3, n_strata=3, n_genes=30)
    cfg = TrainConfig(n_factors=4, epochs=2, batch_size=32, log_every=0,
                      eval_every=0)
    model, _ = fit(X, p, c, s, n, config=cfg)

    rows = np.arange(min(200, len(X)))
    sampler = StratifiedNegativeSampler(
        perturbation_idx=p[rows], context_idx=c[rows], stratum=s[rows],
        n_perturbations=int(p.max()) + 1, n_contexts=int(c.max()) + 1,
        weights={"same_s_other_pert": 1.0},
    )
    m = evaluate(model, torch.from_numpy(X), p, c, rows, sampler,
                 np.random.default_rng(0))
    for key in ("n_rows", "accuracy", "disc", "recon", "real_score_mean",
                "fake_score_mean", "span_residual_median", "span_residual_q90"):
        assert key in m, key
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["span_residual_median"] <= 1.5


# ------------------------------------------------------- three-way split


def test_three_way_split_is_disjoint_and_complete():
    from module_finder.data.augment import make_split

    meta = _toy_meta(n_pert=40, n_ctx=4, n_strata=11)
    split = make_split(meta, level="pair", test_frac=0.1, val_frac=0.1, seed=0)
    assert split.three_way
    assert int(split.train.sum() + split.val.sum() + split.test.sum()) == len(meta)
    assert not (split.train & split.val).any()
    assert not (split.train & split.test).any()
    assert not (split.val & split.test).any()
    assert not set(split.held_out) & set(split.held_out_val)


def test_three_way_split_keeps_pairs_whole():
    from module_finder.data.augment import make_split

    meta = _toy_meta(n_pert=30, n_ctx=3, n_strata=11)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=3)
    half = np.select(
        [split.train, split.val, split.test], ["train", "val", "test"],
        default="none",
    )
    assert "none" not in set(half)
    per_pair = (
        meta.assign(half=half).groupby(["perturbation", "context"])["half"].nunique()
    )
    assert (per_pair == 1).all()


def test_split_rejects_impossible_fractions():
    from module_finder.data.augment import make_split

    meta = _toy_meta(n_pert=4, n_ctx=1, n_strata=2)
    with pytest.raises(ValueError, match="would hold out"):
        make_split(meta, level="pair", test_frac=0.7, val_frac=0.7, seed=0)


def test_early_stopping_restores_the_best_epoch():
    """The last epoch is not the model we want; the selected one is."""
    from module_finder.data.augment import make_split

    meta = _toy_meta(n_pert=16, n_ctx=4, n_strata=3)
    X, p, c, s, n = _toy_dataset(n_pert=16, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=0)

    cfg = TrainConfig(n_factors=4, epochs=30, batch_size=32, log_every=0,
                      eval_every=1, patience=2, seed=0)
    model, history = fit(X, p, c, s, n, config=cfg,
                         train_rows=split.train, val_rows=split.val)

    evaluated = [h for h in history if "val_accuracy" in h]
    best = max(evaluated, key=lambda h: h["val_accuracy"])
    assert history[-1]["selected_epoch"] == best["epoch"]
    # patience must actually have been able to fire
    assert len(history) <= 30


def test_fit_rejects_overlapping_train_and_val():
    X, p, c, s, n = _toy_dataset(n_pert=12, n_ctx=3, n_strata=3, n_genes=30)
    mask = np.zeros(len(X), dtype=bool)
    mask[:] = True
    with pytest.raises(ValueError, match="overlap"):
        fit(X, p, c, s, n, config=TrainConfig(epochs=1, batch_size=32,
                                              log_every=0),
            train_rows=mask, val_rows=mask)
