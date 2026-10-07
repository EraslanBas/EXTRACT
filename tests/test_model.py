"""The implementation must match the method paper, equation by equation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from extract.data.ontarget import mask_coverage, on_target_index
from extract.evaluation import span_residual
from extract.models import NO_MASK, GlobalLoadings, PerComponentHead
from extract.models.label_net import FactorizedLabelNet, ProductLabelNet
from extract.objectives import (
    DEFAULT_WEIGHTS,
    StratifiedNegativeSampler,
    contrastive_loss,
    precision_weights,
)
from extract.train import TrainConfig, fit


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


def test_product_label_net_is_a_per_factor_product():
    torch.manual_seed(0)
    net = ProductLabelNet(6, 3, 5, 4)
    with torch.no_grad():
        net.f.weight.normal_(); net.g.weight.normal_()
    p, c = torch.tensor([1, 1, 4, 4]), torch.tensor([0, 2, 0, 2])
    lam = net(p, c)
    assert lam.shape == (4, 5, 4)
    # rank one in (p, c) for every factor and statistic
    assert torch.allclose(lam[0] * lam[3], lam[1] * lam[2], atol=1e-5)


def test_fit_runs_with_the_product_label_model():
    X, p, c, s, n = _toy_dataset()
    cfg = TrainConfig(n_factors=5, epochs=6, batch_size=64, log_every=0, seed=0,
                      label_model="product", noise_rank=3)
    model, history = fit(X, p, c, s, n, target_col=None, config=cfg)
    assert isinstance(model.label_net, ProductLabelNet)
    assert history[-1]["disc"] < history[0]["disc"]


# ----------------------------------------------------------------- negatives


def _toy_labels(n_pert=12, n_ctx=4, n_strata=11):
    p, c, s = [], [], []
    for pi in range(n_pert):
        for ci in range(n_ctx):
            for si in range(n_strata):
                p.append(pi), c.append(ci), s.append(si)
    return np.array(p), np.array(c), np.array(s)


def _sampler(weights=None):
    p, c, s = _toy_labels()
    return (
        StratifiedNegativeSampler(
            perturbation_idx=p,
            context_idx=c,
            stratum=s,
            n_perturbations=12,
            n_contexts=4,
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
    sampler, p, c, _ = _sampler(weights={"same_s_other_pert": 1.0})
    rng = np.random.default_rng(1)
    rows = np.arange(500)
    neg_p, neg_c = sampler.sample(rows, rng)
    assert np.array_equal(neg_c, c[rows])
    assert (neg_p != p[rows]).all()


def test_other_context_strategy_keeps_the_perturbation():
    sampler, p, c, _ = _sampler(weights={"same_s_other_context": 1.0})
    rng = np.random.default_rng(2)
    rows = np.arange(500)
    neg_p, neg_c = sampler.sample(rows, rng)
    assert np.array_equal(neg_p, p[rows])
    assert (neg_c != c[rows]).all()


def test_single_context_perturbation_swaps_the_other_axis():
    """A perturbation with one observed context has no alternative context.
    The sampler must swap its perturbation instead of drawing a context from
    the full range, which would name an unobserved (possibly held-out) pair."""
    # perturbation 0 only in context 0; perturbations 1-3 in contexts 0-2
    rows = [(0, 0, s) for s in range(2)]
    rows += [(pp, cc, s) for pp in (1, 2, 3) for cc in range(3) for s in range(2)]
    P, C, S = (np.array(v) for v in zip(*rows))
    sampler = StratifiedNegativeSampler(
        perturbation_idx=P, context_idx=C, stratum=S,
        n_perturbations=4, n_contexts=3,
        weights={"same_s_other_context": 1.0},
    )
    observed = set(zip(P.tolist(), C.tolist()))
    rng = np.random.default_rng(0)
    for _ in range(50):
        neg_p, neg_c = sampler.sample(np.arange(len(P)), rng)
        named = set(zip(neg_p.tolist(), neg_c.tolist()))
        assert named <= observed, f"unobserved pairs named: {named - observed}"
        assert not ((neg_p == P) & (neg_c == C)).any()
    # the single-context rows got a perturbation swap, context untouched
    assert (neg_c[:2] == 0).all() and (neg_p[:2] != 0).all()


def test_label_with_no_observed_alternative_raises():
    """If neither axis has an observed alternative there is no legal negative;
    fail loudly rather than name an unobserved pair."""
    sampler = StratifiedNegativeSampler(
        perturbation_idx=np.array([0, 0]), context_idx=np.array([0, 0]),
        stratum=np.array([0, 1]), n_perturbations=2, n_contexts=2,
    )
    with pytest.raises(ValueError, match="no observed alternative"):
        sampler.sample(np.arange(2), np.random.default_rng(0))


def test_unknown_strategy_is_rejected():
    p, c, s = _toy_labels()
    with pytest.raises(ValueError, match="unknown strategies"):
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
    cfg = TrainConfig(subspace="free", n_factors=5, epochs=12, batch_size=64, log_every=0, seed=0)
    model, history = fit(X, p, c, s, n, target_col=None, config=cfg)

    assert len(history) == 12
    assert history[-1]["recon"] < history[0]["recon"]
    assert history[-1]["disc"] < history[0]["disc"]
    assert model.loading_matrix().shape == (5, X.shape[1])


def test_fit_accepts_a_mask_and_custom_negative_weights():
    X, p, c, s, n = _toy_dataset(n_pert=12, n_genes=40)
    target_col = (p % 40).astype(np.int64)
    cfg = TrainConfig(
        n_factors=4, epochs=4, batch_size=48, log_every=0,
        negative_weights={"same_s_other_pert": 0.8, "same_s_other_context": 0.2},
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
    from extract.data.augment import make_split

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
    from extract.data.augment import make_split

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
    from extract.objectives import StratifiedNegativeSampler
    from extract.train import evaluate

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
    assert m["n_synthetic"] == 0 and np.isnan(m["synth_accuracy"])


def test_evaluate_never_scores_synthetic_rows_as_positives():
    """Synthetic rows are negatives under their own label. The permuted-label
    metrics must be exactly what the measured rows alone give, and the
    synthetic rows must be reported on their own."""
    from extract.train import evaluate

    X, p, c, s, n = _toy_dataset(n_pert=10, n_ctx=3, n_strata=3, n_genes=30)
    cfg = TrainConfig(n_factors=4, epochs=2, batch_size=32, log_every=0,
                      eval_every=0)
    model, _ = fit(X, p, c, s, n, config=cfg)

    k = min(200, len(X))
    rng = np.random.default_rng(1)
    fake = np.stack([X[rng.permutation(k), j] for j in range(X.shape[1])], 1)
    Xb = torch.from_numpy(np.vstack([X[:k], fake]).astype(np.float32))
    cat = lambda a: np.concatenate([a[:k], a[:k]])
    pb, cb, sb = cat(p), cat(c), cat(s)
    flag = np.r_[np.ones(k, bool), np.zeros(k, bool)]

    def _eval(rows, is_real):
        sampler = StratifiedNegativeSampler(
            perturbation_idx=pb[rows], context_idx=cb[rows], stratum=sb[rows],
            n_perturbations=int(p.max()) + 1, n_contexts=int(c.max()) + 1,
        )
        return evaluate(model, Xb, pb, cb, rows, sampler,
                        np.random.default_rng(0), is_real=is_real)

    real_only = _eval(np.arange(k), None)
    mixed = _eval(np.arange(2 * k), flag)
    for key in ("n_rows", "accuracy", "disc", "recon", "real_score_mean",
                "fake_score_mean", "span_residual_median"):
        assert mixed[key] == pytest.approx(real_only[key]), key
    assert mixed["n_synthetic"] == k
    assert 0.0 <= mixed["synth_accuracy"] <= 1.0
    assert np.isfinite(mixed["disc_synth"])


# ------------------------------------------------------- three-way split


def test_three_way_split_is_disjoint_and_complete():
    from extract.data.augment import make_split

    meta = _toy_meta(n_pert=40, n_ctx=4, n_strata=11)
    split = make_split(meta, level="pair", test_frac=0.1, val_frac=0.1, seed=0)
    assert split.three_way
    assert int(split.train.sum() + split.val.sum() + split.test.sum()) == len(meta)
    assert not (split.train & split.val).any()
    assert not (split.train & split.test).any()
    assert not (split.val & split.test).any()
    assert not set(split.held_out) & set(split.held_out_val)


def test_three_way_split_keeps_pairs_whole():
    from extract.data.augment import make_split

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
    from extract.data.augment import make_split

    meta = _toy_meta(n_pert=4, n_ctx=1, n_strata=2)
    with pytest.raises(ValueError, match="would hold out"):
        make_split(meta, level="pair", test_frac=0.7, val_frac=0.7, seed=0)


def test_early_stopping_restores_the_best_epoch():
    """The last epoch is not the model we want; the selected one is. Checked
    for BOTH selection rules -- "accuracy" maximises, "total" (eq. 12, the
    quantity actually minimised) minimises and therefore stops at the
    overfitting onset. The two can pick different epochs."""
    from extract.data.augment import make_split

    meta = _toy_meta(n_pert=16, n_ctx=4, n_strata=3)
    X, p, c, s, n = _toy_dataset(n_pert=16, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=0)

    for metric, pick in (("accuracy", max), ("total", min)):
        cfg = TrainConfig(n_factors=4, epochs=30, batch_size=32, log_every=0,
                          eval_every=1, patience=2, seed=0, select_on=metric)
        model, history = fit(X, p, c, s, n, config=cfg,
                             train_rows=split.train, val_rows=split.val)
        key = f"val_{metric}"
        evaluated = [h for h in history if key in h]
        best = pick(evaluated, key=lambda h: h[key])
        assert history[-1]["selected_epoch"] == best["epoch"], metric
        assert len(history) <= 30


def test_total_is_the_objective_on_both_train_and_val():
    """`total` must equal disc + alpha*recon on both sides, so the two curves
    are the same quantity and their crossing is meaningful."""
    from extract.data.augment import make_split

    meta = _toy_meta(n_pert=16, n_ctx=4, n_strata=3)
    X, p, c, s, n = _toy_dataset(n_pert=16, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=0)
    alpha = 700.0
    cfg = TrainConfig(n_factors=4, alpha=alpha, epochs=6, batch_size=32,
                      log_every=0, eval_every=1, patience=0, seed=0)
    _, history = fit(X, p, c, s, n, config=cfg,
                     train_rows=split.train, val_rows=split.val)
    for h in history:
        assert abs(h["total"] - (h["disc"] + alpha * h["recon"])) < 1e-6
        if "val_total" in h:
            assert abs(h["val_total"]
                       - (h["val_disc"] + alpha * h["val_recon"])) < 1e-6



def test_fit_rejects_overlapping_train_and_val():
    X, p, c, s, n = _toy_dataset(n_pert=12, n_ctx=3, n_strata=3, n_genes=30)
    mask = np.zeros(len(X), dtype=bool)
    mask[:] = True
    with pytest.raises(ValueError, match="overlap"):
        fit(X, p, c, s, n, config=TrainConfig(epochs=1, batch_size=32,
                                              log_every=0),
            train_rows=mask, val_rows=mask)


# --- is_real gate and masked span residual ---------------------------------


def test_span_residual_excludes_the_masked_entry():
    """B is trained never to reconstruct the on-target entry, so charging the
    residual for it inflates r. Masked and unmasked must differ, and masking
    must match an explicit computation that drops the column."""
    torch.manual_seed(0)
    L = GlobalLoadings(3, 12, ridge=1e-4)
    x = torch.randn(5, 12)
    col = torch.full((5,), NO_MASK, dtype=torch.long)
    col[:3] = torch.tensor([0, 4, 9])

    r_masked = L.span_residual(x, col)
    r_plain = L.span_residual(x)
    assert not torch.allclose(r_masked[:3], r_plain[:3])
    assert torch.allclose(r_masked[3:], r_plain[3:])       # unmasked rows unchanged

    z = L.project(x, col)
    resid = (x - L.reconstruct(z)).detach()
    xm = x.clone()
    for i in range(3):
        resid[i, col[i]] = 0.0
        xm[i, col[i]] = 0.0
    expect = torch.linalg.norm(resid, dim=1) / torch.linalg.norm(xm, dim=1)
    assert torch.allclose(r_masked, expect, atol=1e-6)


def test_is_real_false_rows_are_excluded_from_reconstruction():
    """Rows flagged not-real must not shape B. Appending pure noise marked
    is_real=False must leave B where it would have been without those rows."""
    rng = np.random.default_rng(0)
    n, G = 400, 24
    X = rng.normal(size=(n, G)).astype(np.float32)
    p = rng.integers(0, 8, n)
    c = rng.integers(0, 3, n)
    s = rng.integers(0, 4, n)
    cells = rng.integers(50, 500, n).astype(float)
    cfg = TrainConfig(n_factors=3, epochs=2, batch_size=64, seed=0,
                      log_every=0, eval_every=0, patience=0)

    m_ref, _ = fit(X, p, c, s, cells, config=cfg)

    fake = (rng.normal(size=(n, G)) * 50).astype(np.float32)   # wildly off-scale
    X2 = np.vstack([X, fake])
    pad = lambda a: np.concatenate([a, a])
    real_flag = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    m_gated, _ = fit(X2, pad(p), pad(c), pad(s), pad(cells),
                     config=cfg, is_real=real_flag)

    # Not bit-identical (batching differs), but the gate must keep B finite and
    # on-scale rather than dragged toward the 50x noise.
    B_ref, B_gated = m_ref.loading_matrix(), m_gated.loading_matrix()
    assert np.isfinite(B_gated).all()
    assert B_gated.std() < 10 * B_ref.std()


def test_is_real_rejects_bad_input():
    rng = np.random.default_rng(0)
    n, G = 300, 16
    X = rng.normal(size=(n, G)).astype(np.float32)
    a = (rng.integers(0, 6, n), rng.integers(0, 3, n),
         rng.integers(0, 4, n), rng.integers(50, 300, n).astype(float))
    cfg = TrainConfig(n_factors=2, epochs=1, batch_size=64, log_every=0,
                      eval_every=0, patience=0)
    with pytest.raises(ValueError, match="is_real has"):
        fit(X, *a, config=cfg, is_real=np.ones(n - 1, bool))
    with pytest.raises(ValueError, match="no real rows"):
        fit(X, *a, config=cfg, is_real=np.zeros(n, bool))


def test_negatives_never_name_a_held_out_pair():
    """The sampler is built on training rows only, and every strategy must draw
    from an OBSERVED (perturbation, context) pair, so a held-out pair is never
    named as a negative."""
    rng = np.random.default_rng(0)
    n_p, n_c, n_s = 40, 6, 3
    rows = [(p, c, s) for p in range(n_p) for c in range(n_c) for s in range(n_s)]
    # hold out a tenth of the (p, c) pairs
    pairs = sorted({(p, c) for p, c, _ in rows})
    held = set(map(tuple, rng.permutation(pairs)[: len(pairs) // 10].tolist()))

    tr = [(p, c, s) for p, c, s in rows if (p, c) not in held]
    P = np.array([r[0] for r in tr]); C = np.array([r[1] for r in tr])
    S = np.array([r[2] for r in tr])

    sampler = StratifiedNegativeSampler(
        perturbation_idx=P, context_idx=C, stratum=S,
        n_perturbations=n_p, n_contexts=n_c, weights=dict(DEFAULT_WEIGHTS),
    )
    neg_p, neg_c = sampler.sample(np.arange(len(tr)), np.random.default_rng(1))

    named_held = [(int(a), int(b)) for a, b in zip(neg_p, neg_c) if (int(a), int(b)) in held]
    assert not named_held, f"{len(named_held)} negatives named a held-out pair, e.g. {named_held[:3]}"
    observed = {(p, c) for p, c, _ in tr}
    unseen = [(int(a), int(b)) for a, b in zip(neg_p, neg_c) if (int(a), int(b)) not in observed]
    assert not unseen, f"{len(unseen)} negatives named an unobserved pair, e.g. {unseen[:3]}"
    assert not ((neg_p == P) & (neg_c == C)).any(), "a negative kept its own label"



def test_synthetic_rows_are_negatives_for_L_disc_and_signed_in_L_recon():
    """Synthetic rows go through the SAME head as negatives -- the question is
    "did p in c produce this?", and a shuffled vector answers no. They must not
    enter L_recon with a plus sign; recon_fake_weight decides whether they are
    dropped (0) or actively pushed away (>0)."""
    rng = np.random.default_rng(0)
    n, G = 256, 20
    X = rng.normal(size=(n, G)).astype(np.float32)
    fake = (rng.normal(size=(n, G)) * 100).astype(np.float32)     # off-scale
    p = rng.integers(0, 6, n); c = rng.integers(0, 3, n)
    st = rng.integers(0, 3, n); cells = rng.integers(50, 400, n).astype(float)
    dbl = lambda a: np.concatenate([a, a])
    flag = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    Xb = np.vstack([X, fake])

    for beta in (0.0, 0.5):
        cfg = TrainConfig(n_factors=3, epochs=2, batch_size=128, seed=0,
                          log_every=0, eval_every=0, patience=0,
                          recon_fake_weight=beta)
        m, hist = fit(Xb, dbl(p), dbl(c), dbl(st), dbl(cells),
                      config=cfg, is_real=flag)
        B = m.loading_matrix()
        assert np.isfinite(B).all(), f"beta={beta}: non-finite B"
        assert hist[-1]["accuracy"] > 0.4, (beta, hist[-1]["accuracy"])
        assert B.std() < 5.0, (beta, B.std())   # not dragged onto the 100x noise


def test_recon_fake_weight_changes_the_objective():
    """beta > 0 must actually alter the fit, else the minus term is inert."""
    rng = np.random.default_rng(1)
    n, G = 192, 16
    X = rng.normal(size=(n, G)).astype(np.float32)
    fake = rng.normal(size=(n, G)).astype(np.float32)
    a = [rng.integers(0, 5, n), rng.integers(0, 3, n),
         rng.integers(0, 3, n), rng.integers(50, 300, n).astype(float)]
    dbl = lambda v: np.concatenate([v, v])
    flag = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    out = {}
    for beta in (0.0, 1.0):
        cfg = TrainConfig(subspace="free", n_factors=3, epochs=3, batch_size=128, seed=0,
                          log_every=0, eval_every=0, patience=0,
                          recon_fake_weight=beta)
        m, _ = fit(np.vstack([X, fake]), *[dbl(v) for v in a],
                   config=cfg, is_real=flag)
        out[beta] = m.loading_matrix()
    assert not np.allclose(out[0.0], out[1.0]), "recon_fake_weight had no effect"


def test_apply_split_keeps_synthetic_rows_with_their_real_twin():
    """A synthetic row inherits a real pair's label, so it must land in the
    SAME half. Re-drawing the split over the combined frame would scatter the
    twins and leak a held-out pair's synthetic copy into training."""
    from extract.data.augment import apply_split, make_split

    rng = np.random.default_rng(0)
    perts = [f"P{i:03d}" for i in range(60)]
    ctxs = [f"C{i}" for i in range(4)]
    rows = [{"perturbation": p, "context": c, "label": f"{p}__sub{s:02d}" if s else p,
             "n_cells": 100 + s}
            for p in perts for c in ctxs for s in range(3)]
    real = pd.DataFrame(rows)
    split = make_split(real, level="pair", test_frac=0.2, val_frac=0.1, seed=0)

    fake = real.copy()                       # one synthetic twin per real row
    combined = pd.concat([real, fake], ignore_index=True)
    ext = apply_split(split, combined)

    n = len(real)
    # every synthetic row is in the same half as its twin
    assert (ext.train[:n] == ext.train[n:]).all()
    assert (ext.test[:n] == ext.test[n:]).all()
    assert (ext.val[:n] == ext.val[n:]).all()
    # and the halves still partition the combined frame
    assert int(ext.train.sum() + ext.test.sum() + ext.val.sum()) == len(combined)
    # no held-out pair appears in train, on either side
    key = combined.perturbation + "|" + combined.context
    assert not (set(key[ext.train]) & set(key[ext.test]))
    assert not (set(key[ext.train]) & set(key[ext.val]))



def test_balance_negatives_matches_negative_count_to_positives():
    """|permuted| + |synthetic| must equal |positives|. With an equal number of
    synthetic rows that means every permuted negative is dropped; with half as
    many, half are kept."""
    from extract.train import Extract
    rng = np.random.default_rng(0)
    n, G = 256, 16
    X = rng.normal(size=(n, G)).astype(np.float32)
    a = [rng.integers(0, 5, n), rng.integers(0, 3, n),
         rng.integers(0, 3, n), rng.integers(50, 300, n).astype(float)]

    seen = {}
    orig = Extract.score
    def spy(self, z, p_i, c_i):
        seen["n"] = seen.get("n", 0) + len(z)
        return orig(self, z, p_i, c_i)

    for n_synth, expect_frac in ((n, 0.0), (n // 2, 0.5)):
        Xf = rng.normal(size=(n_synth, G)).astype(np.float32)
        cat = lambda v: np.concatenate([v, v[:n_synth]])
        flag = np.concatenate([np.ones(n, bool), np.zeros(n_synth, bool)])
        cfg = TrainConfig(n_factors=3, epochs=1, batch_size=n + n_synth,
                          seed=0, log_every=0, eval_every=0, patience=0)
        seen.clear()
        Extract.score = spy
        try:
            fit(np.vstack([X, Xf]), *[cat(v) for v in a], config=cfg, is_real=flag)
        finally:
            Extract.score = orig
        # scored = positives(n) + permuted(~expect_frac*n) + synthetic(n_synth)
        expected = n + expect_frac * n + n_synth
        assert abs(seen["n"] - expected) < 0.25 * n, (n_synth, seen["n"], expected)


def test_balance_negatives_is_inert_without_synthetic_rows():
    rng = np.random.default_rng(3)
    n, G = 256, 16
    X = rng.normal(size=(n, G)).astype(np.float32)
    a = [rng.integers(0, 6, n), rng.integers(0, 3, n),
         rng.integers(0, 3, n), rng.integers(50, 300, n).astype(float)]
    Bs = []
    for bal in (True, False):
        cfg = TrainConfig(n_factors=3, epochs=2, batch_size=128, seed=0,
                          log_every=0, eval_every=0, patience=0,
                          balance_negatives=bal)
        m, _ = fit(X, *a, config=cfg)
        Bs.append(m.loading_matrix())
    assert np.allclose(Bs[0], Bs[1])


def test_single_context_perturbations_always_train():
    """A perturbation with one pair has nothing left to train e_p on if that
    pair is held out, so its row is unpredictable by construction. Those pairs
    must stay in train."""
    from extract.data.augment import make_split

    rows = []
    for i in range(200):                       # 12 contexts each
        for c in range(12):
            rows.append({"perturbation": f"P{i:03d}", "context": f"C{c}",
                         "label": f"P{i:03d}"})
    for i in range(20):                        # single-context perturbations
        rows.append({"perturbation": f"S{i:02d}", "context": "C0",
                     "label": f"S{i:02d}"})
    meta = pd.DataFrame(rows)
    singles = {f"S{i:02d}" for i in range(20)}

    sp = make_split(meta, level="pair", test_frac=0.2, val_frac=0.2, seed=0)
    held = set(meta[sp.test].perturbation) | set(meta[sp.val].perturbation)
    assert not (singles & held), sorted(singles & held)[:3]
    # every perturbation keeps at least one training pair
    assert set(meta.perturbation) <= set(meta[sp.train].perturbation)
    # and the held-out count is still the requested fraction of ALL pairs
    n_pairs = meta.groupby(["perturbation", "context"], observed=True).ngroups
    assert abs(len(sp.held_out) - round(0.2 * n_pairs)) <= 1

    off = make_split(meta, level="pair", test_frac=0.2, val_frac=0.2, seed=0,
                     protect_single_context=False)
    held_off = set(meta[off.test].perturbation) | set(meta[off.val].perturbation)
    assert singles & held_off, "opt-out should let singles be held out"


def test_evaluate_takes_its_device_from_the_model_not_the_data():
    """A caller may deliberately keep the full matrix on the CPU (it is ~11 GB
    on the real split) and let batches move. Reading the device off X_t instead
    of the model sends CPU batches into a CUDA model and the first matmul
    raises. Checked on CPU by putting the model on a meta-free device and
    asserting evaluate() uses `next(model.parameters()).device`."""
    import inspect
    from extract import train as T
    src_ = inspect.getsource(T.evaluate)
    assert "next(model.parameters()).device" in src_, \
        "evaluate() must take its device from the model"
    assert "device = X_t.device" not in src_, \
        "evaluate() must not take its device from the data tensor"
    # every batch slice of the data tensor is moved explicitly
    for frag in ("X_t[idx_t].to(device)", "target_col[idx_t].to(device)"):
        assert frag in src_, f"missing device move: {frag}"


def test_min_epochs_keeps_training_but_still_restores_the_best_epoch():
    """min_epochs decides when training STOPS, not which state is kept: a run
    whose validation bottoms out early must still train to min_epochs, and
    still restore the early best epoch."""
    from extract.data.augment import make_split

    meta = _toy_meta(n_pert=16, n_ctx=4, n_strata=3)
    X, p, c, s, n = _toy_dataset(n_pert=16, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=0)
    cfg = TrainConfig(n_factors=4, epochs=40, batch_size=32, log_every=0,
                      eval_every=1, patience=1, min_epochs=25, seed=0)
    _, history = fit(X, p, c, s, n, config=cfg,
                     train_rows=split.train, val_rows=split.val)
    assert len(history) >= 25, len(history)
    ev = [h for h in history if "val_total" in h]
    best = min(ev, key=lambda h: h["val_total"])
    assert history[-1]["selected_epoch"] == best["epoch"]


def test_on_eval_snapshots_B_at_every_evaluation():
    """on_eval fires once per validation pass with the live model, so a copy
    of B taken there is the B of that epoch -- not the restored one."""
    from extract.data.augment import make_split

    meta = _toy_meta(n_pert=16, n_ctx=4, n_strata=3)
    X, p, c, s, n = _toy_dataset(n_pert=16, n_ctx=4, n_strata=3, n_genes=40)
    split = make_split(meta, level="pair", test_frac=0.12, val_frac=0.12, seed=0)
    cfg = TrainConfig(n_factors=4, epochs=10, batch_size=32, log_every=0,
                      eval_every=3, patience=0, select_on="accuracy", seed=0)
    snaps = {}
    model, history = fit(X, p, c, s, n, config=cfg,
                         train_rows=split.train, val_rows=split.val,
                         on_eval=lambda ep, m: snaps.__setitem__(
                             ep, m.loading_matrix().copy()))
    evals = [h["epoch"] for h in history if "val_accuracy" in h]
    assert sorted(snaps) == evals == [0, 3, 6, 9]
    assert not np.allclose(snaps[0], snaps[9])
    sel = history[-1]["selected_epoch"]
    np.testing.assert_allclose(model.loading_matrix(), snaps[sel])
