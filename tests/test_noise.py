"""Noise model, label-driven subspace, and the fixed / anchored subspace modes."""

import numpy as np
import pandas as pd
import pytest
import torch

from extract.data.noise import NoiseModel, label_subspace, noise_covariance, subsample_residuals
from extract.models import FixedBasisLoadings, GlobalLoadings


def _planted(n_pairs=600, G=60, d=3, r=2, n_sub=6, seed=0):
    """Rows x = z A + noise/sqrt(n), noise with a planted correlated part."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, G))
    U, _ = np.linalg.qr(rng.normal(size=(G, r)))
    s_true = np.array([4.0, 2.0])
    diag = rng.uniform(0.5, 1.5, G)
    L = U * np.sqrt(s_true)
    rows, X = [], []
    for p in range(n_pairs):
        z = rng.laplace(size=d) * 0.5
        n_full = int(rng.integers(200, 2000))
        cells = rng.normal(size=(n_full, G)) * np.sqrt(diag) + rng.normal(size=(n_full, r)) @ L.T
        for s in range(n_sub + 1):
            n = n_full if s == 0 else int(n_full * rng.uniform(0.15, 0.9))
            x = z @ A + cells[:n].mean(0)            # subsets of the same cells
            X.append(x)
            rows.append(dict(perturbation=f"P{p}", context="C0", n_cells=n,
                             variant="main" if s == 0 else "sub", is_real=True))
    return np.asarray(X, np.float32), pd.DataFrame(rows), A, U, s_true, diag


def test_residual_scaling_recovers_the_cell_noise():
    X, meta, A, U, s_true, diag = _planted()
    R = subsample_residuals(X, meta, np.ones(len(meta), bool))
    assert len(R) == (meta.variant == "sub").sum()
    nm = noise_covariance(R, rank=2)
    # per-gene total noise variance and the correlated directions are recovered
    total_true = diag + (U * U * s_true).sum(1)
    total_hat = nm.diag + (nm.U * nm.U * nm.s).sum(1)
    assert np.corrcoef(total_true, total_hat)[0, 1] > 0.9
    cos = np.abs(nm.U.T @ U)
    assert np.linalg.svd(cos, compute_uv=False).min() > 0.9
    assert np.allclose(np.sort(nm.s)[::-1], s_true, rtol=0.25)


def test_metric_square_roots_are_consistent():
    rng = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.normal(size=(20, 3)))
    nm = NoiseModel(diag=rng.uniform(0.5, 2, 20), U=U, s=np.array([5.0, 2.0, 1.0]))
    M = nm.c * np.eye(20) + U @ np.diag(nm.s) @ U.T
    X = rng.normal(size=(4, 20))
    assert np.allclose(nm.metric_sqrt(nm.metric_inv_sqrt(X)), X)
    assert np.allclose(nm.metric_inv_quadratic(X), np.einsum("ij,jk,ik->i", X, np.linalg.inv(M), X))


def test_label_subspace_finds_the_signal_span():
    X, meta, A, *_ = _planted()
    full = (meta.variant == "main").to_numpy()
    R = subsample_residuals(X, meta, np.ones(len(meta), bool))
    V, ev = label_subspace(X[full], meta.n_cells.to_numpy()[full], noise_covariance(R, 2), d=3)
    assert V.shape == (3, X.shape[1]) and np.all(np.diff(ev) <= 0)
    Qa, _ = np.linalg.qr(A.T); Qv, _ = np.linalg.qr(V.T)
    assert np.linalg.svd(Qa.T @ Qv, compute_uv=False).min() > 0.95


def test_fixed_basis_keeps_the_span():
    rng = np.random.default_rng(2)
    V = rng.normal(size=(4, 30)).astype(np.float32)
    fl = FixedBasisLoadings(V)
    assert np.allclose(fl.B.detach().numpy(), V)
    with torch.no_grad():
        fl.A.copy_(torch.as_tensor(rng.normal(size=(4, 4)), dtype=torch.float32))
    x = torch.as_tensor(rng.normal(size=(8, 30)), dtype=torch.float32)
    gl = GlobalLoadings(4, 30)
    with torch.no_grad():
        gl.B.copy_(torch.as_tensor(V))
    # same span -> same reconstruction, whatever A is
    assert torch.allclose(fl.reconstruct(fl.project(x)), gl.reconstruct(gl.project(x)), atol=1e-4)
    assert [n for n, _ in fl.named_parameters()] == ["A"]


def test_noise_metric_reduces_to_euclidean_without_correlated_noise():
    rng = np.random.default_rng(3)
    gl = GlobalLoadings(3, 25)
    x = torch.as_tensor(rng.normal(size=(5, 25)), dtype=torch.float32)
    col = torch.tensor([0, 3, -1, 7, -1])
    z = gl.project(x, col)
    plain = gl.squared_error(x, z, col)
    U, _ = np.linalg.qr(rng.normal(size=(25, 2)))
    gl.set_noise_metric(U, np.array([1e-9, 1e-9]), c=1.0)
    assert torch.allclose(gl.squared_error(x, z, col), plain, atol=1e-5)
    gl.set_noise_metric(U, np.array([10.0, 5.0]), c=1.0)
    assert (gl.squared_error(x, z, col) <= plain + 1e-6).all()


def test_fit_runs_in_each_subspace_mode():
    from extract.train import TrainConfig, fit
    X, meta, *_ = _planted(n_pairs=80, G=30, n_sub=3)
    meta["context"] = np.where(np.arange(len(meta)) % 2, "C0", "C1")
    p = pd.factorize(meta.perturbation)[0]; c = pd.factorize(meta.context)[0]
    s = np.where(meta.variant == "main", 0, 1); n = meta.n_cells.to_numpy()
    R = subsample_residuals(X, meta, np.ones(len(meta), bool))
    nm = noise_covariance(R, rank=2)
    full = (meta.variant == "main").to_numpy()
    V, _ = label_subspace(X[full], n[full], nm, d=3)
    val = meta.perturbation.isin([f"P{i}" for i in range(0, 80, 9)]).to_numpy()
    for mode in ("free", "fixed", "anchored"):
        cfg = TrainConfig(n_factors=3, epochs=2, batch_size=64, eval_every=1, patience=0,
                          log_every=0, select_on="accuracy", subspace=mode)
        model, hist = fit(X, p, c, s, n, config=cfg, train_rows=~val, val_rows=val,
                          subspace_basis=V, noise_model=nm)
        assert "val_accuracy" in hist[-1]
        if mode == "fixed":
            B = model.loading_matrix()
            Qb, _ = np.linalg.qr(B.T); Qv, _ = np.linalg.qr(V.T)
            assert np.linalg.svd(Qb.T @ Qv, compute_uv=False).min() > 0.999
    with pytest.raises(ValueError):
        fit(X, p, c, s, n, config=TrainConfig(n_factors=3, subspace="fixed"), train_rows=~val)


def test_frozen_basis_learns_nothing_in_B():
    from extract.train import TrainConfig, fit
    X, meta, *_ = _planted(n_pairs=80, G=30, n_sub=3)
    meta["context"] = np.where(np.arange(len(meta)) % 2, "C0", "C1")
    p = pd.factorize(meta.perturbation)[0]; c = pd.factorize(meta.context)[0]
    s = np.where(meta.variant == "main", 0, 1); n = meta.n_cells.to_numpy()
    V = np.random.default_rng(5).normal(size=(3, 30)).astype(np.float32)
    val = meta.perturbation.isin([f"P{i}" for i in range(0, 80, 9)]).to_numpy()
    cfg = TrainConfig(n_factors=3, epochs=2, batch_size=64, eval_every=1, patience=0,
                      log_every=0, select_on="accuracy", subspace="frozen")
    model, _ = fit(X, p, c, s, n, config=cfg, train_rows=~val, val_rows=val, subspace_basis=V)
    assert np.allclose(model.loading_matrix(), V)
    assert "loadings.A" not in dict(model.named_parameters())
