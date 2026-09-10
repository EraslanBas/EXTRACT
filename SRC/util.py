import numpy as np
from sklearn.linear_model import LinearRegression
from libraries import *

def sample_adata(adata, frac=0.4, random_state=0):
    """
    Randomly sample a fraction of cells from an AnnData object.
    """
    rng = np.random.default_rng(random_state)
    
    n = adata.n_obs
    k = int(n * frac)

    idx = rng.choice(n, size=k, replace=False)
    return adata[idx, :].copy()


def assess_on_target_knockdown(
    adata_counts_path: str,
    adata_norm_path: str,
    perturbation_column: str = "target_gene",
    control_label: str = "non-targeting"):
    
    adata = sc.read_h5ad(adata_counts_path)

    perts = adata.obs[perturbation_column]
    control_cells = (perts == control_label).values
    
    control_mean = pd.DataFrame(np.mean(adata.X[control_cells,:],axis=0))
    control_mean.columns = adata.var_names
    
    for target_gene in list(set(perts.unique()) - set([control_label])):
        if target_gene in adata.var_names:
            perturbed_cells = adata.obs[perturbation_column] == target_gene
            gene_idx = adata.var_names.get_loc(target_gene)
            

            control_mean_gene = float(control_mean.iloc[0, gene_idx])
            expr_vals = adata[perturbed_cells, :].X[:, gene_idx]
            
            # convert to dense (even if it’s a single-column sparse matrix)
            if sp.issparse(expr_vals):
                expr_vals = expr_vals.toarray().ravel()
            else:
                expr_vals = np.asarray(expr_vals).ravel()
            
            # compute ratio
            if ~np.isclose(control_mean_gene, 0.0):
               ratios = ((expr_vals - float(control_mean_gene))/ (float(control_mean_gene)))   
               adata.obs.loc[perturbed_cells, "KnockDownEfficiency"] = ratios

    KOef = adata.obs["KnockDownEfficiency"]
    adata = sc.read_h5ad(adata_norm_path)
    adata.obs["KnockDownEfficiency"]=KOef
    adata.write(adata_norm_path)
 

def baseline_high_low(target_fold, abs_W_typical, HIGH=None, LOW=None):
    """
    Solve for HIGH or LOW to achieve a target *baseline expression*
    fold-change between active-program and inactive-program genes.

    From ``ControlDataGenerator.generate_expression``:

        log_expr[c, g] = sum_p softplus(activity_p) * W[g, p] + baseline_g + noise

    so a gene that belongs only to an active program (activity=HIGH) and a
    gene that belongs only to an inactive program (activity=LOW) differ
    in log-expression by ``(softplus(HIGH) - softplus(LOW)) * |W_typical|``,
    giving a fold-change of

        fold = exp((softplus(HIGH) - softplus(LOW)) * abs_W_typical).

    Parameters
    ----------
    target_fold : float
        Desired ratio  active-gene / inactive-gene  in linear expression.
        Typical biological range: 2 - 4.
    abs_W_typical : float
        Mean magnitude of non-zero W entries in the GRN.  Compute with
        ``np.abs(grn.W[grn.W != 0]).mean()`` (~0.79 in the default GRN).
    HIGH, LOW : float, optional
        Pass exactly one — the other is solved.

    Returns
    -------
    HIGH, LOW : tuple of float
    """
    if target_fold <= 1.0:
        raise ValueError("target_fold must be > 1")
    if (HIGH is None) == (LOW is None):
        raise ValueError("Pass exactly one of HIGH or LOW; solve for the other.")

    sp_delta = np.log(target_fold) / abs_W_typical  # softplus(HIGH) - softplus(LOW)
    softplus = lambda x: np.log1p(np.exp(x))
    inv_softplus = lambda y: np.log(np.expm1(y))    # softplus^{-1}, valid for y>0

    if HIGH is not None:
        sp_LOW = softplus(HIGH) - sp_delta
        if sp_LOW <= 0:
            raise ValueError(
                f"HIGH={HIGH} too low for fold={target_fold}: requires "
                f"softplus(LOW) <= 0. Increase HIGH or lower target_fold."
            )
        LOW = inv_softplus(sp_LOW)
    else:
        sp_HIGH = softplus(LOW) + sp_delta
        HIGH = inv_softplus(sp_HIGH)
    return float(HIGH), float(LOW)


def gate_params(HIGH, LOW, g_H=0.999, g_L=0.25):
    """
    Solve for ``(sigmoid_threshold, sigmoid_scale)`` such that the
    PerturbSeqSimulator gate hits target values at HIGH and LOW:

        gate(HIGH) = g_H   and   gate(LOW) = g_L

    Gate ratio = g_H / g_L is the factor by which active-context KO
    effects exceed inactive-context KO effects.  The baseline
    similarity between contexts is controlled separately by
    softplus(HIGH) / softplus(LOW), and depends on the HIGH/LOW values
    themselves (not on threshold/scale).

    Parameters
    ----------
    HIGH, LOW : float
        Per-program activity values for "active" and "inactive" contexts.
    g_H, g_L : float in (0, 1)
        Desired gate values at HIGH and LOW.  Default (0.999, 0.25)
        gives a gate ratio of ~4.

    Returns
    -------
    threshold, scale : float
        Pass to PerturbSeqSimulator(... sigmoid_threshold=, sigmoid_scale=).
    """
    logit_H = np.log(g_H / (1.0 - g_H))
    logit_L = np.log(g_L / (1.0 - g_L))
    scale = (logit_H - logit_L) / (HIGH - LOW)
    threshold = HIGH - logit_H / scale
    return float(threshold), float(scale)


def multivariate_r2(Y, X):
    """
    Computes the proportion of total variance in Y explained by covariates X
    (multivariate R^2 via linear projection).
    
    Parameters:
    - Y: (n_samples, n_features) array-like response matrix
    - X: (n_samples,) or (n_samples, n_covariates) covariate(s)
    
    Returns:
    - explained_variance: float, proportion of variance in Y explained by X
    """
    Y = np.asarray(Y)
    X = np.asarray(X)

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Fit linear regression model of Y ~ X
    model = LinearRegression()
    model.fit(X, Y)
    Y_hat = model.predict(X)

    # Center Y
    Y_mean = Y.mean(axis=0)
    Y_centered = Y - Y_mean
    
    # Residuals and total variance
    residuals = Y - Y_hat
    ss_res = np.sum(residuals ** 2) 
    ss_total = np.sum(Y_centered ** 2) 

    return 1 - (ss_res / ss_total)