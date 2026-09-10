from .libraries import *
from .grn import *

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


class PerturbSeqSimulator:
    """
    Simulate Perturb-seq single gene knockouts from observed control cells.

    Given a gene regulatory network (W, V matrices) and a control AnnData
    (raw counts), this class answers the counterfactual question:
    "How would these control cells look if gene g were knocked out?"

    Perturbation model
    ------------------
    Primary effect (step 1):
        The KO removes gene g's contribution to its programs.  V[p, g]
        directly encodes the effect on program p when gene g is knocked out
        (sign(V) = -sign(W) is enforced by GeneRegulatoryNetwork):

            delta_h[p] = V[p, g] * strength

    Secondary effects (steps 2+):
        The primary program shift alters other genes via W.  Those gene
        changes feed back to programs via V, attenuated by a damping
        factor.  The KO gene is excluded from the feedback (it is
        silenced and cannot contribute):

            delta_g_k = W  @ delta_h_k               (programs -> genes)
            delta_g_k[g] = 0                          (KO gene silenced)
            delta_h_{k+1} = V @ delta_g_k * damping   (genes -> programs)

        Each secondary step also updates the gene-level effects: the total
        fold change applied to every gene is fc = exp(W @ total_delta_h),
        which accumulates shifts from all propagation steps.

    The total program shift is stored as a per-gene log-fold-change vector
    in natural log space:
        log_fc = W @ total_delta_h.
    Perturbed cells are produced by sampling control cells and shifting
    each gene's count by log_fc in log1p space:
        log(X_pert + 1) = log(X_sampled_ctrl + 1) + log_fc,
    then taking expm1.  The KO gene is forced to zero.  Counts are kept
    as floats (no integer rounding) so the empirical log1p-mean-difference
    estimator  mean(log(X_pert+1)) - mean(log(X_ctrl+1))  equals log_fc
    exactly for every cell, and its average over cells matches the GT up
    to control-sampling noise.
    """

    def __init__(
        self,
        grn: GeneRegulatoryNetwork,
        control_adata: ad.AnnData,
        counts_layer: Optional[str] = None,
        perturbation_strength: float = 1.0,
        damping_factor: float = 0.5,
        n_propagation_steps: int = 2,
        max_log2_fc: float = 5.0,
        noise_std: float = 0.5,
        nb_sampling: bool = False,
        min_dispersion: float = 5.0,
        program_activity: Optional[np.ndarray] = None,
        sigmoid_threshold: float = 0.0,
        sigmoid_scale: float = 5.0,
        seed: int = 42,
        verbose: bool = False
    ):
        """
        Parameters
        ----------
        grn : GeneRegulatoryNetwork
            Network with W (n_genes, n_programs) and V (n_programs, n_genes).
        control_adata : AnnData
            Control (unperturbed) cells.  Must contain raw counts either in
            .X or in the layer specified by *counts_layer*.
        counts_layer : str, optional
            Layer name that holds raw counts.  None means use .X directly.
        perturbation_strength : float
            Scaling factor for the KO effect (1.0 = full knockout).
        damping_factor : float
            Attenuation applied at each secondary propagation step.
        n_propagation_steps : int
            Number of V->W propagation rounds.
            1 = primary only, 2+ = includes secondary cascade.
        max_log2_fc : float
            Symmetric clip applied to the per-gene log2 fold change before
            simulating counts.  Keeps runaway cascade amplification from
            producing non-physical library sizes.  Default 5.0 (=32x).
        noise_std : float
            Standard deviation of zero-mean Gaussian noise added in log1p
            space before exponentiation.  Preserves the GT-empirical
            mean identity (E[noise]=0) while giving realistic p-value
            gradients on Wilcoxon / count-based DE.  Set to 0 for
            deterministic simulation.
        nb_sampling : bool
            If True, layer a negative-binomial draw on top of the
            deterministic/Gaussian-noised log1p-shift output so the final
            counts are integer with realistic mean-variance structure.
            Needed for count-based DE (DESeq2, glmGamPoi).  Introduces a
            Jensen bias plus a zero-clamp asymmetry on strong
            downregulation from low-baseline genes, so the GT-empirical
            diagonal degrades.  Default False (log1p-space pipelines
            don't need it).
        min_dispersion : float
            Lower bound applied to the per-gene NB dispersion estimated
            from controls.  Higher values mean less overdispersion (less
            count-level noise) and tighter GT-empirical agreement.
            Default 5.0 — controls with mean > 0.1 can go higher if the
            data warrants it; below-0.1-mean genes fall back to this
            floor instead of the legacy r=1.  Only consulted when
            nb_sampling is True.
        program_activity : array-like of shape (n_programs,), optional
            Context-specific activity level for every latent program.
            When provided, the total program shift produced by a KO is
            multiplied by  sigmoid((activity - threshold) * scale)
            elementwise, so KO effects on programs that are inactive in
            this context are attenuated toward zero.  None disables the
            gate and keeps the legacy behavior.
        sigmoid_threshold : float
            Activity level at which the gate is at 0.5.  Default 0.0.
        sigmoid_scale : float
            Steepness of the sigmoid gate.  Higher => sharper on/off
            transition.  Default 5.0 (sigmoid(+/-3) ~ 1.00 / 0.00).
        seed : int
            Random seed.
        verbose : bool
            Print progress information.
        """
        self.grn = grn
        self.control_adata = control_adata
        self.perturbation_strength = perturbation_strength
        self.damping_factor = damping_factor
        self.n_propagation_steps = max(1, n_propagation_steps)
        self.max_log2_fc = max_log2_fc
        self.noise_std = noise_std
        self.nb_sampling = nb_sampling
        self.min_dispersion = min_dispersion
        self.program_activity = (
            np.asarray(program_activity, dtype=np.float64)
            if program_activity is not None else None
        )
        self.sigmoid_threshold = sigmoid_threshold
        self.sigmoid_scale = sigmoid_scale
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        # Extract raw count matrix from AnnData
        raw = control_adata.layers[counts_layer] if counts_layer else control_adata.X
        if sp.issparse(raw):
            self.control_counts = np.asarray(raw.toarray(), dtype=np.float64)
        else:
            self.control_counts = np.asarray(raw, dtype=np.float64)

        # Per-gene dispersion estimated from control counts
        self.dispersion = self._estimate_dispersion()

        self.perturbable_genes = self._get_perturbable_genes()
        self.effects = self._precompute_all_effects()

        if self.verbose:
            print(f"PerturbSeqSimulator initialized:")
            print(f"  Control cells: {self.control_counts.shape[0]}")
            print(f"  Perturbable genes: {len(self.perturbable_genes)}")
            print(f"  Propagation steps: {self.n_propagation_steps}")
            print(f"  Damping factor: {self.damping_factor}")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _estimate_dispersion(self) -> np.ndarray:
        """
        Estimate per-gene NB dispersion from control counts via method of
        moments:  r = mean^2 / (var - mean).  Low-mean genes fall back to
        min_dispersion, and all values are clipped to [min_dispersion, 100].
        """
        mean = self.control_counts.mean(axis=0)
        var = self.control_counts.var(axis=0)
        excess_var = np.maximum(var - mean, 1e-6)
        r = np.where(mean > 0.1, mean ** 2 / excess_var, self.min_dispersion)
        return np.clip(r, self.min_dispersion, 100.0)

    def _get_perturbable_genes(self) -> List[str]:
        """Genes with at least one non-zero entry in V (program members)."""
        v_active = np.any(self.grn.V != 0, axis=0)
        return [self.grn.all_genes[i] for i, active in enumerate(v_active) if active]

    def _get_or_compute_effect(self, gene_name: str) -> Dict:
        """
        Return the cached perturbation effect for *gene_name*, lazily
        computing it if absent.  Genes outside any program have V[:, g] = 0,
        so the computed log_fc is all zeros — a null perturbation in which
        the KO silences the target gene but does not shift any other gene.
        """
        eff = self.effects.get(gene_name)
        if eff is None:
            gene_idx = self.grn.gene_to_idx[gene_name]
            eff = self._compute_perturbation_effect(gene_idx)
            self.effects[gene_name] = eff
        return eff

    # ------------------------------------------------------------------
    # Perturbation-effect computation
    # ------------------------------------------------------------------

    def _compute_perturbation_effect(self, gene_idx: int) -> Dict:
        """
        Compute the program-activity shift and per-gene fold change for
        knocking out one gene.

        V[p, g] already encodes the correct sign for the KO effect
        (sign(V) = -sign(W), enforced by GeneRegulatoryNetwork).

        During secondary propagation the KO gene is excluded from the
        V-feedback (it is silenced and cannot influence programs).
        """
        W = self.grn.W  # (n_genes, n_programs)
        V = self.grn.V  # (n_programs, n_genes)

        # Primary: V directly gives the KO effect on programs
        primary_delta_h = V[:, gene_idx] * self.perturbation_strength
        primary_delta_g = W @ primary_delta_h  # resulting gene-level shift

        total_delta_h = primary_delta_h.copy()

        info: Dict = {
            'primary_program_shift': primary_delta_h.copy(),
            'primary_gene_shift': primary_delta_g.copy(),
        }

        # Secondary propagation
        delta_h = primary_delta_h
        for step in range(1, self.n_propagation_steps):
            delta_g = W @ delta_h                        # programs -> genes
            delta_g[gene_idx] = 0.0                      # KO gene is silenced
            delta_h = V @ delta_g * self.damping_factor  # genes -> programs
            total_delta_h += delta_h
            info[f'step{step + 1}_program_shift'] = delta_h.copy()
            info[f'step{step + 1}_gene_shift'] = (W @ delta_h).copy()

        # Context-specific sigmoid gate: inactive programs don't propagate
        if self.program_activity is not None:
            gate = 1.0 / (1.0 + np.exp(
                -(self.program_activity - self.sigmoid_threshold) * self.sigmoid_scale
            ))
            total_delta_h = total_delta_h * gate
            info['program_gate'] = gate

        info['total_program_shift'] = total_delta_h
        info['total_gene_shift'] = W @ total_delta_h

        # Per-gene multiplicative fold change
        log_fc = W @ total_delta_h              # (n_genes,)
        # Cap biologically implausible cascade amplification
        max_log_fc = self.max_log2_fc * np.log(2)
        log_fc = np.clip(log_fc, -max_log_fc, max_log_fc)
        info['log_fold_change'] = log_fc
        info['fold_change'] = np.exp(log_fc)

        info['affected_programs'] = [
            self.grn.program_names[i]
            for i in range(self.grn.n_programs)
            if abs(total_delta_h[i]) > 1e-4
        ]
        return info

    def _precompute_all_effects(self) -> Dict[str, Dict]:
        """Pre-compute perturbation effects for every perturbable gene."""
        effects: Dict[str, Dict] = {}
        iterable = self.perturbable_genes
        if self.verbose:
            iterable = tqdm(iterable, desc="Pre-computing perturbation effects")

        for gene in iterable:
            gene_idx = self.grn.gene_to_idx[gene]
            effects[gene] = self._compute_perturbation_effect(gene_idx)
        return effects

    # ------------------------------------------------------------------
    # Count resampling
    # ------------------------------------------------------------------

    def _resample_counts(
        self,
        mu: np.ndarray,
        count_distribution: str = 'negative_binomial'
    ) -> np.ndarray:
        """
        Sample integer counts from per-cell, per-gene expected rates.

        Parameters
        ----------
        mu : np.ndarray, shape (n_cells, n_genes)
            Non-negative expected counts.
        count_distribution : str
            'negative_binomial' or 'poisson'.

        Returns
        -------
        counts : np.ndarray of int
        """
        mu = np.maximum(mu, 0.0)

        if count_distribution == 'negative_binomial':
            r = self.dispersion[np.newaxis, :]          # (1, n_genes)
            p = r / (r + mu + 1e-10)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            counts = self.rng.negative_binomial(r, p)
        elif count_distribution == 'poisson':
            counts = self.rng.poisson(mu + 1e-10)
        else:
            raise ValueError(f"Unknown count distribution: {count_distribution}")

        return counts

    # ------------------------------------------------------------------
    # Public simulation API
    # ------------------------------------------------------------------

    def simulate_perturbation(
        self,
        gene_name: str,
        n_cells: int = 100,
        count_distribution: str = 'negative_binomial',
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate a single gene knockout.

        For each of *n_cells* perturbed cells a control cell is sampled
        (with replacement), its counts are scaled by the perturbation fold
        change, and new counts are drawn from NB / Poisson.

        Parameters
        ----------
        gene_name : str
            Gene to knock out.
        n_cells : int
            Number of perturbed cells to generate.
        count_distribution : str
            'negative_binomial' or 'poisson'.

        Returns
        -------
        counts : np.ndarray, shape (n_cells, n_genes)
        effect_info : dict
        """
        if gene_name not in self.grn.gene_to_idx:
            raise ValueError(f"Gene '{gene_name}' not found in network")

        # Lazy: filler genes (V[:, g] = 0) get a null effect on demand.
        effect = self._get_or_compute_effect(gene_name)

        gene_idx = self.grn.gene_to_idx[gene_name]
        log_fc = effect['log_fold_change']  # (n_genes,), natural log

        # Sample control cells as baseline
        n_control = self.control_counts.shape[0]
        idx = self.rng.choice(n_control, size=n_cells, replace=True)
        baseline = self.control_counts[idx, :]  # (n_cells, n_genes)

        # Log1p-shift: log(X_pert + 1) = log(X_ctrl + 1) + log_fc
        # Counts are kept as floats (no rounding, no [0, inf) clamp);
        # expm1(y) lives in (-1, inf), and (X_pert + 1) = exp(y_pert) > 0, so
        # log2(X_pert + 1) is always finite and the identity holds for
        # downregulation-from-zero as well as upregulation.
        y_pert = np.log1p(baseline) + log_fc[np.newaxis, :]
        if self.noise_std > 0:
            y_pert += self.rng.normal(0.0, self.noise_std, size=y_pert.shape)
        y_pert[:, gene_idx] = 0.0  # KO gene: log(0+1) = 0 -> X_ko = 0
        counts = np.expm1(y_pert)

        if self.nb_sampling:
            # Technical/capture noise: integer counts with NB(mu, r) on top
            # of the biological log1p-space variability.
            counts = self._resample_counts(counts, 'negative_binomial')
        counts[:, gene_idx] = 0

        return counts, effect

    def simulate_all_perturbations(
        self,
        n_cells_per_perturbation: int = 100,
        genes_to_perturb: Optional[List[str]] = None,
        count_distribution: str = 'negative_binomial',
        include_control: bool = True,
    ) -> ad.AnnData:
        """
        Simulate single-gene KOs for all (or selected) perturbable genes.

        Parameters
        ----------
        n_cells_per_perturbation : int
            Cells generated per KO condition.
        genes_to_perturb : list of str, optional
            Subset of perturbable genes.  None = all perturbable genes.
        count_distribution : str
            'negative_binomial' or 'poisson'.
        include_control : bool
            Whether to include the original control cells in the output.

        Returns
        -------
        adata : AnnData
            Combined dataset.  Key fields:

            - .obs['perturbation']  : gene name or 'control'
            - .obs['is_control']    : bool
            - .var[<program_name>]  : bool membership per program
            - .varm['W']            : W matrix
            - .uns['V']             : V matrix
            - .uns['perturbation_effects'] : per-gene effect dicts
        """
        if genes_to_perturb is None:
            genes_to_perturb = self.perturbable_genes
        else:
            invalid = [g for g in genes_to_perturb if g not in self.grn.gene_to_idx]
            if invalid:
                raise ValueError(
                    f"Genes not in network: {invalid}"
                )

        count_blocks: List[np.ndarray] = []
        obs_records: List[Dict] = []
        effect_store: Dict[str, Dict] = {}

        # -- Control cells (original, unmodified) --
        if include_control:
            n_ctrl = self.control_counts.shape[0]
            if self.verbose:
                print(f"Including {n_ctrl} control cells from input AnnData ...")
            count_blocks.append(self.control_counts)
            obs_records.extend(
                [{'perturbation': 'control', 'is_control': True}] * n_ctrl
            )

        # -- Perturbed conditions --
        iterable = genes_to_perturb
        if self.verbose:
            iterable = tqdm(iterable, desc="Simulating gene KOs")

        for gene in iterable:
            counts, effect = self.simulate_perturbation(
                gene_name=gene,
                n_cells=n_cells_per_perturbation,
                count_distribution=count_distribution,
            )
            count_blocks.append(counts)
            effect_store[gene] = effect
            obs_records.extend(
                [{'perturbation': gene, 'is_control': False}]
                * n_cells_per_perturbation
            )

        # -- Assemble AnnData --
        X = np.vstack(count_blocks)

        obs = pd.DataFrame(obs_records)
        obs.index = [f"cell_{i}" for i in range(len(obs))]

        var = pd.DataFrame(index=self.grn.all_genes)
        for prog_name, prog_info in self.grn.programs.items():
            var[prog_name] = var.index.isin(prog_info['genes'])

        adata = ad.AnnData(X=X, obs=obs, var=var)

        # Ground-truth log2 fold changes (KO genes x all genes)
        gt_log2fc = self.get_ground_truth_log2fc(genes_to_perturb)
        adata.uns['ground_truth_log2fc'] = gt_log2fc.values
        adata.uns['ground_truth_log2fc_index'] = list(gt_log2fc.index)

        # Store network and metadata
        adata.varm['W'] = self.grn.W
        adata.uns['V'] = self.grn.V
        adata.uns['program_names'] = self.grn.program_names
        adata.uns['perturbation_effects'] = effect_store
        adata.uns['simulator_params'] = {
            'perturbation_strength': self.perturbation_strength,
            'damping_factor': self.damping_factor,
            'n_propagation_steps': self.n_propagation_steps,
            'n_cells_per_perturbation': n_cells_per_perturbation,
            'count_distribution': count_distribution,
        }

        if self.verbose:
            n_pert = len(genes_to_perturb)
            n_ctrl = self.control_counts.shape[0] if include_control else 0
            print(f"\nGenerated AnnData: {adata.shape}")
            print(f"  Control cells: {n_ctrl}")
            print(f"  KO conditions: {n_pert}")
            print(f"  Cells per KO: {n_cells_per_perturbation}")
            print(f"  Total perturbed cells: {n_pert * n_cells_per_perturbation}")

        return adata

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_ground_truth_log2fc(
        self,
        genes_to_perturb: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Ground-truth log2 fold changes used to generate perturbed counts.

        For each KO gene, the simulator multiplies control counts by
        fc = exp(W @ total_delta_h).  This method returns log2(fc) for
        every (KO gene, target gene) pair, plus sets the KO gene itself
        to -inf (complete silencing).

        Parameters
        ----------
        genes_to_perturb : list of str, optional
            Subset of perturbable genes.  None = all perturbable genes.

        Returns
        -------
        df : pd.DataFrame
            Shape (n_perturbed_genes, n_genes).
            Rows = KO genes, columns = all genes in the network.
            Values are log2 fold changes.
        """
        if genes_to_perturb is None:
            genes_to_perturb = self.perturbable_genes

        log2fc_matrix = np.zeros((len(genes_to_perturb), self.grn.n_genes))

        for i, gene in enumerate(genes_to_perturb):
            eff = self._get_or_compute_effect(gene)
            log2fc_matrix[i, :] = eff['log_fold_change'] / np.log(2)
            gene_idx = self.grn.gene_to_idx[gene]
            log2fc_matrix[i, gene_idx] = -np.inf

        return pd.DataFrame(
            log2fc_matrix,
            index=genes_to_perturb,
            columns=self.grn.all_genes,
        )

    def get_effect_summary(self) -> pd.DataFrame:
        """
        One-row-per-gene summary of pre-computed perturbation effects.

        Columns include number of affected programs, max/mean program shift,
        number of downstream genes affected, and a primary-dominance ratio
        (fraction of total effect coming from the primary step alone).
        """
        records = []
        for gene in self.perturbable_genes:
            eff = self.effects[gene]
            total_h = eff['total_program_shift']
            primary_h = eff['primary_program_shift']
            total_g = eff['total_gene_shift']

            nonzero_h = total_h[total_h != 0]
            records.append({
                'gene': gene,
                'n_affected_programs': len(eff['affected_programs']),
                'affected_programs': ', '.join(eff['affected_programs']),
                'max_abs_program_shift': np.max(np.abs(total_h)),
                'mean_abs_program_shift': (
                    np.mean(np.abs(nonzero_h)) if len(nonzero_h) > 0 else 0.0
                ),
                'n_genes_affected': int(np.sum(np.abs(total_g) > 1e-4)),
                'max_abs_gene_shift': np.max(np.abs(total_g)),
                'mean_abs_gene_shift': np.mean(np.abs(total_g)),
                'primary_dominance': (
                    np.linalg.norm(primary_h)
                    / (np.linalg.norm(total_h) + 1e-10)
                ),
            })

        return (
            pd.DataFrame(records)
            .sort_values('max_abs_program_shift', ascending=False)
            .reset_index(drop=True)
        )
