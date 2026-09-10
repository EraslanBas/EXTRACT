from libraries import *
from parameters import *
from util import *
from GetGOPrograms import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import matplotlib.patches as mpatches
from typing import Dict, List, Set
import warnings
warnings.filterwarnings('ignore')

class GeneRegulatoryNetwork:
    """
    Bidirectional gene-program network with mixed-sign regulation.
    
    Each program has its own Gaussian distribution for W and V weights.
    Genes in programs have non-zero weights, genes not in any program are exactly zero.
    """
    
    def __init__(
        self,
        programs: Dict[str, Dict],
        all_genes: List[str],
        w_activation_prob: float = 0.7,
        v_negative_prob: float = 0.7,
        w_mean_range: tuple = (0.6, 1.0),
        w_std_range: tuple = (0.1, 0.3),
        v_mean_range: tuple = (0.15, 0.35),
        v_std_range: tuple = (0.05, 0.15),
        min_magnitude: float = 0.01,
        seed: int = 42,
        verbose: bool = False
    ):
        """
        Parameters:
        -----------
        programs : dict
            Gene programs with their associated genes
        all_genes : list
            All genes (including those not in any program)
        w_activation_prob : float
            Probability that W connection is positive (activation)
        v_negative_prob : float
            Probability that V connection is negative (feedback)
        w_mean_range : tuple
            Range for W distribution means across programs (min, max)
        w_std_range : tuple
            Range for W distribution stds across programs (min, max)
        v_mean_range : tuple
            Range for V distribution means across programs (min, max)
        v_std_range : tuple
            Range for V distribution stds across programs (min, max)
        min_magnitude : float
            Minimum absolute value for non-zero weights
        seed : int
            Random seed
        verbose : bool
            Print statistics
        """
        self.programs = programs
        self.program_names = list(programs.keys())
        self.all_genes = all_genes
        self.n_programs = len(programs)
        self.n_genes = len(all_genes)
        
        self.w_activation_prob = w_activation_prob
        self.v_negative_prob = v_negative_prob
        self.w_mean_range = w_mean_range
        self.w_std_range = w_std_range
        self.v_mean_range = v_mean_range
        self.v_std_range = v_std_range
        self.min_magnitude = min_magnitude
        self.verbose = verbose
        
        self.rng = np.random.default_rng(seed)
        
        self.gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
        self.program_to_idx = {prog: i for i, prog in enumerate(self.program_names)}
        
        # Per-program distribution parameters
        self.program_w_params = {}  # {program_name: {'mean': float, 'std': float}}
        self.program_v_params = {}  # {program_name: {'mean': float, 'std': float}}
        
        self.W = None
        self.V = None
        
        self._initialize_program_distributions()
        self._initialize_weights()
        
        if self.verbose:
            self._print_statistics()
    
    def _initialize_program_distributions(self):
        """
        Initialize unique Gaussian distribution parameters for each program.
        """
        if self.verbose:
            print("\n" + "="*80)
            print("INITIALIZING PER-PROGRAM DISTRIBUTIONS")
            print("="*80)
        
        for prog_name in self.program_names:
            # W distribution parameters
            w_mean = self.rng.uniform(self.w_mean_range[0], self.w_mean_range[1])
            w_std = self.rng.uniform(self.w_std_range[0], self.w_std_range[1])
            
            self.program_w_params[prog_name] = {
                'mean': w_mean,
                'std': w_std
            }
            
            # V distribution parameters
            v_mean = self.rng.uniform(self.v_mean_range[0], self.v_mean_range[1])
            v_std = self.rng.uniform(self.v_std_range[0], self.v_std_range[1])
            
            self.program_v_params[prog_name] = {
                'mean': v_mean,
                'std': v_std
            }
            
            if self.verbose:
                print(f"{prog_name:30s}  W: N(μ={w_mean:.3f}, σ={w_std:.3f})  "
                      f"V: N(μ={v_mean:.3f}, σ={v_std:.3f})")
        
        if self.verbose:
            print("="*80)
    
    def _initialize_weights(self):
        """
        Initialize W and V based on per-program Gaussian distributions.
        Genes in programs: non-zero weights from program-specific Gaussian.
        Genes not in programs: exactly zero.

        Sign convention
        ---------------
        V[p, g] represents the effect on program p when gene g is knocked
        out.  Its sign is always opposite to W[g, p]:

        * W[g, p] > 0  (gene co-active with program)
          → V[p, g] < 0  (KO suppresses the program)
        * W[g, p] < 0  (gene anti-active with program)
          → V[p, g] > 0  (KO enhances the program)
        """

        self.W = np.zeros((self.n_genes, self.n_programs))
        self.V = np.zeros((self.n_programs, self.n_genes))

        # Track which genes are in at least one program
        genes_in_programs = set()
        for prog_info in self.programs.values():
            genes_in_programs.update(prog_info['genes'])

        if self.verbose:
            print(f"\nInitializing weights:")
            print(f"  Total genes: {self.n_genes}")
            print(f"  Genes in programs: {len(genes_in_programs)}")
            print(f"  Genes not in programs: {self.n_genes - len(genes_in_programs)}")

        # For each program, assign weights from its distribution
        for prog_name, prog_info in self.programs.items():
            prog_idx = self.program_to_idx[prog_name]
            prog_genes = prog_info['genes']

            # Get distribution parameters for this program
            w_params = self.program_w_params[prog_name]
            v_params = self.program_v_params[prog_name]

            for gene in prog_genes:
                if gene in self.gene_to_idx:
                    gene_idx = self.gene_to_idx[gene]

                    # === W: program → gene ===
                    # Sample magnitude from Gaussian
                    w_magnitude = self.rng.normal(w_params['mean'], w_params['std'])
                    # Ensure minimum magnitude and positive
                    w_magnitude = max(abs(w_magnitude), self.min_magnitude)

                    # Apply sign based on activation probability
                    w_sign = 1 if self.rng.random() < self.w_activation_prob else -1
                    self.W[gene_idx, prog_idx] = w_sign * w_magnitude

                    # === V: gene → program (KO effect) ===
                    # Sample magnitude from Gaussian
                    v_magnitude = self.rng.normal(v_params['mean'], v_params['std'])
                    # Ensure minimum magnitude and positive
                    v_magnitude = max(abs(v_magnitude), self.min_magnitude)

                    # Sign is opposite to W: KO of a co-active gene
                    # suppresses the program, KO of an anti-active gene
                    # enhances it.
                    v_sign = -w_sign
                    self.V[prog_idx, gene_idx] = v_sign * v_magnitude
        
        # Enforce magnitude constraint
        self._enforce_magnitude_constraint()
        
        # Verify no genes outside programs have non-zero weights
        if self.verbose:
            self._verify_program_gene_constraint(genes_in_programs)
    
    def _enforce_magnitude_constraint(self):
        """Ensure |W_ij| > |V_ji| for connected gene-program pairs."""
        violations_before = 0
        violations_fixed = 0
        
        for gene_idx in range(self.n_genes):
            for prog_idx in range(self.n_programs):
                w_val = abs(self.W[gene_idx, prog_idx])
                v_val = abs(self.V[prog_idx, gene_idx])
                
                if w_val > 0 and v_val > 0:
                    if v_val >= w_val:
                        violations_before += 1
                        # Scale down V to maintain |W| > |V|
                        self.V[prog_idx, gene_idx] *= 0.8 * (w_val / v_val)
                        violations_fixed += 1
        
        if self.verbose and violations_before > 0:
            print(f"\n  Magnitude constraint violations found: {violations_before}")
            print(f"  Violations fixed: {violations_fixed}")
    
    def _verify_program_gene_constraint(self, genes_in_programs):
        """Verify that only genes in programs have non-zero weights."""
        
        # Check W
        w_nonzero_genes = set()
        for gene_idx in range(self.n_genes):
            if np.any(self.W[gene_idx, :] != 0):
                w_nonzero_genes.add(self.all_genes[gene_idx])
        
        # Check V
        v_nonzero_genes = set()
        for gene_idx in range(self.n_genes):
            if np.any(self.V[:, gene_idx] != 0):
                v_nonzero_genes.add(self.all_genes[gene_idx])
        
        # Should be exactly the same
        w_outside = w_nonzero_genes - genes_in_programs
        v_outside = v_nonzero_genes - genes_in_programs
        
        print(f"\n  Verification:")
        print(f"    Genes with non-zero W: {len(w_nonzero_genes)}")
        print(f"    Genes with non-zero V: {len(v_nonzero_genes)}")
        print(f"    Genes in programs: {len(genes_in_programs)}")
        
        if len(w_outside) > 0:
            warnings.warn(f"Found {len(w_outside)} genes with non-zero W outside programs!")
        else:
            print(f"    ✓ All non-zero W are in programs")
        
        if len(v_outside) > 0:
            warnings.warn(f"Found {len(v_outside)} genes with non-zero V outside programs!")
        else:
            print(f"    ✓ All non-zero V are in programs")
    
    def _print_statistics(self):
        """Print network statistics."""
        print("\n" + "="*80)
        print("GENE REGULATORY NETWORK STATISTICS")
        print("="*80)
        
        # Overall W statistics
        w_positive = (self.W > 0).sum()
        w_negative = (self.W < 0).sum()
        w_total = w_positive + w_negative
        
        print(f"\nW (Program → Gene) - OVERALL:")
        print(f"  Shape: {self.W.shape}")
        print(f"  Non-zero: {w_total} / {self.W.size} ({w_total/self.W.size*100:.2f}%)")
        print(f"  Positive (activation): {w_positive} ({w_positive/w_total*100:.1f}%)")
        print(f"  Negative (repression): {w_negative} ({w_negative/w_total*100:.1f}%)")
        print(f"  Mean |W|: {np.abs(self.W[self.W != 0]).mean():.3f}")
        print(f"  Std |W|: {np.abs(self.W[self.W != 0]).std():.3f}")
        
        # Overall V statistics
        v_positive = (self.V > 0).sum()
        v_negative = (self.V < 0).sum()
        v_total = v_positive + v_negative
        
        print(f"\nV (Gene → Program) - OVERALL:")
        print(f"  Shape: {self.V.shape}")
        print(f"  Non-zero: {v_total} / {self.V.size} ({v_total/self.V.size*100:.2f}%)")
        print(f"  Positive (activation): {v_positive} ({v_positive/v_total*100:.1f}%)")
        print(f"  Negative (feedback): {v_negative} ({v_negative/v_total*100:.1f}%)")
        print(f"  Mean |V|: {np.abs(self.V[self.V != 0]).mean():.3f}")
        print(f"  Std |V|: {np.abs(self.V[self.V != 0]).std():.3f}")
        
        # Per-program statistics
        print(f"\nPER-PROGRAM STATISTICS:")
        print(f"{'Program':<30s} {'W Non-Zero':>10s} {'W Mean':>10s} {'W Std':>10s} "
              f"{'V Non-Zero':>10s} {'V Mean':>10s} {'V Std':>10s}")
        print("-" * 100)
        
        for prog_idx, prog_name in enumerate(self.program_names):
            w_col = self.W[:, prog_idx]
            w_nonzero = w_col[w_col != 0]
            
            v_row = self.V[prog_idx, :]
            v_nonzero = v_row[v_row != 0]
            
            w_mean = np.abs(w_nonzero).mean() if len(w_nonzero) > 0 else 0
            w_std = np.abs(w_nonzero).std() if len(w_nonzero) > 0 else 0
            v_mean = np.abs(v_nonzero).mean() if len(v_nonzero) > 0 else 0
            v_std = np.abs(v_nonzero).std() if len(v_nonzero) > 0 else 0
            
            print(f"{prog_name:<30s} {len(w_nonzero):>10d} {w_mean:>10.3f} {w_std:>10.3f} "
                  f"{len(v_nonzero):>10d} {v_mean:>10.3f} {v_std:>10.3f}")
        
        # Magnitude constraint check
        violations = 0
        for i in range(self.n_genes):
            for j in range(self.n_programs):
                if abs(self.W[i, j]) > 0 and abs(self.V[j, i]) > 0:
                    if abs(self.V[j, i]) >= abs(self.W[i, j]):
                        violations += 1
        
        print(f"\n|W| > |V| constraint violations: {violations}")
        print("="*80)
    
    def get_program_distributions(self) -> Dict:
        """Get the distribution parameters for all programs."""
        return {
            'W': self.program_w_params,
            'V': self.program_v_params
        }
    
    def get_genes_in_programs(self) -> set:
        """Get set of genes that belong to at least one program."""
        genes_in_programs = set()
        for prog_info in self.programs.values():
            genes_in_programs.update(prog_info['genes'])
        return genes_in_programs


