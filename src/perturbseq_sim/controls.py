from .libraries import *
from .config import *
from .utils import *
from .programs import *
from .grn import *
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import matplotlib.patches as mpatches
from typing import Dict, List, Set
import warnings
warnings.filterwarnings('ignore')
from matplotlib.patches import Rectangle
from scipy.stats import multivariate_normal, nbinom, poisson



class ControlDataGenerator:
    """
    Generate control single-cell RNA-seq data based on GRN structure.
    
    Gene expression is driven by latent program activities through the W matrix:
    - Positive W values → gene expression increases with program activity
    - Negative W values → gene expression decreases with program activity
    
    Program activities are sampled with realistic correlations based on biological knowledge.
    """
    
    def __init__(
        self,
        grn,
        go_gen,
        seed: int = 42
    ):
        """
        Parameters:
        -----------
        grn : GeneRegulatoryNetwork
            Gene regulatory network with W and V matrices
        go_gen : GOProgramGenerator
            Gene program generator
        seed : int
            Random seed
        """
        self.grn = grn
        self.go_gen = go_gen
        self.rng = np.random.default_rng(seed)
        
        # Program correlation structure based on biology
        self.program_correlations = self._define_program_correlations()
        
        # Expression parameters
        self.baseline_expression = None
        self.dispersion_params = None
        
        self._initialize_expression_parameters()
    
    def _define_program_correlations(
        self,
        std: float = 0.05,
        random_state: int | None = 42,
    ) -> np.ndarray:
        """
        Define biologically realistic correlations between programs.
    
        Correlations are sampled from a normal distribution centered at
        biologically motivated mean values, then the matrix is projected
        to the nearest positive-definite correlation-like matrix.
        """
        n_programs = self.grn.n_programs
        rng = np.random.default_rng(random_state)
    
        # Start with identity (no correlation)
        corr_matrix = np.eye(n_programs)
    
        # Define correlation patterns based on known biology
        program_names = self.grn.program_names
    
        # Helper function to set sampled correlation
        def set_corr(prog1: str, prog2: str, mean_value: float):
            if prog1 in program_names and prog2 in program_names:
                i = program_names.index(prog1)
                j = program_names.index(prog2)
    
                sampled_value = rng.normal(loc=mean_value, scale=std)
                sampled_value = np.clip(sampled_value, -1.0, 1.0)
    
                corr_matrix[i, j] = sampled_value
                corr_matrix[j, i] = sampled_value
    
        # Cell cycle and metabolism (proliferating cells need energy)
        set_corr('cell_cycle', 'metabolism', 0.5)
        set_corr('cell_cycle', 'ribosome_biogenesis', 0.6)
    
        # Apoptosis anti-correlates with cell cycle
        set_corr('apoptosis', 'cell_cycle', -0.4)
    
        # Stress responses correlate
        set_corr('oxidative_stress', 'er_stress', 0.5)
        set_corr('oxidative_stress', 'autophagy', 0.4)
        set_corr('er_stress', 'autophagy', 0.4)
    
        # Hypoxia and angiogenesis
        set_corr('hypoxia_response', 'angiogenesis', 0.6)
        set_corr('hypoxia_response', 'metabolism', 0.3)
    
        # Immune and inflammatory responses
        set_corr('immune_response', 'inflammatory_response', 0.7)
    
        # EMT and cell adhesion (inverse)
        set_corr('emt', 'cell_adhesion', -0.5)
    
        # DNA repair and cell cycle
        set_corr('dna_repair', 'cell_cycle', 0.3)
    
        # Ensure positive definite
        corr_matrix = self._nearest_positive_definite(corr_matrix)
    
        return corr_matrix
        
    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive definite matrix."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self._is_positive_definite(A3):
            return A3
        
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        
        return A3
    
    def _is_positive_definite(self, A: np.ndarray) -> bool:
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def _initialize_expression_parameters(self):
        """Initialize baseline expression and dispersion parameters."""
        n_genes = len(self.grn.all_genes)

        # Heavy-tailed lognormal-like baseline in natural log space, so the
        # linear expression spans 4-5 orders of magnitude as in real scRNA-seq.
        self.baseline_expression = self.rng.normal(loc=0.5, scale=2.0, size=n_genes)

        # Dispersion parameters for negative binomial
        self.dispersion_params = self.rng.uniform(0.1, 2.0, n_genes)
    
    def get_program_correlations(self, as_dataframe: bool = False):
        """
        Get the program correlation matrix.
        
        Parameters:
        -----------
        as_dataframe : bool
            If True, return as pandas DataFrame with program names as index/columns
        
        Returns:
        --------
        correlations : np.ndarray or pd.DataFrame
            Program correlation matrix
        """
        if as_dataframe:
            return pd.DataFrame(
                self.program_correlations,
                index=self.grn.program_names,
                columns=self.grn.program_names
            )
        else:
            return self.program_correlations
    
    def plot_program_correlations(
        self,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'RdBu_r',
        annot: bool = True,
        fmt: str = '.2f',
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot the program correlation matrix as a heatmap.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap name
        annot : bool
            If True, annotate cells with correlation values
        fmt : str
            Format string for annotations
        save_path : str, optional
            If provided, save figure to this path
        show : bool
            If True, display the plot
        """
        # Get correlation matrix as DataFrame
        corr_df = self.get_program_correlations(as_dataframe=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            corr_df,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            annot=annot,
            fmt=fmt,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            ax=ax
        )
        
        # Format
        ax.set_title('Program Activity Correlations', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Programs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Programs', fontsize=12, fontweight='bold')
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, ax
        
    def sample_program_activities(
        self,
        n_cells: int,
        activity_mean=0.0,
        activity_std: float = 1.0,
        use_correlations: bool = True
    ) -> np.ndarray:
        """
        Sample program activities for n_cells.

        Parameters:
        -----------
        n_cells : int
            Number of cells to generate
        activity_mean : float or array-like of shape (n_programs,)
            Mean program activity (in log space).  Scalar applies uniformly
            to all programs; a length-n_programs vector sets a per-program
            mean so different programs can be active at different levels
            (context-specific control cells).
        activity_std : float
            Standard deviation of program activities
        use_correlations : bool
            If True, use correlated program activities

        Returns:
        --------
        activities : np.ndarray
            Shape (n_cells, n_programs)
        """
        n_programs = self.grn.n_programs

        mean_arr = np.broadcast_to(
            np.asarray(activity_mean, dtype=np.float64),
            (n_programs,)
        ).astype(np.float64)

        if use_correlations:
            cov = self.program_correlations * (activity_std ** 2)
            activities = self.rng.multivariate_normal(mean_arr, cov, size=n_cells)
        else:
            activities = self.rng.normal(
                mean_arr,
                activity_std,
                size=(n_cells, n_programs),
            )

        # Ensure non-negative activities (apply softplus)
        activities = np.log1p(np.exp(activities))

        return activities
    
    def generate_expression(
        self,
        program_activities: np.ndarray,
        use_feedback: bool = False,
        cell_size_factor_cv: float = 0.3,
        technical_noise_scale: float = 0.1
    ) -> np.ndarray:
        """
        Generate gene expression from program activities via GRN.
        
        Expression model:
        log(expression) = baseline + W @ activity + noise
        
        Parameters:
        -----------
        program_activities : np.ndarray
            Shape (n_cells, n_programs)
        use_feedback : bool
            If True, include V matrix feedback (experimental)
        cell_size_factor_cv : float
            Coefficient of variation for cell size factors
        technical_noise_scale : float
            Scale of technical noise
        
        Returns:
        --------
        expression : np.ndarray
            Shape (n_cells, n_genes) - log-transformed expression
        """
        n_cells = program_activities.shape[0]
        n_genes = len(self.grn.all_genes)
        
        # Base expression from W matrix: gene_expr = W @ program_activity
        expression = program_activities @ self.grn.W.T  # (n_cells, n_genes)
        
        # Add baseline
        expression += self.baseline_expression[np.newaxis, :]
        
        # Add noise
        biological_noise = self.rng.normal(0, 0.2, size=(n_cells, n_genes))
        expression += biological_noise
        
        
        # Cell size factors
        cell_size_factors = self.rng.lognormal(
            mean=0,
            sigma=cell_size_factor_cv,
            size=n_cells
        )
        expression += np.log(cell_size_factors)[:, np.newaxis]
        
        return expression
    
    def expression_to_counts(
        self,
        expression: np.ndarray,
        count_distribution: str = 'negative_binomial',
        total_counts_per_cell: float = 10000
    ) -> np.ndarray:
        """
        Convert log-expression to count data.
        
        Parameters:
        -----------
        expression : np.ndarray
            Log-transformed expression (n_cells × n_genes)
        count_distribution : str
            'negative_binomial' or 'poisson'
        total_counts_per_cell : float
            Target total counts per cell
        
        Returns:
        --------
        counts : np.ndarray
            Count matrix (n_cells × n_genes)
        """
        n_cells, n_genes = expression.shape
        
        # Convert log-expression to expected counts
        lambda_exp = np.exp(expression)
        
        # Normalize to total counts per cell
        row_sums = lambda_exp.sum(axis=1, keepdims=True)
        lambda_normalized = lambda_exp / row_sums * total_counts_per_cell
        
        # Sample counts
        counts = np.zeros((n_cells, n_genes), dtype=int)
        
        if count_distribution == 'negative_binomial':
            for i in range(n_cells):
                for j in range(n_genes):
                    mu = lambda_normalized[i, j]
                    r = self.dispersion_params[j]
                    
                    p = r / (r + mu)
                    if p > 0 and p < 1:
                        counts[i, j] = self.rng.negative_binomial(r, p)
                    else:
                        counts[i, j] = self.rng.poisson(mu)
        
        elif count_distribution == 'poisson':
            for i in range(n_cells):
                for j in range(n_genes):
                    counts[i, j] = self.rng.poisson(lambda_normalized[i, j])
        
        else:
            raise ValueError(f"Unknown distribution: {count_distribution}")
        
        return counts
    
    def generate_control_data(
        self,
        n_cells: int,
        activity_mean: float = 0.0,
        activity_std: float = 1.0,
        use_correlations: bool = True,
        use_feedback: bool = False,
        count_distribution: str = 'negative_binomial',
        total_counts_per_cell: float = 10000
    ):
        """
        Generate complete control dataset as AnnData object.
        
        Parameters:
        -----------
        n_cells : int
            Number of cells
        activity_mean : float
            Mean program activity
        activity_std : float
            Standard deviation of program activities
        use_correlations : bool
            Use correlated program activities
        use_feedback : bool
            Include V matrix feedback
        count_distribution : str
            'negative_binomial' or 'poisson'
        total_counts_per_cell : float
            Target library size
        
        Returns:
        --------
        adata : AnnData
            AnnData object with count matrix
        """
        try:
            import anndata as ad
        except ImportError:
            raise ImportError("Please install scanpy: pip install scanpy")
        
        # Sample program activities
        activities = self.sample_program_activities(
            n_cells=n_cells,
            activity_mean=activity_mean,
            activity_std=activity_std,
            use_correlations=use_correlations
        )
        
        # Generate expression
        expression = self.generate_expression(
            program_activities=activities,
            use_feedback=use_feedback
        )
        
        # Convert to counts
        counts = self.expression_to_counts(
            expression=expression,
            count_distribution=count_distribution,
            total_counts_per_cell=total_counts_per_cell
        )
        
        # Create AnnData
        adata = ad.AnnData(
            X=counts,
            obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_cells)]),
            var=pd.DataFrame(index=self.grn.all_genes)
        )
        
        return adata
    
    def generate_multiple_conditions(
        self,
        n_conditions: int,
        n_cells_per_condition: int,
        activity_mean_range: Tuple[float, float] = (-0.5, 0.5),
        activity_std_range: Tuple[float, float] = (0.8, 1.2),
        **kwargs
    ):
        """
        Generate multiple control conditions with varying program activities.
        
        Parameters:
        -----------
        n_conditions : int
            Number of different conditions
        n_cells_per_condition : int
            Cells per condition
        activity_mean_range : tuple
            Range for sampling activity means
        activity_std_range : tuple
            Range for sampling activity stds
        **kwargs : 
            Additional arguments passed to generate_control_data
        
        Returns:
        --------
        adata : AnnData
            Combined AnnData object with all conditions
        """
        try:
            import anndata as ad
        except ImportError:
            raise ImportError("Please install scanpy: pip install scanpy")
        
        adatas = []
        
        for i in range(n_conditions):
            # Sample parameters
            activity_mean = self.rng.uniform(*activity_mean_range)
            activity_std = self.rng.uniform(*activity_std_range)
            
            # Generate data
            adata = self.generate_control_data(
                n_cells=n_cells_per_condition,
                activity_mean=activity_mean,
                activity_std=activity_std,
                **kwargs
            )
            
            # Add condition label
            adata.obs['condition'] = f"condition_{i}"
            adata.obs['condition_id'] = i
            
            adatas.append(adata)
        
        # Concatenate
        adata_combined = ad.concat(adatas, axis=0, join='outer')
        adata_combined.obs_names_make_unique()
        
        return adata_combined


