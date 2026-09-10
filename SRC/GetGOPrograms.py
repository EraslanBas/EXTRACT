import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Optional, Union, Tuple
import urllib.request
import gzip
import shutil
import warnings
import time
from collections import defaultdict

# Optional imports
try:
    from goatools.obo_parser import GODag
    GOATOOLS_AVAILABLE = True
except ImportError:
    GOATOOLS_AVAILABLE = False
    warnings.warn("goatools not available. Install with: pip install goatools")

try:
    import mygene
    MYGENE_AVAILABLE = True
except ImportError:
    MYGENE_AVAILABLE = False
    warnings.warn("mygene not available. Install with: pip install mygene")


class GOProgramGenerator:
    """
    Generate gene programs based on Gene Ontology biological processes.
    Connects to real GO database and gene annotations.
    """
    
    def __init__(
        self, 
        n_programs: int = 15,
        organism: str = "human", 
        total_genes: int = 5000,
        min_program_size: int = 20,
        max_program_size: int = 150,
        add_synthetic_program_genes: bool = True,
        n_synthetic_per_program: Union[int, Tuple[int, int], List[int]] = [5,25],
        synthetic_overlap_mean: float = 0.15,
        synthetic_overlap_std: float = 0.05,
        add_filler_genes: bool = True,
        seed: int = 42,
        data_dir: str = "./go_data",
        use_cached: bool = True,
        verbose: bool = False
    ):
        """
        Parameters:
        -----------
        n_programs : int
            Number of gene programs to use (1-15).
        organism : str
            'human' or 'mouse'
        total_genes : int
            Target total number of genes (only used if add_filler_genes=True)
        min_program_size : int
            Minimum genes per program (real genes only)
        max_program_size : int
            Maximum genes per program (real genes only)
        add_synthetic_program_genes : bool
            If True, add synthetic genes to programs.
        n_synthetic_per_program : int or tuple(int, int) or list[int, int]
            Number of synthetic genes to add per program.
        synthetic_overlap_mean : float (0 to 1)
            Mean fraction of synthetic genes shared between programs.
        synthetic_overlap_std : float
            Standard deviation for overlap percentage.
        add_filler_genes : bool
            If True, add GENE_XXXX genes to reach total_genes.
        seed : int
            Random seed
        data_dir : str
            Directory to store GO data files
        use_cached : bool
            Use cached data if available
        verbose : bool
            Print progress messages (kept for compatibility but not used)
        """
        if not 1 <= n_programs <= 15:
            raise ValueError(f"n_programs must be between 1 and 15, got {n_programs}")
        
        self.n_programs = n_programs
        self.organism = organism
        self.total_genes = total_genes
        self.min_program_size = min_program_size
        self.max_program_size = max_program_size
        self.add_synthetic_program_genes = add_synthetic_program_genes
        self.n_synthetic_per_program = n_synthetic_per_program
        self.synthetic_overlap_mean = synthetic_overlap_mean
        self.synthetic_overlap_std = synthetic_overlap_std
        self.add_filler_genes = add_filler_genes
        self.rng = np.random.default_rng(seed)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.use_cached = use_cached
        self.verbose = verbose
        
        # GO database and annotations
        self.go_dag = None
        self.gene_to_go = None
        self.go_to_genes = None
        
        # Programs
        self.predefined_programs = self._get_predefined_programs()
        self.all_genes_list = None
        
        # Synthetic gene tracking
        self.shared_synthetic_genes = []
        self.program_unique_synthetic = {}
        self.synthetic_gene_membership = {}
        
        # Load GO data
        self._load_go_database()
        self._load_gene_annotations()
        
        # Expand programs
        self._expand_programs()
    
    def _download_file(
        self, 
        url: str, 
        output_path: Path, 
        decompress: bool = False,
        max_retries: int = 3
    ):
        """Download file from URL with proper headers and retry logic."""
        if self.use_cached and output_path.exists():
            return True
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; GOProgramGenerator/1.0)',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers=headers)
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(output_path, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
                
                if decompress and output_path.suffix == '.gz':
                    decompressed_path = output_path.with_suffix('')
                    with gzip.open(output_path, 'rb') as f_in:
                        with open(decompressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return False
        
        return False
    
    def _load_go_database(self):
        """Load Gene Ontology OBO file."""
        obo_file = self.data_dir / "go-basic.obo"
        
        urls = [
            "http://current.geneontology.org/ontology/go-basic.obo",
            "http://purl.obolibrary.org/obo/go/go-basic.obo",
        ]
        
        success = False
        if not obo_file.exists() or not self.use_cached:
            for url in urls:
                if self._download_file(url, obo_file):
                    success = True
                    break
        else:
            success = True
        
        if not success:
            return
        
        if GOATOOLS_AVAILABLE:
            try:
                self.go_dag = GODag(str(obo_file))
            except Exception as e:
                self.go_dag = None
    
    def _load_gene_annotations(self):
        """Load gene-to-GO annotations."""
        success = False
        
        if MYGENE_AVAILABLE and not success:
            try:
                success = self._load_annotations_mygene()
            except Exception as e:
                pass
        
        if not success:
            try:
                success = self._load_annotations_ncbi()
            except Exception as e:
                pass
        
        if not success:
            self._load_annotations_fallback()
    
    def _load_annotations_mygene(self) -> bool:
        """Load annotations using mygene library."""
        mg = mygene.MyGeneInfo()
        
        taxon_map = {'human': 9606, 'mouse': 10090}
        if self.organism not in taxon_map:
            raise ValueError(f"Organism {self.organism} not supported")
        
        species = taxon_map[self.organism]
        
        self.go_to_genes = defaultdict(set)
        self.gene_to_go = defaultdict(set)
        
        for prog_name, prog_info in self.predefined_programs.items():
            go_id = prog_info['go_id']
            
            try:
                results = mg.query(
                    f'go:{go_id}',
                    species=species,
                    fields='symbol,go',
                    size=1000
                )
                
                if 'hits' in results:
                    for hit in results['hits']:
                        if 'symbol' in hit:
                            gene_symbol = hit['symbol']
                            self.go_to_genes[go_id].add(gene_symbol)
                            self.gene_to_go[gene_symbol].add(go_id)
                
                time.sleep(0.5)
            
            except Exception as e:
                pass
        
        total_genes = sum(len(genes) for genes in self.go_to_genes.values())
        if total_genes > 0:
            return True
        
        return False
    
    def _load_annotations_ncbi(self) -> bool:
        """Load annotations from NCBI gene2go file."""
        taxon_map = {'human': '9606', 'mouse': '10090'}
        if self.organism not in taxon_map:
            raise ValueError(f"Organism {self.organism} not supported")
        
        taxon_id = taxon_map[self.organism]
        
        # Download files
        gene2go_file = self.data_dir / "gene2go.gz"
        if not self._download_file(
            "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz",
            gene2go_file
        ):
            return False
        
        gene_info_file = self.data_dir / "gene_info.gz"
        organism_urls = {
            'human': "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
            'mouse': "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz"
        }
        
        if self.organism in organism_urls:
            if not self._download_file(organism_urls[self.organism], gene_info_file):
                self._download_file(
                    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz",
                    gene_info_file
                )
        
        # Load gene symbols
        gene_id_to_symbol = {}
        try:
            with gzip.open(gene_info_file, 'rt') as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split('\t')
                    if parts[0] == taxon_id:
                        gene_id_to_symbol[parts[1]] = parts[2]
        except Exception as e:
            return False
        
        # Load GO annotations
        self.go_to_genes = defaultdict(set)
        self.gene_to_go = defaultdict(set)
        
        go_terms = {prog_info['go_id'] for prog_info in self.predefined_programs.values()}
        
        try:
            with gzip.open(gene2go_file, 'rt') as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split('\t')
                    if parts[0] == taxon_id:
                        gene_id = parts[1]
                        go_id = parts[2]
                        
                        if go_id in go_terms and gene_id in gene_id_to_symbol:
                            symbol = gene_id_to_symbol[gene_id]
                            self.go_to_genes[go_id].add(symbol)
                            self.gene_to_go[symbol].add(go_id)
        except Exception as e:
            return False
        
        total_genes = sum(len(genes) for genes in self.go_to_genes.values())
        if total_genes > 0:
            return True
        
        return False
    
    def _load_annotations_fallback(self):
        """Fallback to curated gene lists."""
        self.go_to_genes = defaultdict(set)
        self.gene_to_go = defaultdict(set)
        
        expanded_genes = self._get_expanded_curated_genes()
        
        for prog_name, prog_info in self.predefined_programs.items():
            go_id = prog_info['go_id']
            genes = set(prog_info['core_genes'])
            
            if prog_name in expanded_genes:
                genes.update(expanded_genes[prog_name])
            
            for gene in genes:
                self.go_to_genes[go_id].add(gene)
                self.gene_to_go[gene].add(go_id)
    
    def _get_expanded_curated_genes(self) -> Dict[str, List[str]]:
        """Expanded curated gene sets for fallback."""
        return {
            'cell_cycle': [
                'CCNA2', 'CCNB1', 'CCND1', 'CCNE1', 'CDK1', 'CDK2', 'CDK4', 'CDK6',
                'E2F1', 'E2F2', 'E2F3', 'MKI67', 'PCNA', 'MCM2', 'MCM3', 'MCM4', 'MCM5',
                'MCM6', 'MCM7', 'TOP2A', 'AURKA', 'AURKB', 'PLK1', 'BUB1', 'BUB3',
                'CCNB2', 'CDC20', 'CDC25A', 'CDC25B', 'CDC25C', 'CCNA1', 'CCND2', 'CCND3',
                'CCNE2', 'CDK7', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'RB1', 'TP53',
                'SKP2', 'FBXW7', 'CDC6', 'CDC7', 'ORC1', 'ORC2', 'ORC3', 'ORC4', 'ORC5',
                'ORC6', 'MCM10', 'GINS1', 'GINS2', 'GINS3', 'GINS4', 'POLA1', 'POLA2',
                'POLE', 'POLE2', 'POLE3', 'POLE4', 'RFC1', 'RFC2', 'RFC3', 'RFC4', 'RFC5',
                'CHEK1', 'CHEK2', 'ATM', 'ATR', 'WEE1', 'MYT1', 'CDC45',
                'RPA1', 'RPA2', 'RPA3', 'BRCA1', 'BRCA2', 'RAD51', 'XRCC2', 'XRCC3',
                'MAD2L1', 'BUB1B', 'ESPL1', 'SMC1A', 'SMC3', 'STAG1',
                'STAG2', 'RAD21', 'PDS5A', 'PDS5B', 'WAPL', 'NIPBL'
            ],
            'apoptosis': [
                'TP53', 'BAX', 'BAK1', 'BCL2', 'BCL2L1', 'CASP3', 'CASP8', 'CASP9',
                'APAF1', 'CYCS', 'BID', 'BAD', 'FAS', 'FASLG', 'TNF', 'TNFRSF1A',
                'BBC3', 'PMAIP1', 'MCL1', 'BIRC5', 'XIAP', 'DIABLO', 'HTRA2',
                'BCL2L2', 'BCL2L11', 'BCL2A1', 'BCL2L10', 'BIK', 'BMF', 'BOK',
                'HRK', 'CASP2', 'CASP6', 'CASP7', 'CASP10', 'CFLAR', 'FADD',
                'TRADD', 'TRAF2', 'RIPK1', 'TNFRSF10A', 'TNFRSF10B', 'TNFSF10',
                'ENDOG', 'AIFM1', 'BCL2L13', 'BCL2L14', 'BNIP1', 'BNIP2',
                'BNIP3', 'BNIP3L', 'CRADD', 'DAPK1', 'DAPK2', 'DAPK3'
            ],
        }
    
    def _get_predefined_programs(self) -> Dict[str, Dict]:
        """15 predefined biological programs with GO IDs. Returns first n_programs."""
        all_programs = {
            'cell_cycle': {
                'go_id': 'GO:0007049',
                'name': 'Cell Cycle',
                'core_genes': ['CCNA2', 'CCNB1', 'CCND1', 'CCNE1', 'CDK1', 'CDK2', 
                              'CDK4', 'CDK6', 'E2F1', 'E2F3', 'MKI67', 'PCNA',
                              'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7',
                              'TOP2A', 'AURKA', 'AURKB', 'PLK1', 'BUB1', 'BUB3',
                              'CCNB2', 'CDC20', 'CDC25A', 'CDC25B', 'CDC25C'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'apoptosis': {
                'go_id': 'GO:0006915',
                'name': 'Apoptotic Process',
                'core_genes': ['TP53', 'BAX', 'BAK1', 'BCL2', 'BCL2L1', 'CASP3',
                              'CASP8', 'CASP9', 'APAF1', 'CYCS', 'BID', 'BAD',
                              'FAS', 'FASLG', 'TNF', 'TNFRSF1A', 'BBC3', 'PMAIP1',
                              'MCL1', 'BIRC5', 'XIAP', 'DIABLO', 'HTRA2'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'immune_response': {
                'go_id': 'GO:0006955',
                'name': 'Immune Response',
                'core_genes': ['IFNG', 'IL2', 'IL6', 'IL10', 'TNF', 'NFKB1', 'NFKB2',
                              'RELA', 'STAT1', 'STAT3', 'JAK1', 'JAK2', 'TLR4',
                              'CD4', 'CD8A', 'CD28', 'CTLA4', 'PDCD1', 'CD274',
                              'IL1B', 'IL12A', 'IL12B', 'IFNA1', 'IFNB1', 'IRF3'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'dna_repair': {
                'go_id': 'GO:0006281',
                'name': 'DNA Repair',
                'core_genes': ['TP53', 'BRCA1', 'BRCA2', 'ATM', 'ATR', 'CHEK1', 'CHEK2',
                              'RAD51', 'XRCC1', 'XRCC4', 'LIG4', 'PARP1', 'MLH1',
                              'MSH2', 'MSH6', 'PMS2', 'ERCC1', 'XPA', 'XPC',
                              'RAD50', 'MRE11A', 'NBN', 'BRIP1', 'PALB2'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'metabolism': {
                'go_id': 'GO:0008152',
                'name': 'Metabolic Process',
                'core_genes': ['HK2', 'PFKM', 'ALDOA', 'PKM', 'LDHA', 'G6PD',
                              'IDH1', 'IDH2', 'ACLY', 'FASN', 'SCD', 'ACACA',
                              'CPT1A', 'PPARG', 'SREBF1', 'MTOR', 'PRKAA1',
                              'PFKL', 'PFKP', 'GAPDH', 'ENO1', 'PGK1', 'LDHB'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'ribosome_biogenesis': {
                'go_id': 'GO:0042254',
                'name': 'Ribosome Biogenesis',
                'core_genes': ['RPL5', 'RPL11', 'RPS6', 'RPS14', 'RPS19', 'MYC',
                              'NPM1', 'NCL', 'FBL', 'NOP56', 'NOP58', 'DKC1',
                              'UTP14A', 'UTP15', 'WDR43', 'POLR1A', 'POLR1B',
                              'RPL3', 'RPL4', 'RPS3', 'RPS7', 'BYSL'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'oxidative_stress': {
                'go_id': 'GO:0006979',
                'name': 'Response to Oxidative Stress',
                'core_genes': ['SOD1', 'SOD2', 'CAT', 'GPX1', 'GPX4', 'PRDX1',
                              'PRDX2', 'NQO1', 'HMOX1', 'NFE2L2', 'KEAP1',
                              'TXNRD1', 'GSR', 'GCLC', 'GCLM', 'PRDX3', 'PRDX5',
                              'GPX2', 'GPX3', 'TXN', 'TXNRD2'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'autophagy': {
                'go_id': 'GO:0006914',
                'name': 'Autophagy',
                'core_genes': ['ATG5', 'ATG7', 'ATG12', 'BECN1', 'MAP1LC3A', 'MAP1LC3B',
                              'SQSTM1', 'ULK1', 'PIK3C3', 'WIPI1', 'LAMP1', 'LAMP2',
                              'TFEB', 'MTOR', 'TSC1', 'TSC2', 'ATG3', 'ATG4B',
                              'ATG13', 'ATG14', 'ATG16L1', 'GABARAP'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'cell_adhesion': {
                'go_id': 'GO:0007155',
                'name': 'Cell Adhesion',
                'core_genes': ['CDH1', 'CDH2', 'ITGA5', 'ITGB1', 'ITGB3', 'VCAM1',
                              'ICAM1', 'PECAM1', 'VIM', 'FN1', 'COL1A1', 'COL1A2',
                              'MMP2', 'MMP9', 'TIMP1', 'TIMP2', 'RAC1', 'CDC42',
                              'RHOA', 'ROCK1', 'ROCK2', 'PTK2', 'SRC', 'PTEN'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'angiogenesis': {
                'go_id': 'GO:0001525',
                'name': 'Angiogenesis',
                'core_genes': ['VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR', 'FLT4',
                              'ANGPT1', 'ANGPT2', 'TEK', 'PDGFA', 'PDGFB', 'PDGFRA',
                              'PDGFRB', 'HIF1A', 'EPAS1', 'NRP1', 'NRP2',
                              'PECAM1', 'CDH5', 'NOTCH1', 'DLL4', 'JAG1'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'hypoxia_response': {
                'go_id': 'GO:0071456',
                'name': 'Cellular Response to Hypoxia',
                'core_genes': ['HIF1A', 'EPAS1', 'ARNT', 'VHL', 'EGLN1', 'EGLN2',
                              'EGLN3', 'HIF1AN', 'VEGFA', 'EPO', 'SLC2A1', 'LDHA',
                              'PGK1', 'ENO1', 'BNIP3', 'BNIP3L', 'PDK1',
                              'CA9', 'NDRG1', 'ADM', 'ANGPTL4', 'LOX'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'er_stress': {
                'go_id': 'GO:0034976',
                'name': 'ER Stress Response',
                'core_genes': ['ATF4', 'ATF6', 'XBP1', 'ERN1', 'EIF2AK3', 'HSPA5',
                              'DDIT3', 'PPP1R15A', 'EDEM1', 'DNAJB9', 'CALR',
                              'PDIA4', 'PDIA6', 'HSP90B1', 'HYOU1', 'DNAJC3',
                              'HERPUD1', 'SEL1L', 'DERL1', 'DERL2'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'inflammatory_response': {
                'go_id': 'GO:0006954',
                'name': 'Inflammatory Response',
                'core_genes': ['IL1B', 'IL6', 'CXCL8', 'TNF', 'CXCL1', 'CXCL2',
                              'CXCL10', 'CCL2', 'CCL5', 'PTGS2', 'NFKB1', 'RELA',
                              'NFKBIA', 'IKBKB', 'TLR2', 'TLR4', 'MYD88',
                              'IRAK1', 'TRAF6', 'IL1R1', 'TNFRSF1A', 'NLRP3'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'emt': {
                'go_id': 'GO:0001837',
                'name': 'Epithelial to Mesenchymal Transition',
                'core_genes': ['SNAI1', 'SNAI2', 'TWIST1', 'TWIST2', 'ZEB1', 'ZEB2',
                              'CDH1', 'CDH2', 'VIM', 'FN1', 'MMP2', 'MMP9',
                              'TGFB1', 'TGFB2', 'TGFBR1', 'TGFBR2', 'SMAD2',
                              'SMAD3', 'SMAD4', 'GSK3B', 'AXL', 'FOXC2'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            },
            'chromatin_remodeling': {
                'go_id': 'GO:0006338',
                'name': 'Chromatin Remodeling',
                'core_genes': ['DNMT1', 'DNMT3A', 'DNMT3B', 'TET1', 'TET2', 'TET3',
                              'HDAC1', 'HDAC2', 'HDAC3', 'KDM1A', 'KDM4A', 'KDM5A',
                              'EZH2', 'SUZ12', 'EED', 'SETD2', 'KMT2A', 'KMT2D',
                              'SMARCA4', 'ARID1A', 'CHD4', 'SMARCA2', 'ARID1B'],
                'target_size': self.rng.integers(self.min_program_size, self.max_program_size + 1)
            }
        }
        
        # Select first n_programs
        program_names = list(all_programs.keys())[:self.n_programs]
        selected_programs = {name: all_programs[name] for name in program_names}
        
        return selected_programs
    
    def _generate_synthetic_genes_with_overlap(self):
        """Generate synthetic genes with controlled overlap between programs."""
        
        # Determine how many synthetic genes each program needs
        program_synthetic_counts = {}
        total_synthetic_needed = 0
        
        for prog_name, prog_info in self.predefined_programs.items():
            # Determine n_synthetic for this program
            if isinstance(self.n_synthetic_per_program, (tuple, list)):
                n_synthetic = self.rng.integers(
                    self.n_synthetic_per_program[0],
                    self.n_synthetic_per_program[1] + 1
                )
            else:
                n_synthetic = self.n_synthetic_per_program
            
            # Add synthetic genes unconditionally (not limited by target_size)
            n_synthetic = max(0, n_synthetic)
            
            program_synthetic_counts[prog_name] = n_synthetic
            total_synthetic_needed += n_synthetic
        
        # Estimate pool sizes
        expected_shared_fraction = self.synthetic_overlap_mean
        estimated_shared_pool_size = int(total_synthetic_needed * expected_shared_fraction * 1.5)
        
        # Create shared synthetic gene pool
        self.shared_synthetic_genes = [
            f"SYN_SHARED_{i:04d}" 
            for i in range(estimated_shared_pool_size)
        ]
        
        # Create unique synthetic genes for each program
        for prog_name in self.predefined_programs.keys():
            n_needed = program_synthetic_counts[prog_name]
            
            # Sample overlap percentage for this program
            overlap_pct = self.rng.normal(
                self.synthetic_overlap_mean,
                self.synthetic_overlap_std
            )
            overlap_pct = np.clip(overlap_pct, 0.0, 0.9)
            
            # Determine split
            n_shared = int(n_needed * overlap_pct)
            n_unique = n_needed - n_shared
            
            # Create unique genes for this program
            unique_genes = [
                f"SYN_{prog_name.upper()}_{i:04d}"
                for i in range(n_unique)
            ]
            
            self.program_unique_synthetic[prog_name] = {
                'genes': unique_genes,
                'n_shared': n_shared,
                'n_unique': n_unique,
                'overlap_pct': overlap_pct
            }
    
    def _expand_programs(self):
        """Expand each program with real GO annotations and optionally synthetic genes."""
        
        # First, get real genes for each program
        for prog_name, prog_info in self.predefined_programs.items():
            go_id = prog_info['go_id']
            
            # Get genes from GO database
            real_genes = list(self.go_to_genes.get(go_id, set()))
            
            if len(real_genes) == 0:
                real_genes = prog_info['core_genes']
            
            # Shuffle real genes
            self.rng.shuffle(real_genes)
            target_size = prog_info['target_size']
            
            # Keep real genes up to target_size
            if len(real_genes) > target_size:
                real_genes = real_genes[:target_size]
            
            prog_info['real_genes'] = real_genes
            prog_info['n_real'] = len(real_genes)
        
        # Generate synthetic genes with overlap if requested
        if self.add_synthetic_program_genes:
            self._generate_synthetic_genes_with_overlap()
            
            # Assign synthetic genes to each program
            for prog_name, prog_info in self.predefined_programs.items():
                real_genes = prog_info['real_genes']
                
                # Get synthetic gene allocation
                synth_info = self.program_unique_synthetic[prog_name]
                n_shared = synth_info['n_shared']
                n_unique = synth_info['n_unique']
                
                # Sample from shared pool
                if n_shared > 0 and len(self.shared_synthetic_genes) > 0:
                    shared_genes = [str(x) for x in self.rng.choice(
                                    self.shared_synthetic_genes,
                                    size=min(n_shared, len(self.shared_synthetic_genes)),
                                    replace=False
                                )]
                else:
                    shared_genes = []
                
                # Get unique genes
                unique_genes = synth_info['genes']
                
                # Combine all genes (real + synthetic)
                all_genes = real_genes + shared_genes + unique_genes
                
                prog_info['genes'] = all_genes
                prog_info['synthetic_shared'] = shared_genes
                prog_info['synthetic_unique'] = unique_genes
                prog_info['n_synthetic'] = len(shared_genes) + len(unique_genes)
                prog_info['n_synthetic_shared'] = len(shared_genes)
                prog_info['n_synthetic_unique'] = len(unique_genes)
                
                # Track gene membership
                for gene in shared_genes:
                    if gene not in self.synthetic_gene_membership:
                        self.synthetic_gene_membership[gene] = []
                    self.synthetic_gene_membership[gene].append(prog_name)
                
                for gene in unique_genes:
                    self.synthetic_gene_membership[gene] = [prog_name]
        else:
            # No synthetic genes
            for prog_name, prog_info in self.predefined_programs.items():
                prog_info['genes'] = prog_info['real_genes']
                prog_info['n_synthetic'] = 0
                prog_info['n_synthetic_shared'] = 0
                prog_info['n_synthetic_unique'] = 0
        
        # Create union of all program genes
        all_program_genes = set()
        for prog_info in self.predefined_programs.values():
            all_program_genes.update(prog_info['genes'])
        
        # Add filler genes to reach total_genes
        if self.add_filler_genes:
            n_additional = self.total_genes - len(all_program_genes)
            
            if n_additional > 0:
                additional_genes = [f"GENE_{i:04d}" for i in range(n_additional)]
            else:
                additional_genes = []
        else:
            additional_genes = []
        
        # Final gene list
        self.all_genes_list = sorted(list(all_program_genes) + additional_genes)
    
    def get_programs(self, k: int = None) -> Dict[str, Dict]:
        """Get k gene programs."""
        programs = self.predefined_programs
        
        if k is not None:
            if k > self.n_programs:
                warnings.warn(f"Requested {k} programs but only {self.n_programs} available. "
                            f"Returning all {self.n_programs} programs.")
                k = self.n_programs
            program_names = list(programs.keys())[:k]
            programs = {name: programs[name] for name in program_names}
        
        return programs
    
    def get_all_genes(self, programs: Dict = None) -> List[str]:
        """Get all genes."""
        return self.all_genes_list
    
    def get_real_genes(self) -> Set[str]:
        """Get all real genes from GO database."""
        real_genes = set()
        for prog_info in self.predefined_programs.values():
            real_genes.update(prog_info.get('real_genes', []))
        return real_genes
    
    def get_synthetic_genes(self) -> Set[str]:
        """Get all synthetic genes."""
        synthetic_genes = set()
        for prog_info in self.predefined_programs.values():
            synthetic_genes.update(prog_info.get('synthetic_shared', []))
            synthetic_genes.update(prog_info.get('synthetic_unique', []))
        return synthetic_genes
    
    def get_shared_synthetic_genes(self) -> Set[str]:
        """Get synthetic genes that belong to multiple programs."""
        return {gene for gene, programs in self.synthetic_gene_membership.items()
                if len(programs) > 1}
    
    def get_synthetic_gene_membership(self) -> Dict[str, List[str]]:
        """Get mapping of synthetic genes to programs they belong to."""
        return self.synthetic_gene_membership.copy()
    
    def get_actual_total_genes(self) -> int:
        """Get actual total number of genes."""
        return len(self.all_genes_list)
    
    def print_summary(self):
        """Print summary of gene programs."""
        print("\n" + "="*80)
        print("GENE PROGRAM SUMMARY")
        print("="*80)
        
        print(f"\nSettings:")
        print(f"  Number of programs: {self.n_programs}")
        print(f"  Synthetic program genes: {'Enabled' if self.add_synthetic_program_genes else 'DISABLED'}")
        if self.add_synthetic_program_genes:
            if isinstance(self.n_synthetic_per_program, (tuple, list)):
                print(f"  Synthetic per program: {self.n_synthetic_per_program[0]}-{self.n_synthetic_per_program[1]}")
            else:
                print(f"  Synthetic per program: {self.n_synthetic_per_program}")
            print(f"  Synthetic overlap mean: {self.synthetic_overlap_mean:.1%}")
            print(f"  Synthetic overlap std: {self.synthetic_overlap_std:.1%}")
        print(f"  Filler genes: {'Enabled' if self.add_filler_genes else 'DISABLED'}")
        print(f"  Target total genes: {self.total_genes}")
        print(f"  Actual total genes: {len(self.all_genes_list)}")
        
        print(f"\nPrograms:")
        
        total_real = 0
        total_synthetic = 0
        total_synthetic_shared = 0
        total_synthetic_unique = 0
        
        for i, (prog_name, prog_info) in enumerate(self.predefined_programs.items(), 1):
            n_genes = len(prog_info['genes'])
            n_real = prog_info.get('n_real', 0)
            n_synthetic = prog_info.get('n_synthetic', 0)
            n_syn_shared = prog_info.get('n_synthetic_shared', 0)
            n_syn_unique = prog_info.get('n_synthetic_unique', 0)
            
            total_real += n_real
            total_synthetic += n_synthetic
            total_synthetic_shared += n_syn_shared
            total_synthetic_unique += n_syn_unique
            
            if self.add_synthetic_program_genes:
                print(f"{i:2d}. {prog_info['name']:45s} "
                      f"{n_genes:3d} genes ({n_real:3d} real, "
                      f"{n_syn_shared:2d} shared syn, {n_syn_unique:2d} unique syn)")
            else:
                print(f"{i:2d}. {prog_info['name']:45s} "
                      f"{n_genes:3d} genes ({n_real:3d} real)")
        
        all_program_genes = set()
        for prog_info in self.predefined_programs.values():
            all_program_genes.update(prog_info['genes'])
        
        n_filler = len(self.all_genes_list) - len(all_program_genes)
        
        print(f"\nSummary:")
        print(f"{'  Total genes in programs (with overlap):':<50s} {total_real + total_synthetic:4d}")
        print(f"{'    - Real genes from GO:':<50s} {total_real:4d}")
        if self.add_synthetic_program_genes:
            print(f"{'    - Synthetic genes (total across programs):':<50s} {total_synthetic:4d}")
            print(f"{'      * Shared synthetic:':<50s} {total_synthetic_shared:4d}")
            print(f"{'      * Unique synthetic:':<50s} {total_synthetic_unique:4d}")
            
            all_shared = self.get_shared_synthetic_genes()
            all_synthetic = self.get_synthetic_genes()
            truly_unique = len(all_synthetic - all_shared)
            
            print(f"{'    - Unique synthetic genes (actual):':<50s} {truly_unique:4d}")
            print(f"{'    - Shared synthetic genes (actual):':<50s} {len(all_shared):4d}")
            if len(all_shared) > 0:
                mean_progs = np.mean([len(progs) for gene, progs in self.synthetic_gene_membership.items() if len(progs) > 1])
                print(f"{'      * Mean programs per shared gene:':<50s} {mean_progs:.2f}")
        
        print(f"{'  Total unique genes in programs:':<50s} {len(all_program_genes):4d}")
        print(f"{'  Filler genes (GENE_XXXX):':<50s} {n_filler:4d}")
        print(f"{'  TOTAL GENES:':<50s} {len(self.all_genes_list):4d}")
        print("="*80)
    
    def print_synthetic_overlap_analysis(self):
        """Print detailed analysis of synthetic gene overlap."""
        if not self.add_synthetic_program_genes:
            print("No synthetic genes - overlap analysis not applicable.")
            return
        
        print("\n" + "="*80)
        print("SYNTHETIC GENE OVERLAP ANALYSIS")
        print("="*80)
        
        membership_counts = defaultdict(int)
        for gene, programs in self.synthetic_gene_membership.items():
            membership_counts[len(programs)] += 1
        
        print("\nGenes by program membership:")
        for n_programs in sorted(membership_counts.keys()):
            count = membership_counts[n_programs]
            if n_programs == 1:
                print(f"  {n_programs} program (unique):  {count:4d} genes")
            else:
                print(f"  {n_programs} programs (shared): {count:4d} genes")
        
        shared_genes = [(gene, programs) for gene, programs in self.synthetic_gene_membership.items()
                       if len(programs) > 1]
        
        if len(shared_genes) > 0:
            shared_genes.sort(key=lambda x: len(x[1]), reverse=True)
            
            print(f"\nTop 10 most shared synthetic genes:")
            for gene, programs in shared_genes[:10]:
                print(f"  {gene:30s}: {len(programs)} programs - {', '.join(programs[:3])}"
                      f"{'...' if len(programs) > 3 else ''}")
        
        prog_names = list(self.predefined_programs.keys())
        overlap_matrix = np.zeros((len(prog_names), len(prog_names)), dtype=int)
        
        for i, prog1 in enumerate(prog_names):
            genes1 = set(self.predefined_programs[prog1].get('synthetic_shared', []) +
                        self.predefined_programs[prog1].get('synthetic_unique', []))
            for j, prog2 in enumerate(prog_names):
                if i <= j:
                    genes2 = set(self.predefined_programs[prog2].get('synthetic_shared', []) +
                               self.predefined_programs[prog2].get('synthetic_unique', []))
                    overlap = len(genes1 & genes2)
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
        
        overlaps = []
        for i in range(len(prog_names)):
            for j in range(i+1, len(prog_names)):
                if overlap_matrix[i, j] > 0:
                    overlaps.append((prog_names[i], prog_names[j], overlap_matrix[i, j]))
        
        overlaps.sort(key=lambda x: x[2], reverse=True)
        
        n_to_show = min(15, len(overlaps))
        print(f"\nTop {n_to_show} program pairs by shared synthetic genes:")
        for prog1, prog2, count in overlaps[:n_to_show]:
            print(f"  {prog1:25s} <-> {prog2:25s}: {count:3d} shared genes")
        
        print("="*80)