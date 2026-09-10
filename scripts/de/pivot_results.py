import glob
import pandas as pd

chunks = sorted(glob.glob("results/chunk_*.csv"))
df = pd.concat([pd.read_csv(f) for f in chunks], ignore_index=True)

matrix = df.pivot(index="perturbation", columns="gene", values="mean_diff")
matrix.to_csv("results/mean_diff_matrix.csv")
print(f"Combined {len(chunks)} files: {df.shape[0]} rows")
print(f"Matrix shape: {matrix.shape}")
