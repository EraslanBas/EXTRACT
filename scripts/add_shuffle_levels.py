#!/usr/bin/env python3
"""Add partially shuffled synthetic rows to an existing split.

    python scripts/add_shuffle_levels.py --split-dir $EXTRACT_ROOT/splits/pair_seed0_ctxshuffle

Writes ``<ctx>_shuffled_frac<f>_seed<k>.parquet`` next to the fully shuffled
``<ctx>_shuffled_seed<k>.parquet`` in both partitions: the same context-wide
shuffle before the split, but permuting only a fraction ``f`` of the gene
columns, so the rest of each synthetic row keeps its real values. Real rows,
test pairs and the gene list are untouched. Load them with
``load_partition(..., shuffle_levels=SHUFFLE_LEVELS)`` or
``sweep_grid.py --shuffle-levels 0.2 0.4 0.6 0.8 1.0``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract import paths
from extract.data.splits import add_shuffle_levels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-dir", type=Path,
                   default=paths.root() / "splits" / "pair_seed0_ctxshuffle")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--matrices", type=Path, default=None,
                   help="default: the source_matrices recorded in the manifest")
    args = p.parse_args()
    man = add_shuffle_levels(args.split_dir, tuple(args.fracs), matrices_dir=args.matrices)
    print(json.dumps({"shuffle_levels": man["shuffle_levels"]}))


if __name__ == "__main__":
    main()
