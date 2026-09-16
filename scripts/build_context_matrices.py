#!/usr/bin/env python3
"""CLI for module_finder.de.build_all_matrices.

Assembles per-context (label x gene) matrices of ashr-shrunken logFC. All the
logic lives in the package; this is only argument parsing.

    python scripts/build_context_matrices.py \
        --computese /large_storage/ctc/<user>/ModuleFinder/computese \
        --out-dir   /large_storage/ctc/<user>/ModuleFinder/matrices
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from module_finder.de import build_all_matrices


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--computese", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--contexts", nargs="+", default=None)
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    summary = build_all_matrices(args.computese, args.out_dir,
                                 contexts=args.contexts, fmt=args.format,
                                 overwrite=args.overwrite)
    if len(summary):
        print("\n" + summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
