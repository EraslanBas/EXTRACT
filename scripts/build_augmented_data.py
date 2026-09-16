#!/usr/bin/env python3
"""CLI for module_finder.data.build_augmented.

Writes both kinds of augmented data next to the real matrices:
label permutations (small, pooled) and shuffled matrices (large, per context).

    python scripts/build_augmented_data.py \
        --matrices /large_storage/ctc/<user>/ModuleFinder/matrices \
        --out-dir  /large_storage/ctc/<user>/ModuleFinder/augmented
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from module_finder.data import STRATEGIES, build_augmented


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrices", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--contexts", nargs="+", default=None)
    ap.add_argument("--label-strategies", nargs="+", default=list(STRATEGIES),
                    choices=list(STRATEGIES))
    ap.add_argument("--label-seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--shuffle-seeds", nargs="+", type=int, default=[0],
                    help="one ~1.8 GB file per context per seed")
    ap.add_argument("--no-shuffled", action="store_true",
                    help="labels only; skip the large shuffled matrices")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    manifest = build_augmented(
        args.matrices, args.out_dir,
        label_strategies=tuple(args.label_strategies),
        label_seeds=tuple(args.label_seeds),
        shuffle_seeds=() if args.no_shuffled else tuple(args.shuffle_seeds),
        contexts=args.contexts,
        overwrite=args.overwrite,
    )
    if len(manifest):
        print("\n" + manifest.to_string(index=False))
        print(f"\n{len(manifest)} files, {manifest.MB.sum()/1000:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
