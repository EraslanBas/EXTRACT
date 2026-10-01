#!/usr/bin/env python3
"""The single test read -- the only place test metrics are printed.

Selection is done on validation (``summarise_sweep.py`` -> ``best_configs.csv``,
which carries no test columns). Once a configuration is chosen, run this for
it. Test numbers were recorded for every cell by the sweep, at the epoch
validation selected, so nothing is refitted.

Every read is appended to ``<sweep>/TEST_READS.log`` with a timestamp. Test is
meant to be read once; the log makes a second read visible rather than silent,
and this script refuses a repeat unless ``--again`` is given, so a re-read is a
decision and not an accident. A test number obtained after choosing among
configs by their test scores is no longer an unbiased estimate.

    python scripts/read_test.py <sweep_dir> --tag d4_a1000_b0_K1_seed0
"""
from __future__ import annotations
import argparse, datetime, getpass
from pathlib import Path
import pandas as pd

SHOW = ["disc", "recon_measured", "recon_synth", "accuracy",
        "synth_accuracy", "span_residual_median"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--tag", nargs="+", required=True,
                    help="the configuration(s) chosen ON VALIDATION")
    ap.add_argument("--reason", default="",
                    help="why these configs -- recorded in the log")
    ap.add_argument("--again", action="store_true",
                    help="allow a read when test has been read before")
    args = ap.parse_args()

    log = args.sweep_dir / "TEST_READS.log"
    prior = log.read_text().strip().splitlines() if log.exists() else []
    if prior and not args.again:
        print(f"test has already been read {len(prior)} time(s):")
        for line in prior:
            print(f"  {line}")
        raise SystemExit("refusing a repeat read; pass --again if this is deliberate")

    m = pd.read_csv(args.sweep_dir / "sweep_metrics.csv")
    rows = m[m.tag.isin(args.tag)]
    missing = sorted(set(args.tag) - set(rows.tag))
    if missing:
        raise SystemExit(f"not in sweep_metrics.csv: {missing}")
    rows = rows.drop_duplicates("tag", keep="last")

    cols = ["tag"]
    for k in SHOW:
        for sp in ("val", "test"):
            c = f"{sp}_{k}"
            if c in rows:
                cols.append(c)
    print(rows[cols].to_string(index=False))

    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open(log, "a") as fh:
        fh.write(f"{stamp}\t{getpass.getuser()}\t{','.join(args.tag)}\t{args.reason}\n")
    print(f"\nlogged to {log}  (read #{len(prior) + 1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
