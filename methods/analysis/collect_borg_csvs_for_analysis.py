#!/usr/bin/env python3
"""
Stack all MMBorg result CSVs under outputs/ into one long table for downstream analysis.

Adds columns: source_file, policy, reservoir, nfe, seed, island, mrf_filtered (bool).

Example:

  python -m methods.analysis.collect_borg_csvs_for_analysis -o outputs/analysis_all_borg_runs.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# Final aggregated CSV from Borg (not per-island runtime)
CSV_RE = re.compile(
    r"^MMBorg_(?P<islands>\d+)M_(?P<policy>STARFIT|RBF|PWL)_(?P<res>.+)_nfe(?P<nfe>\d+)"
    r"_seed(?P<seed>\d+)(?P<mrf>(?:_mrffiltered|_mrfmasked)(?:_perfect|_regression)?)?\.csv$"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing MMBorg_*.csv (default: ./outputs)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("outputs/analysis_all_borg_runs.csv"),
        help="Output CSV path",
    )
    args = ap.parse_args()

    root = args.outputs_dir
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("MMBorg_*.csv")):
        m = CSV_RE.match(path.name)
        if not m:
            continue
        df = pd.read_csv(path)
        df.insert(0, "source_file", path.name)
        df.insert(1, "policy", m.group("policy"))
        df.insert(2, "reservoir", m.group("res"))
        df.insert(3, "nfe", int(m.group("nfe")))
        df.insert(4, "seed", int(m.group("seed")))
        mrf_g = m.group("mrf")
        df.insert(5, "mrf_filtered", bool(mrf_g))
        if mrf_g and "_perfect" in mrf_g:
            variant = "perfect"
        elif mrf_g and "_regression" in mrf_g:
            variant = "regression"
        elif mrf_g:
            variant = "legacy_mrf_token"
        else:
            variant = "none"
        df.insert(6, "mrf_filter_variant", variant)
        frames.append(df)

    if not frames:
        raise SystemExit(f"No matching MMBorg_*.csv files under {root}")

    out = pd.concat(frames, axis=0, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows from {len(frames)} files -> {args.out}")


if __name__ == "__main__":
    main()
