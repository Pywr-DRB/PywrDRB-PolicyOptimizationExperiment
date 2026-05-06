#!/usr/bin/env python3
"""
Summarize Borg optimization outputs: objectives used, total solution counts,
and per-reservoir/per-policy counts before and after OBJ_FILTER_BOUNDS filtering.

Respects the same environment as figures (methods.borg_paths.resolve_borg_moea_csv_path):
  CEE_BORG_SEED / CEE_SEED, CEE_BORG_MRF_FILTERED (or deprecated CEE_BORG_MRFMASKED),
  CEE_MRF_FILTER_TAG / CEE_MRF_FILTER_SOURCE (or deprecated CEE_MRF_MASK_*).
  On-disk Borg CSVs use ``*_mrffiltered_regression.csv`` / ``*_mrffiltered_perfect.csv``; ``CEE_MRF_FILTER_TAG`` may be
  user-facing names such as ``regression_disagg`` or ``perfect`` (see ``methods.borg_paths``). Legacy ``*_mrfmasked_*`` is still found.

Usage (from CEE6400Project/):
  CEE_BORG_SEED=72 CEE_BORG_MRF_FILTERED=0 python -m methods.postprocess.summarize_optimization
  CEE_BORG_SEED=71 CEE_BORG_MRF_FILTERED=1 CEE_MRF_FILTER_TAG=regression_disagg python -m methods.postprocess.summarize_optimization -o outputs/summary_regression_disagg.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.config import (
    OBJ_LABELS,
    OBJ_FILTER_BOUNDS,
    RELEASE_METRICS,
    STORAGE_METRICS,
    EPSILONS,
    NFE,
    ISLANDS,
    reservoir_options,
    policy_type_options,
    OUTPUT_DIR,
)
from methods.borg_paths import resolve_borg_moea_csv_path
from methods.load.results import load_results


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Borg CSV solution counts before/after filtering.")
    ap.add_argument(
        "-o",
        "--out-csv",
        default=None,
        help="Write per reservoir/policy summary CSV (optional).",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="Write full summary JSON including metadata (optional).",
    )
    ap.add_argument("--reservoirs", nargs="*", default=list(reservoir_options))
    ap.add_argument("--policies", nargs="*", default=list(policy_type_options))
    args = ap.parse_args()

    _obj_keys = sorted(OBJ_LABELS.keys(), key=lambda k: int("".join(filter(str.isdigit, k)) or 0))
    objectives_order = [OBJ_LABELS[k] for k in _obj_keys]
    meta = {
        "release_metrics": list(RELEASE_METRICS),
        "storage_metrics": list(STORAGE_METRICS),
        "objectives_in_borg_order": objectives_order,
        "objective_labels_for_figures": objectives_order,
        "epsilons": list(EPSILONS),
        "nfe": NFE,
        "islands": ISLANDS,
        "filter_bounds": {k: list(v) for k, v in OBJ_FILTER_BOUNDS.items()},
        "output_dir": OUTPUT_DIR,
        "env": {
            "CEE_BORG_SEED": os.environ.get("CEE_BORG_SEED", ""),
            "CEE_SEED": os.environ.get("CEE_SEED", ""),
            "CEE_BORG_MRF_FILTERED": os.environ.get("CEE_BORG_MRF_FILTERED", ""),
            "CEE_BORG_MRFMASKED": os.environ.get("CEE_BORG_MRFMASKED", ""),
            "CEE_MRF_FILTER_TAG": os.environ.get("CEE_MRF_FILTER_TAG", ""),
            "CEE_MRF_FILTER_SOURCE": os.environ.get("CEE_MRF_FILTER_SOURCE", ""),
            "CEE_MRF_MASK_TAG": os.environ.get("CEE_MRF_MASK_TAG", ""),
            "CEE_MRF_MASK_SOURCE": os.environ.get("CEE_MRF_MASK_SOURCE", ""),
        },
    }

    rows = []
    total_before = 0
    total_after = 0

    for res in args.reservoirs:
        for pol in args.policies:
            path = resolve_borg_moea_csv_path(pol, res)
            p = Path(path)
            if not p.is_file():
                rows.append(
                    {
                        "reservoir": res,
                        "policy": pol,
                        "csv_path": path,
                        "exists": False,
                        "n_before_filter": 0,
                        "n_after_filter": 0,
                    }
                )
                continue
            try:
                obj_all, _ = load_results(
                    path, obj_labels=OBJ_LABELS, filter=False, obj_bounds=OBJ_FILTER_BOUNDS
                )
                obj_filt, _ = load_results(
                    path, obj_labels=OBJ_LABELS, filter=True, obj_bounds=OBJ_FILTER_BOUNDS
                )
            except Exception as e:
                rows.append(
                    {
                        "reservoir": res,
                        "policy": pol,
                        "csv_path": path,
                        "exists": True,
                        "error": str(e),
                        "n_before_filter": 0,
                        "n_after_filter": 0,
                    }
                )
                continue

            nb = len(obj_all)
            na = len(obj_filt)
            total_before += nb
            total_after += na
            rows.append(
                {
                    "reservoir": res,
                    "policy": pol,
                    "csv_path": path,
                    "exists": True,
                    "n_before_filter": nb,
                    "n_after_filter": na,
                }
            )

    df = pd.DataFrame(rows)

    print("=== Optimization summary ===")
    print(f"Objectives (minimize): {meta['objectives_in_borg_order']}")
    print(f"NFE={NFE}, islands={ISLANDS}, epsilons={EPSILONS}")
    print(f"Filter bounds (pretty names): {json.dumps(meta['filter_bounds'], indent=2)}")
    print()
    print(f"Total solutions (sum over existing CSVs): {total_before} before filter → {total_after} after filter")
    print()
    print(df.to_string(index=False))

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"\nWrote {outp}")

    if args.out_json:
        outj = Path(args.out_json)
        outj.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": meta, "rows": rows, "totals": {"before_filter": total_before, "after_filter": total_after}}
        outj.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {outj}")


if __name__ == "__main__":
    main()
