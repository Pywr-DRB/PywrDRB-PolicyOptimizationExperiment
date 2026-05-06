#!/usr/bin/env python3
"""
build_mrf_active_masks.py

Preprocessing script to:
1. Run PywrDRB simulation with default policy (unless --skip-simulation + --existing-output)
2. Extract lower_basin_mrf_contributions DataFrame from the HDF5
3. Export to CSV
4. Build MRF active date ranges JSON for masking optimization objectives

`build_mrf_masking_folder.sh` uses existing pub HDF5 when present (else runs Pywr); the
perfect_information bundle is always read from HDF5 only (never simulated there).

Usage:
    python -m methods.preprocessing.build_mrf_active_masks \
        --output-csv lower_basin_mrf_contributions.csv \
        --output-json lower_basin_mrf_active_ranges.json \
        --epsilon 0.1 \
        --csv-output mrf_active_mask_daily.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Canonical folder for Pywr JSON/HDF5 (avoid cluttering project root)
PYWR_DATA_DIR = Path(os.environ.get("DRB_OUTPUT_DIR", PROJECT_ROOT / "pywr_data"))
DEFAULT_MRF_PUB_SIM_DIR = PYWR_DATA_DIR / "_mrf_pub_sim"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add Pywr-DRB to path
PYWR_DRB_REPO = PROJECT_ROOT.parent / "Release_Policy_DRB"
PYWR_SRC = PYWR_DRB_REPO / "src"
if str(PYWR_SRC) not in sys.path:
    sys.path.insert(0, str(PYWR_SRC))

import pywrdrb
from methods.preprocessing.mrf_masking import build_lower_basin_mrf_active_dict


def run_pywrdrb_simulation(
    inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    start_date: str = "1983-10-01",
    end_date: str = "2023-12-31",
    work_dir: Path | None = None,
) -> Tuple[str, str]:
    """
    Run PywrDRB simulation and return model and output filenames.
    """
    if work_dir is None:
        work_dir = DEFAULT_MRF_PUB_SIM_DIR
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning PywrDRB simulation for inflow type: {inflow_type}")
    print(f"  Start date: {start_date}")
    print(f"  End date: {end_date}")
    print(f"  Work directory: {work_dir}")

    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date=start_date,
        end_date=end_date,
    )
    mb.make_model()

    model_filename = str(work_dir / f"model_{inflow_type}.json")
    output_filename = str(work_dir / f"pywrdrb_output_{inflow_type}.hdf5")

    mb.write_model(model_filename)

    model = pywrdrb.Model.load(model_filename)
    pywrdrb.OutputRecorder(model, output_filename)
    model.run()

    assert os.path.exists(output_filename), f"Simulation output not found: {output_filename}"
    print(f"  ✓ Simulation completed: {output_filename}")

    return model_filename, output_filename


def extract_mrf_contributions(
    output_filename: str,
) -> pd.DataFrame:
    """
    Extract lower_basin_mrf_contributions DataFrame from PywrDRB output.
    """
    print(f"\nExtracting MRF contributions from: {output_filename}")

    data = pywrdrb.Data(
        print_status=True,
        results_sets=["lower_basin_mrf_contributions"],
        output_filenames=[output_filename],
    )
    data.load_output()

    key = Path(output_filename).stem
    df_mrf = data.lower_basin_mrf_contributions[key][0]

    if not isinstance(df_mrf.index, pd.DatetimeIndex):
        df_mrf.index = pd.to_datetime(df_mrf.index)

    print(f"  ✓ Extracted {len(df_mrf)} rows")
    print(f"  ✓ Columns: {list(df_mrf.columns)}")
    print(f"  ✓ Date range: {df_mrf.index.min()} to {df_mrf.index.max()}")
    return df_mrf


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MRF active masks from PywrDRB simulation")
    parser.add_argument("--inflow-type", default="pub_nhmv10_BC_withObsScaled", help="Inflow type for PywrDRB simulation")
    parser.add_argument("--start-date", default="1983-10-01", help="Simulation start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2023-12-31", help="Simulation end date (YYYY-MM-DD)")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for model_*.json and pywrdrb_output_*.hdf5 when running Pywr. Default: <project>/pywr_data/_mrf_pub_sim.",
    )
    parser.add_argument("--output-csv", type=Path, default="lower_basin_mrf_contributions.csv", help="Output CSV path for MRF contributions DataFrame")
    parser.add_argument("--output-json", type=Path, default="lower_basin_mrf_active_ranges.json", help="Output JSON path for MRF active date ranges")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Threshold (MGD) for considering MRF contribution 'active'")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional: also export daily mask CSV for debugging")
    parser.add_argument("--skip-simulation", action="store_true", help="Skip simulation and use existing output file (must provide --existing-output)")
    parser.add_argument("--existing-output", type=Path, default=None, help="Path to existing PywrDRB output HDF5 file (if --skip-simulation)")
    parser.add_argument("--plot-output", type=Path, default=None, help="Optional: path to save diagnostic stacked area plot")
    parser.add_argument("--plot-start", default=None, help="Optional: start date for plot window (YYYY-MM-DD)")
    parser.add_argument("--plot-end", default=None, help="Optional: end date for plot window (YYYY-MM-DD)")
    parser.add_argument(
        "--mask-bundle",
        choices=("pub_reconstruction", "perfect_information"),
        default=None,
        help="Write standard filenames under preprocessing_outputs/masking/<bundle>/",
    )
    args = parser.parse_args()

    if args.work_dir is None:
        args.work_dir = DEFAULT_MRF_PUB_SIM_DIR

    if args.mask_bundle:
        bundle_dir = PROJECT_ROOT / "preprocessing_outputs" / "masking" / args.mask_bundle
        bundle_dir.mkdir(parents=True, exist_ok=True)
        args.output_csv = bundle_dir / "lower_basin_mrf_contributions.csv"
        args.output_json = bundle_dir / "lower_basin_mrf_active_ranges.json"
        if args.csv_output is None:
            args.csv_output = bundle_dir / "mrf_active_mask_daily.csv"

    if args.skip_simulation:
        if args.existing_output is None:
            parser.error("--existing-output required when --skip-simulation is used")
        output_filename = str(args.existing_output)
        print(f"\nUsing existing output: {output_filename}")
    else:
        _, output_filename = run_pywrdrb_simulation(
            inflow_type=args.inflow_type,
            start_date=args.start_date,
            end_date=args.end_date,
            work_dir=args.work_dir,
        )

    df_mrf = extract_mrf_contributions(output_filename=output_filename)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_mrf.to_csv(output_csv)
    print(f"\n✓ Exported MRF contributions to: {output_csv}")

    print(f"\nBuilding MRF active date ranges (epsilon={args.epsilon})")
    reservoirs = ["beltzvilleCombined", "blueMarsh", "nockamixon"]
    prefix = "mrf_trenton_"

    expected_cols = [f"{prefix}{r}" for r in reservoirs]
    missing_cols = [c for c in expected_cols if c not in df_mrf.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")
        print(f"  Available columns: {list(df_mrf.columns)}")
        available_res = [c.replace(prefix, "") for c in df_mrf.columns if c.startswith(prefix)]
        if available_res:
            print(f"  Using available reservoirs: {available_res}")
            reservoirs = available_res

    mrf_ranges_dict, any_mask = build_lower_basin_mrf_active_dict(
        df=df_mrf,
        eps=args.epsilon,
        reservoirs=reservoirs,
        prefix=prefix,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    json_dict = {
        key: [{"start": str(r["start"]), "end": str(r["end"]), "days": r["days"]} for r in ranges]
        for key, ranges in mrf_ranges_dict.items()
    }
    with open(output_json, "w") as f:
        json.dump(json_dict, f, indent=2)
    print(f"✓ Exported MRF active ranges to: {output_json}")

    if args.csv_output:
        mask_df = pd.DataFrame(index=df_mrf.index)
        for key, ranges in mrf_ranges_dict.items():
            mask = pd.Series(False, index=df_mrf.index)
            for r in ranges:
                mask.loc[r["start"] : r["end"]] = True
            mask_df[key] = mask.astype(int)
        mask_df["ANY_lower_basin"] = any_mask.astype(int)
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        mask_df.to_csv(csv_output)
        print(f"\n✓ Exported daily mask CSV to: {csv_output}")

    if args.plot_output:
        print("\nGenerating diagnostic plot...")
        df_plot = df_mrf.copy()
        if args.plot_start or args.plot_end:
            df_plot = df_plot.loc[slice(args.plot_start, args.plot_end)]
            print(f"  Plotting date range: {df_plot.index.min()} to {df_plot.index.max()}")
        else:
            print(f"  Plotting full date range: {df_plot.index.min()} to {df_plot.index.max()}")

        fig, ax = plt.subplots(figsize=(15, 6))
        cols = [f"{prefix}{r}" for r in reservoirs if f"{prefix}{r}" in df_plot.columns]
        label_map = {
            "beltzvilleCombined": "Beltzville Combined",
            "blueMarsh": "Blue Marsh",
            "nockamixon": "Nockamixon",
        }
        labels = [label_map.get(r, r) for r in reservoirs if f"{prefix}{r}" in df_plot.columns]

        if cols:
            ax.stackplot(df_plot.index, *[df_plot[col] for col in cols], labels=labels, alpha=0.8)
            ax.set_title(
                "Lower Basin Reservoir Contributions to Trenton MRF\n"
                f"({df_plot.index.min().strftime('%Y-%m-%d')} to {df_plot.index.max().strftime('%Y-%m-%d')})"
            )
            ax.set_ylabel("Contribution to MRF (MGD)")
            ax.set_xlabel("Date")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plot_path = Path(args.plot_output)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"✓ Exported diagnostic plot to: {plot_path}")
        else:
            print("  WARNING: No MRF contribution columns found for plotting")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
