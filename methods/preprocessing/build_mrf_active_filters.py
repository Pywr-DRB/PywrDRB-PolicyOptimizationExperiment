#!/usr/bin/env python3
"""
build_mrf_active_filters.py

Preprocessing script to:
1. Run PywrDRB simulation with default policy (unless --skip-simulation + --existing-output)
2. Extract lower_basin_mrf_contributions from HDF5 using pywrdrb.Data loader
3. Export contributions and daily filter CSV
4. Optionally export compact MRF date-range JSON (not required by optimization)

`build_mrf_filtering_folder.sh` uses existing pub HDF5 when present (else runs Pywr); the
mode bundles are organized by optimization flow mode:
- regression_disagg
- perfect_foresight

Usage:
    python -m methods.preprocessing.build_mrf_active_filters \
        --output-csv lower_basin_mrf_contributions.csv \
        --epsilon 0.1 \
        --csv-output mrf_active_filter_daily.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

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
from methods.config import MRF_NORMAL_OPS_BUFFER_DAYS
from methods.preprocessing.mrf_filtering import (
    build_lower_basin_mrf_active_dict,
    build_normal_ops_filter,
    plot_mrf_contributions_active_periods,
    plot_mrf_filtering_for_optimization,
    summarize_filter_fractions,
    write_filter_summary_txt,
)


def run_pywrdrb_simulation(
    inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    flow_prediction_mode: str = "regression_disagg",
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
    print(f"  Flow prediction mode: {flow_prediction_mode}")
    print(f"  Start date: {start_date}")
    print(f"  End date: {end_date}")
    print(f"  Work directory: {work_dir}")

    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date=start_date,
        end_date=end_date,
        options={"flow_prediction_mode": flow_prediction_mode},
    )
    mb.make_model()

    model_filename = str(work_dir / f"model_{inflow_type}_{flow_prediction_mode}.json")
    output_filename = str(work_dir / f"pywrdrb_output_{inflow_type}_{flow_prediction_mode}.hdf5")

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
    parser = argparse.ArgumentParser(description="Build MRF active filters from PywrDRB simulation")
    parser.add_argument("--inflow-type", default="pub_nhmv10_BC_withObsScaled", help="Inflow type for PywrDRB simulation")
    parser.add_argument(
        "--flow-prediction-mode",
        choices=("regression_disagg", "perfect_foresight", "gage_flow"),
        default="regression_disagg",
        help="Pywr-DRB flow prediction mode used by ModelBuilder when simulation is run.",
    )
    parser.add_argument("--start-date", default="1983-10-01", help="Simulation start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2023-12-31", help="Simulation end date (YYYY-MM-DD)")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for model_*.json and pywrdrb_output_*.hdf5 when running Pywr. Default: <project>/pywr_data/_mrf_pub_sim.",
    )
    parser.add_argument("--output-csv", type=Path, default="lower_basin_mrf_contributions.csv", help="Output CSV path for MRF contributions DataFrame")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path for MRF active date ranges (not required by optimization).")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Threshold (MGD) for considering MRF contribution 'active'")
    parser.add_argument(
        "--buffer-days",
        type=int,
        default=None,
        help=f"Days before/after each active range excluded from objectives (default: MRF_NORMAL_OPS_BUFFER_DAYS={MRF_NORMAL_OPS_BUFFER_DAYS})",
    )
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional: also export daily filter CSV for debugging")
    parser.add_argument("--skip-simulation", action="store_true", help="Skip simulation and use existing output file (must provide --existing-output)")
    parser.add_argument("--existing-output", type=Path, default=None, help="Path to existing PywrDRB output HDF5 file (if --skip-simulation)")
    parser.add_argument("--plot-output", type=Path, default=None, help="Optional: path to save diagnostic stacked area plot")
    parser.add_argument("--plot-start", default=None, help="Optional: start date for plot window (YYYY-MM-DD)")
    parser.add_argument("--plot-end", default=None, help="Optional: end date for plot window (YYYY-MM-DD)")
    parser.add_argument(
        "--filter-bundle",
        choices=("regression_disagg", "perfect_foresight"),
        default=None,
        help="Write standard filenames under preprocessing_outputs/filtering/<bundle>/",
    )
    args = parser.parse_args()
    buffer_days = (
        int(args.buffer_days)
        if args.buffer_days is not None
        else int(MRF_NORMAL_OPS_BUFFER_DAYS)
    )

    if args.work_dir is None:
        args.work_dir = DEFAULT_MRF_PUB_SIM_DIR

    if args.filter_bundle:
        bundle_dir = PROJECT_ROOT / "preprocessing_outputs" / "filtering" / args.filter_bundle
        bundle_dir.mkdir(parents=True, exist_ok=True)
        args.output_csv = bundle_dir / "lower_basin_mrf_contributions.csv"
        if args.csv_output is None:
            args.csv_output = bundle_dir / "mrf_active_filter_daily.csv"
        if args.plot_output is None:
            args.plot_output = bundle_dir / "lower_basin_mrf_contributions_active_periods.png"

    if args.skip_simulation:
        if args.existing_output is None:
            parser.error("--existing-output required when --skip-simulation is used")
        output_filename = str(args.existing_output)
        print(f"\nUsing existing output: {output_filename}")
    else:
        _, output_filename = run_pywrdrb_simulation(
            inflow_type=args.inflow_type,
            flow_prediction_mode=args.flow_prediction_mode,
            start_date=args.start_date,
            end_date=args.end_date,
            work_dir=args.work_dir,
        )

    df_mrf = extract_mrf_contributions(output_filename=output_filename)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_mrf.to_csv(output_csv)
    print(f"\n✓ Exported MRF contributions to: {output_csv}")

    print(f"\nBuilding MRF active filters (epsilon={args.epsilon})")
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

    mrf_ranges_dict, any_filter = build_lower_basin_mrf_active_dict(
        df=df_mrf,
        eps=args.epsilon,
        reservoirs=reservoirs,
        prefix=prefix,
    )

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        json_dict = {
            key: [{"start": str(r["start"]), "end": str(r["end"]), "days": r["days"]} for r in ranges]
            for key, ranges in mrf_ranges_dict.items()
        }
        with open(output_json, "w") as f:
            json.dump(json_dict, f, indent=2)
        print(f"✓ Exported MRF active ranges to: {output_json}")

    if args.filter_bundle:
        trace_path = bundle_dir / "traceability.txt"
        trace_path.write_text(
            "\n".join(
                [
                    f"bundle={args.filter_bundle}",
                    f"inflow_type={args.inflow_type}",
                    f"flow_prediction_mode={args.flow_prediction_mode}",
                    f"source_hdf5_path_used_for_extraction_build={output_filename}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"✓ Wrote traceability metadata: {trace_path}")

    normal_ops = build_normal_ops_filter(
        df_mrf.index,
        mrf_ranges_dict,
        mode="ANY",
        buffer_days=buffer_days,
    )
    raw_active = any_filter.values.astype(bool)
    stats_full = summarize_filter_fractions(df_mrf.index, raw_active, normal_ops)

    training_stats = None
    training_index = None
    try:
        from methods.load.observations import get_observational_training_data
        from methods.config import PROCESSED_DATA_DIR

        inflow_df, _, _ = get_observational_training_data(
            reservoir_name="blueMarsh",
            data_dir=PROCESSED_DATA_DIR,
            as_numpy=False,
            inflow_type="inflow_pub",
        )
        training_index = inflow_df.index
        tr_raw = any_filter.reindex(training_index).fillna(False).astype(bool).values
        tr_normal = build_normal_ops_filter(
            training_index,
            mrf_ranges_dict,
            mode="ANY",
            buffer_days=buffer_days,
        )
        training_stats = summarize_filter_fractions(training_index, tr_raw, tr_normal)
    except Exception as exc:
        print(f"  Note: could not compute Borg training overlap stats: {exc}")

    if args.csv_output:
        filter_df = pd.DataFrame(index=df_mrf.index)
        for key, ranges in mrf_ranges_dict.items():
            active_filter = pd.Series(False, index=df_mrf.index)
            for r in ranges:
                active_filter.loc[r["start"] : r["end"]] = True
            filter_df[key] = active_filter.astype(int)
        filter_df["ANY_lower_basin"] = any_filter.astype(int)
        filter_df["normal_ops_for_objectives"] = normal_ops.astype(int)
        filter_df["excluded_from_objectives"] = (~normal_ops).astype(int)
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        filter_df.to_csv(csv_output)
        print(f"\n✓ Exported daily filter CSV to: {csv_output}")

    summary_dir = Path(args.output_csv).parent
    write_filter_summary_txt(
        summary_dir / "mrf_filtering_summary.txt",
        bundle_label=args.filter_bundle or "custom",
        eps_mgd=args.epsilon,
        buffer_days=buffer_days,
        stats=stats_full,
        training_index=training_index,
        training_stats=training_stats,
    )
    print(open(summary_dir / "mrf_filtering_summary.txt").read())

    if args.plot_output:
        bundle_title = (args.filter_bundle or args.flow_prediction_mode).replace("_", " ")
        plot_mrf_contributions_active_periods(
            df_mrf,
            args.plot_output,
            eps_mgd=args.epsilon,
            title=f"{bundle_title}: lower-basin MRF (raw active > {args.epsilon} MGD)",
        )
        print(f"✓ Exported active-period plot: {args.plot_output}")
        opt_plot = summary_dir / "mrf_filtering_for_optimization.png"
        plot_mrf_filtering_for_optimization(
            df_mrf,
            raw_active,
            normal_ops,
            opt_plot,
            buffer_days=buffer_days,
            eps_mgd=args.epsilon,
            title=f"{bundle_title}: MRF filtering for Borg objectives",
        )
        print(f"✓ Exported optimization filter plot: {opt_plot}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
