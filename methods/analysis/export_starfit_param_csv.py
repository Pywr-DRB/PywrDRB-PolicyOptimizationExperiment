#!/usr/bin/env python3
"""
Export selected STARFIT policies to a PywrDRB-ready starfit.csv.

Workflow:
1) Build a selection manifest from figure picks (one row per selected policy).
2) Run this script to extract selected parameter vectors from optimization CSVs.
3) Copy the output CSV into PywrDRB operational constants as `starfit.csv`.

Expected manifest columns:
- reservoir           (required)
- policy_id           (required)
- source_csv          (required; path to optimization CSV containing selected row)
- row_index           (required; 0-based row index in source_csv)

Optional manifest columns:
- moea_policy         (if present, must be STARFIT)
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from methods.config import STARFIT_PARAM_NAMES
from methods.utils.policy_parameter_naming import rename_vars_with_param_names


REQUIRED_MANIFEST_COLS = {"reservoir", "policy_id", "source_csv", "row_index"}


def _validate_manifest(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_MANIFEST_COLS.difference(df.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    if df.empty:
        raise ValueError("Manifest is empty.")

    if "moea_policy" in df.columns:
        bad = df.loc[df["moea_policy"].astype(str).str.upper() != "STARFIT"]
        if not bad.empty:
            raise ValueError(
                "Manifest includes non-STARFIT rows in 'moea_policy'. "
                "This exporter only supports STARFIT selections."
            )


def _extract_params_from_source_row(source_df: pd.DataFrame, row_index: int) -> Dict[str, float]:
    if row_index < 0 or row_index >= len(source_df):
        raise IndexError(f"row_index={row_index} out of bounds for source file with {len(source_df)} rows.")

    row = source_df.iloc[int(row_index)]

    # Case A: source CSV already has STARFIT parameter names.
    if all(p in source_df.columns for p in STARFIT_PARAM_NAMES):
        return {p: float(row[p]) for p in STARFIT_PARAM_NAMES}

    # Case B: source CSV uses var1..varN columns.
    var_cols = [c for c in source_df.columns if str(c).lower().startswith("var")]
    if not var_cols:
        raise ValueError(
            "Source CSV must contain STARFIT parameter columns or var* columns. "
            f"Columns found: {list(source_df.columns)[:30]}"
        )
    renamed = rename_vars_with_param_names(source_df[var_cols].copy(), "STARFIT")
    if not all(p in renamed.columns for p in STARFIT_PARAM_NAMES):
        raise ValueError(
            "Could not map var* columns to STARFIT names. "
            f"Mapped columns: {list(renamed.columns)}"
        )
    renamed_row = renamed.iloc[int(row_index)]
    return {p: float(renamed_row[p]) for p in STARFIT_PARAM_NAMES}


def _load_defaults_from_istarf(path: Path, reservoirs: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"reservoir", "policy_id", *STARFIT_PARAM_NAMES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"ISTARF default source missing required columns: {missing}")
    defaults = df[df["policy_id"].astype(str) == "default"].copy()
    defaults = defaults[defaults["reservoir"].astype(str).isin(reservoirs)]
    defaults = defaults.drop_duplicates(subset=["reservoir"], keep="first")
    out_cols = ["reservoir", "policy_id", *STARFIT_PARAM_NAMES]
    return defaults[out_cols].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="CSV manifest with selected figure picks.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output starfit.csv path.",
    )
    parser.add_argument(
        "--include-defaults-from-istarf",
        action="store_true",
        help="Append policy_id=default rows per reservoir from ISTARF source if missing.",
    )
    parser.add_argument(
        "--istarf-source",
        type=Path,
        default=Path("obs_data/drb_model_istarf_conus.csv"),
        help="ISTARF CSV used for default fallback rows.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    _validate_manifest(manifest)

    rows = []
    cache: Dict[str, pd.DataFrame] = {}

    for _, pick in manifest.iterrows():
        source_csv = str(pick["source_csv"])
        source_path = Path(source_csv)
        if not source_path.is_absolute():
            source_path = (args.manifest.parent / source_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Source CSV not found: {source_path}")

        if str(source_path) not in cache:
            cache[str(source_path)] = pd.read_csv(source_path)
        source_df = cache[str(source_path)]

        params = _extract_params_from_source_row(source_df, int(pick["row_index"]))
        row = {
            "reservoir": str(pick["reservoir"]),
            "policy_id": str(pick["policy_id"]),
        }
        row.update(params)
        rows.append(row)

    out_cols = ["reservoir", "policy_id", *STARFIT_PARAM_NAMES]
    out_df = pd.DataFrame(rows)[out_cols].copy()

    if args.include_defaults_from_istarf:
        reservoirs = sorted(out_df["reservoir"].astype(str).unique())
        defaults = _load_defaults_from_istarf(args.istarf_source, reservoirs)
        have_default = set(
            out_df.loc[out_df["policy_id"].astype(str) == "default", "reservoir"].astype(str).tolist()
        )
        defaults_to_add = defaults.loc[~defaults["reservoir"].astype(str).isin(have_default)].copy()
        if not defaults_to_add.empty:
            out_df = pd.concat([out_df, defaults_to_add[out_cols]], axis=0, ignore_index=True)

    out_df = out_df.sort_values(["reservoir", "policy_id"]).reset_index(drop=True)

    dup = out_df.duplicated(subset=["reservoir", "policy_id"], keep=False)
    if dup.any():
        bad = out_df.loc[dup, ["reservoir", "policy_id"]]
        raise ValueError(
            "Duplicate (reservoir, policy_id) in output. "
            f"Please make policy_id unique per reservoir.\n{bad.to_string(index=False)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"Wrote {len(out_df)} rows to {args.output}")
    print("Rows per reservoir:")
    print(out_df.groupby("reservoir")["policy_id"].count().to_string())


if __name__ == "__main__":
    main()
