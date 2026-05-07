"""
MRF filtering helpers used by optimization and preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd


def filter_to_ranges(active_filter: pd.Series) -> List[Dict[str, any]]:
    """Convert a boolean filter to contiguous date-range dictionaries."""
    if active_filter.empty:
        return []

    ranges = []
    in_range = False
    start_idx = None

    for i, (date, is_active) in enumerate(active_filter.items()):
        if is_active and not in_range:
            start_idx = i
            in_range = True
        elif not is_active and in_range:
            end_date = active_filter.index[i - 1]
            start_date = active_filter.index[start_idx]
            ranges.append({"start": start_date, "end": end_date, "days": (end_date - start_date).days + 1})
            in_range = False
            start_idx = None

    if in_range and start_idx is not None:
        end_date = active_filter.index[-1]
        start_date = active_filter.index[start_idx]
        ranges.append({"start": start_date, "end": end_date, "days": (end_date - start_date).days + 1})

    return ranges


def ranges_to_filter(ranges: List[Dict], dt_index: pd.DatetimeIndex) -> pd.Series:
    """Convert date ranges back to a boolean filter aligned to a datetime index."""
    active_filter = pd.Series(False, index=dt_index)
    for r in ranges:
        start = pd.to_datetime(r["start"])
        end = pd.to_datetime(r["end"])
        active_filter.loc[start:end] = True
    return active_filter


def build_lower_basin_mrf_active_dict(
    df: pd.DataFrame,
    eps: float,
    reservoirs: List[str],
    prefix: str = "mrf_trenton_",
) -> Tuple[Dict[str, List[Dict]], pd.Series]:
    """Build active-date ranges for each lower-basin reservoir and ANY_lower_basin."""
    mrf_ranges_dict = {}
    any_filter = pd.Series(False, index=df.index)

    for res in reservoirs:
        col_name = f"{prefix}{res}"
        if col_name not in df.columns:
            print(f"  WARNING: Column '{col_name}' not found, skipping {res}")
            continue

        active_filter = df[col_name] > eps
        ranges = filter_to_ranges(active_filter)
        mrf_ranges_dict[res] = ranges
        any_filter = any_filter | active_filter

    mrf_ranges_dict["ANY_lower_basin"] = filter_to_ranges(any_filter)
    return mrf_ranges_dict, any_filter


def load_mrf_ranges(json_path: str | Path) -> Dict:
    """Load MRF active date ranges from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    result = {}
    for key, ranges in data.items():
        result[key] = [{"start": pd.Timestamp(r["start"]), "end": pd.Timestamp(r["end"]), "days": r["days"]} for r in ranges]
    return result


def load_mrf_daily_filter_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load daily MRF-active filter CSV (0/1 columns) indexed by datetime."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def build_normal_ops_filter_from_daily(
    datetime_index: pd.DatetimeIndex,
    daily_filter_df: pd.DataFrame,
    reservoir_name: str | None = None,
    mode: str = "ANY",
    buffer_days: int = 0,
) -> np.ndarray:
    """Create normal-ops filter from daily MRF-active 0/1 CSV using range-buffer logic."""
    if mode == "ANY":
        key = "ANY_lower_basin"
    elif mode == "RES":
        if reservoir_name is None:
            raise ValueError("reservoir_name required when mode='RES'")
        key = reservoir_name
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'ANY' or 'RES'")

    if key not in daily_filter_df.columns:
        raise KeyError(f"Column '{key}' not found in daily filter CSV. Available: {list(daily_filter_df.columns)}")

    active = daily_filter_df[key].astype(float).reindex(datetime_index).fillna(0.0).astype(int).astype(bool)
    mrf_ranges_dict = {key: filter_to_ranges(active)}
    return build_normal_ops_filter(
        datetime_index=datetime_index,
        mrf_ranges_dict=mrf_ranges_dict,
        reservoir_name=(reservoir_name if mode == "RES" else None),
        mode=mode,
        buffer_days=buffer_days,
    )


def build_normal_ops_filter(
    datetime_index: pd.DatetimeIndex,
    mrf_ranges_dict: Dict,
    reservoir_name: str | None = None,
    mode: str = "ANY",
    buffer_days: int = 0,
) -> np.ndarray:
    """Create a boolean filter for normal operations (NOT MRF-active)."""
    if mode == "ANY":
        key = "ANY_lower_basin"
    elif mode == "RES":
        if reservoir_name is None:
            raise ValueError("reservoir_name required when mode='RES'")
        key = reservoir_name
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'ANY' or 'RES'")

    if key not in mrf_ranges_dict:
        raise KeyError(f"Key '{key}' not found in mrf_ranges_dict. Available: {list(mrf_ranges_dict.keys())}")

    idx_min = datetime_index.min()
    idx_max = datetime_index.max()
    delta = pd.Timedelta(days=buffer_days)

    mrf_active_filter = pd.Series(False, index=datetime_index)
    for r in mrf_ranges_dict[key]:
        start = pd.to_datetime(r["start"])
        end = pd.to_datetime(r["end"])
        start_buf = max(idx_min, start - delta)
        end_buf = min(idx_max, end + delta)
        mrf_active_filter.loc[start_buf:end_buf] = True

    normal_ops_filter = ~mrf_active_filter
    return normal_ops_filter.values


def validate_filter_alignment(
    datetime_index: pd.DatetimeIndex,
    normal_ops_filter: np.ndarray,
    min_normal_days: int = 365,
) -> Tuple[bool, str]:
    """Validate filter length/alignment and minimum number of normal-operation days."""
    if len(normal_ops_filter) != len(datetime_index):
        return False, f"Filter length ({len(normal_ops_filter)}) != datetime length ({len(datetime_index)})"

    n_normal = np.sum(normal_ops_filter)
    n_total = len(normal_ops_filter)
    pct_normal = 100.0 * n_normal / n_total if n_total > 0 else 0.0

    if n_normal < min_normal_days:
        return False, f"Insufficient normal ops days: {n_normal} < {min_normal_days}"

    return True, f"Valid filter: {n_normal}/{n_total} normal ops days ({pct_normal:.1f}%)"

