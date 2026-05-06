"""
mrf_masking.py

Helper functions for MRF (Minimum Release Flow) masking in optimization.
Used to identify "normal operations" periods vs. drought operations periods.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd


def mask_to_ranges(mask: pd.Series) -> List[Dict[str, any]]:
    """
    Convert a boolean mask to a list of contiguous date range dictionaries.
    
    Parameters
    ----------
    mask : pd.Series
        Boolean Series with DatetimeIndex
        
    Returns
    -------
    ranges : List[Dict]
        List of dictionaries with keys: 'start', 'end', 'days'
    """
    if mask.empty:
        return []
    
    ranges = []
    in_range = False
    start_idx = None
    
    for i, (date, is_active) in enumerate(mask.items()):
        if is_active and not in_range:
            # Start of new active period
            start_idx = i
            in_range = True
        elif not is_active and in_range:
            # End of active period
            end_date = mask.index[i - 1]  # Last True day
            start_date = mask.index[start_idx]
            ranges.append({
                "start": start_date,
                "end": end_date,
                "days": (end_date - start_date).days + 1
            })
            in_range = False
            start_idx = None
    
    # Handle case where mask ends while still in active period
    if in_range and start_idx is not None:
        end_date = mask.index[-1]
        start_date = mask.index[start_idx]
        ranges.append({
            "start": start_date,
            "end": end_date,
            "days": (end_date - start_date).days + 1
        })
    
    return ranges


def ranges_to_mask(ranges: List[Dict], dt_index: pd.DatetimeIndex) -> pd.Series:
    """
    Convert date ranges back to a boolean mask aligned to datetime index.
    
    Parameters
    ----------
    ranges : List[Dict]
        List of dictionaries with 'start' and 'end' keys (pd.Timestamp or string)
    dt_index : pd.DatetimeIndex
        Target datetime index to align mask to
        
    Returns
    -------
    mask : pd.Series
        Boolean Series with same index as dt_index
    """
    mask = pd.Series(False, index=dt_index)
    
    for r in ranges:
        start = pd.to_datetime(r["start"])
        end = pd.to_datetime(r["end"])
        mask.loc[start:end] = True
    
    return mask


def build_lower_basin_mrf_active_dict(
    df: pd.DataFrame,
    eps: float,
    reservoirs: List[str],
    prefix: str = "mrf_trenton_",
) -> Tuple[Dict[str, List[Dict]], pd.Series]:
    """
    Build MRF active date ranges dictionary for lower basin reservoirs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MRF contribution columns (datetime index)
    eps : float
        Threshold (MGD) for considering contribution 'active'
    reservoirs : List[str]
        List of reservoir names (without prefix)
    prefix : str
        Column name prefix (e.g., "mrf_trenton_")
        
    Returns
    -------
    mrf_ranges_dict : Dict[str, List[Dict]]
        Dictionary mapping reservoir names to list of active date ranges
    any_mask : pd.Series
        Boolean mask that is True when ANY reservoir is active
    """
    mrf_ranges_dict = {}
    any_mask = pd.Series(False, index=df.index)
    
    for res in reservoirs:
        col_name = f"{prefix}{res}"
        if col_name not in df.columns:
            print(f"  WARNING: Column '{col_name}' not found, skipping {res}")
            continue
        
        # Create mask: active when contribution > eps
        mask = df[col_name] > eps
        
        # Convert to ranges
        ranges = mask_to_ranges(mask)
        mrf_ranges_dict[res] = ranges
        
        # Update ANY mask
        any_mask = any_mask | mask
    
    # Add ANY_lower_basin key
    any_ranges = mask_to_ranges(any_mask)
    mrf_ranges_dict["ANY_lower_basin"] = any_ranges
    
    return mrf_ranges_dict, any_mask


def load_mrf_ranges(json_path: str | Path) -> Dict:
    """
    Load MRF active date ranges from JSON file.
    
    Parameters
    ----------
    json_path : str or Path
        Path to JSON file
        
    Returns
    -------
    mrf_ranges_dict : Dict
        Dictionary mapping keys to lists of date range dicts
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Convert string dates back to Timestamp objects
    result = {}
    for key, ranges in data.items():
        result[key] = [
            {
                "start": pd.Timestamp(r["start"]),
                "end": pd.Timestamp(r["end"]),
                "days": r["days"]
            }
            for r in ranges
        ]
    
    return result


def build_normal_ops_mask(
    datetime_index: pd.DatetimeIndex,
    mrf_ranges_dict: Dict,
    reservoir_name: str | None = None,
    mode: str = "ANY",
    buffer_days: int = 0,
) -> np.ndarray:
    """
    Create a boolean mask for "normal operations" (i.e., NOT MRF-active).
    
    Optionally expands each MRF-active range by buffer_days before and after,
    so transition/anticipation periods (e.g. operators preparing for drought)
    are excluded from normal ops and do not influence objectives.
    
    Parameters
    ----------
    datetime_index : pd.DatetimeIndex
        Target datetime index
    mrf_ranges_dict : Dict
        Dictionary of MRF active ranges (from load_mrf_ranges)
    reservoir_name : str, optional
        Reservoir name (if mode="RES")
    mode : str
        "ANY" to use ANY_lower_basin, "RES" to use reservoir-specific
    buffer_days : int
        Days to add before and after each MRF-active range (default 0).
        Use a few days (e.g. 3–7) to exclude drought transition/anticipation.
        
    Returns
    -------
    normal_ops_mask : np.ndarray
        Boolean array: True = normal ops, False = MRF-active (including buffer)
    """
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
    
    # Create mask: True = MRF active (with optional buffer)
    mrf_active_mask = pd.Series(False, index=datetime_index)
    for r in mrf_ranges_dict[key]:
        start = pd.to_datetime(r["start"])
        end = pd.to_datetime(r["end"])
        start_buf = max(idx_min, start - delta)
        end_buf = min(idx_max, end + delta)
        mrf_active_mask.loc[start_buf:end_buf] = True
    
    # Normal ops = NOT MRF active
    normal_ops_mask = ~mrf_active_mask
    
    return normal_ops_mask.values


def validate_mask_alignment(
    datetime_index: pd.DatetimeIndex,
    normal_ops_mask: np.ndarray,
    min_normal_days: int = 365,
) -> Tuple[bool, str]:
    """
    Validate that a normal ops mask is properly aligned and has sufficient days.
    
    Parameters
    ----------
    datetime_index : pd.DatetimeIndex
        Expected datetime index
    normal_ops_mask : np.ndarray
        Boolean array: True = normal ops, False = MRF-active
    min_normal_days : int
        Minimum number of normal ops days required
        
    Returns
    -------
    is_valid : bool
        True if mask is valid
    msg : str
        Validation message
    """
    if len(normal_ops_mask) != len(datetime_index):
        return False, f"Mask length ({len(normal_ops_mask)}) != datetime length ({len(datetime_index)})"
    
    n_normal = np.sum(normal_ops_mask)
    n_total = len(normal_ops_mask)
    pct_normal = 100.0 * n_normal / n_total if n_total > 0 else 0.0
    
    if n_normal < min_normal_days:
        return False, f"Insufficient normal ops days: {n_normal} < {min_normal_days}"
    
    return True, f"Valid mask: {n_normal}/{n_total} normal ops days ({pct_normal:.1f}%)"
