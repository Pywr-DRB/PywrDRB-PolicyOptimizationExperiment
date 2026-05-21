"""
MRF filtering helpers used by optimization and preprocessing.

Lower-basin reservoirs sometimes release for Trenton MRF targets (Pywr
``lower_basin_mrf_contributions``, MGD). Borg evaluates objectives only on
**normal operations** days: full simulation runs every timestep; the filter
excludes MRF-active periods (plus an optional day buffer). See ``MRF_FILTERING_README.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union
import json

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


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


def summarize_filter_fractions(
    datetime_index: pd.DatetimeIndex,
    mrf_active_raw: np.ndarray,
    normal_ops_filter: np.ndarray,
) -> Dict[str, float]:
    """Day counts and percentages for raw MRF-active vs buffered normal-ops filter."""
    n_total = len(datetime_index)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_mrf_active_raw": 0,
            "pct_mrf_active_raw": 0.0,
            "n_normal_ops": 0,
            "pct_normal_ops": 0.0,
            "n_excluded_from_objectives": 0,
            "pct_excluded_from_objectives": 0.0,
        }
    n_raw = int(np.sum(mrf_active_raw))
    n_normal = int(np.sum(normal_ops_filter))
    n_excl = n_total - n_normal
    return {
        "n_total": n_total,
        "n_mrf_active_raw": n_raw,
        "pct_mrf_active_raw": 100.0 * n_raw / n_total,
        "n_normal_ops": n_normal,
        "pct_normal_ops": 100.0 * n_normal / n_total,
        "n_excluded_from_objectives": n_excl,
        "pct_excluded_from_objectives": 100.0 * n_excl / n_total,
    }


def write_filter_summary_txt(
    out_path: Union[str, Path],
    *,
    bundle_label: str,
    eps_mgd: float,
    buffer_days: int,
    stats: Dict[str, float],
    training_index: pd.DatetimeIndex | None = None,
    training_stats: Dict[str, float] | None = None,
) -> None:
    """Human-readable normal-ops percentages for lab review."""
    out_path = Path(out_path)
    lines = [
        f"MRF filtering summary — {bundle_label}",
        f"Active threshold: mrf_trenton_* > {eps_mgd} MGD (Pywr lower_basin_mrf_contributions)",
        f"Objective filter buffer: {buffer_days} days before and after each raw active range",
        "",
        "Full simulation period (HDF5 datetime index):",
        f"  Total days: {int(stats['n_total'])}",
        f"  Raw MRF-active days (any lower basin > eps): {int(stats['n_mrf_active_raw'])} "
        f"({stats['pct_mrf_active_raw']:.2f}%)",
        f"  Days used for Borg objectives (normal ops, after buffer): "
        f"{int(stats['n_normal_ops'])} ({stats['pct_normal_ops']:.2f}%)",
        f"  Days excluded from objectives (active + buffer): "
        f"{int(stats['n_excluded_from_objectives'])} ({stats['pct_excluded_from_objectives']:.2f}%)",
    ]
    if training_index is not None and training_stats is not None:
        lines.extend(
            [
                "",
                f"Borg training overlap ({training_index.min().date()} .. {training_index.max().date()}):",
                f"  Total days: {int(training_stats['n_total'])}",
                f"  Raw MRF-active: {int(training_stats['n_mrf_active_raw'])} "
                f"({training_stats['pct_mrf_active_raw']:.2f}%)",
                f"  Normal ops (buffered): {int(training_stats['n_normal_ops'])} "
                f"({training_stats['pct_normal_ops']:.2f}%)",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_mrf_contributions_active_periods(
    contrib_df: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    eps_mgd: float,
    title: str = "Lower-basin MRF contributions (Pywr-DRB)",
) -> None:
    """Stacked mrf_trenton_* with shading where any contribution > eps."""
    if plt is None:
        raise ImportError("matplotlib required for MRF diagnostic plots")
    out_path = Path(out_path)
    df = contrib_df.copy()
    df.index = pd.to_datetime(df.index)
    cols = [c for c in df.columns if str(c).startswith("mrf_trenton_")]
    if not cols:
        raise ValueError("No mrf_trenton_* columns in contributions dataframe")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.stackplot(df.index, *[df[c].values for c in cols], labels=cols, alpha=0.85)
    any_active = (df[cols] > eps_mgd).any(axis=1)
    _shade_active_ranges(ax, df.index, any_active.values, color="#fdd", label=f"active > {eps_mgd} MGD")
    ax.set_ylabel("MGD")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mrf_filtering_for_optimization(
    contrib_df: pd.DataFrame,
    mrf_active_raw: np.ndarray,
    normal_ops_filter: np.ndarray,
    out_path: Union[str, Path],
    *,
    buffer_days: int,
    eps_mgd: float,
    title: str = "MRF filtering for Borg objectives",
) -> None:
    """Companion plot: raw active, buffer-only exclusion, normal-ops days."""
    if plt is None:
        raise ImportError("matplotlib required for MRF diagnostic plots")
    out_path = Path(out_path)
    idx = pd.to_datetime(contrib_df.index)
    raw = np.asarray(mrf_active_raw, dtype=bool)
    normal = np.asarray(normal_ops_filter, dtype=bool)
    buffer_only = (~normal) & (~raw)

    cols = [c for c in contrib_df.columns if str(c).startswith("mrf_trenton_")]
    total_mrf = contrib_df[cols].sum(axis=1) if cols else pd.Series(0.0, index=idx)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    ax0 = axes[0]
    ax0.plot(idx, total_mrf, color="#333", lw=0.8, label="Sum lower-basin MRF (MGD)")
    ax0.axhline(eps_mgd, color="#999", ls="--", lw=0.8, label=f"eps = {eps_mgd} MGD")
    _shade_active_ranges(ax0, idx, raw, color="#fcc", alpha=0.35)
    ax0.set_ylabel("MGD")
    ax0.set_title(title)
    ax0.legend(loc="upper left", fontsize=8)

    ax1 = axes[1]
    _shade_active_ranges(ax1, idx, raw, color="#e74c3c", alpha=0.5, label="Raw MRF-active")
    _shade_active_ranges(ax1, idx, buffer_only, color="#f39c12", alpha=0.45, label="Buffer only")
    ax1.set_ylabel("Filter")
    ax1.set_yticks([])
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = axes[2]
    _shade_active_ranges(ax2, idx, normal, color="#27ae60", alpha=0.35, label="Normal ops (objectives ON)")
    _shade_active_ranges(ax2, idx, ~normal, color="#bdc3c7", alpha=0.5, label="Excluded")
    ax2.set_ylabel("Borg fit")
    ax2.set_yticks([])
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_xlabel("Date")
    fig.suptitle(f"Buffer = {buffer_days} days each side of raw active ranges", fontsize=10, y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _shade_active_ranges(ax, index: pd.DatetimeIndex, mask: np.ndarray, **kwargs) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return
    label = kwargs.pop("label", None)
    labeled = False
    in_run = False
    start = None
    for i, flag in enumerate(mask):
        if flag and not in_run:
            start = index[i]
            in_run = True
        elif not flag and in_run:
            kw = dict(kwargs)
            if label and not labeled:
                kw["label"] = label
                labeled = True
            elif label:
                kw.pop("label", None)
            ax.axvspan(start, index[i - 1], **kw)
            in_run = False
    if in_run:
        kw = dict(kwargs)
        if label and not labeled:
            kw["label"] = label
        ax.axvspan(start, index[-1], **kw)

