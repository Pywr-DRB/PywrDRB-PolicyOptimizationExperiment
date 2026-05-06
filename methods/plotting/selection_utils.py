"""
Advanced Pareto pick helpers used by the figure pipeline (``methods.postprocess.figures_primary`` / ``figures_validation``).

Delegates to :mod:`methods.plotting.selection_unified` (single source of truth).
"""

from __future__ import annotations

import pandas as pd

from methods.plotting.pick_labels import MAX_HV_CONTRIBUTION_2D
from methods.plotting.selection_unified import build_unified_picks, stamp_highlight
from methods.plotting.theme import ADVANCED_COLORS

__all__ = [
    "ADVANCED_COLORS",
    "apply_combined_selection_column",
    "compute_and_apply_advanced_highlights",
]


def _merge_legacy_adv_label(leg: str, adv: str) -> str:
    """Single legend category: if both name different picks, join; never drop one methodology."""
    lo = str(leg).strip()
    ao = str(adv).strip()
    if lo == "Other" and ao == "Other":
        return "Other"
    if lo == "Other":
        return ao
    if ao == "Other":
        return lo
    if lo == ao:
        return lo
    return f"{lo} · {ao}"


def apply_combined_selection_column(
    obj_df: pd.DataFrame,
    *,
    out_col: str = "highlight_selection",
) -> pd.DataFrame:
    """
    One categorical label per row for parallel-axis coloring: merges ``highlight`` (scalarization-style
    picks) and ``highlight_adv`` (literature-style picks). If a row is named by **both** with different
    labels, the category is ``\"legacy · adv\"`` so both appear in one legend—not two separate figures
    and not silent override by either side.
    """
    if obj_df is None or obj_df.empty:
        return obj_df
    if "highlight" not in obj_df.columns or "highlight_adv" not in obj_df.columns:
        raise ValueError("apply_combined_selection_column requires 'highlight' and 'highlight_adv' columns")
    out = obj_df.copy()
    leg = out["highlight"].astype(str)
    adv = out["highlight_adv"].astype(str)
    out[out_col] = [_merge_legacy_adv_label(l, a) for l, a in zip(leg, adv)]
    return out


def compute_and_apply_advanced_highlights(
    obj_df,
    objectives,
    senses=None,
    bounds=None,
    eps_qs=(0.5, 0.8),
    add_k_diverse=2,
    include_hv=False,
    out_label_col="highlight_adv",
):
    """
    Add ``out_label_col`` (default ``highlight_adv``) and return candidate table + map.

    Parameters ``senses``, ``eps_qs``, and ``add_k_diverse`` are accepted for API
    compatibility; picking logic follows ``selection_unified`` (uses config bounds/senses).
    """
    del senses, eps_qs, add_k_diverse  # unified implementation uses config; keep call sites stable
    _ = bounds  # reserved for future wiring; filtering uses OBJ_FILTER_BOUNDS in load_results

    picks = build_unified_picks(obj_df, objectives)
    if not include_hv:
        picks.pop(MAX_HV_CONTRIBUTION_2D, None)

    obj_aug = stamp_highlight(obj_df, picks, label_col=out_label_col)
    cand_map = {k: int(v) for k, v in picks.items()}
    cand_df = pd.DataFrame(list(picks.items()), columns=["pick", "row_index"])
    return obj_aug, cand_df, cand_map
