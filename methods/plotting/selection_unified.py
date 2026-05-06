# methods/plotting/selection_unified.py
from __future__ import annotations
import re
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from methods.config import (
    BASELINE_ALIASES,
    BASELINE_VALUE_COL,
    OBJ_FILTER_BOUNDS,
    OBJ_LABELS,
    SENSES_ALL,
)
from methods.plotting.pick_labels import (
    DESIRED_PICKS_ORDER,
    DIVERSE_SOLUTION_1,
    DIVERSE_SOLUTION_2,
    MAX_HV_CONTRIBUTION_2D,
    MAX_CURVATURE_KNEE_POINT,
    MIN_EUCLIDEAN_L2_DISTANCE_TO_IP,
    MIN_MANHATTAN_L1_DISTANCE_TO_IP,
    MIN_WEIGHTED_CHEBYSHEV_DISTANCE_TO_IP,
    NORMALIZED_EQUAL_WEIGHT_MEAN_OPTIMUM,
    epsilon_constraint_label,
    objective_optimum_label,
)

# ===================== Helper wiring (robust & opinionated) =====================

def _obj_cols_from_config() -> List[str]:
    """Axis order from config."""
    return list(OBJ_LABELS.values())

def _present_objectives(df: pd.DataFrame, objectives: List[str]) -> List[str]:
    """Keep only objectives that exist in df and have at least one non-NaN."""
    keep = []
    for c in objectives:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            keep.append(c)
    return keep

def _sense_is_max(col: str) -> bool:
    return str(SENSES_ALL.get(col, "max")).lower().startswith("max")

def normalize_objectives(df: pd.DataFrame,
                         objectives: List[str],
                         bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    """
    Create sense-aware [0,1] normalized columns `${col}__norm` for each objective.
    - Uses OBJ_FILTER_BOUNDS when provided; otherwise per-DF min/max.
    - If spread is ~0, force a 1e-12 spread to avoid NaN.
    - Leaves NaNs from the source as NaNs (callers handle neutrality).
    """
    out = df.copy()
    for col in objectives:
        raw = pd.to_numeric(out[col], errors="coerce")
        lo, hi = (None, None)
        if bounds and col in bounds:
            lo, hi = bounds[col]
        if lo is None or hi is None:
            if raw.notna().any():
                lo, hi = float(raw.min()), float(raw.max())
            else:
                lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1e-12

        if _sense_is_max(col):
            norm = (raw - lo) / (hi - lo)
        else:
            norm = (hi - raw) / (hi - lo)

        out[f"{col}__norm"] = np.clip(norm, 0.0, 1.0)
    return out

def _canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _flip_if_neg(metric_key: str, val: float) -> float:
    k = str(metric_key).lower()
    return -float(val) if (k.startswith("neg_") or k.endswith("_neg") or "_neg_" in k) else float(val)

def baseline_series_from_df(
    baseline_df: Optional[pd.DataFrame],
    obj_cols: List[str],
    *,
    aliases: Optional[Mapping[str, str]] = None,
    value_col: str = BASELINE_VALUE_COL,
) -> pd.Series:
    """
    Map a baseline CSV (columns: ['metric', value_col]) to a Series keyed by
    pretty axis labels in obj_cols, with 'neg_*' metrics flipped.
    """
    if baseline_df is None or baseline_df.empty:
        return pd.Series(index=obj_cols, dtype=float)

    aliases = aliases or BASELINE_ALIASES
    if "metric" not in baseline_df.columns or value_col not in baseline_df.columns:
        # graceful empty return; caller can still proceed without baseline
        return pd.Series(index=obj_cols, dtype=float)

    # Build lookup: metric_name -> value
    metric_vals = {
        str(m): float(v)
        for m, v in zip(baseline_df["metric"], baseline_df[value_col])
    }
    canon_metric = {_canon(k): k for k in metric_vals}

    out = {}
    for pretty in obj_cols:
        # 1) explicit alias in config
        key = aliases.get(pretty)

        # 2) if OBJ_LABELS key maps to this pretty, try that name too
        if key is None:
            maybe_keys = [k for k, p in OBJ_LABELS.items() if p == pretty]
            if maybe_keys:
                key = canon_metric.get(_canon(maybe_keys[0]))

        # 3) fallback: canonicalize the pretty label itself
        if key is None:
            key = canon_metric.get(_canon(pretty))

        if key is None or key not in metric_vals:
            out[pretty] = np.nan
        else:
            out[pretty] = _flip_if_neg(key, metric_vals[key])

    return pd.Series(out, index=obj_cols, dtype=float)
# ====================== Baseline helpers (unchanged API) =======================
def append_baseline_row(
    obj_df: pd.DataFrame,
    baseline: Optional[pd.Series | pd.DataFrame],
    *,
    label_col: str,
    label_value: str = "Baseline",
    obj_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Append a baseline row to obj_df. Accepts either a ready Series or the raw
    baseline DataFrame; if DataFrame is provided, it is mapped via
    baseline_series_from_df().
    """
    if obj_df is None or obj_df.empty:
        return obj_df

    obj_cols = obj_cols or [c for c in OBJ_LABELS.values() if c in obj_df.columns]

    if isinstance(baseline, pd.DataFrame):
        baseline_ser = baseline_series_from_df(baseline, obj_cols)
    else:
        baseline_ser = baseline if isinstance(baseline, pd.Series) else None

    if baseline_ser is None or baseline_ser.empty:
        return obj_df

    row = {c: float(baseline_ser.get(c, np.nan)) for c in obj_cols}
    # Fill non-objective columns with first row (or NaN if none)
    for c in obj_df.columns:
        if c not in row:
            row[c] = obj_df.iloc[0][c] if len(obj_df) else np.nan
    row[label_col] = label_value

    out = pd.concat([obj_df, pd.DataFrame([row], columns=obj_df.columns)], ignore_index=True)
    for c in obj_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def filter_better_than_baseline(obj_df, baseline_metrics_df, margin=0.0,
                                obj_cols=None, senses=None, require_all=True):
    if baseline_metrics_df is None or baseline_metrics_df.empty or obj_df.empty:
        return obj_df
    senses   = senses   or SENSES_ALL
    obj_cols = obj_cols or [c for c in OBJ_LABELS.values() if c in obj_df.columns]
    bl = baseline_series_from_df(baseline_metrics_df, obj_cols)

    better_any = pd.Series(False, index=obj_df.index)
    better_all = pd.Series(True,  index=obj_df.index)

    for col in obj_cols:
        if col not in obj_df.columns or pd.isna(bl.get(col)):
            continue
        vec = pd.to_numeric(obj_df[col], errors="coerce")
        b   = float(bl[col])
        if str(senses.get(col, "max")).lower().startswith("max"):
            win = vec >= (b + margin)
        else:
            win = vec <= (b - margin)
        better_any |= win
        better_all &= win

    keep = (better_all if require_all else better_any)
    return obj_df.loc[keep].copy()

# ============================ Core pick strategies =============================

def _safe_best_average_index(obj_df: pd.DataFrame,
                             objectives: List[str]) -> int:
    """
    Equal-weight average across AVAILABLE objectives (normalized).
    - Ignores objectives that are absent or all-NaN.
    - Averages with skipna; if all-NaN, falls back to neutral 0.5 fill.
    - Always returns a valid row index from obj_df.
    """
    if obj_df is None or obj_df.empty:
        # shouldn't happen, but be defensive
        return 0

    avail = _present_objectives(obj_df, objectives)
    if not avail:
        return int(obj_df.index[0])

    nd = normalize_objectives(obj_df, avail, bounds=OBJ_FILTER_BOUNDS)
    cols = [f"{c}__norm" for c in avail]

    avg = nd[cols].mean(axis=1, skipna=True)
    if avg.notna().any():
        return int(avg.idxmax())

    # If literally everything is NaN, treat as neutral 0.5
    neutral = nd[cols].fillna(0.5).mean(axis=1)
    return int(neutral.idxmax())


def best_indices_all(obj_df: pd.DataFrame,
                     objectives: List[str]) -> Dict[str, int]:
    """
    Best per-objective + robust best equal-weight average across ALL objectives.
    Only considers objectives that exist and have data.
    """
    picks: Dict[str, int] = {}
    avail = _present_objectives(obj_df, objectives)

    # per-objective bests
    for col in avail:
        vec = pd.to_numeric(obj_df[col], errors="coerce")
        if not vec.notna().any():
            continue
        idx = int(vec.idxmax() if _sense_is_max(col) else vec.idxmin())
        picks[objective_optimum_label(col)] = idx

    # Unified multi-objective average (sense-aware norms). Distinct name from legacy
    # ``Best Average All`` (scaled min–max in ``stage1``) to avoid duplicate indices.
    picks[NORMALIZED_EQUAL_WEIGHT_MEAN_OPTIMUM] = _safe_best_average_index(obj_df, objectives)
    return picks


def _lp_distance_to_ideal(nd: pd.DataFrame, objectives: List[str], p: float) -> pd.Series:
    cols = [f"{c}__norm" for c in objectives if f"{c}__norm" in nd.columns]
    X = nd[cols].fillna(0.5).to_numpy(float)  # neutral fill for missing
    if X.size == 0:
        # return zeros series aligned to df index
        return pd.Series(np.zeros(len(nd), dtype=float), index=nd.index)

    w = np.ones(X.shape[1], dtype=float) / X.shape[1]
    D = np.abs(1.0 - X)  # distance to ideal = 1
    if np.isinf(p):
        d = (w * D).max(axis=1)
    elif p == 1:
        d = (w * D).sum(axis=1)
    else:
        d = np.power((w * np.power(D, p)).sum(axis=1), 1.0 / p)
    return pd.Series(d, index=nd.index)


def _eps_constraint(nd: pd.DataFrame, eps_on: str, optimize: str,
                    qs=(0.5, 0.8)) -> Dict[str, int]:
    out: Dict[str, int] = {}
    c_eps, c_opt = f"{eps_on}__norm", f"{optimize}__norm"
    if c_eps not in nd.columns or c_opt not in nd.columns:
        return out
    thr = nd[c_eps].fillna(0.5).quantile(qs, interpolation="nearest").to_numpy()
    for q, t in zip(qs, thr):
        sub = nd[nd[c_eps].fillna(0.5) >= t]
        if len(sub):
            q_int = int(round(100 * float(q)))
            out[epsilon_constraint_label(eps_on, q_int, optimize)] = int(sub[c_opt].idxmax())
    return out

def _knee_point_2d(nd: pd.DataFrame, objectives: List[str]) -> Optional[int]:
    # Return the ORIGINAL index label of the knee point (not a 0..n-1 position)
    if len(objectives) != 2 or len(nd) < 3:
        return None
    x = nd[f"{objectives[0]}__norm"].fillna(0.5).to_numpy(float)
    y = nd[f"{objectives[1]}__norm"].fillna(0.5).to_numpy(float)

    i1_pos = int(np.argmax(x))
    i2_pos = int(np.argmax(y))
    P1, P2 = np.array([x[i1_pos], y[i1_pos]]), np.array([x[i2_pos], y[i2_pos]])
    v = P2 - P1
    den = (np.hypot(v[0], v[1]) + 1e-12)

    d = np.abs(v[0]*(y - P1[1]) - v[1]*(x - P1[0])) / den
    d[[i1_pos, i2_pos]] = -np.inf  # exclude the extremes
    j_pos = int(np.argmax(d))
    if not (np.isfinite(d[j_pos]) and d[j_pos] >= 0):
        return None

    # Map 0..n-1 position back to original index label
    return nd.index[j_pos]


def _fps_diverse(nd: pd.DataFrame, objectives: List[str], k: int = 2) -> Dict[str, int]:
    """
    Farthest-Point Sampling on normalized objectives.
    Works in positional space, returns ORIGINAL index labels.
    """
    cols = [f"{c}__norm" for c in objectives if f"{c}__norm" in nd.columns]
    X = nd[cols].fillna(0.5).to_numpy(float)
    out: Dict[str, int] = {}
    if X.shape[0] == 0:
        return out

    # Seed picks by positions, not labels
    picks_pos: List[int] = []
    if len(cols) >= 2:
        picks_pos = [int(np.argmax(X[:, 0])), int(np.argmax(X[:, 1]))]
    else:
        picks_pos = [int(np.argmax(X.sum(axis=1)))]

    # Greedy FPS loop in positional space
    while len(out) < k and len(picks_pos) < len(nd):
        # distance to the closest picked point
        dists = np.linalg.norm(X[:, None, :] - X[np.array(picks_pos)][None, :, :], axis=2)
        dmin = np.min(dists, axis=1)
        dmin[np.array(picks_pos, dtype=int)] = -np.inf  # don't re-pick

        nxt_pos = int(np.argmax(dmin))
        if dmin[nxt_pos] < 0:
            break
        picks_pos.append(nxt_pos)
        # Return the ORIGINAL index label for this position
        labels = (DIVERSE_SOLUTION_1, DIVERSE_SOLUTION_2)
        if len(out) < len(labels):
            out[labels[len(out)]] = nd.index[nxt_pos]

    return out


def _hv_contrib_2d(nd: pd.DataFrame, objectives: List[str]) -> Optional[int]:
    cols = [f"{c}__norm" for c in objectives if f"{c}__norm" in nd.columns][:2]
    if len(cols) != 2:
        return None
    P = nd[cols].fillna(0.5).to_numpy(float)
    ids = nd.index.to_numpy()
    order = np.argsort(-P[:, 0], kind="mergesort")
    keep, besty = [], -np.inf
    for k in order:
        if P[k, 1] > besty + 1e-12:
            keep.append(k); besty = P[k, 1]
    Pn = P[keep]; ids = ids[keep]
    if len(Pn) == 0:
        return None

    def hv(points):
        pts = points[np.argsort(-points[:, 0], kind="mergesort")]
        total, ymax = 0.0, 0.0
        for i in range(len(pts)):
            x, y = pts[i]
            xnext = pts[i+1, 0] if i < len(pts)-1 else 0.0
            ymax = max(ymax, y)
            total += ymax * max(x - xnext, 0.0)
        return float(total)

    H = hv(Pn)
    contrib = []
    for j in range(len(Pn)):
        Pm = np.delete(Pn, j, axis=0) if len(Pn) > 1 else np.empty((0, 2))
        contrib.append(H - hv(Pm))
    return int(ids[int(np.argmax(contrib))])

# ======================== Public selection API (wired) =========================
# DESIRED_PICKS_ORDER is defined in pick_labels.py (canonical names + filename slugs).

def sophisticated_picks(obj_df: pd.DataFrame,
                        objectives: List[str]) -> Dict[str, int]:
    """Distance-, knee-, ε-constraint-, diversity-, and HV-based picks (robust to NaNs)."""
    avail = _present_objectives(obj_df, objectives)
    if not avail:
        return {}

    nd = normalize_objectives(obj_df, avail, bounds=OBJ_FILTER_BOUNDS)
    out: Dict[str, int] = {}
    out[MIN_EUCLIDEAN_L2_DISTANCE_TO_IP] = int(_lp_distance_to_ideal(nd, avail, 2).idxmin())
    out[MIN_WEIGHTED_CHEBYSHEV_DISTANCE_TO_IP] = int(_lp_distance_to_ideal(nd, avail, np.inf).idxmin())
    out[MIN_MANHATTAN_L1_DISTANCE_TO_IP] = int(_lp_distance_to_ideal(nd, avail, 1).idxmin())

    if len(avail) >= 2:
        kp = _knee_point_2d(nd, avail)
        if kp is not None:
            out[MAX_CURVATURE_KNEE_POINT] = kp
        # Anchor ε-constraint on Release NSE when present so labels match ``stage1`` default picks
        if "Release NSE" in avail:
            eps_on = "Release NSE"
            rest = [c for c in avail if c != eps_on]
            optimize = "Storage NSE" if "Storage NSE" in rest else rest[0]
        else:
            eps_on, optimize = avail[0], avail[1]
        out.update(_eps_constraint(nd, eps_on, optimize, qs=(0.5, 0.8)))
        out.update(_fps_diverse(nd, avail, k=2))
        hv = _hv_contrib_2d(nd, avail)
        if hv is not None:
            out[MAX_HV_CONTRIBUTION_2D] = hv
    return out


def build_unified_picks(obj_df: pd.DataFrame,
                        objectives: Optional[List[str]] = None) -> Dict[str, int]:
    """
    One-stop “wiring”:
      - Auto-detect present objectives
      - Always add normalized equal-weight mean optimum (unified multi-objective average; sense-aware)
      - Add distance/knee/ε/diversity/HV picks when meaningful
      - Add per-objective optima
      - Deduplicate by solution index, preserving a sensible order
    """
    if objectives is None:
        objectives = _obj_cols_from_config()

    # Build the picks
    picks = {}
    picks.update(best_indices_all(obj_df, objectives))
    picks.update(sophisticated_picks(obj_df, objectives))

    # Order + dedupe by index
    seen, ordered = set(), {}

    # 1) Desired canonical order
    for lab in DESIRED_PICKS_ORDER:
        if lab in picks:
            idx = int(picks[lab])
            if idx not in seen:
                ordered[lab] = idx
                seen.add(idx)

    # 2) Ensure all per-objective optima are included (in stable order)
    for col in _present_objectives(obj_df, objectives):
        lab = objective_optimum_label(col)
        if lab in picks:
            idx = int(picks[lab])
            if idx not in seen:
                ordered[lab] = idx
                seen.add(idx)

    # 3) Any remaining picks (e.g. ε-constraint with a non-default secondary objective)
    for lab, idx in picks.items():
        if lab in ordered:
            continue
        idx = int(idx)
        if idx not in seen:
            ordered[lab] = idx
            seen.add(idx)

    return ordered


def stamp_highlight(obj_df: pd.DataFrame,
                    pick_map: Dict[str, int],
                    label_col: str = "highlight") -> pd.DataFrame:
    """
    Mark picked rows in a categorical column (default 'highlight').
    Non-picked rows are labeled 'Other'.
    """
    out = obj_df.copy()
    out[label_col] = "Other"
    for lab, idx in pick_map.items():
        if idx in out.index:
            out.loc[idx, label_col] = lab
    return out
