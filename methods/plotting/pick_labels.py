# methods/plotting/pick_labels.py
"""Canonical Pareto solution pick labels, legacy aliases, and short filename slugs."""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional

# --- Display names (canonical; used in highlight / highlight_adv / legends) ---

RELEASE_NSE_OBJECTIVE_OPTIMUM = "Release NSE objective optimum"
STORAGE_NSE_OBJECTIVE_OPTIMUM = "Storage NSE objective optimum"
AVERAGE_NSE_OBJECTIVE_OPTIMUM = "Average NSE objective optimum"
BEST_AVERAGE_ALL = "Best Average All"

NORMALIZED_EQUAL_WEIGHT_MEAN_OPTIMUM = "Normalized equal-weight mean optimum"

MIN_EUCLIDEAN_L2_DISTANCE_TO_IP = "Minimum Euclidean (L2) distance to IP"
MIN_WEIGHTED_CHEBYSHEV_DISTANCE_TO_IP = "Minimum weighted Chebyshev distance to IP"
MIN_MANHATTAN_L1_DISTANCE_TO_IP = "Minimum Manhattan (L1) distance to IP"

MAX_CURVATURE_KNEE_POINT = "Maximum-curvature knee point (along the Pareto front)"

DIVERSE_SOLUTION_1 = (
    "Diverse solution 1 — maximin distance to two objective-axis extrema (normalized objectives)"
)
DIVERSE_SOLUTION_2 = (
    "Diverse solution 2 — maximin distance to extrema and Diverse solution 1 (normalized objectives)"
)

MAX_HV_CONTRIBUTION_2D = "Maximum hypervolume contribution (2D nondominated front)"


def objective_optimum_label(objective_col: str) -> str:
    return f"{objective_col} objective optimum"


def epsilon_constraint_label(eps_on: str, q: int, optimize: str) -> str:
    """q is 50 or 80 for Q50 / Q80 style thresholds."""
    return f"Epsilon-constraint: feasible set {eps_on} ≥ Q{q}; maximize {optimize}"


def _slug_component(s: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")
    return t or "x"


def _slug_objective_optimum(objective_col: str) -> str:
    return f"objopt_{_slug_component(objective_col)}"


def _slug_epsilon_constraint(eps_on: str, q: int, optimize: str) -> str:
    return f"eps_{_slug_component(eps_on)}_Q{q}_max_{_slug_component(optimize)}"


# Fixed canonical label -> short token for paths (HDF5, PNG stems)
FILENAME_SLUG_BY_CANONICAL: Dict[str, str] = {
    RELEASE_NSE_OBJECTIVE_OPTIMUM: "rel_nse_objopt",
    STORAGE_NSE_OBJECTIVE_OPTIMUM: "stg_nse_objopt",
    AVERAGE_NSE_OBJECTIVE_OPTIMUM: "avg_nse_objopt",
    BEST_AVERAGE_ALL: "best_avg_all",
    NORMALIZED_EQUAL_WEIGHT_MEAN_OPTIMUM: "norm_eq_mean_opt",
    MIN_EUCLIDEAN_L2_DISTANCE_TO_IP: "l2_dist_IP",
    MIN_WEIGHTED_CHEBYSHEV_DISTANCE_TO_IP: "cheb_dist_IP",
    MIN_MANHATTAN_L1_DISTANCE_TO_IP: "l1_dist_IP",
    MAX_CURVATURE_KNEE_POINT: "knee_max_curv",
    DIVERSE_SOLUTION_1: "diverse_1_maximin",
    DIVERSE_SOLUTION_2: "diverse_2_maximin",
    MAX_HV_CONTRIBUTION_2D: "hv_contrib_2d",
}


def _build_legacy_to_canonical() -> Dict[str, str]:
    from methods.config import OBJ_LABELS

    d: Dict[str, str] = {
        "Best Release NSE": RELEASE_NSE_OBJECTIVE_OPTIMUM,
        "Best Storage NSE": STORAGE_NSE_OBJECTIVE_OPTIMUM,
        "Best Average NSE": AVERAGE_NSE_OBJECTIVE_OPTIMUM,
        "Best Average (All Objectives)": NORMALIZED_EQUAL_WEIGHT_MEAN_OPTIMUM,
        "Best Average All": BEST_AVERAGE_ALL,
        "Compromise L2 (Euclidean)": MIN_EUCLIDEAN_L2_DISTANCE_TO_IP,
        "Tchebycheff L∞": MIN_WEIGHTED_CHEBYSHEV_DISTANCE_TO_IP,
        "Manhattan L1": MIN_MANHATTAN_L1_DISTANCE_TO_IP,
        "Knee (max curvature)": MAX_CURVATURE_KNEE_POINT,
        "ε-constraint Release NSE ≥ Q50": epsilon_constraint_label("Release NSE", 50, "Storage NSE"),
        "ε-constraint Release NSE ≥ Q80": epsilon_constraint_label("Release NSE", 80, "Storage NSE"),
        "Diverse #1 (FPS)": DIVERSE_SOLUTION_1,
        "Diverse #2 (FPS)": DIVERSE_SOLUTION_2,
        "Max HV Contribution": MAX_HV_CONTRIBUTION_2D,
    }
    for pretty in OBJ_LABELS.values():
        d[f"Best {pretty}"] = objective_optimum_label(pretty)
    return d


LEGACY_TO_CANONICAL: Mapping[str, str] = _build_legacy_to_canonical()


def normalize_pick_label(label: str) -> str:
    """Map legacy pick strings to canonical display names; unknown strings pass through."""
    s = str(label).strip()
    if s in LEGACY_TO_CANONICAL:
        return LEGACY_TO_CANONICAL[s]
    return s


def pick_filename_slug(label: str) -> str:
    """
    Short, stable token for HDF5/PNG paths. Uses FILENAME_SLUG_BY_CANONICAL for known picks;
    derives compact slugs for dynamic objective-optimum and epsilon-constraint labels.
    """
    s = normalize_pick_label(label)
    if s in FILENAME_SLUG_BY_CANONICAL:
        return FILENAME_SLUG_BY_CANONICAL[s]
    m = re.match(r"^(.+) objective optimum$", s)
    if m:
        return _slug_objective_optimum(m.group(1).strip())
    m = re.match(
        r"^Epsilon-constraint: feasible set (.+?) ≥ Q(\d+); maximize (.+)$",
        s,
    )
    if m:
        return _slug_epsilon_constraint(m.group(1).strip(), int(m.group(2)), m.group(3).strip())
    # Fallback: sanitize (shorter than full display string)
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return (t[:96] if len(t) > 96 else t) or "pick"


def iter_pick_lookup_labels(label: str) -> List[str]:
    """Unique label strings to try when matching highlight / highlight_adv / cand_map."""
    canon = normalize_pick_label(label)
    out: List[str] = []
    for k in (canon, str(label).strip()):
        if k and k not in out:
            out.append(k)
    for old, new in LEGACY_TO_CANONICAL.items():
        if new == canon and old not in out:
            out.append(old)
    return out


def resolve_cand_map_value(cand_map: Mapping[str, object], label: str) -> Optional[object]:
    """Return cand_map[value] for label or legacy alias; epsilon keys disambiguate by prefix if needed."""
    for k in iter_pick_lookup_labels(label):
        if k in cand_map:
            return cand_map[k]
    canon = normalize_pick_label(label)
    m = re.match(
        r"^Epsilon-constraint: feasible set (.+?) ≥ Q(\d+); maximize .+$",
        canon,
    )
    if m:
        prefix = f"Epsilon-constraint: feasible set {m.group(1)} ≥ Q{m.group(2)}; maximize"
        matches = [ck for ck in cand_map if str(ck).startswith(prefix)]
        if len(matches) == 1:
            return cand_map[matches[0]]
    return None


# Default pick lists (stage 1 / validation short list — no knee, Q80, HV unless included elsewhere)
DEFAULT_STAGE1_PICKS: List[str] = [
    RELEASE_NSE_OBJECTIVE_OPTIMUM,
    STORAGE_NSE_OBJECTIVE_OPTIMUM,
    AVERAGE_NSE_OBJECTIVE_OPTIMUM,
    BEST_AVERAGE_ALL,
    NORMALIZED_EQUAL_WEIGHT_MEAN_OPTIMUM,
    MIN_EUCLIDEAN_L2_DISTANCE_TO_IP,
    MIN_WEIGHTED_CHEBYSHEV_DISTANCE_TO_IP,
    MIN_MANHATTAN_L1_DISTANCE_TO_IP,
    epsilon_constraint_label("Release NSE", 50, "Storage NSE"),
    DIVERSE_SOLUTION_1,
    DIVERSE_SOLUTION_2,
]

# Full ordering for build_unified_picks (advanced / ordering only)
DESIRED_PICKS_ORDER: List[str] = [
    NORMALIZED_EQUAL_WEIGHT_MEAN_OPTIMUM,
    MIN_EUCLIDEAN_L2_DISTANCE_TO_IP,
    MIN_WEIGHTED_CHEBYSHEV_DISTANCE_TO_IP,
    MIN_MANHATTAN_L1_DISTANCE_TO_IP,
    MAX_CURVATURE_KNEE_POINT,
    epsilon_constraint_label("Release NSE", 50, "Storage NSE"),
    epsilon_constraint_label("Release NSE", 80, "Storage NSE"),
    DIVERSE_SOLUTION_1,
    DIVERSE_SOLUTION_2,
    MAX_HV_CONTRIBUTION_2D,
]
