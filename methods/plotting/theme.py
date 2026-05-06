"""
Unified plotting theme: policy colors, advanced Pareto pick colors, and time-series styles.

Single module for policy colors, advanced pick colors, and time-series styles (import this instead of legacy shims).
"""

from __future__ import annotations

import os

from methods.plotting.pick_labels import (
    AVERAGE_NSE_OBJECTIVE_OPTIMUM,
    BEST_AVERAGE_ALL,
    DESIRED_PICKS_ORDER,
    DIVERSE_SOLUTION_1,
    DIVERSE_SOLUTION_2,
    MAX_CURVATURE_KNEE_POINT,
    MAX_HV_CONTRIBUTION_2D,
    MIN_EUCLIDEAN_L2_DISTANCE_TO_IP,
    MIN_MANHATTAN_L1_DISTANCE_TO_IP,
    MIN_WEIGHTED_CHEBYSHEV_DISTANCE_TO_IP,
    NORMALIZED_EQUAL_WEIGHT_MEAN_OPTIMUM,
    RELEASE_NSE_OBJECTIVE_OPTIMUM,
    STORAGE_NSE_OBJECTIVE_OPTIMUM,
    epsilon_constraint_label,
)

# --- Policy / MOEA ---
policy_type_colors = {
    "PWL": "blue",
    "RBF": "orange",
    "STARFIT": "green",
    "Baseline": "black",
}

# Borg / MOEA policy hues for **comparison** plots (Fig 1 Pareto overlay, ε-nondominated plots).
# STARFIT=blue, RBF=orange, PWL=green — stable across facets and independent of how many policies are present.
POLICY_COMPARISON_COLORS = {
    "STARFIT": "blue",
    "RBF": "orange",
    "PWL": "green",
}

# Footnote for parallel-axis plots using ``highlight_adv`` (see selection_unified.py).
ADVANCED_PICKS_CITATION = (
    "Advanced picks: L1/L2/Chebyshev distance to the ideal point (IP), epsilon-constraint selections, "
    "maximum-curvature knee, farthest-point diversity in normalized objective space, "
    "and 2-D hypervolume contribution; see Deb (2001) Multi-Objective Optimization Using "
    "Evolutionary Algorithms (Wiley)."
)

_EPS_Q50 = epsilon_constraint_label("Release NSE", 50, "Storage NSE")
_EPS_Q80 = epsilon_constraint_label("Release NSE", 80, "Storage NSE")

ADVANCED_COLORS = {
    NORMALIZED_EQUAL_WEIGHT_MEAN_OPTIMUM: "#1f77b4",
    "Other": "#d3d3d3",
    MIN_EUCLIDEAN_L2_DISTANCE_TO_IP: "#7b6cff",
    MIN_WEIGHTED_CHEBYSHEV_DISTANCE_TO_IP: "#c266ff",
    MIN_MANHATTAN_L1_DISTANCE_TO_IP: "#ff66c4",
    MAX_CURVATURE_KNEE_POINT: "#ff914d",
    _EPS_Q50: "#00c2a8",
    _EPS_Q80: "#008eaa",
    DIVERSE_SOLUTION_1: "#8bd3dd",
    DIVERSE_SOLUTION_2: "#a0e7e5",
    MAX_HV_CONTRIBUTION_2D: "#ffd166",
}

# Footnote for parallel-axis unified selection (scalarization + literature picks on one panel).
COMBINED_SELECTION_FOOTNOTE = (
    "Highlighted lines are nondominated solutions chosen under multiple criteria on the same plot: "
    "scalarization-style picks (best release NSE, storage NSE, mean of the two, min–max scaled compromise) "
    "and literature-based picks (L1/L2/Chebyshev distance to the IP, ε-constraints, knee, diversity, etc.). "
    "When one solution satisfies more than one criterion, its label lists both. "
    + ADVANCED_PICKS_CITATION
)


def color_dict_for_selection_parplot(
    present_labels: list[str],
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build a color map for parallel-axis selection: extends ``combined_selection_color_dict()`` with
    merged labels ``\"A · B\"`` (reuse color from the first matching component).
    """
    out = dict(base or combined_selection_color_dict())
    fallback = out.get("Other", "#d3d3d3")
    for lab in present_labels:
        if not lab or lab == "Other" or lab in out:
            continue
        if " · " in lab:
            color = None
            for part in (p.strip() for p in lab.split("·")):
                if part in out:
                    color = out[part]
                    break
            out[lab] = color if color is not None else fallback
        else:
            out.setdefault(lab, fallback)
    return out


def combined_selection_color_dict() -> dict[str, str]:
    """
    Ordered palette for parallel-axis combined selection: legacy single-objective / compromise picks first,
    then literature-grounded picks (``DESIRED_PICKS_ORDER``), then any remaining advanced colors.
    """
    legacy = {
        RELEASE_NSE_OBJECTIVE_OPTIMUM: "red",
        STORAGE_NSE_OBJECTIVE_OPTIMUM: "green",
        AVERAGE_NSE_OBJECTIVE_OPTIMUM: "purple",
        BEST_AVERAGE_ALL: "blue",
    }
    out: dict[str, str] = {}
    for k, v in legacy.items():
        out[k] = v
    for k in DESIRED_PICKS_ORDER:
        if k in ADVANCED_COLORS:
            out.setdefault(k, ADVANCED_COLORS[k])
    for k, v in ADVANCED_COLORS.items():
        out.setdefault(k, v)
    out["Other"] = "#d3d3d3"
    return out


# --- Pywr parametric mode (flow prediction) ---
PARAMETRIC_MODE_COLORS = {
    "perfect_foresight": "#ff7f0e",
    "regression_disagg": "#17becf",
    "gage_flow": "#8c564b",
}


def get_parametric_mode() -> str:
    return os.environ.get("CEE_PYWR_FLOW_PREDICTION_MODE", "regression_disagg").strip() or "regression_disagg"


def get_parametric_color(flow_mode: str | None = None) -> str:
    mode = (flow_mode or get_parametric_mode()).strip()
    return PARAMETRIC_MODE_COLORS.get(mode, PARAMETRIC_MODE_COLORS["regression_disagg"])


SERIES_COLORS = {
    "observed": "black",
    "parametric": get_parametric_color(),
    "default": "#1f77b4",
    "independent": "#9467bd",
}

SERIES_LINESTYLES = {
    "observed": "--",
    "parametric": "-",
    "default": "-",
    "independent": "-",
}

SERIES_LINEWIDTHS = {
    "observed": 1.4,
    "parametric": 1.6,
    "default": 1.6,
    "independent": 1.6,
}
