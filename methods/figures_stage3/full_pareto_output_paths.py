"""
Output layout for full-Pareto HDF5 manifest figures under each Borg bundle folder.

Writes figures **14–23** (after validation figures 1–13), one subfolder per figure with a
descriptive name and a stable ``figN_*.png`` filename.
"""

from __future__ import annotations

import os
from typing import Dict, Final, List, Tuple

# (lookup_key, fig_num, subdir under bundle, png basename)
_FULL_PARETO_LAYOUT: Final[
    Tuple[Tuple[str, int, str, str], ...]
] = (
    (
        "multipanel_daily",
        14,
        "fig14_full_pareto_multipanel_daily_uncertainty",
        "fig14_multipanel_daily_uncertainty.png",
    ),
    (
        "multipanel_monthly",
        15,
        "fig15_full_pareto_multipanel_monthly_uncertainty",
        "fig15_multipanel_monthly_uncertainty.png",
    ),
    (
        "bias_surface",
        16,
        "fig16_full_pareto_release_bias_vs_inflow",
        "fig16_release_bias_vs_inflow.png",
    ),
    (
        "trenton_attribution",
        17,
        "fig17_full_pareto_trenton_lower_basin_mrf_attribution",
        "fig17_trenton_lower_basin_mrf_attribution.png",
    ),
    (
        "failure_alignment",
        18,
        "fig18_full_pareto_trenton_below_target_alignment",
        "fig18_trenton_below_target_alignment.png",
    ),
    (
        "reliability_storage",
        19,
        "fig19_full_pareto_reliability_vs_storage_nse",
        "fig19_reliability_vs_storage_nse.png",
    ),
    (
        "flow_regime_split",
        20,
        "fig20_full_pareto_reliability_by_inflow_regime",
        "fig20_reliability_by_inflow_regime.png",
    ),
    (
        "temporal_lag",
        21,
        "fig21_full_pareto_release_trenton_lag_correlation",
        "fig21_release_trenton_lag_correlation.png",
    ),
    (
        "policy_surface",
        22,
        "fig22_full_pareto_release_doy_storage_hexbin",
        "fig22_release_doy_storage_hexbin.png",
    ),
    (
        "extreme_event",
        23,
        "fig23_full_pareto_drought_window_case_study",
        "fig23_drought_window_case_study.png",
    ),
)

_BY_KEY: Dict[str, Tuple[int, str, str]] = {t[0]: (t[1], t[2], t[3]) for t in _FULL_PARETO_LAYOUT}


def full_pareto_png_path(bundle_root: str, key: str) -> str:
    """
    Return a path ``bundle_root/<subdir>/<png>``, creating ``subdir`` if needed.

    ``bundle_root`` is typically ``figures/borg_full_series`` (or an MRF-filtered sibling).
    """
    if key not in _BY_KEY:
        raise KeyError(
            f"unknown full-pareto figure key {key!r}; expected one of {sorted(_BY_KEY)}"
        )
    _n, sub, fn = _BY_KEY[key]
    d = os.path.join(bundle_root, sub)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fn)


def full_pareto_figure_index_rows() -> List[Tuple[int, str, str]]:
    """(fig_num, subdir, basename) for docs / debugging."""
    return [(t[1], t[2], t[3]) for t in _FULL_PARETO_LAYOUT]
