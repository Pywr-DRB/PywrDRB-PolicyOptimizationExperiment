#!/usr/bin/env python3
"""
Fig 1–2 style plots from ``eps_nondominated_<reservoir>.csv`` files produced by
``methods/analysis/mmborg_eps_nondominated_set.py`` (same ``load_results_with_metadata`` path as docstring examples).

Typical use after building three Borg bundles (full / regression / perfect)::

  for variant in full regression perfect; do
    python -m methods.analysis.mmborg_eps_nondominated_set --per-reservoir \\
      --out-dir outputs/pareto_eps_nondominated_\${variant} --borg-variant \"\$variant\" --print-counts
  done
  python -m methods.analysis.plot_eps_nondominated_figures --variants full regression perfect

Outputs under ``figures/eps_nondominated_<variant>/fig1_pareto_front_comparison/`` and
``.../fig2_parallel_axes/`` by default.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit(
        "Requires Python 3.10+ (same as 04_make_figures.py). "
        f"Got {sys.executable} — {sys.version.split()[0]}"
    )

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.config import (
    BASELINE_DIR_NAME,
    BASELINE_INFLOW_TAG,
    FIG_DIR,
    OBJ_FILTER_BOUNDS,
    OBJ_LABELS,
    SENSES_ALL,
    VAL_END,
    VAL_START,
    policy_type_options,
)
from methods.load.results import load_results_with_metadata
from methods.plotting.plot_parallel_axis import custom_parallel_coordinates
from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison
from methods.plotting.selection_unified import baseline_series_from_df
from methods.plotting.theme import POLICY_COMPARISON_COLORS
from methods.utils.policy_parameter_naming import safe_name

RE_EPS_CSV = re.compile(r"^eps_nondominated_(.+)\.csv$", re.IGNORECASE)

POLICY_LABELS = {"STARFIT": "STARFIT", "RBF": "RBF", "PWL": "PWL"}
FIG1_COLS = ["Release NSE", "Storage NSE"]
IDEAL_RS = [1.0, 1.0]


def _baseline_xy(reservoir: str, enabled: bool) -> tuple[float, float] | None:
    if not enabled:
        return None
    p = (
        Path(FIG_DIR)
        / f"{BASELINE_DIR_NAME}_{BASELINE_INFLOW_TAG}"
        / f"baseline_objectives_{reservoir}_{VAL_START}_to_{VAL_END}.csv"
    )
    if not p.is_file():
        return None
    try:
        bdf = pd.read_csv(p)
        bls = baseline_series_from_df(bdf, FIG1_COLS)
        r_nse = bls.get("Release NSE")
        s_nse = bls.get("Storage NSE")
        if np.isfinite(r_nse) and np.isfinite(s_nse):
            return (float(r_nse), float(s_nse))
    except Exception:
        return None
    return None


def _split_by_policy(obj_df: pd.DataFrame, meta_df: pd.DataFrame) -> tuple[list[pd.DataFrame], list[str]]:
    base = obj_df.copy()
    base["policy_number"] = base.index.astype(int)
    if meta_df is None or meta_df.empty or "moea_policy" not in meta_df.columns:
        return [base], ["ε-nondominated"]
    pol = meta_df["moea_policy"].astype(str).str.strip()
    obj_dfs: list[pd.DataFrame] = []
    labels: list[str] = []
    for p in policy_type_options:
        filter = pol.str.upper() == str(p).upper()
        if filter.any():
            obj_dfs.append(base.loc[filter].copy())
            labels.append(POLICY_LABELS.get(p, p))
    return (obj_dfs, labels) if obj_dfs else ([base], ["ε-nondominated"])


def _series_colors_for_labels(labels: list[str]) -> list[str]:
    """One matplotlib color per policy trace; uses ``POLICY_COMPARISON_COLORS``."""
    out = []
    for lab in labels:
        key = str(lab).strip().upper()
        out.append(POLICY_COMPARISON_COLORS.get(key, "#888888"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["full", "regression", "perfect"],
        metavar="NAME",
        help="Borg bundle names matching outputs/pareto_eps_nondominated_<NAME>/ (default: full regression perfect)",
    )
    ap.add_argument(
        "--in-root",
        type=Path,
        default=Path("outputs"),
        help="Parent of pareto_eps_nondominated_<variant>/ (default: ./outputs)",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("figures"),
        help="Parent of eps_nondominated_<variant>/... (default: ./figures)",
    )
    ap.add_argument(
        "--figures",
        nargs="+",
        choices=("1", "2"),
        default=["1", "2"],
        help="Which figure types to write (default: both)",
    )
    ap.add_argument(
        "--filter",
        action="store_true",
        help="Apply OBJ_FILTER_BOUNDS when loading (same as stage 1)",
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Add Pywr default NSE marker on Fig 1 when baseline_objectives CSV exists",
    )
    ap.add_argument(
        "--annotate-policy-number",
        action="store_true",
        help="Annotate each Fig 1 point with policy_number (row index in eps_nondominated CSV).",
    )
    args = ap.parse_args()

    want1 = "1" in args.figures
    want2 = "2" in args.figures
    in_root = args.in_root.resolve()
    out_root = args.out_root.resolve()
    obj_labels = OBJ_LABELS
    obj_cols = list(obj_labels.values())
    minmaxs_all = ["max" if SENSES_ALL[c] == "max" else "min" for c in obj_cols]

    for variant in args.variants:
        indir = in_root / f"pareto_eps_nondominated_{variant}"
        if not indir.is_dir():
            print(f"[plot_eps] skip (missing): {indir}", flush=True)
            continue
        if want1:
            (out_root / f"eps_nondominated_{variant}" / "fig1_pareto_front_comparison").mkdir(
                parents=True, exist_ok=True
            )
        if want2:
            (out_root / f"eps_nondominated_{variant}" / "fig2_parallel_axes").mkdir(
                parents=True, exist_ok=True
            )

        for csv_path in sorted(indir.glob("eps_nondominated_*.csv")):
            m = RE_EPS_CSV.match(csv_path.name)
            if not m:
                continue
            res_key = m.group(1)

            obj_df, _var, meta = load_results_with_metadata(
                str(csv_path),
                obj_labels=obj_labels,
                filter=args.filter,
                obj_bounds=OBJ_FILTER_BOUNDS,
            )
            if obj_df.empty:
                print(f"[plot_eps] skip empty: {csv_path}", flush=True)
                continue

            obj_dfs, labels = _split_by_policy(obj_df, meta)
            sn = safe_name(res_key)
            bp = _baseline_xy(res_key, args.baseline)

            if want1:
                f1 = (
                    out_root
                    / f"eps_nondominated_{variant}"
                    / "fig1_pareto_front_comparison"
                    / f"{sn}.png"
                )
                plot_pareto_front_comparison(
                    obj_dfs,
                    labels,
                    obj_cols=FIG1_COLS,
                    ideal=IDEAL_RS,
                    title=f"ε-nondominated ({variant}) — {res_key}",
                    fname=str(f1),
                    baseline_point=bp,
                    series_colors=_series_colors_for_labels(labels),
                    annotate_id_col="policy_number" if args.annotate_policy_number else None,
                )
                print(f"[plot_eps] Fig1 -> {f1}", flush=True)

            if want2:
                plot_df = obj_df.copy()
                if meta is not None and not meta.empty and "moea_policy" in meta.columns:
                    plot_df["moea_policy"] = meta["moea_policy"].astype(str).str.strip()
                    cats = pd.unique(plot_df["moea_policy"].astype(str))
                    color_dict = {
                        str(c).strip(): POLICY_COMPARISON_COLORS.get(
                            str(c).strip().upper(), "#888888"
                        )
                        for c in cats
                    }
                    cat_col = "moea_policy"
                else:
                    plot_df["policy"] = "ε-nondominated"
                    cat_col = "policy"
                    color_dict = {"ε-nondominated": "0.35"}

                f2 = (
                    out_root
                    / f"eps_nondominated_{variant}"
                    / "fig2_parallel_axes"
                    / f"fig2_eps_{sn}.png"
                )
                custom_parallel_coordinates(
                    objs=plot_df,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction="top",
                    minmaxs=minmaxs_all,
                    color_by_categorical=cat_col,
                    color_dict_categorical=color_dict,
                    fname=str(f2),
                    figsize=(12, 7.5),
                    alpha_base=0.85,
                    bottom_pad=0.24,
                    legend_pad=0.1,
                )
                print(f"[plot_eps] Fig2 -> {f2}", flush=True)

    print(f"[plot_eps] done (under {out_root})", flush=True)


if __name__ == "__main__":
    main()
