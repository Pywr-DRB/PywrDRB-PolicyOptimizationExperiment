"""Stage 1 figure pipeline (figures 1-6) for optimization/postprocess outputs.

This module loads filtered Borg tables, computes focal selections, optionally
runs/caches independent and Pywr simulations for selected policies, and writes
the primary figure set used before validation-stage plots.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import re
import os
import time
import tempfile
import pywrdrb
from pywrdrb.utils.dates import model_date_ranges
warnings.filterwarnings("ignore")

# ---------------- config & project imports ----------------
from methods.config import (
    OBJ_LABELS, OBJ_FILTER_BOUNDS,
    reservoir_options, policy_type_options,
    FIG_DIR, PROCESSED_DATA_DIR,
    reservoir_capacity, n_rbfs, n_rbf_inputs, n_segments, n_pwl_inputs,
    BASELINE_DIR_NAME, BASELINE_INFLOW_TAG, VAL_START, VAL_END,
    get_pywr_work_dir,
)

from methods.reservoir.model import Reservoir
from methods.load.results import load_results
from methods.load.observations import get_observational_training_data

from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison
from methods.plotting.plot_parallel_axis import custom_parallel_coordinates
from methods.plotting.plot_release_storage_9panel import plot_release_storage_9panel
from methods.plotting.plot_reservoir_storage_release_distributions import (
    plot_storage_release_distributions_independent_vs_pywr_split,
)
from methods.plotting.starfit_nor import try_compute_starfit_nor_pct_by_doy
from methods.plotting.theme import (
    COMBINED_SELECTION_FOOTNOTE,
    POLICY_COMPARISON_COLORS,
    color_dict_for_selection_parplot,
)
from methods.plotting.selection_unified import baseline_series_from_df
from methods.postprocess.pywr_output_metadata import normalize_borg_row_label
from methods.postprocess.compute_baseline_metrics import compute_baseline_objectives_for_reservoir
from methods.postprocess.pywr_parametric_run import (
    parametric_hdf5_stem,
    run_pywr_parametric_multi,
    parametric_result_from_h5_path,
)
from methods.plotting.selection_utils import (
    apply_combined_selection_column,
    compute_and_apply_advanced_highlights,
)
from methods.plotting.plot_policy_surfaces_v2 import save_policy_figure6_v2
from methods.plotting.pick_labels import (
    AVERAGE_NSE_OBJECTIVE_OPTIMUM,
    BEST_AVERAGE_ALL,
    DEFAULT_STAGE1_PICKS,
    DESIRED_PICKS_ORDER,
    RELEASE_NSE_OBJECTIVE_OPTIMUM,
    STORAGE_NSE_OBJECTIVE_OPTIMUM,
    iter_pick_lookup_labels,
    normalize_pick_label,
    pick_filename_slug,
    resolve_cand_map_value,
)
from methods.utils.policy_parameter_naming import (
    safe_name, get_param_names_for_policy, rename_vars_with_param_names,
    print_params_flat, print_params_pretty, has_solutions, reservoir_has_any
)
from methods.borg_paths import resolve_borg_moea_csv_path, resolve_figure_root

_DEFAULT_SERIES_CACHE: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}
_BASELINE_OBJ_CACHE: dict[str, pd.DataFrame] = {}


def _resolve_default_hdf5_path(tag: str = "1983-10-01_2023-12-31_pub_nhmv10_BC_withObsScaled") -> Path:
    if os.environ.get("CEE_DEFAULT_HDF5", "").strip():
        p = Path(os.environ["CEE_DEFAULT_HDF5"]).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2]
        out_dir = Path(os.environ.get("DRB_OUTPUT_DIR", str(root / "pywr_data")))
        cache = Path(os.environ.get("DRB_DEFAULT_CACHE", str(out_dir / "_pywr_default_cache")))
        p = cache / f"output_default_{tag}.hdf5"
    if not p.is_file():
        raise FileNotFoundError(f"Default Pywr HDF5 not found: {p}")
    return p


def _load_default_release_storage_series(reservoir_name: str):
    """Load cached Pywr default release/storage from baseline HDF5 via standard loader."""
    h5 = _resolve_default_hdf5_path()
    key = (str(h5), reservoir_name)
    if key not in _DEFAULT_SERIES_CACHE:
        result = parametric_result_from_h5_path(
            str(h5), {reservoir_name: "STARFIT"}, scenario_id=0, fetch_prompton_nwis=False
        )
        _DEFAULT_SERIES_CACHE[key] = result["by_res"][reservoir_name]
    rel, sto = _DEFAULT_SERIES_CACHE[key]
    return rel.rename("default_release"), sto.rename("default_storage")


def _pywr_release_series_for_reservoir(
    pywr_data: pywrdrb.Data,
    kP: str,
    scenario_id: int,
    reservoir_name: str,
    *,
    log_tag: str,
) -> pd.Series | None:
    """
    Simulated release from Pywr HDF5 for comparison plots.

    ``res_release`` (outflow + spill at the reservoir) is the complete simulated
    release for modeled reservoirs. ``reservoir_downstream_gage`` is often a subset
    of reservoirs (e.g. missing when there is no ``link_<gage>`` recorder in the
    output), so it is only used if ``res_release`` does not carry the column.
    """
    rs = getattr(pywr_data, "res_release", None)
    if rs is not None:
        try:
            df_r = rs[kP][scenario_id]
        except (KeyError, TypeError, IndexError):
            df_r = None
        if df_r is not None and reservoir_name in getattr(df_r, "columns", pd.Index([])):
            return df_r[reservoir_name].astype(float)

    try:
        df_g = pywr_data.reservoir_downstream_gage[kP][scenario_id]
    except (KeyError, TypeError, IndexError, AttributeError):
        df_g = None
    if df_g is not None and reservoir_name in getattr(df_g, "columns", pd.Index([])):
        print(
            f"[{log_tag}] {reservoir_name} scenario {scenario_id}: "
            "`res_release` missing column; using `reservoir_downstream_gage`.",
            flush=True,
        )
        return df_g[reservoir_name].astype(float)

    print(
        f"[{log_tag}] Skip {reservoir_name} scenario {scenario_id}: "
        "no column in `res_release` or `reservoir_downstream_gage`.",
        flush=True,
    )
    return None


def _pywr_flow_prediction_mode() -> str:
    """ModelBuilder ``options['flow_prediction_mode']``; override with CEE_PYWR_FLOW_PREDICTION_MODE."""
    v = os.environ.get("CEE_PYWR_FLOW_PREDICTION_MODE", "").strip()
    return v if v else "regression_disagg"


def _flow_mode_from_hdf5_stem(stem: str) -> str:
    """Infer flow mode from cached Pywr HDF5 stem for truthful figure labels."""
    s = str(stem)
    if "perfect_foresight" in s:
        return "perfect_foresight"
    if "regression_disagg" in s:
        return "regression_disagg"
    if "gage_flow" in s:
        return "gage_flow"
    return _pywr_flow_prediction_mode()


def _idxmin_safe(series: pd.Series):
    """Return idxmin while handling all-NA/non-finite rows safely."""
    if series is None or len(series) == 0:
        return None
    ser = pd.to_numeric(series, errors="coerce")
    ser = ser.replace([np.inf, -np.inf], np.nan)
    if ser.notna().any():
        return ser.idxmin()
    return series.index[0]
# ---------------- helpers ----------------
# def safe_name(label: str) -> str:
#     """Turn any label into a filesystem-friendly token."""
#     s = re.sub(r'[^A-Za-z0-9._-]+', '_', str(label))
#     s = re.sub(r'_+', '_', s).strip('_')
#     return s if s else "pick"

# def get_param_names_for_policy(policy: str):
#     """Return ordered parameter names matching the CSV var order for each policy."""
#     policy = str(policy).upper()
#     if policy == "STARFIT":
#         return [
#             "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
#             "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
#             "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
#             "Release_c", "Release_p1", "Release_p2",
#         ]
#     if policy == "RBF":
#         labels = ["storage", "inflow", "doy"][:n_rbf_inputs]
#         names = []
#         # weights
#         for i in range(1, n_rbfs + 1):
#             names.append(f"w{i}")
#         # centers
#         for i in range(1, n_rbfs + 1):
#             for v in labels:
#                 names.append(f"c{i}_{v}")
#         # radii/scales
#         for i in range(1, n_rbfs + 1):
#             for v in labels:
#                 names.append(f"r{i}_{v}")
#         return names
#     if policy == "PWL":
#         names = []
#         block_labels = ["storage", "inflow", "day"][:n_pwl_inputs]
#         for lab in block_labels:
#             for k in range(1, n_segments):
#                 names.append(f"{lab}_x{k}")
#             for k in range(1, n_segments + 1):
#                 names.append(f"{lab}_theta{k}")
#         return names
#     raise ValueError(f"Unknown policy '{policy}'")

# def rename_vars_with_param_names(var_df: pd.DataFrame, policy_type: str) -> pd.DataFrame:
#     """Rename var1..varN to policy parameter names. Leaves extra columns as var* if any."""
#     if var_df is None or var_df.empty:
#         return var_df
#     out = var_df.copy()
#     # Identify var* columns in order
#     var_cols = [c for c in out.columns if c.lower().startswith("var")]
#     # Keep order as in CSV (var1, var2, ...)
#     var_cols_sorted = sorted(var_cols, key=lambda x: int(re.sub(r'[^0-9]', '', x) or 0))
#     names = get_param_names_for_policy(policy_type)
#     k = min(len(names), len(var_cols_sorted))
#     # Map first k vars to names; keep any extras as-is
#     rename_map = {var_cols_sorted[i]: names[i] for i in range(k)}
#     out.rename(columns=rename_map, inplace=True)
#     return out

# def print_params_flat(policy_type: str, params_1d):
#     """Flat index → name → value (works for all policies)."""
#     names = get_param_names_for_policy(policy_type)
#     assert len(params_1d) == len(names), f"Length mismatch: got {len(params_1d)} values, expected {len(names)}"
#     print(f"\n--- Parameters ({policy_type}) ---")
#     for i, (n, v) in enumerate(zip(names, params_1d)):
#         print(f"[{i:02d}] {n:16s} = {float(v): .6f}")

# def print_params_pretty(policy_type: str, params_1d):
#     """Grouped printing for RBF/PWL; STARFIT remains flat."""
#     policy = policy_type.upper()
#     names = get_param_names_for_policy(policy)
#     assert len(params_1d) == len(names), f"Length mismatch: got {len(params_1d)} values, expected {len(names)}"

#     if policy == "STARFIT":
#         print_params_flat(policy, params_1d)
#         return

#     if policy == "RBF":
#         print(f"\n--- Parameters (RBF) n_rbfs={n_rbfs}, n_inputs={n_rbf_inputs} ---")
#         idx = 0
#         print("Weights:")
#         for i in range(1, n_rbfs + 1):
#             print(f"  w{i} = {float(params_1d[idx]): .6f}")
#             idx += 1
#         print("Centers c[i, var]:")
#         for i in range(1, n_rbfs + 1):
#             row = []
#             for var in ["storage", "inflow", "doy"][:n_rbf_inputs]:
#                 row.append(float(params_1d[idx])); idx += 1
#             print(f"  c{i} = {row}")
#         print("Scales r[i, var]:")
#         for i in range(1, n_rbfs + 1):
#             row = []
#             for var in ["storage", "inflow", "doy"][:n_rbf_inputs]:
#                 row.append(float(params_1d[idx])); idx += 1
#             print(f"  r{i} = {row}")
#         return

#     if policy == "PWL":
#         print(f"\n--- Parameters (PWL) n_segments={n_segments}, n_inputs={n_pwl_inputs} ---")
#         per_block = 2 * n_segments - 1
#         blocks = ["storage", "inflow", "day"][:n_pwl_inputs]
#         for b, lab in enumerate(blocks):
#             block = params_1d[b*per_block:(b+1)*per_block]
#             xs     = block[:n_segments-1]
#             thetas = block[n_segments-1:]
#             print(f"{lab.capitalize()} block:")
#             for i, x in enumerate(xs, start=1):
#                 print(f"  x{i}     = {float(x): .6f}")
#             for i, th in enumerate(thetas, start=1):
#                 print(f"  theta{i} = {float(th): .6f}")
#         return

#     print_params_flat(policy, params_1d)

# def has_solutions(solution_objs, reservoir_name: str, policy_type: str) -> bool:
#     return (
#         reservoir_name in solution_objs and
#         policy_type in solution_objs[reservoir_name] and
#         solution_objs[reservoir_name][policy_type] is not None and
#         len(solution_objs[reservoir_name][policy_type]) > 0
#     )

# def reservoir_has_any(solution_objs, reservoir_name: str) -> bool:
#     d = solution_objs.get(reservoir_name, {})
#     return any((df is not None) and (len(df) > 0) for df in d.values())

def _params_for_row(var_df: pd.DataFrame, row_idx):
    """Row-safe accessor: prefer .loc by index label, fall back to .iloc if needed."""
    try:
        return var_df.loc[row_idx].values
    except Exception:
        return var_df.iloc[int(row_idx)].values

def summarize_ranges(solution_objs, cols):
    """Print ranges for *objectives*."""
    for res, pols in solution_objs.items():
        print(f"\n[RANGES] {res}")
        for pol, df in pols.items():
            print(f"  {pol}:")
            for c in cols:
                if c in df.columns:
                    v = pd.to_numeric(df[c], errors="coerce")
                    print(
                        f"    {c:24s} "
                        f"min={v.min():.4g} p25={v.quantile(0.25):.4g} "
                        f"med={v.median():.4g} p75={v.quantile(0.75):.4g} "
                        f"max={v.max():.4g} NaN={v.isna().sum()}"
                    )

def summarize_param_ranges(solution_vars: dict):
    """Print ranges for *decision variables* (renamed to parameter names)."""
    for res, pols in solution_vars.items():
        print(f"\n[PARAM RANGES] {res}")
        for pol, df in pols.items():
            if df is None or df.empty:
                continue
            print(f"  {pol}:")
            for c in df.columns:
                if not c.lower().startswith("obj"):  # only parameters (var/renamed), not objectives
                    v = pd.to_numeric(df[c], errors="coerce")
                    if v.notna().any():
                        print(
                            f"    {c:24s} "
                            f"min={v.min():.4g} p25={v.quantile(0.25):.4g} "
                            f"med={v.median():.4g} p75={v.quantile(0.75):.4g} "
                            f"max={v.max():.4g} NaN={v.isna().sum()}"
                        )

# ---------------- labels & plot settings ----------------
POLICY_TYPES = policy_type_options
RESERVOIR_NAMES = reservoir_options

reservoir_labels = {
    'beltzvilleCombined': 'Beltzville',
    'fewalter': 'FE Walter',
    'prompton': 'Prompton',
    'blueMarsh': 'Blue Marsh',
}
policy_labels = {'STARFIT': 'STARFIT', 'RBF': 'RBF', 'PWL': 'PWL'}
policy_colors = POLICY_COMPARISON_COLORS

# Direction per axis for plotting (match 3 objectives)
senses_all = {
    "Release NSE": "max",
    "Q20 Abs % Bias (Release)": "min",
    "Storage NSE": "max",
    "Q80 Abs % Bias (Storage)": "min",
}

DESIRED_PICKS = list(DEFAULT_STAGE1_PICKS)

REMAKE_PARALLEL_PLOTS = True
REMAKE_DYNAMICS_PLOTS = True
EXPORT_SELECTED_TIMESERIES_HDF5 = False
SELECTED_TIMESERIES_HDF5_NAME = "selected_solution_timeseries.h5"


def _env_override_remake(module_default: bool, env_name: str) -> bool:
    """If ``env_name`` is set, 0/false/off → False, 1/true/on → True; else ``module_default``."""
    v = os.environ.get(env_name, "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return module_default


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _figure_picks_from_env() -> list[str]:
    """Subset picks via ``CEE_DESIRED_PICKS`` (comma-separated labels). Empty → full ``DESIRED_PICKS``."""
    raw = os.environ.get("CEE_DESIRED_PICKS", "").strip()
    if not raw:
        return list(DESIRED_PICKS)
    out = [normalize_pick_label(p.strip()) for p in raw.split(",") if p.strip()]
    return out if out else list(DESIRED_PICKS)


def _figure_policies_from_env() -> list[str]:
    """Subset policies via ``CEE_FIGURE_POLICIES`` (comma-separated, e.g. ``STARFIT,RBF``). Empty → all."""
    raw = os.environ.get("CEE_FIGURE_POLICIES", "").strip()
    if not raw:
        return list(POLICY_TYPES)
    allowed = {str(p).upper() for p in POLICY_TYPES}
    out = []
    for tok in raw.split(","):
        t = tok.strip().upper()
        if t in allowed and t not in out:
            out.append(t)
    return out if out else list(POLICY_TYPES)


def _env_first_nonempty(*keys: str) -> str:
    for k in keys:
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return ""


def _fig4_reservoirs_from_env(all_names: list[str]) -> list[str]:
    """``CEE_FIG4_RESERVOIRS`` or legacy ``CEE_FIG3_RESERVOIRS`` (comma-separated). Empty → all ``all_names``."""
    raw = _env_first_nonempty("CEE_FIG4_RESERVOIRS", "CEE_FIG3_RESERVOIRS")
    if not raw:
        return list(all_names)
    want = {x.strip() for x in raw.split(",") if x.strip()}
    inter = [r for r in all_names if r in want]
    return inter if inter else list(all_names)


def _fig4_policies_from_env(base: list[str]) -> list[str]:
    """``CEE_FIG4_POLICIES`` or legacy ``CEE_FIG3_POLICIES`` — subset of ``base`` for Fig 4–6 (dynamics/surfaces)."""
    raw = _env_first_nonempty("CEE_FIG4_POLICIES", "CEE_FIG3_POLICIES")
    if not raw:
        return base
    want = {tok.strip().upper() for tok in raw.split(",") if tok.strip()}
    out = [p for p in base if p in want]
    return out if out else base


def _fig4_picks_from_env(base: list[str]) -> list[str]:
    """``CEE_FIG4_PICKS`` or legacy ``CEE_FIG3_PICKS`` — subset of focal picks for Fig 4–6."""
    raw = _env_first_nonempty("CEE_FIG4_PICKS", "CEE_FIG3_PICKS")
    if not raw:
        return base
    want = [normalize_pick_label(p.strip()) for p in raw.split(",") if p.strip()]
    inter = [p for p in base if p in want]
    return inter if inter else base


def _fig4_k_from_env() -> int | None:
    """``CEE_FIG4_K`` or legacy ``CEE_FIG3_K`` — only this solution index ``k`` (1-based per pick)."""
    raw = _env_first_nonempty("CEE_FIG4_K", "CEE_FIG3_K")
    if not raw:
        return None
    return int(raw)


def _eval_window_stage1() -> tuple[str, str]:
    """
    Evaluation window for Fig 4–5 (annual aggregation / 9-panel). Override with ``CEE_EVAL_START`` /
    ``CEE_EVAL_END`` (YYYY-MM-DD); filenames include this range.
    """
    s = os.environ.get("CEE_EVAL_START", "").strip() or "1980-01-01"
    e = os.environ.get("CEE_EVAL_END", "").strip() or "2018-12-31"
    return s, e


def _fig5_windows_stage1(
    default_start: str,
    default_end: str,
    reservoir_name: str | None = None,
) -> list[tuple[str, str, str]]:
    """
    Figure 5 windows: drought + normal slices by default.
    Override with ``CEE_FIG5_WINDOWS`` as:
      ``label:start:end,label2:start:end``.
    """
    raw = os.environ.get("CEE_FIG5_WINDOWS", "").strip()
    if raw:
        out = []
        for token in raw.split(","):
            t = token.strip()
            if not t:
                continue
            parts = [p.strip() for p in t.split(":")]
            if len(parts) != 3:
                continue
            lbl, s, e = parts
            out.append((lbl or "window", s, e))
        if out:
            return out
    if reservoir_name == "fewalter":
        return [
            ("zoom_2017_2019", "2017-01-01", "2019-12-31"),
        ]
    return [
        ("drought_2010_2012", "2010-01-01", "2012-12-31"),
        ("normal_2012_2014", "2012-01-01", "2014-12-31"),
    ]


def _eval_window_file_token(start: str, end: str) -> str:
    return f"{safe_name(str(start)[:10])}_to_{safe_name(str(end)[:10])}"


def _fig1_default_point_mode() -> str:
    """
    Pywr default baseline marker on Pareto scatter (Fig 1).

    ``CEE_FIG1_INCLUDE_BASELINE`` (alias ``CEE_FIG1_DEFAULT_POINT``): unset / ``both`` → two PNGs
    (``*_with_pywr_default.png``, ``*_no_pywr_default.png``); ``with`` / ``1`` → only with marker;
    ``without`` / ``0`` → only without (avoids scale skew when the default is far from the front).
    """
    raw = _env_first_nonempty("CEE_FIG1_INCLUDE_BASELINE", "CEE_FIG1_DEFAULT_POINT")
    v = raw.lower()
    if v in ("", "both", "all", "2", "two"):
        return "both"
    if v in ("1", "true", "yes", "on", "with", "include", "with_default", "only_with"):
        return "with_only"
    if v in ("0", "false", "no", "off", "without", "exclude", "omit", "no_default", "only_without"):
        return "without_only"
    return "both"


def _timing_tick(phase: str, t0: float) -> float:
    """Log elapsed wall seconds since ``t0``; return new reference time (for bash/Slurm logs)."""
    t1 = time.perf_counter()
    print(f"[timing] {phase}: {t1 - t0:.2f}s", flush=True)
    return t1

def get_pick_indices(solution_objs, solution_adv_maps, reservoir_name: str, policy_type: str, label: str):
    """
    Return a list of row indices for the requested label.
    Works with legacy 'highlight' and advanced picks from cand_map or 'highlight_adv'.
    Accepts legacy pick names (e.g. ``Best Release NSE``) via :func:`normalize_pick_label`.
    """
    out = []

    df = solution_objs.get(reservoir_name, {}).get(policy_type)
    cand_map = solution_adv_maps.get(reservoir_name, {}).get(policy_type, {}) or {}

    if df is not None and "highlight" in df.columns:
        for lbl in iter_pick_lookup_labels(label):
            out += df.index[df["highlight"] == lbl].tolist()

    val = resolve_cand_map_value(cand_map, label)
    if val is not None:
        if isinstance(val, (list, tuple, np.ndarray, pd.Index, pd.Series)):
            out += list(pd.Index(val))
        else:
            out.append(val)

    if df is not None and "highlight_adv" in df.columns:
        for lbl in iter_pick_lookup_labels(label):
            out += df.index[df["highlight_adv"] == lbl].tolist()

    # dedupe, preserve order
    seen, deduped = set(), []
    for idx in out:
        try:
            key = int(idx)
        except Exception:
            key = idx
        if key not in seen:
            seen.add(key)
            deduped.append(idx)
    return deduped

def _run_selected_solution_cache(
    solution_objs,
    solution_vars,
    solution_adv_maps,
    selected_labels: list[str],
    policy_types: list[str],
    *,
    skip_indie_runs: bool = False,
):
    """
    Run each unique selected solution once, then reuse across figure workflows.
    Cache key: (reservoir_name, policy_type, idx_row)

    If ``skip_indie_runs`` is True, only ``obs_cache`` and ``pick_index_map`` are filled
    (for Pareto/parallel/parameter-range figures); no independent simulator runs.
    """
    selected_labels = set(selected_labels)
    run_cache = {}
    obs_cache = {}
    pick_index_map = {}
    initial_storage_by_reservoir = {}
    shared_init_frac_raw = os.environ.get("CEE_SHARED_INITIAL_STORAGE_FRAC", "").strip()
    shared_init_frac = None
    if shared_init_frac_raw:
        try:
            shared_init_frac = float(shared_init_frac_raw)
            shared_init_frac = max(0.0, min(1.0, shared_init_frac))
        except Exception:
            print(
                f"[init] invalid CEE_SHARED_INITIAL_STORAGE_FRAC={shared_init_frac_raw!r}; "
                "ignoring explicit shared initial fraction.",
                flush=True,
            )
            shared_init_frac = None

    for reservoir_name in RESERVOIR_NAMES:
        pick_index_map.setdefault(reservoir_name, {})
        inflow_df, release_df, storage_df = get_observational_training_data(
            reservoir_name=reservoir_name,
            data_dir=PROCESSED_DATA_DIR,
            as_numpy=False,
            inflow_type='inflow_pub'
        )
        if shared_init_frac is not None:
            cap = float(reservoir_capacity[reservoir_name])
            s0_shared = shared_init_frac * cap
            initial_storage_by_reservoir[reservoir_name] = float(s0_shared)

        if inflow_df.empty or storage_df.empty:
            print(f"[cache] Skip {reservoir_name}: missing obs.")
            continue

        datetime = inflow_df.index
        inflow_obs = inflow_df.values
        release_obs = release_df.values if release_df is not None else None
        storage_obs = storage_df.values
        storage_obs_flat = np.asarray(storage_obs, dtype=float).flatten()
        finite_storage = storage_obs_flat[np.isfinite(storage_obs_flat)]
        initial_storage_obs = float(finite_storage[0]) if finite_storage.size else None
        if shared_init_frac is None and initial_storage_obs is None:
            print(
                f"[init] {reservoir_name}: no finite observed storage found; "
                "independent/Pywr defaults will be used unless explicit "
                "CEE_SHARED_INITIAL_STORAGE_FRAC is set.",
                flush=True,
            )
        obs_cache[reservoir_name] = {
            "datetime": datetime,
            "inflow": inflow_obs,
            "release": release_obs,
            "storage": storage_obs,
        }

        for policy_type in policy_types:
            pick_index_map[reservoir_name].setdefault(policy_type, {})
            if not has_solutions(solution_objs, reservoir_name, policy_type):
                continue
            var_df = solution_vars.get(reservoir_name, {}).get(policy_type)
            if var_df is None or var_df.empty:
                continue

            for pick_label in selected_labels:
                idxs = get_pick_indices(solution_objs, solution_adv_maps, reservoir_name, policy_type, pick_label)
                if not idxs:
                    continue
                pick_index_map[reservoir_name][policy_type][pick_label] = idxs

                if skip_indie_runs:
                    continue

                for idx_row in idxs:
                    cache_key = (reservoir_name, policy_type, idx_row)
                    if cache_key in run_cache:
                        continue
                    params = _params_for_row(var_df, idx_row)
                    reservoir = Reservoir(
                        inflow=inflow_obs, dates=datetime,
                        capacity=reservoir_capacity[reservoir_name],
                        policy_type=policy_type, policy_params=params,
                        initial_storage=initial_storage_by_reservoir.get(reservoir_name),
                        name=reservoir_name,
                    )
                    reservoir.run()
                    sim_storage = reservoir.storage_array.flatten()
                    sim_release = (reservoir.release_array + reservoir.spill_array).flatten()
                    run_cache[cache_key] = {
                        "params": np.asarray(params, dtype=float).flatten(),
                        "sim_storage": sim_storage,
                        "sim_release": sim_release,
                    }
    if initial_storage_by_reservoir:
        _pairs = ", ".join(
            f"{r}={v:.2f} MG" for r, v in sorted(initial_storage_by_reservoir.items())
        )
        if shared_init_frac is not None:
            print(
                f"[init] explicit shared initial storage fraction={shared_init_frac:.3f} "
                f"applied by reservoir: {_pairs}",
                flush=True,
            )
        else:
            print(
                f"[init] shared initial storage override map: {_pairs}",
                flush=True,
            )
    else:
        print(
            "[init] no explicit shared initial storage override; model defaults remain active.",
            flush=True,
        )

    return run_cache, obs_cache, pick_index_map, initial_storage_by_reservoir

def _export_selected_timeseries_hdf5(fig_root, run_cache, obs_cache, pick_index_map):
    """
    Export selected pick time series into one HDF5 for sharing/reanalysis.
    """
    out_h5 = Path(fig_root) / SELECTED_TIMESERIES_HDF5_NAME
    with pd.HDFStore(str(out_h5), mode="w", complevel=5, complib="zlib") as store:
        for reservoir_name, pols in pick_index_map.items():
            if reservoir_name not in obs_cache:
                continue
            dt_index = obs_cache[reservoir_name]["datetime"]
            obs_storage = obs_cache[reservoir_name]["storage"].flatten()
            obs_release_arr = obs_cache[reservoir_name]["release"]
            obs_release = obs_release_arr.flatten() if obs_release_arr is not None else None

            for policy_type, picks in pols.items():
                for pick_label, idxs in picks.items():
                    for k, idx_row in enumerate(idxs, start=1):
                        cache_key = (reservoir_name, policy_type, idx_row)
                        if cache_key not in run_cache:
                            continue
                        run = run_cache[cache_key]
                        group = (
                            f"/selected/{pick_filename_slug(pick_label)}/"
                            f"{safe_name(policy_type)}/{safe_name(reservoir_name)}/k{k}"
                        )
                        df = pd.DataFrame(
                            {
                                "sim_storage": run["sim_storage"],
                                "sim_release": run["sim_release"],
                                "obs_storage": obs_storage,
                            },
                            index=dt_index,
                        )
                        if obs_release is not None:
                            df["obs_release"] = obs_release
                        store.put(f"{group}/timeseries", df, format="table")
                        store.put(
                            f"{group}/params",
                            pd.DataFrame({"param": run["params"]}),
                            format="table",
                        )
    print(f"[export] Saved selected solution time series: {out_h5}")

def _ensure_pywr_pick_hdf5(
    *,
    fig_root,
    policy_type: str,
    pick_label: str,
    pick_index_map: dict,
    run_cache: dict,
    pywr_inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    flow_prediction_mode: str | None = None,
    reservoirs_to_include: list[str] | None = None,
    overwrite: bool = False,
    tmp_dir: Path | None = None,
    solution_vars: dict | None = None,
    initial_storage_by_reservoir: dict[str, float] | None = None,
):
    """
    One Pywr-DRB simulation per (policy_type, pick_label).

    ``release_policy_dict`` includes every reservoir in ``reservoirs_to_include`` with that
    reservoir's *first* selected parameter row for this pick (FE Walter, Beltzville, Blue Marsh,
    Prompton in one run — lower-basin reservoirs are not in series, so one model run suffices).

    OutputRecorder writes a *single* standard pywrdrb HDF5 (scenario index 0) — no post-merge of
    multiple runs. Load results with ``pywrdrb.Data(...).load_output()`` as usual.

    ``flow_prediction_mode`` defaults from ``_pywr_flow_prediction_mode()`` (env
    ``CEE_PYWR_FLOW_PREDICTION_MODE``, else ``regression_disagg``).

    Optional ``solution_vars`` enables storing Borg dataframe iloc positions alongside row labels
    in ``*_cee_meta.json`` and HDF5 ``cee_meta`` attributes.

    HDF5 is written under ``CEE_PYWR_WORK_DIR`` (same path convention as stage 2 / validation).
    Filename includes ``flow_prediction_mode``, inflow type, and a Borg-bundle tag from env
    (``CEE_FIG_SUBDIR``, seed, MRF flags) via :func:`methods.postprocess.pywr_parametric_run.parametric_hdf5_stem`.
    """
    del fig_root  # figure root does not select the cache path (only env tags do)

    work_dir = str(tmp_dir) if tmp_dir is not None else get_pywr_work_dir()
    os.makedirs(work_dir, exist_ok=True)

    if pywr_inflow_type not in model_date_ranges:
        raise KeyError(f"Unknown pywr_inflow_type={pywr_inflow_type!r} in model_date_ranges")
    pywr_start, pywr_end = model_date_ranges[pywr_inflow_type]
    if reservoirs_to_include is None:
        reservoirs_to_include = list(RESERVOIR_NAMES)
    if flow_prediction_mode is None:
        flow_prediction_mode = _pywr_flow_prediction_mode()

    stem_base = parametric_hdf5_stem(
        policy_type, pick_label, flow_prediction_mode, pywr_inflow_type
    )
    out_h5 = Path(work_dir) / f"{stem_base}.hdf5"

    if len(reservoirs_to_include) == 0:
        return out_h5

    release_policy_dict = {}
    row_index_labels_by_reservoir: dict = {}
    row_indices_by_reservoir: dict = {}
    for reservoir_name in reservoirs_to_include:
        idxs = pick_index_map.get(reservoir_name, {}).get(policy_type, {}).get(pick_label, [])
        if len(idxs) < 1:
            raise RuntimeError(
                f"[pywr] Missing idx for reservoir={reservoir_name} policy={policy_type} pick={pick_label}"
            )
        if len(idxs) > 1:
            print(
                f"[pywr] Using first index only for {reservoir_name}/{pick_label} "
                f"({len(idxs)} indices in pick list; single combined run)"
            )
        idx_row = idxs[0]
        cache_key = (reservoir_name, policy_type, idx_row)
        if cache_key not in run_cache:
            raise RuntimeError(f"[pywr] Missing independent cache row for {cache_key}")
        params = run_cache[cache_key]["params"]
        release_policy_dict[reservoir_name] = {
            "class_type": "ParametricReservoirRelease",
            "policy_type": policy_type,
            "policy_id": "inline",
            "params": ",".join(str(x) for x in np.asarray(params, float).tolist()),
        }
        row_index_labels_by_reservoir[reservoir_name] = normalize_borg_row_label(idx_row)
        if solution_vars is not None:
            var_df = solution_vars.get(reservoir_name, {}).get(policy_type)
            if var_df is not None:
                try:
                    loc = var_df.index.get_loc(idx_row)
                    if isinstance(loc, slice):
                        row_indices_by_reservoir[reservoir_name] = int(loc.start)
                    elif isinstance(loc, (np.ndarray, list)):
                        row_indices_by_reservoir[reservoir_name] = int(loc[0]) if len(loc) else None
                    else:
                        row_indices_by_reservoir[reservoir_name] = int(loc)
                except Exception:
                    pass

    print(
        f"[pywr] pick HDF5: parametric reservoirs={sorted(release_policy_dict.keys())} "
        f"(n={len(release_policy_dict)}) | policy_type={policy_type} | pick={pick_label!r} | stem={stem_base}",
        flush=True,
    )

    pywr_run_metadata = {
        "pick_label": pick_label,
        "policy_type": policy_type,
        "row_index_labels_by_reservoir": row_index_labels_by_reservoir,
        "row_indices_by_reservoir": row_indices_by_reservoir if row_indices_by_reservoir else None,
    }
    t0 = time.perf_counter()
    extra_model_options = None
    if initial_storage_by_reservoir:
        initial_volume_frac_dict = {}
        for reservoir_name in reservoirs_to_include:
            s0 = initial_storage_by_reservoir.get(reservoir_name)
            cap = float(reservoir_capacity[reservoir_name])
            if s0 is None or not np.isfinite(float(s0)) or cap <= 0:
                continue
            frac = max(0.0, min(1.0, float(s0) / cap))
            initial_volume_frac_dict[reservoir_name] = frac
        if initial_volume_frac_dict:
            extra_model_options = {"initial_volume_frac_dict": initial_volume_frac_dict}
            _frac_msg = ", ".join(
                f"{r}={f:.4f}" for r, f in sorted(initial_volume_frac_dict.items())
            )
            _mg_msg = ", ".join(
                f"{r}={float(initial_storage_by_reservoir[r]):.2f} MG"
                for r in sorted(initial_volume_frac_dict.keys())
            )
            print(
                f"[init] applying shared initial storage to Pywr "
                f"({policy_type}/{pick_label}): {_mg_msg} | fractions: {_frac_msg}",
                flush=True,
            )
        else:
            print(
                f"[init] no valid shared initial storage fractions for Pywr "
                f"({policy_type}/{pick_label}).",
                flush=True,
            )

    run_pywr_parametric_multi(
        release_policy_dict,
        str(pywr_start),
        str(pywr_end),
        pywr_inflow_type,
        work_dir,
        stem_base,
        flow_prediction_mode,
        pywr_run_metadata=pywr_run_metadata,
        extra_model_options=extra_model_options,
        force_rerun=overwrite,
    )
    print(
        f"[timing] pywr parametric multi {policy_type} / {pick_label!r}: "
        f"{time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return out_h5


def load_filtered_borg_solution_tables(
    reservoir_names: list[str],
    policy_types: list[str],
    *,
    verbose: bool = True,
    borg_seed: int | None = None,
    borg_mrf_filtered: bool | None = None,
    borg_mrf_filter_tag: str | None = None,
) -> tuple[dict, dict, dict, dict]:
    """
    Load filtered Borg MOEA objective tables, renamed decision variables, and advanced
    highlight maps — same data as the stage1 CSV load block (for reuse by simulation CLIs).

    Optional ``borg_*`` arguments are forwarded to :func:`methods.borg_paths.resolve_borg_moea_csv_path`
    so callers can load ``full`` vs ``_mrffiltered_regression`` vs ``_mrffiltered_perfect`` CSVs without
    mutating global environment (see :func:`methods.borg_paths.borg_variant_resolve_kwargs`).
    """
    solution_objs: dict = {}
    solution_vars: dict = {}
    solution_adv_maps: dict = {}
    solution_adv_cands: dict = {}

    obj_labels = OBJ_LABELS
    obj_cols = list(obj_labels.values())

    for reservoir_name in reservoir_names:
        solution_objs[reservoir_name] = {}
        solution_vars[reservoir_name] = {}

        for policy_type in policy_types:
            fname = resolve_borg_moea_csv_path(
                policy_type,
                reservoir_name,
                seed=borg_seed,
                mrf_filtered=borg_mrf_filtered,
                mrf_filter_tag=borg_mrf_filter_tag,
            )

            try:
                obj_df_raw, var_df_raw = load_results(
                    fname, obj_labels=obj_labels, filter=False, obj_bounds=None
                )
                if verbose:
                    print(f"[RAW] {reservoir_name}/{policy_type}: {len(obj_df_raw)} rows before filter")
            except Exception as e:
                if verbose:
                    print(f"[WARN] Could not load RAW for {reservoir_name}/{policy_type}: {e}")
                obj_df_raw, var_df_raw = pd.DataFrame(), pd.DataFrame()

            try:
                obj_df, var_df = load_results(
                    fname, obj_labels=obj_labels, filter=True, obj_bounds=OBJ_FILTER_BOUNDS
                )
                if verbose:
                    print(f"[FLT] {reservoir_name}/{policy_type}: {len(obj_df)} rows after filter")
            except Exception as e:
                if verbose:
                    print(f"[WARN] Could not load FILTERED for {reservoir_name}/{policy_type}: {e}")
                obj_df, var_df = pd.DataFrame(), pd.DataFrame()

            if len(obj_df) == 0:
                if verbose:
                    print(
                        f"Warning: No solutions found for {policy_type} with {reservoir_name}. Skipping."
                    )
                continue

            var_df_named = rename_vars_with_param_names(var_df, policy_type)

            solution_objs[reservoir_name][policy_type] = obj_df
            solution_vars[reservoir_name][policy_type] = var_df_named

            idx_best_release = obj_df["Release NSE"].idxmax()
            idx_best_storage = obj_df["Storage NSE"].idxmax()
            idx_best_average = obj_df[["Release NSE", "Storage NSE"]].mean(axis=1).idxmax()

            if verbose:
                print(f"Stats for {policy_type} {reservoir_name}:")
                print(
                    f"  {RELEASE_NSE_OBJECTIVE_OPTIMUM}: {idx_best_release} = "
                    f"{obj_df['Release NSE'][idx_best_release]}"
                )
                print(
                    f"  {STORAGE_NSE_OBJECTIVE_OPTIMUM}: {idx_best_storage} = "
                    f"{obj_df['Storage NSE'][idx_best_storage]}"
                )
                print(
                    f"  {AVERAGE_NSE_OBJECTIVE_OPTIMUM}: {idx_best_average} = "
                    f"{obj_df[['Release NSE','Storage NSE']].mean(axis=1)[idx_best_average]}"
                )

            min_obj_df = obj_df.copy()
            min_obj_df["Release NSE"] = -min_obj_df["Release NSE"]
            min_obj_df["Storage NSE"] = -min_obj_df["Storage NSE"]
            scaled_min_obj_df = (min_obj_df - min_obj_df.min()) / (min_obj_df.max() - min_obj_df.min())
            idx_best_all_avg = _idxmin_safe(scaled_min_obj_df.mean(axis=1))

            highlight_label_dict = {
                idx_best_release: RELEASE_NSE_OBJECTIVE_OPTIMUM,
                idx_best_storage: STORAGE_NSE_OBJECTIVE_OPTIMUM,
                idx_best_average: AVERAGE_NSE_OBJECTIVE_OPTIMUM,
                idx_best_all_avg: BEST_AVERAGE_ALL,
            }
            obj_df["highlight"] = [highlight_label_dict.get(idx, "Other") for idx in obj_df.index]

            obj_df_aug, cand_df, cand_map = compute_and_apply_advanced_highlights(
                obj_df,
                objectives=obj_cols,
                senses=senses_all,
                bounds=OBJ_FILTER_BOUNDS,
                eps_qs=(0.5, 0.8),
                add_k_diverse=2,
                include_hv=False,
                out_label_col="highlight_adv",
            )

            solution_objs[reservoir_name][policy_type] = obj_df_aug
            solution_adv_maps.setdefault(reservoir_name, {})[policy_type] = cand_map
            solution_adv_cands.setdefault(reservoir_name, {})[policy_type] = cand_df

    return solution_objs, solution_vars, solution_adv_maps, solution_adv_cands


# ---------------- main ----------------
def main_stage1():
        def _parse_figure_numbers() -> set[int] | None:
            raw = os.environ.get("CEE_FIGURE_NUMBERS", "").strip()
            if not raw:
                return None
            out: set[int] = set()
            for token in raw.replace(",", " ").split():
                if "-" in token:
                    a, b = token.split("-", 1)
                    lo, hi = sorted((int(a), int(b)))
                    out.update(range(lo, hi + 1))
                else:
                    out.add(int(token))
            return out

        def _want_figure(n: int, enabled: set[int] | None) -> bool:
            return True if enabled is None else (int(n) in enabled)

        _fig = _parse_figure_numbers()

        def _want(n):
            return _want_figure(n, _fig)

        if _fig is not None and _fig.isdisjoint({1, 2, 3, 4, 5, 6}):
            print("[stage1] skipping — CEE_FIGURE_NUMBERS does not include figures 1–6.")
            return

        desired_picks = _figure_picks_from_env()
        figure_policy_types = _figure_policies_from_env()
        remake_parallel = _env_override_remake(REMAKE_PARALLEL_PLOTS, "CEE_REMAKE_PARALLEL_PLOTS")
        remake_dynamics = _env_override_remake(REMAKE_DYNAMICS_PLOTS, "CEE_REMAKE_DYNAMICS_PLOTS")
        skip_pywr = _env_truthy("CEE_SKIP_PYWR")

        t_run = time.perf_counter()
        t0 = t_run
        print("[timing] --- figure pipeline start ---", flush=True)

        fig_root = resolve_figure_root(FIG_DIR)
        fig4_reservoirs = _fig4_reservoirs_from_env(RESERVOIR_NAMES)
        fig4_policies = _fig4_policies_from_env(figure_policy_types)
        fig4_picks = _fig4_picks_from_env(desired_picks)
        fig4_k = _fig4_k_from_env()
        if any(
            os.environ.get(k, "").strip()
            for k in (
                "CEE_FIG4_RESERVOIRS",
                "CEE_FIG4_POLICIES",
                "CEE_FIG4_PICKS",
                "CEE_FIG4_K",
                "CEE_FIG3_RESERVOIRS",
                "CEE_FIG3_POLICIES",
                "CEE_FIG3_PICKS",
                "CEE_FIG3_K",
            )
        ):
            print(
                f"[Fig4 filter] CEE_FIG4_* / legacy CEE_FIG3_* active: reservoirs={fig4_reservoirs} "
                f"policies={fig4_policies} picks={fig4_picks} k={fig4_k!r}"
            )

        print(
            f"[speed] picks={len(desired_picks)} policies={figure_policy_types} "
            f"remake_parallel={remake_parallel} remake_dynamics={remake_dynamics} skip_pywr={skip_pywr}"
        )
        print(
            f"[paths] figures -> {fig_root} "
            f"(CEE_FIG_SUBDIR={os.environ.get('CEE_FIG_SUBDIR', '')!r})"
        )
        print(
            "[paths] Borg CSVs: CEE_BORG_OUTPUT_DIR (default: outputs/), CEE_BORG_SEED or CEE_SEED, "
            f"CEE_BORG_MRF_FILTERED={os.environ.get('CEE_BORG_MRF_FILTERED', '')!r} "
            f"(deprecated alias CEE_BORG_MRFMASKED={os.environ.get('CEE_BORG_MRFMASKED', '')!r}); "
            f"CEE_MRF_FILTER_TAG={os.environ.get('CEE_MRF_FILTER_TAG', '')!r}"
        )
        print(
            f"[pywr] ModelBuilder flow_prediction_mode={_pywr_flow_prediction_mode()!r} "
            "(override: CEE_PYWR_FLOW_PREDICTION_MODE)"
        )
        print(f"[pywr] pywrdrb package: {pywrdrb.__file__}")

        # Quick sanity print from one reservoir
        inflow_obs, release_obs, storage_obs = get_observational_training_data(
            reservoir_name='prompton',
            data_dir=PROCESSED_DATA_DIR,
            as_numpy=False,
            inflow_type='inflow_pub'
        )
        print(f"Inflows shape: {inflow_obs.shape}")

        # ensure figure subfolders exist
        Path(fig_root, "fig01_pareto_front_comparison").mkdir(parents=True, exist_ok=True)
        Path(fig_root, "fig02_parallel_axes").mkdir(parents=True, exist_ok=True)
        Path(fig_root, "fig03_parameter_ranges").mkdir(parents=True, exist_ok=True)
        Path(fig_root, "fig04_dynamics").mkdir(parents=True, exist_ok=True)
        Path(fig_root, "fig05_temporal_aggregation_evaluation").mkdir(parents=True, exist_ok=True)
        Path(fig_root, "fig06_policy_surfaces").mkdir(parents=True, exist_ok=True)

        obj_labels = OBJ_LABELS
        obj_cols = list(obj_labels.values())
        minmaxs_all = ['max' if senses_all[c] == 'max' else 'min' for c in obj_cols]

        # ---------- load raw + filtered ----------
        solution_objs, solution_vars, solution_adv_maps, solution_adv_cands = (
            load_filtered_borg_solution_tables(RESERVOIR_NAMES, figure_policy_types, verbose=True)
        )

        t0 = _timing_tick("load Borg CSVs + MOEA / advanced highlight selections", t0)

        # ---------- print ranges ----------
        summarize_ranges(solution_objs, obj_cols)       # objectives
        summarize_param_ranges(solution_vars)           # decision variables (parameters)

        t0 = _timing_tick("summarize objective & parameter ranges (print only)", t0)

        _wants_sim_figs = _fig is None or bool(_fig & {4, 5, 6})

        # ---------- plotting: figures 1–2 (Borg CSVs), then figure 3 (parameter ranges) ----------
        t_fig = time.perf_counter()
        plot_obj_cols = ["Release NSE", "Storage NSE"]
        ideal_point = [1.0, 1.0]

        if _want(1):
            print("#### Figure 1 - Pareto Front Comparison #####")
            dp_mode = _fig1_default_point_mode()
            for reservoir in RESERVOIR_NAMES:
                if not reservoir_has_any(solution_objs, reservoir):
                    print(f"[Fig1] Skip {reservoir}: no solutions for any policy.")
                    continue

                obj_dfs, labels, series_colors = [], [], []
                for policy in figure_policy_types:
                    if not has_solutions(solution_objs, reservoir, policy):
                        print(f"[Fig1] Skip {reservoir}/{policy}: no solutions.")
                        continue
                    obj_dfs.append(solution_objs[reservoir][policy])
                    labels.append(policy_labels[policy])
                    series_colors.append(policy_colors[policy])

                if not obj_dfs:
                    continue

                baseline_path = (
                    Path(FIG_DIR)
                    / f"{BASELINE_DIR_NAME}_{BASELINE_INFLOW_TAG}"
                    / f"baseline_objectives_{reservoir}_{VAL_START}_to_{VAL_END}.csv"
                )
                baseline_xy = None
                if reservoir not in _BASELINE_OBJ_CACHE:
                    try:
                        _BASELINE_OBJ_CACHE[reservoir] = compute_baseline_objectives_for_reservoir(
                            reservoir,
                            start=str(VAL_START),
                            end=str(VAL_END),
                        )
                    except Exception as e:
                        print(f"[Fig1] Baseline objective load failed for {reservoir}: {e}")
                        _BASELINE_OBJ_CACHE[reservoir] = pd.DataFrame(
                            columns=["metric", "pywr_baseline"]
                        )
                bdf = _BASELINE_OBJ_CACHE[reservoir]
                if not bdf.empty:
                    bls = baseline_series_from_df(bdf, plot_obj_cols)
                    r_nse = bls.get("Release NSE")
                    s_nse = bls.get("Storage NSE")
                    if np.isfinite(r_nse) and np.isfinite(s_nse):
                        baseline_xy = (float(r_nse), float(s_nse))
                elif baseline_path.is_file():
                    # Backward-compatible fallback for older cached runs.
                    bdf = pd.read_csv(baseline_path)
                    bls = baseline_series_from_df(bdf, plot_obj_cols)
                    r_nse = bls.get("Release NSE")
                    s_nse = bls.get("Storage NSE")
                    if np.isfinite(r_nse) and np.isfinite(s_nse):
                        baseline_xy = (float(r_nse), float(s_nse))

                base_title = f"Pareto Front Comparison - {reservoir_labels.get(reservoir, reservoir)}"
                if dp_mode == "both":
                    variants = [
                        ("with_pywr_default", baseline_xy),
                        ("no_pywr_default", None),
                    ]
                elif dp_mode == "with_only":
                    variants = [("with_pywr_default", baseline_xy)]
                else:
                    variants = [("no_pywr_default", None)]

                for suffix, bp in variants:
                    fname = f"{fig_root}/fig01_pareto_front_comparison/{safe_name(reservoir)}_{suffix}.png"
                    sub = f" ({suffix.replace('_', ' ')})" if dp_mode == "both" else ""
                    plot_pareto_front_comparison(
                        obj_dfs,
                        labels,
                        obj_cols=plot_obj_cols,
                        ideal=ideal_point,
                        title=base_title + sub,
                        fname=fname,
                        baseline_point=bp,
                        series_colors=series_colors,
                    )

        t0 = _timing_tick("Figure 1 (Pareto comparison)", t_fig)

        t_fig2 = time.perf_counter()
        if _want(2) and remake_parallel:
            print("#### Figure 2 - Parallel Axis Plot #####")
            # (A) All solutions per reservoir & policy
            print("Plotting all solutions for each reservoir & policy...")
            for reservoir_name in RESERVOIR_NAMES:
                for policy_type in figure_policy_types:
                    if not has_solutions(solution_objs, reservoir_name, policy_type):
                        print(f"[Fig2-all] Skip {reservoir_name}/{policy_type}: no solutions.")
                        continue
                    obj_df = solution_objs[reservoir_name][policy_type].copy()
                    fname1 = (
                        f"{fig_root}/fig02_parallel_axes/"
                        f"fig02_all_objectives_{safe_name(reservoir_name)}_{policy_type}.png"
                    )
                    custom_parallel_coordinates(
                        objs=obj_df,
                        columns_axes=obj_cols,
                        axis_labels=obj_cols,
                        ideal_direction='top',
                        minmaxs=minmaxs_all,
                        color_by_continuous=0,
                        color_palette_continuous=None,
                        color_by_categorical=None,
                        color_palette_categorical=None,
                        colorbar_ticks_continuous=None,
                        color_dict_categorical=None,
                        zorder_by=0,
                        zorder_num_classes=10,
                        zorder_direction='ascending',
                        alpha_base=0.7,
                        brushing_dict=None,
                        alpha_brush=0.05,
                        lw_base=1.5,
                        fontsize=10,
                        figsize=(12, 7.5),
                        bottom_pad=0.22,
                        legend_pad=0.08,
                        fname=fname1
                    )

            # (B) Single PAP: scalarization + literature picks merged (one legend, one file)
            print("Plotting unified Pareto selection (scalarization + literature picks) for each reservoir & policy...")
            for reservoir_name in RESERVOIR_NAMES:
                for policy_type in figure_policy_types:
                    if not has_solutions(solution_objs, reservoir_name, policy_type):
                        print(f"[Fig2-sel] Skip {reservoir_name}/{policy_type}: no solutions.")
                        continue
                    obj_df_sel = apply_combined_selection_column(
                        solution_objs[reservoir_name][policy_type].copy()
                    )
                    present = pd.unique(obj_df_sel["highlight_selection"].astype(str))
                    selection_colors = color_dict_for_selection_parplot(list(present))
                    fname_sel = (
                        f"{fig_root}/fig02_parallel_axes/"
                        f"fig02_selection_unified_{safe_name(reservoir_name)}_{policy_type}.png"
                    )
                    custom_parallel_coordinates(
                        objs=obj_df_sel,
                        columns_axes=obj_cols,
                        axis_labels=obj_cols,
                        ideal_direction='top',
                        minmaxs=minmaxs_all,
                        color_by_continuous=None,
                        color_palette_continuous=None,
                        color_by_categorical='highlight_selection',
                        color_palette_categorical=None,
                        colorbar_ticks_continuous=None,
                        color_dict_categorical=selection_colors,
                        zorder_by=0,
                        zorder_num_classes=10,
                        zorder_direction='ascending',
                        alpha_base=0.9,
                        brushing_dict=None,
                        alpha_brush=0.1,
                        lw_base=1.5,
                        fontsize=10,
                        figsize=(12, 8),
                        bottom_pad=0.34,
                        legend_pad=0.14,
                        legend_ncol=2,
                        legend_label_width=36,
                        fname=fname_sel,
                        footnote=COMBINED_SELECTION_FOOTNOTE,
                    )

            # (D) Presentation: one PNG per selected pick — single highlighted line, all others grey
            if _env_override_remake(True, "CEE_FIG2_PRESENTATION_HIGHLIGHT"):
                print(
                    "Plotting parallel-axis presentation panels "
                    "(fig02_parallel_axes/presentation_one_pick/)..."
                )
                _pick_order = {lab: i for i, lab in enumerate(DESIRED_PICKS_ORDER)}
                _pres_foot = (
                    "Highlighted line: the chosen nondominated solution for this criterion; "
                    "grey lines: all other solutions on the same reservoir–policy Pareto front."
                )
                pres_dir = Path(fig_root) / "fig02_parallel_axes" / "presentation_one_pick"
                pres_dir.mkdir(parents=True, exist_ok=True)
                for reservoir_name in RESERVOIR_NAMES:
                    for policy_type in figure_policy_types:
                        if not has_solutions(solution_objs, reservoir_name, policy_type):
                            continue
                        obj_df_sel = apply_combined_selection_column(
                            solution_objs[reservoir_name][policy_type].copy()
                        )
                        present = [
                            p
                            for p in pd.unique(obj_df_sel["highlight_selection"].astype(str))
                            if str(p) != "Other"
                        ]
                        present.sort(
                            key=lambda p: (_pick_order.get(normalize_pick_label(p), len(DESIRED_PICKS_ORDER)), str(p))
                        )
                        for pick_label in present:
                            df_one = obj_df_sel.copy()
                            sel = df_one["highlight_selection"].astype(str) == str(pick_label)
                            if not sel.any():
                                continue
                            df_one["highlight_single_pick"] = np.where(
                                sel,
                                str(pick_label),
                                "Other",
                            )
                            pal = color_dict_for_selection_parplot([str(pick_label), "Other"])
                            hl_col = pal.get(str(pick_label), "#1f77b4")
                            cd = {str(pick_label): hl_col, "Other": pal.get("Other", "#bdbdbd")}
                            slug = pick_filename_slug(pick_label)
                            fname_p = pres_dir / (
                                f"fig02_pap_onepick_{safe_name(reservoir_name)}_{policy_type}_{slug}.png"
                            )
                            custom_parallel_coordinates(
                                objs=df_one,
                                columns_axes=obj_cols,
                                axis_labels=obj_cols,
                                ideal_direction="top",
                                minmaxs=minmaxs_all,
                                color_by_continuous=None,
                                color_palette_continuous=None,
                                color_by_categorical="highlight_single_pick",
                                color_palette_categorical=None,
                                colorbar_ticks_continuous=None,
                                color_dict_categorical=cd,
                                zorder_by=0,
                                zorder_num_classes=10,
                                zorder_direction="ascending",
                                alpha_base=0.95,
                                brushing_dict=None,
                                alpha_brush=0.12,
                                lw_base=1.5,
                                fontsize=10,
                                figsize=(12, 7.5),
                                bottom_pad=0.28,
                                legend_pad=0.1,
                                legend_ncol=1,
                                legend_label_width=48,
                                fname=str(fname_p),
                                footnote=_pres_foot,
                            )

            # (C) All policies combined per reservoir
            print("Plotting all solutions, all policies for each reservoir...")
            for reservoir_name in RESERVOIR_NAMES:
                if not reservoir_has_any(solution_objs, reservoir_name):
                    print(f"[Fig2-allpol] Skip {reservoir_name}: no solutions.")
                    continue

                obj_list = []
                for policy_type in figure_policy_types:
                    if has_solutions(solution_objs, reservoir_name, policy_type):
                        df = solution_objs[reservoir_name][policy_type].copy()
                        df['policy'] = policy_type
                        obj_list.append(df)

                if not obj_list:
                    continue

                combined_df = pd.concat(obj_list, axis=0)
                if combined_df.empty:
                    print(f"[Fig2-allpol] Skip {reservoir_name}: combined empty.")
                    continue

                combined_df = combined_df.sample(frac=1).reset_index(drop=True)
                fname1 = (
                    f"{fig_root}/fig02_parallel_axes/"
                    f"fig02_all_policies_overlay_{safe_name(reservoir_name)}.png"
                )
                custom_parallel_coordinates(
                    objs=combined_df,
                    columns_axes=obj_cols,
                    axis_labels=obj_cols,
                    ideal_direction='top',
                    minmaxs=minmaxs_all,
                    color_by_continuous=None,
                    color_palette_continuous=None,
                    color_by_categorical='policy',
                    color_palette_categorical=None,
                    colorbar_ticks_continuous=None,
                    color_dict_categorical=policy_colors,
                    zorder_by=None,
                    zorder_num_classes=None,
                    zorder_direction='ascending',
                    alpha_base=0.3,
                    brushing_dict=None,
                    alpha_brush=0.1,
                    lw_base=1.0,
                    fontsize=10,
                    figsize=(12, 7.5),
                    bottom_pad=0.22,
                    legend_pad=0.08,
                    fname=fname1
                )

        if _want(2) and remake_parallel:
            t0 = _timing_tick("Figure 2 (parallel axes, all panels)", t_fig2)
        else:
            t0 = _timing_tick("Figure 2 (skipped)", t_fig2)


        t_fig_pr = time.perf_counter()
        if _want(3):
            from methods.plotting.plot_bounds_tables import make_reservoir_visual

            for reservoir in RESERVOIR_NAMES:
                try:
                    make_reservoir_visual(reservoir, outdir=Path(fig_root) / "fig03_parameter_ranges")
                except Exception as e:
                    print(f"[Fig3] parameter ranges skipped ({reservoir}): {e}")
        t0 = _timing_tick("Figure 3 (parameter ranges from optimization)", t_fig_pr)

        # ---------- run selected picks once (shared cache for Fig 4–6 + optional export) ----------
        t_cache0 = time.perf_counter()
        (
            run_cache,
            obs_cache,
            pick_index_map,
            initial_storage_by_reservoir,
        ) = _run_selected_solution_cache(
            solution_objs,
            solution_vars,
            solution_adv_maps,
            selected_labels=desired_picks,
            policy_types=figure_policy_types,
            skip_indie_runs=not _wants_sim_figs,
        )
        t0 = _timing_tick("independent Reservoir cache (all selected picks)", t_cache0)
        if EXPORT_SELECTED_TIMESERIES_HDF5 and run_cache:
            t_ex0 = time.perf_counter()
            _export_selected_timeseries_hdf5(
                fig_root=fig_root,
                run_cache=run_cache,
                obs_cache=obs_cache,
                pick_index_map=pick_index_map,
            )
            t0 = _timing_tick("export selected-solution timeseries HDF5", t_ex0)

        # ---------- run Pywr-DRB once per (policy, pick) ----------
        # One full-basin simulation per (policy, pick): release_policy_dict lists each
        # reservoir's params (first selected row per reservoir). OutputRecorder writes one
        # HDF5 (scenario 0); pywrdrb.Data.load_output() reads it — no merge of batch files.
        # Use CEE_SKIP_PYWR=1 or CEE_DESIRED_PICKS / CEE_FIGURE_POLICIES to reduce work;
        # existing HDF5 under CEE_PYWR_WORK_DIR is reused when valid (shared with validation).
        PYWR_INFLOW_TYPE = (
            os.environ.get("CEE_PYWR_INFLOW_TYPE", "pub_nhmv10_BC_withObsScaled").strip()
            or "pub_nhmv10_BC_withObsScaled"
        )
        pywr_kmax_map: dict[tuple[str, str], int] = {}
        pywr_hdf5_path_map: dict[tuple[str, str], Path] = {}
        pywr_reservoirs_map: dict[tuple[str, str], set[str]] = {}

        if skip_pywr or not _wants_sim_figs:
            if skip_pywr:
                print("[speed] CEE_SKIP_PYWR=1 — skipping Pywr HDF5 generation; Fig 4–6 Pywr panels skipped.")
            else:
                print("[speed] Figures 4–6 not requested — skipping Pywr HDF5 generation.")
            t0 = _timing_tick("Pywr HDF5 generation (skipped)", t0)
        else:
            t_pywr_block = time.perf_counter()
            for policy_type in figure_policy_types:
                for pick_label in desired_picks:
                    idxs_by_res = {
                        reservoir_name: pick_index_map.get(reservoir_name, {}).get(policy_type, {}).get(pick_label, [])
                        for reservoir_name in RESERVOIR_NAMES
                    }
                    reservoirs_for_pick = [r for r, idxs in idxs_by_res.items() if len(idxs) > 0]
                    if not reservoirs_for_pick:
                        continue

                    pywr_kmax_map[(policy_type, pick_label)] = 1
                    pywr_reservoirs_map[(policy_type, pick_label)] = set(reservoirs_for_pick)
                    t_combo = time.perf_counter()
                    out_h5 = _ensure_pywr_pick_hdf5(
                        fig_root=fig_root,
                        policy_type=policy_type,
                        pick_label=pick_label,
                        pick_index_map=pick_index_map,
                        run_cache=run_cache,
                        pywr_inflow_type=PYWR_INFLOW_TYPE,
                        reservoirs_to_include=reservoirs_for_pick,
                        solution_vars=solution_vars,
                        initial_storage_by_reservoir=initial_storage_by_reservoir,
                    )
                    pywr_hdf5_path_map[(policy_type, pick_label)] = out_h5
                    _timing_tick(
                        f"Pywr combo {policy_type} / {pick_label!r} (incl. model.run if rebuilt)",
                        t_combo,
                    )
            t0 = _timing_tick("Pywr HDF5 generation (all combos total)", t_pywr_block)

        # cache loaded pywrdrb HDF5 objects to avoid re-reading during plotting
        _pywr_data_cache: dict[str, pywrdrb.Data] = {}
        PYWR_RESULTS_SETS = ["res_storage", "res_release", "reservoir_downstream_gage"]

        def _get_pywr_data(path: Path) -> pywrdrb.Data:
            key = str(path)
            data = _pywr_data_cache.get(key)
            if data is None:
                data = pywrdrb.Data(
                    print_status=False,
                    results_sets=PYWR_RESULTS_SETS,
                    output_filenames=[str(path)],
                )
                data.load_output()
                _pywr_data_cache[key] = data
            return data

        print("##### Figure 4 - annual aggregation: independent vs Pywr-DRB (same evaluation window) #####")
        eval_start, eval_end = _eval_window_stage1()
        eval_token = _eval_window_file_token(eval_start, eval_end)
        print(f"[Fig4–5] Evaluation window: {eval_start} → {eval_end} (override with CEE_EVAL_START / CEE_EVAL_END)")
        t_fig4_dyn = time.perf_counter()
        if remake_dynamics and not skip_pywr and _wants_sim_figs and _want(4):
            for reservoir_name in fig4_reservoirs:
                if reservoir_name not in obs_cache:
                    print(f"[Fig4] Skip {reservoir_name}: missing obs.")
                    continue

                dt_all = obs_cache[reservoir_name]["datetime"]
                inflow_obs_ser = pd.Series(obs_cache[reservoir_name]["inflow"].flatten(), index=dt_all, name=f"{reservoir_name}_inflow")
                obs_storage_ser = pd.Series(obs_cache[reservoir_name]["storage"].flatten(), index=dt_all, name=f"{reservoir_name}_storage")
                obs_release_arr = obs_cache[reservoir_name]["release"]
                if obs_release_arr is None:
                    obs_release_ser = pd.Series(np.full_like(obs_storage_ser.values, np.nan, dtype=float), index=dt_all, name=f"{reservoir_name}_release")
                else:
                    obs_release_ser = pd.Series(np.asarray(obs_release_arr).flatten(), index=dt_all, name=f"{reservoir_name}_release")

                for policy_type in fig4_policies:
                    if not has_solutions(solution_objs, reservoir_name, policy_type):
                        print(f"[Fig4] Skip {reservoir_name}/{policy_type}: no solutions.")
                        continue

                    for pick_label in fig4_picks:
                        idxs = pick_index_map.get(reservoir_name, {}).get(policy_type, {}).get(pick_label, [])
                        if not idxs:
                            print(f"[Fig4] {reservoir_name}/{policy_type}: no '{pick_label}' pick; skip.")
                            continue
                        if reservoir_name not in pywr_reservoirs_map.get((policy_type, pick_label), set()):
                            print(f"[Fig4] Skip {reservoir_name}/{policy_type}/{pick_label}: not included in Pywr pick set.")
                            continue

                        kmax_common = pywr_kmax_map.get((policy_type, pick_label), 0)
                        if kmax_common <= 0:
                            print(f"[Fig4] Skip {reservoir_name}/{policy_type}/{pick_label}: no shared Pywr kmax.")
                            continue

                        idxs_common = idxs[:kmax_common]
                        pywr_h5_path = pywr_hdf5_path_map[(policy_type, pick_label)]
                        pywr_data = _get_pywr_data(pywr_h5_path)
                        kP = pywr_h5_path.stem

                        for k, idx_row in enumerate(idxs_common, start=1):
                            if fig4_k is not None and k != fig4_k:
                                continue
                            cache_key = (reservoir_name, policy_type, idx_row)
                            run = run_cache.get(cache_key)
                            if run is None:
                                print(f"[Fig4] {reservoir_name}/{policy_type}/{pick_label}: missing cache row {idx_row}; skip.")
                                continue

                            scenario_id = k - 1
                            pywr_storage_ser = pywr_data.res_storage[kP][scenario_id][reservoir_name].astype(float)
                            pywr_release_ser = _pywr_release_series_for_reservoir(
                                pywr_data, kP, scenario_id, reservoir_name, log_tag="Fig4"
                            )
                            if pywr_release_ser is None:
                                continue

                            indie_storage_ser = pd.Series(
                                np.asarray(run["sim_storage"], float).flatten(), index=dt_all, name=reservoir_name
                            )
                            indie_release_ser = pd.Series(
                                np.asarray(run["sim_release"], float).flatten(), index=dt_all, name=reservoir_name
                            )

                            idx_all = (
                                dt_all.intersection(pywr_storage_ser.index)
                                .intersection(pywr_release_ser.index)
                                .intersection(indie_storage_ser.index)
                                .intersection(indie_release_ser.index)
                            )
                            if len(idx_all) == 0:
                                print(f"[Fig4] {reservoir_name}/{policy_type}/{pick_label}/k{k}: no overlap indie/Pywr; skip.")
                                continue

                            idx_sorted = idx_all.sort_values()
                            idx_win = idx_sorted[
                                (idx_sorted >= eval_start) & (idx_sorted <= eval_end)
                            ]
                            if len(idx_win) == 0:
                                print(
                                    f"[Fig4] {reservoir_name}/{policy_type}/{pick_label}/k{k}: "
                                    f"empty window {eval_start}–{eval_end}; skip."
                                )
                                continue

                            base = (
                                f"{fig_root}/fig04_dynamics/"
                                f"{safe_name(reservoir_name)}_{policy_type}_{pick_filename_slug(pick_label)}_k{k}"
                            )
                            compare_out = f"{base}_{eval_token}_annual_agg_independent_vs_pywr.png"
                            plot_storage_release_distributions_independent_vs_pywr_split(
                                obs_storage=obs_storage_ser.loc[idx_win].values,
                                obs_release=obs_release_ser.loc[idx_win].values,
                                indie_storage=indie_storage_ser.loc[idx_win].values,
                                indie_release=indie_release_ser.loc[idx_win].values,
                                pywr_storage=pywr_storage_ser.loc[idx_win].values,
                                pywr_release=pywr_release_ser.loc[idx_win].values,
                                obs_inflow=inflow_obs_ser.loc[idx_win].values,
                                datetime=idx_win,
                                fname=compare_out,
                                eval_period_label=f"{eval_start} to {eval_end}",
                                pick_label=str(pick_label),
                                pywr_mode_label=_pywr_flow_prediction_mode(),
                            )
                            if policy_type == "STARFIT":
                                lo366, hi366 = try_compute_starfit_nor_pct_by_doy(
                                    run["params"], reservoir_name
                                )
                                if lo366 is not None:
                                    compare_4a = (
                                        f"{base}_{eval_token}_annual_agg_independent_vs_pywr_pct_nor.png"
                                    )
                                    plot_storage_release_distributions_independent_vs_pywr_split(
                                        obs_storage=obs_storage_ser.loc[idx_win].values,
                                        obs_release=obs_release_ser.loc[idx_win].values,
                                        indie_storage=indie_storage_ser.loc[idx_win].values,
                                        indie_release=indie_release_ser.loc[idx_win].values,
                                        pywr_storage=pywr_storage_ser.loc[idx_win].values,
                                        pywr_release=pywr_release_ser.loc[idx_win].values,
                                        obs_inflow=inflow_obs_ser.loc[idx_win].values,
                                        datetime=idx_win,
                                        fname=compare_4a,
                                        eval_period_label=f"{eval_start} to {eval_end}",
                                        capacity_mg=float(reservoir_capacity[reservoir_name]),
                                        nor_lo_pct_by_doy=lo366,
                                        nor_hi_pct_by_doy=hi366,
                                        pick_label=str(pick_label),
                                        pywr_mode_label=_pywr_flow_prediction_mode(),
                                        suptitle_extra=(
                                            "Storage: % of capacity; shaded band = STARFIT NOR (seasonal)."
                                        ),
                                    )
                                else:
                                    print(
                                        f"[Fig4a] STARFIT NOR overlay skipped for "
                                        f"{reservoir_name}/{pick_label}/k{k} (setup/import)."
                                    )
        elif remake_dynamics and skip_pywr:
            print("[Fig4] Skipped (CEE_SKIP_PYWR=1 — no Pywr HDF5 for comparison).")

        if remake_dynamics and not skip_pywr and _wants_sim_figs and _want(4):
            t0 = _timing_tick("Figure 4 (dynamics)", t_fig4_dyn)
        elif remake_dynamics:
            t0 = _timing_tick("Figure 4 (skipped)", t_fig4_dyn)
        else:
            t0 = _timing_tick("Figure 4 (skipped, CEE_REMAKE_DYNAMICS_PLOTS=0)", t_fig4_dyn)

        print("##### Figure 6 - policy surfaces (optional) #####")
        t_fig6_surf = time.perf_counter()
        _fig6_style = os.environ.get("CEE_FIG6_STYLE", "v2").strip().lower()
        if _fig6_style not in ("v2", "legacy", "both"):
            print(f"[Fig6] Unknown CEE_FIG6_STYLE={_fig6_style!r}; using 'v2'.")
            _fig6_style = "v2"
        if (
            remake_dynamics
            and not skip_pywr
            and _wants_sim_figs
            and _want(6)
            and not _env_truthy("CEE_SKIP_POLICY_SURFACES")
        ):
            for reservoir_name in fig4_reservoirs:
                if reservoir_name not in obs_cache:
                    print(f"[Fig6] Skip {reservoir_name}: missing obs.")
                    continue

                dt_all = obs_cache[reservoir_name]["datetime"]
                inflow_obs_ser = pd.Series(obs_cache[reservoir_name]["inflow"].flatten(), index=dt_all, name=f"{reservoir_name}_inflow")
                obs_storage_ser = pd.Series(obs_cache[reservoir_name]["storage"].flatten(), index=dt_all, name=f"{reservoir_name}_storage")
                obs_release_arr = obs_cache[reservoir_name]["release"]
                if obs_release_arr is None:
                    obs_release_ser = pd.Series(np.full_like(obs_storage_ser.values, np.nan, dtype=float), index=dt_all, name=f"{reservoir_name}_release")
                else:
                    obs_release_ser = pd.Series(np.asarray(obs_release_arr).flatten(), index=dt_all, name=f"{reservoir_name}_release")

                for policy_type in fig4_policies:
                    if not has_solutions(solution_objs, reservoir_name, policy_type):
                        print(f"[Fig6] Skip {reservoir_name}/{policy_type}: no solutions.")
                        continue

                    for pick_label in fig4_picks:
                        idxs = pick_index_map.get(reservoir_name, {}).get(policy_type, {}).get(pick_label, [])
                        if not idxs:
                            print(f"[Fig6] {reservoir_name}/{policy_type}: no '{pick_label}' pick; skip.")
                            continue
                        if reservoir_name not in pywr_reservoirs_map.get((policy_type, pick_label), set()):
                            print(f"[Fig6] Skip {reservoir_name}/{policy_type}/{pick_label}: not included in Pywr pick set.")
                            continue

                        kmax_common = pywr_kmax_map.get((policy_type, pick_label), 0)
                        if kmax_common <= 0:
                            print(f"[Fig6] Skip {reservoir_name}/{policy_type}/{pick_label}: no shared Pywr kmax.")
                            continue

                        idxs_common = idxs[:kmax_common]
                        pywr_h5_path = pywr_hdf5_path_map[(policy_type, pick_label)]
                        pywr_data = _get_pywr_data(pywr_h5_path)
                        kP = pywr_h5_path.stem

                        for k, idx_row in enumerate(idxs_common, start=1):
                            if fig4_k is not None and k != fig4_k:
                                continue
                            cache_key = (reservoir_name, policy_type, idx_row)
                            run = run_cache.get(cache_key)
                            if run is None:
                                print(f"[Fig6] {reservoir_name}/{policy_type}/{pick_label}: missing cache row {idx_row}; skip.")
                                continue

                            params = run["params"]

                            scenario_id = k - 1
                            pywr_storage_ser = pywr_data.res_storage[kP][scenario_id][reservoir_name].astype(float)
                            pywr_release_ser = _pywr_release_series_for_reservoir(
                                pywr_data, kP, scenario_id, reservoir_name, log_tag="Fig6"
                            )
                            if pywr_release_ser is None:
                                continue

                            idx_common = dt_all.intersection(pywr_storage_ser.index).intersection(pywr_release_ser.index)
                            if len(idx_common) == 0:
                                print(f"[Fig6] {reservoir_name}/{policy_type}/{pick_label}/k{k}: no overlap with Pywr output; skip.")
                                continue

                            try:
                                reservoir = Reservoir(
                                    inflow=inflow_obs_ser.values,
                                    dates=dt_all,
                                    capacity=reservoir_capacity[reservoir_name],
                                    policy_type=policy_type,
                                    policy_params=params,
                                    name=reservoir_name,
                                )
                                surf_dir = Path(fig_root, "fig06_policy_surfaces")
                                surf_dir.mkdir(parents=True, exist_ok=True)
                                stem = f"{safe_name(reservoir_name)}_{policy_type}_{pick_filename_slug(pick_label)}_{k}"
                                if _fig6_style in ("legacy", "both"):
                                    if hasattr(reservoir.policy, "plot_surfaces_for_different_weeks"):
                                        reservoir.policy.plot_surfaces_for_different_weeks(
                                            fname=str(surf_dir / f"{stem}_surface.png"),
                                            save=True,
                                            grid=40,
                                            n_weeks=5,
                                        )
                                    elif hasattr(reservoir.policy, "plot_policy_surface"):
                                        reservoir.policy.plot_policy_surface(
                                            save=True,
                                            fname=str(surf_dir / f"{stem}_policy_surface.png"),
                                        )
                                if _fig6_style in ("v2", "both"):
                                    sim_storage_ser = pd.Series(
                                        np.asarray(reservoir.storage_array).flatten(),
                                        index=dt_all,
                                    ).loc[idx_common]
                                    sim_release_ser = pd.Series(
                                        np.asarray(reservoir.release_array + reservoir.spill_array).flatten(),
                                        index=dt_all,
                                    ).loc[idx_common]
                                    save_policy_figure6_v2(
                                        reservoir.policy,
                                        surf_dir / f"{stem}_fig06v2.png",
                                        policy_type=policy_type,
                                        reservoir_name=reservoir_name,
                                        pick_slug=pick_filename_slug(pick_label),
                                        k=k,
                                        obs_storage_series=obs_storage_ser.loc[idx_common],
                                        obs_release_series=obs_release_ser.loc[idx_common],
                                        sim_storage_series=sim_storage_ser,
                                        sim_release_series=sim_release_ser,
                                        overlay_years=10,
                                    )
                            except Exception as e:
                                print(f"[Fig6] Policy surface skipped ({reservoir_name}/{policy_type}/{pick_label}/k{k}): {e}")
        elif remake_dynamics and skip_pywr and _want(6):
            print("[Fig6] Skipped (CEE_SKIP_PYWR=1 — no Pywr HDF5 for policy surfaces).")

        if remake_dynamics and not skip_pywr and _wants_sim_figs and _want(6) and not _env_truthy("CEE_SKIP_POLICY_SURFACES"):
            t0 = _timing_tick("Figure 6 (policy surfaces)", t_fig6_surf)
        elif remake_dynamics:
            t0 = _timing_tick("Figure 6 (skipped)", t_fig6_surf)
        else:
            t0 = _timing_tick("Figure 6 (skipped, CEE_REMAKE_DYNAMICS_PLOTS=0)", t_fig6_surf)

        print("##### Figure 5 - Temporal aggregation evaluation (9-panel overlay) #####")
        t_fig5_trends = time.perf_counter()
        if remake_dynamics and not skip_pywr and _wants_sim_figs and _want(5):
            for reservoir_name in RESERVOIR_NAMES:
                if reservoir_name not in obs_cache:
                    continue
                fig5_windows = _fig5_windows_stage1(
                    eval_start,
                    eval_end,
                    reservoir_name=reservoir_name,
                )

                dt_all = obs_cache[reservoir_name]["datetime"]

                for policy_type in figure_policy_types:
                    if not has_solutions(solution_objs, reservoir_name, policy_type):
                        print(f"[Fig5] Skip {reservoir_name}/{policy_type}: no solutions.")
                        continue

                    for pick_label in desired_picks:
                        idxs = pick_index_map.get(reservoir_name, {}).get(policy_type, {}).get(pick_label, [])
                        if not idxs:
                            continue
                        if reservoir_name not in pywr_reservoirs_map.get((policy_type, pick_label), set()):
                            continue
                        kmax_common = pywr_kmax_map.get((policy_type, pick_label), 0)
                        if kmax_common <= 0:
                            continue

                        idxs_common = idxs[:kmax_common]
                        pywr_h5_path = pywr_hdf5_path_map[(policy_type, pick_label)]
                        pywr_data = _get_pywr_data(pywr_h5_path)
                        kP = pywr_h5_path.stem

                        for k, idx_row in enumerate(idxs_common, start=1):
                            cache_key = (reservoir_name, policy_type, idx_row)
                            run = run_cache.get(cache_key)
                            if run is None:
                                continue

                            scenario_id = k - 1
                            pywr_storage_full = pywr_data.res_storage[kP][scenario_id][reservoir_name].astype(float)
                            _pywr_rel = _pywr_release_series_for_reservoir(
                                pywr_data, kP, scenario_id, reservoir_name, log_tag="Fig5"
                            )
                            if _pywr_rel is None:
                                continue
                            pywr_release_full = _pywr_rel

                            for win_label, win_start, win_end in fig5_windows:
                                slicer = slice(win_start, win_end)
                                obs_storage_win = pd.Series(
                                    obs_cache[reservoir_name]["storage"].flatten(),
                                    index=dt_all,
                                    name=reservoir_name,
                                ).loc[slicer]
                                obs_release_arr = obs_cache[reservoir_name]["release"]
                                if obs_release_arr is None:
                                    obs_release_win = None
                                else:
                                    obs_release_win = pd.Series(
                                        np.asarray(obs_release_arr).flatten(),
                                        index=dt_all,
                                        name=reservoir_name,
                                    ).loc[slicer]
                                if len(obs_storage_win) == 0:
                                    print(
                                        f"[Fig5] Skip {reservoir_name}/{policy_type}/{pick_label}/k{k}/{win_label}: "
                                        f"no observed storage in {win_start}→{win_end} "
                                        f"(obs span {dt_all.min().date()}→{dt_all.max().date()}).",
                                        flush=True,
                                    )
                                    continue

                                indie_storage_ser = pd.Series(
                                    np.asarray(run["sim_storage"], float).flatten(),
                                    index=dt_all,
                                    name=reservoir_name,
                                ).loc[slicer]
                                indie_release_ser = pd.Series(
                                    np.asarray(run["sim_release"], float).flatten(),
                                    index=dt_all,
                                    name=reservoir_name,
                                ).loc[slicer]
                                pywr_storage_ser = pywr_storage_full.loc[slicer]
                                pywr_release_ser = pywr_release_full.loc[slicer]

                                def_rel, def_sto = _load_default_release_storage_series(reservoir_name)
                                if def_rel is None or def_sto is None:
                                    raise RuntimeError(
                                        f"Default baseline HDF5 is missing required series for {reservoir_name}: "
                                        f"release={'ok' if def_rel is not None else 'missing'}, "
                                        f"storage={'ok' if def_sto is not None else 'missing'}."
                                    )
                                def_rel_w = def_rel.loc[slicer] if def_rel is not None else None
                                def_sto_w = def_sto.loc[slicer] if def_sto is not None else None
                                flow_mode = _flow_mode_from_hdf5_stem(kP)
                                window_token = _eval_window_file_token(win_start, win_end)
                                out_9 = (
                                    f"{fig_root}/fig05_temporal_aggregation_evaluation/"
                                    f"{safe_name(reservoir_name)}_{policy_type}_{pick_filename_slug(pick_label)}_k{k}"
                                    f"_{safe_name(win_label)}_temporal_agg_eval_{window_token}_9panel.png"
                                )
                                plot_release_storage_9panel(
                                    reservoir=reservoir_name,
                                    sim_release=pywr_release_ser,
                                    sim_storage_MG=pywr_storage_ser,
                                    secondary_release=indie_release_ser,
                                    secondary_storage_MG=indie_storage_ser,
                                    obs_release=obs_release_win if obs_release_win is not None else None,
                                    obs_storage_MG=obs_storage_win,
                                    pywr_default_release=def_rel_w,
                                    pywr_default_storage_MG=def_sto_w,
                                    start=win_start,
                                    end=win_end,
                                    policy_label=policy_type,
                                    pick_label=str(pick_label),
                                    sim_label=f"Simulated (Pywr-DRB parametric, {flow_mode})",
                                    secondary_sim_label="Simulated (independent reservoir model)",
                                    run_type_note=(
                                        "Overlay: Pywr-DRB parametric, independent model, Pywr default operation, and observations."
                                    ),
                                    save_path=out_9,
                                )

            t0 = _timing_tick("Figure 5 (validation 9-panel + comparisons)", t_fig5_trends)
        elif remake_dynamics:
            t0 = _timing_tick("Figure 5 (skipped)", t_fig5_trends)
        else:
            t0 = _timing_tick("Figure 5 (skipped, CEE_REMAKE_DYNAMICS_PLOTS=0)", t_fig5_trends)

        # nothing to close: pywrdrb.Data loaders keep pandas objects in memory

        print(f"[timing] TOTAL wall-clock: {time.perf_counter() - t_run:.2f}s", flush=True)
        print("DONE!")

if __name__ == "__main__":
    main_stage1()
