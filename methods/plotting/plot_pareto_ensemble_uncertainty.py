"""
Pareto ensemble uncertainty helpers (aggregators, envelopes, optional composite PNGs).

Used by ``methods/figures_stage3`` multipanels and diagnostics; not wired to ``04_make_figures.py``.

Runs filtered Pareto rows through Pywr-DRB (``run_pywr_parametric_multi`` + data loader),
summarizes releases and Trenton flow into median / quantile envelope dictionaries, then
renders:

  * **Option 2** — main figure: 4× inflow–release + Trenton time + FDC + reliability
  * **Option 1** — full figure: adds DOY–release row

Inflow–release panels: policy envelopes, then observed daily inflow–release (training loader),
with flow-regime shading from **observed inflow** Q20/Q80 (shared across policies per reservoir).

Optional: load precomputed envelope dicts from pickle (``CEE_PARETO_ENSEMBLE_SUMMARY_PATH``).

Trenton target handling:

- Reads ``CEE_TRENTON_TARGET_MGD`` when building ensemble summaries/plots.
- Falls back to ``DEFAULT_TRENTON_TARGET_MGD`` from
  ``methods.figures_stage3.constants`` when the env var is unset.
- Trenton target env vars are shared with ``methods/figures_stage3``.
"""
from __future__ import annotations

import os
import pickle
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from methods.config import PROCESSED_DATA_DIR
from methods.figures_stage3.constants import DEFAULT_TRENTON_TARGET_MGD
from methods.load.observations import get_observational_training_data

# ---------------------------------------------------------------------------
# Display names (figure panels) vs internal reservoir keys (Borg / Pywr)
# Order matches user template: Blue Marsh, Beltzville, F.E. Walter, Prompton
# ---------------------------------------------------------------------------
RESERVOIR_PANEL_ORDER: Tuple[Tuple[str, str], ...] = (
    ("blueMarsh", "Blue Marsh"),
    ("beltzvilleCombined", "Beltzville"),
    ("fewalter", "F.E. Walter"),
    ("prompton", "Prompton"),
)

# One Pywr-DRB ``ModelBuilder`` run = one ``release_policy_dict`` listing these four parametric nodes
# (each policy type STARFIT | RBF | PWL uses one shared basin simulation per pick / Pareto row).
PARAMETRIC_MOEA_RESERVOIR_KEYS: Tuple[str, ...] = tuple(k for k, _ in RESERVOIR_PANEL_ORDER)

POLICY_ORDER: Tuple[str, ...] = ("STARFIT", "PWL", "RBF")

POLICY_COLORS = {
    "STARFIT": "tab:blue",
    "PWL": "tab:green",
    "RBF": "orange",
}


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def _max_trenton_runs() -> int:
    return max(10, int(os.environ.get("CEE_TRENTON_ENSEMBLE_MAX_RUNS", "50")))


def _pywr_flow_mode() -> str:
    v = os.environ.get("CEE_PYWR_FLOW_PREDICTION_MODE", "").strip()
    return v if v else "regression_disagg"


def _parse_inflow_ensemble_indices_env() -> Optional[List[int]]:
    """Optional ``CEE_INFLOW_ENSEMBLE_INDICES`` — comma-separated ints for Pywr ensemble run."""
    raw = os.environ.get("CEE_INFLOW_ENSEMBLE_INDICES", "").strip()
    if not raw:
        return None
    out: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.append(int(p))
    return out if out else None


# =============================================================================
# Summarization: raw matrices -> envelope dicts expected by Option 1 / 2 plots
# =============================================================================


def envelope_doy_from_matrix(
    dates: pd.DatetimeIndex, mat: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    ``mat`` (T, n_sols): pool all values by calendar day-of-year -> quantiles.
    Returns keys x (1..366), median, q25, q75, q10, q90.
    """
    if mat.size == 0:
        return _empty_envelope_doy()
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    T, n = mat.shape
    if len(doy) != T:
        doy = doy[:T]
    x = np.arange(1, 367, dtype=float)
    out = {k: np.full(366, np.nan) for k in ("median", "q25", "q75", "q10", "q90")}
    for d in range(1, 367):
        mask = doy == d
        chunk = mat[mask, :].ravel()
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size == 0:
            continue
        out["median"][d - 1] = float(np.nanmedian(chunk))
        out["q25"][d - 1] = float(np.nanquantile(chunk, 0.25))
        out["q75"][d - 1] = float(np.nanquantile(chunk, 0.75))
        out["q10"][d - 1] = float(np.nanquantile(chunk, 0.10))
        out["q90"][d - 1] = float(np.nanquantile(chunk, 0.90))
    return {**out, "x": x}


def _empty_envelope_doy() -> Dict[str, np.ndarray]:
    x = np.arange(1, 367, dtype=float)
    empty = np.full(366, np.nan)
    return {
        "x": x,
        "median": empty.copy(),
        "q25": empty.copy(),
        "q75": empty.copy(),
        "q10": empty.copy(),
        "q90": empty.copy(),
    }


def envelope_month_from_matrix(
    dates: pd.DatetimeIndex, mat: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    ``mat`` (T, n_sols): pool all values by calendar month (1–12) -> quantiles.
    Returns keys x (1..12), median, q25, q75, q10, q90 (length 12 each).
    """
    if mat.size == 0:
        x = np.arange(1, 13, dtype=float)
        empty = np.full(12, np.nan)
        return {"x": x, "median": empty, "q25": empty, "q75": empty, "q10": empty, "q90": empty}
    mo = pd.DatetimeIndex(dates).month.to_numpy()
    T, n = mat.shape
    if len(mo) != T:
        mo = mo[:T]
    x = np.arange(1, 13, dtype=float)
    out = {k: np.full(12, np.nan) for k in ("median", "q25", "q75", "q10", "q90")}
    for m in range(1, 13):
        mask = mo == m
        chunk = mat[mask, :].ravel()
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size == 0:
            continue
        out["median"][m - 1] = float(np.nanmedian(chunk))
        out["q25"][m - 1] = float(np.nanquantile(chunk, 0.25))
        out["q75"][m - 1] = float(np.nanquantile(chunk, 0.75))
        out["q10"][m - 1] = float(np.nanquantile(chunk, 0.10))
        out["q90"][m - 1] = float(np.nanquantile(chunk, 0.90))
    return {**out, "x": x}


def envelope_ir_binned(
    inflow_1d: np.ndarray,
    release_mat: np.ndarray,
    n_bins: int = 80,
) -> Dict[str, np.ndarray]:
    """Binned inflow (x) vs release quantiles (across solutions and timesteps in each bin)."""
    T, n = release_mat.shape
    inf = np.asarray(inflow_1d, dtype=float)[:T]
    if inf.shape[0] != T:
        m = min(len(inf), T)
        inf = inf[:m]
        release_mat = release_mat[:m, :]

    order = np.argsort(inf)
    inf_s = inf[order]
    rel_s = release_mat[order, :]
    lo, hi = np.nanpercentile(inf_s, 2), np.nanpercentile(inf_s, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = np.nanmin(inf_s), np.nanmax(inf_s)
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = []
    med, q25, q75, q10, q90 = [], [], [], [], []
    for j in range(n_bins):
        m = (inf_s >= edges[j]) & (inf_s <= edges[j + 1])
        if m.sum() < 3:
            continue
        chunk = rel_s[m, :].ravel()
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size < 3:
            continue
        centers.append(0.5 * (edges[j] + edges[j + 1]))
        med.append(float(np.nanmedian(chunk)))
        q25.append(float(np.nanquantile(chunk, 0.25)))
        q75.append(float(np.nanquantile(chunk, 0.75)))
        q10.append(float(np.nanquantile(chunk, 0.10)))
        q90.append(float(np.nanquantile(chunk, 0.90)))
    if not centers:
        z = np.array([np.nan])
        return {"x": z, "median": z, "q25": z, "q75": z, "q10": z, "q90": z}
    return {
        "x": np.asarray(centers, dtype=float),
        "median": np.asarray(med, dtype=float),
        "q25": np.asarray(q25, dtype=float),
        "q75": np.asarray(q75, dtype=float),
        "q10": np.asarray(q10, dtype=float),
        "q90": np.asarray(q90, dtype=float),
    }


def fdc_quantiles_matrix(flow_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """``flow_matrix`` (T, n): FDC per column, quantiles along rank axis."""
    T, n = flow_matrix.shape
    exceed = np.linspace(100.0 / (T + 1), 100.0 * T / (T + 1), T)
    sorted_desc = np.sort(flow_matrix, axis=0)[::-1, :]
    curves: Dict[str, np.ndarray] = {}
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        curves[f"q{int(q * 100)}"] = np.nanquantile(sorted_desc, q, axis=1)
    return exceed, curves


def envelope_fdc_from_matrix(trenton_mat: np.ndarray) -> Dict[str, np.ndarray]:
    ex, fc = fdc_quantiles_matrix(trenton_mat)
    return {
        "x": ex,
        "median": fc["q50"],
        "q25": fc["q25"],
        "q75": fc["q75"],
        "q10": fc["q10"],
        "q90": fc["q90"],
    }


def observed_fdc_dict(obs: pd.Series) -> Dict[str, np.ndarray]:
    y = np.sort(obs.to_numpy(dtype=float))[::-1]
    y = y[np.isfinite(y)]
    n = len(y)
    if n == 0:
        return {"x": np.array([0.0, 100.0]), "y": np.array([np.nan, np.nan])}
    x = np.linspace(100.0 / (n + 1), 100.0 * n / (n + 1), n)
    return {"x": x, "y": y}


def _inflow_series_on_dates(internal: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """Align publication inflow to simulation ``dates`` for inflow–release envelopes."""
    inf_df, _, _ = get_observational_training_data(
        reservoir_name=internal,
        data_dir=PROCESSED_DATA_DIR,
        as_numpy=False,
        inflow_type="inflow_pub",
    )
    s = inf_df[internal].reindex(dates)
    s = s.interpolate(limit_direction="both").bfill().ffill()
    return s.to_numpy(dtype=float)


def observed_inflow_release_training(internal: str) -> Dict[str, Any]:
    """
    Daily observed inflow–release from the training loader (``inflow_pub``).
    ``q20`` / ``q80`` are percentiles of **inflow** only — shared across policies for
    flow-regime shading on IR panels.
    """
    inf_df, rel_df, _ = get_observational_training_data(
        reservoir_name=internal,
        data_dir=PROCESSED_DATA_DIR,
        as_numpy=False,
        inflow_type="inflow_pub",
    )
    inf = inf_df[internal].astype(float).to_numpy()
    rel = rel_df[internal].astype(float).to_numpy()
    mask = np.isfinite(inf) & np.isfinite(rel) & (inf > 0) & (rel > 0)
    obs_x = inf[mask]
    obs_y = rel[mask]
    if obs_x.size == 0:
        return {"obs_x": obs_x, "obs_y": obs_y, "q20": np.nan, "q80": np.nan}
    q20 = float(np.percentile(obs_x, 20))
    q80 = float(np.percentile(obs_x, 80))
    return {"obs_x": obs_x, "obs_y": obs_y, "q20": q20, "q80": q80}


def build_observed_ir_by_display(reservoir_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Map panel display title -> observed IR scatter + hydrologic Q20/Q80 (inflow)."""
    out: Dict[str, Dict[str, Any]] = {}
    for internal, display in RESERVOIR_PANEL_ORDER:
        if internal not in reservoir_keys:
            continue
        meta = observed_inflow_release_training(internal)
        if meta["obs_x"].size:
            out[display] = meta
    return out


def add_flow_regime_shading(ax, q20, q80, xmin=None, xmax=None):
    """
    Vertical bands on **inflow** (x) axis: low / mid / high from observed inflow Q20, Q80.
    Call after axis limits are set (log scales).
    """
    xmin = xmin if xmin is not None else ax.get_xlim()[0]
    xmax = xmax if xmax is not None else ax.get_xlim()[1]
    if not np.isfinite(q20) or not np.isfinite(q80) or q20 <= 0 or q80 <= 0 or q20 >= q80:
        return
    ax.axvspan(xmin, q20, color="0.85", alpha=0.18, zorder=0)
    ax.axvspan(q80, xmax, color="0.85", alpha=0.18, zorder=0)

    ymax = ax.get_ylim()[1]
    if not np.isfinite(ymax) or ymax <= 0:
        return

    def _gmean(a: float, b: float) -> float:
        return float(np.sqrt(max(a, 1e-12) * max(b, 1e-12)))

    ax.text(_gmean(xmin, q20), ymax * 0.92, "Low flow", ha="center", va="top", fontsize=9, zorder=1)
    ax.text(_gmean(q20, q80), ymax * 0.92, "Mid flow", ha="center", va="top", fontsize=9, zorder=1)
    ax.text(_gmean(q80, xmax), ymax * 0.92, "High flow", ha="center", va="top", fontsize=9, zorder=1)


def _finish_ir_panel(
    ax,
    reservoir_display: str,
    observed_ir: Optional[Dict[str, Dict[str, Any]]],
) -> None:
    """Log scales, x-limits from observed inflow, regime shading (background), observed IR on top."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ent = observed_ir.get(reservoir_display) if observed_ir else None
    if ent is None or ent["obs_x"].size == 0:
        _style_axis(ax, grid=True)
        return

    obs_x = np.asarray(ent["obs_x"], dtype=float)
    obs_y = np.asarray(ent["obs_y"], dtype=float)
    q20, q80 = ent["q20"], ent["q80"]
    pos = obs_x > 0
    if not np.any(pos):
        _style_axis(ax, grid=True)
        return
    xmin, xmax = float(np.min(obs_x[pos])), float(np.max(obs_x[pos]))
    ax.set_xlim(xmin, xmax)
    ax.relim()
    ax.autoscale(axis="y")
    add_flow_regime_shading(ax, q20=q20, q80=q80, xmin=xmin, xmax=xmax)
    ax.plot(obs_x, obs_y, color="black", lw=2.5, label="Observed", zorder=5, alpha=0.9)
    ax.relim()
    ax.autoscale(axis="y")
    _style_axis(ax, grid=True)


# =============================================================================
# Pywr-DRB ensemble (one multi-reservoir model run per Pareto alignment row)
# =============================================================================


def collect_pywr_ensemble_for_policy(
    policy: str,
    solution_vars: Dict[str, Dict[str, pd.DataFrame]],
    reservoirs: List[str],
    pywr_inflow_type: str,
    max_runs: int,
) -> Optional[Dict[str, Any]]:
    """
    For each aligned Pareto index ``i``, build **one** ``release_policy_dict`` containing
    the four MOEA parametric reservoirs (``PARAMETRIC_MOEA_RESERVOIR_KEYS``) so
    ``run_pywr_parametric_multi`` runs a single basin model per row — **not** one run per
    reservoir. The ``reservoirs`` argument is ignored in favor of that fixed set (callers
    may still pass a list for compatibility).

    Extracts downstream-gage release per reservoir and Trenton from the Pywr data loader
    (``_parametric_result_from_h5_path`` / ``obs_trenton``).
    """
    try:
        from pywrdrb.utils.dates import model_date_ranges
        from methods.postprocess.pywr_output_metadata import normalize_borg_row_label
        from methods.postprocess.pywr_parametric_run import run_pywr_parametric_multi
    except Exception as e:
        print(f"[Fig12] Pywr ensemble import failed: {e}")
        return None

    # One combined run per policy × alignment index; always the four MOEA parametric nodes.
    moea_reservoirs = [
        k
        for k in PARAMETRIC_MOEA_RESERVOIR_KEYS
        if k in solution_vars and policy in solution_vars.get(k, {})
    ]
    _ = reservoirs  # callers may pass a wider list; fixed MOEA set is used above

    lengths = []
    for res in moea_reservoirs:
        df = solution_vars.get(res, {}).get(policy)
        if df is None or df.empty:
            return None
        lengths.append(len(df))
    n_align = min(lengths)
    n_runs = min(n_align, max_runs)
    if n_runs < 1:
        return None

    print(
        f"[Fig12] Pywr ensemble: reservoirs={moea_reservoirs} | per-reservoir row counts={dict(zip(moea_reservoirs, lengths))} "
        f"| n_align={n_align} | capped_runs={n_runs}",
        flush=True,
    )

    start, end = model_date_ranges[pywr_inflow_type]
    start, end = str(start), str(end)
    from methods.config import get_pywr_work_dir
    from methods.ensemble.policy_manifest import ensemble_stem_slug

    work_dir = get_pywr_work_dir()
    mode = _pywr_flow_mode()
    ens_idx = _parse_inflow_ensemble_indices_env()
    extra_pywr_opts = (
        {"inflow_ensemble_indices": ens_idx} if ens_idx is not None else None
    )
    ens_slug = ensemble_stem_slug(ens_idx)

    trenton_cols: List[np.ndarray] = []
    reliabilities: List[float] = []
    release_cols: Dict[str, List[np.ndarray]] = {r: [] for r in moea_reservoirs}
    dates_out: Optional[pd.DatetimeIndex] = None
    obs_trenton: Optional[pd.Series] = None

    tgt_raw = os.environ.get("CEE_TRENTON_TARGET_MGD", "").strip()
    target_mgd = float(tgt_raw) if tgt_raw else float(DEFAULT_TRENTON_TARGET_MGD)

    for i in range(n_runs):
        # One dict → one ModelBuilder run: all reservoirs share alignment index ``i``.
        release_policy_dict: Dict[str, Any] = {}
        for res in moea_reservoirs:
            var_df = solution_vars[res][policy]
            row = var_df.iloc[i]
            params = row.values.astype(float)
            release_policy_dict[res] = {
                "class_type": "ParametricReservoirRelease",
                "policy_type": policy,
                "policy_id": "inline",
                "params": ",".join(str(x) for x in params.tolist()),
            }
        stem = f"output_Parametric_{policy}_align{i:05d}_{_safe(mode)}_{ens_slug}"
        row_labels: Dict[str, Any] = {}
        row_ilocs: Dict[str, int] = {}
        for res in moea_reservoirs:
            var_df = solution_vars[res][policy]
            row_labels[res] = normalize_borg_row_label(var_df.index[i])
            row_ilocs[res] = int(i)
        pywr_run_metadata = {
            "policy_type": policy,
            "alignment_index": i,
            "row_index_labels_by_reservoir": row_labels,
            "row_indices_by_reservoir": row_ilocs,
            "extra": {"stem_base": stem, "pywr_work_dir": work_dir},
        }
        try:
            multi_out = run_pywr_parametric_multi(
                release_policy_dict,
                start,
                end,
                pywr_inflow_type,
                work_dir,
                stem,
                mode,
                pywr_run_metadata=pywr_run_metadata,
                extra_model_options=extra_pywr_opts,
            )
        except Exception as e:
            print(f"[Fig12] Pywr row {i} failed: {e}")
            continue
        t = multi_out["trenton"]
        if t is None or len(t) == 0:
            continue
        if dates_out is None:
            dates_out = pd.DatetimeIndex(t.index)
        if obs_trenton is None and multi_out.get("obs_trenton") is not None:
            obs_trenton = multi_out["obs_trenton"]
        trenton_cols.append(t.to_numpy(dtype=float))
        if target_mgd is not None and np.isfinite(target_mgd):
            reliabilities.append(float(np.mean(t.to_numpy(dtype=float) >= target_mgd)))
        else:
            reliabilities.append(float("nan"))

        for rname in moea_reservoirs:
            r0, _ = multi_out["by_res"][rname]
            release_cols[rname].append(r0.to_numpy(dtype=float))

    if not trenton_cols:
        return None

    Tm = max(len(c) for c in trenton_cols)
    trenton_mat = np.full((Tm, len(trenton_cols)), np.nan)
    for j, c in enumerate(trenton_cols):
        trenton_mat[: len(c), j] = c

    release_by_res: Dict[str, np.ndarray] = {}
    for res in moea_reservoirs:
        cols = release_cols.get(res, [])
        if not cols:
            return None
        Tr = max(len(c) for c in cols)
        M = np.full((Tr, len(cols)), np.nan)
        for j, c in enumerate(cols):
            M[: len(c), j] = c
        release_by_res[res] = M

    idx = dates_out if dates_out is not None else pd.date_range(start, periods=Tm, freq="D")
    return {
        "dates": idx,
        "release_by_res": release_by_res,
        "trenton_mat": trenton_mat,
        "obs_trenton": obs_trenton,
        "reliabilities": reliabilities,
        "source": "pywr",
    }


# =============================================================================
# Plotting (user template): Option 2 = main, Option 1 = full
# =============================================================================


def _plot_envelope(
    ax,
    x,
    median,
    q25,
    q75,
    q10,
    q90,
    color,
    label=None,
    lw=2.0,
    alpha_outer=0.08,
    alpha_inner=0.20,
):
    ax.fill_between(x, q10, q90, color=color, alpha=alpha_outer, linewidth=0)
    ax.fill_between(x, q25, q75, color=color, alpha=alpha_inner, linewidth=0)
    ax.plot(x, median, color=color, lw=lw, label=label)


def _style_axis(ax, grid=True):
    if grid:
        ax.grid(True, alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _policy_legend_handles():
    return [
        Line2D([0], [0], color=POLICY_COLORS["STARFIT"], lw=2.5, label="STARFIT"),
        Line2D([0], [0], color=POLICY_COLORS["PWL"], lw=2.5, label="PWL"),
        Line2D([0], [0], color=POLICY_COLORS["RBF"], lw=2.5, label="RBF"),
    ]


def _ir_legend_handles():
    return _policy_legend_handles() + [
        Line2D([0], [0], color="black", lw=2.5, label="Observed"),
    ]


def _plot_reliability_violin(ax, reliability_data, policies):
    pol_ok = [p for p in policies if len(np.asarray(reliability_data.get(p, [])).ravel()) > 0]
    if not pol_ok:
        ax.text(
            0.5,
            0.5,
            "No reliability data\n(Trenton ensemble not run or no target)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.set_axis_off()
        return

    data = [np.asarray(reliability_data[p]).ravel() for p in pol_ok]

    violin = ax.violinplot(
        data,
        positions=np.arange(1, len(pol_ok) + 1),
        widths=0.8,
        showmedians=True,
        showmeans=False,
        showextrema=False,
    )

    for body, policy in zip(violin["bodies"], pol_ok):
        body.set_facecolor(POLICY_COLORS[policy])
        body.set_edgecolor(POLICY_COLORS[policy])
        body.set_alpha(0.22)

    violin["cmedians"].set_color("black")
    violin["cmedians"].set_linewidth(2)

    rng = np.random.default_rng(42)
    for i, policy in enumerate(pol_ok, start=1):
        vals = np.asarray(reliability_data[policy]).ravel()
        x_jitter = rng.normal(i, 0.05, size=len(vals))
        ax.scatter(
            x_jitter,
            vals,
            s=10,
            alpha=0.22,
            color=POLICY_COLORS[policy],
            edgecolors="none",
        )

    ax.set_xticks(range(1, len(pol_ok) + 1))
    ax.set_xticklabels(pol_ok)
    ax.set_ylabel("Reliability\n(% time Trenton ≥ target)")
    ax.set_title("Reliability Distribution", fontsize=13, weight="bold")
    _style_axis(ax, grid=True)


def _plot_reliability_cdf(ax, reliability_data, policies):
    for policy in policies:
        vals = np.sort(np.asarray(reliability_data.get(policy, [])).ravel())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, color=POLICY_COLORS[policy], lw=2.2, label=policy)

    ax.set_xlabel("Reliability")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Reliability CDF", fontsize=13, weight="bold")
    _style_axis(ax, grid=True)


def make_option2_main_figure(
    release_ir_data,
    trenton_time_data,
    trenton_fdc_data,
    reliability_data,
    observed_fdc,
    reservoirs=("Blue Marsh", "Beltzville", "F.E. Walter", "Prompton"),
    policies=("STARFIT", "PWL", "RBF"),
    trenton_target=None,
    observed_ir: Optional[Dict[str, Dict[str, Any]]] = None,
    title="Main Figure: Policy Structure, Trenton Response, and Reliability",
    figsize=(18, 8),
    savepath=None,
):
    fig, axs = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)

    for j, reservoir in enumerate(reservoirs):
        ax = axs[0, j]
        for policy in policies:
            if reservoir not in release_ir_data or policy not in release_ir_data[reservoir]:
                continue
            d = release_ir_data[reservoir][policy]
            _plot_envelope(
                ax=ax,
                x=d["x"],
                median=d["median"],
                q25=d["q25"],
                q75=d["q75"],
                q10=d["q10"],
                q90=d["q90"],
                color=POLICY_COLORS[policy],
                label=policy,
            )

        ax.set_title(reservoir, fontsize=13, weight="bold")
        ax.set_xlabel("Inflow (MGD)")
        if j == 0:
            ax.set_ylabel("Release (MGD)")
        _finish_ir_panel(ax, reservoir, observed_ir)

        if j == len(reservoirs) - 1:
            ax.legend(handles=_ir_legend_handles(), frameon=True, fontsize=10, loc="upper left")

    ax = axs[1, 0]
    for policy in policies:
        if policy not in trenton_time_data:
            continue
        d = trenton_time_data[policy]
        _plot_envelope(
            ax=ax,
            x=d["x"],
            median=d["median"],
            q25=d["q25"],
            q75=d["q75"],
            q10=d["q10"],
            q90=d["q90"],
            color=POLICY_COLORS[policy],
            label=policy,
        )

    if trenton_target is not None:
        ax.axhline(trenton_target, color="black", lw=2, ls="--", label="Trenton target")

    ax.set_title("Trenton Flow Envelope (Time)", fontsize=13, weight="bold")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Flow (MGD)")
    _style_axis(ax, grid=True)

    ax = axs[1, 1]
    for policy in policies:
        if policy not in trenton_fdc_data:
            continue
        d = trenton_fdc_data[policy]
        _plot_envelope(
            ax=ax,
            x=d["x"],
            median=d["median"],
            q25=d["q25"],
            q75=d["q75"],
            q10=d["q10"],
            q90=d["q90"],
            color=POLICY_COLORS[policy],
            label=policy,
        )

    ax.plot(observed_fdc["x"], observed_fdc["y"], color="black", lw=2.5, label="Observed")

    if trenton_target is not None:
        ax.axhline(trenton_target, color="black", lw=1.6, ls="--", alpha=0.85)

    ax.set_yscale("log")
    ax.set_title("Trenton Flow Duration Curve", fontsize=13, weight="bold")
    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel("Flow (MGD)")
    _style_axis(ax, grid=True)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    _plot_reliability_violin(axs[1, 2], reliability_data, policies)
    _plot_reliability_cdf(axs[1, 3], reliability_data, policies)
    axs[1, 3].legend(frameon=True, fontsize=10, loc="lower right")

    fig.suptitle(title, fontsize=17, weight="bold")

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axs


def make_option1_full_figure(
    release_doy_data,
    release_ir_data,
    trenton_time_data,
    trenton_fdc_data,
    reliability_data,
    observed_fdc,
    reservoirs=("Blue Marsh", "Beltzville", "F.E. Walter", "Prompton"),
    policies=("STARFIT", "PWL", "RBF"),
    trenton_target=None,
    observed_ir: Optional[Dict[str, Dict[str, Any]]] = None,
    title="Full Figure: Seasonal Behavior, Policy Structure, Trenton Response, and Reliability",
    figsize=(18, 12),
    savepath=None,
):
    fig, axs = plt.subplots(3, 4, figsize=figsize, constrained_layout=True)

    for j, reservoir in enumerate(reservoirs):
        ax = axs[0, j]
        for policy in policies:
            if reservoir not in release_doy_data or policy not in release_doy_data[reservoir]:
                continue
            d = release_doy_data[reservoir][policy]
            _plot_envelope(
                ax=ax,
                x=d["x"],
                median=d["median"],
                q25=d["q25"],
                q75=d["q75"],
                q10=d["q10"],
                q90=d["q90"],
                color=POLICY_COLORS[policy],
                label=policy,
            )

        ax.set_title(reservoir, fontsize=13, weight="bold")
        ax.set_xlabel("Day of Year")
        if j == 0:
            ax.set_ylabel("Release (MGD)")
        _style_axis(ax, grid=True)

        if j == len(reservoirs) - 1:
            ax.legend(handles=_policy_legend_handles(), frameon=True, fontsize=10, loc="upper left")

    for j, reservoir in enumerate(reservoirs):
        ax = axs[1, j]
        for policy in policies:
            if reservoir not in release_ir_data or policy not in release_ir_data[reservoir]:
                continue
            d = release_ir_data[reservoir][policy]
            _plot_envelope(
                ax=ax,
                x=d["x"],
                median=d["median"],
                q25=d["q25"],
                q75=d["q75"],
                q10=d["q10"],
                q90=d["q90"],
                color=POLICY_COLORS[policy],
                label=policy,
            )

        ax.set_xlabel("Inflow (MGD)")
        if j == 0:
            ax.set_ylabel("Release (MGD)")
        _finish_ir_panel(ax, reservoir, observed_ir)

        if j == len(reservoirs) - 1:
            ax.legend(handles=_ir_legend_handles(), frameon=True, fontsize=10, loc="upper left")

    ax = axs[2, 0]
    for policy in policies:
        if policy not in trenton_time_data:
            continue
        d = trenton_time_data[policy]
        _plot_envelope(
            ax=ax,
            x=d["x"],
            median=d["median"],
            q25=d["q25"],
            q75=d["q75"],
            q10=d["q10"],
            q90=d["q90"],
            color=POLICY_COLORS[policy],
            label=policy,
        )

    if trenton_target is not None:
        ax.axhline(trenton_target, color="black", lw=2, ls="--")

    ax.set_title("Trenton Flow Envelope (Time)", fontsize=13, weight="bold")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Flow (MGD)")
    _style_axis(ax, grid=True)

    ax = axs[2, 1]
    for policy in policies:
        if policy not in trenton_fdc_data:
            continue
        d = trenton_fdc_data[policy]
        _plot_envelope(
            ax=ax,
            x=d["x"],
            median=d["median"],
            q25=d["q25"],
            q75=d["q75"],
            q10=d["q10"],
            q90=d["q90"],
            color=POLICY_COLORS[policy],
            label=policy,
        )

    ax.plot(observed_fdc["x"], observed_fdc["y"], color="black", lw=2.5, label="Observed")

    if trenton_target is not None:
        ax.axhline(trenton_target, color="black", lw=1.6, ls="--", alpha=0.85)

    ax.set_yscale("log")
    ax.set_title("Trenton Flow Duration Curve", fontsize=13, weight="bold")
    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel("Flow (MGD)")
    _style_axis(ax, grid=True)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    _plot_reliability_violin(axs[2, 2], reliability_data, policies)
    _plot_reliability_cdf(axs[2, 3], reliability_data, policies)
    axs[2, 3].legend(frameon=True, fontsize=10, loc="lower right")

    fig.suptitle(title, fontsize=17, weight="bold")

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axs


# =============================================================================
# Orchestration: build envelope dicts + save figures + optional pickle
# =============================================================================


def raw_runs_to_plot_payload(
    raw_by_policy: Dict[str, Any],
    reservoir_keys: Sequence[str],
) -> Dict[str, Any]:
    """
    Convert per-policy Pywr bundles into envelope dicts for
    ``make_option1_full_figure`` / ``make_option2_main_figure``.

    ``observed_ir`` uses the observational training loader (daily inflow–release);
    Q20/Q80 are **inflow** percentiles only, identical for every policy on a reservoir panel.
    """
    release_doy_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    release_ir_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    trenton_time_data: Dict[str, Dict[str, np.ndarray]] = {}
    trenton_fdc_data: Dict[str, Dict[str, np.ndarray]] = {}
    reliability_data: Dict[str, np.ndarray] = {}

    displays = [d for k, d in RESERVOIR_PANEL_ORDER if k in reservoir_keys]
    observed_fdc = {"x": np.array([0.0]), "y": np.array([np.nan])}

    for pol, bundle in raw_by_policy.items():
        dates = bundle.get("dates")
        release_by_res = bundle.get("release_by_res") or {}
        trenton_mat = bundle.get("trenton_mat")
        rels = bundle.get("reliabilities") or []

        if dates is None:
            continue

        dates = pd.DatetimeIndex(dates)

        for internal, display in RESERVOIR_PANEL_ORDER:
            if internal not in reservoir_keys:
                continue
            mat = release_by_res.get(internal)
            if mat is None or mat.size == 0:
                continue
            release_doy_data.setdefault(display, {})[pol] = envelope_doy_from_matrix(dates, mat)
            inf_aln = _inflow_series_on_dates(internal, dates)
            Tr = min(len(inf_aln), mat.shape[0])
            release_ir_data.setdefault(display, {})[pol] = envelope_ir_binned(
                inf_aln[:Tr], mat[:Tr, :]
            )

        if trenton_mat is not None and trenton_mat.size and np.isfinite(trenton_mat).any():
            trenton_time_data[pol] = envelope_doy_from_matrix(dates, trenton_mat)
            trenton_fdc_data[pol] = envelope_fdc_from_matrix(trenton_mat)

        reliability_data[pol] = np.asarray(rels, dtype=float)

    for bundle in raw_by_policy.values():
        o = bundle.get("obs_trenton")
        if o is not None and len(o):
            observed_fdc = observed_fdc_dict(o)
            break

    tgt_raw = os.environ.get("CEE_TRENTON_TARGET_MGD", "").strip()
    trenton_target = float(tgt_raw) if tgt_raw else float(DEFAULT_TRENTON_TARGET_MGD)

    policies_out = tuple(p for p in POLICY_ORDER if p in raw_by_policy)
    observed_ir = build_observed_ir_by_display(reservoir_keys)

    return {
        "release_doy_data": release_doy_data,
        "release_ir_data": release_ir_data,
        "trenton_time_data": trenton_time_data,
        "trenton_fdc_data": trenton_fdc_data,
        "reliability_data": reliability_data,
        "observed_fdc": observed_fdc,
        "observed_ir": observed_ir,
        "trenton_target": trenton_target,
        "reservoirs": tuple(displays),
        "reservoir_keys_internal": tuple(reservoir_keys),
        "policies": policies_out,
    }


def build_pareto_ensemble_figure_suite(
    fig_root: str,
    solution_vars: Dict[str, Dict[str, pd.DataFrame]],
    reservoirs_for_alignment: Sequence[str],
    pywr_inflow_type: str = "pub_nhmv10_BC_withObsScaled",
) -> None:
    """
    Load or compute ensemble summaries, then write Option 2 (main) and Option 1 (full) PNGs.

    Resolution order:

    1. ``CEE_PARETO_SIM_BUNDLE`` — raw Pywr ensemble pickle from ``run_pareto_simulations simulate --mode ensemble``
    2. ``CEE_PARETO_ENSEMBLE_SUMMARY_PATH`` — plot-ready envelope pickle (legacy)
    3. Live Pywr runs from ``solution_vars`` (requires Pywr)

    Environment:
      * ``CEE_PARETO_SIM_BUNDLE``, ``CEE_PARETO_ENSEMBLE_SUMMARY_PATH``
      * ``CEE_PARETO_ENSEMBLE_SAVE_SUMMARY`` — save envelope bundle next to figures (when computing)
      * ``CEE_TRENTON_ENSEMBLE_MAX_RUNS``, ``CEE_TRENTON_TARGET_MGD``, ``CEE_PYWR_FLOW_PREDICTION_MODE``
    """
    out_dir = os.path.join(fig_root, "fig12_pareto_uncertainty")
    os.makedirs(out_dir, exist_ok=True)

    bundle_path = os.environ.get("CEE_PARETO_SIM_BUNDLE", "").strip()
    if bundle_path and os.path.isfile(bundle_path):
        from methods.ensemble.pareto_simulation_cache import load_pareto_sim_bundle

        data = load_pareto_sim_bundle(bundle_path)
        if data.get("kind") != "ensemble_pywr":
            raise ValueError(
                f"CEE_PARETO_SIM_BUNDLE must be kind=ensemble_pywr; got {data.get('kind')!r}"
            )
        meta = data.get("meta", {})
        rk = meta.get("reservoir_keys") or list(reservoirs_for_alignment)
        payload = raw_runs_to_plot_payload(data["raw_by_policy"], rk)
        print(f"[Fig12] loaded Pywr simulation bundle: {bundle_path}")
    else:
        cache_path = os.environ.get("CEE_PARETO_ENSEMBLE_SUMMARY_PATH", "").strip()
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            print(f"[Fig12] loaded ensemble summary from {cache_path}")
            if not payload.get("observed_ir"):
                rk = list(payload.get("reservoir_keys_internal") or [])
                if not rk:
                    rk = [k for k, _ in RESERVOIR_PANEL_ORDER if k in reservoirs_for_alignment]
                payload["observed_ir"] = build_observed_ir_by_display(rk)
        else:
            payload = _compute_ensemble_payload(
                solution_vars, reservoirs_for_alignment, pywr_inflow_type
            )
            if os.environ.get("CEE_PARETO_ENSEMBLE_SAVE_SUMMARY", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                pkl = os.path.join(out_dir, "pareto_ensemble_envelopes.pkl")
                with open(pkl, "wb") as f:
                    pickle.dump(payload, f)
                print(f"[Fig12] saved envelope summary {pkl}")

    reservoirs = payload.get("reservoirs") or tuple(d for _, d in RESERVOIR_PANEL_ORDER)
    policies = payload.get("policies") or POLICY_ORDER
    policies = tuple(p for p in POLICY_ORDER if p in policies)

    tgt = payload.get("trenton_target")
    if tgt is not None and not np.isfinite(tgt):
        tgt = None

    opt2 = os.path.join(out_dir, "fig12_main_option2_pareto_ensemble.png")
    make_option2_main_figure(
        release_ir_data=payload["release_ir_data"],
        trenton_time_data=payload["trenton_time_data"],
        trenton_fdc_data=payload["trenton_fdc_data"],
        reliability_data=payload["reliability_data"],
        observed_fdc=payload["observed_fdc"],
        reservoirs=reservoirs,
        policies=policies,
        trenton_target=tgt,
        observed_ir=payload.get("observed_ir"),
        savepath=opt2,
    )
    plt.close("all")
    print(f"[Fig12] saved {opt2}")

    opt1 = os.path.join(out_dir, "fig12_full_option1_pareto_ensemble.png")
    make_option1_full_figure(
        release_doy_data=payload["release_doy_data"],
        release_ir_data=payload["release_ir_data"],
        trenton_time_data=payload["trenton_time_data"],
        trenton_fdc_data=payload["trenton_fdc_data"],
        reliability_data=payload["reliability_data"],
        observed_fdc=payload["observed_fdc"],
        reservoirs=reservoirs,
        policies=policies,
        trenton_target=tgt,
        observed_ir=payload.get("observed_ir"),
        savepath=opt1,
    )
    plt.close("all")
    print(f"[Fig12] saved {opt1}")


def _compute_ensemble_payload(
    solution_vars: Dict[str, Dict[str, pd.DataFrame]],
    reservoirs_for_alignment: Sequence[str],
    pywr_inflow_type: str,
) -> Dict[str, Any]:
    max_tr = _max_trenton_runs()

    res_keys = [k for k, _ in RESERVOIR_PANEL_ORDER if k in reservoirs_for_alignment]
    policies_avail = [p for p in POLICY_ORDER if any(p in solution_vars.get(r, {}) for r in res_keys)]
    if not policies_avail:
        raise RuntimeError("[Fig12] No policy CSVs for configured reservoirs.")

    raw_by_policy: Dict[str, Any] = {}
    for pol in policies_avail:
        b = collect_pywr_ensemble_for_policy(
            pol, solution_vars, list(res_keys), pywr_inflow_type, max_tr
        )
        if b is not None:
            raw_by_policy[pol] = b

    if not raw_by_policy:
        raise RuntimeError(
            "[Fig12] Pywr ensemble produced no results. "
            "Check Borg CSVs, Pywr install, CEE_SKIP_PYWR, and CEE_PYWR_WORK_DIR."
        )

    return raw_runs_to_plot_payload(raw_by_policy, res_keys)


# Backwards-compatible name for stage1
build_figure12_pareto_uncertainty = build_pareto_ensemble_figure_suite

# Explicit name for notebooks: raw Pywr bundles -> envelope dicts for Option 1/2
summarize_raw_runs_to_envelope_dicts = raw_runs_to_plot_payload
