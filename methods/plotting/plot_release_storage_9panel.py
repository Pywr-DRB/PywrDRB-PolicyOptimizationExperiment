#!/usr/bin/env python3
from pathlib import Path
from typing import Tuple, Optional, Sequence, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from methods.config import reservoir_capacity
from methods.plotting.theme import SERIES_COLORS, SERIES_LINESTYLES

# ===== Fixed palette & styles for consistency across all panels =====
PALETTE = {
    "sim": SERIES_COLORS["parametric"],
    "sim2": SERIES_COLORS["independent"],
    "obs": SERIES_COLORS["observed"],
    "pywr_param": SERIES_COLORS["parametric"],
    "pywr_default": SERIES_COLORS["default"],
    "nor": "0.45",
}
LW = {  # line widths
    "sim": 1.05,
    "sim2": 1.0,
    "obs": 1.0,
    "pywr_param": 0.7,
    "pywr_default": 0.7,
    "nor": 0.8,
}
ALPHA = {  # opacities
    "sim": 1.0,
    "obs": 0.95,
    "pywr_param": 0.95,
    "pywr_default": 0.95,
    "nor": 1.0,
}
MS = 3  # marker size for monthly/annual points

# ---------- helpers ----------
def to_percent_storage(s: pd.Series, reservoir: str) -> pd.Series:
    cap = float(reservoir_capacity[reservoir])
    return 100.0 * (pd.to_numeric(s, errors="coerce") / cap)

def _fdc(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty: return np.array([]), np.array([])
    vals = np.sort(s.values)[::-1]
    p = np.linspace(0, 100, len(vals))
    return p, vals

def _fdc_positive(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    p, v = _fdc(series); keep = v > 0
    return p[keep], v[keep]

def _safe_monthly_means(s: Optional[pd.Series]) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(index=range(1,13), dtype=float)
    m = s.resample("ME").mean()
    mg = m.groupby(m.index.month).mean()
    out = pd.Series(index=range(1,13), dtype=float); out.loc[mg.index] = mg.values
    return out

def _safe_annual_means(s: Optional[pd.Series]) -> pd.Series:
    if s is None or s.empty: return pd.Series(dtype=float)
    y = s.resample("YE").mean(); y.index = y.index.year
    return y

def _nice_datetime_axis(ax):
    loc = AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    for label in ax.get_xticklabels():
        label.set_rotation(0)

def _restrict_to_overlapping_dates(series_list):
    present = [s for s in series_list if s is not None]
    if len(present) <= 1:
        return series_list

    idx = present[0].index
    for s in present[1:]:
        idx = idx.intersection(s.index)
    idx = idx.sort_values()
    if len(idx) == 0:
        return [None for _ in series_list]
    return [s.loc[idx] if s is not None else None for s in series_list]

# ---------- main ----------
def plot_release_storage_9panel(
    *,
    reservoir: str,
    # independent sim (required)
    sim_release: pd.Series,            # DatetimeIndex, MGD
    sim_storage_MG: pd.Series,         # DatetimeIndex, MG
    # observations (optional)
    obs_release: Optional[pd.Series] = None,
    obs_storage_MG: Optional[pd.Series] = None,
    # optional overlays from Pywr runs
    pywr_param_release: Optional[pd.Series] = None,
    pywr_param_storage_MG: Optional[pd.Series] = None,
    pywr_default_release: Optional[pd.Series] = None,
    pywr_default_storage_MG: Optional[pd.Series] = None,
    # optional NOR bands (% capacity; aligned to DatetimeIndex)
    nor_lo_pct: Optional[pd.Series] = None,
    nor_hi_pct: Optional[pd.Series] = None,
    # labels / window
    start: str = None,
    end: str = None,
    ylabel: str = "Flow (MGD)",
    storage_ylabel: str = "Storage (% cap)",
    policy_label: Optional[str] = None,
    pick_label: Optional[str] = None,
    # Reproducibility: same θ used for independent sim and Pywr (Param)
    param_vector: Optional[Union[Sequence[float], np.ndarray]] = None,
    # Optional second simulated series (e.g. independent model) overlaid on all panels
    secondary_release: Optional[pd.Series] = None,
    secondary_storage_MG: Optional[pd.Series] = None,
    sim_label: str = "Simulated (Pywr-DRB parametric)",
    secondary_sim_label: str = "Simulated (independent reservoir model)",
    obs_label: str = "Observed",
    run_type_note: Optional[str] = None,
    save_path: Optional[str] = None,
):
    # time windowing
    if start is not None and end is not None:
        slicer = slice(start, end)
        sim_release = sim_release.loc[slicer]
        sim_storage_MG = sim_storage_MG.loc[slicer]
        if obs_release is not None:
            obs_release = obs_release.loc[slicer]
        if obs_storage_MG is not None:
            obs_storage_MG = obs_storage_MG.loc[slicer]
        if pywr_param_release is not None:
            pywr_param_release = pywr_param_release.loc[slicer]
        if pywr_param_storage_MG is not None:
            pywr_param_storage_MG = pywr_param_storage_MG.loc[slicer]
        if pywr_default_release is not None:
            pywr_default_release = pywr_default_release.loc[slicer]
        if pywr_default_storage_MG is not None:
            pywr_default_storage_MG = pywr_default_storage_MG.loc[slicer]
        if nor_lo_pct is not None:
            nor_lo_pct = nor_lo_pct.loc[slicer]
        if nor_hi_pct is not None:
            nor_hi_pct = nor_hi_pct.loc[slicer]
        if secondary_release is not None:
            secondary_release = secondary_release.loc[slicer]
        if secondary_storage_MG is not None:
            secondary_storage_MG = secondary_storage_MG.loc[slicer]

    (
        sim_release,
        obs_release,
        pywr_param_release,
        pywr_default_release,
        secondary_release,
    ) = _restrict_to_overlapping_dates(
        [
            sim_release,
            obs_release,
            pywr_param_release,
            pywr_default_release,
            secondary_release,
        ]
    )
    (
        sim_storage_MG,
        obs_storage_MG,
        pywr_param_storage_MG,
        pywr_default_storage_MG,
        secondary_storage_MG,
    ) = _restrict_to_overlapping_dates(
        [
            sim_storage_MG,
            obs_storage_MG,
            pywr_param_storage_MG,
            pywr_default_storage_MG,
            secondary_storage_MG,
        ]
    )
    if nor_lo_pct is not None and sim_storage_MG is not None:
        nor_lo_pct = nor_lo_pct.reindex(sim_storage_MG.index)
    if nor_hi_pct is not None and sim_storage_MG is not None:
        nor_hi_pct = nor_hi_pct.reindex(sim_storage_MG.index)

    if sim_release is None or sim_storage_MG is None or len(sim_release) == 0 or len(sim_storage_MG) == 0:
        print(
            f"[plot_release_storage_9panel] Skip {reservoir} ({policy_label}, {pick_label}): "
            "no overlapping dates across requested series."
        )
        return

    # unit conversions
    sim_s_pct = to_percent_storage(sim_storage_MG, reservoir)
    obs_s_pct = to_percent_storage(obs_storage_MG, reservoir) if obs_storage_MG is not None else None
    sec_s_pct = (
        to_percent_storage(secondary_storage_MG, reservoir) if secondary_storage_MG is not None else None
    )
    pywr_param_s_pct   = to_percent_storage(pywr_param_storage_MG, reservoir)   if pywr_param_storage_MG is not None else None
    pywr_default_s_pct = to_percent_storage(pywr_default_storage_MG, reservoir) if pywr_default_storage_MG is not None else None

    # monthly/annual aggregates
    m_sim_r = _safe_monthly_means(sim_release);       m_sim_s = _safe_monthly_means(sim_s_pct)
    m_sec_r = _safe_monthly_means(secondary_release) if secondary_release is not None else None
    m_sec_s = _safe_monthly_means(sec_s_pct) if sec_s_pct is not None else None
    m_obs_r = _safe_monthly_means(obs_release);       m_obs_s = _safe_monthly_means(obs_s_pct)
    y_sim_r = _safe_annual_means(sim_release);        y_sim_s = _safe_annual_means(sim_s_pct)
    y_sec_r = _safe_annual_means(secondary_release) if secondary_release is not None else None
    y_sec_s = _safe_annual_means(sec_s_pct) if sec_s_pct is not None else None
    y_obs_r = _safe_annual_means(obs_release);        y_obs_s = _safe_annual_means(obs_s_pct)

    # FDCs
    fdc_d_sim = _fdc_positive(sim_release);  fdc_m_sim = _fdc_positive(m_sim_r);  fdc_y_sim = _fdc_positive(y_sim_r)
    fdc_d_sec = _fdc_positive(secondary_release) if secondary_release is not None else (np.array([]), np.array([]))
    fdc_m_sec = _fdc_positive(m_sec_r) if m_sec_r is not None else (np.array([]), np.array([]))
    fdc_y_sec = _fdc_positive(y_sec_r) if y_sec_r is not None else (np.array([]), np.array([]))
    fdc_d_obs = _fdc_positive(obs_release) if obs_release is not None else (np.array([]), np.array([]))
    fdc_m_obs = _fdc_positive(m_obs_r)     if obs_release is not None else (np.array([]), np.array([]))
    fdc_y_obs = _fdc_positive(y_obs_r)     if obs_release is not None else (np.array([]), np.array([]))

    # canvas
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    # ------ Row 1: Daily ------
    # Storage (daily)
    ax = axs[0,0]
    ax.plot(
        sim_s_pct.index, sim_s_pct.values,
            label=sim_label, lw=LW["sim"], color=PALETTE["sim"], alpha=ALPHA["sim"], ls=SERIES_LINESTYLES["parametric"]
    )
    if sec_s_pct is not None:
        ax.plot(
            sec_s_pct.index, sec_s_pct.values,
            label=secondary_sim_label, lw=LW["sim2"], color=PALETTE["sim2"], alpha=0.95, ls=SERIES_LINESTYLES["independent"]
        )
    if obs_s_pct is not None:
        ax.plot(
            obs_s_pct.index, obs_s_pct.values,
            label=obs_label, lw=LW["obs"], color=PALETTE["obs"], alpha=ALPHA["obs"], ls=SERIES_LINESTYLES["observed"]
        )
    if pywr_param_s_pct is not None:
        ax.plot(
            pywr_param_s_pct.index, pywr_param_s_pct.values,
            label="Pywr (Param)", lw=LW["pywr_param"], color=PALETTE["pywr_param"], alpha=ALPHA["pywr_param"], ls=SERIES_LINESTYLES["parametric"]
        )
    if pywr_default_s_pct is not None:
        ax.plot(
            pywr_default_s_pct.index, pywr_default_s_pct.values,
            label="Pywr (Default)", lw=LW["pywr_default"], color=PALETTE["pywr_default"], alpha=ALPHA["pywr_default"], ls=SERIES_LINESTYLES["default"]
        )
    if nor_lo_pct is not None:
        ax.plot(
            nor_lo_pct.index, nor_lo_pct.values,
            ls=":", lw=LW["nor"], color=PALETTE["nor"], label="NOR lo"
        )
    if nor_hi_pct is not None:
        ax.plot(
            nor_hi_pct.index, nor_hi_pct.values,
            ls=":", lw=LW["nor"], color=PALETTE["nor"], label="NOR hi"
        )
    ax.set_title("Daily Storage (%)"); ax.set_ylabel(storage_ylabel); ax.grid(True, alpha=0.3); ax.margins(y=0.06); _nice_datetime_axis(ax)

    # Release (daily)
    ax = axs[0,1]
    if obs_release is not None:
        ax.plot(
            obs_release.index, obs_release.values,
            label=obs_label, lw=LW["obs"], color=PALETTE["obs"], alpha=ALPHA["obs"], ls=SERIES_LINESTYLES["observed"]
        )
    ax.plot(
        sim_release.index, sim_release.values,
        label=sim_label, lw=LW["sim"], color=PALETTE["sim"], alpha=ALPHA["sim"], ls=SERIES_LINESTYLES["parametric"]
    )
    if secondary_release is not None:
        ax.plot(
            secondary_release.index, secondary_release.values,
            label=secondary_sim_label, lw=LW["sim2"], color=PALETTE["sim2"], alpha=0.95, ls=SERIES_LINESTYLES["independent"]
        )
    if pywr_param_release is not None:
        ax.plot(
            pywr_param_release.index, pywr_param_release.values,
            label="Pywr (Param)", lw=LW["pywr_param"], color=PALETTE["pywr_param"], alpha=ALPHA["pywr_param"], ls=SERIES_LINESTYLES["parametric"]
        )
    if pywr_default_release is not None:
        ax.plot(
            pywr_default_release.index, pywr_default_release.values,
            label="Pywr (Default)", lw=LW["pywr_default"], color=PALETTE["pywr_default"], alpha=ALPHA["pywr_default"], ls=SERIES_LINESTYLES["default"]
        )
    ax.set_title("Daily Release"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.margins(y=0.06); _nice_datetime_axis(ax)

    # FDC daily (release)
    ax = axs[0,2]
    if len(fdc_d_obs[0]):
        ax.plot(
            *fdc_d_obs, label=obs_label, lw=LW["obs"], color=PALETTE["obs"], alpha=ALPHA["obs"]
        )
    if len(fdc_d_sim[0]):
        ax.plot(
            *fdc_d_sim, label=sim_label, lw=LW["sim"], color=PALETTE["sim"], alpha=ALPHA["sim"], ls=SERIES_LINESTYLES["parametric"]
        )
    if len(fdc_d_sec[0]):
        ax.plot(
            *fdc_d_sec, label=secondary_sim_label, lw=LW["sim2"], color=PALETTE["sim2"], alpha=0.95, ls=SERIES_LINESTYLES["independent"]
        )
    ax.set_title("Daily Release FDC"); ax.set_xlabel("Exceedance (%)"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.margins(y=0.06)

    # ------ Row 2: Monthly means ------
    # Storage (monthly avg)
    ax = axs[1,0]
    ax.plot(
        month_labels, _safe_monthly_means(sim_s_pct).values,
        label=sim_label, marker="o", lw=LW["sim"], ms=MS, color=PALETTE["sim"], alpha=ALPHA["sim"]
    )
    if m_sec_s is not None and m_sec_s.notna().any():
        ax.plot(
            month_labels, m_sec_s.values,
            label=secondary_sim_label, marker="s", lw=LW["sim2"], ms=MS, color=PALETTE["sim2"], alpha=0.95, ls="-."
        )
    if obs_s_pct is not None:
        ax.plot(
            month_labels, _safe_monthly_means(obs_s_pct).values,
            label=obs_label, marker="o", lw=LW["obs"], ms=MS, color=PALETTE["obs"], alpha=ALPHA["obs"], ls=SERIES_LINESTYLES["observed"]
        )
    if pywr_param_s_pct is not None:
        ax.plot(
            month_labels, _safe_monthly_means(pywr_param_s_pct).values,
            label="Pywr (Param)", marker="o", lw=LW["pywr_param"], ms=MS, color=PALETTE["pywr_param"], alpha=ALPHA["pywr_param"], ls=SERIES_LINESTYLES["parametric"]
        )
    if pywr_default_s_pct is not None:
        ax.plot(
            month_labels, _safe_monthly_means(pywr_default_s_pct).values,
            label="Pywr (Default)", marker="o", lw=LW["pywr_default"], ms=MS, color=PALETTE["pywr_default"], alpha=ALPHA["pywr_default"], ls=SERIES_LINESTYLES["default"]
        )
    ax.set_title("Monthly Avg Storage"); ax.set_ylabel(storage_ylabel); ax.grid(True, alpha=0.3); ax.margins(y=0.06)

    # Release (monthly avg)
    ax = axs[1,1]
    if obs_release is not None:
        ax.plot(
            month_labels, _safe_monthly_means(obs_release).values,
            label=obs_label, marker="o", lw=LW["obs"], ms=MS, color=PALETTE["obs"], alpha=ALPHA["obs"], ls=SERIES_LINESTYLES["observed"]
        )
    ax.plot(
        month_labels, m_sim_r.values,
        label=sim_label, ls=SERIES_LINESTYLES["parametric"], marker="o", lw=LW["sim"], ms=MS, color=PALETTE["sim"], alpha=ALPHA["sim"]
    )
    if m_sec_r is not None and m_sec_r.notna().any():
        ax.plot(
            month_labels, m_sec_r.values,
            label=secondary_sim_label, ls=SERIES_LINESTYLES["independent"], marker="s", lw=LW["sim2"], ms=MS, color=PALETTE["sim2"], alpha=0.95
        )
    if pywr_param_release is not None:
        ax.plot(
            month_labels, _safe_monthly_means(pywr_param_release).values,
            label="Pywr (Param)", marker="o", lw=LW["pywr_param"], ms=MS, color=PALETTE["pywr_param"], alpha=ALPHA["pywr_param"], ls=SERIES_LINESTYLES["parametric"]
        )
    if pywr_default_release is not None:
        ax.plot(
            month_labels, _safe_monthly_means(pywr_default_release).values,
            label="Pywr (Default)", marker="o", lw=LW["pywr_default"], ms=MS, color=PALETTE["pywr_default"], alpha=ALPHA["pywr_default"], ls=SERIES_LINESTYLES["default"]
        )
    ax.set_title("Monthly Avg Release"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.margins(y=0.06)

    # Monthly FDC (release)
    ax = axs[1,2]
    if len(fdc_m_obs[0]):
        ax.plot(
            *fdc_m_obs, label=obs_label, lw=LW["obs"], color=PALETTE["obs"], alpha=ALPHA["obs"]
        )
    if len(fdc_m_sim[0]):
        ax.plot(
            *fdc_m_sim, label=sim_label, ls=SERIES_LINESTYLES["parametric"], lw=LW["sim"], color=PALETTE["sim"], alpha=ALPHA["sim"]
        )
    if len(fdc_m_sec[0]):
        ax.plot(
            *fdc_m_sec, label=secondary_sim_label, ls=SERIES_LINESTYLES["independent"], lw=LW["sim2"], color=PALETTE["sim2"], alpha=0.95
        )
    ax.set_title("Monthly Release FDC"); ax.set_xlabel("Exceedance (%)"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.margins(y=0.06)

    # ------ Row 3: Annual means ------
    # Storage (annual avg)
    ax = axs[2,0]
    ax.plot(
        y_sim_s.index, y_sim_s.values,
        label=sim_label, marker="o", lw=LW["sim"], ms=MS, color=PALETTE["sim"], alpha=ALPHA["sim"]
    )
    if y_sec_s is not None and not y_sec_s.empty:
        ax.plot(
            y_sec_s.index, y_sec_s.values,
            label=secondary_sim_label, marker="s", lw=LW["sim2"], ms=MS, color=PALETTE["sim2"], alpha=0.95, ls="-."
        )
    y_obs_s_safe = _safe_annual_means(obs_s_pct)
    if not y_obs_s_safe.empty:
        ax.plot(
            y_obs_s_safe.index, y_obs_s_safe.values,
            label=obs_label, marker="o", lw=LW["obs"], ms=MS, color=PALETTE["obs"], alpha=ALPHA["obs"], ls=SERIES_LINESTYLES["observed"]
        )
    if pywr_param_s_pct is not None:
        y_pp = _safe_annual_means(pywr_param_s_pct)
        ax.plot(
            y_pp.index, y_pp.values,
            label="Pywr (Param)", marker="o", lw=LW["pywr_param"], ms=MS, color=PALETTE["pywr_param"], alpha=ALPHA["pywr_param"], ls=SERIES_LINESTYLES["parametric"]
        )
    if pywr_default_s_pct is not None:
        y_pd = _safe_annual_means(pywr_default_s_pct)
        ax.plot(
            y_pd.index, y_pd.values,
            label="Pywr (Default)", marker="o", lw=LW["pywr_default"], ms=MS, color=PALETTE["pywr_default"], alpha=ALPHA["pywr_default"], ls=SERIES_LINESTYLES["default"]
        )
    ax.set_title("Annual Avg Storage"); ax.set_xlabel("Year"); ax.set_ylabel(storage_ylabel); ax.grid(True, alpha=0.3); ax.margins(y=0.06)

    # Release (annual avg)
    ax = axs[2,1]
    if not y_obs_r.empty:
        ax.plot(
            y_obs_r.index, y_obs_r.values,
            label=obs_label, marker="o", lw=LW["obs"], ms=MS, color=PALETTE["obs"], alpha=ALPHA["obs"], ls=SERIES_LINESTYLES["observed"]
        )
    ax.plot(
        y_sim_r.index, y_sim_r.values,
        label=sim_label, ls=SERIES_LINESTYLES["parametric"], marker="o", lw=LW["sim"], ms=MS, color=PALETTE["sim"], alpha=ALPHA["sim"]
    )
    if y_sec_r is not None and not y_sec_r.empty:
        ax.plot(
            y_sec_r.index, y_sec_r.values,
            label=secondary_sim_label, ls=SERIES_LINESTYLES["independent"], marker="s", lw=LW["sim2"], ms=MS, color=PALETTE["sim2"], alpha=0.95
        )
    if pywr_param_release is not None:
        y_pr = _safe_annual_means(pywr_param_release)
        ax.plot(
            y_pr.index, y_pr.values,
            label="Pywr (Param)", marker="o", lw=LW["pywr_param"], ms=MS, color=PALETTE["pywr_param"], alpha=ALPHA["pywr_param"], ls=SERIES_LINESTYLES["parametric"]
        )
    if pywr_default_release is not None:
        y_pd = _safe_annual_means(pywr_default_release)
        ax.plot(
            y_pd.index, y_pd.values,
            label="Pywr (Default)", marker="o", lw=LW["pywr_default"], ms=MS, color=PALETTE["pywr_default"], alpha=ALPHA["pywr_default"], ls=SERIES_LINESTYLES["default"]
        )
    ax.set_title("Annual Avg Release"); ax.set_xlabel("Year"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3); ax.margins(y=0.06)

    # Annual FDC (release)
    ax = axs[2,2]
    if len(fdc_y_obs[0]):
        ax.plot(
            *fdc_y_obs, label=obs_label, lw=LW["obs"], color=PALETTE["obs"], alpha=ALPHA["obs"]
        )
    if len(fdc_y_sim[0]):
        ax.plot(
            *fdc_y_sim, label=sim_label, ls=SERIES_LINESTYLES["parametric"], lw=LW["sim"], color=PALETTE["sim"], alpha=ALPHA["sim"]
        )
    if len(fdc_y_sec[0]):
        ax.plot(
            *fdc_y_sec, label=secondary_sim_label, ls=SERIES_LINESTYLES["independent"], lw=LW["sim2"], color=PALETTE["sim2"], alpha=0.95
        )
    ax.set_title("Annual Release FDC"); ax.set_xlabel("Exceedance (%)"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.margins(y=0.06)

    # ----- Title + shared legend -----
    effective_start = None
    effective_end = None
    if len(sim_release.index):
        effective_start = str(sim_release.index.min().date())
        effective_end = str(sim_release.index.max().date())

    title_bits = [reservoir]
    if policy_label: title_bits.append(policy_label)
    if pick_label:   title_bits.append(pick_label)
    if effective_start and effective_end:
        title_bits.append(f"{effective_start} to {effective_end}")
    elif start and end:
        title_bits.append(f"{start} to {end}")
    fig = plt.gcf()
    _st = " — ".join(title_bits)
    if run_type_note:
        _st = _st + "\n" + run_type_note
    fig.suptitle(_st, fontsize=14, weight="bold")

    # Small reference: pick name + θ (same vector for independent sim and Pywr) for reproducibility
    ref_lines = []
    if pick_label:
        ref_lines.append(f"Pick: {pick_label}")
    if param_vector is not None and len(param_vector) > 0:
        arr = np.asarray(param_vector, dtype=float)
        n_show = min(12, len(arr))
        theta_str = ", ".join(f"{x:.4g}" for x in arr[:n_show])
        if len(arr) > n_show:
            theta_str += f", ... ({len(arr)} params)"
        ref_lines.append("θ (indep. & Pywr): [" + theta_str + "]")
    if ref_lines:
        fig.text(0.02, 0.02, "\n".join(ref_lines), fontsize=7, family="monospace",
                 verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    # shared legend
    handles, labels = [], []
    for ax in axs.ravel():
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in labels:
                handles.append(hh); labels.append(ll)
        if ax.get_legend() is not None:
            ax.legend_.remove()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            framealpha=0.92,
            title="Series",
            ncol=min(5, max(2, len(labels))),
        )

    fig.tight_layout(rect=[0, 0.12, 1.0, 0.90])  # leave bottom margin for legend + θ ref box

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
