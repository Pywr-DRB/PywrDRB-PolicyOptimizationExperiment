#!/usr/bin/env python3
# methods/plotting/plot_dynamics_2x1.py

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from methods.metrics.objectives import ObjectiveCalculator
from methods.plotting.theme import SERIES_COLORS, SERIES_LINESTYLES


def _release_nse_vs_obs(obs: pd.Series, sim: pd.Series) -> float:
    """NSE from ObjectiveCalculator (same as optimization release neg_nse)."""
    idx = obs.index.intersection(sim.index)
    o = pd.to_numeric(obs.loc[idx], errors="coerce").astype(float)
    s = pd.to_numeric(sim.loc[idx], errors="coerce").astype(float)
    m = o.notna() & s.notna()
    o = o.loc[m].to_numpy(dtype=np.float64)
    s = s.loc[m].to_numpy(dtype=np.float64)
    if len(o) < 3:
        return float("nan")
    oc = ObjectiveCalculator(metrics=["neg_nse"])
    neg = oc.calculate(obs=o, sim=s)[0]
    return float(-neg)


# ---------- helpers ----------
def _robust_limits(series_list, lo: float = 0.2, hi: float = 99.8, pad: float = 0.12):
    """Percentile-based limits with extra padding to avoid clipping."""
    vals = pd.concat([pd.Series(s, dtype=float) for s in series_list if s is not None], axis=0)
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    ql, qh = np.nanpercentile(vals.values, [lo, hi])
    span = max(qh - ql, 1e-9)
    return (ql - pad * span, qh + pad * span)


def _positive_log_limits(series_list, lo: float = 0.2, hi: float = 99.8, pad: float = 0.12):
    """Percentile limits using only positive values; returns (lo, hi) or None."""
    vals = []
    for s in series_list:
        if s is None:
            continue
        v = (
            pd.Series(s, dtype=float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        v = v[v > 0]
        if v.size:
            vals.append(v)
    if not vals:
        return None
    vals = np.concatenate(vals)
    ql, qh = np.nanpercentile(vals, [lo, hi])
    span = max(qh - ql, 1e-9)
    ql = max(ql * 0.8, 1e-6)  # keep lower bound above 0 on log scale
    return (ql, qh + pad * span)


def _mask_nonpositive(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None:
        return None
    out = s.astype(float).copy()
    out[out <= 0] = np.nan
    return out


def _apply_scale_then_limits(
    ax,
    values_concat: np.ndarray,
    scale: Optional[str],
    ylims: Optional[Tuple[float, float]],
    linthresh: float,
):
    """Set the y-scale first, then apply limits (so ticks are computed in that space)."""
    if scale in (None, "linear"):
        pass
    elif scale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    elif scale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh)
    else:
        raise ValueError("yscale must be None|'linear'|'log'|'symlog'")
    if ylims is not None:
        ax.set_ylim(*ylims)


def _downsample(s: Optional[pd.Series], step: Optional[int]) -> Optional[pd.Series]:
    if s is None or step is None or step <= 1:
        return s
    return s.iloc[::step]

def _release_fdc(series: Optional[pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
    if series is None:
        return np.array([]), np.array([])
    vals = (
        pd.Series(series, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
    )
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.array([]), np.array([])
    vals = np.sort(vals)[::-1]
    exceed = np.linspace(0, 100, vals.size)
    return exceed, vals


# ---------- main ----------
def plot_2x1_dynamics(
    reservoir: str,
    policy: str,
    indie_R: pd.Series, indie_S: pd.Series,
    pywr_R: pd.Series,  pywr_S: pd.Series,
    def_R: Optional[pd.Series] = None,   def_S: Optional[pd.Series] = None,
    obs_R: Optional[pd.Series] = None,
    obs_S: Optional[pd.Series] = None,
    date_label: Optional[str] = None,
    ylims_storage: Optional[Tuple[float, float]] = None,
    ylims_release: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    # display knobs
    yscale_storage: Optional[str] = None,      # None | 'linear' | 'log' | 'symlog'
    yscale_release: Optional[str] = None,      # None | 'linear' | 'log' | 'symlog'
    linthresh_release: float = 10.0,           # used only for symlog
    max_date_ticks: int = 10,
    downsample_step: Optional[int] = 1,
    line_width: float = 1.35,
    pick_label: Optional[str] = None,
    param_text: Optional[str] = None,
    show_release_nse_panel: bool = True,
    add_release_fdc_panel: bool = False,
    storage_as_pct_capacity: bool = False,
    capacity_mg: float | None = None,
    nor_lo_pct: Optional[pd.Series] = None,
    nor_hi_pct: Optional[pd.Series] = None,
    parametric_label: str = "Pywr-DRB Parametric",
):
    """
    2×1 overlay:
      top  = Storage (MG), or % of capacity if ``storage_as_pct_capacity`` and ``capacity_mg``
      bot  = Release/Flow (MGD)

    Series: Observed (if provided), Independent, Pywr-DRB Parametric, Pywr-DRB Default.

    Readability:
      - line_width / downsample_step control visibility of dense daily series
      - relaxed auto-limits (0.2..99.8 pctile + 12% padding)
      - optional right column with release NSE vs observations
    """
    lw_obs = max(line_width * 1.1, 1.0)
    lw_sim = line_width

    # aligned index across all series
    idx = indie_R.index
    for s in (pywr_R, def_R, obs_R, indie_S, pywr_S, def_S, obs_S):
        if s is not None:
            idx = idx.intersection(s.index)
    idx = idx.sort_values()
    if len(idx) == 0:
        print(f"[Plot] {reservoir}/{policy}: no overlap for 2x1 dynamics; skipping.")
        return

    # align
    R1, R2 = indie_R.reindex(idx), pywr_R.reindex(idx)
    S1, S2 = indie_S.reindex(idx), pywr_S.reindex(idx)
    R3 = def_R.reindex(idx) if def_R is not None else None
    S3 = def_S.reindex(idx) if def_S is not None else None
    Robs = obs_R.reindex(idx) if obs_R is not None else None
    Sobs = obs_S.reindex(idx) if obs_S is not None else None

    storage_ylabel = "Storage (MG)"
    if storage_as_pct_capacity and capacity_mg is not None and float(capacity_mg) > 0:
        cap = float(capacity_mg)
        S1 = S1 * (100.0 / cap)
        S2 = S2 * (100.0 / cap)
        if S3 is not None:
            S3 = S3 * (100.0 / cap)
        if Sobs is not None:
            Sobs = Sobs * (100.0 / cap)
        storage_ylabel = "Storage (% of capacity)"

    nor_lo_a = nor_hi_a = None
    if nor_lo_pct is not None and nor_hi_pct is not None:
        nor_lo_a = pd.to_numeric(nor_lo_pct, errors="coerce").reindex(idx)
        nor_hi_a = pd.to_numeric(nor_hi_pct, errors="coerce").reindex(idx)
        if storage_as_pct_capacity:
            nor_lo_a = nor_lo_a.clip(lower=0.0, upper=100.0)
            nor_hi_a = nor_hi_a.clip(lower=0.0, upper=100.0)
        # Guard against occasional inverted bounds from bad inputs.
        bad = (nor_hi_a < nor_lo_a) & nor_hi_a.notna() & nor_lo_a.notna()
        if bad.any():
            lo_fix = nor_lo_a.copy()
            nor_lo_a.loc[bad] = nor_hi_a.loc[bad]
            nor_hi_a.loc[bad] = lo_fix.loc[bad]

    # log-aware prep (mask <=0, compute limits in the right space)
    if yscale_release == "log":
        ylims_release = ylims_release or _positive_log_limits([R1, R2, R3, Robs])
        R1p, R2p = _mask_nonpositive(R1), _mask_nonpositive(R2)
        R3p, Robsp = _mask_nonpositive(R3), _mask_nonpositive(Robs)
    else:
        R1p, R2p, R3p, Robsp = R1, R2, R3, Robs
        if ylims_release is None:
            ylims_release = _robust_limits([R1, R2, R3, Robs])

    if yscale_storage == "log":
        ylims_storage = ylims_storage or _positive_log_limits([S1, S2, S3, Sobs])
        S1p, S2p = _mask_nonpositive(S1), _mask_nonpositive(S2)
        S3p, Sobsp = _mask_nonpositive(S3), _mask_nonpositive(Sobs)
    else:
        S1p, S2p, S3p, Sobsp = S1, S2, S3, Sobs
        if ylims_storage is None:
            lim_src = [S1, S2, S3, Sobs]
            if nor_lo_a is not None and nor_hi_a is not None:
                lim_src.extend([nor_lo_a, nor_hi_a])
            ylims_storage = _robust_limits(lim_src)

    # optional downsampling
    R1p = _downsample(R1p, downsample_step); R2p = _downsample(R2p, downsample_step)
    R3p = _downsample(R3p, downsample_step); Robsp = _downsample(Robsp, downsample_step)
    S1p = _downsample(S1p, downsample_step); S2p = _downsample(S2p, downsample_step)
    S3p = _downsample(S3p, downsample_step); Sobsp = _downsample(Sobsp, downsample_step)
    if nor_lo_a is not None:
        nor_lo_a = _downsample(nor_lo_a, downsample_step)
    if nor_hi_a is not None:
        nor_hi_a = _downsample(nor_hi_a, downsample_step)

    # figure: time series left, optional NSE + param text right
    fig = plt.figure(figsize=(15.5, 8.2))
    if add_release_fdc_panel:
        fig.set_size_inches(15.5, 11.0)
        if show_release_nse_panel and obs_R is not None:
            gs = GridSpec(3, 2, figure=fig, width_ratios=[4.0, 1.05], wspace=0.22, hspace=0.16)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
            ax2 = fig.add_subplot(gs[2, 0])
            ax_nse = fig.add_subplot(gs[:, 1])
            ax_nse.axis("off")
        else:
            gs = GridSpec(3, 1, figure=fig, hspace=0.16)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
            ax2 = fig.add_subplot(gs[2, 0])
            ax_nse = None
    else:
        if show_release_nse_panel and obs_R is not None:
            gs = GridSpec(2, 2, figure=fig, width_ratios=[4.0, 1.05], wspace=0.22, hspace=0.12)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
            ax_nse = fig.add_subplot(gs[:, 1])
            ax_nse.axis("off")
        else:
            gs = GridSpec(2, 1, figure=fig, hspace=0.12)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
            ax_nse = None

    axes = [ax0, ax1]

    # ----- Storage (top) -----
    storage_values = pd.concat([x for x in [S1p, S2p, S3p, Sobsp] if x is not None]).values
    _apply_scale_then_limits(ax0, storage_values, yscale_storage, ylims_storage, linthresh_release)

    if nor_lo_a is not None and nor_hi_a is not None:
        ax0.fill_between(
            nor_lo_a.index,
            nor_lo_a.values,
            nor_hi_a.values,
            color="0.75",
            alpha=0.32,
            zorder=1,
            linewidth=0,
            label="STARFIT NOR",
        )

    if Sobsp is not None:
        ax0.plot(
            Sobsp.index, Sobsp.values, label="Observed", lw=lw_obs,
            color=SERIES_COLORS["observed"], ls=SERIES_LINESTYLES["observed"], zorder=10
        )
    ax0.plot(
        S1p.index, S1p.values, label="Independent", lw=lw_sim, alpha=0.95,
        color=SERIES_COLORS["independent"], ls=SERIES_LINESTYLES["independent"], zorder=6
    )
    ax0.plot(
        S2p.index, S2p.values, label=parametric_label, lw=lw_sim, alpha=0.95,
        color=SERIES_COLORS["parametric"], ls=SERIES_LINESTYLES["parametric"], zorder=5
    )
    if S3p is not None:
        ax0.plot(
            S3p.index, S3p.values, label="Pywr-DRB Default", lw=lw_sim, alpha=0.95,
            color=SERIES_COLORS["default"], ls=SERIES_LINESTYLES["default"], zorder=4
        )
    ax0.set_ylabel(storage_ylabel)
    ax0.grid(True, alpha=0.3)
    ax0.margins(y=0.04)

    # ----- Release (bottom) -----
    release_values = pd.concat([x for x in [R1p, R2p, R3p, Robsp] if x is not None]).values
    _apply_scale_then_limits(ax1, release_values, yscale_release, ylims_release, linthresh_release)

    if Robsp is not None:
        ax1.plot(
            Robsp.index, Robsp.values, label="Observed", lw=lw_obs,
            color=SERIES_COLORS["observed"], ls=SERIES_LINESTYLES["observed"], zorder=10
        )
    ax1.plot(
        R1p.index, R1p.values, label="Independent (release+spill)", lw=lw_sim, alpha=0.95,
        color=SERIES_COLORS["independent"], ls=SERIES_LINESTYLES["independent"], zorder=6
    )
    ax1.plot(
        R2p.index, R2p.values, label=parametric_label, lw=lw_sim, alpha=0.95,
        color=SERIES_COLORS["parametric"], ls=SERIES_LINESTYLES["parametric"], zorder=5
    )
    if R3p is not None:
        ax1.plot(
            R3p.index, R3p.values, label="Pywr-DRB Default", lw=lw_sim, alpha=0.95,
            color=SERIES_COLORS["default"], ls=SERIES_LINESTYLES["default"], zorder=4
        )
    ax1.set_ylabel("Release (MGD)")
    ax1.set_xlabel("Date")
    ax1.grid(True, alpha=0.3)
    ax1.margins(y=0.04)

    # Date ticks
    locator = mdates.AutoDateLocator(minticks=5, maxticks=max_date_ticks)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    if add_release_fdc_panel:
        fdc_obs = _release_fdc(Robsp)
        fdc_indie = _release_fdc(R1p)
        fdc_pywr = _release_fdc(R2p)
        fdc_def = _release_fdc(R3p)

        if len(fdc_obs[0]):
            ax2.plot(*fdc_obs, lw=lw_obs, color=SERIES_COLORS["observed"], ls=SERIES_LINESTYLES["observed"], label="Observed", zorder=10)
        if len(fdc_indie[0]):
            ax2.plot(*fdc_indie, lw=lw_sim, alpha=0.95, color=SERIES_COLORS["independent"], ls=SERIES_LINESTYLES["independent"], label="Independent (release+spill)", zorder=6)
        if len(fdc_pywr[0]):
            ax2.plot(*fdc_pywr, lw=lw_sim, alpha=0.95, color=SERIES_COLORS["parametric"], ls=SERIES_LINESTYLES["parametric"], label=parametric_label, zorder=5)
        if len(fdc_def[0]):
            ax2.plot(*fdc_def, lw=lw_sim, alpha=0.95, color=SERIES_COLORS["default"], ls=SERIES_LINESTYLES["default"], label="Pywr-DRB Default", zorder=4)

        ax2.set_title("Release FDC")
        ax2.set_xlabel("Exceedance (%)")
        ax2.set_ylabel("Release (MGD)")
        ax2.set_yscale("log")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.margins(y=0.04)
        # keep only a single figure-level legend (avoid duplicated legends)

    # Title + legend outside
    header = f"{reservoir} — {policy}"
    if pick_label:
        header += f" — {pick_label}"
    if date_label:
        header += f"\n{date_label}"
    fig.suptitle(header, fontsize=13, fontweight="bold", y=0.98)

    handles_all, labels_all = [], []
    legend_axes = [ax0, ax1] + ([ax2] if add_release_fdc_panel else [])
    for ax_ in legend_axes:
        h, l = ax_.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in labels_all:
                handles_all.append(hh)
                labels_all.append(ll)
    if handles_all:
        if ax_nse is not None:
            ax_nse.legend(
                handles_all,
                labels_all,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                borderaxespad=0.0,
                framealpha=0.9,
                fontsize=9,
                title="Series",
            )
        else:
            fig.legend(
                handles_all,
                labels_all,
                loc="center left",
                bbox_to_anchor=(0.86, 0.5),
                framealpha=0.9,
                fontsize=9,
                title="Series",
            )

    if ax_nse is not None and obs_R is not None:
        lines = [
            ("Independent", R1),
            (parametric_label, R2),
        ]
        if R3 is not None:
            lines.append(("Pywr-DRB Default", R3))
        parts = ["Release NSE vs obs", "-" * 22]
        for name, ser in lines:
            if ser is None:
                continue
            nse = _release_nse_vs_obs(obs_R, ser)
            parts.append(f"{name}: {nse:.4f}" if np.isfinite(nse) else f"{name}: n/a")
        ax_nse.text(
            0.02,
            0.72,
            "\n".join(parts),
            transform=ax_nse.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.95),
        )

    if param_text:
        if ax_nse is not None:
            ax_nse.text(
                0.02,
                0.02,
                param_text,
                transform=ax_nse.transAxes,
                fontsize=7,
                fontfamily="monospace",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.88),
            )
        else:
            fig.text(
                0.86,
                0.02,
                param_text,
                fontsize=7,
                family="monospace",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.88),
            )

    # Avoid tight_layout warnings with mixed/extra axes by using explicit subplot spacing.
    fig.subplots_adjust(
        left=0.08,
        bottom=0.08,
        right=(0.84 if ax_nse is not None else 0.95),
        top=0.94,
        hspace=0.22,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_two_column_indie_pywr_dynamics(
    reservoir: str,
    policy: str,
    indie_R: pd.Series,
    indie_S: pd.Series,
    pywr_R: pd.Series,
    pywr_S: pd.Series,
    obs_R: Optional[pd.Series] = None,
    obs_S: Optional[pd.Series] = None,
    date_label: Optional[str] = None,
    save_path: Optional[str] = None,
    downsample_step: Optional[int] = 2,
    line_width: float = 1.35,
):
    """
    Two-column layout: Independent (left) vs Pywr-DRB parametric (right); 2×1 storage/release per column.
    Observed series overlaid on both columns when provided.
    """
    idx = indie_R.index
    for s in (pywr_R, obs_R, indie_S, pywr_S, obs_S):
        if s is not None:
            idx = idx.intersection(s.index)
    idx = idx.sort_values()
    if len(idx) == 0:
        print(f"[Plot] {reservoir}/{policy}: no overlap for two-column dynamics; skipping.")
        return

    R_i = _downsample(indie_R.reindex(idx), downsample_step)
    S_i = _downsample(indie_S.reindex(idx), downsample_step)
    R_p = _downsample(pywr_R.reindex(idx), downsample_step)
    S_p = _downsample(pywr_S.reindex(idx), downsample_step)
    Robs = _downsample(obs_R.reindex(idx), downsample_step) if obs_R is not None else None
    Sobs = _downsample(obs_S.reindex(idx), downsample_step) if obs_S is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 7.2), sharex="col")
    ax_is, ax_ir = axes[0, 0], axes[1, 0]
    ax_ps, ax_pr = axes[0, 1], axes[1, 1]

    lw_obs = max(line_width * 1.1, 1.0)
    lw = line_width

    def _plot_col(ax_s, ax_r, S_sim, R_sim, col_title: str, sim_color: str):
        if Sobs is not None:
            ax_s.plot(
                Sobs.index, Sobs.values, label="Observed", lw=lw_obs,
                color=SERIES_COLORS["observed"], ls=SERIES_LINESTYLES["observed"], zorder=10
            )
        ax_s.plot(
            S_sim.index, S_sim.values, label="Simulated", lw=lw, alpha=0.95,
            color=sim_color,
            ls="-", zorder=5
        )
        ax_s.set_ylabel("Storage (MG)")
        ax_s.set_title(f"{col_title} — storage")
        ax_s.grid(True, alpha=0.3)

        if Robs is not None:
            ax_r.plot(
                Robs.index, Robs.values, label="Observed", lw=lw_obs,
                color=SERIES_COLORS["observed"], ls=SERIES_LINESTYLES["observed"], zorder=10
            )
        ax_r.plot(
            R_sim.index, R_sim.values, label="Simulated", lw=lw, alpha=0.95,
            color=sim_color,
            ls="-", zorder=5
        )
        ax_r.set_ylabel("Release (MGD)")
        ax_r.set_title(f"{col_title} — release")
        ax_r.grid(True, alpha=0.3)

    _plot_col(ax_is, ax_ir, S_i, R_i, "Independent reservoir model", SERIES_COLORS["independent"])
    _plot_col(ax_ps, ax_pr, S_p, R_p, "Pywr-DRB parametric", SERIES_COLORS["parametric"])

    for ax in (ax_is, ax_ps):
        ax.legend(loc="upper right", fontsize=8)
    for ax in (ax_ir, ax_pr):
        ax.legend(loc="upper right", fontsize=8)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    supt = f"{reservoir} — {policy}"
    if date_label:
        supt += f" — {date_label}"
    fig.suptitle(supt, fontsize=12, weight="semibold")
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.22)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
