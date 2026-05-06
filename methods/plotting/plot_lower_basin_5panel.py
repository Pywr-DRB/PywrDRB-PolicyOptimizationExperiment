#!/usr/bin/env python3
"""Five-panel figure: inflow, storage, release, lower-basin MRF at Trenton, Trenton flow."""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from methods.plotting.theme import SERIES_COLORS, SERIES_LINESTYLES, SERIES_LINEWIDTHS


def _nice_dates(ax, max_ticks: int = 10):
    loc = AutoDateLocator(minticks=4, maxticks=max_ticks)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))


def _mrf_reservoir_columns(df: pd.DataFrame, prefix: str = "mrf_trenton_") -> list[str]:
    return [c for c in df.columns if str(c).startswith(prefix)]


def plot_lower_basin_fivepanel(
    *,
    reservoir: str,
    obs_inflow: Optional[pd.Series],
    obs_storage_mg: Optional[pd.Series],
    obs_release: Optional[pd.Series],
    sim_storage_mg: Optional[pd.Series] = None,
    sim_release: Optional[pd.Series] = None,
    default_storage_mg: Optional[pd.Series] = None,
    default_release: Optional[pd.Series] = None,
    mrf_df: Optional[pd.DataFrame] = None,
    default_mrf_df: Optional[pd.DataFrame] = None,
    trenton_obs: Optional[pd.Series] = None,
    trenton_sim: Optional[pd.Series] = None,
    trenton_default: Optional[pd.Series] = None,
    start: str,
    end: str,
    policy: str = "",
    pick_label: str = "",
    param_text: str = "",
    mrf_style: str = "rolling",
    rolling_days: int = 7,
    mrf_line_width: float = 2.0,
    trenton_line_width: float = 2.2,
    save_path: Optional[str] = None,
):
    """
    mrf_style: 'rolling' — stacked area of rolling-mean MRF contributions;
               'bar_monthly' — monthly mean stacked bars for MRF columns.
    """
    slicer = slice(start, end)
    q_in = (
        pd.to_numeric(obs_inflow.loc[slicer], errors="coerce").dropna().astype(float)
        if obs_inflow is not None
        else pd.Series(dtype=float)
    )
    q_s = (
        pd.to_numeric(obs_storage_mg.loc[slicer], errors="coerce").dropna().astype(float)
        if obs_storage_mg is not None
        else pd.Series(dtype=float)
    )
    q_r = (
        pd.to_numeric(obs_release.loc[slicer], errors="coerce").dropna().astype(float)
        if obs_release is not None
        else pd.Series(dtype=float)
    )

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True, constrained_layout=False)

    if len(q_in):
        axes[0].plot(q_in.index, q_in.values, color="tab:blue", lw=1.2)
    else:
        axes[0].text(0.5, 0.5, "Observed inflow unavailable in this window", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_ylabel("Inflow (MGD)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Observed inflow (pub)")

    if len(q_s):
        axes[1].plot(
            q_s.index, q_s.values,
            color=SERIES_COLORS["observed"],
            lw=SERIES_LINEWIDTHS["observed"],
            ls=SERIES_LINESTYLES["observed"],
            label="Observed",
        )
    if sim_storage_mg is not None and len(sim_storage_mg):
        s_s = sim_storage_mg.loc[slicer].astype(float)
        sim_storage_label = "Simulated (Parametric)" if (default_storage_mg is not None and len(default_storage_mg)) else "Simulated"
        axes[1].plot(
            s_s.index, s_s.values,
            color=SERIES_COLORS["parametric"],
            lw=SERIES_LINEWIDTHS["parametric"],
            ls=SERIES_LINESTYLES["parametric"],
            label=sim_storage_label,
        )
    if default_storage_mg is not None and len(default_storage_mg):
        d_s = default_storage_mg.loc[slicer].astype(float)
        axes[1].plot(
            d_s.index, d_s.values,
            color=SERIES_COLORS["default"],
            lw=SERIES_LINEWIDTHS["default"],
            ls=SERIES_LINESTYLES["default"],
            label="Simulated (Pywr Default)",
        )
    axes[1].set_ylabel("Storage (MG)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Storage (observed vs simulated)")
    if axes[1].has_data():
        axes[1].legend(loc="upper left", fontsize=8)

    if len(q_r):
        axes[2].plot(
            q_r.index, q_r.values,
            color=SERIES_COLORS["observed"],
            lw=SERIES_LINEWIDTHS["observed"],
            ls=SERIES_LINESTYLES["observed"],
            label="Observed",
        )
    if sim_release is not None and len(sim_release):
        s_r = sim_release.loc[slicer].astype(float)
        sim_release_label = "Simulated (Parametric)" if (default_release is not None and len(default_release)) else "Simulated"
        axes[2].plot(
            s_r.index, s_r.values,
            color=SERIES_COLORS["parametric"],
            lw=SERIES_LINEWIDTHS["parametric"],
            ls=SERIES_LINESTYLES["parametric"],
            label=sim_release_label,
        )
    if default_release is not None and len(default_release):
        d_r = default_release.loc[slicer].astype(float)
        axes[2].plot(
            d_r.index, d_r.values,
            color=SERIES_COLORS["default"],
            lw=SERIES_LINEWIDTHS["default"],
            ls=SERIES_LINESTYLES["default"],
            label="Simulated (Pywr Default)",
        )
    axes[2].set_ylabel("Release (MGD)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Release (observed vs simulated)")
    if axes[2].has_data():
        axes[2].legend(loc="upper left", fontsize=8)

    # --- MRF contributions ---
    ax_m = axes[3]
    if mrf_df is not None and len(mrf_df) > 0:
        sub = mrf_df.loc[slicer].astype(float)
        cols = _mrf_reservoir_columns(sub)
        if cols:
            labels = [c.replace("mrf_trenton_", "") for c in cols]
            if mrf_style == "bar_monthly":
                m = sub[cols].resample("ME").mean()
                x = np.arange(len(m.index))
                ax_m.bar(x, m.iloc[:, 0].values, label=labels[0], width=0.8)
                bottom = m.iloc[:, 0].values.copy()
                for j in range(1, len(cols)):
                    ax_m.bar(x, m.iloc[:, j].values, bottom=bottom, label=labels[j], width=0.8)
                    bottom = bottom + m.iloc[:, j].values
                ax_m.set_xticks(x[:: max(1, len(x) // 12)])
                ax_m.set_xticklabels([d.strftime("%Y-%m") for d in m.index[:: max(1, len(x) // 12)]], rotation=45, ha="right")
                ax_m.set_ylabel("MRF contrib. (MGD)")
            else:
                roll = sub[cols].rolling(f"{rolling_days}D", min_periods=1).mean()
                ax_m.stackplot(
                    roll.index,
                    *[roll[c] for c in cols],
                    labels=labels,
                    alpha=0.85,
                    linewidth=mrf_line_width,
                    edgecolor="white",
                )
                ax_m.set_ylabel(f"MRF contrib. ({rolling_days}d mean, MGD)")
                ax_m.legend(loc="upper left", fontsize=8)
            ax_m.set_title(f"Lower-basin MRF contributions at Trenton ({mrf_style})")
            if default_mrf_df is not None and len(default_mrf_df) > 0:
                dsub = default_mrf_df.loc[slicer].astype(float)
                dcols = [c for c in cols if c in dsub.columns]
                if dcols:
                    droll = dsub[dcols].rolling(f"{rolling_days}D", min_periods=1).mean().sum(axis=1)
                    ax_m.plot(
                        droll.index,
                        droll.values,
                        color=SERIES_COLORS["default"],
                        lw=2.0,
                        ls="--",
                        label="Pywr Default total MRF",
                    )
                    ax_m.legend(loc="upper left", fontsize=8)
        else:
            ax_m.text(0.5, 0.5, "No mrf_trenton_* columns", ha="center", va="center", transform=ax_m.transAxes)
    else:
        ax_m.text(0.5, 0.5, "MRF data unavailable", ha="center", va="center", transform=ax_m.transAxes)
    ax_m.grid(True, alpha=0.3)

    ax_t = axes[4]
    if trenton_obs is not None and len(trenton_obs):
        o = trenton_obs.loc[slicer].astype(float)
        ax_t.plot(
            o.index, o.values,
            color=SERIES_COLORS["observed"],
            lw=max(trenton_line_width, SERIES_LINEWIDTHS["observed"]),
            ls=SERIES_LINESTYLES["observed"],
            label="Observed delTrenton",
        )
    if trenton_sim is not None and len(trenton_sim):
        s = trenton_sim.loc[slicer].astype(float)
        sim_trenton_label = "Simulated delTrenton (Parametric)" if (trenton_default is not None and len(trenton_default)) else "Simulated delTrenton"
        ax_t.plot(
            s.index, s.values,
            color=SERIES_COLORS["parametric"],
            lw=max(trenton_line_width, SERIES_LINEWIDTHS["parametric"]),
            ls=SERIES_LINESTYLES["parametric"],
            label=sim_trenton_label,
        )
    if trenton_default is not None and len(trenton_default):
        d = trenton_default.loc[slicer].astype(float)
        ax_t.plot(
            d.index, d.values,
            color=SERIES_COLORS["default"],
            lw=max(trenton_line_width, SERIES_LINEWIDTHS["default"]),
            ls=SERIES_LINESTYLES["default"],
            label="Simulated delTrenton (Pywr Default)",
        )
    ax_t.set_ylabel("Flow (MGD)")
    ax_t.set_xlabel("Date")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper left")
    ax_t.set_title("Trenton flow")

    for ax in axes[:3]:
        _nice_dates(ax)
    if mrf_style != "bar_monthly":
        _nice_dates(ax_m)
    _nice_dates(ax_t)

    bits = [reservoir, policy, pick_label, f"{start} → {end}"]
    fig.suptitle(" — ".join(b for b in bits if b), fontsize=14, fontweight="bold", y=1.01)
    if param_text:
        fig.text(
            0.02,
            0.01,
            param_text,
            fontsize=7,
            family="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.88),
            transform=fig.transFigure,
        )

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    if save_path:
        p = Path(save_path)
        if p.parent.parts:
            p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
