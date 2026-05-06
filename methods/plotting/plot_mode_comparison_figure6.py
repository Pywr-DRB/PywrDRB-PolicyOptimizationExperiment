#!/usr/bin/env python3
"""Figure 6: compare parametric modes against observed/default."""

from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from methods.plotting.theme import SERIES_COLORS, SERIES_LINESTYLES, SERIES_LINEWIDTHS


def _overlap_index(series_list):
    present = [s for s in series_list if s is not None]
    if not present:
        return None
    idx = present[0].index
    for s in present[1:]:
        idx = idx.intersection(s.index)
    return idx.sort_values() if len(idx) else None


def plot_figure6_mode_comparison(
    *,
    reservoir: str,
    policy: str,
    pick_label: str,
    start: str,
    end: str,
    obs_release: Optional[pd.Series],
    obs_storage: Optional[pd.Series],
    default_release: Optional[pd.Series],
    default_storage: Optional[pd.Series],
    param_reg_release: Optional[pd.Series],
    param_reg_storage: Optional[pd.Series],
    param_perfect_release: Optional[pd.Series],
    param_perfect_storage: Optional[pd.Series],
    save_path: str,
):
    idx = _overlap_index(
        [
            obs_release,
            obs_storage,
            default_release,
            default_storage,
            param_reg_release,
            param_reg_storage,
            param_perfect_release,
            param_perfect_storage,
        ]
    )
    if idx is None:
        return

    def _sel(s):
        return s.loc[idx] if s is not None else None

    obs_release = _sel(obs_release)
    obs_storage = _sel(obs_storage)
    default_release = _sel(default_release)
    default_storage = _sel(default_storage)
    param_reg_release = _sel(param_reg_release)
    param_reg_storage = _sel(param_reg_storage)
    param_perfect_release = _sel(param_perfect_release)
    param_perfect_storage = _sel(param_perfect_storage)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Storage
    if obs_storage is not None:
        ax0.plot(
            obs_storage.index,
            obs_storage.values,
            color=SERIES_COLORS["observed"],
            ls=SERIES_LINESTYLES["observed"],
            lw=SERIES_LINEWIDTHS["observed"],
            label="Observed",
        )
    if default_storage is not None:
        ax0.plot(
            default_storage.index,
            default_storage.values,
            color=SERIES_COLORS["default"],
            ls=SERIES_LINESTYLES["default"],
            lw=SERIES_LINEWIDTHS["default"],
            label="Pywr Default",
        )
    if param_reg_storage is not None:
        ax0.plot(
            param_reg_storage.index,
            param_reg_storage.values,
            color="#17becf",
            ls=SERIES_LINESTYLES["parametric"],
            lw=SERIES_LINEWIDTHS["parametric"],
            label="Pywr Parametric (regression_disagg)",
        )
    if param_perfect_storage is not None:
        ax0.plot(
            param_perfect_storage.index,
            param_perfect_storage.values,
            color="#ff7f0e",
            ls=SERIES_LINESTYLES["parametric"],
            lw=SERIES_LINEWIDTHS["parametric"],
            label="Pywr Parametric (perfect_foresight)",
        )
    ax0.set_ylabel("Storage (MG)")
    ax0.grid(alpha=0.3)
    ax0.set_title("Figure 6 — Storage comparison")

    # Release
    if obs_release is not None:
        ax1.plot(
            obs_release.index,
            obs_release.values,
            color=SERIES_COLORS["observed"],
            ls=SERIES_LINESTYLES["observed"],
            lw=SERIES_LINEWIDTHS["observed"],
            label="Observed",
        )
    if default_release is not None:
        ax1.plot(
            default_release.index,
            default_release.values,
            color=SERIES_COLORS["default"],
            ls=SERIES_LINESTYLES["default"],
            lw=SERIES_LINEWIDTHS["default"],
            label="Pywr Default",
        )
    if param_reg_release is not None:
        ax1.plot(
            param_reg_release.index,
            param_reg_release.values,
            color="#17becf",
            ls=SERIES_LINESTYLES["parametric"],
            lw=SERIES_LINEWIDTHS["parametric"],
            label="Pywr Parametric (regression_disagg)",
        )
    if param_perfect_release is not None:
        ax1.plot(
            param_perfect_release.index,
            param_perfect_release.values,
            color="#ff7f0e",
            ls=SERIES_LINESTYLES["parametric"],
            lw=SERIES_LINEWIDTHS["parametric"],
            label="Pywr Parametric (perfect_foresight)",
        )
    ax1.set_ylabel("Release (MGD)")
    ax1.set_xlabel("Date")
    ax1.grid(alpha=0.3)
    ax1.set_title("Figure 6 — Release comparison")

    loc = mdates.AutoDateLocator(minticks=5, maxticks=10)
    fmt = mdates.ConciseDateFormatter(loc)
    ax1.xaxis.set_major_locator(loc)
    ax1.xaxis.set_major_formatter(fmt)

    title = f"{reservoir} — {policy} — {pick_label}\n{start} to {end}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
