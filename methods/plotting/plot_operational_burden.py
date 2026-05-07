"""
Engaging static plots for operational-burden metrics (beyond plain time series).

Uses matplotlib only. Optional Plotly figure in ``plot_tradeoff_interactive`` if plotly is installed.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import Patch


def _combined_stress_score(
    storage_pct: pd.Series,
    nor_low_pct: float,
    nor_high_pct: float,
) -> pd.Series:
    s = storage_pct.astype(float)
    drought = (nor_low_pct - s).clip(lower=0.0)
    flood = (s - nor_high_pct).clip(lower=0.0)
    return (drought + flood).rename("stress_pct_points")


def plot_stress_calendar_heatmap(
    storage_pct: pd.Series,
    nor_low_pct: float,
    nor_high_pct: float,
    *,
    ax: Optional[plt.Axes] = None,
    cmap: str = "magma_r",
    title: str = "Operating stress (distance outside NOR)",
    fontsize: int = 9,
):
    """
    Year × day-of-year heatmap: drought + flood distance from NOR band (%-points).
    """
    score = _combined_stress_score(storage_pct, nor_low_pct, nor_high_pct)
    df = score.to_frame("v")
    df["year"] = df.index.year
    df["doy"] = df.index.dayofyear
    pivot = df.pivot_table(index="year", columns="doy", values="v", aggfunc="mean")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(11, 4.2), dpi=140)

    arr = pivot.values.astype(float)
    vmax = float(np.nanpercentile(arr, 99)) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(
        arr,
        aspect="auto",
        cmap=cmap,
        norm=mcolors.PowerNorm(gamma=0.65, vmin=0, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int), fontsize=fontsize - 1)
    ax.set_xlabel("Day of year", fontsize=fontsize)
    ax.set_ylabel("Year", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Stress (combined % pts)", fontsize=fontsize - 1)

    if created:
        plt.tight_layout()
    return ax, im


def plot_spell_duration_distribution(
    spells: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    color: str = "#2a6f97",
    title: str = "Spell duration distribution",
    fontsize: int = 9,
):
    """Histogram of contiguous spell lengths (e.g. below NOR or binding at Trenton)."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5.5, 3.4), dpi=140)

    if spells is None or spells.empty or "duration_days" not in spells.columns:
        ax.text(0.5, 0.5, "No spells", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    d = spells["duration_days"].astype(int)
    mx = int(d.max())
    if mx <= 45:
        bins = np.arange(0.5, mx + 1.5, 1.0)
    else:
        bins = np.linspace(0.5, mx + 0.5, 26)
    ax.hist(d, bins=bins, color=color, edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.set_xlabel("Duration (days)", fontsize=fontsize)
    ax.set_ylabel("Count", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created:
        plt.tight_layout()
    return ax


# Palette aligned with DRB-Historic-Reconstruction ``flow_contribution.py`` (T. Amestoy).
_FLOW_CONTRIB_PALETTE = [
    "#2166ac",
    "#4393c3",
    "#92c5de",
    "#d1e5f0",
    "#f6e8c3",
    "#dfc27d",
    "#bf812d",
    "#8c510a",
    "#BF92D2",
]


def _color_for_mrf_column(col_name: str, i: int) -> str:
    """Prefer stable colors for common lower-basin names; else cycle palette."""
    key = str(col_name).lower().replace("mrf_trenton_", "")
    special = {
        "beltzvillecombined": "#bf812d",
        "bluemarsh": "#dfc27d",
        "nockamixon": "#f6e8c3",
    }
    if key in special:
        return special[key]
    return _FLOW_CONTRIB_PALETTE[i % len(_FLOW_CONTRIB_PALETTE)]


def plot_lower_basin_mrf_flow_contributions(
    contrib_wide: pd.DataFrame,
    total_flow_sim: pd.Series,
    total_flow_obs: pd.Series | None = None,
    *,
    stack_normalization: str = "lower_basin_total",
    ax: Optional[plt.Axes] = None,
    title: str = "Lower-basin MRF contributions at Trenton",
    fontsize: int = 10,
    rolling_days: int | None = 7,
    contribution_fill_alpha: float = 0.9,
    sim_line_color: str = "#4393c3",
    units: str = "MGD",
    dpi: int = 200,
):
    """
    Twin-axis figure in the style of ``plot_ensemble_node_flow_contributions`` / DRB flow
    contribution plots: **primary** axis = total Trenton flow (sim, optional obs); **secondary**
    = stacked percent from ``mrf_trenton_*`` columns.

    ``stack_normalization``:

    - ``lower_basin_total`` (default): each ``mrf_trenton_i / sum(mrf_trenton_*) × 100``. The stack
      always sums to **100%** and shows **how lower-basin MRF at Trenton is split** across
      reservoirs (internal composition).
    - ``trenton``: each ``mrf_trenton_i / delTrenton × 100``. Shows each term as **% of total
      Trenton flow**; the stack height equals combined lower-basin MRF as a share of Trenton
      (typically well below 100% because other processes contribute).

    Only ``mrf_trenton_*`` columns are plotted; NYC releases and uncontrolled flow are not included.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(1, 1, figsize=(7.0, 3.0), dpi=dpi)

    cols = [c for c in contrib_wide.columns if str(c).startswith("mrf_trenton_")]
    if not cols:
        ax.text(0.5, 0.5, "No mrf_trenton_* columns", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        if created:
            plt.tight_layout()
        return ax

    flow_s = pd.to_numeric(total_flow_sim, errors="coerce").astype(float).sort_index()
    flow_s = flow_s[~flow_s.index.duplicated(keep="first")]

    idx = flow_s.index.intersection(contrib_wide.index)
    idx = idx.sort_values()
    flow_s = flow_s.reindex(idx)
    df = contrib_wide[cols].astype(float).reindex(idx).fillna(0.0)

    if rolling_days and rolling_days > 1:
        df = df.rolling(rolling_days, min_periods=1).mean()
        flow_s = flow_s.rolling(rolling_days, min_periods=1).mean()

    if stack_normalization == "lower_basin_total":
        mrf_sum = df.sum(axis=1)
        denom = mrf_sum.replace(0.0, np.nan)
        pct = df.divide(denom, axis=0) * 100.0
        pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        twin_ylabel = "Share of lower-basin MRF (%)"
    elif stack_normalization == "trenton":
        denom = flow_s.replace(0.0, np.nan)
        pct = df.divide(denom, axis=0) * 100.0
        pct = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        twin_ylabel = "Flow contribution (% of Trenton)"
    else:
        raise ValueError(
            f"stack_normalization must be 'lower_basin_total' or 'trenton', got {stack_normalization!r}"
        )

    x = pct.index
    ax_twin = ax.twinx()
    ax_twin.set_ylim(0.0, 100.0)

    y_lo = np.zeros(len(pct), dtype=float)
    for i, c in enumerate(cols):
        y_hi = y_lo + pct[c].values.astype(float)
        color = _color_for_mrf_column(c, i)
        lab = str(c).replace("mrf_trenton_", "").replace("_", " ")
        ax_twin.fill_between(
            x,
            y_lo,
            y_hi,
            color=color,
            alpha=contribution_fill_alpha,
            lw=0,
            zorder=1,
            label=lab,
        )
        y_lo = y_hi

    ax.plot(
        flow_s.index,
        flow_s.values,
        color=sim_line_color,
        lw=1.2,
        zorder=10,
        label="Simulated delTrenton",
    )

    if total_flow_obs is not None and len(total_flow_obs):
        fo = pd.to_numeric(total_flow_obs, errors="coerce").astype(float)
        fo = fo[~fo.index.duplicated(keep="first")].sort_index()
        fo = fo.reindex(idx).dropna(how="all")
        if rolling_days and rolling_days > 1:
            fo = fo.rolling(rolling_days, min_periods=1).mean()
        if len(fo):
            ax.plot(fo.index, fo.values, color="k", ls="--", lw=1.0, zorder=10, label="Observed delTrenton")

    ax.set_ylabel(f"Total flow ({units})", fontsize=fontsize)
    ax_twin.set_ylabel(twin_ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0.0, max(float(np.nanmax(flow_s.values)) * 1.05, 1e-6))

    ax_twin.set_zorder(1)
    ax.set_zorder(10)
    ax.patch.set_visible(False)

    h_line, lab_line = ax.get_legend_handles_labels()
    h_fill, lab_fill = ax_twin.get_legend_handles_labels()
    ax.legend(
        h_line + h_fill,
        lab_line + lab_fill,
        frameon=False,
        fontsize=fontsize - 1,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(4, max(1, len(h_line) + len(h_fill))),
    )

    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    fig = ax.get_figure()
    if fig is not None:
        fig.autofmt_xdate()

    if created:
        plt.tight_layout()
    return ax


def plot_contribution_vs_depletion_scatter(
    tradeoff_table: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    point_color: str = "#bc4749",
    fontsize: int = 9,
    annotate: bool = True,
):
    """
    Scatter: contribution share vs stress exposure (e.g. fraction time below threshold).

    ``tradeoff_table`` rows indexed by reservoir; columns ``contribution_share``,
    ``fraction_time_stressed`` (from ``contribution_vs_depletion_tradeoff``).
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5.5, 4.5), dpi=140)

    if tradeoff_table.empty:
        ax.text(0.5, 0.5, "Empty table", ha="center", va="center", transform=ax.transAxes)
        return ax

    x = tradeoff_table["fraction_time_stressed"].astype(float)
    y = tradeoff_table["contribution_share"].astype(float)
    ax.scatter(
        x * 100.0,
        y * 100.0,
        s=120,
        c=point_color,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    if annotate:
        for idx, row in tradeoff_table.iterrows():
            ax.annotate(
                str(idx),
                (row["fraction_time_stressed"] * 100.0, row["contribution_share"] * 100.0),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=fontsize - 1,
                alpha=0.9,
            )

    ax.set_xlabel("% time below depletion threshold", fontsize=fontsize)
    ax.set_ylabel("Contribution share (%)", fontsize=fontsize)
    ax.set_title("Trade-off: depletion exposure vs. operational contribution", fontsize=fontsize + 1)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created:
        plt.tight_layout()
    return ax


def plot_annual_stress_bars(
    annual_table: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    fontsize: int = 9,
):
    """Grouped bars: events per year and stress days per year (twin axis)."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8.0, 3.5), dpi=140)

    if annual_table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return ax, ax

    years = annual_table.index.astype(int)
    x = np.arange(len(years))
    w = 0.38
    ax.bar(x - w / 2, annual_table["n_events"], width=w, label="Stress events", color="#457b9d")
    ax.set_ylabel("Event count", color="#457b9d", fontsize=fontsize)
    ax.tick_params(axis="y", labelcolor="#457b9d")

    ax2 = ax.twinx()
    ax2.bar(x + w / 2, annual_table["stress_days"], width=w, label="Stress days", color="#e76f51", alpha=0.85)
    ax2.set_ylabel("Stress days (overlap)", color="#e76f51", fontsize=fontsize)
    ax2.tick_params(axis="y", labelcolor="#e76f51")

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=fontsize - 1)
    ax.set_xlabel("Year", fontsize=fontsize)
    ax.set_title("Annual stress frequency", fontsize=fontsize + 1)

    h1 = [Patch(facecolor="#457b9d", label="Events")]
    h2 = [Patch(facecolor="#e76f51", label="Stress days")]
    ax.legend(handles=h1 + h2, loc="upper left", frameon=False, fontsize=fontsize - 1)

    if created:
        plt.tight_layout()
    return ax, ax2


def plot_storage_ecdf_stress_split(
    storage_pct: pd.Series,
    stress_filter: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    label_stress: str = "Stress days",
    label_other: str = "Other days",
    fontsize: int = 9,
):
    """
    ECDF of storage % on stress days vs non-stress days — highlights breaking-point tail.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5.0, 4.0), dpi=140)

    s = storage_pct.astype(float).reindex(stress_filter.index).dropna()
    m = stress_filter.reindex(s.index).fillna(False).astype(bool)

    def ecdf(a: np.ndarray):
        a = np.sort(a[np.isfinite(a)])
        if len(a) == 0:
            return np.array([]), np.array([])
        y = np.arange(1, len(a) + 1) / len(a)
        return a, y

    for filter, lab, c in [(m, label_stress, "#d62828"), (~m, label_other, "#2a9d8f")]:
        xs, ys = ecdf(s.loc[filter].values)
        if len(xs):
            ax.plot(xs, ys, lw=2.0, label=lab, color=c)

    ax.set_xlabel("Storage (% capacity)", fontsize=fontsize)
    ax.set_ylabel("Empirical CDF", fontsize=fontsize)
    ax.set_title("Storage distribution: stress vs normal conditions", fontsize=fontsize + 1)
    ax.legend(frameon=False, loc="lower right", fontsize=fontsize - 1)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created:
        plt.tight_layout()
    return ax


def plot_trenton_shortfall_lollipop(
    shortfall: pd.Series,
    *,
    freq: str = "ME",
    ax: Optional[plt.Axes] = None,
    fontsize: int = 9,
):
    """
    Aggregated shortfall (e.g. monthly mean) as horizontal lollipop — quick regime view.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8.0, 3.2), dpi=140)

    g = shortfall.astype(float).clip(lower=0).resample(freq).mean()
    y = np.arange(len(g))
    ax.hlines(y, 0, g.values, color="#6c757d", lw=1)
    ax.scatter(g.values, y, color="#e63946", zorder=3, s=36)
    ax.set_yticks(y)
    ax.set_yticklabels([d.strftime("%Y-%m") for d in g.index], fontsize=fontsize - 2)
    ax.set_xlabel(f"Mean shortfall ({freq})", fontsize=fontsize)
    ax.set_title("Trenton target shortfall (aggregated)", fontsize=fontsize + 1)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created:
        plt.tight_layout()
    return ax


def plot_tradeoff_interactive(
    tradeoff_table: pd.DataFrame,
    *,
    title: str = "Contribution vs depletion",
):
    """
    Optional Plotly scatter (hover labels). Returns None if plotly is missing.
    """
    try:
        import plotly.express as px
    except ImportError:
        return None

    df = tradeoff_table.reset_index().rename(columns={"index": "reservoir"})
    fig = px.scatter(
        df,
        x=df["fraction_time_stressed"] * 100.0,
        y=df["contribution_share"] * 100.0,
        text="reservoir",
        title=title,
        labels={
            "x": "% time below threshold",
            "y": "Contribution share (%)",
        },
    )
    fig.update_traces(textposition="top center", marker=dict(size=12))
    return fig


def plot_operational_burden_summary_figure(
    storage_pct: pd.Series,
    nor_low_pct: float,
    nor_high_pct: float,
    stress_threshold_pct: float,
    annual_stress: pd.DataFrame,
    spells_below: pd.DataFrame,
    tradeoff_table: Optional[pd.DataFrame] = None,
    trenton_shortfall: Optional[pd.Series] = None,
    figsize: tuple[float, float] = (12.0, 10.0),
):
    """
    Multi-panel figure tying calendar, spells, annual rates, ECDF, optional trade-off.
    """
    fig = plt.figure(figsize=figsize, dpi=140)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.35, wspace=0.28)

    ax_cal = fig.add_subplot(gs[0, :])
    plot_stress_calendar_heatmap(
        storage_pct, nor_low_pct, nor_high_pct, ax=ax_cal, title="Stress outside NOR (calendar)"
    )

    ax_sp = fig.add_subplot(gs[1, 0])
    plot_spell_duration_distribution(spells_below, ax=ax_sp, title="Drought-side spell lengths")

    ax_an = fig.add_subplot(gs[1, 1])
    plot_annual_stress_bars(annual_stress, ax=ax_an)

    stress_filter = storage_pct.astype(float) < stress_threshold_pct
    ax_ec = fig.add_subplot(gs[2, 0])
    plot_storage_ecdf_stress_split(
        storage_pct,
        stress_filter,
        ax=ax_ec,
        label_stress=f"Below {stress_threshold_pct:g}%",
    )

    ax_tr = fig.add_subplot(gs[2, 1])
    if tradeoff_table is not None and not tradeoff_table.empty:
        plot_contribution_vs_depletion_scatter(tradeoff_table, ax=ax_tr)
    elif trenton_shortfall is not None and len(trenton_shortfall):
        plot_trenton_shortfall_lollipop(trenton_shortfall, ax=ax_tr)
    else:
        ax_tr.axis("off")

    fig.suptitle("Operational burden summary", fontsize=12, fontweight="bold", y=1.02)
    return fig
