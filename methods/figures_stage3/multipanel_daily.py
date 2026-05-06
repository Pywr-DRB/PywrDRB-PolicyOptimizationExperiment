"""Multipanel figure: daily reservoir releases, Trenton time/FDC, reliability (Stage 3)."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from methods.plotting.theme import SERIES_LINESTYLES, SERIES_LINEWIDTHS

from .constants import (
    POLICY_ORDER,
    RESERVOIR_KEYS,
    STAGE3_OBSERVED_COLOR,
    STAGE3_POLICY_COLORS,
    STAGE3_TRENTON_TARGET_COLOR,
)
from .data_loading import MultipanelDailyBundle

# Denser day-of-year ticks (row 1 + Trenton time); avoid overcrowding
_DOY_XTICKS = np.unique(np.concatenate((np.arange(1, 366, 30), [365])))


def _first_band_x(bundle: MultipanelDailyBundle, reservoir_key: str):
    for pol in POLICY_ORDER:
        if pol in bundle.reservoir_release.get(reservoir_key, {}):
            return bundle.reservoir_release[reservoir_key][pol].x
    return bundle.days


def _first_trenton_x(bundle: MultipanelDailyBundle):
    for pol in POLICY_ORDER:
        if pol in bundle.trenton:
            return bundle.trenton[pol].x
    return bundle.days


def _trim_obs_to_x(obs: np.ndarray, x: np.ndarray):
    """Observed DOY curves may be 365 or 366 points; align to envelope ``x`` grid."""
    o = np.asarray(obs, dtype=float)
    if o.size == 0 or x is None or len(x) == 0:
        return None
    n = min(len(x), len(o))
    if n < 1:
        return None
    return x[:n], o[:n]


def _band_alphas() -> Tuple[float, float]:
    """Low alpha limits additive color mixing when all three policies overlap (see Beltzville)."""
    return 0.045, 0.11


def _fig_bottom_legend_handles(colors: dict) -> List:
    """Match dynamics-style hierarchy: policy medians, band meaning, observed, target."""
    outer_a, inner_a = _band_alphas()
    handles: List = []
    for p in POLICY_ORDER:
        handles.append(Line2D([0], [0], color=colors[p], lw=2.0, label=f"{p} median"))
    handles.append(
        Patch(
            facecolor="0.4",
            edgecolor="0.35",
            linewidth=0.8,
            alpha=max(outer_a * 3, 0.12),
            label="10–90% across Pareto runs (per policy)",
        )
    )
    handles.append(
        Patch(
            facecolor="0.4",
            edgecolor="0.35",
            linewidth=0.8,
            alpha=min(inner_a * 3.5, 0.35),
            label="25–75% across Pareto runs (per policy)",
        )
    )
    obs_ls = SERIES_LINESTYLES.get("observed", "-")
    obs_lw = SERIES_LINEWIDTHS.get("observed", 1.8) + 0.4
    handles.append(
        Line2D(
            [0],
            [0],
            color=STAGE3_OBSERVED_COLOR,
            ls=obs_ls,
            lw=obs_lw,
            label="Observed (smoothed DOY trend)",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=STAGE3_TRENTON_TARGET_COLOR,
            ls="--",
            lw=2.0,
            label="Trenton target",
        )
    )
    return handles


def plot_multipanel_daily_uncertainty(
    data: MultipanelDailyBundle,
    *,
    title: str = "Multi-panel policy uncertainty: reservoir releases to Trenton reliability",
    footnote: str = (
        "Rows: reservoir-level release envelopes plus observed releases; Trenton time/FDC propagation; "
        "reliability distributions across Pareto-optimal solutions. "
        "Translucent bands overlap across policies; where all three overlap the mix reads grey—"
        "lower opacity here limits the effect."
    ),
    save_path: Optional[str] = None,
    dpi: int = 220,
) -> str:
    """
    Row 1: four reservoir release envelopes (policy median + quantile bands) + observed.
    Row 2: Trenton flow time series + FDC with observed FDC.
    Row 3: reliability violin + CDF.
    """
    colors = STAGE3_POLICY_COLORS
    outer_a, inner_a = _band_alphas()
    exceed = data.fdc_exceedance_pct
    reservoirs = [k for k in RESERVOIR_KEYS if k in data.reservoir_release]

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 0.95], hspace=0.38, wspace=0.22)

    for j, r in enumerate(reservoirs):
        ax = fig.add_subplot(gs[0, j])
        for zi, p in enumerate(POLICY_ORDER):
            if p not in data.reservoir_release[r]:
                continue
            b = data.reservoir_release[r][p]
            zf = 1 + zi
            ax.fill_between(
                b.x, b.p10, b.p90, color=colors[p], alpha=outer_a, zorder=zf, linewidths=0
            )
            ax.fill_between(
                b.x,
                b.p25,
                b.p75,
                color=colors[p],
                alpha=inner_a,
                zorder=3 + zi,
                linewidths=0,
            )
            ax.plot(b.x, b.p50, color=colors[p], lw=2.0, zorder=8, label=None)

        obs = data.reservoir_release_observed.get(r) if data.reservoir_release_observed else None
        xy = _trim_obs_to_x(obs, _first_band_x(data, r)) if obs is not None else None
        if xy is not None:
            xo, yo = xy
            ax.plot(
                xo,
                yo,
                color=STAGE3_OBSERVED_COLOR,
                ls=SERIES_LINESTYLES.get("observed", "-"),
                lw=SERIES_LINEWIDTHS.get("observed", 1.4) + 0.5,
                zorder=10,
                label=None,
            )

        ax.set_title(data.reservoir_display_names.get(r, r), fontsize=13, weight="bold")
        ax.set_xlim(1, 365)
        ax.set_xticks(_DOY_XTICKS)
        ax.set_xlabel("Day of Year")
        if j == 0:
            ax.set_ylabel("Release (MGD)")
        ax.grid(True, alpha=0.25)

    ax5 = fig.add_subplot(gs[1, 0:2])
    for zi, p in enumerate(POLICY_ORDER):
        b = data.trenton[p]
        ax5.fill_between(
            b.x, b.p10, b.p90, color=colors[p], alpha=outer_a, zorder=1 + zi, linewidths=0
        )
        ax5.fill_between(
            b.x, b.p25, b.p75, color=colors[p], alpha=inner_a, zorder=3 + zi, linewidths=0
        )
        ax5.plot(b.x, b.p50, color=colors[p], lw=2.2, zorder=8, label=None)
    ax5.axhline(
        data.trenton_target_mgd, color=STAGE3_TRENTON_TARGET_COLOR, ls="--", lw=2, zorder=7, label=None
    )
    tx = _first_trenton_x(data)
    t_xy = (
        _trim_obs_to_x(data.trenton_doy_observed, tx)
        if data.trenton_doy_observed is not None
        else None
    )
    if t_xy is not None:
        xo, yo = t_xy
        ax5.plot(
            xo,
            yo,
            color=STAGE3_OBSERVED_COLOR,
            ls=SERIES_LINESTYLES.get("observed", "-"),
            lw=SERIES_LINEWIDTHS.get("observed", 1.4) + 0.5,
            zorder=10,
            label=None,
        )
    ax5.set_title("Trenton Flow Envelope (Time)", fontsize=14, weight="bold")
    ax5.set_xlim(1, 365)
    ax5.set_xticks(_DOY_XTICKS)
    ax5.set_xlabel("Day of Year")
    ax5.set_ylabel("Flow (MGD)")
    ax5.grid(True, alpha=0.25)

    ax6 = fig.add_subplot(gs[1, 2:4])
    for zi, p in enumerate(POLICY_ORDER):
        b = data.trenton_fdc[p]
        ax6.fill_between(
            b.x, b.p10, b.p90, color=colors[p], alpha=outer_a, zorder=1 + zi, linewidths=0
        )
        ax6.fill_between(
            b.x, b.p25, b.p75, color=colors[p], alpha=inner_a, zorder=3 + zi, linewidths=0
        )
        ax6.plot(b.x, b.p50, color=colors[p], lw=2.2, zorder=8, label=None)
    ax6.plot(exceed, data.trenton_fdc_observed, color=STAGE3_OBSERVED_COLOR, lw=2.5, zorder=10, label=None)
    ax6.axhline(data.trenton_target_mgd, color=STAGE3_TRENTON_TARGET_COLOR, ls="--", lw=1.8, alpha=0.9)
    ax6.set_yscale("log")
    ax6.set_title("Trenton Flow Duration Curve Envelope", fontsize=14, weight="bold")
    ax6.set_xlabel("Exceedance Probability (%)")
    ax6.set_ylabel("Flow (MGD)")
    ax6.grid(True, alpha=0.25, which="both")

    ax7 = fig.add_subplot(gs[2, 0:2])
    rel_lists = [data.reliability_by_policy[p] for p in POLICY_ORDER]
    v = ax7.violinplot(
        rel_lists,
        positions=[1, 2, 3],
        widths=0.8,
        showmedians=True,
        showmeans=False,
        showextrema=False,
    )
    for body, p in zip(v["bodies"], POLICY_ORDER):
        body.set_facecolor(colors[p])
        body.set_edgecolor(colors[p])
        body.set_alpha(0.22)
    v["cmedians"].set_color("black")
    v["cmedians"].set_linewidth(2)
    rng = np.random.RandomState(0)
    for i, p in enumerate(POLICY_ORDER, start=1):
        x = rng.normal(i, 0.05, len(data.reliability_by_policy[p]))
        ax7.scatter(x, data.reliability_by_policy[p], s=10, color=colors[p], alpha=0.22)
    ax7.set_xticks([1, 2, 3])
    ax7.set_xticklabels(POLICY_ORDER)
    ax7.set_ylim(0.55, 1.01)
    ax7.set_ylabel("Reliability\n(% time Trenton ≥ target)")
    ax7.set_title("Reliability Distribution (Violin + Points)", fontsize=14, weight="bold")
    ax7.grid(True, axis="y", alpha=0.25)

    ax8 = fig.add_subplot(gs[2, 2:4])
    for p in POLICY_ORDER:
        vals = np.sort(data.reliability_by_policy[p])
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax8.plot(vals, cdf, color=colors[p], lw=2.4, label=p)
    ax8.set_xlim(0.55, 1.0)
    ax8.set_ylim(0, 1.02)
    ax8.set_xlabel("Reliability")
    ax8.set_ylabel("Cumulative Probability")
    ax8.set_title("Reliability CDF", fontsize=14, weight="bold")
    ax8.grid(True, alpha=0.25)
    ax8.legend(
        frameon=True,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.92,
        borderaxespad=0,
    )

    fig.subplots_adjust(bottom=0.18, top=0.92, right=0.98)
    leg_handles = _fig_bottom_legend_handles(colors)
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=True,
        fontsize=9,
        framealpha=0.95,
        handlelength=2.6,
        columnspacing=1.2,
    )

    fig.suptitle(title, fontsize=19, weight="bold", y=0.97)
    fig.text(0.5, 0.06, footnote, ha="center", fontsize=10)

    out = save_path or "fig12_multipanel_daily_uncertainty.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
