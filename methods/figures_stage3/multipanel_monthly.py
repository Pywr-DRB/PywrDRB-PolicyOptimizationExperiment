"""Multipanel figure: monthly release envelopes + inflow–release + Trenton + reliability."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from methods.plotting.plot_pareto_ensemble_uncertainty import envelope_ir_binned

from .axes_helpers import add_flow_regime_shading
from .constants import (
    MONTH_LABELS,
    POLICY_ORDER,
    RESERVOIR_KEYS,
    STAGE3_OBSERVED_COLOR,
    STAGE3_POLICY_COLORS,
    STAGE3_TRENTON_TARGET_COLOR,
)
from .data_loading import MultipanelMonthlyBundle


def plot_multipanel_monthly_uncertainty(
    data: MultipanelMonthlyBundle,
    *,
    title: str = "Monthly seasonal summary + inflow–release structure + Trenton reliability",
    footnote: str = (
        "Top row: monthly release envelopes across solutions; middle row: inflow–release with "
        "observed curve and low/mid/high flow regimes; bottom: Trenton and reliability."
    ),
    save_path: Optional[str] = None,
    dpi: int = 220,
) -> str:
    colors = STAGE3_POLICY_COLORS
    months = data.months
    exceed = data.fdc_exceedance_pct
    reservoirs = [k for k in RESERVOIR_KEYS if k in data.monthly_release]

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.05, 0.95], hspace=0.38, wspace=0.22)

    for j, r in enumerate(reservoirs):
        ax = fig.add_subplot(gs[0, j])
        for p in POLICY_ORDER:
            b = data.monthly_release[r][p]
            ax.fill_between(b.x, b.p10, b.p90, color=colors[p], alpha=0.08, zorder=2)
            ax.fill_between(b.x, b.p25, b.p75, color=colors[p], alpha=0.18, zorder=2)
            ax.plot(b.x, b.p50, color=colors[p], lw=2.2, label=None, zorder=3)
        obs_m = data.monthly_release_observed.get(r) if data.monthly_release_observed else None
        if obs_m is not None and len(obs_m) == len(months):
            ax.plot(
                months,
                obs_m,
                color=STAGE3_OBSERVED_COLOR,
                lw=2.4,
                ls="-",
                zorder=5,
                label=None,
            )
        ax.set_title(data.reservoir_display_names.get(r, r), fontsize=13, weight="bold")
        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_xlabel("Month")
        if j == 0:
            ax.set_ylabel("Release (MGD)")
        ax.grid(True, alpha=0.25)

    for j, r in enumerate(reservoirs):
        ax = fig.add_subplot(gs[1, j])
        if r not in data.inflow_release:
            ax.text(0.5, 0.5, "No inflow–release data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ir = data.inflow_release[r]
        x = ir.inflow_mgd
        if (
            ir.regime_q20 is not None
            and ir.regime_q80 is not None
            and np.isfinite(ir.regime_q20)
            and np.isfinite(ir.regime_q80)
        ):
            q20, q80 = float(ir.regime_q20), float(ir.regime_q80)
        else:
            q20 = float(np.percentile(x, 20))
            q80 = float(np.percentile(x, 80))
        x_pos = np.asarray(x, dtype=float)
        x_pos = x_pos[np.isfinite(x_pos) & (x_pos > 0)]
        raw_xmin = float(np.nanmin(x_pos)) if x_pos.size else float(x.min())
        raw_xmax = float(np.nanmax(x_pos)) if x_pos.size else float(x.max())
        # Widen span so observed Q20/Q80 sit inside limits (regime shading + labels on every panel).
        xmin = float(min(raw_xmin, q20 * 0.92))
        xmax = float(max(raw_xmax, q80 * 1.08))
        xmin = max(xmin, 1e-8)

        band_max = []
        for p in POLICY_ORDER:
            if p in ir.policy_bands:
                band_max.append(float(np.nanmax(ir.policy_bands[p].p90)))
        obs_y_arr = np.asarray(ir.observed_release, dtype=float)
        ymax = float(
            max(
                float(np.nanmax(obs_y_arr)) if obs_y_arr.size else 1.0,
                max(band_max) if band_max else 1.0,
            )
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(max(ymax * 0.02, 0.5), ymax * 1.12)

        add_flow_regime_shading(
            ax,
            q20,
            q80,
            xmin=xmin,
            xmax=xmax,
            label_low="Low flow",
            label_mid="Mid flow",
            label_high="High flow",
            with_labels=True,
        )

        for p in POLICY_ORDER:
            if p not in ir.policy_bands:
                continue
            b = ir.policy_bands[p]
            ax.fill_between(b.x, b.p10, b.p90, color=colors[p], alpha=0.08, zorder=2)
            ax.fill_between(b.x, b.p25, b.p75, color=colors[p], alpha=0.18, zorder=2)
            ax.plot(b.x, b.p50, color=colors[p], lw=2.0, zorder=4)

        # Observed IR: bin by inflow (do not plot raw daily pairs in time order — reads as black scribbles).
        ox = np.asarray(ir.observed_inflow_mgd, dtype=float) if ir.observed_inflow_mgd is not None else None
        oy = np.asarray(ir.observed_release, dtype=float)
        if ox is not None and ox.size == oy.size:
            m = np.isfinite(ox) & np.isfinite(oy) & (ox > 0) & (oy > 0)
            ox, oy = ox[m], oy[m]
        else:
            ox, oy = np.array([]), np.array([])
        if ox.size >= 8:
            nbin = int(min(55, max(18, ox.size // 25)))
            ir_obs = envelope_ir_binned(ox, oy.reshape(-1, 1), n_bins=nbin)
            xc = np.asarray(ir_obs["x"], dtype=float)
            yc = np.asarray(ir_obs["median"], dtype=float)
            ok = np.isfinite(xc) & np.isfinite(yc) & (xc > 0) & (yc > 0)
            if np.any(ok):
                ax.plot(
                    xc[ok],
                    yc[ok],
                    color=STAGE3_OBSERVED_COLOR,
                    lw=2.6,
                    zorder=7,
                    solid_capstyle="round",
                )

        ax.set_xlabel("Inflow (MGD)")
        if j == 0:
            ax.set_ylabel("Release (MGD)")
        ax.grid(True, alpha=0.25, which="both")

    ax9 = fig.add_subplot(gs[2, 0])
    for p in POLICY_ORDER:
        b = data.trenton[p]
        ax9.fill_between(b.x, b.p10, b.p90, color=colors[p], alpha=0.08)
        ax9.fill_between(b.x, b.p25, b.p75, color=colors[p], alpha=0.18)
        ax9.plot(b.x, b.p50, color=colors[p], lw=2.2, label=None)
    ax9.axhline(data.trenton_target_mgd, color=STAGE3_TRENTON_TARGET_COLOR, ls="--", lw=2, label=None)
    ax9.set_title("Trenton Flow Envelope (Time)", fontsize=13, weight="bold")
    ax9.set_xlabel("Day of Year")
    ax9.set_ylabel("Flow (MGD)")
    ax9.grid(True, alpha=0.25)

    ax10 = fig.add_subplot(gs[2, 1])
    for p in POLICY_ORDER:
        b = data.trenton_fdc[p]
        ax10.fill_between(b.x, b.p10, b.p90, color=colors[p], alpha=0.08)
        ax10.fill_between(b.x, b.p25, b.p75, color=colors[p], alpha=0.18)
        ax10.plot(b.x, b.p50, color=colors[p], lw=2.2, label=None)
    ax10.plot(exceed, data.trenton_fdc_observed, color=STAGE3_OBSERVED_COLOR, lw=2.5, label=None)
    ax10.axhline(data.trenton_target_mgd, color=STAGE3_TRENTON_TARGET_COLOR, ls="--", lw=1.7, alpha=0.9)
    ax10.set_yscale("log")
    ax10.set_title("Trenton Flow Duration Curve", fontsize=13, weight="bold")
    ax10.set_xlabel("Exceedance Probability (%)")
    ax10.set_ylabel("Flow (MGD)")
    ax10.grid(True, alpha=0.25, which="both")

    ax11 = fig.add_subplot(gs[2, 2])
    rel_lists = [data.reliability_by_policy[p] for p in POLICY_ORDER]
    v = ax11.violinplot(
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
    rng = np.random.RandomState(1)
    for i, p in enumerate(POLICY_ORDER, start=1):
        xj = rng.normal(i, 0.05, len(data.reliability_by_policy[p]))
        ax11.scatter(xj, data.reliability_by_policy[p], s=10, color=colors[p], alpha=0.22)
    ax11.set_xticks([1, 2, 3])
    ax11.set_xticklabels(POLICY_ORDER)
    ax11.set_ylim(0.55, 1.01)
    ax11.set_ylabel("Reliability\n(% time Trenton ≥ target)")
    ax11.set_title("Reliability Distribution", fontsize=13, weight="bold")
    ax11.grid(True, axis="y", alpha=0.25)

    ax12 = fig.add_subplot(gs[2, 3])
    for p in POLICY_ORDER:
        vals = np.sort(data.reliability_by_policy[p])
        cdf_vals = np.arange(1, len(vals) + 1) / len(vals)
        ax12.plot(vals, cdf_vals, color=colors[p], lw=2.3, label=None)
    ax12.set_xlim(0.55, 1.0)
    ax12.set_ylim(0, 1.02)
    ax12.set_xlabel("Reliability")
    ax12.set_ylabel("Cumulative Probability")
    ax12.set_title("Reliability CDF", fontsize=13, weight="bold")
    ax12.grid(True, alpha=0.25)

    pol_handles = [plt.Line2D([0], [0], color=colors[p], lw=2.2, label=p) for p in POLICY_ORDER]
    obs_handle = plt.Line2D(
        [0],
        [0],
        color=STAGE3_OBSERVED_COLOR,
        lw=2.4,
        label="Observed (monthly release, binned IR, Trenton FDC)",
    )
    tgt_handle = plt.Line2D(
        [0], [0], color=STAGE3_TRENTON_TARGET_COLOR, lw=2, ls="--", label="Trenton target"
    )
    regime_handles = [
        Patch(facecolor="#c8d4e6", edgecolor="none", alpha=0.35, label="Low inflow"),
        Patch(facecolor="#f5f5f5", edgecolor="none", alpha=0.35, label="Mid inflow"),
        Patch(facecolor="#e6ddd0", edgecolor="none", alpha=0.35, label="High inflow"),
    ]
    fig.legend(
        handles=pol_handles + [obs_handle, tgt_handle] + regime_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=True,
        fontsize=9,
        framealpha=0.95,
    )

    fig.suptitle(title, fontsize=19, weight="bold", y=0.98)
    fig.text(0.5, 0.04, footnote, ha="center", fontsize=11)

    fig.subplots_adjust(bottom=0.12)

    out = save_path or "fig13_multipanel_monthly_uncertainty.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
