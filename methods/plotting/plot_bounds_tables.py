# plot_bounds_tables.py
# One-reservoir visual: for each available policy, show objective and decision ranges.

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# ==== Project imports (your repo layout) ====
from methods.config import (
    OBJ_LABELS, OBJ_FILTER_BOUNDS,
    reservoir_options, policy_type_options,
    FIG_DIR,
    n_rbfs, n_rbf_inputs, n_segments, n_pwl_inputs
)
from methods.borg_paths import resolve_borg_moea_csv_path, resolve_figure_root
from methods.load.results import load_results


# ---------- Pretty helpers ----------
POL_COLORS = {"STARFIT": "#3778c2", "RBF": "#f28e2b", "PWL": "#59a14f"}
POLICY_LABELS = {"STARFIT": "STARFIT", "RBF": "RBF", "PWL": "PWL"}


def safe(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))
    return re.sub(r'_+', '_', s).strip('_') or "out"


def get_param_names_for_policy(policy: str):
    policy = str(policy).upper()
    if policy == "STARFIT":
        return [
            "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
            "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
            "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
            "Release_c", "Release_p1", "Release_p2",
        ]
    if policy == "RBF":
        labels = ["storage", "inflow", "doy"][:n_rbf_inputs]
        names = []
        for i in range(1, n_rbfs + 1):
            names.append(f"w{i}")
        for i in range(1, n_rbfs + 1):
            for v in labels:
                names.append(f"c{i}_{v}")
        for i in range(1, n_rbfs + 1):
            for v in labels:
                names.append(f"r{i}_{v}")
        return names
    if policy == "PWL":
        names = []
        block_labels = ["storage", "inflow", "day"][:n_pwl_inputs]
        for lab in block_labels:
            for k in range(1, n_segments):
                names.append(f"{lab}_x{k}")
            for k in range(1, n_segments + 1):
                names.append(f"{lab}_theta{k}")
        return names
    raise ValueError(f"Unknown policy '{policy}'")


def summarize_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        out.append({
            "Metric": c,
            "Min": np.nanmin(v),
            "P25": np.nanquantile(v, 0.25),
            "Median": np.nanmedian(v),
            "P75": np.nanquantile(v, 0.75),
            "Max": np.nanmax(v),
            "NaN": int(pd.isna(v).sum())
        })
    return pd.DataFrame(out)


def _normalize_row(row):
    mn, mx = row["Min"], row["Max"]
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        row[["Min", "P25", "Median", "P75", "Max"]] = 0.5
    else:
        for k in ["Min", "P25", "Median", "P75", "Max"]:
            row[k] = (row[k] - mn) / (mx - mn)
    return row


def draw_rangebars(ax, stats: pd.DataFrame, color: str, title: str, x_label: str, normalize=False):
    """Horizontal min–max bar, IQR band, and median dot for each metric."""
    if stats.empty:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no data)", fontsize=11)
        return

    D = stats.copy()
    if normalize:
        D = D.apply(_normalize_row, axis=1)
    # Positional rows only: iterrows() labels are not 0..n-1 after sort/head, but y[i] must be.
    D = D.reset_index(drop=True)

    y = np.arange(len(D))[::-1]  # top to bottom
    ax.set_yticks(y)
    ax.set_yticklabels(D["Metric"], fontsize=9)
    if normalize:
        ax.set_xlim(0, 1)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_title(title, fontsize=12, pad=6, weight="semibold")

    for i, row in D.iterrows():
        # min–max
        ax.plot([row["Min"], row["Max"]], [y[i], y[i]],
                lw=2.5, color=color, alpha=0.9, solid_capstyle="round")
        # IQR band
        ax.add_patch(Rectangle((row["P25"], y[i] - 0.18), row["P75"] - row["P25"], 0.36,
                               facecolor=color, alpha=0.25, edgecolor="none"))
        # median
        ax.scatter([row["Median"]], [y[i]], s=32, color=color, zorder=3)

    ax.grid(axis="x", alpha=0.15)
    ax.tick_params(axis='y', which='both', pad=2)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


def load_policy_frames(reservoir: str, policy: str):
    """Load filtered results; return objective DF, variable DF (with renamed parameters)."""
    obj_cols = list(OBJ_LABELS.values())
    fname = resolve_borg_moea_csv_path(policy, reservoir)
    try:
        obj_df, var_df = load_results(fname, obj_labels=OBJ_LABELS, filter=True, obj_bounds=OBJ_FILTER_BOUNDS)
    except Exception as e:
        print(f"[WARN] load failed for {reservoir}/{policy}: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # rename var_df columns to policy-specific param names (if counts match)
    param_names = get_param_names_for_policy(policy)
    if not var_df.empty and len(param_names) == var_df.shape[1]:
        var_df = var_df.copy()
        var_df.columns = param_names
    return obj_df, var_df


def add_ranges_legend(fig, policy_colors: dict, loc="lower center"):
    """Two-part legend: (a) range semantics, (b) policy colors."""
    sem_handles = [
        Line2D([0], [0], lw=6, color="0.25", alpha=0.9, label="min–max (bar)"),
        Patch(facecolor="0.6", alpha=0.25, label="25–75% (band)"),
        Line2D([0], [0], marker="o", ms=7, color="0.25", lw=0, label="median (dot)")
    ]
    pol_handles = [Line2D([0], [0], lw=6, color=c, label=lab)
                   for lab, c in policy_colors.items()]

    leg1 = fig.legend(
        handles=sem_handles,  # explicit handles fixes MPL ambiguity
        ncol=3, loc=loc, bbox_to_anchor=(0.5, 0.02),
        frameon=False, borderaxespad=0.0, columnspacing=1.2, handlelength=2.8
    )
    leg2 = fig.legend(
        handles=pol_handles,
        ncol=min(5, len(pol_handles)), loc=loc, bbox_to_anchor=(0.5, 0.085),
        frameon=False, borderaxespad=0.0, columnspacing=1.2, handlelength=2.8
    )
    fig.add_artist(leg1)


def tighten(fig, suptitle_y=0.985, hspace=0.42, wspace=0.28, bottom=0.16):
    """Tight layout with space left for legends."""
    fig.subplots_adjust(hspace=hspace, wspace=wspace, bottom=bottom)
    try:
        fig.tight_layout(rect=[0, bottom, 1, suptitle_y - 0.005])
    except Exception:
        pass


def make_reservoir_visual(reservoir: str, outdir: str | Path | None = None):
    reservoir = str(reservoir)
    all_policies = list(policy_type_options)  # typically ["STARFIT","RBF","PWL"]
    obj_cols = list(OBJ_LABELS.values())

    # collect stats per policy; keep only policies with any data
    per_pol = {}
    for pol in all_policies:
        obj_df, var_df = load_policy_frames(reservoir, pol)
        obj_stats = summarize_cols(obj_df, obj_cols) if not obj_df.empty else pd.DataFrame()
        dec_stats = summarize_cols(var_df, list(var_df.columns)) if not var_df.empty else pd.DataFrame()
        if not obj_stats.empty or not dec_stats.empty:
            per_pol[pol] = (obj_stats, dec_stats)

    if not per_pol:
        print(f"[INFO] No data available for reservoir '{reservoir}'. Nothing to plot.")
        return

    policies = list(per_pol.keys())
    ncol = len(policies)

    fig = plt.figure(figsize=(4.0 * ncol + 1.5, 8.8))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 2.0], hspace=0.35, wspace=0.35)

    fig.suptitle(f"{reservoir} – Objective & Decision Ranges by Policy",
                 y=0.985, fontsize=20, weight="bold")

    add_ranges_legend(fig, {p: POL_COLORS[p] for p in policies})
    tighten(fig, suptitle_y=0.985, hspace=0.42, wspace=0.28, bottom=0.16)

    # panels
    for j, pol in enumerate(policies):
        color = POL_COLORS.get(pol, "black")
        obj_stats, dec_stats = per_pol[pol]

        ax1 = fig.add_subplot(gs[0, j])
        draw_rangebars(
            ax1, obj_stats, color=color,
            title=f"{POLICY_LABELS.get(pol, pol)} – Objectives",
            x_label="value", normalize=False
        )

        ax2 = fig.add_subplot(gs[1, j])
        if not dec_stats.empty:
            # pick most varying parameters to keep readable (top 18)
            dec_stats["_var"] = (dec_stats["Max"] - dec_stats["Min"]).abs()
            dec_stats = dec_stats.sort_values("_var", ascending=False).drop(columns="_var")
            dec_top = dec_stats.head(18)
        else:
            dec_top = dec_stats

        draw_rangebars(
            ax2, dec_top, color=color,
            title=f"{POLICY_LABELS.get(pol, pol)} – Decisions (normalized per-param)",
            x_label="0 … 1 (min–max per param)", normalize=True
        )

    fig.text(0.01, 0.01,
             "Bars = min–max, band = 25–75%, dot = median. Decision ranges normalized per parameter.",
             fontsize=9, alpha=0.7)

    if outdir is None:
        outdir = Path(resolve_figure_root(FIG_DIR)) / "fig3_parameter_ranges"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"ranges_{safe(reservoir)}.png"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"[SAVED] {out}")


def main():
    parser = argparse.ArgumentParser(description="Make per-policy objective & decision range visuals.")
    parser.add_argument("--reservoir", default="fewalter",
                        help=f"Reservoir name (default: fewalter). Options: {', '.join(reservoir_options)}")
    parser.add_argument("--outdir", default=None, help="Output directory (defaults to FIG_DIR/policy_range_viz).")
    args = parser.parse_args()
    make_reservoir_visual(args.reservoir, args.outdir)


if __name__ == "__main__":
    main()
