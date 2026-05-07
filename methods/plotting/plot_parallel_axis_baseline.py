# methods/plotting/plot_parallel_axis_baseline.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def get_color(value, color_by_continuous, color_palette_continuous,
              color_by_categorical, color_dict_categorical):
    if color_by_continuous is not None:
        return colormaps.get_cmap(color_palette_continuous)(value)
    elif color_by_categorical is not None:
        return color_dict_categorical[value]
    return "k"

def get_zorder(norm_value, zorder_num_classes, zorder_direction):
    xgrid = np.arange(0, 1.001, 1 / zorder_num_classes)
    if zorder_direction == "ascending":
        return 4 + np.sum(norm_value > xgrid)
    elif zorder_direction == "descending":
        return 4 + np.sum(norm_value < xgrid)
    return 4

def reorganize_objs(objs, columns_axes, ideal_direction, minmaxs):
    if minmaxs is None:
        minmaxs = ["max"] * len(columns_axes)

    objs_reorg = objs[columns_axes].copy()
    for c in objs_reorg.columns:
        objs_reorg[c] = pd.to_numeric(objs_reorg[c], errors="coerce")

    if ideal_direction == "bottom":
        tops = objs_reorg.min(axis=0)
        bottoms = objs_reorg.max(axis=0)
        for i, minmax in enumerate(minmaxs):
            col = objs_reorg.iloc[:, i].astype(float)
            mn, mx = col.min(), col.max()
            span = mx - mn
            if not np.isfinite(span) or span == 0:
                objs_reorg.iloc[:, i] = 0.5
                continue
            if minmax == "max":
                # larger is better, but ideal is 'bottom' so invert
                objs_reorg.iloc[:, i] = (mx - col) / (mx - mn)
            else:
                # smaller is better: swap annotation ends so top shows the *min* value
                bottoms[i], tops[i] = tops[i], bottoms[i]
                objs_reorg.iloc[:, i] = (col - mn) / (mx - mn)
    else:  # ideal_direction == "top"
        tops = objs_reorg.max(axis=0)
        bottoms = objs_reorg.min(axis=0)
        for i, minmax in enumerate(minmaxs):
            col = objs_reorg.iloc[:, i].astype(float)
            mn, mx = col.min(), col.max()
            span = mx - mn
            if not np.isfinite(span) or span == 0:
                objs_reorg.iloc[:, i] = 0.5
                continue
            if minmax == "max":
                objs_reorg.iloc[:, i] = (col - mn) / (mx - mn)
            else:
                # smaller is better: put the *min* at the top annotation
                bottoms[i], tops[i] = tops[i], bottoms[i]
                objs_reorg.iloc[:, i] = (mx - col) / (mx - mn)

    objs_reorg = objs_reorg.where(np.isfinite(objs_reorg), 0.5)
    return objs_reorg, tops, bottoms


def custom_parallel_coordinates(
    objs, columns_axes=None, axis_labels=None,
    ideal_direction="top", minmaxs=None,
    color_by_continuous=None, color_palette_continuous=None,
    color_by_categorical=None, color_palette_categorical=None,
    colorbar_ticks_continuous=None, color_dict_categorical=None,
    zorder_by=None, zorder_num_classes=10, zorder_direction="ascending",
    alpha_base=0.8, brushing_dict=None, alpha_brush=0.05,
    lw_base=1.5, fontsize=14,
    figsize=(11, 6), fname=None,
    bottom_pad=0.18, legend_pad=0.06, legend_ncol=4
):
    assert ideal_direction in ["top", "bottom"]
    assert zorder_direction in ["ascending", "descending"]
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ["max", "min"]
    assert (color_by_continuous is None) or (color_by_categorical is None)

    columns_axes = columns_axes if (columns_axes is not None) else objs.columns
    axis_labels = axis_labels if (axis_labels is not None) else columns_axes

    # --- quick visibility sanity check for Baseline row ---
    if color_by_categorical is not None and "Baseline" in set(objs[color_by_categorical].astype(str)):
        bl_filter = objs[color_by_categorical].astype(str) == "Baseline"
        missing_axes = [c for c in columns_axes if not np.isfinite(pd.to_numeric(objs.loc[bl_filter, c], errors="coerce")).all()]
        if missing_axes:
            print(f"[WARN] Baseline has missing values on: {', '.join(missing_axes)} (will render at mid-axis for those).")

    fig, ax = plt.subplots(1, 1, figsize=figsize, gridspec_kw={"hspace": 0.1, "wspace": 0.1})
    fig.subplots_adjust(bottom=bottom_pad)

    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)

    satisfice = None
    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == "<":
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] < threshold)
            elif operator == "<=":
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] <= threshold)
            elif operator == ">":
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] > threshold)
            elif operator == ">=":
                satisfice = np.logical_and(satisfice, objs.iloc[:, col_idx] >= threshold)

            threshold_norm = (threshold - bottoms[col_idx]) / (tops[col_idx] - bottoms[col_idx])
            if ideal_direction == "top" and minmaxs[col_idx] == "max":
                rect = Rectangle([col_idx - 0.05, 0 if operator in [">", ">="] else threshold_norm],
                                 0.1, threshold_norm if operator in [">", ">="] else 1 - threshold_norm)
            elif ideal_direction == "top" and minmaxs[col_idx] == "min":
                rect = Rectangle([col_idx - 0.05, 0 if operator in ["<", "<="] else threshold_norm],
                                 0.1, threshold_norm if operator in ["<", "<="] else 1 - threshold_norm)
            elif ideal_direction == "bottom" and minmaxs[col_idx] == "max":
                rect = Rectangle([col_idx - 0.05, 0 if operator in ["<", "<="] else threshold_norm],
                                 0.1, threshold_norm if operator in ["<", "<="] else 1 - threshold_norm)
            else:
                rect = Rectangle([col_idx - 0.05, 0 if operator in [">", ">="] else threshold_norm],
                                 0.1, threshold_norm if operator in [">", ">="] else 1 - threshold_norm)
            ax.add_collection(PatchCollection([rect], facecolor="grey", alpha=0.5, zorder=3))

    for i in range(objs_reorg.shape[0]):
        if color_by_continuous is not None:
            color = get_color(
                objs_reorg[columns_axes[color_by_continuous]].iloc[i],
                color_by_continuous, color_palette_continuous,
                color_by_categorical, color_dict_categorical
            )
        elif color_by_categorical is not None:
            color = get_color(
                objs[color_by_categorical].iloc[i],
                color_by_continuous, color_palette_continuous,
                color_by_categorical, color_dict_categorical
            )
        else:
            color = "k"

        if zorder_by is None:
            zorder = 4
        else:
            zorder = get_zorder(objs_reorg[columns_axes[zorder_by]].iloc[i], zorder_num_classes, zorder_direction)

        if satisfice is not None:
            if satisfice.iloc[i]:
                alpha = alpha_base; lw = lw_base
            else:
                alpha = alpha_brush; lw = 1.0; zorder = 2
        else:
            alpha = alpha_base; lw = lw_base

        lab = ""
        if color_by_categorical is not None:
            lab = str(objs[color_by_categorical].iloc[i])
        is_hl_col = (color_by_categorical is not None and str(color_by_categorical).lower().startswith("highlight"))

        is_baseline = (
            lab == "Baseline" or
            ("policy"        in objs.columns and str(objs["policy"].iloc[i])        == "Baseline") or
            ("highlight"     in objs.columns and str(objs["highlight"].iloc[i])     == "Baseline") or
            ("highlight_adv" in objs.columns and str(objs["highlight_adv"].iloc[i]) == "Baseline")
        )

        if is_baseline:
            alpha = alpha_base
            lw = max(lw_base, 3.0)
            zorder = 100
        elif is_hl_col and lab != "Other":
            alpha = alpha_base; lw = max(lw_base, 2.5); zorder = 10
        elif is_hl_col:  # "Other"
            alpha = alpha_brush; lw = 1.0; zorder = 2

        for j in range(objs_reorg.shape[1] - 1):
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j + 1]]
            x = [j, j + 1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)

    for j in range(len(columns_axes)):
        ax.annotate(str(round(tops[j], 1)), [j, 1.02], ha="center", va="bottom", zorder=5, fontsize=fontsize)
        ax.annotate(str(round(bottoms[j], 1)), [j, -0.02], ha="center", va="top", zorder=5, fontsize=fontsize)
        ax.plot([j, j], [0, 1], c="k", zorder=1)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(False)

    if ideal_direction == "top":
        ax.arrow(-0.15, 0.1, 0, 0.7, head_width=0.08, head_length=0.05, color="k", lw=1.5)
    else:
        ax.arrow(-0.15, 0.9, 0, -0.7, head_width=0.08, head_length=0.05, color="k", lw=1.5)
    ax.annotate("Direction of preference", xy=(-0.3, 0.5), ha="center", va="center",
                rotation=90, fontsize=fontsize)

    n_axes = len(columns_axes)
    ax.set_xlim(-0.5, n_axes - 0.5)
    ax.set_ylim(-0.4, 1.1)
    for i, l in enumerate(axis_labels):
        ax.annotate(l, xy=(i, -0.12), ha="center", va="top", fontsize=fontsize)
    ax.patch.set_alpha(0)

    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        vmin = objs[columns_axes[color_by_continuous]].min()
        vmax = objs[columns_axes[color_by_continuous]].max()
        mappable.set_clim(vmin=vmin, vmax=vmax)
        cb = plt.colorbar(mappable, ax=ax, orientation="horizontal", location="bottom",
                          shrink=0.4, label=axis_labels[color_by_continuous], pad=0.00, alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous, fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)

    elif color_by_categorical is not None and color_dict_categorical is not None:
        present = pd.unique(objs[color_by_categorical].astype(str)).tolist()
        labels_for_legend = [lab for lab in color_dict_categorical.keys() if lab in present]
        is_hl = str(color_by_categorical).lower().startswith("highlight")
        handles = []
        for lab in labels_for_legend:
            col = color_dict_categorical[lab]
            a = (alpha_brush if (is_hl and lab == "Other") else alpha_base)
            if lab == "Baseline":
                lw_leg = max(lw_base, 3.0)
            else:
                lw_leg = (1.0 if (is_hl and lab == "Other") else max(lw_base, 2.5 if is_hl else lw_base))
            handles.append(Line2D([0], [0], color=col, lw=lw_leg, alpha=a, label=lab))
        if handles:
            ncols = min(legend_ncol if "legend_ncol" in locals() else 4, len(handles))
            leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, legend_pad),
                             ncol=ncols, frameon=False, fontsize=fontsize)
            fig.canvas.draw()
            bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            needed = bb.height + legend_pad + 0.01
            if fig.subplotpars.bottom < needed:
                fig.subplots_adjust(bottom=needed)

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
    return
