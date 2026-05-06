# methods/plotting/plot_parallel_axis.py
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates

from methods.config import OUTPUT_DIR, FIG_DIR
### function to get color based on continuous color map or categorical map
def get_color(value, color_by_continuous, color_palette_continuous, 
              color_by_categorical, color_dict_categorical):
    if color_by_continuous is not None:
        color = colormaps.get_cmap(color_palette_continuous)(value)
    elif color_by_categorical is not None:
        # NaN / missing labels → treat as "Other"; unknown labels → fallback (not in palette)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            key = "Other"
        else:
            key = str(value).strip()
        if key in ("nan", "NaN", "None"):
            key = "Other"
        default = color_dict_categorical.get("Other", "lightgray") if color_dict_categorical else "lightgray"
        color = color_dict_categorical.get(key, default) if color_dict_categorical else default
    return color

### function to get zorder value for ordering lines on plot. 
### This works by binning a given axis' values and mapping to discrete classes.
def get_zorder(norm_value, zorder_num_classes, zorder_direction):
    xgrid = np.arange(0, 1.001, 1/zorder_num_classes)
    if zorder_direction == 'ascending':
        return 4 + np.sum(norm_value > xgrid)
    elif zorder_direction == 'descending':
        return 4 + np.sum(norm_value < xgrid)
    
        
### function to normalize data based on direction of preference and whether each objective is minimized or maximized
###   -> output dataframe will have values ranging from 0 (which maps to bottom of figure) to 1 (which maps to top)
def reorganize_objs(objs, columns_axes, ideal_direction, minmaxs):
    ### if min/max directions not given for each axis, assume all should be maximized
    if minmaxs is None:
        minmaxs = ['max']*len(columns_axes)
        
    ### get subset of dataframe columns that will be shown as parallel axes
    objs_reorg = objs[columns_axes]
    
    ### reorganize & normalize data to go from 0 (bottom of figure) to 1 (top of figure), 
    if ideal_direction == 'bottom':
        # 1d numpy: avoid Series integer [i] (deprecated positional access in future pandas).
        tops = objs_reorg.min(axis=0).to_numpy(dtype=float, copy=True)
        bottoms = objs_reorg.max(axis=0).to_numpy(dtype=float, copy=True)
        for i, minmax in enumerate(minmaxs):
            col = objs_reorg.iloc[:, i].astype(float)
            mn, mx = col.min(), col.max()
            span = mx - mn
            if not np.isfinite(span) or span == 0:
                # center a flat axis so lines render instead of NaNs
                objs_reorg.iloc[:, i] = 0.5
                continue
            if minmax == 'max':
                objs_reorg.iloc[:, i] = (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i]) / \
                                        (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i].min(axis=0))
            else:
                bottoms[i], tops[i] = tops[i], bottoms[i]
                objs_reorg.iloc[:, i] = (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i]) / \
                                        (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i].min(axis=0))
    elif ideal_direction == 'top':
        tops = objs_reorg.max(axis=0).to_numpy(dtype=float, copy=True)
        bottoms = objs_reorg.min(axis=0).to_numpy(dtype=float, copy=True)
        for i, minmax in enumerate(minmaxs):
            col = objs_reorg.iloc[:, i].astype(float)
            mn, mx = col.min(), col.max()
            span = mx - mn
            if not np.isfinite(span) or span == 0:
                # center a flat axis so lines render instead of NaNs
                objs_reorg.iloc[:, i] = 0.5
                continue
            if minmax == 'max':
                objs_reorg.iloc[:, i] = (objs_reorg.iloc[:, i] - objs_reorg.iloc[:, i].min(axis=0)) / \
                                        (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i].min(axis=0))
            else:
                bottoms[i], tops[i] = tops[i], bottoms[i]
                objs_reorg.iloc[:, i] = (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i]) / \
                                        (objs_reorg.iloc[:, i].max(axis=0) - objs_reorg.iloc[:, i].min(axis=0))
    # Shuffle so that solutions are mixed
    # objs_reorg = objs_reorg.sample(frac=1).reset_index(drop=True)

    return objs_reorg, tops, bottoms


def _wrap_legend_label(text: str, width: int) -> str:
    """Break long pick names into multiple lines for readable legends."""
    if width <= 0 or not str(text).strip():
        return text
    s = str(text).strip()
    if len(s) <= width:
        return s
    return textwrap.fill(s, width=width, break_long_words=True, break_on_hyphens=True)


### customizable parallel coordinates plot
def custom_parallel_coordinates(objs, columns_axes=None, axis_labels=None, 
                                ideal_direction='top', minmaxs=None, 
                                color_by_continuous=None, color_palette_continuous=None, 
                                color_by_categorical=None, color_palette_categorical=None,
                                colorbar_ticks_continuous=None, color_dict_categorical=None,
                                zorder_by=None, zorder_num_classes=10, zorder_direction='ascending', 
                                alpha_base=0.8, brushing_dict=None, alpha_brush=0.05, 
                                lw_base=1.5, fontsize=14, 
                                figsize=(12, 7.5), fname=None,
                                bottom_pad=0.22,
                                legend_pad=0.06,
                                legend_ncol=2,
                                legend_label_width=40,
                                legend_fontsize=None,
                                subplot_left=0.10,
                                subplot_right=0.97,
                                subplot_top=0.92,
                                footnote=None,
                                ):

    ### verify that all inputs take supported values
    assert ideal_direction in ['top','bottom']
    assert zorder_direction in ['ascending', 'descending']
    if minmaxs is not None:
        for minmax in minmaxs:
            assert minmax in ['max','min']
    assert color_by_continuous is None or color_by_categorical is None
    columns_axes = columns_axes if (columns_axes is not None) else objs.columns
    axis_labels = axis_labels if (axis_labels is not None) else columns_axes
    
    ### create figure (taller default figsize + margins so axes fill width without huge side gaps)
    fig,ax = plt.subplots(1,1,figsize=figsize, gridspec_kw={'hspace':0.1, 'wspace':0.1})
    leg_fs = legend_fontsize if legend_fontsize is not None else max(9, fontsize + 1)
    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        top=subplot_top,
        bottom=bottom_pad,
    )

    ### reorganize & normalize objective data
    objs_reorg, tops, bottoms = reorganize_objs(objs, columns_axes, ideal_direction, minmaxs)

    ### apply any brushing criteria
    if brushing_dict is not None:
        satisfice = np.zeros(objs.shape[0]) == 0.
        ### iteratively apply all brushing criteria to get satisficing set of solutions
        for col_idx, (threshold, operator) in brushing_dict.items():
            if operator == '<':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] < threshold)
            elif operator == '<=':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] <= threshold)
            elif operator == '>':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] > threshold)
            elif operator == '>=':
                satisfice = np.logical_and(satisfice, objs.iloc[:,col_idx] >= threshold)

            ### add rectangle patch to plot to represent brushing
            threshold_norm = (threshold - bottoms[col_idx]) / (tops[col_idx] - bottoms[col_idx])
            if ideal_direction == 'top' and minmaxs[col_idx] == 'max':
                if operator in ['<', '<=']:
                    rect = Rectangle([col_idx-0.05, threshold_norm], 0.1, 1-threshold_norm)
                elif operator in ['>', '>=']:
                    rect = Rectangle([col_idx-0.05, 0], 0.1, threshold_norm)
            elif ideal_direction == 'top' and minmaxs[col_idx] == 'min':
                if operator in ['<', '<=']:
                    rect = Rectangle([col_idx-0.05, 0], 0.1, threshold_norm)
                elif operator in ['>', '>=']:
                    rect = Rectangle([col_idx-0.05, threshold_norm], 0.1, 1-threshold_norm)
            if ideal_direction == 'bottom' and minmaxs[col_idx] == 'max':
                if operator in ['<', '<=']:
                    rect = Rectangle([col_idx-0.05, 0], 0.1, threshold_norm)
                elif operator in ['>', '>=']:
                    rect = Rectangle([col_idx-0.05, threshold_norm], 0.1, 1-threshold_norm)
            elif ideal_direction == 'bottom' and minmaxs[col_idx] == 'min':
                if operator in ['<', '<=']:
                    rect = Rectangle([col_idx-0.05, threshold_norm], 0.1, 1-threshold_norm)
                elif operator in ['>', '>=']:
                    rect = Rectangle([col_idx-0.05, 0], 0.1, threshold_norm)
                    
            pc = PatchCollection([rect], facecolor='grey', alpha=0.5, zorder=3)
            ax.add_collection(pc)
    
    ### loop over all solutions/rows & plot on parallel axis plot
    for i in range(objs_reorg.shape[0]):
        if color_by_continuous is not None:
            color = get_color(objs_reorg[columns_axes[color_by_continuous]].iloc[i], 
                              color_by_continuous, color_palette_continuous, 
                              color_by_categorical, color_dict_categorical)
        elif color_by_categorical is not None:
            color = get_color(objs[color_by_categorical].iloc[i], 
                              color_by_continuous, color_palette_continuous, 
                              color_by_categorical, color_dict_categorical)
        
 
        ### order lines according to ascending or descending values of one of the objectives?
        if zorder_by is None:
            zorder = 4
        else:
            zorder = get_zorder(objs_reorg[columns_axes[zorder_by]].iloc[i], 
                                zorder_num_classes, zorder_direction)
            
        ### apply any brushing?
        if brushing_dict is not None:
            if satisfice.iloc[i]:
                alpha = alpha_base
                lw = lw_base
            else:
                alpha = alpha_brush
                lw = 1
                zorder = 2
        else:
            alpha = alpha_base
            lw = lw_base
            
        ### loop over objective/column pairs & plot lines between parallel axes
        for j in range(objs_reorg.shape[1]-1):

            # --- style boost for highlight* columns (works for 'highlight' or 'highlight_adv') ---
            is_hl = (color_by_categorical is not None
                    and str(color_by_categorical).lower().startswith("highlight"))
            if is_hl:
                lab = str(objs[color_by_categorical].iloc[i])
                if lab != "Other":
                    alpha = alpha_base
                    lw = 2.5
                    zorder = 10
                else:
                    alpha = alpha_brush
                    lw = 1.0
                    zorder = 2
            
            y = [objs_reorg.iloc[i, j], objs_reorg.iloc[i, j+1]]
            x = [j, j+1]
            ax.plot(x, y, c=color, alpha=alpha, zorder=zorder, lw=lw)
            
            
    ### add top/bottom ranges
    for j in range(len(columns_axes)):
        ax.annotate(str(round(tops[j], 1)), [j, 1.02], ha='center', va='bottom', 
                    zorder=5, fontsize=fontsize)
 
        ax.annotate(str(round(bottoms[j], 1)), [j, -0.02], ha='center', va='top', 
                    zorder=5, fontsize=fontsize)    

        ax.plot([j,j], [0,1], c='k', zorder=1)
    
    ### other aesthetics
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ['top','bottom','left','right']:
        ax.spines[spine].set_visible(False)

    if ideal_direction == 'top':
        ax.arrow(-0.15,0.1,0,0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    elif ideal_direction == 'bottom':
        ax.arrow(-0.15,0.9,0,-0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    ax.annotate('Direction of preference', xy=(-0.3,0.5), ha='center', va='center',
                rotation=90, fontsize=fontsize)

    n_axes = len(columns_axes)
    ax.set_xlim(-0.5, n_axes - 0.5)
    ax.set_ylim(-0.4,1.1)
    
    for i,l in enumerate(axis_labels):
        ax.annotate(l, xy=(i,-0.12), ha='center', va='top', fontsize=fontsize)
    ax.patch.set_alpha(0)
    

    ### colorbar for continuous legend
    if color_by_continuous is not None:
        mappable = cm.ScalarMappable(cmap=color_palette_continuous)
        mappable.set_clim(vmin=objs[columns_axes[color_by_continuous]].min(), 
                          vmax=objs[columns_axes[color_by_continuous]].max())
        cb = plt.colorbar(mappable, ax=ax, 
                          orientation='horizontal', 
                          location = 'bottom', 
                          shrink=0.4, 
                          label=axis_labels[color_by_continuous], pad=0.00, 
                          alpha=alpha_base)
        if colorbar_ticks_continuous is not None:
            _ = cb.ax.set_xticks(colorbar_ticks_continuous, colorbar_ticks_continuous, 
                                 fontsize=fontsize)
        _ = cb.ax.set_xlabel(cb.ax.get_xlabel(), fontsize=fontsize)  
    ### categorical legend
    # --- categorical legend (figure-level) ---
    elif color_by_categorical is not None and color_dict_categorical is not None:
        # Keep only labels actually present in the plotted data
        present = pd.unique(objs[color_by_categorical].astype(str)).tolist()
        labels_for_legend = [lab for lab in color_dict_categorical.keys() if lab in present]

        # Build handles with the same “highlight vs Other” styling you used
        is_hl = str(color_by_categorical).lower().startswith("highlight")
        handles = []
        for lab in labels_for_legend:
            col = color_dict_categorical[lab]
            a = (alpha_brush if (is_hl and lab == "Other") else alpha_base)
            lw_leg = (1.0 if (is_hl and lab == "Other") else max(lw_base, 2.5 if is_hl else lw_base))
            wrapped = _wrap_legend_label(lab, legend_label_width)
            handles.append(Line2D([0], [0], color=col, lw=lw_leg, alpha=a, label=wrapped))

        if handles:
            # Fewer columns + wrapped labels so entries stay readable (not one ultra-wide row).
            ncols = max(1, min(legend_ncol, len(handles)))
            leg = fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, legend_pad),
                ncol=ncols,
                frameon=False,
                fontsize=leg_fs,
                handlelength=2.4,
                handletextpad=0.6,
                columnspacing=1.4,
                borderaxespad=0.8,
            )

            # Ensure the reserved bottom margin is big enough for the legend
            fig.canvas.draw()  # need renderer for accurate bbox
            bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            needed = bb.height + legend_pad + 0.02
            if footnote:
                needed += 0.06
            if fig.subplotpars.bottom < needed:
                fig.subplots_adjust(bottom=needed)
        
    if footnote:
        foot_w = max(48, min(100, int(figsize[0] * 8)))
        foot_text = textwrap.fill(footnote, width=foot_w, break_long_words=False)
        foot_fs = max(8, fontsize - 2)
        fig.text(
            0.5,
            0.012,
            foot_text,
            ha="center",
            va="bottom",
            fontsize=foot_fs,
            linespacing=1.25,
        )

     ### save figure
    if fname is not None:
         plt.savefig(fname, bbox_inches='tight', dpi=300)
         plt.close(fig)
    
    #plt.show()
    return