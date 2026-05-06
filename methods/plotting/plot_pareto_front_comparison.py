# methods/plotting/plot_pareto_front_comparison.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_pareto_front_comparison(
    obj_dfs,
    labels,
    obj_cols,
    title="Pareto Front Comparison",
    x_lims=None,
    y_lims=None,
    ideal=None,
    fname=None,
    baseline_point=None,
    baseline_label="Pywr default operation",
    series_colors=None,
    annotate_id_col=None,
    annotate_fontsize=7,
):
    """
    Scatter Pareto fronts per policy. Optional ``baseline_point`` is (x, y) in ``obj_cols`` space
    for the current default Pywr operation (e.g. from baseline_objectives CSV).

    ``series_colors``: optional sequence of matplotlib colors, one per ``obj_dfs`` row, so each
    policy keeps the same hue regardless of plotting order or how many policies are present
    (e.g. :data:`methods.plotting.theme.POLICY_COMPARISON_COLORS`).

    ``annotate_id_col``: optional column name to annotate each plotted point (e.g. policy row id).
    """
    assert len(obj_dfs) == len(labels), "Number of dataframes must match number of labels"
    if series_colors is not None and len(series_colors) != len(obj_dfs):
        raise ValueError("series_colors must be the same length as obj_dfs (or None)")
    for i, df in enumerate(obj_dfs):
        for col in obj_cols:
            assert col in df.columns, f"Column {col} not found in {i}th dataframe"

    fig, ax = plt.subplots()
    for i, df in enumerate(obj_dfs):
        sc_kw = {}
        if series_colors is not None:
            sc_kw["color"] = series_colors[i]
        ax.scatter(
            df[obj_cols[0]], df[obj_cols[1]], label=labels[i], alpha=0.3, zorder=5, **sc_kw
        )
        if annotate_id_col is not None and annotate_id_col in df.columns:
            for _, row in df.iterrows():
                ax.annotate(
                    str(row[annotate_id_col]),
                    (row[obj_cols[0]], row[obj_cols[1]]),
                    fontsize=annotate_fontsize,
                    alpha=0.75,
                )
        ax.set_xlabel(obj_cols[0], fontsize=12)
        ax.set_ylabel(obj_cols[1], fontsize=12)

    if ideal is not None:
        ax.scatter(
            ideal[0],
            ideal[1],
            color="gold",
            label="Ideal",
            marker="*",
            s=500,
            zorder=6,
        )

    if baseline_point is not None:
        bx, by = float(baseline_point[0]), float(baseline_point[1])
        if np.isfinite(bx) and np.isfinite(by):
            ax.scatter(
                bx,
                by,
                color="tab:blue",
                marker="X",
                s=160,
                zorder=7,
                label=baseline_label,
                edgecolors="navy",
                linewidths=0.8,
            )

    if x_lims is not None:
        ax.set_xlim(x_lims[0], x_lims[1])
    if y_lims is not None:
        ax.set_ylim(y_lims[0], y_lims[1])
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", fontsize=10, ncol=4)
    ax.set_title(title)
    ax.grid(zorder=0)
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return
