"""
Diagnostic twin-axis figure: Trenton flow + FFMP target vs. lower-basin reservoir MRF shares.

Mirrors the PywrDRB flow-contribution style (total flow on the left axis, stacked
percentages on the right). Lower-basin outputs from PywrDRB use columns
``mrf_trenton_<reservoir>``; see ``pywrdrb.load.output_loader`` and
``drbc_lower_basin_reservoirs``.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ShareMode = Literal["among_lower_basin", "of_target"]
EnsembleReduce = Literal["mean", "median"]
TrentonFlowInput = Union[pd.Series, pd.DataFrame, Mapping[Any, pd.Series], Mapping[Any, pd.DataFrame]]
LowerBasinInput = Union[pd.DataFrame, Mapping[Any, pd.DataFrame]]
TargetInput = Union[pd.Series, pd.DataFrame, Mapping[Any, pd.Series], Mapping[Any, pd.DataFrame]]


LOWER_BASIN_COLORS: dict[str, str] = {
    "beltzvilleCombined": "#2166ac",
    "blueMarsh": "#4393c3",
    "nockamixon": "#92c5de",
}

DEFAULT_DISPLAY_NAMES: dict[str, str] = {
    "beltzvilleCombined": "Beltzville",
    "blueMarsh": "Blue Marsh",
    "nockamixon": "Nockamixon",
}


def _subset(df: pd.DataFrame | pd.Series, start, end) -> pd.DataFrame | pd.Series:
    if start is None and end is None:
        return df
    return df.loc[start:end]


def _dataframe_with_reservoir_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        s = str(c)
        if s.startswith("mrf_trenton_"):
            rename[c] = s.split("mrf_trenton_", 1)[-1]
    if rename:
        df = df.rename(columns=rename)
    return df


def _apply_lower_basin_routing_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align each reservoir's MRF contribution with arrival at Trenton / output_del,
    matching ``pywrdrb.post.metrics.get_lagged_lower_basin_contributions``.
    """
    try:
        from pywrdrb.pywr_drb_node_data import (
            downstream_node_lags,
            immediate_downstream_nodes_dict,
        )
    except ImportError:
        return df

    out = df.copy()
    for c in out.columns:
        if c not in downstream_node_lags:
            continue
        lag = downstream_node_lags[c]
        downstream_node = immediate_downstream_nodes_dict[c]
        while downstream_node != "output_del":
            lag += downstream_node_lags[downstream_node]
            downstream_node = immediate_downstream_nodes_dict[downstream_node]
        if lag > 0 and len(out.index) > lag:
            lag_start = out.index[lag]
            lag_end = out.index[-lag]
            out.loc[lag_start:, c] = out.loc[:lag_end, c].values
    return out


def _ensemble_flow_stats(
    flow: pd.DataFrame,
    smoothing_window: int,
    q_low: float,
    q_high: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    rolled = flow.rolling(smoothing_window, center=True).mean()
    med = rolled.median(axis=1)
    lo = rolled.quantile(q_low, axis=1)
    hi = rolled.quantile(q_high, axis=1)
    return med, lo, hi


def _is_mapping(x: object) -> bool:
    return isinstance(x, Mapping) and not isinstance(x, (str, bytes))


def _reduce_frame_dict(
    d: Mapping[Any, pd.DataFrame],
    how: EnsembleReduce,
) -> pd.DataFrame:
    if len(d) == 0:
        raise ValueError("Empty mapping for ensemble-style input.")
    keys = list(d.keys())
    standardized = [_dataframe_with_reservoir_columns(df.astype(float)) for df in d.values()]
    idx = standardized[0].index
    for i, df in enumerate(standardized):
        standardized[i] = df.reindex(idx)
    stacked = pd.concat(standardized, keys=keys, axis=0)
    if how == "mean":
        return stacked.groupby(level=1).mean()
    return stacked.groupby(level=1).median()


def _reduce_series_dict(d: Mapping[Any, pd.Series], how: EnsembleReduce) -> pd.Series:
    if len(d) == 0:
        raise ValueError("Empty mapping for ensemble-style input.")
    df = pd.concat(dict(d), axis=1)
    return df.median(axis=1) if how == "median" else df.mean(axis=1)


def _target_series_from_value(v: pd.Series | pd.DataFrame, column: str) -> pd.Series:
    if isinstance(v, pd.Series):
        return v
    if column in v.columns:
        return v[column]
    if len(v.columns) == 1:
        return v.iloc[:, 0]
    raise ValueError(
        f"Expected column {column!r} in target DataFrame, or a single-column frame; got {list(v.columns)}."
    )


def _coerce_target(
    mrf_target_del_trenton: TargetInput,
    *,
    mrf_target_column: str,
    ensemble_reduce: EnsembleReduce,
) -> pd.Series:
    if isinstance(mrf_target_del_trenton, pd.Series):
        return mrf_target_del_trenton
    if isinstance(mrf_target_del_trenton, pd.DataFrame):
        return _target_series_from_value(mrf_target_del_trenton, mrf_target_column)
    if _is_mapping(mrf_target_del_trenton):
        pieces = []
        for v in mrf_target_del_trenton.values():
            if not isinstance(v, (pd.Series, pd.DataFrame)):
                raise TypeError("Each target mapping value must be a Series or DataFrame.")
            pieces.append(_target_series_from_value(v, mrf_target_column))
        if len(pieces) == 1:
            return pieces[0]
        return _reduce_series_dict(dict(enumerate(pieces)), ensemble_reduce)
    raise TypeError(
        "mrf_target_del_trenton must be a Series, DataFrame, or mapping of Series/DataFrame per realization."
    )


def _coalesce_lower_basin(
    lower_basin_mrf: LowerBasinInput,
    *,
    ensemble_reduce: EnsembleReduce,
) -> pd.DataFrame:
    if isinstance(lower_basin_mrf, pd.DataFrame):
        return lower_basin_mrf
    if _is_mapping(lower_basin_mrf):
        return _reduce_frame_dict(lower_basin_mrf, ensemble_reduce)
    raise TypeError(
        "lower_basin_mrf must be a DataFrame or a mapping (realization id -> DataFrame)."
    )


def _coalesce_trenton_flow(
    trenton_flow: TrentonFlowInput,
    *,
    ensemble_reduce: EnsembleReduce,
    trenton_flow_column: Optional[str],
    draw_ensemble_band: bool,
    smoothing_window: int,
    q_low: float,
    q_high: float,
) -> tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], bool, str]:
    """
    Returns flow_line, lo, hi, use_ensemble_band, line_label_suffix.

    Single column / single realization: no quantile band. Multi-column or multi-realization:
    either draw a band (median line + quantiles) or collapse to mean/median across members.
    """

    def from_wide(tf: pd.DataFrame) -> tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], bool, str]:
        if tf.shape[1] == 1:
            return tf.iloc[:, 0], None, None, False, ""
        if draw_ensemble_band:
            med, lo, hi = _ensemble_flow_stats(tf, smoothing_window, q_low, q_high)
            return med, lo, hi, True, " (median)"
        reduced = tf.median(axis=1) if ensemble_reduce == "median" else tf.mean(axis=1)
        return reduced, None, None, False, f" ({ensemble_reduce})"

    if isinstance(trenton_flow, pd.Series):
        return trenton_flow, None, None, False, ""

    if isinstance(trenton_flow, pd.DataFrame):
        tf = trenton_flow.astype(float)
        if trenton_flow_column is not None:
            if trenton_flow_column not in tf.columns:
                raise KeyError(
                    f"trenton_flow_column {trenton_flow_column!r} not in columns {list(tf.columns)}"
                )
            return tf[trenton_flow_column], None, None, False, ""
        if tf.shape[1] == 1:
            return tf.iloc[:, 0], None, None, False, ""
        return from_wide(tf)

    if _is_mapping(trenton_flow):
        d = dict(trenton_flow)
        if len(d) == 0:
            raise ValueError("trenton_flow mapping is empty.")
        if len(d) == 1:
            only = next(iter(d.values()))
            if isinstance(only, pd.DataFrame):
                only = only.astype(float)
                if trenton_flow_column is not None:
                    if trenton_flow_column not in only.columns:
                        raise KeyError(
                            f"trenton_flow_column {trenton_flow_column!r} not in columns {list(only.columns)}"
                        )
                    return only[trenton_flow_column], None, None, False, ""
                if only.shape[1] == 1:
                    return only.iloc[:, 0], None, None, False, ""
                return from_wide(only)
            return only.astype(float), None, None, False, ""

        cols = []
        for v in d.values():
            if isinstance(v, pd.DataFrame):
                if trenton_flow_column is None:
                    raise ValueError(
                        "trenton_flow is a mapping of DataFrames; set trenton_flow_column (e.g. 'delTrenton')."
                    )
                v = v.astype(float)
                if trenton_flow_column not in v.columns:
                    raise KeyError(
                        f"trenton_flow_column {trenton_flow_column!r} not in columns {list(v.columns)}"
                    )
                cols.append(v[trenton_flow_column])
            else:
                cols.append(v.astype(float))
        wide = pd.concat(cols, axis=1)
        return from_wide(wide)

    raise TypeError(
        "trenton_flow must be a Series, DataFrame, or mapping (realization -> Series or flow DataFrame)."
    )


def plot_lower_basin_trenton_target_contributions(
    trenton_flow: TrentonFlowInput,
    mrf_target_del_trenton: TargetInput,
    lower_basin_mrf: LowerBasinInput,
    trenton_flow_obs: Optional[pd.Series] = None,
    *,
    start_date=None,
    end_date=None,
    share_mode: ShareMode = "among_lower_basin",
    apply_routing_lag: bool = True,
    smoothing_window: int = 1,
    contribution_fill_alpha: float = 0.88,
    ensemble_q_low: float = 0.05,
    ensemble_q_high: float = 0.95,
    draw_ensemble_band: bool = False,
    ensemble_reduce: EnsembleReduce = "mean",
    trenton_flow_column: Optional[str] = None,
    mrf_target_column: str = "mrf_target_delTrenton",
    ensemble_color: str = "#7b6cff",
    obs_color: str = "k",
    target_color: str = "k",
    target_linestyle: str = ":",
    units: str = "MGD",
    fontsize: int = 10,
    figsize: tuple[float, float] = (8.0, 3.2),
    dpi: int = 200,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    display_names: Optional[Mapping[str, str]] = None,
    colors: Optional[Mapping[str, str]] = None,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Twin-axis diagnostic: Trenton flow (+ optional ensemble band, target, obs) and
    stacked lower-basin MRF percentages.

    Parameters
    ----------
    trenton_flow
        Simulated Trenton flow at ``delTrenton``: a ``Series``; a ``DataFrame`` whose
        columns are ensemble members; or a mapping ``realization_id -> Series`` /
        ``-> DataFrame`` (use ``trenton_flow_column`` for the latter).         With multiple members, set ``draw_ensemble_band=True`` to draw a 5–95% band;
        otherwise (default) the line is the mean or median across members only.
    mrf_target_del_trenton
        Target as a ``Series``, a one-column or named-column ``DataFrame``, or a
        mapping per realization (e.g. ``all_mrf[model]``). Column name defaults to
        ``mrf_target_delTrenton``; override with ``mrf_target_column``.
    lower_basin_mrf
        ``DataFrame`` of ``mrf_trenton_*`` / reservoir columns, or a mapping
        ``realization_id -> DataFrame`` (collapsed with ``ensemble_reduce``).
    trenton_flow_obs
        Optional observed Trenton flow (same units as simulation).
    draw_ensemble_band
        If True and there are multiple flow members, show quantile shading; if False,
        collapse to ``ensemble_reduce`` without a band.
    ensemble_reduce
        How to collapse multiple lower-basin frames or non-banded flow members.
    trenton_flow_column
        Required when passing a mapping of DataFrames (e.g. ``'delTrenton'``).
    mrf_target_column
        Column to read when targets are given as DataFrames or mapping of DataFrames.

    Returns
    -------
    ax, ax_twin
        Primary axis (flow / target) and twin (stacked shares).
    """
    names = {**DEFAULT_DISPLAY_NAMES, **(display_names or {})}
    colormap = {**LOWER_BASIN_COLORS, **(colors or {})}

    flow_line, lo, hi, use_ensemble_band, flow_label_suffix = _coalesce_trenton_flow(
        trenton_flow,
        ensemble_reduce=ensemble_reduce,
        trenton_flow_column=trenton_flow_column,
        draw_ensemble_band=draw_ensemble_band,
        smoothing_window=smoothing_window,
        q_low=ensemble_q_low,
        q_high=ensemble_q_high,
    )

    flow_line = _subset(flow_line.copy(), start_date, end_date).astype(float)
    if lo is not None:
        lo = _subset(lo, start_date, end_date).astype(float)
        hi = _subset(hi, start_date, end_date).astype(float)
    if not use_ensemble_band and smoothing_window > 1:
        flow_line = flow_line.rolling(smoothing_window, center=True).mean()

    target_full = _coerce_target(
        mrf_target_del_trenton,
        mrf_target_column=mrf_target_column,
        ensemble_reduce=ensemble_reduce,
    )
    target = _subset(target_full.copy(), start_date, end_date).astype(float)

    lb = _coalesce_lower_basin(lower_basin_mrf, ensemble_reduce=ensemble_reduce)
    lb = _subset(lb.copy(), start_date, end_date).astype(float)
    lb = _dataframe_with_reservoir_columns(lb)
    if apply_routing_lag:
        lb = _apply_lower_basin_routing_lag(lb)

    try:
        from pywrdrb.utils.lists import drbc_lower_basin_reservoirs
    except ImportError:
        drbc_lower_basin_reservoirs = list(DEFAULT_DISPLAY_NAMES.keys())

    res_cols = [r for r in drbc_lower_basin_reservoirs if r in lb.columns]
    if not res_cols:
        res_cols = list(lb.columns)

    if not res_cols:
        raise ValueError("lower_basin_mrf has no recognizable reservoir columns.")

    lb = lb[res_cols].fillna(0.0)
    if smoothing_window > 1:
        lb = lb.rolling(smoothing_window, center=True).mean().fillna(0.0)

    target = target.reindex(flow_line.index).ffill().bfill()
    lb = lb.reindex(flow_line.index).fillna(0.0)

    if share_mode == "among_lower_basin":
        denom = lb.sum(axis=1).replace(0, np.nan)
        pct = lb.div(denom, axis=0) * 100.0
        pct = pct.fillna(0.0)
    elif share_mode == "of_target":
        denom = target.replace(0, np.nan)
        pct = lb.div(denom, axis=0) * 100.0
        pct = pct.fillna(0.0)
    else:
        raise ValueError(f"Unknown share_mode: {share_mode!r}")

    x = flow_line.index
    y0 = np.zeros(len(x))
    layers: list[tuple[str, np.ndarray]] = []
    acc = y0.copy()
    for c in res_cols:
        acc = acc + pct[c].values
        layers.append((c, acc.copy()))
    stack_top = layers[-1][1] if layers else y0

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if use_ensemble_band and lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=ensemble_color, alpha=0.35, linewidth=0, zorder=2)
    ax.plot(
        x,
        flow_line,
        color=ensemble_color,
        lw=1.1,
        zorder=3,
        label=f"Sim. Trenton{flow_label_suffix}",
    )

    ax.plot(
        x,
        target.values,
        color=target_color,
        ls=target_linestyle,
        lw=1.0,
        zorder=4,
        label="Min. flow target (Trenton)",
    )

    if trenton_flow_obs is not None:
        obs = _subset(trenton_flow_obs.copy(), start_date, end_date).astype(float)
        if smoothing_window > 1:
            obs = obs.rolling(smoothing_window, center=True).mean()
        obs = obs.reindex(x).dropna()
        if len(obs):
            ax.plot(
                obs.index,
                obs.values,
                color=obs_color,
                ls="--",
                lw=1.0,
                zorder=5,
                label="Obs. Trenton",
            )

    ax.set_ylabel(f"Trenton flow ({units})", fontsize=fontsize)
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, alpha=0.25)

    ax_twin = ax.twinx()
    prev = y0
    for i, (c, top) in enumerate(layers):
        color = colormap.get(c, f"C{i % 10}")
        label = names.get(c, c)
        ax_twin.fill_between(x, prev, top, color=color, alpha=contribution_fill_alpha, lw=0, zorder=1, label=label)
        prev = top

    if share_mode == "among_lower_basin":
        ax_twin.set_ylim(0, 100)
    else:
        hi = float(np.nanmax(stack_top)) if len(stack_top) else 100.0
        ax_twin.set_ylim(0, max(100.0, hi * 1.05))
    ax_twin.set_ylabel(
        "Lower-basin MRF (% of sum)" if share_mode == "among_lower_basin" else "Lower-basin MRF (% of target)",
        fontsize=fontsize,
    )

    ax_twin.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    if legend:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_twin.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=fontsize - 1, ncol=2)

    if created_fig:
        plt.tight_layout()

    return ax, ax_twin
