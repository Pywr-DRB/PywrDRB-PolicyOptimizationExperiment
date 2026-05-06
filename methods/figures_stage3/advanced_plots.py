"""
Stage 3 diagnostic figures (full-Pareto ensemble): bias, attribution, failure map, Pareto projection,
flow-regime split, lag/propagation, empirical policy surface, drought trace.

All are invoked from :func:`methods.figures_stage3.stage3_analysis.run_stage3_full_pareto_analysis`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methods.borg_paths import borg_variant_resolve_kwargs, normalize_borg_variant
from methods.plotting.plot_pareto_ensemble_uncertainty import (
    PARAMETRIC_MOEA_RESERVOIR_KEYS,
    envelope_ir_binned,
    _inflow_series_on_dates,
)
from methods.postprocess.figures_primary import load_filtered_borg_solution_tables

from .constants import (
    POLICY_ORDER,
    RESERVOIR_DISPLAY_NAMES,
    RESERVOIR_KEYS,
    STAGE3_POLICY_COLORS,
)
from .data_loading import Stage3DiagnosticContext
from .full_pareto_output_paths import full_pareto_png_path


def _storage_nse_for_reservoir(
    solution_objs: Dict[str, Any],
    policy: str,
    alignment_index: int,
    reservoir: str,
) -> float:
    objdf = (solution_objs.get(reservoir) or {}).get(policy)
    if objdf is None or alignment_index >= len(objdf):
        return float("nan")
    if "Storage NSE" not in objdf.columns:
        return float("nan")
    return float(objdf["Storage NSE"].iloc[alignment_index])


def run_all_diagnostic_figures(
    ctx: Stage3DiagnosticContext, out_dir: str
) -> List[str]:
    """Run all diagnostic PNGs; return written paths."""
    out: List[str] = []
    if not ctx.runs:
        print("[full-pareto figs] diagnostic: no runs in context — skipping PNGs", flush=True)
        return out

    raw_variants = {str(r.get("borg_variant", "")).strip() or "full" for r in ctx.runs}
    sol_cache: Dict[str, Tuple[Any, Any]] = {}
    for v in raw_variants:
        if not v:
            continue
        canon = normalize_borg_variant(v)
        if canon in sol_cache:
            continue
        try:
            vk = borg_variant_resolve_kwargs(canon)
            so, sv, _, _ = load_filtered_borg_solution_tables(
                list(PARAMETRIC_MOEA_RESERVOIR_KEYS),
                list(POLICY_ORDER),
                verbose=False,
                **vk,
            )
            sol_cache[canon] = (so, sv)
        except Exception as e:
            print(f"[full-pareto figs] Borg objectives skip variant={v}: {e}", flush=True)
            sol_cache[canon] = (None, None)

    writers = (
        plot_bias_surface,
        plot_trenton_attribution,
        plot_failure_alignment,
        plot_reliability_storage_pareto,
        plot_flow_regime_performance_split,
        plot_temporal_lag_propagation,
        plot_policy_surface_contour,
        plot_extreme_event_case_study,
    )
    for fn in writers:
        try:
            path_or_paths = fn(ctx, sol_cache, out_dir)
            if not path_or_paths:
                continue
            if isinstance(path_or_paths, (list, tuple)):
                for path in path_or_paths:
                    if path:
                        out.append(path)
                        print(f"[full-pareto figs] wrote {path}", flush=True)
            else:
                out.append(path_or_paths)
                print(f"[full-pareto figs] wrote {path_or_paths}", flush=True)
        except Exception as e:
            print(f"[full-pareto figs] {fn.__name__} failed: {e}", flush=True)
    return out


def plot_bias_surface(
    ctx: Stage3DiagnosticContext,
    sol_cache: Dict[str, Tuple[Any, Any]],
    out_dir: str,
) -> str:
    """ΔR = R_policy − R_obs vs binned inflow (log), median + 10–90% per reservoir."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
    dates = ctx.dates
    if dates is None:
        raise ValueError("no dates in context")
    for ax, r in zip(axes, RESERVOIR_KEYS):
        orel = ctx.obs_release_by_res.get(r)
        if orel is None:
            continue
        inf = _inflow_series_on_dates(r, dates)
        for pol in POLICY_ORDER:
            deltas = []
            for run in ctx.runs:
                if run["policy"] != pol:
                    continue
                sim = run["releases"].get(r)
                if sim is None or len(sim) != len(orel):
                    continue
                dlt = sim[: len(orel)] - orel[: len(sim)]
                deltas.append(dlt)
            if not deltas:
                continue
            T = max(len(x) for x in deltas)
            M = np.full((T, len(deltas)), np.nan)
            for j, col in enumerate(deltas):
                M[: len(col), j] = col
            ir = envelope_ir_binned(inf[:T], M, n_bins=60)
            xb, m = ir["x"], ir["median"]
            ax.plot(xb, m, color=STAGE3_POLICY_COLORS[pol], lw=2, label=pol)
            ax.fill_between(xb, ir["q10"], ir["q90"], color=STAGE3_POLICY_COLORS[pol], alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("Inflow (MGD)")
        ax.set_title(RESERVOIR_DISPLAY_NAMES.get(r, r))
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("ΔR (policy − obs) MGD")
    fig.suptitle("Policy deviation from observed release (bias vs inflow)", fontsize=14, fontweight="bold")
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=STAGE3_POLICY_COLORS[p], lw=2, label=p) for p in POLICY_ORDER
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
    )
    out = full_pareto_png_path(out_dir, "bias_surface")
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_trenton_attribution(
    ctx: Stage3DiagnosticContext,
    sol_cache: Dict[str, Tuple[Any, Any]],
    out_dir: str,
) -> str:
    """Stacked area of lower-basin MRF contributions to Trenton (median across runs per policy)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    policy_to_mrf: Dict[str, List[pd.DataFrame]] = {p: [] for p in POLICY_ORDER}
    for run in ctx.runs:
        mrf = run.get("mrf")
        if mrf is None or not isinstance(mrf, pd.DataFrame) or mrf.empty:
            continue
        policy_to_mrf[run["policy"]].append(mrf)

    res_cols: List[str] = []
    for pol in POLICY_ORDER:
        lst = policy_to_mrf.get(pol) or []
        if lst:
            res_cols = [c for c in lst[0].columns if np.issubdtype(lst[0][c].dtype, np.number)]
            break
    if not res_cols:
        for run in ctx.runs:
            mrf = run.get("mrf")
            if mrf is not None and isinstance(mrf, pd.DataFrame) and not mrf.empty:
                res_cols = [c for c in mrf.columns if np.issubdtype(mrf[c].dtype, np.number)]
                break

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(res_cols), 1)))

    for ax, pol in zip(axes, POLICY_ORDER):
        dfs = policy_to_mrf.get(pol) or []
        if not dfs:
            ax.set_title(f"{pol} (no MRF)")
            continue
        use_cols = [c for c in res_cols if c in dfs[0].columns][:12]
        if not use_cols:
            ax.set_title(f"{pol} (no numeric cols)")
            continue
        arrs = [df[use_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) for df in dfs]
        min_t = min(a.shape[0] for a in arrs)
        stack = np.stack([a[:min_t] for a in arrs], axis=0)
        median_df = pd.DataFrame(np.nanmedian(stack, axis=0), columns=use_cols)
        idx = dfs[0].index[:min_t]
        median_df.index = idx
        bottom = np.zeros(len(idx))
        for i, c in enumerate(use_cols):
            y = np.asarray(median_df[c].fillna(0), dtype=float)
            ax.fill_between(
                idx,
                bottom,
                bottom + y,
                alpha=0.55,
                color=colors[i % len(colors)],
                label=str(c),
            )
            bottom = bottom + y
        ax.set_title(pol)
        ax.set_ylabel("Contribution (MGD)")
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("Time")
    fig.suptitle("Lower-basin MRF contributions toward Trenton (median across solutions)", fontsize=13, fontweight="bold")
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=min(4, len(l)), bbox_to_anchor=(0.5, -0.05))
    out = full_pareto_png_path(out_dir, "trenton_attribution")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_failure_alignment(
    ctx: Stage3DiagnosticContext,
    sol_cache: Dict[str, Tuple[Any, Any]],
    out_dir: str,
) -> str:
    """Heatmap: solutions × time (subsampled) — Trenton below target."""
    dates = ctx.dates
    if dates is None:
        raise ValueError("no dates")
    rows = []
    meta = []
    for run in ctx.runs:
        tr = run["trenton"]
        fail = (tr < ctx.target_mgd).astype(float)
        rows.append(fail)
        meta.append(f"{run['policy'][:2]}")
    max_r = 200
    if len(rows) > max_r:
        idx = np.linspace(0, len(rows) - 1, max_r, dtype=int)
        rows = [rows[i] for i in idx]
    T = max(len(r) for r in rows)
    M = np.ones((len(rows), T)) * np.nan
    for i, r in enumerate(rows):
        M[i, : len(r)] = r[:T]
    # compress time: weekly max
    step = max(1, T // 400)
    Msub = M[:, ::step]
    fig, ax = plt.subplots(figsize=(12, min(0.12 * len(rows) + 1, 24)))
    im = ax.imshow(Msub, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
    ax.set_xlabel(f"Time index (step={step} d)")
    ax.set_ylabel("Solution (subsampled)")
    ax.set_title("Failure alignment: Trenton below target (1 = failure)")
    plt.colorbar(im, ax=ax, fraction=0.02)
    out = full_pareto_png_path(out_dir, "failure_alignment")
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_reliability_storage_pareto(
    ctx: Stage3DiagnosticContext,
    sol_cache: Dict[str, Tuple[Any, Any]],
    out_dir: str,
) -> str:
    """2×2 panels: Storage NSE (per reservoir) vs Trenton reliability; color = policy."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9), sharey=True)
    axes_flat = axes.ravel()
    h = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=STAGE3_POLICY_COLORS[p], label=p)
        for p in POLICY_ORDER
    ]
    for ax, res in zip(axes_flat, RESERVOIR_KEYS):
        xs: List[float] = []
        ys: List[float] = []
        cs: List[str] = []
        for run in ctx.runs:
            try:
                canon = normalize_borg_variant(str(run.get("borg_variant", "") or "full"))
            except Exception:
                continue
            so, _ = sol_cache.get(canon, (None, None))
            if so is None:
                continue
            sx = _storage_nse_for_reservoir(so, run["policy"], run["alignment_index"], res)
            if not np.isfinite(sx):
                continue
            xs.append(sx)
            ys.append(run["reliability"])
            cs.append(STAGE3_POLICY_COLORS.get(run["policy"], "#333"))
        if xs:
            ax.scatter(xs, ys, c=cs, s=22, alpha=0.65, edgecolors="none")
        ax.set_title(RESERVOIR_DISPLAY_NAMES.get(res, res))
        ax.set_xlabel("Storage NSE (Borg)")
        ax.grid(True, alpha=0.3)
        if xs:
            y0, y1 = ax.get_ylim()
            if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
                pad = 0.02 * (y1 - y0)
                ax.set_ylim(max(0, y0 - pad), min(1.02, y1 + pad))
        else:
            ax.set_ylim(0, 1.02)
    axes_flat[0].set_ylabel("Simulated Trenton reliability")
    axes_flat[2].set_ylabel("Simulated Trenton reliability")
    fig.legend(handles=h, title="Policy", loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Reliability vs storage NSE by reservoir", fontsize=13, fontweight="bold", y=1.03)
    out = full_pareto_png_path(out_dir, "reliability_storage")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_flow_regime_performance_split(
    ctx: Stage3DiagnosticContext,
    sol_cache: Dict[str, Tuple[Any, Any]],
    out_dir: str,
) -> str:
    """Per-reservoir rows: Trenton reliability by observed inflow regime; policy hues; y scaled to data."""
    dates = ctx.dates
    if dates is None:
        raise ValueError("no dates")

    fig, axes = plt.subplots(len(RESERVOIR_KEYS), 3, figsize=(12, 13), sharex=True)
    if len(RESERVOIR_KEYS) == 1:
        axes = np.asarray([axes])
    labels = ("Low inflow", "Mid inflow", "High inflow")

    for row, res in enumerate(RESERVOIR_KEYS):
        inf0 = _inflow_series_on_dates(res, dates)
        q20, q80 = np.nanpercentile(inf0, [20, 80])
        regime = np.where(inf0 < q20, 0, np.where(inf0 <= q80, 1, 2))

        buckets: Dict[str, Dict[int, List[float]]] = {p: {0: [], 1: [], 2: []} for p in POLICY_ORDER}
        for run in ctx.runs:
            tr = run["trenton"]
            m = min(len(tr), len(regime))
            for g in (0, 1, 2):
                mask = regime[:m] == g
                if not np.any(mask):
                    continue
                rel = float(np.mean(tr[:m][mask] >= ctx.target_mgd))
                buckets[run["policy"]][g].append(rel)

        row_axes = axes[row]
        ymax = 0.05
        for g, ax in enumerate(row_axes):
            data = [buckets[p][g] for p in POLICY_ORDER]
            parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True, showextrema=False)
            for b, p in zip(parts["bodies"], POLICY_ORDER):
                b.set_facecolor(STAGE3_POLICY_COLORS[p])
                b.set_edgecolor(STAGE3_POLICY_COLORS[p])
                b.set_alpha(0.55)
            flat = [v for sub in data for v in sub]
            if flat:
                ymax = max(ymax, float(np.nanmax(flat)))
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(POLICY_ORDER, fontsize=8)
            ax.set_title(labels[g] if row == 0 else "")
            ax.grid(True, axis="y", alpha=0.25)
            ymin = 0.0
            ytop = min(1.02, ymax * 1.08 + 0.02) if ymax > 0 else 1.02
            ax.set_ylim(ymin, ytop)
        row_axes[0].set_ylabel(
            f"{RESERVOIR_DISPLAY_NAMES.get(res, res)}\nTrenton reliability",
            fontsize=10,
        )

    fig.suptitle(
        "Reliability by inflow regime (thresholds from each reservoir's observed inflow)",
        fontsize=12,
        fontweight="bold",
    )
    out = full_pareto_png_path(out_dir, "flow_regime_split")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_temporal_lag_propagation(
    ctx: Stage3DiagnosticContext,
    sol_cache: Dict[str, Tuple[Any, Any]],
    out_dir: str,
) -> str:
    """Correlation(release, Trenton) vs lag (days); median across solutions per policy × reservoir."""
    lags = np.arange(0, 21)
    corr_acc = {
        r: {p: [] for p in POLICY_ORDER} for r in RESERVOIR_KEYS
    }
    for run in ctx.runs:
        tr = run["trenton"] - np.nanmean(run["trenton"])
        pol = run["policy"]
        for r in RESERVOIR_KEYS:
            rel = run["releases"].get(r)
            if rel is None or len(rel) != len(tr):
                continue
            x = rel - np.nanmean(rel)
            best = []
            for lag in lags:
                if lag == 0:
                    a, b = x, tr
                else:
                    a, b = x[:-lag], tr[lag:]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 30:
                    best.append(np.nan)
                    continue
                best.append(float(np.corrcoef(a[m], b[m])[0, 1]))
            corr_acc[r][pol].append(best)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, r in zip(axes, RESERVOIR_KEYS):
        for pol in POLICY_ORDER:
            arrs = corr_acc[r].get(pol) or []
            if not arrs:
                continue
            med = np.nanmedian(np.array(arrs), axis=0)
            ax.plot(lags, med, color=STAGE3_POLICY_COLORS[pol], lw=2, label=pol)
        ax.set_title(RESERVOIR_DISPLAY_NAMES.get(r, r))
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Correlation")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Release–Trenton correlation vs lag (median across ensemble)", fontsize=13, fontweight="bold")
    out = full_pareto_png_path(out_dir, "temporal_lag")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_policy_surface_contour(
    ctx: Stage3DiagnosticContext,
    sol_cache: Dict[str, Tuple[Any, Any]],
    out_dir: str,
) -> str:
    """4×3 grid: rows = reservoirs, columns = policies; hexbin median release vs DOY × storage fraction."""
    from methods.config import reservoir_capacity

    fig, axes = plt.subplots(
        len(RESERVOIR_KEYS),
        len(POLICY_ORDER),
        figsize=(14, 18),
        sharex=True,
        sharey=True,
    )

    for i, focus in enumerate(RESERVOIR_KEYS):
        cap = float(reservoir_capacity.get(focus, 1.0) or 1.0)
        for j, pol in enumerate(POLICY_ORDER):
            ax = axes[i, j]
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            for run in ctx.runs:
                if run["policy"] != pol:
                    continue
                s = run["storage"].get(focus)
                r = run["releases"].get(focus)
                if s is None or r is None:
                    continue
                sfrac = np.clip(s.to_numpy(dtype=float) if hasattr(s, "to_numpy") else s, 0, None) / cap
                sfrac = np.clip(sfrac, 0, 1)
                if ctx.dates is not None and len(ctx.dates) == len(r):
                    doy = pd.DatetimeIndex(ctx.dates).dayofyear.to_numpy()[: len(r)]
                else:
                    doy = (np.arange(len(r)) % 366) + 1
                rv = r.to_numpy(dtype=float) if hasattr(r, "to_numpy") else np.asarray(r, dtype=float)
                m = len(rv)
                for k in range(m):
                    xs.append(float(doy[k]))
                    ys.append(float(sfrac[k]))
                    zs.append(float(rv[k]))
            if len(xs) < 50:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            else:
                hb = ax.hexbin(
                    xs,
                    ys,
                    C=zs,
                    reduce_C_function=np.median,
                    gridsize=22,
                    cmap="viridis",
                    mincnt=3,
                )
                cb = plt.colorbar(hb, ax=ax, fraction=0.035)
                if j == len(POLICY_ORDER) - 1:
                    cb.set_label("Release (MGD)", fontsize=8)
            if i == 0:
                ax.set_title(pol)
            if j == 0:
                ax.set_ylabel(f"{RESERVOIR_DISPLAY_NAMES.get(focus, focus)}\nStorage (fraction)")
            if i == len(RESERVOIR_KEYS) - 1:
                ax.set_xlabel("Day of year")

    fig.suptitle(
        "Empirical release (hexbin median): day of year × storage fraction",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = full_pareto_png_path(out_dir, "policy_surface")
    plt.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_extreme_event_case_study(
    ctx: Stage3DiagnosticContext,
    sol_cache: Dict[str, Tuple[Any, Any]],
    out_dir: str,
) -> str:
    """Worst observed-flow summer window: overlay policy median Trenton + releases."""
    dates = ctx.dates
    if dates is None:
        raise ValueError("no dates")
    inf = _inflow_series_on_dates("beltzvilleCombined", dates)
    # 120-day window minimizing mean inflow
    win = 120
    best_i = 0
    best_score = np.inf
    for i in range(0, len(inf) - win):
        j = i + win
        score = float(np.nanmean(inf[i:j]))
        if score < best_score:
            best_score = score
            best_i = i
    sl = slice(best_i, best_i + win)
    tidx = np.arange(win)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for pol in POLICY_ORDER:
        trs = [run["trenton"][sl] for run in ctx.runs if run["policy"] == pol and len(run["trenton"]) >= best_i + win]
        if not trs:
            continue
        mat = np.full((win, len(trs)), np.nan)
        for j, tr in enumerate(trs):
            mat[: len(tr), j] = tr[:win]
        med = np.nanmedian(mat, axis=1)
        axes[0].plot(tidx, med, lw=2, color=STAGE3_POLICY_COLORS[pol], label=pol)
    axes[0].axhline(ctx.target_mgd, color="k", ls="--")
    axes[0].set_ylabel("Trenton (MGD)")
    axes[0].set_title(f"Drought case study: {dates[best_i].date()} — {dates[best_i + win - 1].date()}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    for pol in POLICY_ORDER:
        rs = [
            run["releases"]["prompton"][sl]
            for run in ctx.runs
            if run["policy"] == pol and "prompton" in run["releases"] and len(run["releases"]["prompton"]) >= best_i + win
        ]
        if not rs:
            continue
        mat = np.full((win, len(rs)), np.nan)
        for j, tr in enumerate(rs):
            mat[: len(tr), j] = tr[:win]
        med = np.nanmedian(mat, axis=1)
        axes[1].plot(tidx, med, lw=2, color=STAGE3_POLICY_COLORS[pol], label=pol)
    axes[1].set_xlabel("Day within window")
    axes[1].set_ylabel("Prompton release (MGD)")
    axes[1].grid(True, alpha=0.25)

    out = full_pareto_png_path(out_dir, "extreme_event")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
