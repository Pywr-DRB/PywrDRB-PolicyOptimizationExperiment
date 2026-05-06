"""
Figure 6 v2: policy diagnostics without Pywr-DRB plotting entrypoints.

Row A — policy-specific (NOR band, PWL marginals, or RBF slices).
Row B — z and release vs storage at median inflow, for two day-of-year slices.

Saves with plt.savefig + plt.close (no plt.show) for headless batch runs.
"""

from pathlib import Path
from typing import Any, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _require_context(policy: Any) -> None:
    if getattr(policy, "x_min", None) is None or getattr(policy, "x_max", None) is None:
        raise RuntimeError("Policy surface v2: set_context(...) must be called before plotting.")


def _d_norm_from_doy(policy: Any, doy: float) -> float:
    lo = float(policy.x_min[2])
    hi = float(policy.x_max[2])
    return float(np.clip((float(doy) - lo) / (hi - lo), 0.0, 1.0))


def _denorm_axis(policy: Any, axis: int, x_norm: float) -> float:
    lo = float(policy.x_min[axis])
    hi = float(policy.x_max[axis])
    return lo + float(x_norm) * (hi - lo)


def _median_inflow_norm(policy: Any) -> float:
    return 0.5


def _z_grid(policy: Any, d_norm: float, grid: int = 65) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s1 = np.linspace(0.0, 1.0, grid)
    i1 = np.linspace(0.0, 1.0, grid)
    z = np.zeros((grid, grid), dtype=float)
    for si, s in enumerate(s1):
        for ii, inf in enumerate(i1):
            z[ii, si] = float(policy.evaluate([float(s), float(inf), float(d_norm)]))
    return s1, i1, z


def _slice_z_vs_storage(
    policy: Any,
    *,
    i_norm: float,
    d_norm: float,
    n: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    s = np.linspace(0.0, 1.0, n)
    z = np.array([float(policy.evaluate([float(sv), float(i_norm), float(d_norm)])) for sv in s], dtype=float)
    return s, z


def _axis_honesty_suffix(policy: Any) -> str:
    rmax = float(getattr(policy, "release_max", None) or 0.0)
    if rmax > 0:
        return f"Policy output z ∈ [0, 1]; release target = z × R_max (R_max = {rmax:.3f} MGD)"
    return "Policy output z ∈ [0, 1]; release target = z × R_max"


# --- STARFIT -----------------------------------------------------------------


def _plot_starfit(
    policy: Any,
    out_path: Path,
    *,
    title: str,
    doy_pair: Tuple[int, int] = (200, 320),
    grid: int = 65,
    obs_storage_series: pd.Series | None = None,
    obs_release_series: pd.Series | None = None,
    sim_storage_series: pd.Series | None = None,
    sim_release_series: pd.Series | None = None,
    overlay_years: int | None = 10,
) -> None:
    def _last_n_years(series: pd.Series, n_years: int | None):
        s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
        if len(s) == 0:
            return s
        if n_years is None or int(n_years) <= 0:
            return s
        years = sorted(pd.Index(s.index.year).unique().tolist())
        keep = set(years[-int(n_years):])
        return s[s.index.year.isin(keep)]

    def _plot_annual_background(
        ax: plt.Axes,
        series: pd.Series,
        *,
        transform,
        color: str,
        alpha: float = 0.14,
        lw: float = 0.8,
    ) -> None:
        """Plot one faint line per year on a shared hydrologic-year (Oct->Sep) axis."""
        for _, grp in series.groupby(series.index.year):
            g = pd.to_numeric(grp, errors="coerce").dropna()
            if len(g) == 0:
                continue
            # Map month/day to a canonical water year timeline using a leap-safe
            # reference year pair so Feb 29 is always representable.
            # Oct-Dec -> 2019, Jan-Sep -> 2020.
            ts = pd.DatetimeIndex(g.index)
            gdt = pd.to_datetime(
                [
                    pd.Timestamp(
                        year=2019 if t.month >= 10 else 2020,
                        month=t.month,
                        day=t.day,
                    )
                    for t in ts
                ]
            )
            order = np.argsort(gdt.values)
            gy = transform(g.values.astype(float))
            ax.plot(gdt.values[order], gy[order], color=color, alpha=alpha, lw=lw, zorder=1)

    _require_context(policy)
    if getattr(policy, "I_bar", None) is None:
        raise RuntimeError("STARFIT surface v2: I_bar must be set.")

    # Build a canonical hydrologic year axis (Oct->Sep) for policy curves and overlays.
    doy = np.arange(1, 367, dtype=float)
    c = np.pi * doy / 365.0
    nhi = np.clip(
        float(policy.NORhi_mu) + float(policy.NORhi_alpha) * np.sin(2.0 * c) + float(policy.NORhi_beta) * np.cos(2.0 * c),
        float(policy.NORhi_min),
        float(policy.NORhi_max),
    )
    nlo = np.clip(
        float(policy.NORlo_mu) + float(policy.NORlo_alpha) * np.sin(2.0 * c) + float(policy.NORlo_beta) * np.cos(2.0 * c),
        float(policy.NORlo_min),
        float(policy.NORlo_max),
    )
    cal_dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(doy - 1, unit="D")
    wy_dates = pd.to_datetime(
        [
            pd.Timestamp(
                year=2019 if d.month >= 10 else 2020,
                month=d.month,
                day=d.day,
            )
            for d in cal_dates
        ]
    )
    wy_order = np.argsort(wy_dates.values)
    dates = wy_dates.values[wy_order]
    nlo = nlo[wy_order]
    nhi = nhi[wy_order]
    cap = float(policy.storage_capacity)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), layout="constrained")
    fig.suptitle(title + "\nSTARFIT mechanistic structure (paper-style)", fontsize=11)
    ax = axes[0]
    ax.fill_between(dates, nlo, nhi, color="C0", alpha=0.30, label="NOR band")
    ax.plot(dates, nlo, color="C0", lw=1.2, ls="--", label="NOR low")
    ax.plot(dates, nhi, color="C0", lw=1.4, label="NOR high")
    if obs_storage_series is not None and len(obs_storage_series) > 0:
        s = _last_n_years(obs_storage_series, overlay_years)
        if len(s):
            _plot_annual_background(
                ax,
                s,
                transform=lambda y: y / cap,
                color="0.25",
            )
    if sim_storage_series is not None and len(sim_storage_series) > 0:
        ssim = _last_n_years(sim_storage_series, overlay_years)
        if len(ssim):
            _plot_annual_background(
                ax,
                ssim,
                transform=lambda y: y / cap,
                color="C2",
            )
    ax.set_ylabel("Storage (% capacity)")
    ax.set_title("Normal operating range (NOR)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    ax_mg = ax.twinx()
    ax_mg.set_ylabel("NOR (MG)", color="C1", fontsize=9)
    ax_mg.tick_params(axis="y", labelcolor="C1")

    axr = axes[1]
    doy = np.arange(1, 367, dtype=float)
    c = np.pi * doy / 365.0
    harmonic = (
        float(policy.Release_alpha1) * np.sin(2.0 * c)
        + float(policy.Release_alpha2) * np.sin(4.0 * c)
        + float(policy.Release_beta1) * np.cos(2.0 * c)
        + float(policy.Release_beta2) * np.cos(4.0 * c)
    )
    ibar = float(policy.I_bar)
    dt = dates
    harmonic = harmonic[wy_order]
    axr.plot(dt, harmonic, color="C3", lw=1.9, label="Seasonal release harmonic")
    if obs_release_series is not None and len(obs_release_series) > 0:
        r = _last_n_years(obs_release_series, overlay_years)
        if len(r):
            _plot_annual_background(
                axr,
                r,
                transform=(lambda y: (y / ibar) - 1.0) if ibar > 0 else (lambda y: y * np.nan),
                color="0.2",
            )
    if sim_release_series is not None and len(sim_release_series) > 0:
        rsim = _last_n_years(sim_release_series, overlay_years)
        if len(rsim):
            _plot_annual_background(
                axr,
                rsim,
                transform=(lambda y: (y / ibar) - 1.0) if ibar > 0 else (lambda y: y * np.nan),
                color="C2",
            )
    axr.set_title("Seasonal release harmonic + observed annual release trajectories")
    axr.set_ylabel("Release deviation (R / Ibar - 1)")
    axr.axhline(0.0, color="0.4", lw=0.9, ls="--", alpha=0.8)
    axr.grid(True, alpha=0.3)
    axr.legend(loc="upper right", fontsize=8)
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axr.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axr.set_xlabel("Hydrologic year (Oct-Sep)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- PWL ---------------------------------------------------------------------


def _pwl_marginal_curve(policy: Any, axis: str, n: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, n)
    if axis == "S":
        bounds, slopes, intercepts = policy.storage_bounds, policy.storage_slopes, policy.storage_intercepts
    elif axis == "I":
        bounds, slopes, intercepts = policy.inflow_bounds, policy.inflow_slopes, policy.inflow_intercepts
    else:
        bounds, slopes, intercepts = policy.day_bounds, policy.day_slopes, policy.day_intercepts
    y = np.array(
        [max(0.0, min(1.0, policy._segment_eval(float(xi), bounds, slopes, intercepts))) for xi in x],
        dtype=float,
    )
    return x, y


def _plot_pwl(
    policy: Any,
    out_path: Path,
    *,
    title: str,
    doy_pair: Tuple[int, int] = (200, 320),
) -> None:
    _require_context(policy)
    if policy.storage_bounds is None:
        raise RuntimeError("PWL surface v2: parse_policy_params did not run.")

    fig, axd = plt.subplot_mosaic(
        [["s", "i", "d"], ["b1", "b2", "."]],
        figsize=(14.5, 9.0),
        layout="constrained",
        height_ratios=[1.05, 0.95],
    )
    fig.suptitle(title + "\n" + _axis_honesty_suffix(policy), fontsize=11)

    labels = [
        ("s", "S", "Storage", 0),
        ("i", "I", "Inflow", 1),
        ("d", "D", "Day-of-year", 2),
    ]
    for key, axis, name, dim in labels:
        ax = axd[key]
        xv, yv = _pwl_marginal_curve(policy, axis)
        ax.plot(xv, yv, color="k", lw=1.6)
        bounds = getattr(policy, f"{('storage' if axis=='S' else 'inflow' if axis=='I' else 'day')}_bounds")
        knots = bounds[1:-1]
        for j, xk in enumerate(knots):
            ax.axvline(xk, color="C3", ls="--", lw=1.0, alpha=0.85)
            phys = _denorm_axis(policy, dim, xk)
            ax.text(
                xk,
                0.04,
                f"x{j+1}",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=8,
                color="C3",
            )
            unit = "MG" if dim == 0 else ("MGD" if dim == 1 else "doy")
            ax.text(
                xk,
                -0.09,
                f"{phys:.3g} {unit}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=7,
                clip_on=False,
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(f"{name} (normalized)")
        ax.set_ylabel(f"z_{axis} (clamped)")
        ax.set_title(f"PWL marginal: {name}")
        ax.grid(True, alpha=0.3)

    d1, d2 = doy_pair
    i_med = _median_inflow_norm(policy)
    s_knots = policy.storage_bounds[1:-1]
    for key, doy, tag in (("b1", d1, f"DoY {d1}"), ("b2", d2, f"DoY {d2}")):
        dn = _d_norm_from_doy(policy, doy)
        s, z = _slice_z_vs_storage(policy, i_norm=i_med, d_norm=dn)
        a = axd[key]
        s_mg = np.array([_denorm_axis(policy, 0, sv) for sv in s], dtype=float)
        rmax = float(policy.release_max)
        a.plot(s_mg, z, "k-", lw=1.8, label="z (avg of z_S, z_I, z_D)")
        a.plot(s_mg, z * rmax, "C3--", lw=1.4, label="z × R_max (MGD)")
        for xk in s_knots:
            s_phys = _denorm_axis(policy, 0, xk)
            a.axvline(s_phys, color="C3", ls=":", lw=1.0, alpha=0.9)
        a.set_xlabel("Storage (MG)")
        a.set_ylabel("z / release target")
        a.set_title(f"Slice: I_norm = {i_med:.2f}, {tag}; vertical ticks = storage knots")
        a.grid(True, alpha=0.3)
        a.legend(loc="best", fontsize=8)
        a.set_ylim(bottom=0.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- RBF ---------------------------------------------------------------------


def _plot_rbf(
    policy: Any,
    out_path: Path,
    *,
    title: str,
    d_norm_quads: Sequence[float] | None = None,
    doy_pair: Tuple[int, int] = (200, 320),
    grid: int = 55,
) -> None:
    _require_context(policy)
    w = np.asarray(policy.w, dtype=float)
    c = np.asarray(policy.c, dtype=float)
    r = np.asarray(policy.r, dtype=float)
    n = int(policy.nRBFs)
    d_in = int(getattr(policy, "n_inputs", 3))
    if c.ndim == 1:
        c = c.reshape(n, d_in)
    if r.ndim == 1:
        r = r.reshape(n, d_in)

    if d_norm_quads is None:
        d_norm_quads = (0.2, 0.45, 0.55, 0.8)

    fig = plt.figure(figsize=(14.5, 11.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.85], hspace=0.35, wspace=0.28)
    fig.suptitle(title + "\n" + _axis_honesty_suffix(policy), fontsize=11)

    r_s_i = np.sqrt(np.maximum(r[:, 0] * r[:, 1], 1e-12))
    r_scale = 80.0 * (r_s_i / (r_s_i.max() + 1e-9))

    for idx, dn in enumerate(d_norm_quads):
        ridx, cidx = divmod(idx, 2)
        ax = fig.add_subplot(gs[ridx, cidx])
        s1, i1, z = _z_grid(policy, float(dn), grid=grid)
        cf = ax.contourf(s1, i1, z, levels=16, cmap="plasma", vmin=0.0, vmax=1.0)
        ax.contour(s1, i1, z, levels=8, colors="w", linewidths=0.3, alpha=0.5)
        doy_est = _denorm_axis(policy, 2, float(dn))
        ax.scatter(
            c[:, 0],
            c[:, 1],
            s=r_scale,
            c="w",
            edgecolors="k",
            linewidths=0.6,
            alpha=0.9,
            zorder=5,
            label="Centers (size ~ √(r_S r_I))",
        )
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.02, label="z")
        ax.set_xlabel("S_norm")
        ax.set_ylabel("I_norm")
        ax.set_title(f"z(S_norm, I_norm) at D_norm = {dn:.2f} (~doy {doy_est:.0f})")
        ax.set_aspect("auto")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)

    i_med = _median_inflow_norm(policy)
    d1, d2 = doy_pair
    for j, (doy, tag) in enumerate(((d1, f"DoY {d1}"), (d2, f"DoY {d2}"))):
        ax = fig.add_subplot(gs[2, j])
        dn = _d_norm_from_doy(policy, doy)
        s, zv = _slice_z_vs_storage(policy, i_norm=i_med, d_norm=dn)
        s_mg = np.array([_denorm_axis(policy, 0, sv) for sv in s], dtype=float)
        rmax = float(policy.release_max)
        ax.plot(s_mg, zv, "k-", lw=1.8, label="z")
        ax.plot(s_mg, zv * rmax, "C3--", lw=1.4, label="z × R_max (MGD)")
        ax.set_xlabel("Storage (MG)")
        ax.set_ylabel("z / release target")
        ax.set_title(f"Slice: I_norm = {i_med:.2f}, {tag}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(bottom=0.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- public API --------------------------------------------------------------


def save_policy_figure6_v2(
    policy: Any,
    out_path: str | Path,
    *,
    policy_type: str,
    reservoir_name: str,
    pick_slug: str,
    k: int,
    doy_pair: Tuple[int, int] = (200, 320),
    obs_storage_series: pd.Series | None = None,
    obs_release_series: pd.Series | None = None,
    sim_storage_series: pd.Series | None = None,
    sim_release_series: pd.Series | None = None,
    overlay_years: int | None = 10,
) -> None:
    """
    Write one composite PNG for the given policy instance (already set_context + validated).
    """
    title = f"{reservoir_name} — {policy_type} — {pick_slug} — k={k}"
    out_path = Path(out_path)
    pt = policy_type.strip().upper()
    if pt == "STARFIT":
        _plot_starfit(
            policy,
            out_path,
            title=title,
            doy_pair=doy_pair,
            obs_storage_series=obs_storage_series,
            obs_release_series=obs_release_series,
            sim_storage_series=sim_storage_series,
            sim_release_series=sim_release_series,
            overlay_years=overlay_years,
        )
    elif pt == "PWL":
        _plot_pwl(policy, out_path, title=title, doy_pair=doy_pair)
    elif pt == "RBF":
        _plot_rbf(policy, out_path, title=title, doy_pair=doy_pair)
    else:
        raise ValueError(f"Unknown policy_type for figure6 v2: {policy_type!r}")
