"""Shared matplotlib helpers (flow-regime shading, labels)."""

from typing import Optional

import numpy as np


def add_flow_regime_shading(
    ax,
    q20: float,
    q80: float,
    *,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    low_color: str = "#c8d4e6",
    mid_color: str = "#f5f5f5",
    high_color: str = "#e6ddd0",
    alpha: float = 0.18,
    zorder: float = 0,
    label_low: str = "Low flow",
    label_mid: str = "Mid flow",
    label_high: str = "High flow",
    x_scale: str = "log",
    with_labels: bool = True,
) -> None:
    """
    Vertical bands for low / mid / high inflow regimes using **one** pair of thresholds (Q20, Q80)
    from the **observed** inflow record for that reservoir — shared across all policy curves.

    ``xmin`` / ``xmax`` default to current axis limits (call after setting xlim for log axes).
    Use ``x_scale='log'`` for geometric-mean label positions; ``'linear'`` for midpoints.
    """
    xmin = float(xmin if xmin is not None else ax.get_xlim()[0])
    xmax = float(xmax if xmax is not None else ax.get_xlim()[1])
    q20, q80 = float(q20), float(q80)
    if not (np.isfinite(xmin) and np.isfinite(xmax) and xmin > 0 and xmax > xmin):
        return
    if not (np.isfinite(q20) and np.isfinite(q80) and q20 > 0 and q80 > q20):
        return
    # Clamp regime edges into the plotted inflow span so shading works when bin centers
    # omit tails but Q20/Q80 come from the full observed record (common for IR panels).
    lo = xmin * (1.0 + 1e-5)
    hi = xmax * (1.0 - 1e-5)
    if lo >= hi:
        return
    q20u = float(np.clip(q20, lo, hi))
    q80u = float(np.clip(q80, max(q20u * (1.0 + 1e-5), lo), hi))
    if q80u <= q20u:
        return

    ax.axvspan(xmin, q20u, facecolor=low_color, alpha=alpha, zorder=zorder, linewidth=0)
    ax.axvspan(q20u, q80u, facecolor=mid_color, alpha=alpha * 0.9, zorder=zorder, linewidth=0)
    ax.axvspan(q80u, xmax, facecolor=high_color, alpha=alpha, zorder=zorder, linewidth=0)

    def _cx(a: float, b: float) -> float:
        if x_scale == "log":
            return float(np.sqrt(max(a * b, 1e-30)))
        return 0.5 * (a + b)

    if not with_labels:
        return

    ymax = ax.get_ylim()[1]
    ax.text(_cx(xmin, q20u), ymax * 0.92, label_low, ha="center", va="top", fontsize=8, color="0.35", zorder=15)
    ax.text(_cx(q20u, q80u), ymax * 0.92, label_mid, ha="center", va="top", fontsize=8, color="0.35", zorder=15)
    ax.text(_cx(q80u, xmax), ymax * 0.92, label_high, ha="center", va="top", fontsize=8, color="0.35", zorder=15)
