"""
Operational burden metrics for reservoir systems and Trenton-style flow targets.

Outputs are plain pandas objects (and small dict bundles) so you can save tables,
join to MOEA results, or pass them to plotting/analysis notebooks.

Typical inputs (PywrDRB or your simulator):
- Storage as % of capacity (combined NYC or per-reservoir).
- FFMP level labels or ordered level codes.
- Lower-basin ``mrf_trenton_*`` contributions (wide DataFrame).
- Trenton flow and ``mrf_target_delTrenton``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Spell / event utilities
# ---------------------------------------------------------------------------


def find_spells(mask: pd.Series, min_duration: int = 1) -> pd.DataFrame:
    """
    Contiguous True segments along a boolean mask (daily index).

    Returns
    -------
    pd.DataFrame
        Columns: start, end, duration_days
    """
    m = mask.astype(bool).reindex(mask.index).fillna(False)
    if m.empty:
        return pd.DataFrame(columns=["start", "end", "duration_days"])

    change = m.ne(m.shift(fill_value=False))
    gid = change.cumsum()
    rows = []
    for _, sub in m.groupby(gid):
        if not sub.iloc[0]:
            continue
        d = int(len(sub))
        if d < min_duration:
            continue
        rows.append({"start": sub.index[0], "end": sub.index[-1], "duration_days": d})
    return pd.DataFrame(rows)


def spell_summary(mask: pd.Series, min_duration: int = 1) -> dict:
    """Scalar summaries for a boolean stress / deficit mask."""
    spells = find_spells(mask, min_duration=min_duration)
    n = len(mask)
    frac = float(mask.astype(float).mean()) if n else 0.0
    mean_dur = float(spells["duration_days"].mean()) if len(spells) else 0.0
    max_dur = int(spells["duration_days"].max()) if len(spells) else 0
    return {
        "spells": spells,
        "fraction_time": frac,
        "n_spells": int(len(spells)),
        "mean_spell_duration_days": mean_dur,
        "max_spell_duration_days": max_dur,
    }


# ---------------------------------------------------------------------------
# NOR (normal operating range) — storage % band
# ---------------------------------------------------------------------------


def nor_mask(
    storage_pct: pd.Series,
    nor_low_pct: float,
    nor_high_pct: float,
) -> pd.Series:
    """
    True when storage (percent of capacity) lies inside [nor_low_pct, nor_high_pct].
    """
    s = storage_pct.astype(float)
    return (s >= nor_low_pct) & (s <= nor_high_pct)


def nor_operational_burden_metrics(
    storage_pct: pd.Series,
    nor_low_pct: float,
    nor_high_pct: float,
    min_spell_duration: int = 1,
) -> dict:
    """
    Time *outside* NOR and spell statistics for the complementary mask.

    Drought-side stress is often ``storage < nor_low``; flood-side ``storage > nor_high``.
    """
    inside = nor_mask(storage_pct, nor_low_pct, nor_high_pct)
    below = storage_pct.astype(float) < nor_low_pct
    above = storage_pct.astype(float) > nor_high_pct
    outside = ~inside

    return {
        "nor_low_pct": nor_low_pct,
        "nor_high_pct": nor_high_pct,
        "fraction_inside_nor": float(inside.mean()),
        "fraction_below_nor": float(below.mean()),
        "fraction_above_nor": float(above.mean()),
        "outside_nor": spell_summary(outside, min_duration=min_spell_duration),
        "below_nor": spell_summary(below, min_duration=min_spell_duration),
        "above_nor": spell_summary(above, min_duration=min_spell_duration),
    }


def recovery_times_after_spells(
    storage_pct: pd.Series,
    spells_low: pd.DataFrame,
    recovery_level_pct: float,
) -> pd.Series:
    """
    For each spell in ``spells_low`` (from ``find_spells(storage < thresh)``),
    days from spell *end* until storage first reaches ``recovery_level_pct``.

    Returns a Series indexed by spell start (or reindex with range(len)).
    """
    s = storage_pct.astype(float)
    out = []
    for _, row in spells_low.iterrows():
        end = row["end"]
        after = s.loc[s.index > end]
        hit = after[after >= recovery_level_pct]
        if hit.empty:
            out.append(np.nan)
        else:
            delta = (hit.index[0] - end).days
            out.append(float(delta))
    idx = spells_low["start"].values if len(spells_low) else []
    return pd.Series(out, index=pd.DatetimeIndex(idx), name="recovery_lag_days")


# ---------------------------------------------------------------------------
# Storage depletion / stress events
# ---------------------------------------------------------------------------


def stress_event_catalog(
    storage_pct: pd.Series,
    stress_threshold_pct: float,
    min_duration_days: int = 3,
) -> pd.DataFrame:
    """
    Catalog drought-direction stress events (storage below threshold).

    Adds min / mean storage during spell and a simple depletion index
    (mean deficit below threshold, %·days).
    """
    m = storage_pct.astype(float) < stress_threshold_pct
    spells = find_spells(m, min_duration=min_duration_days)
    if spells.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "duration_days",
                "min_storage_pct",
                "mean_storage_pct",
                "deficit_pct_days",
            ]
        )

    mins, means, deficit = [], [], []
    for _, row in spells.iterrows():
        seg = storage_pct.loc[row["start"] : row["end"]].astype(float)
        mins.append(float(seg.min()))
        means.append(float(seg.mean()))
        deficit.append(float((stress_threshold_pct - seg).clip(lower=0).sum()))
    spells = spells.copy()
    spells["min_storage_pct"] = mins
    spells["mean_storage_pct"] = means
    spells["deficit_pct_days"] = deficit
    return spells


def annual_stress_rates(
    event_catalog: pd.DataFrame,
    calendar_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Events per calendar year and stress days per year (overlap with calendar year).
    """
    years = sorted(set(pd.DatetimeIndex(calendar_index).year))
    if not years:
        return pd.DataFrame(columns=["n_events", "stress_days"])

    if event_catalog.empty:
        return pd.DataFrame(
            {"n_events": 0, "stress_days": 0},
            index=pd.Index(years, name="year"),
        )

    rows = []
    starts = pd.to_datetime(event_catalog["start"])
    for y in years:
        y0, y1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
        n_ev = int((starts.dt.year == y).sum())
        stress_days = 0
        for _, ev in event_catalog.iterrows():
            s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
            if e < y0 or s > y1:
                continue
            lo, hi = max(s, y0), min(e, y1)
            stress_days += (hi - lo).days + 1
        rows.append({"year": y, "n_events": n_ev, "stress_days": stress_days})
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Contribution by reservoir
# ---------------------------------------------------------------------------


def contribution_shares(
    contributions_wide: pd.DataFrame,
    mode: Literal["mean_flow", "integrated", "stress_weighted"] = "mean_flow",
    stress_weights: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Per-column share of total lower-basin (or arbitrary) contribution.

    Parameters
    ----------
    contributions_wide
        Daily flows/volumes per reservoir (only numeric columns used).
    mode
        ``mean_flow``: mean of each column / sum of column means.
        ``integrated``: sum over time of each column / grand total sum.
        ``stress_weighted``: column means after multiplying by ``stress_weights``.
    stress_weights
        Required for ``stress_weighted``; aligned index (e.g. shortfall indicator).
    """
    df = contributions_wide.select_dtypes(include=[np.number]).fillna(0.0)
    if df.empty:
        return pd.Series(dtype=float)

    if mode == "mean_flow":
        num = df.mean(axis=0)
    elif mode == "integrated":
        num = df.sum(axis=0)
    elif mode == "stress_weighted":
        if stress_weights is None:
            raise ValueError("stress_weights required for stress_weighted mode.")
        w = stress_weights.reindex(df.index).fillna(0.0).astype(float)
        num = (df.mul(w, axis=0)).mean(axis=0)
    else:
        raise ValueError(mode)

    tot = float(num.sum())
    if tot == 0:
        return pd.Series(0.0, index=num.index)
    return (num / tot).rename("share")


def contribution_daily_fractions(contributions_wide: pd.DataFrame) -> pd.DataFrame:
    """Per-day shares that sum to 1 (or NaN row if daily total is 0)."""
    df = contributions_wide.select_dtypes(include=[np.number]).fillna(0.0)
    tot = df.sum(axis=1).replace(0, np.nan)
    return df.div(tot, axis=0)


# ---------------------------------------------------------------------------
# Trenton target dynamics
# ---------------------------------------------------------------------------


def trenton_target_metrics(
    flow: pd.Series,
    target: pd.Series,
    shortfall_epsilon: float = 0.0,
    surplus_fraction: float = 0.0,
) -> dict:
    """
    Shortfall / surplus relative to minimum-flow target.

    ``shortfall_epsilon``: treat as binding if ``flow < target - epsilon``.
    ``surplus_fraction``: flood-side flag if ``flow > target * (1 + surplus_fraction)``.
    """
    q = flow.astype(float).reindex(target.index)
    tgt = target.astype(float)
    shortfall = (tgt - q).clip(lower=0.0)
    binding = q < (tgt - shortfall_epsilon)
    surplus = q > (tgt * (1.0 + surplus_fraction))

    return {
        "shortfall": shortfall.rename("shortfall"),
        "binding_mask": binding.rename("binding"),
        "surplus_mask": surplus.rename("surplus"),
        "shortfall_spell_summary": spell_summary(shortfall > shortfall_epsilon),
        "binding_spell_summary": spell_summary(binding),
        "mean_shortfall_when_binding": float(shortfall[binding].mean()) if binding.any() else 0.0,
        "fraction_days_binding": float(binding.mean()),
        "fraction_days_surplus": float(surplus.mean()),
    }


def rolling_covariance(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Aligned rolling covariance (min_periods = max(3, window//4))."""
    a = x.astype(float).align(y.astype(float), join="inner")
    xa, ya = a[0], a[1]
    mp = max(3, min(window // 4, window))
    return xa.rolling(window, min_periods=mp).cov(ya)


# ---------------------------------------------------------------------------
# Trade-off: contribution vs depletion (per reservoir)
# ---------------------------------------------------------------------------


@dataclass
class TradeoffSummary:
    """Rows = reservoirs; columns ready for scatter / Pareto-style plots."""

    table: pd.DataFrame

    def to_scatter_frame(self) -> pd.DataFrame:
        return self.table.copy()


def contribution_vs_depletion_tradeoff(
    contributions_wide: pd.DataFrame,
    storage_pct_by_reservoir: Optional[pd.DataFrame] = None,
    combined_storage_pct: Optional[pd.Series] = None,
    depletion_threshold_pct: float = 40.0,
    contribution_mode: Literal["mean_flow", "integrated", "stress_weighted"] = "integrated",
    stress_weights: Optional[pd.Series] = None,
) -> TradeoffSummary:
    """
    Build a per-reservoir table: operational contribution vs drought stress exposure.

    If ``storage_pct_by_reservoir`` is provided, stress days use that column.
    Otherwise all reservoirs get the same ``combined_storage_pct`` stress mask
    (useful for NYC combined pool vs lower-basin contributions).
    """
    shares = contribution_shares(
        contributions_wide,
        mode=contribution_mode,
        stress_weights=stress_weights,
    )

    rows = []
    for col in shares.index:
        if storage_pct_by_reservoir is not None and col in storage_pct_by_reservoir.columns:
            s_pct = storage_pct_by_reservoir[col].astype(float)
        elif combined_storage_pct is not None:
            s_pct = combined_storage_pct.astype(float)
        else:
            raise ValueError(
                "Provide storage_pct_by_reservoir (with reservoir columns) or combined_storage_pct."
            )
        below = s_pct < depletion_threshold_pct
        rows.append(
            {
                "reservoir": col,
                "contribution_share": float(shares[col]),
                "stress_days_below_thresh": int(below.sum()),
                "fraction_time_stressed": float(below.mean()),
            }
        )
    return TradeoffSummary(table=pd.DataFrame(rows).set_index("reservoir"))


def aggregate_operational_burden_bundle(
    *,
    storage_pct: pd.Series,
    nor_low_pct: float,
    nor_high_pct: float,
    stress_threshold_pct: float,
    contributions_wide: Optional[pd.DataFrame] = None,
    trenton_flow: Optional[pd.Series] = None,
    trenton_target: Optional[pd.Series] = None,
    min_spell_duration: int = 1,
) -> dict:
    """
    One-call dict for dashboards: NOR, stress catalog, optional Trenton + shares.

    Safe to serialize keys to JSON if you drop DataFrames or use ``to_dict``.
    """
    out: dict = {}
    out["nor"] = nor_operational_burden_metrics(
        storage_pct,
        nor_low_pct,
        nor_high_pct,
        min_spell_duration=min_spell_duration,
    )
    out["drought_stress_catalog"] = stress_event_catalog(
        storage_pct,
        stress_threshold_pct,
        min_duration_days=max(1, min_spell_duration),
    )
    out["annual_stress"] = annual_stress_rates(
        out["drought_stress_catalog"],
        storage_pct.index,
    )
    if contributions_wide is not None:
        out["contribution_shares_mean_flow"] = contribution_shares(
            contributions_wide, mode="mean_flow"
        )
        out["contribution_shares_integrated"] = contribution_shares(
            contributions_wide, mode="integrated"
        )
    if trenton_flow is not None and trenton_target is not None:
        out["trenton"] = trenton_target_metrics(trenton_flow, trenton_target)
    return out
