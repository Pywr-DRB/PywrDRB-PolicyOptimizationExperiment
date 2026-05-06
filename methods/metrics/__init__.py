"""Metrics for optimization objectives and operational diagnostics."""

from .operational_burden import (
    TradeoffSummary,
    aggregate_operational_burden_bundle,
    annual_stress_rates,
    contribution_daily_fractions,
    contribution_shares,
    contribution_vs_depletion_tradeoff,
    find_spells,
    nor_mask,
    nor_operational_burden_metrics,
    recovery_times_after_spells,
    rolling_covariance,
    spell_summary,
    stress_event_catalog,
    trenton_target_metrics,
)
