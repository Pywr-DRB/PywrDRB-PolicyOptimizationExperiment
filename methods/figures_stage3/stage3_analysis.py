"""
Unified Stage 3 full-Pareto analysis: multipanel figures + diagnostic PNGs in one pipeline.

Called from ``methods/figures_stage3/plot_stage3_full_pareto_figures.py`` via :func:`run_stage3_full_pareto_analysis`.

Trenton target handling:

- Reads ``CEE_TRENTON_TARGET_MGD`` for reliability/threshold calculations.
- Falls back to ``DEFAULT_TRENTON_TARGET_MGD`` from ``.constants`` if unset.
- This module is one of the target-load points used by stage 3 figure scripts.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from methods.config import PROCESSED_DATA_DIR
from methods.load.observations import get_observational_training_data

from .constants import DEFAULT_TRENTON_TARGET_MGD, POLICY_ORDER, RESERVOIR_KEYS, STAGE3_BORG_VARIANT_SUBDIRS
from .full_pareto_output_paths import full_pareto_png_path
from .data_loading import (
    Stage3DiagnosticContext,
    _load_cee_meta_payload,
    _release_policy_dict_stub_from_cee_meta,
    aggregate_stage3_multipanels_from_manifest,
    build_multipanel_daily_synthetic,
    build_multipanel_monthly_synthetic,
    load_full_pareto_manifest,
    manifest_record_borg_variant,
)
from .multipanel_daily import plot_multipanel_daily_uncertainty
from .multipanel_monthly import plot_multipanel_monthly_uncertainty


def _obs_release_series_aligned(reservoir: str, dates: pd.DatetimeIndex) -> np.ndarray:
    _, rel_df, _ = get_observational_training_data(
        reservoir_name=reservoir,
        data_dir=PROCESSED_DATA_DIR,
        as_numpy=False,
        inflow_type="inflow_pub",
    )
    s = rel_df[reservoir].reindex(dates)
    s = s.interpolate(limit_direction="both").bfill().ffill()
    return s.to_numpy(dtype=float)


def _obs_inflow_series_aligned(reservoir: str, dates: pd.DatetimeIndex) -> np.ndarray:
    inf_df, _, _ = get_observational_training_data(
        reservoir_name=reservoir,
        data_dir=PROCESSED_DATA_DIR,
        as_numpy=False,
        inflow_type="inflow_pub",
    )
    s = inf_df[reservoir].reindex(dates)
    s = s.interpolate(limit_direction="both").bfill().ffill()
    return s.to_numpy(dtype=float)


def build_diagnostic_context(
    manifest_path: str,
    *,
    max_runs: Optional[int] = None,
    borg_variant: Optional[str] = None,
) -> Stage3DiagnosticContext:
    """
    Single pass over manifest ``ok`` rows: collect Trenton, releases, storage, optional MRF frame
    per simulation for diagnostic figures.
    """
    from methods.postprocess.pywr_parametric_run import parametric_result_from_h5_path

    manifest = load_full_pareto_manifest(manifest_path)
    bv_env = (os.environ.get("CEE_STAGE3_BORG_VARIANT", "") or "").strip()
    bv_filter = (borg_variant or bv_env or None) and str(borg_variant or bv_env).strip().lower()

    raw_entries: List[Dict[str, Any]] = []
    for rec in manifest.get("results", []) or []:
        if not isinstance(rec, dict) or not rec.get("ok"):
            continue
        h5 = rec.get("hdf5")
        pol = rec.get("policy")
        if not h5 or not pol:
            continue
        if bv_filter and manifest_record_borg_variant(rec) != bv_filter:
            continue
        raw_entries.append(rec)

    by_policy: Dict[str, List[Dict[str, Any]]] = {p: [] for p in POLICY_ORDER}
    for rec in raw_entries:
        p = str(rec.get("policy", "")).upper()
        if p in by_policy:
            by_policy[p].append(rec)
    for p in POLICY_ORDER:
        by_policy[p].sort(key=lambda r: (int(r.get("alignment_index") or 0), str(r.get("borg_variant", ""))))

    tgt_raw = os.environ.get("CEE_TRENTON_TARGET_MGD", "").strip()
    target_mgd = float(tgt_raw) if tgt_raw else float(DEFAULT_TRENTON_TARGET_MGD)

    runs: List[Dict[str, Any]] = []
    dates_ref: Optional[pd.DatetimeIndex] = None
    obs_release_by_res: Dict[str, np.ndarray] = {}
    obs_inflow_by_res: Dict[str, np.ndarray] = {}

    for pol in POLICY_ORDER:
        rows = by_policy[pol]
        if max_runs is not None:
            rows = rows[: int(max_runs)]
        for rec in rows:
            h5 = os.path.abspath(str(rec["hdf5"]))
            if not os.path.isfile(h5):
                continue
            meta = _load_cee_meta_payload(h5)
            if not meta:
                continue
            try:
                rdict = _release_policy_dict_stub_from_cee_meta(meta, pol)
            except Exception:
                continue
            try:
                multi_out = parametric_result_from_h5_path(
                    h5, rdict, scenario_id=0, fetch_prompton_nwis=False
                )
            except Exception:
                continue
            tser = multi_out.get("trenton")
            if tser is None or len(tser) == 0:
                continue
            if dates_ref is None:
                dates_ref = pd.DatetimeIndex(tser.index)
            tr = tser.to_numpy(dtype=float)
            rel: Dict[str, np.ndarray] = {}
            sto: Dict[str, np.ndarray] = {}
            br = multi_out.get("by_res") or {}
            for rname in RESERVOIR_KEYS:
                if rname not in br:
                    rel[rname] = np.full(len(tr), np.nan)
                    sto[rname] = np.full(len(tr), np.nan)
                    continue
                r0, s0 = br[rname]
                rel[rname] = r0.to_numpy(dtype=float)
                sto[rname] = s0.to_numpy(dtype=float)
            mrf = multi_out.get("mrf")
            rel_val = float(np.mean(tr >= target_mgd)) if np.isfinite(target_mgd) else float("nan")
            idx = int(rec.get("alignment_index") or 0)
            runs.append(
                {
                    "policy": pol,
                    "borg_variant": str(rec.get("borg_variant", "")),
                    "alignment_index": idx,
                    "hdf5": h5,
                    "trenton": tr,
                    "releases": rel,
                    "storage": sto,
                    "reliability": rel_val,
                    "mrf": mrf.copy() if mrf is not None else None,
                }
            )

    if dates_ref is not None:
        for r in RESERVOIR_KEYS:
            try:
                obs_release_by_res[r] = _obs_release_series_aligned(r, dates_ref)
                obs_inflow_by_res[r] = _obs_inflow_series_aligned(r, dates_ref)
            except Exception:
                obs_release_by_res[r] = np.full(len(dates_ref), np.nan)
                obs_inflow_by_res[r] = np.full(len(dates_ref), np.nan)

    return Stage3DiagnosticContext(
        target_mgd=target_mgd,
        dates=dates_ref,
        runs=runs,
        obs_release_by_res=obs_release_by_res,
        obs_inflow_by_res=obs_inflow_by_res,
    )


def _run_stage3_once(
    *,
    manifest: Optional[str],
    out_dir: str,
    mock: bool,
    borg_variant: Optional[str],
    max_runs: Optional[int],
    which: str,
    skip_monthly: bool,
    skip_diagnostics: bool,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    outputs: List[str] = []

    w = which.strip().lower()
    want_daily = w in ("daily", "multipanels", "all")
    want_monthly = w in ("monthly", "multipanels", "all") and not skip_monthly
    want_diag = w in ("diagnostics", "all") and not skip_diagnostics

    need_multipanel = mock or not manifest or want_daily or want_monthly
    daily_data = None
    monthly_data = None
    if need_multipanel:
        if mock or not manifest:
            daily_data = build_multipanel_daily_synthetic()
            monthly_data = build_multipanel_monthly_synthetic()
        else:
            daily_data, monthly_data = aggregate_stage3_multipanels_from_manifest(
                manifest, max_runs=max_runs, borg_variant=borg_variant
            )

    if want_daily and daily_data is not None:
        p = full_pareto_png_path(out_dir, "multipanel_daily")
        outputs.append(plot_multipanel_daily_uncertainty(daily_data, save_path=p))
    if want_monthly and monthly_data is not None:
        p = full_pareto_png_path(out_dir, "multipanel_monthly")
        outputs.append(plot_multipanel_monthly_uncertainty(monthly_data, save_path=p))

    if want_diag and manifest and not mock:
        ctx = build_diagnostic_context(manifest, max_runs=max_runs, borg_variant=borg_variant)
        from .advanced_plots import run_all_diagnostic_figures

        outputs.extend(run_all_diagnostic_figures(ctx, out_dir))
    elif want_diag and mock:
        print("[full-pareto figs] diagnostics require --manifest (skipping diagnostic PNGs)", flush=True)

    return outputs


def run_stage3_full_pareto_analysis(
    *,
    manifest: Optional[str],
    out_dir: str,
    mock: bool,
    borg_variant: Optional[str],
    max_runs: Optional[int],
    which: str,
    skip_monthly: bool = False,
    skip_diagnostics: bool = False,
) -> List[str]:
    """
    One pipeline: multipanel daily/monthly (unless skipped) + diagnostic PNGs.

    ``which``: ``daily`` | ``monthly`` | ``multipanels`` (both) | ``diagnostics`` | ``all``.

    When ``manifest`` is set (not mock), by default writes **separate** figure sets under
    ``out_dir/borg_full_series/``, ``out_dir/borg_mrffiltered_regression/``, and
    ``out_dir/borg_mrffiltered_perfect_foresight/`` (see :data:`STAGE3_BORG_VARIANT_SUBDIRS`).
    Disable with ``CEE_STAGE3_SPLIT_VARIANTS=0`` or ``--no-split-borg-variants``.
    """
    explicit_bv = (borg_variant or os.environ.get("CEE_STAGE3_BORG_VARIANT", "")).strip().lower()
    split_env = os.environ.get("CEE_STAGE3_SPLIT_VARIANTS", "1").strip().lower()
    want_split = split_env not in ("0", "false", "no", "off")

    if want_split and manifest and not mock and not explicit_bv:
        merged: List[str] = []
        for v in ("full", "regression", "perfect"):
            sub = os.path.join(out_dir, STAGE3_BORG_VARIANT_SUBDIRS[v])
            try:
                merged.extend(
                    _run_stage3_once(
                        manifest=manifest,
                        out_dir=sub,
                        mock=mock,
                        borg_variant=v,
                        max_runs=max_runs,
                        which=which,
                        skip_monthly=skip_monthly,
                        skip_diagnostics=skip_diagnostics,
                    )
                )
            except ValueError as e:
                print(f"[full-pareto figs] skip variant {v}: {e}", flush=True)
        return merged

    bv_pass = explicit_bv if explicit_bv else None
    return _run_stage3_once(
        manifest=manifest,
        out_dir=out_dir,
        mock=mock,
        borg_variant=bv_pass,
        max_runs=max_runs,
        which=which,
        skip_monthly=skip_monthly,
        skip_diagnostics=skip_diagnostics,
    )
