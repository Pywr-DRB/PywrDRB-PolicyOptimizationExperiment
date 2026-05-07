"""
Data structures and loading for Stage 3 (full-Pareto ensemble) figures.

**Implemented**

- :func:`build_multipanel_daily_synthetic` / :func:`build_multipanel_monthly_synthetic` — development
  stand-ins matching the mockup envelopes (replace with HDF5-backed loaders when runs exist).

- :func:`load_full_pareto_manifest` — reads ``_full_pareto_manifest.json`` from
  :func:`methods.config.get_pywr_full_pareto_work_dir` (or a given directory).

**HDF5 (full Pareto)**

- :func:`aggregate_multipanel_daily_from_manifest` walks ``_full_pareto_manifest.json`` ``results``,
  loads each HDF5 via :func:`methods.postprocess.pywr_parametric_run.parametric_result_from_h5_path`
  (using ``cee_meta`` for parametric reservoir keys), and pools day-of-year / FDC quantiles by policy.

- :func:`aggregate_stage3_multipanels_from_manifest` performs one manifest walk and builds both the
  daily bundle and HDF5-backed monthly multipanel data (month-of-year releases, inflow–release bins,
  observed inflow Q20/Q80 for regime shading).

Trenton target handling:

- Reads ``CEE_TRENTON_TARGET_MGD`` when assembling stage 3 bundles.
- Falls back to ``DEFAULT_TRENTON_TARGET_MGD`` from ``.constants`` when unset.
- Called by ``methods.figures_stage3.stage3_analysis`` and stage 3 plotting wrappers.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np


def manifest_record_borg_variant(rec: Mapping[str, Any]) -> str:
    """
    Canonical Borg phase label for manifest rows: ``full`` | ``regression`` | ``perfect``.

    Empty or missing ``borg_variant`` is treated as ``full`` (legacy manifests).
    """
    v = str(rec.get("borg_variant", "")).strip().lower().replace("-", "_")
    return v if v else "full"


@dataclass
class Stage3DiagnosticContext:
    """Per-run arrays from manifest HDF5s for diagnostic figures."""

    target_mgd: float
    dates: Optional[Any]
    runs: List[Dict[str, Any]] = field(default_factory=list)
    obs_release_by_res: Dict[str, np.ndarray] = field(default_factory=dict)
    obs_inflow_by_res: Dict[str, np.ndarray] = field(default_factory=dict)

from methods.config import get_pywr_full_pareto_work_dir
from methods.postprocess.pywr_output_metadata import cee_meta_json_path

from .constants import (
    DEFAULT_TRENTON_TARGET_MGD,
    POLICY_ORDER,
    RESERVOIR_DISPLAY_NAMES,
    RESERVOIR_KEYS,
)


class EnvelopeBands(object):
    """Pointwise quantiles along ``x`` (e.g. day-of-year or exceedance %)."""

    __slots__ = ("x", "p50", "p25", "p75", "p10", "p90")

    def __init__(self, x, p50, p25, p75, p10, p90):
        self.x = x
        self.p50 = p50
        self.p25 = p25
        self.p75 = p75
        self.p10 = p10
        self.p90 = p90


class MultipanelDailyBundle(object):
    """Row-1 reservoir release envelopes + Trenton time/FDC + reliability samples (synthetic or aggregated)."""

    __slots__ = (
        "days",
        "reservoir_release",
        "reservoir_release_observed",
        "trenton",
        "trenton_doy_observed",
        "trenton_target_mgd",
        "fdc_exceedance_pct",
        "trenton_fdc",
        "trenton_fdc_observed",
        "reliability_by_policy",
        "reservoir_display_names",
    )

    def __init__(
        self,
        days,
        reservoir_release,
        trenton,
        trenton_target_mgd,
        fdc_exceedance_pct,
        trenton_fdc,
        trenton_fdc_observed,
        reliability_by_policy,
        reservoir_display_names=None,
        reservoir_release_observed=None,
        trenton_doy_observed=None,
    ):
        self.days = days
        self.reservoir_release = reservoir_release
        self.reservoir_release_observed = reservoir_release_observed or {}
        self.trenton = trenton
        self.trenton_doy_observed = trenton_doy_observed
        self.trenton_target_mgd = trenton_target_mgd
        self.fdc_exceedance_pct = fdc_exceedance_pct
        self.trenton_fdc = trenton_fdc
        self.trenton_fdc_observed = trenton_fdc_observed
        self.reliability_by_policy = reliability_by_policy
        self.reservoir_display_names = (
            dict(RESERVOIR_DISPLAY_NAMES) if reservoir_display_names is None else reservoir_display_names
        )


class InflowReleaseSynthetic(object):
    """Synthetic inflow grid + observed curve for IR panels (replace with real obs in production)."""

    __slots__ = (
        "inflow_mgd",
        "observed_release",
        "policy_bands",
        "regime_q20",
        "regime_q80",
        "observed_inflow_mgd",
    )

    def __init__(
        self,
        inflow_mgd,
        observed_release,
        policy_bands,
        regime_q20=None,
        regime_q80=None,
        observed_inflow_mgd=None,
    ):
        self.inflow_mgd = inflow_mgd
        self.observed_release = observed_release
        self.policy_bands = policy_bands
        self.regime_q20 = regime_q20
        self.regime_q80 = regime_q80
        self.observed_inflow_mgd = observed_inflow_mgd


class MultipanelMonthlyBundle(object):
    """Monthly row + inflow–release + Trenton + reliability (extended layout)."""

    __slots__ = (
        "months",
        "monthly_release",
        "monthly_release_observed",
        "inflow_release",
        "days",
        "trenton",
        "trenton_target_mgd",
        "fdc_exceedance_pct",
        "trenton_fdc",
        "trenton_fdc_observed",
        "reliability_by_policy",
        "reservoir_display_names",
    )

    def __init__(
        self,
        months,
        monthly_release,
        inflow_release,
        days,
        trenton,
        trenton_target_mgd,
        fdc_exceedance_pct,
        trenton_fdc,
        trenton_fdc_observed,
        reliability_by_policy,
        reservoir_display_names=None,
        monthly_release_observed=None,
    ):
        self.months = months
        self.monthly_release = monthly_release
        self.monthly_release_observed = monthly_release_observed or {}
        self.inflow_release = inflow_release
        self.days = days
        self.trenton = trenton
        self.trenton_target_mgd = trenton_target_mgd
        self.fdc_exceedance_pct = fdc_exceedance_pct
        self.trenton_fdc = trenton_fdc
        self.trenton_fdc_observed = trenton_fdc_observed
        self.reliability_by_policy = reliability_by_policy
        self.reservoir_display_names = (
            dict(RESERVOIR_DISPLAY_NAMES) if reservoir_display_names is None else reservoir_display_names
        )


def default_manifest_path(work_dir: Optional[str] = None) -> str:
    wd = work_dir or get_pywr_full_pareto_work_dir()
    return os.path.join(os.path.abspath(wd), "_full_pareto_manifest.json")


def load_full_pareto_manifest(path: Optional[str] = None) -> Dict[str, Any]:
    p = path or default_manifest_path()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def iter_manifest_hdf5_results(manifest: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield result records that contain an ``hdf5`` path (whether or not ``ok``)."""
    for rec in manifest.get("results", []) or []:
        if isinstance(rec, dict) and rec.get("hdf5"):
            yield rec


def list_ok_hdf5_paths(manifest: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for rec in iter_manifest_hdf5_results(manifest):
        if rec.get("ok") and rec.get("hdf5"):
            out.append(os.path.abspath(str(rec["hdf5"])))
    return out


def _load_cee_meta_payload(h5_path: str) -> Optional[Dict[str, Any]]:
    """Load CEE JSON sidecar or decode ``cee_meta`` HDF5 root attribute."""
    jp = cee_meta_json_path(h5_path)
    if os.path.isfile(jp):
        with open(jp, encoding="utf-8") as f:
            return json.load(f)
    try:
        import h5py

        with h5py.File(h5_path, "r") as f:
            raw = f.attrs.get("cee_meta")
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(str(raw))
    except Exception:
        return None


def _release_policy_dict_stub_from_cee_meta(payload: Mapping[str, Any], policy: str) -> Dict[str, Any]:
    """
    Rebuild a minimal ``release_policy_dict`` for :func:`parametric_result_from_h5_path`.

    Only reservoir keys and policy type are required for loading columns from existing HDF5.
    """
    res_list = list(payload.get("parametric_reservoirs") or [])
    pol = str(payload.get("policy_type") or policy).upper()
    if not res_list:
        raise ValueError("cee_meta payload missing parametric_reservoirs")
    return {
        res: {
            "class_type": "ParametricReservoirRelease",
            "policy_type": pol,
            "policy_id": "inline",
            "params": "",
        }
        for res in res_list
    }


def _envelope_bands_from_doy_dict(d: Mapping[str, np.ndarray]) -> EnvelopeBands:
    return EnvelopeBands(
        x=d["x"],
        p50=d["median"],
        p25=d["q25"],
        p75=d["q75"],
        p10=d["q10"],
        p90=d["q90"],
    )


def _envelope_bands_from_month_dict(d: Mapping[str, np.ndarray]) -> EnvelopeBands:
    return EnvelopeBands(
        x=d["x"],
        p50=d["median"],
        p25=d["q25"],
        p75=d["q75"],
        p10=d["q10"],
        p90=d["q90"],
    )


def _envelope_bands_from_ir_bin_dict(d: Mapping[str, np.ndarray]) -> EnvelopeBands:
    return EnvelopeBands(
        x=d["x"],
        p50=d["median"],
        p25=d["q25"],
        p75=d["q75"],
        p10=d["q10"],
        p90=d["q90"],
    )


def _envelope_bands_from_fdc_dict(d: Mapping[str, np.ndarray]) -> EnvelopeBands:
    return EnvelopeBands(
        x=d["x"],
        p50=d["median"],
        p25=d["q25"],
        p75=d["q75"],
        p10=d["q10"],
        p90=d["q90"],
    )


def _interp_fdc_onto_exceed(exceed: np.ndarray, obs_fdc: Mapping[str, np.ndarray]) -> np.ndarray:
    """Map observed FDC points onto the simulation FDC exceedance grid (for plotting)."""
    ox = np.asarray(obs_fdc["x"], dtype=float)
    oy = np.asarray(obs_fdc["y"], dtype=float)
    m = np.isfinite(ox) & np.isfinite(oy)
    if not np.any(m):
        return np.full_like(exceed, np.nan, dtype=float)
    ox = ox[m]
    oy = oy[m]
    order = np.argsort(ox)
    return np.interp(
        np.asarray(exceed, dtype=float),
        ox[order],
        oy[order],
        left=np.nan,
        right=np.nan,
    )


def _smooth_circular_doy(values: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Smooth a 365/366-length DOY climatology with circular rolling mean.

    This preserves seasonality while reducing day-to-day noise so the observed
    reference line reflects trend shape rather than jagged or near-flat medians.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 3:
        return v.copy()
    w = int(max(3, window))
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if n % 2 == 0 else n
        if w < 3:
            return v.copy()
    pad = w // 2
    ext = np.concatenate([v[-pad:], v, v[:pad]])
    with np.errstate(invalid="ignore"):
        out = np.array(
            [np.nanmean(ext[i : i + w]) for i in range(n)],
            dtype=float,
        )
    return out


def _nan_envelope_doy() -> EnvelopeBands:
    x = np.arange(1, 367, dtype=float)
    nan366 = np.full(366, np.nan, dtype=float)
    return EnvelopeBands(x=x, p50=nan366, p25=nan366, p75=nan366, p10=nan366, p90=nan366)


def _nan_envelope_fdc(exceed: np.ndarray) -> EnvelopeBands:
    x = np.asarray(exceed, dtype=float)
    nanf = np.full(len(x), np.nan, dtype=float)
    return EnvelopeBands(x=x, p50=nanf, p25=nanf, p75=nanf, p10=nanf, p90=nanf)


# --- Synthetic generators (mockup parity) ---------------------------------


def _release_envelope(
    days: np.ndarray,
    base: float,
    amp: float,
    *,
    phase: float = 0.0,
    trend: float = 0.0,
) -> Tuple[np.ndarray, ...]:
    med = base + amp * np.sin(2 * np.pi * (days + phase) / 365.0) + trend * np.linspace(0, 1, len(days))
    q25 = med - (12 + 4 * np.sin(2 * np.pi * (days + phase + 25) / 365.0))
    q75 = med + (12 + 4 * np.sin(2 * np.pi * (days + phase + 25) / 365.0))
    q10 = med - (28 + 6 * np.cos(2 * np.pi * (days + phase - 15) / 365.0))
    q90 = med + (28 + 6 * np.cos(2 * np.pi * (days + phase - 15) / 365.0))
    return med, q25, q75, q10, q90


def _envelope_from_tuple(x: np.ndarray, t: Tuple[np.ndarray, ...]) -> EnvelopeBands:
    med, q25, q75, q10, q90 = t
    return EnvelopeBands(x=x, p50=med, p25=q25, p75=q75, p10=q10, p90=q90)


def build_multipanel_daily_synthetic(seed: int = 11) -> MultipanelDailyBundle:
    rng = np.random.RandomState(seed)
    days = np.arange(1, 366)
    base_map = {
        "blueMarsh": 130.0,
        "beltzvilleCombined": 110.0,
        "fewalter": 95.0,
        "prompton": 70.0,
    }
    reservoir_release: Dict[str, Dict[str, EnvelopeBands]] = {}
    for i, r in enumerate(RESERVOIR_KEYS):
        reservoir_release[r] = {}
        for j, p in enumerate(POLICY_ORDER):
            t = _release_envelope(
                days,
                base_map[r],
                18 + i * 2 + j,
                phase=10 * i + 3 * j,
                trend=3.0 * (-1) ** j,
            )
            reservoir_release[r][p] = _envelope_from_tuple(days, t)

    def _trenton_envelope(base: float, seasonal: float, shock: float) -> Tuple[np.ndarray, ...]:
        med = base + seasonal * np.sin(2 * np.pi * (days - 25) / 365.0) - shock * np.exp(-((days - 285) / 50.0) ** 2)
        q25 = med - 110
        q75 = med + 110
        q10 = med - 250
        q90 = med + 250
        return med, q25, q75, q10, q90

    trenton_raw = {
        "STARFIT": _trenton_envelope(3150, 145, 320),
        "PWL": _trenton_envelope(3380, 170, 170),
        "RBF": _trenton_envelope(3075, 130, 255),
    }
    trenton = {p: _envelope_from_tuple(days, trenton_raw[p]) for p in POLICY_ORDER}

    exceed = np.linspace(0, 100, 200)

    def _fdc_envelope(level: float, spread: float, steep: float) -> Tuple[np.ndarray, ...]:
        med = level * np.exp(-steep * exceed / 100) + 35
        q25 = med * 0.93
        q75 = med * 1.07
        q10 = med * (1 - spread)
        q90 = med * (1 + spread)
        return med, q25, q75, q10, q90

    fdc_raw = {
        "STARFIT": _fdc_envelope(820, 0.16, 2.55),
        "PWL": _fdc_envelope(930, 0.13, 2.35),
        "RBF": _fdc_envelope(760, 0.18, 2.65),
    }
    trenton_fdc = {p: _envelope_from_tuple(exceed, fdc_raw[p]) for p in POLICY_ORDER}
    obs_fdc = 880 * np.exp(-2.45 * exceed / 100) + 45

    n = 90
    reliability_by_policy = {
        "STARFIT": np.clip(rng.normal(0.88, 0.045, n), 0.67, 0.99),
        "PWL": np.clip(rng.normal(0.94, 0.025, n), 0.78, 1.00),
        "RBF": np.clip(rng.normal(0.84, 0.060, n), 0.58, 0.99),
    }

    # Synthetic observed DOY curves (smooth) for multipanel legend testing
    reservoir_release_observed = {}
    for r in RESERVOIR_KEYS:
        reservoir_release_observed[r] = base_map[r] + 8.0 * np.sin(2 * np.pi * (days - 40) / 365.0)
    trenton_doy_obs = 3050 + 90 * np.sin(2 * np.pi * (days - 30) / 365.0)

    return MultipanelDailyBundle(
        days=days,
        reservoir_release=reservoir_release,
        trenton=trenton,
        trenton_target_mgd=DEFAULT_TRENTON_TARGET_MGD,
        fdc_exceedance_pct=exceed,
        trenton_fdc=trenton_fdc,
        trenton_fdc_observed=obs_fdc,
        reliability_by_policy=reliability_by_policy,
        reservoir_release_observed=reservoir_release_observed,
        trenton_doy_observed=trenton_doy_obs,
    )


def build_multipanel_monthly_synthetic(seed: int = 21) -> MultipanelMonthlyBundle:
    rng = np.random.RandomState(seed)
    months = np.arange(1, 13)
    days = np.arange(1, 366)
    exceed = np.linspace(0, 100, 200)
    inflow = np.logspace(1, 3, 200)

    def _monthly_env(base: float, amp: float, phase: float) -> Tuple[np.ndarray, ...]:
        med = base + amp * np.sin(2 * np.pi * (months + phase) / 12.0)
        q25 = med - 10
        q75 = med + 10
        q10 = med - 22
        q90 = med + 22
        return med, q25, q75, q10, q90

    base_map = {
        "blueMarsh": 135.0,
        "beltzvilleCombined": 115.0,
        "fewalter": 95.0,
        "prompton": 75.0,
    }
    monthly_release: Dict[str, Dict[str, EnvelopeBands]] = {}
    monthly_release_observed: Dict[str, np.ndarray] = {}
    for i, r in enumerate(RESERVOIR_KEYS):
        monthly_release[r] = {}
        for j, p in enumerate(POLICY_ORDER):
            raw = _monthly_env(base_map[r], 16 + i * 2 + j, 0.2 * i + 0.1 * j)
            monthly_release[r][p] = _envelope_from_tuple(months, raw)
        monthly_release_observed[r] = base_map[r] + 6.0 * np.sin(2 * np.pi * (months - 2) / 12.0)

    def _ir_env(scale: float, exponent: float) -> EnvelopeBands:
        med = scale * inflow**exponent
        q25 = med * 0.86
        q75 = med * 1.14
        q10 = med * 0.68
        q90 = med * 1.32
        return EnvelopeBands(x=inflow, p50=med, p25=q25, p75=q75, p10=q10, p90=q90)

    inflow_release: Dict[str, InflowReleaseSynthetic] = {}
    for i, r in enumerate(RESERVOIR_KEYS):
        policy_bands = {
            "STARFIT": _ir_env(0.90 + i * 0.08, 0.83),
            "PWL": _ir_env(1.15 + i * 0.08, 0.88),
            "RBF": _ir_env(0.72 + i * 0.08, 0.80),
        }
        obs_y = (1.0 + i * 0.08) * inflow**0.84
        inflow_release[r] = InflowReleaseSynthetic(
            inflow_mgd=inflow,
            observed_release=obs_y,
            policy_bands=policy_bands,
            regime_q20=None,
            regime_q80=None,
        )

    def _trenton_env(base: float, seasonal: float, shock: float) -> EnvelopeBands:
        med = base + seasonal * np.sin(2 * np.pi * (days - 20) / 365.0) - shock * np.exp(-((days - 285) / 55.0) ** 2)
        q25 = med - 110
        q75 = med + 110
        q10 = med - 250
        q90 = med + 250
        return _envelope_from_tuple(days, (med, q25, q75, q10, q90))

    trenton = {
        "STARFIT": _trenton_env(3150, 145, 300),
        "PWL": _trenton_env(3380, 170, 165),
        "RBF": _trenton_env(3070, 130, 245),
    }

    def _fdc_env(level: float, spread: float, steep: float) -> EnvelopeBands:
        med = level * np.exp(-steep * exceed / 100) + 35
        q25 = med * 0.93
        q75 = med * 1.07
        q10 = med * (1 - spread)
        q90 = med * (1 + spread)
        return _envelope_from_tuple(exceed, (med, q25, q75, q10, q90))

    trenton_fdc = {
        "STARFIT": _fdc_env(820, 0.16, 2.55),
        "PWL": _fdc_env(930, 0.13, 2.35),
        "RBF": _fdc_env(760, 0.18, 2.65),
    }
    obs_fdc = 880 * np.exp(-2.45 * exceed / 100) + 45

    n = 90
    reliability_by_policy = {
        "STARFIT": np.clip(rng.normal(0.88, 0.045, n), 0.67, 0.99),
        "PWL": np.clip(rng.normal(0.94, 0.025, n), 0.78, 1.00),
        "RBF": np.clip(rng.normal(0.84, 0.060, n), 0.58, 0.99),
    }

    return MultipanelMonthlyBundle(
        months=months,
        monthly_release=monthly_release,
        inflow_release=inflow_release,
        days=days,
        trenton=trenton,
        trenton_target_mgd=DEFAULT_TRENTON_TARGET_MGD,
        fdc_exceedance_pct=exceed,
        trenton_fdc=trenton_fdc,
        trenton_fdc_observed=obs_fdc,
        reliability_by_policy=reliability_by_policy,
        monthly_release_observed=monthly_release_observed,
    )


def aggregate_stage3_multipanels_from_manifest(
    manifest_path: Optional[str] = None,
    *,
    max_runs: Optional[int] = None,
    borg_variant: Optional[str] = None,
) -> Tuple[MultipanelDailyBundle, MultipanelMonthlyBundle]:
    """
    Single manifest walk: build :class:`MultipanelDailyBundle` and HDF5-backed
    :class:`MultipanelMonthlyBundle` (monthly release envelopes + inflow–release bands).

    **NWIS:** Uses ``fetch_prompton_nwis=False`` on parametric loads (same as daily-only path).
    """
    import pandas as pd

    from methods.plotting.plot_pareto_ensemble_uncertainty import (
        _inflow_series_on_dates,
        envelope_doy_from_matrix,
        envelope_fdc_from_matrix,
        envelope_ir_binned,
        envelope_month_from_matrix,
        observed_fdc_dict,
        observed_inflow_release_training,
    )
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

    reservoir_release: Dict[str, Dict[str, EnvelopeBands]] = {r: {} for r in RESERVOIR_KEYS}
    monthly_release: Dict[str, Dict[str, EnvelopeBands]] = {r: {} for r in RESERVOIR_KEYS}
    reservoir_ir_bands: Dict[str, Dict[str, EnvelopeBands]] = {r: {} for r in RESERVOIR_KEYS}
    trenton: Dict[str, EnvelopeBands] = {}
    trenton_fdc: Dict[str, EnvelopeBands] = {}
    reliability_by_policy: Dict[str, np.ndarray] = {}

    obs_fdc_raw: Optional[Dict[str, np.ndarray]] = None
    fdc_exceed_ref: Optional[np.ndarray] = None

    reference_idx: Optional[pd.DatetimeIndex] = None
    obs_trenton_series = None  # optional pd.Series of observed Trenton flow (aligned when possible)

    for p in POLICY_ORDER:
        rows = by_policy[p]
        if max_runs is not None:
            rows = rows[: int(max_runs)]
        if not rows:
            continue

        trenton_cols: List[np.ndarray] = []
        reliabilities: List[float] = []
        release_cols: Dict[str, List[np.ndarray]] = {r: [] for r in RESERVOIR_KEYS}
        dates_p: Optional[pd.DatetimeIndex] = None

        for rec in rows:
            h5 = os.path.abspath(str(rec["hdf5"]))
            if not os.path.isfile(h5):
                print(f"[stage3] skip missing HDF5: {h5}", flush=True)
                continue
            meta = _load_cee_meta_payload(h5)
            if not meta:
                print(f"[stage3] skip (no cee_meta for {h5})", flush=True)
                continue
            try:
                rdict = _release_policy_dict_stub_from_cee_meta(meta, p)
            except Exception as e:
                print(f"[stage3] skip cee_meta parse for {h5}: {e}", flush=True)
                continue
            try:
                multi_out = parametric_result_from_h5_path(
                    h5, rdict, scenario_id=0, fetch_prompton_nwis=False
                )
            except Exception as e:
                print(f"[stage3] skip load failure {h5}: {e}", flush=True)
                continue

            t = multi_out.get("trenton")
            if t is None or len(t) == 0:
                continue
            if dates_p is None:
                dates_p = pd.DatetimeIndex(t.index)
            if reference_idx is None:
                reference_idx = pd.DatetimeIndex(t.index)
            trenton_cols.append(t.to_numpy(dtype=float))
            if target_mgd is not None and np.isfinite(target_mgd):
                reliabilities.append(float(np.mean(t.to_numpy(dtype=float) >= target_mgd)))
            else:
                reliabilities.append(float("nan"))

            if obs_fdc_raw is None and multi_out.get("obs_trenton") is not None:
                obs_s = multi_out["obs_trenton"]
                if obs_s is not None and len(obs_s):
                    obs_fdc_raw = observed_fdc_dict(obs_s)
            if obs_trenton_series is None and multi_out.get("obs_trenton") is not None:
                obs_s = multi_out["obs_trenton"]
                if obs_s is not None and len(obs_s):
                    obs_trenton_series = pd.to_numeric(obs_s, errors="coerce").astype(float)

            for rname in RESERVOIR_KEYS:
                br = multi_out.get("by_res") or {}
                if rname not in br:
                    release_cols[rname].append(np.full(len(t), np.nan))
                    continue
                r0, _ = br[rname]
                release_cols[rname].append(r0.to_numpy(dtype=float))

        if not trenton_cols:
            continue

        Tm = max(len(c) for c in trenton_cols)
        trenton_mat = np.full((Tm, len(trenton_cols)), np.nan)
        for j, c in enumerate(trenton_cols):
            trenton_mat[: len(c), j] = c

        idx_t = (
            dates_p[:Tm]
            if dates_p is not None and len(dates_p) >= Tm
            else pd.date_range("1980-01-01", periods=Tm, freq="D")
        )
        t_doy = envelope_doy_from_matrix(idx_t, trenton_mat)
        trenton[p] = _envelope_bands_from_doy_dict(t_doy)

        fdc = envelope_fdc_from_matrix(trenton_mat)
        trenton_fdc[p] = _envelope_bands_from_fdc_dict(fdc)
        if fdc_exceed_ref is None:
            fdc_exceed_ref = np.asarray(fdc["x"], dtype=float)

        reliability_by_policy[p] = np.asarray(reliabilities, dtype=float)

        for rname in RESERVOIR_KEYS:
            cols = release_cols.get(rname, [])
            if not cols:
                continue
            Tr = max(len(c) for c in cols)
            M = np.full((Tr, len(cols)), np.nan)
            for j, c in enumerate(cols):
                M[: len(c), j] = c
            if not np.isfinite(M).any():
                continue
            idx_r = (
                dates_p[:Tr]
                if dates_p is not None and len(dates_p) >= Tr
                else pd.date_range("1980-01-01", periods=Tr, freq="D")
            )
            d = envelope_doy_from_matrix(idx_r, M)
            reservoir_release[rname][p] = _envelope_bands_from_doy_dict(d)
            d_mon = envelope_month_from_matrix(idx_r, M)
            monthly_release[rname][p] = _envelope_bands_from_month_dict(d_mon)
            try:
                inf1 = _inflow_series_on_dates(rname, idx_r)
                ir_d = envelope_ir_binned(inf1, M, n_bins=80)
                reservoir_ir_bands[rname][p] = _envelope_bands_from_ir_bin_dict(ir_d)
            except Exception as e:
                print(f"[stage3] IR bins skip {rname}/{p}: {e}", flush=True)

    if not trenton:
        raise ValueError(
            "No manifest HDF5 rows could be aggregated into daily envelopes. "
            "Check _full_pareto_manifest.json, cee_meta sidecars, and Pywr imports."
        )

    exceed_plot = fdc_exceed_ref
    if exceed_plot is None:
        for pol in POLICY_ORDER:
            if pol in trenton_fdc:
                exceed_plot = np.asarray(trenton_fdc[pol].x, dtype=float)
                break
    if exceed_plot is None:
        exceed_plot = np.linspace(0.0, 100.0, 200)

    if obs_fdc_raw is not None:
        trenton_fdc_observed = _interp_fdc_onto_exceed(exceed_plot, obs_fdc_raw)
    else:
        trenton_fdc_observed = np.full_like(exceed_plot, np.nan, dtype=float)

    reservoir_release_observed: Dict[str, np.ndarray] = {}
    monthly_release_observed: Dict[str, np.ndarray] = {}
    trenton_doy_observed: Optional[np.ndarray] = None
    if reference_idx is not None:
        from methods.config import PROCESSED_DATA_DIR
        from methods.load.observations import get_observational_training_data

        for rname in RESERVOIR_KEYS:
            try:
                _, rel_df, _ = get_observational_training_data(
                    reservoir_name=rname,
                    data_dir=PROCESSED_DATA_DIR,
                    as_numpy=False,
                    inflow_type="inflow_pub",
                )
                s = rel_df[rname].reindex(reference_idx)
                s = s.interpolate(limit_direction="both").bfill().ffill()
                mat = np.asarray(s.to_numpy(dtype=float), dtype=float).reshape(-1, 1)
                env = envelope_doy_from_matrix(reference_idx, mat)
                reservoir_release_observed[rname] = _smooth_circular_doy(
                    np.asarray(env["median"], dtype=float),
                    window=21,
                )
            except Exception as e:
                print(f"[stage3] observed release DOY {rname}: {e}", flush=True)
                reservoir_release_observed[rname] = np.full(366, np.nan, dtype=float)
        for rname in RESERVOIR_KEYS:
            try:
                _, rel_df, _ = get_observational_training_data(
                    reservoir_name=rname,
                    data_dir=PROCESSED_DATA_DIR,
                    as_numpy=False,
                    inflow_type="inflow_pub",
                )
                s = rel_df[rname].reindex(reference_idx)
                s = s.interpolate(limit_direction="both").bfill().ffill()
                arr = np.full(12, np.nan, dtype=float)
                for m in range(1, 13):
                    filter = reference_idx.month == m
                    if np.any(filter):
                        arr[m - 1] = float(np.nanmedian(np.asarray(s[filter].to_numpy(dtype=float), dtype=float)))
                monthly_release_observed[rname] = arr
            except Exception as e:
                print(f"[stage3] observed monthly release {rname}: {e}", flush=True)
                monthly_release_observed[rname] = np.full(12, np.nan, dtype=float)
        if obs_trenton_series is not None:
            try:
                aligned = obs_trenton_series.reindex(reference_idx)
                aligned = aligned.interpolate(limit_direction="both").bfill().ffill()
                mat = np.asarray(aligned.to_numpy(dtype=float), dtype=float).reshape(-1, 1)
                env = envelope_doy_from_matrix(reference_idx, mat)
                trenton_doy_observed = _smooth_circular_doy(
                    np.asarray(env["median"], dtype=float),
                    window=21,
                )
            except Exception as e:
                print(f"[stage3] observed Trenton DOY: {e}", flush=True)
    else:
        for rname in RESERVOIR_KEYS:
            reservoir_release_observed[rname] = np.full(366, np.nan, dtype=float)
            monthly_release_observed[rname] = np.full(12, np.nan, dtype=float)

    for pol in POLICY_ORDER:
        if pol not in trenton:
            trenton[pol] = _nan_envelope_doy()
        if pol not in trenton_fdc:
            trenton_fdc[pol] = _nan_envelope_fdc(exceed_plot)
        reliability_by_policy.setdefault(pol, np.array([np.nan], dtype=float))

    daily = MultipanelDailyBundle(
        days=np.arange(1, 366, dtype=float),
        reservoir_release=reservoir_release,
        trenton=trenton,
        trenton_target_mgd=target_mgd,
        fdc_exceedance_pct=exceed_plot,
        trenton_fdc=trenton_fdc,
        trenton_fdc_observed=trenton_fdc_observed,
        reliability_by_policy=reliability_by_policy,
        reservoir_release_observed=reservoir_release_observed,
        trenton_doy_observed=trenton_doy_observed,
    )

    inflow_release: Dict[str, InflowReleaseSynthetic] = {}
    months = np.arange(1, 13, dtype=float)
    for rname in RESERVOIR_KEYS:
        meta = observed_inflow_release_training(rname)
        obs_x = np.asarray(meta["obs_x"], dtype=float)
        obs_y = np.asarray(meta["obs_y"], dtype=float)
        pb = {pol: reservoir_ir_bands[rname][pol] for pol in POLICY_ORDER if pol in reservoir_ir_bands.get(rname, {})}
        if not pb:
            continue
        first_pol = next(pp for pp in POLICY_ORDER if pp in pb)
        x_ref = pb[first_pol].x
        inflow_release[rname] = InflowReleaseSynthetic(
            inflow_mgd=x_ref,
            observed_release=obs_y,
            policy_bands=pb,
            regime_q20=float(meta["q20"]) if np.isfinite(float(meta["q20"])) else None,
            regime_q80=float(meta["q80"]) if np.isfinite(float(meta["q80"])) else None,
            observed_inflow_mgd=obs_x,
        )

    for res in RESERVOIR_KEYS:
        for pol in POLICY_ORDER:
            monthly_release[res].setdefault(pol, _nan_envelope_month())

    monthly = MultipanelMonthlyBundle(
        months=months,
        monthly_release=monthly_release,
        inflow_release=inflow_release,
        days=np.arange(1, 366, dtype=float),
        trenton=trenton,
        trenton_target_mgd=target_mgd,
        fdc_exceedance_pct=exceed_plot,
        trenton_fdc=trenton_fdc,
        trenton_fdc_observed=trenton_fdc_observed,
        reliability_by_policy=reliability_by_policy,
        monthly_release_observed=monthly_release_observed,
    )

    return daily, monthly


def _nan_envelope_month() -> EnvelopeBands:
    m = np.arange(1, 13, dtype=float)
    nan12 = np.full(12, np.nan, dtype=float)
    return EnvelopeBands(x=m, p50=nan12, p25=nan12, p75=nan12, p10=nan12, p90=nan12)


def aggregate_multipanel_daily_from_manifest(
    manifest_path: Optional[str] = None,
    *,
    max_runs: Optional[int] = None,
    borg_variant: Optional[str] = None,
) -> MultipanelDailyBundle:
    """
    Aggregate HDF5 outputs listed in the full-Pareto manifest into :class:`MultipanelDailyBundle`.

    Walks manifest ``results`` with ``ok: true``, groups by ``policy``, opens each ``hdf5`` via
    :func:`methods.postprocess.pywr_parametric_run.parametric_result_from_h5_path`, and pools
    calendar days / FDC points across solutions (same strategy as Fig.~12
    :func:`methods.plotting.plot_pareto_ensemble_uncertainty.collect_pywr_ensemble_for_policy`).

    ``release_policy_dict`` is recovered from ``*_cee_meta.json`` or HDF5 ``cee_meta`` attributes
    (written by :func:`methods.postprocess.pywr_output_metadata.write_pywr_run_artifacts`).

    ``borg_variant`` optionally filters manifest rows (e.g. ``\"regression\"``). If ``None``,
    the environment variable ``CEE_STAGE3_BORG_VARIANT`` is used when set.

    ``max_runs`` caps the number of HDF5 rows accepted **per policy** (after sorting by
    ``alignment_index``) for faster iteration while developing figures.

    **NWIS:** This path calls :func:`~methods.postprocess.pywr_parametric_run.parametric_result_from_h5_path`
    with ``fetch_prompton_nwis=False`` so opening hundreds of HDF5 files does not repeat USGS
    Prompton DV downloads (Stage 3 envelopes use simulated Trenton / releases). To force NWIS
    when calling loaders yourself, pass ``fetch_prompton_nwis=True`` or unset ``CEE_SKIP_PROMPTON_NWIS``.
    """
    daily, _ = aggregate_stage3_multipanels_from_manifest(
        manifest_path, max_runs=max_runs, borg_variant=borg_variant
    )
    return daily


def aggregate_multipanel_monthly_from_manifest(
    manifest_path: Optional[str] = None,
    *,
    max_runs: Optional[int] = None,
    borg_variant: Optional[str] = None,
) -> MultipanelMonthlyBundle:
    """HDF5-backed monthly bundle (same manifest walk as :func:`aggregate_multipanel_daily_from_manifest`)."""
    _, monthly = aggregate_stage3_multipanels_from_manifest(
        manifest_path, max_runs=max_runs, borg_variant=borg_variant
    )
    return monthly
