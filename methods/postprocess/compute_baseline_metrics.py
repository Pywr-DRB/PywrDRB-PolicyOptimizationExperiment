#!/usr/bin/env python3
# methods/postprocess/compute_baseline_metrics.py
"""
Compute Pywr-DRB default-run baseline objective values per reservoir, using the
same ObjectiveCalculator and metric sets as optimization.

Inputs
------
- Default baseline run from cached HDF5 produced by build_default_timeseries.py:
    {DRB_DEFAULT_CACHE}/output_default_{tag}.hdf5
- Observations via get_observational_training_data(reservoir_name=..., data_dir=PROCESSED_DATA_DIR)
- Config-driven objective definitions (RELEASE_METRICS, STORAGE_METRICS), dates (VAL_START/VAL_END),
  reservoir capacities, inertia settings, etc.

Outputs
-------
- Baseline objective metrics computed in-memory from HDF5 + observations.
- Optional CSV export (only when ``--write-csv`` is passed):
    {FIG_DIR}/{BASELINE_DIR_NAME}_{BASELINE_INFLOW_TAG}/baseline_objectives_*.csv

Notes
-----
- If a reservoir lacks observed releases, release metrics are skipped (left out).
- If a reservoir is missing simulated release/storage in the baseline HDF5, this script fails fast.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]   # .../CEE6400Project
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import os

# ---------------- project config & utilities ----------------
from methods.config import (
    FIG_DIR, PROCESSED_DATA_DIR,
    reservoir_options as RESERVOIR_NAMES,
    BASELINE_DIR_NAME, BASELINE_INFLOW_TAG,
    VAL_START, VAL_END,
    RELEASE_METRICS, STORAGE_METRICS,
    reservoir_capacity, INERTIA_BY_RESERVOIR, release_max_by_reservoir,
)

from methods.metrics.objectives import ObjectiveCalculator
from methods.load.observations import get_observational_training_data
from methods.postprocess.pywr_parametric_run import parametric_result_from_h5_path

# --------------- helpers ----------------
_DEFAULT_CACHE: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}


def _resolve_default_hdf5_path(
    hdf5_path: str | None,
    cache_dir: str | None,
    tag: str,
) -> Path:
    if hdf5_path:
        p = Path(hdf5_path).expanduser().resolve()
    else:
        out_dir = Path(os.environ.get("DRB_OUTPUT_DIR", str(ROOT / "pywr_data")))
        cache = Path(cache_dir) if cache_dir else Path(
            os.environ.get("DRB_DEFAULT_CACHE", str(out_dir / "_pywr_default_cache"))
        )
        p = cache.expanduser().resolve() / f"output_default_{tag}.hdf5"
    if not p.is_file():
        raise FileNotFoundError(f"Default Pywr HDF5 not found: {p}")
    return p


def _load_default_release_storage(
    reservoir: str,
    *,
    hdf5_path: str | None,
    cache_dir: str | None,
    tag: str,
) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    p = _resolve_default_hdf5_path(hdf5_path=hdf5_path, cache_dir=cache_dir, tag=tag)
    key = (str(p), reservoir)
    if key not in _DEFAULT_CACHE:
        result = parametric_result_from_h5_path(
            str(p), {reservoir: "STARFIT"}, scenario_id=0, fetch_prompton_nwis=False
        )
        _DEFAULT_CACHE[key] = result["by_res"][reservoir]
    rel, sto = _DEFAULT_CACHE[key]
    return rel.rename("default_release"), sto.rename("default_storage")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _align(*series: pd.Series, start: str, end: str) -> List[pd.Series]:
    """Intersect indices across series, then slice [start, end]. Drops NAs."""
    idx = None
    for s in series:
        if s is None:
            continue
        idx = s.index if idx is None else idx.intersection(s.index)
    if idx is None or len(idx) == 0:
        return [pd.Series(dtype=float) for _ in series]
    idx = idx[(idx >= pd.to_datetime(start)) & (idx <= pd.to_datetime(end))]
    out = []
    for s in series:
        if s is None:
            out.append(pd.Series(dtype=float))
        else:
            out.append(s.loc[idx].astype(float).dropna())
    # re-intersect after dropna to enforce equal length
    idx2 = None
    for s in out:
        if len(s) == 0:
            continue
        idx2 = s.index if idx2 is None else idx2.intersection(s.index)
    if idx2 is None or len(idx2) == 0:
        return [pd.Series(dtype=float) for _ in series]
    return [s.loc[idx2] if len(s) else s for s in out]

def _calc_release_metrics(res: str, obs: pd.Series, sim: pd.Series) -> Optional[List[float]]:
    if obs is None or sim is None or len(obs) == 0 or len(sim) == 0:
        return None
    R_MAX = release_max_by_reservoir[res]
    iset  = INERTIA_BY_RESERVOIR[res]["release"]
    oc = ObjectiveCalculator(
        metrics=RELEASE_METRICS,
        inertia_tau=iset["tau"],
        inertia_scale_release=iset["scale"],
        inertia_release_scale_value=(R_MAX if iset["scale"] == "value" else None),
    )
    vals = oc.calculate(obs=obs.values.astype(np.float64),
                        sim=sim.values.astype(np.float64))
    return vals

def _calc_storage_metrics(res: str, obs: pd.Series, sim: pd.Series) -> Optional[List[float]]:
    if obs is None or sim is None or len(obs) == 0 or len(sim) == 0:
        return None
    iset  = INERTIA_BY_RESERVOIR[res]["storage"]
    oc = ObjectiveCalculator(
        metrics=STORAGE_METRICS,
        capacity_mg=reservoir_capacity[res],
        inertia_tau=iset["tau"],
        inertia_scale_storage=iset["scale"],
        inertia_storage_scale_value=iset["scale_value"],
    )
    vals = oc.calculate(obs=obs.values.astype(np.float64),
                        sim=sim.values.astype(np.float64))
    return vals


def compute_baseline_objectives_for_reservoir(
    res: str,
    *,
    start: str,
    end: str,
    default_hdf5: str | None = None,
    default_cache_dir: str | None = None,
    default_tag: str = "1983-10-01_2023-12-31_pub_nhmv10_BC_withObsScaled",
) -> pd.DataFrame:
    """Compute baseline objective rows (metric, pywr_baseline) for one reservoir."""
    rel_def, sto_def = _load_default_release_storage(
        reservoir=res,
        hdf5_path=default_hdf5,
        cache_dir=default_cache_dir,
        tag=default_tag,
    )
    if rel_def is None:
        raise RuntimeError(
            f"Default HDF5 is missing release series for reservoir '{res}'. "
            "This indicates an incomplete recorded output in the simulation pipeline."
        )
    if sto_def is None:
        raise RuntimeError(
            f"Default HDF5 is missing storage series for reservoir '{res}'. "
            "This indicates an incomplete recorded output in the simulation pipeline."
        )

    inflow_df, release_df, storage_df = get_observational_training_data(
        reservoir_name=res, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
    )
    rel_obs = None if release_df is None or release_df.empty else release_df.squeeze().astype(float).rename("obs_release")
    sto_obs = None if storage_df is None or storage_df.empty else storage_df.squeeze().astype(float).rename("obs_storage")

    rel_obs_al, sto_obs_al, rel_def_al, sto_def_al = _align(
        rel_obs, sto_obs, rel_def, sto_def, start=start, end=end
    )

    rows = []
    r_vals = _calc_release_metrics(res, rel_obs_al, rel_def_al)
    if r_vals is not None:
        for name, val in zip(RELEASE_METRICS, r_vals):
            rows.append((name, float(val)))

    s_vals = _calc_storage_metrics(res, sto_obs_al, sto_def_al)
    if s_vals is not None:
        for name, val in zip(STORAGE_METRICS, s_vals):
            rows.append((name, float(val)))

    if not rows:
        return pd.DataFrame(columns=["metric", "pywr_baseline"])
    return pd.DataFrame(rows, columns=["metric", "pywr_baseline"])

# --------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Compute per-reservoir Pywr baseline objectives from default HDF5 + observations.")
    ap.add_argument("--default-hdf5", default=None, help="Path to default HDF5 (defaults to DRB_DEFAULT_CACHE + default tag)")
    ap.add_argument("--default-cache-dir", default=None, help="Override DRB_DEFAULT_CACHE when --default-hdf5 is not provided")
    ap.add_argument("--default-tag", default="1983-10-01_2023-12-31_pub_nhmv10_BC_withObsScaled", help="Tag used in output_default_<tag>.hdf5")
    ap.add_argument("--outdir", default=None, help="Output dir for baseline CSVs (defaults to FIG_DIR/{BASELINE_DIR_NAME}_{BASELINE_INFLOW_TAG})")
    ap.add_argument("--start", default=str(VAL_START), help="Validation window start (YYYY-MM-DD)")
    ap.add_argument("--end",   default=str(VAL_END),   help="Validation window end (YYYY-MM-DD)")
    ap.add_argument("--reservoirs", nargs="*", default=list(RESERVOIR_NAMES), help="Subset of reservoirs to compute")
    ap.add_argument("--write-csv", action="store_true", help="Write baseline_objectives_*.csv files (default: disabled)")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve() if args.outdir else \
             _ensure_dir(Path(FIG_DIR) / f"{BASELINE_DIR_NAME}_{BASELINE_INFLOW_TAG}")
    _ensure_dir(outdir)

    print("[INFO] Source       : default Pywr HDF5 (no default CSVs)")
    print(f"[INFO] Output dir   : {outdir}")
    print(f"[INFO] Window       : {args.start} to {args.end}")

    for res in args.reservoirs:
        print(f"\n=== Reservoir: {res} ===")

        out = compute_baseline_objectives_for_reservoir(
            res,
            start=args.start,
            end=args.end,
            default_hdf5=args.default_hdf5,
            default_cache_dir=args.default_cache_dir,
            default_tag=args.default_tag,
        )
        if out.empty:
            print(f"[SKIP] No comparable metrics could be computed for {res} (missing data).")
            continue

        if args.write_csv:
            out_path = outdir / f"baseline_objectives_{res}_{args.start}_to_{args.end}.csv"
            out.to_csv(out_path, index=False)
            print(f"[OK] Wrote {out_path} with {len(out)} metrics")
        else:
            print(f"[OK] Computed {len(out)} metrics for {res} (CSV disabled)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(2)
