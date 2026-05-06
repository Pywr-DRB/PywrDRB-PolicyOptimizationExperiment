#!/usr/bin/env python3
"""
Plot observed vs Pywr default release/storage for the same window used in
compute_baseline_metrics.py (defaults: VAL_START/VAL_END, baseline folder).

Writes PNGs next to baseline_objectives_*.csv:
  baseline_dynamics_{res}_{start}_to_{end}.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.config import (
    FIG_DIR,
    PROCESSED_DATA_DIR,
    BASELINE_DIR_NAME,
    BASELINE_INFLOW_TAG,
    VAL_START,
    VAL_END,
    reservoir_options,
)
from methods.load.observations import get_observational_training_data
from methods.postprocess.pywr_parametric_run import parametric_result_from_h5_path

_DEFAULT_CACHE: dict[tuple[str, str], tuple] = {}


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
):
    p = _resolve_default_hdf5_path(hdf5_path=hdf5_path, cache_dir=cache_dir, tag=tag)
    key = (str(p), reservoir)
    if key not in _DEFAULT_CACHE:
        result = parametric_result_from_h5_path(
            str(p), {reservoir: "STARFIT"}, scenario_id=0, fetch_prompton_nwis=False
        )
        _DEFAULT_CACHE[key] = result["by_res"][reservoir]
    rel, sto = _DEFAULT_CACHE[key]
    return rel.rename("default_release"), sto.rename("default_storage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--default-hdf5", default=None, help="Path to default HDF5 (defaults to DRB_DEFAULT_CACHE + default tag)")
    ap.add_argument("--default-cache-dir", default=None, help="Override DRB_DEFAULT_CACHE when --default-hdf5 is not provided")
    ap.add_argument("--default-tag", default="1983-10-01_2023-12-31_pub_nhmv10_BC_withObsScaled", help="Tag used in output_default_<tag>.hdf5")
    ap.add_argument("--start", default=str(VAL_START))
    ap.add_argument("--end", default=str(VAL_END))
    ap.add_argument(
        "--baseline-outdir",
        default=None,
        help="Folder with baseline_objectives_*.csv (default FIG_DIR/{BASELINE_DIR_NAME}_{BASELINE_INFLOW_TAG})",
    )
    ap.add_argument("--reservoirs", nargs="*", default=list(reservoir_options))
    args = ap.parse_args()

    outdir = (
        Path(args.baseline_outdir).resolve()
        if args.baseline_outdir
        else Path(FIG_DIR) / f"{BASELINE_DIR_NAME}_{BASELINE_INFLOW_TAG}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    sl = slice(args.start, args.end)

    for res in args.reservoirs:
        rel_def, sto_def = _load_default_release_storage(
            reservoir=res,
            hdf5_path=args.default_hdf5,
            cache_dir=args.default_cache_dir,
            tag=args.default_tag,
        )
        if rel_def is None:
            raise RuntimeError(
                f"Default HDF5 is missing release series for reservoir '{res}'. "
                "Fix simulation outputs before plotting baseline dynamics."
            )
        if sto_def is None:
            raise RuntimeError(
                f"Default HDF5 is missing storage series for reservoir '{res}'. "
                "Fix simulation outputs before plotting baseline dynamics."
            )
        _, rel_obs_df, sto_obs_df = get_observational_training_data(
            reservoir_name=res, data_dir=PROCESSED_DATA_DIR, as_numpy=False, inflow_type="inflow_pub"
        )
        rel_obs = rel_obs_df.squeeze().astype(float) if rel_obs_df is not None else None
        sto_obs = sto_obs_df.squeeze().astype(float) if sto_obs_df is not None else None

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        if sto_obs is not None and sto_def is not None:
            o = sto_obs.loc[sl]
            s = sto_def.loc[sl]
            idx = o.index.intersection(s.index)
            axes[0].plot(idx, o.loc[idx], color="k", lw=1.2, label="Observed storage")
            axes[0].plot(idx, s.loc[idx], color="tab:orange", lw=1.0, ls="--", label="Pywr default storage")
            axes[0].set_ylabel("Storage (MG)")
            axes[0].legend(loc="upper right")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "Storage series missing", ha="center", transform=axes[0].transAxes)

        if rel_obs is not None and rel_def is not None:
            o = rel_obs.loc[sl]
            s = rel_def.loc[sl]
            idx = o.index.intersection(s.index)
            axes[1].plot(idx, o.loc[idx], color="k", lw=1.2, label="Observed release")
            axes[1].plot(idx, s.loc[idx], color="tab:blue", lw=1.0, ls="--", label="Pywr default release")
            axes[1].set_ylabel("Release (MGD)")
            axes[1].set_xlabel("Date")
            axes[1].legend(loc="upper right")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Release series missing", ha="center", transform=axes[1].transAxes)

        fig.suptitle(
            f"{res} — baseline dynamics (metrics window {args.start} to {args.end})\n"
            "Same series used for compute_baseline_metrics.py",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        png = outdir / f"baseline_dynamics_{res}_{args.start}_to_{args.end}.png"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] {png}")


if __name__ == "__main__":
    main()
