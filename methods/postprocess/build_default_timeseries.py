#!/usr/bin/env python3
"""Build or reuse the default Pywr-DRB baseline run cache.

This script creates the canonical default-operation HDF5 used by downstream
baseline metrics and plotting scripts. It is designed to be safe in workflow
pipelines: if the cache already exists, callers can reuse it unless they
explicitly request a rebuild.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import os

# ---------------------- Paths & config ----------------------------------------
HERE = Path(__file__).parent.resolve()
# Put outputs at project root (two levels above methods/postprocess/)
ROOT = HERE.parents[1]
OUTPUTS_DIR   = Path(os.environ.get("DRB_OUTPUT_DIR", (ROOT / "pywr_data")))
CACHE_DIR = Path(os.environ.get("DRB_DEFAULT_CACHE", (OUTPUTS_DIR / "_pywr_default_cache")))

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

# ------------------------------ pywrdrb ---------------------------------------
try:
    import pywrdrb  # top-level package (your editable install)
    from pywrdrb import Model, ModelBuilder, OutputRecorder
except Exception as e:
    print("[ERROR] pywrdrb failed to import or initialize.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    err = str(e).lower()
    if "only_folders" in err or "set_all_to_sc" in err:
        print(
            "Hint: pywrdrb.path_manager requires pathnavigator>=0.6 "
            "(Folder.set_all_to_sc only_folders=…). Upgrade with:\n"
            '  pip install -U "pathnavigator>=0.6"\n'
            "Then reinstall pywrdrb if needed: pip install -e /path/to/Pywr-DRB",
            file=sys.stderr,
        )
    else:
        print("Is your venv activated and pywrdrb installed (e.g. pip install -e Pywr-DRB)?", file=sys.stderr)
    sys.exit(1)

for sym in ("Model", "ModelBuilder", "OutputRecorder", "Data"):
    if not hasattr(pywrdrb, sym):
        print(
            f"[WARN] pywrdrb.{sym} not found. If your fork exposes it elsewhere, "
            f"adjust imports (e.g., from pywrdrb.postprocessing.data import Data).",
            file=sys.stderr,
        )

# ----------------------- Default run build & load ------------------------------
def save_default_pywr_run(
    start_date: str = "1983-10-01",
    end_date: str   = "2023-12-31",
    inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    outdir: Path = CACHE_DIR,
    overwrite: bool = False,
) -> Path:
    """
    Build & run the default Pywr-DRB model once and cache HDF5.
    Returns the HDF5 path.
    """
    ensure_dir(outdir)
    tag = f"{start_date}_{end_date}_{inflow_type}"
    h5_path = outdir / f"output_default_{tag}.hdf5"
    model_json = outdir / f"model_default_{tag}.json"

    if h5_path.exists() and not overwrite:
        print(f"[default] Reusing cached HDF5: {h5_path}")
        return h5_path

    print(f"[default] Building model: {start_date} → {end_date} | inflow={inflow_type}")
    mb = ModelBuilder(start_date=start_date, end_date=end_date, inflow_type=inflow_type)
    mb.make_model()
    mb.write_model(str(model_json))

    print(f"[default] Running model → {h5_path.name}")
    m = Model.load(str(model_json))
    rec = OutputRecorder(m, str(h5_path))
    _ = m.run()

    if not h5_path.exists():
        raise RuntimeError("Default run finished but HDF5 not found. Check pywrdrb setup.")
    print(f"[default] Saved HDF5: {h5_path}")
    return h5_path


def _supports_release_for_all_model_reservoirs(h5_path: Path) -> None:
    """
    Fail fast when simulated release output is missing for a modeled reservoir.

    Contract:
    - `res_storage` and `res_release` are simulated outputs and should cover all modeled reservoirs.
    - `reservoir_downstream_gage` may be a subset and is not used for this completeness check.
    """
    rel_set = set()
    sto_set = set()
    dataD = pywrdrb.Data(
        print_status=False,
        results_sets=["res_storage", "res_release"],
        output_filenames=[str(h5_path)],
    )
    dataD.load_output()
    key = h5_path.stem
    if key in dataD.res_storage:
        sto_set = {c for c in dataD.res_storage[key][0].columns if isinstance(c, str)}
    if key in dataD.res_release:
        rel_set = {c for c in dataD.res_release[key][0].columns if isinstance(c, str)}

    missing_release = sorted(sto_set - rel_set)
    if missing_release:
        raise RuntimeError(
            "Default HDF5 is missing simulated res_release series for: "
            f"{missing_release}. Refusing to continue with partial baseline simulation output."
        )

# ----------------------------------- CLI --------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run default Pywr-DRB and build/cache baseline HDF5. No observed data needed."
    )
    ap.add_argument("--start", default="1983-10-01", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end",   default="2023-12-31", help="End date (YYYY-MM-DD)")
    ap.add_argument("--inflow-type", default="pub_nhmv10_BC_withObsScaled", help="Pywr-DRB inflow_type")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR), help="Cache dir for model JSON / HDF5")
    # When CEE_FORCE_DEFAULT_RERUN=1 (e.g. in run_postprocessing_and_figures.sh), always re-run so default uses latest inflows
    _default_overwrite = os.environ.get("CEE_FORCE_DEFAULT_RERUN", "").strip().lower() in ("1", "true", "yes")
    ap.add_argument("--overwrite", action="store_true", default=_default_overwrite, help="Force re-run of default model (default: True if CEE_FORCE_DEFAULT_RERUN=1)")
    ap.add_argument("--no-overwrite", action="store_false", dest="overwrite", help="Use cached HDF5 if present")
    ap.add_argument("--strict-release-completeness", action="store_true", default=True, help="Fail if HDF5 has storage reservoirs without matching release series")
    ap.add_argument("--no-strict-release-completeness", action="store_false", dest="strict_release_completeness", help="Allow partial reservoir_downstream_gage outputs")
    args = ap.parse_args()

    cache     = Path(args.cache_dir).resolve()
    ensure_dir(cache)

    # 1) Run (or reuse) default model and get HDF5
    h5 = save_default_pywr_run(
        start_date=args.start, end_date=args.end,
        inflow_type=args.inflow_type, outdir=cache,
        overwrite=args.overwrite,
    )

    if args.strict_release_completeness:
        _supports_release_for_all_model_reservoirs(h5)

    print(f"[default] HDF5 ready: {h5}")

if __name__ == "__main__":
    main()
