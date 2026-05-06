#!/usr/bin/env python
"""
check_parametric_initial_volume.py

Quick sanity check for per-reservoir initial_volume_frac:

- Runs Pywr-DRB Parametric once for PROMPTON + RBF
- Prints the first few storage values for:
    (a) pywr_start = 1983-10-01
    (b) pywr_start = 2019-01-01

You should see:
- Early-1980s storage start near 0.10 * capacity (override)
- 2019 run starting much closer to the observed / spin-up state
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1) Bootstrap paths (same style as 04c_make_9panel_with_pywr_and_default.py)
# ---------------------------------------------------------------------
def _find_up(name: str, start: Path | None = None) -> Path:
    p = Path.cwd() if start is None else Path(start).resolve()
    for q in [p, *p.parents]:
        cand = q / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find '{name}' above {p}")

PO_REPO       = _find_up("CEE6400Project")
PYWR_DRB_REPO = _find_up("Release_Policy_DRB")
PYWR_SRC      = PYWR_DRB_REPO / "src"

for p in [str(PYWR_SRC), str(PO_REPO)]:
    if p in sys.path:
        sys.path.remove(p)
for p in [str(PYWR_SRC), str(PO_REPO)]:
    sys.path.insert(0, p)

def _ensure_dir(p: Path | str) -> Path:
    p = p if isinstance(p, Path) else Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------------------------------------------------------------
# 2) Imports from your project
# ---------------------------------------------------------------------
import pywrdrb  # noqa: E402
from methods.config import (
    NFE, SEED, ISLANDS,
    OUTPUT_DIR as CFG_OUTPUT_DIR,
)
from methods.load.results import load_results
from methods.config import FIG_DIR as CFG_FIG_DIR
outdir = (Path(CFG_FIG_DIR) / "fig4_validation_9panel")
def run_pywr_parametric_once(
    RES: str, POL: str, theta: np.ndarray,
    pywr_start: str, pywr_end: str,
    inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    work_dir: Path | str = (Path(CFG_FIG_DIR) / "_pywr_parametric")
) -> tuple[pd.Series, pd.Series]:
    work_dir   = _ensure_dir(work_dir)
    tmp_models = _ensure_dir(PYWR_DRB_REPO / "_tmp_models")
    params_str = ",".join(str(float(x)) for x in np.asarray(theta, float).ravel().tolist())
    tag        = f"Parametric_{POL}_{RES}"

    options = {
        "release_policy_dict": {
            RES: {
                "class_type": "ParametricReservoirRelease",
                "policy_type": POL,
                "policy_id":   "inline",
                "params":      params_str,
            }
        }
    }

    model_json = tmp_models / f"model_{tag}.json"
    h5         = Path(work_dir) / f"output_{tag}.hdf5"

    mb = pywrdrb.ModelBuilder(start_date=pywr_start, end_date=pywr_end,
                              inflow_type=inflow_type, options=options)
    mb.make_model()
    mb.write_model(str(model_json))

    model = pywrdrb.Model.load(str(model_json))
    rec   = pywrdrb.OutputRecorder(model, str(h5))
    _     = model.run()

    dataP = pywrdrb.Data(print_status=False,
                         results_sets=["res_storage", "reservoir_downstream_gage"],
                         output_filenames=[str(h5)])
    dataP.load_output()
    key = h5.stem

    R_full = dataP.reservoir_downstream_gage[key][0][RES].astype(float).rename("pywr_parametric_release")
    S_full = dataP.res_storage[key][0][RES].astype(float).rename("pywr_parametric_storage")
    return R_full, S_full


# ---------------------------------------------------------------------
# 3) Choose reservoir / policy and load one theta
# ---------------------------------------------------------------------
RES = "prompton"
POL = "RBF"   # or "STARFIT" etc.

csv = Path(CFG_OUTPUT_DIR) / f"MMBorg_{ISLANDS}M_{POL}_{RES}_nfe{NFE}_seed{SEED}.csv"
print(f"[info] Loading optimization results from: {csv}")

obj_df, var_df = load_results(str(csv))
if obj_df is None or obj_df.empty or var_df is None or var_df.empty:
    raise RuntimeError("No optimization results found for this reservoir/policy.")

# just grab the first candidate’s parameters
theta = var_df.iloc[0].to_numpy(dtype=float)
print(f"[info] Using theta of length {len(theta)} from first solution")

# ---------------------------------------------------------------------
# 4) Run parametric once with long spin-up (1983 start) and print head
# ---------------------------------------------------------------------
print("\n=== Pywr Parametric run (start=1983-10-01) ===")
R_full_83, S_full_83 = run_pywr_parametric_once(
    RES, POL, theta,
    pywr_start="1983-10-01",
    pywr_end="2023-12-31",
    inflow_type="pub_nhmv10_BC_withObsScaled",
)
print("Storage head (1983 start):")
print(S_full_83.head(10))  # should be near 0.10 * capacity for PROMPTON

# ---------------------------------------------------------------------
# 5) Run parametric again with validation-aligned start (2019) and print head
# ---------------------------------------------------------------------
print("\n=== Pywr Parametric run (start=2019-01-01) ===")
R_full_19, S_full_19 = run_pywr_parametric_once(
    RES, POL, theta,
    pywr_start="2019-01-01",
    pywr_end="2023-12-31",
    inflow_type="pub_nhmv10_BC_withObsScaled",
)
print("Storage head (2019 start):")
print(S_full_19.head(10))

print("\nDone. Compare these two heads to see the effect of initial_volume_frac + spin-up.")
