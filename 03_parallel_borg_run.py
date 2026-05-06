"""
Runs MMBorgMOEA optimization.

Credit to Chung-Yi Lin & Sai Veena Sunkara for the original code,
which has been modified for use with the reservoir problem.

For more info, see:
https://github.com/philip928lin/BorgTraining/tree/main (Private)

Optional MRF masking: objectives can be evaluated only on "normal operations"
days for DRB calibration reservoirs (F.E. Walter, Prompton, Beltzville, Blue Marsh)
when extra CLI args are provided (see below). Eligibility matches ``reservoir_options``
in ``methods.config`` / pywrdrb.

Slurm: ``USE_MRF=true|false`` in ``run_parallel_mmborg.sh``. Env ``CEE_USE_MRF`` overrides argv when set.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from pathnavigator import PathNavigator

from methods.reservoir.model import Reservoir
from methods.load.observations import get_observational_training_data
from methods.metrics.objectives import ObjectiveCalculator
from methods.config import SEED, RELEASE_METRICS, STORAGE_METRICS, EPSILONS, NFE, ISLANDS
from methods.config import DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MRF_NORMAL_OPS_BUFFER_DAYS
from methods.config import policy_n_params, policy_param_bounds, get_starfit_param_bounds
from methods.config import reservoir_capacity, INERTIA_BY_RESERVOIR, release_max_by_reservoir
from methods.config import reservoir_options

from methods.preprocessing.mrf_masking import (
    load_mrf_ranges,
    build_normal_ops_mask,
    validate_mask_alignment,
)
from methods.borg_paths import mrf_filtered_file_suffix

# Project root: do not rely on process CWD under Slurm/MPI (often job spool on some ranks).
_CEE_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MRF_JSON = "preprocessing_outputs/masking/pub_reconstruction/lower_basin_mrf_active_ranges.json"
_LEGACY_MRF_JSON = "preprocessing_outputs/masking/lower_basin_mrf_active_ranges.json"
root_dir = _CEE_ROOT
pn = PathNavigator(root_dir, max_depth=2)
pn.chdir()

os.makedirs(DATA_DIR, exist_ok=True)


def _mpi_rank() -> int:
    for k in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MPI_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


def _log(msg: str) -> None:
    if _mpi_rank() == 0:
        print(msg, flush=True)


def _resolve_mrf_ranges_path(path_str: str) -> Path:
    """Resolve MRF JSON; fall back to legacy flat masking path if the bundle path is missing."""
    p = Path(path_str)
    cand = p if p.is_absolute() else Path(_CEE_ROOT) / p
    if cand.exists():
        return cand
    leg = Path(_CEE_ROOT) / _LEGACY_MRF_JSON
    if leg.exists():
        _log(f"[MRF Masking] Primary JSON missing ({cand}); using legacy path: {leg}")
        return leg
    return cand


### CLI: POLICY_TYPE RESERVOIR_NAME [seed] [mrf_json] [use_mrf]
#  - seed: optional; default SEED from config
#  - mrf_json: optional path to MRF ranges JSON (default _DEFAULT_MRF_JSON under project root)
#  - use_mrf: "true"/"false" — if true, apply masking for reservoirs in reservoir_options
assert len(sys.argv) > 2, "POLICY_TYPE and RESERVOIR_NAME must be provided by command line."

POLICY_TYPE = str(sys.argv[1])
RESERVOIR_NAME = str(sys.argv[2])

borg_seed = int(sys.argv[3]) if len(sys.argv) > 3 and str(sys.argv[3]).strip() != "" else SEED
MRF_RANGES_JSON = (
    sys.argv[4] if len(sys.argv) > 4 and str(sys.argv[4]).strip() != ""
    else _DEFAULT_MRF_JSON
)
cee_mrf = os.environ.get("CEE_USE_MRF", "").strip().lower()
if cee_mrf in ("0", "false", "no", "off"):
    USE_MRF_MASKING = False
elif cee_mrf in ("1", "true", "yes", "on"):
    USE_MRF_MASKING = True
else:
    USE_MRF_MASKING = (
        str(sys.argv[5]).strip().lower() in ("1", "true", "yes", "on")
        if len(sys.argv) > 5
        else False
    )

##### Settings ####################################################################

NVARS = policy_n_params[POLICY_TYPE]
if POLICY_TYPE == "STARFIT":
    BOUNDS = get_starfit_param_bounds(RESERVOIR_NAME)
else:
    BOUNDS = policy_param_bounds[POLICY_TYPE]

METRICS = RELEASE_METRICS + STORAGE_METRICS
NOBJS = len(METRICS)

NCONSTRS = 1 if POLICY_TYPE == "STARFIT" else 0

runtime_freq = 250
islands = ISLANDS

### Load observed data #######################################

inflow_obs, release_obs, storage_obs = get_observational_training_data(
    reservoir_name=RESERVOIR_NAME,
    data_dir=PROCESSED_DATA_DIR,
    as_numpy=False,
    inflow_type="inflow_pub",
)

datetime = inflow_obs.index

if len(datetime) < 365:
    print(f"Warning: Only {len(datetime)} days of data available for reservoir '{RESERVOIR_NAME}'.")

inflow_obs = inflow_obs.values.flatten().astype(np.float64)
release_obs = release_obs.values.flatten().astype(np.float64)
storage_obs = storage_obs.values.flatten().astype(np.float64)

initial_storage_obs = storage_obs[0]
R_MAX = release_max_by_reservoir[RESERVOIR_NAME]
iset = INERTIA_BY_RESERVOIR[RESERVOIR_NAME]

normal_ops_mask = None
mrf_ranges_dict = None

if USE_MRF_MASKING:
    # fewalter, prompton, beltzvilleCombined, blueMarsh — same as methods.config / pywrdrb
    if RESERVOIR_NAME in reservoir_options:
        _log(f"[MRF Masking] Requested for {RESERVOIR_NAME}")

        mrf_ranges_path = _resolve_mrf_ranges_path(MRF_RANGES_JSON)

        if not mrf_ranges_path.exists():
            _log(f"[WARNING] MRF ranges JSON not found: {mrf_ranges_path}")
            _log("[WARNING] Continuing without MRF masking.")
            USE_MRF_MASKING = False
        else:
            _log(f"[MRF Masking] Loading MRF ranges from: {mrf_ranges_path}")
            mrf_ranges_dict = load_mrf_ranges(str(mrf_ranges_path))

            mode = "RES" if RESERVOIR_NAME in mrf_ranges_dict else "ANY"
            if mode == "ANY":
                _log("[MRF Masking] Using ANY_lower_basin ranges (reservoir-specific not found)")

            normal_ops_mask = build_normal_ops_mask(
                datetime_index=datetime,
                mrf_ranges_dict=mrf_ranges_dict,
                reservoir_name=RESERVOIR_NAME,
                mode=mode,
                buffer_days=MRF_NORMAL_OPS_BUFFER_DAYS,
            )

            is_valid, msg = validate_mask_alignment(
                datetime_index=datetime,
                normal_ops_mask=normal_ops_mask,
                min_normal_days=365,
            )

            if not is_valid:
                _log(f"[ERROR] MRF mask validation failed: {msg}")
                _log("[ERROR] Disabling MRF masking.")
                USE_MRF_MASKING = False
                normal_ops_mask = None
            else:
                n_normal = int(np.sum(normal_ops_mask))
                n_total = len(normal_ops_mask)
                pct_normal = 100.0 * n_normal / n_total
                _log(f"[MRF Masking] {msg}")
                _log(
                    f"[MRF Masking] Buffer: {MRF_NORMAL_OPS_BUFFER_DAYS} days "
                    "before/after MRF-active (excluded from normal ops)"
                )
                _log(
                    f"[MRF Masking] Will evaluate objectives on {n_normal}/{n_total} days "
                    f"({pct_normal:.1f}%)"
                )
    else:
        _log(
            f"[MRF Masking] Ignored for {RESERVOIR_NAME} "
            f"(not in reservoir_options: {list(reservoir_options)})."
        )
        USE_MRF_MASKING = False
else:
    _log("[MRF Masking] Off (Slurm: USE_MRF=false; argv: omit or pass false; env: CEE_USE_MRF=0).")

release_obj_func = ObjectiveCalculator(
    metrics=RELEASE_METRICS,
    inertia_tau=iset["release"]["tau"],
    inertia_scale_release=iset["release"]["scale"],
    inertia_release_scale_value=(R_MAX if iset["release"]["scale"] == "value" else None),
)

storage_obj_func = ObjectiveCalculator(
    metrics=STORAGE_METRICS,
    capacity_mg=reservoir_capacity[RESERVOIR_NAME],
    inertia_tau=iset["storage"]["tau"],
    inertia_scale_storage=iset["storage"]["scale"],
    inertia_storage_scale_value=iset["storage"]["scale_value"],
)


def evaluate(*vars):
    """One Borg evaluation: simulate full series; objectives on full or masked days."""

    reservoir = Reservoir(
        inflow=inflow_obs,
        dates=datetime,
        capacity=reservoir_capacity[RESERVOIR_NAME],
        policy_type=POLICY_TYPE,
        policy_params=list(vars),
        initial_storage=initial_storage_obs,
        name=RESERVOIR_NAME,
    )
    reservoir.policy.debug = False

    if POLICY_TYPE == "STARFIT":
        valid = reservoir.policy.test_nor_constraint()
        if not valid:
            with open("violated_params_borg.log", "a") as f:
                f.write(f"[FAIL] {RESERVOIR_NAME} @ {str(pd.Timestamp.now())}:\n")
                f.write(f"{list(vars)}\n\n")
            return [9999.99] * NOBJS, [1.0]

    reservoir.reset()
    reservoir.run()

    if (reservoir.storage_array < -1e-9).any():
        print("[WARN] Negative storage encountered.")
    avail = np.r_[initial_storage_obs, reservoir.storage_array[:-1]] + inflow_obs
    if (reservoir.release_array - avail > 1e-6).any():
        print("[WARN] Release > available water at some steps.")

    sim_release = reservoir.release_array.astype(np.float64)
    sim_release += reservoir.spill_array.astype(np.float64)
    sim_storage = reservoir.storage_array.astype(np.float64)

    if np.isnan(sim_release).any() or np.isnan(sim_storage).any():
        print(f"Simulation generated NaN for {RESERVOIR_NAME}, {POLICY_TYPE} with parameters {vars}.")

    if USE_MRF_MASKING and normal_ops_mask is not None:
        obs_release_masked = release_obs[normal_ops_mask]
        obs_storage_masked = storage_obs[normal_ops_mask]
        sim_release_masked = sim_release[normal_ops_mask]
        sim_storage_masked = sim_storage[normal_ops_mask]

        if len(obs_release_masked) < 365:
            print(f"[WARN] Insufficient normal-ops days after masking: {len(obs_release_masked)}")
            if NCONSTRS > 0:
                return [9999.99] * NOBJS, [1.0]
            return [9999.99] * NOBJS,

        release_objs = release_obj_func.calculate(obs=obs_release_masked, sim=sim_release_masked)
        storage_objs = storage_obj_func.calculate(obs=obs_storage_masked, sim=sim_storage_masked)
    else:
        release_objs = release_obj_func.calculate(obs=release_obs, sim=sim_release)
        storage_objs = storage_obj_func.calculate(obs=storage_obs, sim=sim_storage)

    objectives = list(release_objs) + list(storage_objs)

    if NCONSTRS > 0:
        return objectives, [0.0]
    return objectives,


borg_settings = {
    "numberOfVariables": NVARS,
    "numberOfObjectives": NOBJS,
    "numberOfConstraints": NCONSTRS,
    "function": evaluate,
    "epsilons": EPSILONS,
    "bounds": BOUNDS,
    "directions": None,
    "seed": borg_seed,
}

if __name__ == "__main__":

    from borg import *
    Configuration.startMPI()

    borg = Borg(**borg_settings)

    pn.mkdir("outputs")
    pn.outputs.mkdir("checkpoints")

    if islands == 1:
        fname_base = pn.outputs.get() / f"MWBorg_{POLICY_TYPE}_{RESERVOIR_NAME}_nfe{NFE}_seed{borg_seed}"
    else:
        fname_base = (
            pn.outputs.get()
            / f"MMBorg_{islands}M_{POLICY_TYPE}_{RESERVOIR_NAME}_nfe{NFE}_seed{borg_seed}"
        )

    if USE_MRF_MASKING and normal_ops_mask is not None:
        fname_base = pn.outputs.get() / f"{fname_base.name}{mrf_filtered_file_suffix()}"

    if islands == 1:
        runtime_filename = f"{fname_base}.runtime"
    else:
        runtime_filename = f"{fname_base}_%d.runtime"

    solvempi_settings = {
        "islands": islands,
        "maxTime": None,
        "maxEvaluations": NFE,
        "initialization": None,
        "runtime": runtime_filename,
        "allEvaluations": None,
        "frequency": runtime_freq,
    }

    result = borg.solveMPI(**solvempi_settings)

    if result is not None:
        with open(f"{fname_base}.csv", "w") as file:
            file.write(
                ",".join(
                    [f"var{i+1}" for i in range(NVARS)]
                    + [f"obj{i+1}" for i in range(NOBJS)]
                    + [f"constr{i+1}" for i in range(NCONSTRS)]
                )
                + "\n"
            )
            result.display(out=file, separator=",")

        with open(f"{fname_base}.set", "w") as file:
            file.write("# Version=5\n")
            file.write(f"# NumberOfVariables={NVARS}\n")
            file.write(f"# NumberOfObjectives={NOBJS}\n")
            file.write(f"# NumberOfConstraints={NCONSTRS}\n")
            for i, bound in enumerate(borg_settings["bounds"]):
                file.write(f"# Variable.{i+1}.Definition=RealVariable({bound[0]},{bound[1]})\n")
            if borg_settings.get("directions") is None:
                for i in range(NOBJS):
                    file.write(f"# Objective.{i+1}.Definition=Minimize\n")
            else:
                for i, direction in enumerate(borg_settings["directions"]):
                    if direction == "min":
                        file.write(f"# Objective.{i+1}.Definition=Minimize\n")
                    elif direction == "max":
                        file.write(f"# Objective.{i+1}.Definition=Maximize\n")
            file.write(f"//NFE={NFE}\n")
            result.display(out=file, separator=" ")
            file.write("#\n")

        with open(f"{fname_base}.info", "w") as file:
            file.write("\nBorg settings\n")
            file.write("=================\n")
            for key, value in borg_settings.items():
                file.write(f"{key}: {value}\n")
            file.write("\nBorg solveMPI settings\n")
            file.write("=================\n")
            for key, value in solvempi_settings.items():
                file.write(f"{key}: {value}\n")
            file.write("\nMRF objective filtering\n")
            file.write("=================\n")
            file.write(f"enabled: {bool(USE_MRF_MASKING and normal_ops_mask is not None)}\n")
            file.write(f"mrf_json: {MRF_RANGES_JSON}\n")
            if USE_MRF_MASKING and normal_ops_mask is not None:
                file.write(f"mrf_filename_suffix: {mrf_filtered_file_suffix()}\n")
            if normal_ops_mask is not None:
                n_normal = int(np.sum(normal_ops_mask))
                n_total = len(normal_ops_mask)
                file.write(f"normal_ops_days: {n_normal}/{n_total} ({100.0 * n_normal / n_total:.1f}%)\n")

        if islands == 1:
            print(f"Master: Completed {fname_base}")
        elif islands > 1:
            print(f"Multi-master controller: Completed {fname_base}")

    Configuration.stopMPI()
