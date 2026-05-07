"""
Optimization and project paths for CEE6400.

Canonical MOEA/objective/reservoir/policy settings come from
``pywrdrb.release_policies.config`` (Pywr-DRB). This module re-exports that
namespace and overrides only filesystem paths so data and outputs live under
the CEE6400 project tree, plus MRF filtering settings used by this repo.
"""
import os

from pywrdrb.release_policies import config as _pywr_rp_config
from pywrdrb.release_policies.config import *  # noqa: F403

# Parent of ``methods/`` = CEE6400 project root when imported from this tree
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Borg CSV ``*_seed{N}_mrffiltered_*`` uses the MOEA ``SEED`` from pywrdrb; unfiltered
# ``*_seed{N}.csv`` uses ``BORG_SEED_FULL`` in ``pywrdrb.release_policies.config`` (fallback 72).
BORG_SEED_MRF = SEED
BORG_SEED_FULL = int(getattr(_pywr_rp_config, "BORG_SEED_FULL", 72))

DATA_DIR = os.path.join(_PROJECT_ROOT, "obs_data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PUB_RECON_DIR = os.path.join(DATA_DIR, "pub_reconstruction")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")
FIG_DIR = os.path.join(_PROJECT_ROOT, "figures")

# Pywr-DRB model artifacts (JSON + HDF5 from ModelBuilder/OutputRecorder — not figure outputs)
PYWR_DATA_DIR = os.path.join(_PROJECT_ROOT, "pywr_data")
PYWR_TMP_RUNS_DIR = os.path.join(PYWR_DATA_DIR, "pywr_tmp_runs")
# Full-Pareto MPI sweep (``methods/ensemble/run_full_pareto_pywr_mpi.py``) — separate from figure pipelines.
PYWR_FULL_PARETO_RUNS_DIR = os.path.join(PYWR_DATA_DIR, "full_pareto_runs")
PYWR_PICK_HDF5_DIR = os.path.join(PYWR_DATA_DIR, "pywr_pick_hdf5")


def get_pywr_work_dir() -> str:
    """Working directory for Pywr model JSON + HDF5 used by figures / validation (override: ``CEE_PYWR_WORK_DIR``)."""
    v = os.environ.get("CEE_PYWR_WORK_DIR", "").strip()
    return os.path.abspath(v) if v else PYWR_TMP_RUNS_DIR


def get_pywr_full_pareto_work_dir() -> str:
    """Working directory for full-Pareto MPI Pywr outputs (override: ``CEE_FULL_PARETO_WORK_DIR``)."""
    v = os.environ.get("CEE_FULL_PARETO_WORK_DIR", "").strip()
    return os.path.abspath(v) if v else PYWR_FULL_PARETO_RUNS_DIR


def get_pywr_pick_hdf5_dir() -> str:
    """Legacy path helper (override: ``CEE_PYWR_PICK_HDF5_DIR``). Parametric pick HDF5 now uses :func:`get_pywr_work_dir`."""
    v = os.environ.get("CEE_PYWR_PICK_HDF5_DIR", "").strip()
    return os.path.abspath(v) if v else PYWR_PICK_HDF5_DIR


# MRF filtering assets (see methods.preprocessing.build_mrf_active_filters and build_mrf_filtering_folder.sh)
MRF_FILTERING_ROOT = os.path.join(_PROJECT_ROOT, "preprocessing_outputs", "filtering")
MRF_FILTER_JSON_PUB_RECON = os.path.join(
    MRF_FILTERING_ROOT, "pub_reconstruction", "lower_basin_mrf_active_ranges.json"
)
MRF_FILTER_JSON_PERFECT_INFO = os.path.join(
    MRF_FILTERING_ROOT, "perfect_information", "lower_basin_mrf_active_ranges.json"
)

# Buffer (days) around MRF “normal ops” windows for filtering (not in pywrdrb config)
MRF_NORMAL_OPS_BUFFER_DAYS = 5

# Normal operating ranges (NOR) by reservoir, lifted from
# ``BASE_POLICY_CONTEXT_BY_RESERVOIR`` for quick discoverability in this repo.
# Values are expressed as storage fractions of capacity and absolute storage MG.
NORMAL_OPERATING_RANGE_BY_RESERVOIR = {}
for _name in reservoir_options:
    _ctx = BASE_POLICY_CONTEXT_BY_RESERVOIR[_name]
    _cap = float(reservoir_capacity[_name])
    _nor_min_frac = float(
        _ctx.get("nor_min_frac", _ctx.get("nor_lo_frac", _ctx.get("normal_min_frac", 0.0)))
    )
    _nor_max_frac = float(
        _ctx.get("nor_max_frac", _ctx.get("nor_hi_frac", _ctx.get("normal_max_frac", 1.0)))
    )
    NORMAL_OPERATING_RANGE_BY_RESERVOIR[_name] = {
        "nor_min_frac": _nor_min_frac,
        "nor_max_frac": _nor_max_frac,
        "nor_min_storage_mg": _nor_min_frac * _cap,
        "nor_max_storage_mg": _nor_max_frac * _cap,
    }


def get_normal_operating_range(reservoir_name: str) -> dict:
    """Return NOR bounds for one reservoir from :data:`NORMAL_OPERATING_RANGE_BY_RESERVOIR`."""
    return NORMAL_OPERATING_RANGE_BY_RESERVOIR[reservoir_name]

# Legacy flat dicts kept for older scripts
reservoir_min_release = {
    name: float(BASE_POLICY_CONTEXT_BY_RESERVOIR[name]["release_min"])
    for name in reservoir_options
}
reservoir_max_release = {
    name: float(BASE_POLICY_CONTEXT_BY_RESERVOIR[name]["release_max"])
    for name in reservoir_options
}
