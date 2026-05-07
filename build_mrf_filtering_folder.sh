#!/usr/bin/env bash
# Populate preprocessing_outputs/filtering/ with two optimization-mode bundles:
#   regression_disagg/  — regression-disaggregation filtering assets
#   perfect_foresight/  — perfect-foresight filtering assets
#
# Each bundle contains:
#   lower_basin_mrf_contributions.csv
#   mrf_active_filter_daily.csv
#
# Usage (from CEE6400Project/, after module + venv — or this script loads them by default):
#   bash build_mrf_filtering_folder.sh
#
# Environment (all optional):
#   SKIP_MRF_PREBUILD=1           Skip entire script (no files written).
#   SKIP_PERFECT_MRF_PREBUILD=1   Only build regression_disagg/.
#   FORCE_MRF_PYWR_SIM=1          Always run Pywr for regression_disagg (ignore existing HDF5).
#   MRF_FORCE_PUB_SIM=1           Backward-compatible alias for FORCE_MRF_PYWR_SIM.
#   FORCE_MRF_FILTER_BUILD=1      Regenerate both regression + perfect simulations.
#   PUB_HDF5 / PERFECT_HDF5       Explicit HDF5 path (checked first in search order).
#   SKIP_PUB_SIM=1                Never run regression_disagg simulation; exit if no HDF5 is found.
#   REGRESSION_INFLOW             Default pub_nhmv10_BC_withObsScaled
#   PERFECT_INFLOW                Default pub_nhmv10_BC_withObsScaled
#   MRF_BUILD_START / MRF_BUILD_END  Default 1983-10-01 .. 2023-12-31 (used for pub cache filename)
#   SKIP_MRF_BUILD_MODULE_LOAD=1  Do not run module load / venv (caller already activated env)
#
# regression_disagg bundle uses flow_prediction_mode=regression_disagg.
# perfect_foresight bundle uses flow_prediction_mode=perfect_foresight.
# If matching HDF5 exists under preprocessing_outputs/pywr/, extract only;
# otherwise local Pywr simulation writes into preprocessing_outputs/pywr/.
#
# Virtualenv path: edit VENV_ACTIVATE if your borg-env lives elsewhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

VENV_ACTIVATE="${VENV_ACTIVATE:-/home/fs02/pmr82_0001/ms3654/envs/borg-env/bin/activate}"

if [[ "${SKIP_MRF_PREBUILD:-0}" == "1" ]]; then
  echo "[build_mrf_filtering_folder] SKIP_MRF_PREBUILD=1 — exiting without changes."
  exit 0
fi

if [[ "${SKIP_MRF_BUILD_MODULE_LOAD:-0}" != "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load python/3.11.5
  fi
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

REGRESSION_INFLOW="${REGRESSION_INFLOW:-pub_nhmv10_BC_withObsScaled}"
PERFECT_INFLOW="${PERFECT_INFLOW:-pub_nhmv10_BC_withObsScaled}"
MRF_BUILD_START="${MRF_BUILD_START:-1983-10-01}"
MRF_BUILD_END="${MRF_BUILD_END:-2023-12-31}"
PYWR_FLOW_ROOT="${PYWR_FLOW_ROOT:-${ROOT}/../Release_Policy_DRB/Pywr-DRB/src/pywrdrb/data/flows}"

mrf_first_existing() {
  local p
  for p in "$@"; do
    [[ -z "$p" ]] && continue
    if [[ -f "$p" ]]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

_force_pub=0
if [[ "${FORCE_MRF_PYWR_SIM:-0}" == "1" ]]; then
  _force_pub=1
fi
[[ "${MRF_FORCE_PUB_SIM:-0}" == "1" ]] && _force_pub=1
[[ "${FORCE_MRF_FILTER_BUILD:-0}" == "1" ]] && _force_pub=1
_force_perfect=0
[[ "${FORCE_MRF_FILTER_BUILD:-0}" == "1" ]] && _force_perfect=1

echo "[build_mrf_filtering_folder] project: $ROOT"

# --- (1/2) regression_disagg ---
echo "[build_mrf_filtering_folder] (1/2) regression_disagg → preprocessing_outputs/filtering/regression_disagg/"
_pub_hdf5=""
if [[ "$_force_pub" -eq 0 ]]; then
  _pub_hdf5="$(mrf_first_existing \
    "${PUB_HDF5:-}" \
    "${ROOT}/preprocessing_outputs/pywr/pywrdrb_output_${REGRESSION_INFLOW}_regression_disagg.hdf5" \
    "${ROOT}/pywrdrb_output_${REGRESSION_INFLOW}.hdf5")" || true
fi
if [[ -n "$_pub_hdf5" ]]; then
  echo "[build_mrf_filtering_folder] regression_disagg: using existing HDF5 (set FORCE_MRF_PYWR_SIM=1 to re-run Pywr): ${_pub_hdf5}"
  python -m methods.preprocessing.build_mrf_active_filters \
    --filter-bundle regression_disagg \
    --inflow-type "$REGRESSION_INFLOW" \
    --flow-prediction-mode regression_disagg \
    --skip-simulation \
    --existing-output "$_pub_hdf5"
elif [[ "${SKIP_PUB_SIM:-0}" == "1" ]]; then
  echo "ERROR: SKIP_PUB_SIM=1 but no regression HDF5 found. Set PUB_HDF5= or run without SKIP_PUB_SIM." >&2
  echo "  Tried: PUB_HDF5, preprocessing_outputs/pywr/pywrdrb_output_\${REGRESSION_INFLOW}_regression_disagg.hdf5, pywrdrb_output_\${REGRESSION_INFLOW}.hdf5" >&2
  exit 1
else
  echo "[build_mrf_filtering_folder] regression_disagg: no existing HDF5 matched — running Pywr simulation"
  python -m methods.preprocessing.build_mrf_active_filters \
    --filter-bundle regression_disagg \
    --inflow-type "$REGRESSION_INFLOW" \
    --flow-prediction-mode regression_disagg \
    --start-date "$MRF_BUILD_START" \
    --end-date "$MRF_BUILD_END" \
    --work-dir "${ROOT}/preprocessing_outputs/pywr"
fi

if [[ "${SKIP_PERFECT_MRF_PREBUILD:-0}" == "1" ]]; then
  echo "[build_mrf_filtering_folder] SKIP_PERFECT_MRF_PREBUILD=1 — done (regression_disagg only)."
  exit 0
fi

# --- (2/2) perfect_foresight ---
echo "[build_mrf_filtering_folder] (2/2) perfect_foresight → preprocessing_outputs/filtering/perfect_foresight/"
_perf_hdf5=""
if [[ "$_force_perfect" -eq 0 ]]; then
  _perf_hdf5="$(mrf_first_existing \
    "${PERFECT_HDF5:-}" \
    "${ROOT}/preprocessing_outputs/pywr/pywrdrb_output_${PERFECT_INFLOW}_perfect_foresight.hdf5" \
    "${ROOT}/pywrdrb_output_${PERFECT_INFLOW}.hdf5")" || true
fi
if [[ -n "$_perf_hdf5" ]]; then
  echo "[build_mrf_filtering_folder] perfect_foresight: extracting from HDF5: ${_perf_hdf5}"
  python -m methods.preprocessing.build_mrf_active_filters \
    --filter-bundle perfect_foresight \
    --inflow-type "$PERFECT_INFLOW" \
    --flow-prediction-mode perfect_foresight \
    --skip-simulation \
    --existing-output "$_perf_hdf5"
else
  _perfect_inflow_for_sim="$PERFECT_INFLOW"
  if [[ ! -d "${PYWR_FLOW_ROOT}/${_perfect_inflow_for_sim}" ]]; then
    if [[ -d "${PYWR_FLOW_ROOT}/${REGRESSION_INFLOW}" ]]; then
      echo "[build_mrf_filtering_folder] WARNING: flows/${_perfect_inflow_for_sim} not found; using REGRESSION_INFLOW=${REGRESSION_INFLOW} for local perfect_foresight fallback simulation."
      _perfect_inflow_for_sim="$REGRESSION_INFLOW"
    else
      echo "ERROR: flows/${_perfect_inflow_for_sim} not found and fallback flows/${REGRESSION_INFLOW} also missing." >&2
      echo "  Set PERFECT_INFLOW= and/or PYWR_FLOW_ROOT= to a valid flow directory in your Pywr-DRB checkout." >&2
      exit 1
    fi
  fi
  if [[ "$_force_perfect" -eq 1 ]]; then
    echo "[build_mrf_filtering_folder] perfect_foresight: FORCE_MRF_FILTER_BUILD=1 -> re-running Pywr simulation"
  else
    echo "[build_mrf_filtering_folder] perfect_foresight: no existing HDF5 found; running Pywr simulation"
  fi
  python -m methods.preprocessing.build_mrf_active_filters \
    --filter-bundle perfect_foresight \
    --inflow-type "$_perfect_inflow_for_sim" \
    --flow-prediction-mode perfect_foresight \
    --start-date "$MRF_BUILD_START" \
    --end-date "$MRF_BUILD_END" \
    --work-dir "${ROOT}/preprocessing_outputs/pywr"
fi

echo "[build_mrf_filtering_folder] done."
echo "  $ROOT/preprocessing_outputs/filtering/regression_disagg/mrf_active_filter_daily.csv"
echo "  $ROOT/preprocessing_outputs/filtering/perfect_foresight/mrf_active_filter_daily.csv"
