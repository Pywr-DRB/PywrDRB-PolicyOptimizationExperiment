#!/usr/bin/env bash
# Populate preprocessing_outputs/filtering/ with two bundles:
#   pub_reconstruction/     — Pywr-DRB simulation if no pub HDF5 is found, else extract only
#   perfect_information/    — always from an existing HDF5 only (never runs Pywr)
#
# Each bundle contains:
#   lower_basin_mrf_contributions.csv
#   lower_basin_mrf_active_ranges.json
#   mrf_active_filter_daily.csv
#
# Usage (from CEE6400Project/, after module + venv — or this script loads them by default):
#   bash build_mrf_filtering_folder.sh
#
# Environment (all optional):
#   SKIP_MRF_PREBUILD=1           Skip entire script (no files written).
#   SKIP_PERFECT_MRF_PREBUILD=1   Only build pub_reconstruction/.
#   FORCE_MRF_PYWR_SIM=1          Always run Pywr for pub (ignore existing pub HDF5).
#   MRF_FORCE_PUB_SIM=1           Same, pub only.
#   PUB_HDF5 / PERFECT_HDF5       Explicit HDF5 path (checked first in search order).
#   SKIP_PUB_SIM=1                Never run Pywr for pub; exit if no pub HDF5 is found.
#   PUB_INFLOW                    Default pub_nhmv10_BC_withObsScaled
#   PERFECT_INFLOW                Default pub_nhmv10_BC_withObsScaled_perfect_information
#   MRF_BUILD_START / MRF_BUILD_END  Default 1983-10-01 .. 2023-12-31 (used for pub cache filename)
#   SKIP_MRF_BUILD_MODULE_LOAD=1  Do not run module load / venv (caller already activated env)
#
# Pub: if a matching HDF5 exists under pywr_data/ (or legacy copy at project root), only
# extract — otherwise Pywr runs into pywr_data/_mrf_pub_sim/. Perfect: HDF5 must exist;
# Pywr is never run for that bundle.
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

PUB_INFLOW="${PUB_INFLOW:-pub_nhmv10_BC_withObsScaled}"
PERFECT_INFLOW="${PERFECT_INFLOW:-pub_nhmv10_BC_withObsScaled_perfect_information}"
MRF_BUILD_START="${MRF_BUILD_START:-1983-10-01}"
MRF_BUILD_END="${MRF_BUILD_END:-2023-12-31}"

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

echo "[build_mrf_filtering_folder] project: $ROOT"

# --- (1/2) pub_reconstruction ---
echo "[build_mrf_filtering_folder] (1/2) pub reconstruction → preprocessing_outputs/filtering/pub_reconstruction/"
_pub_hdf5=""
if [[ "$_force_pub" -eq 0 ]]; then
  _pub_hdf5="$(mrf_first_existing \
    "${PUB_HDF5:-}" \
    "${ROOT}/pywr_data/_mrf_pub_sim/pywrdrb_output_${PUB_INFLOW}.hdf5" \
    "${ROOT}/pywr_data/_pywr_default_cache/output_default_${MRF_BUILD_START}_${MRF_BUILD_END}_${PUB_INFLOW}.hdf5" \
    "${ROOT}/pywr_data/_pywr_default/pywrdrb_output_${PUB_INFLOW}.hdf5" \
    "${ROOT}/pywr_data/_pywr_default/${PUB_INFLOW}.hdf5" \
    "${ROOT}/pywrdrb_output_${PUB_INFLOW}.hdf5")" || true
fi
if [[ -n "$_pub_hdf5" ]]; then
  echo "[build_mrf_filtering_folder] pub: using existing HDF5 (set FORCE_MRF_PYWR_SIM=1 to re-run Pywr): ${_pub_hdf5}"
  python -m methods.preprocessing.build_mrf_active_filters \
    --filter-bundle pub_reconstruction \
    --inflow-type "$PUB_INFLOW" \
    --skip-simulation \
    --existing-output "$_pub_hdf5"
elif [[ "${SKIP_PUB_SIM:-0}" == "1" ]]; then
  echo "ERROR: SKIP_PUB_SIM=1 but no pub HDF5 found. Set PUB_HDF5= or run without SKIP_PUB_SIM." >&2
  echo "  Tried: PUB_HDF5, pywrdrb_output_\${PUB_INFLOW}.hdf5, _pywr_default_cache/output_default_<dates>_\${PUB_INFLOW}.hdf5, ..." >&2
  exit 1
else
  echo "[build_mrf_filtering_folder] pub: no existing HDF5 matched — running Pywr simulation"
  python -m methods.preprocessing.build_mrf_active_filters \
    --filter-bundle pub_reconstruction \
    --inflow-type "$PUB_INFLOW" \
    --start-date "$MRF_BUILD_START" \
    --end-date "$MRF_BUILD_END" \
    --work-dir "${ROOT}/pywr_data/_mrf_pub_sim"
fi

if [[ "${SKIP_PERFECT_MRF_PREBUILD:-0}" == "1" ]]; then
  echo "[build_mrf_filtering_folder] SKIP_PERFECT_MRF_PREBUILD=1 — done (pub only)."
  exit 0
fi

# --- (2/2) perfect_information (HDF5 only — never Pywr) ---
echo "[build_mrf_filtering_folder] (2/2) perfect information → preprocessing_outputs/filtering/perfect_information/"
_perf_hdf5="$(mrf_first_existing \
  "${PERFECT_HDF5:-}" \
  "${ROOT}/pywr_data/_pywr_perfect_information/${PERFECT_INFLOW}.hdf5" \
  "${ROOT}/pywr_data/_pywr_default/_pywr_perfect_information/${PERFECT_INFLOW}.hdf5" \
  "${ROOT}/pywrdrb_output_${PERFECT_INFLOW}.hdf5")" || true
if [[ -n "$_perf_hdf5" ]]; then
  echo "[build_mrf_filtering_folder] perfect: extracting from HDF5 (Pywr is not run for this bundle): ${_perf_hdf5}"
  python -m methods.preprocessing.build_mrf_active_filters \
    --filter-bundle perfect_information \
    --inflow-type "$PERFECT_INFLOW" \
    --skip-simulation \
    --existing-output "$_perf_hdf5"
else
  echo "ERROR: perfect_information filtering needs an existing HDF5; this script never runs Pywr for perfect." >&2
  echo "  Set PERFECT_HDF5=/path/to/${PERFECT_INFLOW}.hdf5 or copy that file to:" >&2
  echo "    ${ROOT}/pywr_data/_pywr_perfect_information/${PERFECT_INFLOW}.hdf5" >&2
  exit 1
fi

echo "[build_mrf_filtering_folder] done."
echo "  $ROOT/preprocessing_outputs/filtering/pub_reconstruction/lower_basin_mrf_active_ranges.json"
echo "  $ROOT/preprocessing_outputs/filtering/perfect_information/lower_basin_mrf_active_ranges.json"
