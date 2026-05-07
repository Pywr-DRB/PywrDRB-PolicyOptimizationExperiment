#!/bin/bash
# Default behavior: run three sequential sweeps over the same policy x reservoir grid.
#   1) full       -> USE_MRF=false, outputs MMBorg_*_seed72.csv (no _mrffiltered_)
#   2) regression -> USE_MRF=true + pub-reconstruction JSON, outputs *_seed71_mrffiltered_regression.csv
#   3) perfect    -> USE_MRF=true + perfect-information JSON, outputs *_seed71_mrffiltered_perfect.csv
#
# Main phase selector:
#   CEE_BORG_MODES=full,regression,perfect
#   (Use only canonical tokens: full, regression, perfect.)
#
# Other env overrides (VAR=value sbatch run_parallel_mmborg.sh):
#   BORG_SEED_FILTERED
#   BORG_SEED_UNFILTERED
#   MRF_RANGES_JSON
#   CEE_MRF_FILTER_TAG
#   CEE_BORG_RESERVOIRS
#   CEE_BORG_POLICY_TYPES / CEE_BORG_STARFIT_ONLY
#SBATCH --job-name=ResBorg
# Log paths: ./logs must exist when the job starts (Slurm opens them before this script runs).
# The repo tracks logs/.gitkeep so clone has ./logs; otherwise: mkdir -p logs && sbatch ...
#SBATCH --output=./logs/ResBorg.out
#SBATCH --error=./logs/ResBorg.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --mail-type=END                    # Send email at job end
#SBATCH --mail-user=ms3654@cornell.edu     # Email for notifications

# -----------------------------------------------------------------------------
# Runtime context: always execute from submit directory/repo root so relative
# paths (Borg script, filtering JSON, logs, outputs) resolve the same on all ranks.
# -----------------------------------------------------------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR" || exit 1
mkdir -p "${SCRIPT_DIR}/logs"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
echo "[paths] SCRIPT_DIR=${SCRIPT_DIR}  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"

BORG_SCRIPT="${SCRIPT_DIR}/03_parallel_borg_run.py"
_PUB_JSON="${SCRIPT_DIR}/preprocessing_outputs/filtering/pub_reconstruction/lower_basin_mrf_active_ranges.json"
_PERF_JSON="${SCRIPT_DIR}/preprocessing_outputs/filtering/perfect_information/lower_basin_mrf_active_ranges.json"

# -----------------------------------------------------------------------------
# Environment/bootstrap: pin Python + BLAS/OpenMP threading for stable MPI runs.
# -----------------------------------------------------------------------------
module load python/3.11.5
source /home/fs02/pmr82_0001/ms3654/envs/borg-env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONOPTIMIZE=1

BORG_SEED_FILTERED="${BORG_SEED_FILTERED:-71}"
BORG_SEED_UNFILTERED="${BORG_SEED_UNFILTERED:-72}"

# -----------------------------------------------------------------------------
# Worklist selection: default lower-basin calibration set, with env overrides
# for subsets (useful for debug runs or reduced sweeps).
# -----------------------------------------------------------------------------
RESERVOIR_NAMES=("fewalter" "prompton" "beltzvilleCombined" "blueMarsh")
if [[ -n "${CEE_BORG_RESERVOIRS:-}" ]]; then
  read -r -a RESERVOIR_NAMES <<< "${CEE_BORG_RESERVOIRS}"
  echo "[subset] CEE_BORG_RESERVOIRS -> ${RESERVOIR_NAMES[*]}"
fi
MRF_RESERVOIRS=("fewalter" "prompton" "beltzvilleCombined" "blueMarsh")

_stf_only="$(echo "${CEE_BORG_STARFIT_ONLY:-0}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${CEE_BORG_POLICY_TYPES:-}" ]]; then
  read -r -a POLICY_TYPES <<< "${CEE_BORG_POLICY_TYPES}"
  echo "[subset] CEE_BORG_POLICY_TYPES -> ${POLICY_TYPES[*]}"
elif [[ "$_stf_only" =~ ^(1|true|yes|on)$ ]]; then
  POLICY_TYPES=("STARFIT")
  echo "[subset] CEE_BORG_STARFIT_ONLY -> STARFIT only"
else
  POLICY_TYPES=("RBF" "PWL" "STARFIT")
fi

_normalize_borg_phase() {
  local x
  x="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  case "$x" in
    full|regression|perfect) echo "$x" ;;
    *) echo "$x" ;;
  esac
}

# --- Default: multi-phase (full + regression MRF + perfect MRF) ---
IFS=',' read -r -a _PHASES_RAW <<< "${CEE_BORG_MODES:-full,regression,perfect}"
PHASES=()
for _p in "${_PHASES_RAW[@]}"; do
  _n="$(_normalize_borg_phase "${_p}")"
  PHASES+=("$_n")
done
echo "[phases] CEE_BORG_MODES -> ${PHASES[*]}"

# Fail fast before starting MPI if required filter JSONs are missing.
for ph in "${PHASES[@]}"; do
  case "$ph" in
    full) ;;
    regression)
      if [[ -z "${MRF_RANGES_JSON:-}" && ! -f "$_PUB_JSON" ]]; then
        echo "ERROR: regression MRF phase requires JSON: $_PUB_JSON" >&2
        exit 1
      fi
      ;;
    perfect)
      if [[ -z "${MRF_RANGES_JSON:-}" && ! -f "$_PERF_JSON" ]]; then
        echo "ERROR: perfect MRF phase requires JSON: $_PERF_JSON" >&2
        exit 1
      fi
      ;;
    *)
      echo "ERROR: unknown phase '$ph' (use full, regression, or perfect)" >&2
      exit 1
      ;;
  esac
done

submit_job() {
    local POLICY_TYPE=$1
    local RESERVOIR_NAME=$2

    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

    # Per-job phase context: determines whether objectives are filtered and
    # which seed / JSON / suffix are used in output naming.
    local RUN_SEED
    if [[ "$USE_MRF_FLAG" -eq 1 ]]; then
      if [[ -z "${MRF_RANGES_JSON:-}" ]]; then
        case "${CEE_MRF_FILTER_SOURCE:-regression_disagg}" in
          perfect)
            MRF_RANGES_JSON="$_PERF_JSON"
            ;;
          *)
            MRF_RANGES_JSON="$_PUB_JSON"
            ;;
        esac
      fi
    else
      unset MRF_RANGES_JSON
    fi

    if [[ " ${MRF_RESERVOIRS[@]} " =~ " ${RESERVOIR_NAME} " ]]; then
        if [[ "$USE_MRF_FLAG" -eq 1 ]]; then
            RUN_SEED="$BORG_SEED_FILTERED"
            if [[ -z "${CEE_MRF_FILTER_TAG:-}" ]]; then
              case "${CEE_MRF_FILTER_SOURCE:-regression_disagg}" in
                perfect) export CEE_MRF_FILTER_TAG=perfect ;;
                *) export CEE_MRF_FILTER_TAG=regression_disagg ;;
              esac
            fi
            echo "Running: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME seed=$RUN_SEED USE_MRF=true (MRF-filtered objectives, tag=_mrffiltered_${CEE_MRF_FILTER_TAG})"
            echo "Datetime: $datetime"
            echo "Total processors: $n_processors"
            echo "  [MRF] Filtered objectives — JSON: $MRF_RANGES_JSON"
            time mpirun --wdir "$SCRIPT_DIR" --bind-to core --map-by ppr:${SLURM_NTASKS_PER_NODE}:node -np "$n_processors" \
                python "$BORG_SCRIPT" "$POLICY_TYPE" "$RESERVOIR_NAME" "$RUN_SEED" "$MRF_RANGES_JSON" "true"
        else
            unset CEE_MRF_FILTER_TAG
            RUN_SEED="$BORG_SEED_UNFILTERED"
            echo "Running: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME seed=$RUN_SEED USE_MRF=false (full series)"
            echo "Datetime: $datetime"
            echo "Total processors: $n_processors"
            time mpirun --wdir "$SCRIPT_DIR" --bind-to core --map-by ppr:${SLURM_NTASKS_PER_NODE}:node -np "$n_processors" \
                python "$BORG_SCRIPT" "$POLICY_TYPE" "$RESERVOIR_NAME" "$RUN_SEED"
        fi
    else
        unset CEE_MRF_FILTER_TAG
        RUN_SEED="$BORG_SEED_UNFILTERED"
        echo "Running: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME seed=$RUN_SEED (non-DRB, full series)"
        echo "Datetime: $datetime"
        echo "Total processors: $n_processors"
        time mpirun --wdir "$SCRIPT_DIR" --bind-to core --map-by ppr:${SLURM_NTASKS_PER_NODE}:node -np "$n_processors" \
            python "$BORG_SCRIPT" "$POLICY_TYPE" "$RESERVOIR_NAME" "$RUN_SEED"
    fi
    echo "Finished: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME"
    echo "#############################################"

    wait
}

# Phase loop sets global env flags consumed by both shell wrapper and
# 03_parallel_borg_run.py (CEE_USE_MRF / CEE_MRF_FILTER_SOURCE / JSON path).
for PHASE in "${PHASES[@]}"; do
  echo ""
  echo "###################################################################"
  echo "# Phase: ${PHASE}  (policies=${#POLICY_TYPES[@]} reservoirs=${#RESERVOIR_NAMES[@]})"
  echo "###################################################################"
  unset MRF_RANGES_JSON
  case "$PHASE" in
    full)
      USE_MRF_FLAG=0
      unset CEE_MRF_FILTER_TAG
      export CEE_USE_MRF=0
      export CEE_MRF_FILTER_SOURCE=regression_disagg
      ;;
    regression)
      USE_MRF_FLAG=1
      export CEE_USE_MRF=1
      export CEE_MRF_FILTER_SOURCE=regression_disagg
      unset CEE_MRF_FILTER_TAG
      export MRF_RANGES_JSON="$_PUB_JSON"
      echo "[phase regression] JSON: $MRF_RANGES_JSON"
      ;;
    perfect)
      USE_MRF_FLAG=1
      export CEE_USE_MRF=1
      export CEE_MRF_FILTER_SOURCE=perfect
      unset CEE_MRF_FILTER_TAG
      export MRF_RANGES_JSON="$_PERF_JSON"
      echo "[phase perfect] JSON: $MRF_RANGES_JSON"
      ;;
  esac

  for POLICY_TYPE in "${POLICY_TYPES[@]}"; do
    for RESERVOIR_NAME in "${RESERVOIR_NAMES[@]}"; do
      echo "Submitting job for ${PHASE}: $POLICY_TYPE - $RESERVOIR_NAME"
      submit_job "$POLICY_TYPE" "$RESERVOIR_NAME"
    done
  done
done

echo "All phases completed."
