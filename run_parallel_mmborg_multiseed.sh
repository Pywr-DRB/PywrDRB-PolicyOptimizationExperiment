#!/bin/bash
# --- Default: THREE sequential phase sweeps × multiseed loop (policy × reservoir × seeds) ---
#   Same phases as run_parallel_mmborg.sh; each Borg invocation uses the loop seed (not 71/72).
#   1) full        — USE_MRF=false → MMBorg_*_seed<S>.csv (no _mrffiltered_)
#   2) regression  — USE_MRF=true, pub-reconstruction JSON → *_seed<S>_mrffiltered_regression.csv
#   3) perfect     — USE_MRF=true, perfect-information JSON → *_seed<S>_mrffiltered_perfect.csv
#
#   CEE_MULTISEED_FROM / CEE_MULTISEED_TO — inclusive seed range (default 1..10)
#   CEE_BORG_MODES — comma-separated (default: full,regression,perfect)
#   CEE_BORG_RESERVOIRS / CEE_BORG_POLICY_TYPES / CEE_BORG_STARFIT_ONLY — same as run_parallel_mmborg.sh
#
#   Legacy single sweep: CEE_BORG_SINGLE_PHASE=1 and set USE_MRF / CEE_MRF_MASK_SOURCE (no phase loop)
#
#SBATCH --job-name=CustomMultiseed
# ./logs must exist at job start; repo tracks logs/.gitkeep — or: mkdir -p logs && sbatch ...
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=80
#SBATCH --exclusive
#SBATCH --mail-type=END
#SBATCH --mail-user=ms3654@cornell.edu

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
_PUB_JSON="${SCRIPT_DIR}/preprocessing_outputs/masking/pub_reconstruction/lower_basin_mrf_active_ranges.json"
_PERF_JSON="${SCRIPT_DIR}/preprocessing_outputs/masking/perfect_information/lower_basin_mrf_active_ranges.json"

module load python/3.11.5
source /home/fs02/pmr82_0001/ms3654/envs/borg-env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONOPTIMIZE=1

CEE_MULTISEED_FROM="${CEE_MULTISEED_FROM:-1}"
CEE_MULTISEED_TO="${CEE_MULTISEED_TO:-10}"

RESERVOIR_NAMES=("fewalter" "prompton" "beltzvilleCombined" "blueMarsh")
if [[ -n "${CEE_BORG_RESERVOIRS:-}" ]]; then
  read -r -a RESERVOIR_NAMES <<< "${CEE_BORG_RESERVOIRS}"
  echo "[subset] CEE_BORG_RESERVOIRS -> ${RESERVOIR_NAMES[*]}"
fi
MRF_RESERVOIRS=("${RESERVOIR_NAMES[@]}")

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
    full|full_series|unmasked) echo full ;;
    regression|regression_disagg|mrffiltered_regression) echo regression ;;
    perfect|pi|perfect_information|mrffiltered_perfect) echo perfect ;;
    *) echo "$x" ;;
  esac
}

submit_job() {
  local RUN_SEED=$1
  local POLICY_TYPE=$2
  local RESERVOIR_NAME=$3

  datetime=$(date '+%Y-%m-%d %H:%M:%S')
  n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

  if [[ "$USE_MRF_FLAG" -eq 1 ]]; then
    if [[ -z "${MRF_RANGES_JSON:-}" ]]; then
      case "${CEE_MRF_MASK_SOURCE:-regression_disagg}" in
        perfect|perfect_information|pi)
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
      if [[ -z "${CEE_MRF_MASK_TAG:-}" ]]; then
        case "${CEE_MRF_MASK_SOURCE:-regression_disagg}" in
          perfect|perfect_information|pi) export CEE_MRF_MASK_TAG=perfect ;;
          *) export CEE_MRF_MASK_TAG=regression_disagg ;;
        esac
      fi
      echo "[JobID ${SLURM_JOB_ID:-local}] USE_MRF=true loop_seed=${RUN_SEED} tag=_mrffiltered_${CEE_MRF_MASK_TAG} ..."
    else
      unset CEE_MRF_MASK_TAG
      echo "[JobID ${SLURM_JOB_ID:-local}] USE_MRF=false loop_seed=${RUN_SEED} ..."
    fi
  else
    unset CEE_MRF_MASK_TAG
    echo "[JobID ${SLURM_JOB_ID:-local}] non-DRB loop_seed=${RUN_SEED} ..."
  fi

  echo "Number of nodes: $SLURM_NNODES"
  echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
  echo "Running: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME borg_seed=$RUN_SEED"
  echo "Datetime: $datetime"
  echo "Total processors: $n_processors"

  if [[ " ${MRF_RESERVOIRS[@]} " =~ " ${RESERVOIR_NAME} " ]]; then
    if [[ "$USE_MRF_FLAG" -eq 1 ]]; then
      echo "  [MRF] Masked objectives — JSON: $MRF_RANGES_JSON"
      time mpirun --wdir "$SCRIPT_DIR" --oversubscribe -np "$n_processors" \
        python "$BORG_SCRIPT" "$POLICY_TYPE" "$RESERVOIR_NAME" "$RUN_SEED" "$MRF_RANGES_JSON" "true"
    else
      time mpirun --wdir "$SCRIPT_DIR" --oversubscribe -np "$n_processors" \
        python "$BORG_SCRIPT" "$POLICY_TYPE" "$RESERVOIR_NAME" "$RUN_SEED"
    fi
  else
    time mpirun --wdir "$SCRIPT_DIR" --oversubscribe -np "$n_processors" \
      python "$BORG_SCRIPT" "$POLICY_TYPE" "$RESERVOIR_NAME" "$RUN_SEED"
  fi
  echo "Finished: POLICY_TYPE=$POLICY_TYPE, RESERVOIR_NAME=$RESERVOIR_NAME seed=$RUN_SEED"
  echo "#############################################"
  wait
}

# --- Legacy: one mode only (no phase loop), seeds CEE_MULTISEED_FROM..TO ---
_sing="$(echo "${CEE_BORG_SINGLE_PHASE:-0}" | tr '[:upper:]' '[:lower:]')"
if [[ "$_sing" =~ ^(1|true|yes|on)$ ]]; then
  _mrf_raw="$(echo "${USE_MRF:-true}" | tr '[:upper:]' '[:lower:]')"
  case "$_mrf_raw" in
    true|1|yes|on) USE_MRF_FLAG=1 ;;
    *) USE_MRF_FLAG=0 ;;
  esac
  if [[ "$USE_MRF_FLAG" -eq 1 ]]; then
    if [[ -n "${MRF_RANGES_JSON:-}" ]]; then
      _chk="${MRF_RANGES_JSON}"
    else
      case "${CEE_MRF_MASK_SOURCE:-regression_disagg}" in
        perfect|perfect_information|pi) _chk="$_PERF_JSON" ;;
        *) _chk="$_PUB_JSON" ;;
      esac
    fi
    if [[ ! -f "$_chk" ]]; then
      echo "ERROR: USE_MRF=true but MRF JSON missing: ${_chk}" >&2
      exit 1
    fi
    echo "[MRF] required JSON present: ${_chk}"
  fi

  for ((seed = CEE_MULTISEED_FROM; seed <= CEE_MULTISEED_TO; seed++)); do
    for POLICY_TYPE in "${POLICY_TYPES[@]}"; do
      for RESERVOIR_NAME in "${RESERVOIR_NAMES[@]}"; do
        echo "Submitting single-phase job: $POLICY_TYPE - $RESERVOIR_NAME - SEED $seed"
        submit_job "$seed" "$POLICY_TYPE" "$RESERVOIR_NAME"
      done
    done
  done
  echo "All single-phase multi-seed jobs completed."
  exit 0
fi

IFS=',' read -r -a _PHASES_RAW <<< "${CEE_BORG_MODES:-full,regression,perfect}"
PHASES=()
for _p in "${_PHASES_RAW[@]}"; do
  _n="$(_normalize_borg_phase "${_p}")"
  PHASES+=("$_n")
done
echo "[phases] CEE_BORG_MODES -> ${PHASES[*]}  | seeds ${CEE_MULTISEED_FROM}..${CEE_MULTISEED_TO}"

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

for PHASE in "${PHASES[@]}"; do
  echo ""
  echo "###################################################################"
  echo "# Phase: ${PHASE}  multiseed ${CEE_MULTISEED_FROM}-${CEE_MULTISEED_TO}"
  echo "###################################################################"
  unset MRF_RANGES_JSON
  case "$PHASE" in
    full)
      USE_MRF_FLAG=0
      unset CEE_MRF_MASK_TAG
      export CEE_USE_MRF=0
      export CEE_MRF_MASK_SOURCE=regression_disagg
      ;;
    regression)
      USE_MRF_FLAG=1
      export CEE_USE_MRF=1
      export CEE_MRF_MASK_SOURCE=regression_disagg
      unset CEE_MRF_MASK_TAG
      export MRF_RANGES_JSON="$_PUB_JSON"
      echo "[phase regression] JSON: $MRF_RANGES_JSON"
      ;;
    perfect)
      USE_MRF_FLAG=1
      export CEE_USE_MRF=1
      export CEE_MRF_MASK_SOURCE=perfect
      unset CEE_MRF_MASK_TAG
      export MRF_RANGES_JSON="$_PERF_JSON"
      echo "[phase perfect] JSON: $MRF_RANGES_JSON"
      ;;
  esac

  for ((seed = CEE_MULTISEED_FROM; seed <= CEE_MULTISEED_TO; seed++)); do
    for POLICY_TYPE in "${POLICY_TYPES[@]}"; do
      for RESERVOIR_NAME in "${RESERVOIR_NAMES[@]}"; do
        echo "Submitting ${PHASE}: $POLICY_TYPE - $RESERVOIR_NAME - SEED $seed"
        submit_job "$seed" "$POLICY_TYPE" "$RESERVOIR_NAME"
      done
    done
  done
done

echo "All multi-phase multi-seed jobs completed."
