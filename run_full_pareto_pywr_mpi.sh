#!/bin/bash
# Full-Pareto Pywr-DRB MPI sweep: one MPI rank per (Borg variant × policy × alignment row).
# Outputs go to pywr_data/full_pareto_runs (override: CEE_FULL_PARETO_WORK_DIR) — not CEE_PYWR_WORK_DIR.
#
# Submit from repo root:
#   ./check_slurm_availability.sh    # optional: see idle nodes / queues
#   mkdir -p logs
#   sbatch run_full_pareto_pywr_mpi.sh
#
# Scale MPI ranks at submit time (each rank = one Pywr simulation), e.g. lighter load:
#   sbatch --nodes=2 --ntasks-per-node=40 run_full_pareto_pywr_mpi.sh
#
# Extra Python CLI args (quoted):
#   FULL_PARETO_EXTRA_ARGS='--dry-run' sbatch run_full_pareto_pywr_mpi.sh
#   FULL_PARETO_EXTRA_ARGS='--max-runs 100 --variants full,regression' sbatch run_full_pareto_pywr_mpi.sh
#
# Env (examples):
#   CEE_FULL_PARETO_WORK_DIR=/path/to/pywr_runs
#   CEE_FULL_PARETO_MERGE_WAIT_SEC=604800   # rank-0 merge waits for all rank JSON (default 7d; shorten if debugging)
#   CEE_BORG_RUN_VARIANTS=full,regression,perfect
#   CEE_FIGURE_POLICIES=STARFIT,PWL,RBF
#
# After MPI completes, one Stage 3 pass (plot_stage3_full_pareto_figures.py): multipanel daily+monthly + diagnostics.
#   SKIP_STAGE3_FIGURES=1 sbatch ...              # skip plotting (sims only)
#   CEE_STAGE3_MANIFEST=/path/_full_pareto_manifest.json   # override manifest path
#   CEE_STAGE3_FIG_OUT_DIR=/path/figures   # bundle subdirs created under here (default: ./figures)
#   STAGE3_WHICH=all|daily|monthly|multipanels|diagnostics   # default: all (HDF5-backed monthly when manifest exists)
#   STAGE3_SKIP_MONTHLY=1                         # skip monthly multipanel only
#   STAGE3_SKIP_DIAGNOSTICS=1                     # skip diagnostic PNGs (bias, attribution, failure map, …)
#   STAGE3_EXTRA_ARGS='--borg-variant regression --max-runs 500'   # extra CLI passed to the plot script
#
#SBATCH --job-name=full_pareto_pywr
#SBATCH --output=./logs/full_pareto_pywr_%j.out
#SBATCH --error=./logs/full_pareto_pywr_%j.err
# --- Default: 6 nodes × 40 tasks/node = 240 MPI ranks (leave headroom vs full partition; override anytime) ---
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
# #SBATCH --partition=YOUR_PARTITION   # uncomment if your cluster requires it
#SBATCH --mail-type=END
#SBATCH --mail-user=ms3654@cornell.edu

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR" || exit 1
mkdir -p "${SCRIPT_DIR}/logs"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Same Python env as run_parallel_mmborg.sh (edit path if your venv moves)
module load python/3.11.5
source /home/fs02/pmr82_0001/ms3654/envs/borg-env/bin/activate

# One OS thread per MPI rank (Pywr / numpy stay single-threaded per process)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

n_processors=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
echo "[full_pareto_pywr] start $(date -Is)"
echo "[full_pareto_pywr] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[full_pareto_pywr] SLURM_JOB_ID=${SLURM_JOB_ID:-} nodes=${SLURM_NNODES} tasks_per_node=${SLURM_NTASKS_PER_NODE} -> MPI ranks=${n_processors}"
echo "[full_pareto_pywr] CEE_FULL_PARETO_WORK_DIR=${CEE_FULL_PARETO_WORK_DIR:-<default pywr_data/full_pareto_runs>}"
echo "[full_pareto_pywr] FULL_PARETO_EXTRA_ARGS=${FULL_PARETO_EXTRA_ARGS:-<none>}"

# shellcheck disable=SC2086
time mpirun --wdir "$SCRIPT_DIR" --oversubscribe -np "${n_processors}" \
  python "${SCRIPT_DIR}/methods/ensemble/run_full_pareto_pywr_mpi.py" ${FULL_PARETO_EXTRA_ARGS:-}

echo "[full_pareto_pywr] mpi finished $(date -Is)"

# --- Stage 3 full-Pareto multipanel figures (HDF5 + manifest; see README) ---
WORK_ROOT="${CEE_FULL_PARETO_WORK_DIR:-${SCRIPT_DIR}/pywr_data/full_pareto_runs}"
MANIFEST_PATH="${CEE_STAGE3_MANIFEST:-${WORK_ROOT}/_full_pareto_manifest.json}"
STAGE3_OUT="${CEE_STAGE3_FIG_OUT_DIR:-${SCRIPT_DIR}/figures}"
STAGE3_WHICH="${STAGE3_WHICH:-all}"

if [[ "${SKIP_STAGE3_FIGURES:-0}" == "1" ]]; then
  echo "[full_pareto_pywr] SKIP_STAGE3_FIGURES=1 — skipping methods/figures_stage3/plot_stage3_full_pareto_figures.py"
elif [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[full_pareto_pywr] WARN: manifest not found: $MANIFEST_PATH"
  echo "[full_pareto_pywr]        Set CEE_STAGE3_MANIFEST or run plot_stage3_full_pareto_figures.py after merge."
else
  echo "[full_pareto_pywr] stage3 plot: manifest=${MANIFEST_PATH} out_dir=${STAGE3_OUT} --which ${STAGE3_WHICH}"
  echo "[full_pareto_pywr] STAGE3_EXTRA_ARGS=${STAGE3_EXTRA_ARGS:-<none>} STAGE3_SKIP_MONTHLY=${STAGE3_SKIP_MONTHLY:-0} STAGE3_SKIP_DIAGNOSTICS=${STAGE3_SKIP_DIAGNOSTICS:-0}"
  # shellcheck disable=SC2086
  time python "${SCRIPT_DIR}/methods/figures_stage3/plot_stage3_full_pareto_figures.py" \
    --manifest "$MANIFEST_PATH" \
    --out-dir "$STAGE3_OUT" \
    --which "$STAGE3_WHICH" \
    ${STAGE3_EXTRA_ARGS:-}
fi

echo "[full_pareto_pywr] end $(date -Is)"
