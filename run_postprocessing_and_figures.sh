#!/bin/bash
#SBATCH --job-name=cee_fig_pipeline
# ./logs must exist at job start; repo tracks logs/.gitkeep — or: mkdir -p logs && sbatch ...
#SBATCH --output=./logs/figure_pipeline_%j.out
#SBATCH --error=./logs/figure_pipeline_%j.err
# --- One Python process at a time (serial). Extra Slurm CPUs/nodes do not speed this up.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ms3654@cornell.edu
# Main figure pipeline (figures 1–11 via 04_make_figures). Figures 12–21 (full-Pareto manifest)
# are separate: python -m methods.figures_stage3.plot_stage3_full_pareto_figures ...
# Submit from repo root:
#   sbatch run_postprocessing_and_figures.sh
#
# Override any default with env vars at submit time (see below).
#
# Trenton target override (VERY IMPORTANT):
# - This script does not set the Trenton target itself; it forwards the environment.
# - Any child call that reads ``CEE_TRENTON_TARGET_MGD`` (e.g., validation plots; full-Pareto figs 12–21 when run separately)
#   will use the value you export at submit time.
# - Default in code is 1938.950669 MGD (Pywr-DRB baseline).
# - +30% sensitivity example:
#     CEE_TRENTON_TARGET_MGD=2520.6358697 sbatch run_postprocessing_and_figures.sh
#
#   | Borg bundle | CEE_FIG_SUBDIR                     | Pywr flow (env)              |
#   |--------------------------------|------------------------------------|------------------------------|
#   | Full-series (no MRF filter)    | borg_full_series                   | FLOW_MODE_FULL |
#   | MRF-filtered regression bundle   | borg_mrffiltered_regression        | FLOW_MODE_REGRESSION_DISAGG  |
#   | MRF-filtered perfect           | borg_mrffiltered_perfect_foresight | FLOW_MODE_PERFECT            |
#
# Core defaults: CEE_POSTPROCESS_BUNDLE=all, FLOW_MODE_FULL=perfect_foresight,
# BORG_SEED_FILTERED=71, BORG_SEED_UNMASKED=72. All focal picks + all policies: leave CEE_DESIRED_PICKS /
# CEE_FIGURE_POLICIES unset.
#
# Speed tips: CEE_FIG12_FIRST=0 skips the extra Fig1/2-only pass; FIG_SCRIPT_ARGS="--plots-only" reuses
# cached Pywr; SKIP_BASELINE_BUILD=1 skips default Pywr + baseline plots; DEBUG_FAST=1 trims scope.
#
# Requires matching Borg CSVs per bundle (methods/borg_paths.py).
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR" || exit 1
mkdir -p logs data/postprocess_runs/logs
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-2}"

module load python/3.11.5
source /home/fs02/pmr82_0001/ms3654/envs/borg-env/bin/activate

# Reuse cached default Pywr run unless explicitly forced by caller.
# Set CEE_FORCE_DEFAULT_RERUN=1 at submission time to force a fresh default run
# (re-run methods/postprocess/build_default_timeseries.py even if cache + CSVs already exist).
export CEE_FORCE_DEFAULT_RERUN="${CEE_FORCE_DEFAULT_RERUN:-0}"

# Paths aligned with methods/postprocess/build_default_timeseries.py (defaults; override via DRB_* env).
# Export so Python subprocesses inherit exactly the same locations.
export DRB_OUTPUT_DIR="${DRB_OUTPUT_DIR:-${SCRIPT_DIR}/pywr_data}"
export DRB_DEFAULT_CACHE="${DRB_DEFAULT_CACHE:-${DRB_OUTPUT_DIR}/_pywr_default_cache}"
# Same tag as build_default_timeseries.py CLI defaults (--start/--end/--inflow-type)
DEFAULT_PYWR_TAG="1983-10-01_2023-12-31_pub_nhmv10_BC_withObsScaled"
DEFAULT_PYWR_H5="${DRB_DEFAULT_CACHE}/output_default_${DEFAULT_PYWR_TAG}.hdf5"

_force_default_rerun() {
  local v="${CEE_FORCE_DEFAULT_RERUN:-0}"
  v="${v,,}"
  [[ "${v}" == "1" || "${v}" == "true" || "${v}" == "yes" || "${v}" == "on" ]]
}

_default_timeseries_ready() {
  [[ -f "${DEFAULT_PYWR_H5}" ]]
}

# Borg seed for MRF-filtered objective runs (preferred name); legacy: BORG_SEED_MASKED.
BORG_SEED_FILTERED="${BORG_SEED_FILTERED:-${BORG_SEED_MASKED:-71}}"
BORG_SEED_UNMASKED="${BORG_SEED_UNMASKED:-72}"

FIG_SCRIPT="${FIG_SCRIPT:-04_make_figures.py}"
# Extra args for 04_make_figures.py (e.g. --plots-only).
FIG_SCRIPT_ARGS="${FIG_SCRIPT_ARGS:-}"
VALIDATE_ARGS="${VALIDATE_ARGS:-}"
CEE_FIG12_FIRST="${CEE_FIG12_FIRST:-1}"
# Optional per-bundle figure subset passed to 04_make_figures.py --figures.
# Accepts the same tokens as 04_make_figures.py (e.g. "1 2 5-7" or "1-6 10 11").
# - CEE_FIGURES_ALL applies to every bundle unless bundle-specific override is set.
# - CEE_FIGURES_FULL / CEE_FIGURES_REGRESSION_DISAGG / CEE_FIGURES_PERFECT override CEE_FIGURES_ALL.
# Examples:
#   CEE_FIGURES_ALL="1-2 7-8" sbatch run_postprocessing_and_figures.sh
#   CEE_FIGURES_FULL="1-6" CEE_FIGURES_PERFECT="7-11" sbatch run_postprocessing_and_figures.sh
CEE_FIGURES_ALL="${CEE_FIGURES_ALL:-}"
CEE_FIGURES_FULL="${CEE_FIGURES_FULL:-}"
CEE_FIGURES_REGRESSION_DISAGG="${CEE_FIGURES_REGRESSION_DISAGG:-}"
CEE_FIGURES_PERFECT="${CEE_FIGURES_PERFECT:-}"
# Pywr ModelBuilder flow_prediction_mode per bundle (orthogonal to which Borg CSV / MRF variant).
FLOW_MODE_FULL="${FLOW_MODE_FULL:-perfect_foresight}"
FLOW_MODE_REGRESSION_DISAGG="${FLOW_MODE_REGRESSION_DISAGG:-regression_disagg}"
FLOW_MODE_PERFECT="${FLOW_MODE_PERFECT:-perfect_foresight}"

# Quick debug profile: single reservoir/policy/pick to validate dynamics quickly.
# For Fig 3/9 only (without changing global CEE_FIGURE_POLICIES), use e.g.:
#   CEE_FIG4_RESERVOIRS=beltzvilleCombined CEE_FIG4_POLICIES=PWL CEE_FIG4_K=1
# Example:
#   DEBUG_FAST=1 DEBUG_ONLY_PERFECT=1 DEBUG_ONE_RESERVOIR=beltzvilleCombined \
#   DEBUG_ONE_POLICY=PWL DEBUG_ONE_PICK="Best Average NSE" sbatch run_postprocessing_and_figures.sh
DEBUG_FAST="${DEBUG_FAST:-0}"
# When DEBUG_FAST=1, default is still all three Borg bundles (full + MRF regression_disagg + MRF perfect).
# Set DEBUG_ONLY_PERFECT=1 to run only the perfect-foresight filtered bundle (faster iteration).
DEBUG_ONLY_PERFECT="${DEBUG_ONLY_PERFECT:-0}"
DEBUG_SKIP_BASELINE="${DEBUG_SKIP_BASELINE:-1}"
DEBUG_SKIP_SUMMARY="${DEBUG_SKIP_SUMMARY:-1}"
DEBUG_VALIDATE_ONLY="${DEBUG_VALIDATE_ONLY:-0}"
DEBUG_ONE_RESERVOIR="${DEBUG_ONE_RESERVOIR:-beltzvilleCombined}"
DEBUG_ONE_POLICY="${DEBUG_ONE_POLICY:-PWL}"
DEBUG_ONE_PICK="${DEBUG_ONE_PICK:-Best Average NSE}"
DEBUG_SKIP_DEFAULT_RUN="${DEBUG_SKIP_DEFAULT_RUN:-1}"

if [[ "${DEBUG_FAST}" == "1" ]]; then
  RUN_FULL=0
  RUN_REGRESSION_DISAGG=0
  RUN_PERFECT=1
  if [[ "${DEBUG_ONLY_PERFECT}" != "1" ]]; then
    RUN_FULL=1
    RUN_REGRESSION_DISAGG=1
  fi
  # Focus figure pipeline to one policy/pick when used in debug mode.
  export CEE_FIGURE_POLICIES="${DEBUG_ONE_POLICY}"
  export CEE_DESIRED_PICKS="${DEBUG_ONE_PICK}"
else
  RUN_FULL=1
  RUN_REGRESSION_DISAGG=1
  RUN_PERFECT=1
fi

# Which Borg CSV / MRF-filtered bundle(s) to run: overrides DEBUG_FAST RUN_* when set.
#   full              — unmasked Borg -> figures/borg_full_series/ (Pywr: FLOW_MODE_FULL)
#   regression_disagg — regression-disaggregation MRF-filtered Borg -> borg_mrffiltered_regression/ (Pywr: FLOW_MODE_REGRESSION_DISAGG)
#   perfect           — perfect-foresight MRF-filtered Borg -> borg_mrffiltered_perfect_foresight/ (Pywr: FLOW_MODE_PERFECT)
#   all               — all three (same as full,regression_disagg,perfect). Comma-separated OK, e.g. regression_disagg,perfect
#
# Default: all three bundles. DEBUG_FAST leaves CEE_POSTPROCESS_BUNDLE unset unless you export it, so RUN_* from the
# DEBUG_FAST block apply; with DEBUG_ONLY_PERFECT=0 (default), that is full + both filtered.
if [[ "${DEBUG_FAST}" != "1" ]]; then
  CEE_POSTPROCESS_BUNDLE="${CEE_POSTPROCESS_BUNDLE:-all}"
else
  CEE_POSTPROCESS_BUNDLE="${CEE_POSTPROCESS_BUNDLE:-}"
fi
if [[ -n "${CEE_POSTPROCESS_BUNDLE}" ]]; then
  RUN_FULL=0
  RUN_REGRESSION_DISAGG=0
  RUN_PERFECT=0
  _norm="${CEE_POSTPROCESS_BUNDLE// /}"
  _norm="${_norm,,}"
  if [[ "${_norm}" == "all" ]]; then
    RUN_FULL=1
    RUN_REGRESSION_DISAGG=1
    RUN_PERFECT=1
  else
    IFS=',' read -ra _cee_bundle_toks <<< "${_norm}"
    for _b in "${_cee_bundle_toks[@]}"; do
      [[ -z "${_b}" ]] && continue
      case "${_b}" in
        full|full_series|borg_full_series)
          RUN_FULL=1
          ;;
        regression_disagg|regression-disagg|mrffiltered_regression|borg_mrffiltered_regression)
          RUN_REGRESSION_DISAGG=1
          ;;
        perfect|mrffiltered_perfect|borg_mrffiltered_perfect_foresight)
          RUN_PERFECT=1
          ;;
        *)
          echo "[run_postprocessing] Unknown CEE_POSTPROCESS_BUNDLE token '${_b}' (use full, regression_disagg, perfect, mrffiltered_regression, borg_mrffiltered_regression, all, or comma-separated)" >&2
          exit 1
          ;;
      esac
    done
  fi
  if [[ "$((RUN_FULL + RUN_REGRESSION_DISAGG + RUN_PERFECT))" -eq 0 ]]; then
    echo "[run_postprocessing] CEE_POSTPROCESS_BUNDLE produced no bundles" >&2
    exit 1
  fi
  echo "[run_postprocessing] CEE_POSTPROCESS_BUNDLE=${CEE_POSTPROCESS_BUNDLE} -> RUN_FULL=${RUN_FULL} RUN_REGRESSION_DISAGG=${RUN_REGRESSION_DISAGG} RUN_PERFECT=${RUN_PERFECT} (Pywr: FULL=${FLOW_MODE_FULL} REGRESSION_DISAGG=${FLOW_MODE_REGRESSION_DISAGG} PERFECT=${FLOW_MODE_PERFECT})"
fi

SKIP_BASELINE_BUILD="${SKIP_BASELINE_BUILD:-0}"
if [[ "${SKIP_BASELINE_BUILD}" == "1" ]]; then
  echo "[skip] SKIP_BASELINE_BUILD=1 — skipping build_default_timeseries, baseline metrics, baseline dynamics."
elif [[ "${DEBUG_FAST}" == "1" && "${DEBUG_SKIP_BASELINE}" == "1" ]]; then
  echo "[debug-fast] skipping baseline timeseries/metrics/dynamics."
else
  echo "building default timeseries and computing baseline metrics..."
  if _force_default_rerun || ! _default_timeseries_ready; then
    if _force_default_rerun; then
      echo "[default] CEE_FORCE_DEFAULT_RERUN=1 — rebuilding default Pywr cache and per-reservoir CSVs..."
    else
      echo "[default] Missing default cache (${DEFAULT_PYWR_H5}) — running build_default_timeseries.py"
    fi
    python -m methods.postprocess.build_default_timeseries
    echo "Default timeseries built."
  else
    echo "[default] Reusing existing default Pywr HDF5 (set CEE_FORCE_DEFAULT_RERUN=1 to rebuild):"
    echo "         ${DEFAULT_PYWR_H5}"
  fi

  echo "Computing baseline metrics..."
  python -m methods.postprocess.compute_baseline_metrics
  echo "Baseline metrics computed."

  echo "Plotting baseline dynamics (obs vs default Pywr for metric windows)..."
  python -m methods.postprocess.plot_baseline_dynamics
fi

_summ_filtered() {
  local filter_tag="$1"
  CEE_BORG_SEED="${BORG_SEED_FILTERED}" CEE_BORG_MRF_FILTERED=1 CEE_MRF_FILTER_TAG="${filter_tag}" \
    python -m methods.postprocess.summarize_optimization \
    -o "${SCRIPT_DIR}/outputs/optimization_summary_seed${BORG_SEED_FILTERED}_mrffiltered_${filter_tag}.csv" \
    --out-json "${SCRIPT_DIR}/outputs/optimization_summary_seed${BORG_SEED_FILTERED}_mrffiltered_${filter_tag}.json"
}

if [[ "${DEBUG_FAST}" == "1" && "${DEBUG_SKIP_SUMMARY}" == "1" ]]; then
  echo "[debug-fast] skipping Borg summary tables."
else
  if [[ "${RUN_REGRESSION_DISAGG}" == "1" ]]; then
    echo "Summarizing Borg (MRF-filtered regression-disaggregation bundle, seed ${BORG_SEED_FILTERED})..."
    _summ_filtered regression_disagg || true
  fi
  if [[ "${RUN_PERFECT}" == "1" ]]; then
    echo "Summarizing Borg (MRF-filtered perfect-foresight bundle, seed ${BORG_SEED_FILTERED})..."
    _summ_filtered perfect || true
  fi
  if [[ "${RUN_FULL}" == "1" ]]; then
    echo "Summarizing Borg solution counts (full series, seed ${BORG_SEED_UNMASKED})..."
    CEE_BORG_SEED="${BORG_SEED_UNMASKED}" CEE_BORG_MRF_FILTERED=0 \
      python -m methods.postprocess.summarize_optimization \
      -o "${SCRIPT_DIR}/outputs/optimization_summary_seed${BORG_SEED_UNMASKED}_full.csv" \
      --out-json "${SCRIPT_DIR}/outputs/optimization_summary_seed${BORG_SEED_UNMASKED}_full.json" || true
  fi
fi

_run_fig_validate() {
  local filter_tag="$1"
  local sub="$2"
  local flow_mode="$3"
  local figures_spec="$4"
  local validate_extra_args=""
  local -a fig_args=()
  local -a figures_tokens=()
  if [[ -n "${figures_spec}" ]]; then
    # shellcheck disable=SC2206
    figures_tokens=( ${figures_spec} )
    fig_args=(--figures "${figures_tokens[@]}")
    echo "[figures] ${sub}: ${figures_spec}"
  fi
  if [[ "${DEBUG_FAST}" == "1" ]]; then
    validate_extra_args="--reservoirs ${DEBUG_ONE_RESERVOIR} --policies ${DEBUG_ONE_POLICY} --picks \"${DEBUG_ONE_PICK}\""
    if [[ "${DEBUG_SKIP_DEFAULT_RUN}" == "1" ]]; then
      validate_extra_args="${validate_extra_args} --skip-default-run"
    fi
  fi
  echo "== Figures + validation: ${sub} (seed ${BORG_SEED_FILTERED}) =="
  if [[ "${CEE_FIG12_FIRST}" == "1" && "${DEBUG_VALIDATE_ONLY}" != "1" ]]; then
    echo "[phase] Fig1/Fig2 first pass (${sub})"
    CEE_BORG_SEED="${BORG_SEED_FILTERED}" CEE_BORG_MRF_FILTERED=1 CEE_MRF_FILTER_TAG="${filter_tag}" CEE_FIG_SUBDIR="${sub}" CEE_SKIP_PYWR=1 CEE_REMAKE_DYNAMICS_PLOTS=0 CEE_PYWR_FLOW_PREDICTION_MODE="${flow_mode}" python "${SCRIPT_DIR}/${FIG_SCRIPT}" ${FIG_SCRIPT_ARGS} "${fig_args[@]}" --skip-stage2 || true
  fi
  if [[ "${DEBUG_VALIDATE_ONLY}" != "1" ]]; then
    CEE_BORG_SEED="${BORG_SEED_FILTERED}" CEE_BORG_MRF_FILTERED=1 CEE_MRF_FILTER_TAG="${filter_tag}" CEE_FIG_SUBDIR="${sub}" CEE_PYWR_FLOW_PREDICTION_MODE="${flow_mode}" python "${SCRIPT_DIR}/${FIG_SCRIPT}" ${FIG_SCRIPT_ARGS} "${fig_args[@]}" || true
  else
    echo "[debug-fast] skipping ${FIG_SCRIPT}; running focused validate only."
    CEE_BORG_SEED="${BORG_SEED_FILTERED}" CEE_BORG_MRF_FILTERED=1 CEE_MRF_FILTER_TAG="${filter_tag}" CEE_FIG_SUBDIR="${sub}" CEE_PYWR_FLOW_PREDICTION_MODE="${flow_mode}" bash -lc "python -m methods.postprocess.figures_validation ${VALIDATE_ARGS} ${validate_extra_args}" || true
  fi
}

FIGURES_SPEC_FULL="${CEE_FIGURES_FULL:-${CEE_FIGURES_ALL}}"
FIGURES_SPEC_REGRESSION_DISAGG="${CEE_FIGURES_REGRESSION_DISAGG:-${CEE_FIGURES_ALL}}"
FIGURES_SPEC_PERFECT="${CEE_FIGURES_PERFECT:-${CEE_FIGURES_ALL}}"

if [[ "${RUN_FULL}" == "1" ]]; then
  echo "== Full-series Borg figures + validation (seed ${BORG_SEED_UNMASKED}) =="
  full_validate_extra_args=""
  full_figures_spec="${FIGURES_SPEC_FULL}"
  full_fig_args=""
  if [[ -n "${full_figures_spec}" ]]; then
    full_fig_args="--figures ${full_figures_spec}"
    echo "[figures] borg_full_series: ${full_figures_spec}"
  fi
  if [[ "${DEBUG_FAST}" == "1" ]]; then
    full_validate_extra_args="--reservoirs ${DEBUG_ONE_RESERVOIR} --policies ${DEBUG_ONE_POLICY} --picks \"${DEBUG_ONE_PICK}\""
    if [[ "${DEBUG_SKIP_DEFAULT_RUN}" == "1" ]]; then
      full_validate_extra_args="${full_validate_extra_args} --skip-default-run"
    fi
  fi
  if [[ "${DEBUG_VALIDATE_ONLY}" == "1" ]]; then
    # Single-line env prefix avoids fragile '\' continuations (trailing space / CRLF can break assignments).
    CEE_BORG_SEED="${BORG_SEED_UNMASKED}" CEE_BORG_MRF_FILTERED=0 CEE_FIG_SUBDIR=borg_full_series CEE_PYWR_FLOW_PREDICTION_MODE="${FLOW_MODE_FULL}" bash -lc "python -m methods.postprocess.figures_validation ${VALIDATE_ARGS} ${full_validate_extra_args}" || true
  elif [[ "${CEE_FIG12_FIRST}" == "1" && "${DEBUG_FAST}" != "1" ]]; then
    echo "[phase] Fig1/Fig2 first pass (borg_full_series)"
    CEE_BORG_SEED="${BORG_SEED_UNMASKED}" CEE_BORG_MRF_FILTERED=0 CEE_FIG_SUBDIR=borg_full_series CEE_SKIP_PYWR=1 CEE_REMAKE_DYNAMICS_PLOTS=0 CEE_PYWR_FLOW_PREDICTION_MODE="${FLOW_MODE_FULL}" python "${SCRIPT_DIR}/${FIG_SCRIPT}" ${FIG_SCRIPT_ARGS} ${full_fig_args} --skip-stage2 || true
    CEE_BORG_SEED="${BORG_SEED_UNMASKED}" CEE_BORG_MRF_FILTERED=0 CEE_FIG_SUBDIR=borg_full_series CEE_PYWR_FLOW_PREDICTION_MODE="${FLOW_MODE_FULL}" python "${SCRIPT_DIR}/${FIG_SCRIPT}" ${FIG_SCRIPT_ARGS} ${full_fig_args} || true
  else
    CEE_BORG_SEED="${BORG_SEED_UNMASKED}" CEE_BORG_MRF_FILTERED=0 CEE_FIG_SUBDIR=borg_full_series CEE_PYWR_FLOW_PREDICTION_MODE="${FLOW_MODE_FULL}" python "${SCRIPT_DIR}/${FIG_SCRIPT}" ${FIG_SCRIPT_ARGS} ${full_fig_args} || true
  fi
  echo "Full-series figures -> figures/borg_full_series/"
fi

if [[ "${RUN_REGRESSION_DISAGG}" == "1" ]]; then
  _run_fig_validate regression_disagg borg_mrffiltered_regression "${FLOW_MODE_REGRESSION_DISAGG}" "${FIGURES_SPEC_REGRESSION_DISAGG}"
  echo "MRF-filtered regression-disaggregation figures -> figures/borg_mrffiltered_regression/"
fi

if [[ "${RUN_PERFECT}" == "1" ]]; then
  _run_fig_validate perfect borg_mrffiltered_perfect_foresight "${FLOW_MODE_PERFECT}" "${FIGURES_SPEC_PERFECT}"
  echo "MRF-filtered perfect-foresight figures -> figures/borg_mrffiltered_perfect_foresight/"
fi

echo "Post-processing and figure generation complete."
