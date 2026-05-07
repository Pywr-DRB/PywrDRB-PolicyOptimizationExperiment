# PywrDRB Policy Optimization Experiment
Formerly known as the CEE project repo.

This repository contains the optimization, postprocessing, and figure-generation workflow for parametric reservoir release policies (STARFIT, RBF, PWL). It is organized so a new user can run the pipeline in order: preprocessing -> MRF filter preparation -> mmBorg optimization -> postprocessing/figures -> optional advanced analyses.

## Outline

- [Quick Start](#quick-start)
- [Installation and Dependencies](#installation-and-dependencies)
- [Repository Structure](#repository-structure)
- [Workflow (Run in Order)](#workflow-run-in-order)
- [Core Settings and Configuration Source of Truth](#core-settings-and-configuration-source-of-truth)
- [Advanced Workflows](#advanced-workflows)
- [Resources](#resources)

## Quick Start

```bash
git clone https://github.com/Pywr-DRB/PywrDRB-PolicyOptimizationExperiment.git
cd PywrDRB-PolicyOptimizationExperiment

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then follow the ordered commands in [Workflow (Run in Order)](#workflow-run-in-order).

## Installation and Dependencies

### Python

- Use Python 3.10+ (3.11 recommended).
- If your default Python is older, load/activate a newer Python before running scripts.

### Pywr-DRB source branch (required)

This workflow expects `pywrdrb` from branch `feature/release-policy-refactor`, because canonical optimization settings are sourced from `pywrdrb.release_policies.config`.

```bash
cd /path/to/Pywr-DRB
git fetch origin
git checkout feature/release-policy-refactor
pip install -e .
```

Verify in your active environment:

```bash
python - <<'PY'
import pywrdrb
from pywrdrb.release_policies import config
print("pywrdrb:", pywrdrb.__file__)
print("release_policies config:", config.__file__)
PY
```

### Borg/MMBorg runtime files

This repo expects Borg Python wrapper/shared libraries and MOEAFramework assets to be available in the repository root (for example from `BorgTraining` artifacts).

Expected files/folders include:

- `borg.py`
- `libborg.so`, `libborgms.so`, `libborgmm.so`
- `MOEAFramework-5.0/`
- `moeaframework/`

If you are setting up from `BorgTraining`, run:

```bash
cd /path/to/projects
git clone https://github.com/philip928lin/BorgTraining.git

cp ./BorgTraining/borg.py ./PywrDRB-PolicyOptimizationExperiment/
cp ./BorgTraining/libborg.so ./PywrDRB-PolicyOptimizationExperiment/
cp ./BorgTraining/libborgms.so ./PywrDRB-PolicyOptimizationExperiment/
cp ./BorgTraining/libborgmm.so ./PywrDRB-PolicyOptimizationExperiment/
cp -r ./BorgTraining/MOEAFramework-5.0 ./PywrDRB-PolicyOptimizationExperiment/
cp -r ./BorgTraining/moeaframework ./PywrDRB-PolicyOptimizationExperiment/
```

Quick check:

```bash
cd /path/to/PywrDRB-PolicyOptimizationExperiment
ls borg.py libborg.so libborgms.so libborgmm.so
ls MOEAFramework-5.0 moeaframework
```

If the shared libraries are missing or incompatible on your system, compile/rebuild Borg for your environment before running optimization.

## Repository Structure

Top-level layout (folder-focused):

- `methods/` - core Python package
  - `methods/preprocessing/` - MRF filtering preparation utilities
  - `methods/postprocess/` - baseline metrics + stage 1/2 figure orchestration
  - `methods/figures_stage3/` - full-Pareto manifest figure pipeline
  - `methods/ensemble/` - batch simulation/manifests/full-Pareto MPI runner
  - `methods/analysis/` - analysis helpers (aggregation, epsilon nondominated, migration)
  - `methods/load/` - data/results loading
  - `methods/plotting/` - plotting primitives used by stage pipelines
  - `methods/config.py` - local adapter/wrapper around canonical config
- `obs_data/` - observational inputs (`raw`, `processed`, `pub_reconstruction`)
- `outputs/` - Borg outputs and summary artifacts
- `figures/` - generated figures
- `pywr_data/` - local-only Pywr JSON/HDF5 artifacts and caches for postprocessing/figure workflows (ignored by git)
- `preprocessing_outputs/` - mode-specific filtering inputs for optimization + local preprocessing Pywr traces
- `moeaframework/` - MOEA workflow scripts/helpers
- `old/` - archived scripts kept for reference only

Primary entrypoint scripts:

- `run_preprocessing.sh`
- `scripts/prepare_preprocessing_outputs.sh`
- `build_mrf_filtering_folder.sh`
- `run_parallel_mmborg.sh`
- `run_parallel_mmborg_multiseed.sh`
- `run_postprocessing_and_figures.sh`
- `run_full_pareto_pywr_mpi.sh`
- `03_parallel_borg_run.py`
- `04_make_figures.py`

## Workflow (Run in Order)

### Stage definitions

- Stage 1: optimization-processing figures (no simulation).
- Stage 2: simulation + validation figures.
- Stage 3: full-Pareto manifest figures (separate pipeline).

### 1) Preprocess observational data

```bash
sbatch run_preprocessing.sh
```

`run_preprocessing.sh` runs `02_process_data.py` with inflow mode `pub_only` by default.
Default behavior (recommended) is public inflow for all reservoirs:

```bash
sbatch run_preprocessing.sh
```

Other valid values:
- `pub_only`
- `observed_with_bluemarsh_pub` (use observed where available, but always force `blueMarsh` from `inflow_pub`)

Preprocessing inflow source settings (via `CEE_INFLOW_SOURCE_MODE`):

```bash
# Default/recommended: use inflow_pub for all reservoirs
CEE_INFLOW_SOURCE_MODE=pub_only sbatch run_preprocessing.sh

# Mixed mode: observed where available, but force blueMarsh from inflow_pub
CEE_INFLOW_SOURCE_MODE=observed_with_bluemarsh_pub sbatch run_preprocessing.sh
```

Reproducible local helper (runs only when missing, unless forced):

```bash
bash scripts/prepare_preprocessing_outputs.sh
```

Force full regeneration of both MRF modes:

```bash
FORCE_MRF_FILTER_BUILD=1 bash scripts/prepare_preprocessing_outputs.sh
```

### 2) Build MRF filtering bundles

```bash
bash build_mrf_filtering_folder.sh
```

This creates filtering assets under:

- `preprocessing_outputs/filtering/regression_disagg/`
- `preprocessing_outputs/filtering/perfect_foresight/`

Each bundle includes:

- `lower_basin_mrf_contributions.csv` (tracked)
- `mrf_active_filter_daily.csv` (tracked)

### Preprocessing output contract (for reproducibility + clean pushes)

- Commit: CSV/PNG outputs needed by downstream scripts and figures.
- Do not commit: generated model `.json`/`.hdf5` artifacts.
- Local-only MRF preprocessing model artifacts are written under `preprocessing_outputs/pywr/`.
- `pywr_data/` is still used by postprocessing/figure pipelines and full-Pareto simulation workflows.
- Filtering outputs live only under `preprocessing_outputs/filtering/{regression_disagg,perfect_foresight}/`.
- If legacy folders like `preprocessing_outputs/contributions/` appear in local clones, treat them as archival and remove them.

### 3) Run mmBorg optimization

Default phase sweep:

```bash
sbatch run_parallel_mmborg.sh
```

Defaults:

- `CEE_BORG_MODES=full,regression,perfect`
- full-series seed: `BORG_SEED_UNFILTERED=72`
- filtered seed: `BORG_SEED_FILTERED=71`

Common variants:

```bash
# subset of phases
CEE_BORG_MODES=regression,perfect sbatch run_parallel_mmborg.sh

# multiseed sweep
CEE_MULTISEED_FROM=1 CEE_MULTISEED_TO=10 sbatch run_parallel_mmborg_multiseed.sh
```

Filtered output naming:

- `*_mrffiltered_regression` for regression-disaggregation filtering
- `*_mrffiltered_perfect` for perfect-information filtering

### 4) Run postprocessing and figures (stages 1-2)

```bash
sbatch run_postprocessing_and_figures.sh
```

Optional: run selected bundle only.

```bash
CEE_POSTPROCESS_BUNDLE=full sbatch run_postprocessing_and_figures.sh
# also supports: regression_disagg, perfect, all
```

### 5) Optional stage 3 full-Pareto workflow

```bash
sbatch run_full_pareto_pywr_mpi.sh
```

Or manual plotting after manifest exists:

```bash
python -m methods.figures_stage3.plot_stage3_full_pareto_figures \
  --manifest pywr_data/full_pareto_runs/_full_pareto_manifest.json \
  --out-dir figures \
  --which all
```

### Quick checks

```bash
git status
ls outputs | rg "MMBorg_"
ls figures
```

## Core Settings and Configuration Source of Truth

Canonical optimization settings live in:

- `pywrdrb.release_policies.config` (from branch `feature/release-policy-refactor`)

This repository's `methods/config.py` is a local adapter that re-exports canonical settings and adds local path helpers/wrappers.

Settings commonly adjusted for experiments include:

- `SEED`, `NFE`, `ISLANDS`
- objective sets and epsilon values
- policy parameter bounds
- reservoir context/capacity mappings
- normal operating range helper mapping (`NORMAL_OPERATING_RANGE_BY_RESERVOIR`)

Primary workflow flags:

- Optimization:
  - `CEE_BORG_MODES`
  - `CEE_BORG_SINGLE_PHASE`
  - `USE_MRF`
  - `CEE_MRF_FILTER_SOURCE`
  - `BORG_SEED_UNFILTERED`, `BORG_SEED_FILTERED`
- Postprocessing:
  - `CEE_POSTPROCESS_BUNDLE`
  - `FLOW_MODE_FULL`, `FLOW_MODE_REGRESSION_DISAGG`, `FLOW_MODE_PERFECT`
  - `BORG_SEED_UNFILTERED`, `BORG_SEED_FILTERED`

## Advanced Workflows

### A) Epsilon nondominated analysis

```bash
python -m methods.analysis.mmborg_eps_nondominated_set --help
python -m methods.analysis.plot_eps_nondominated_figures --help
```

### B) Collect all Borg CSVs into one analysis table

```bash
python -m methods.analysis.collect_borg_csvs_for_analysis -o outputs/analysis_all_borg_runs.csv
```

### C) MOEAFramework metrics pipeline

```bash
cd moeaframework
python organize_borg_outputs.py --src ../outputs --dst ./outputs
bash run_moea_workflow.sh
```

### D) Post-optimization simulation bundles

```bash
python -m methods.ensemble.postprocess_sim simulate --help
```

## Resources

- [Pywr-DRB Repository](https://github.com/Pywr-DRB/Pywr-DRB)
- [BorgTraining Repository](https://github.com/philip928lin/BorgTraining)
- [WaterProgramming Borg + Python wrapper guide](https://waterprogramming.wordpress.com/2025/02/19/everything-you-need-to-run-borg-moea-and-python-wrapper-part-2/)

