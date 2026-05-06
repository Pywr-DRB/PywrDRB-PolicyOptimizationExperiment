# CEE6400Project
Marilyn & Trevor's course project for CEE6400 Spring 2025

Parametric reservoir policies (STARFIT, RBF, PWL) are optimized with MMBorgMOEA and validated in both a standalone reservoir simulator and Pywr-DRB. This repo provides a reproducible workflow from data prep -> optimization -> **optional** post-opt scenario runs (`python -m methods.ensemble.postprocess_sim`) -> figures (`04_make_figures.py`) and validation.

## What Changed (Current Repo State)

This README now reflects the refactor that moved most reusable logic from root scripts into `methods/*` modules and SLURM driver scripts.

- Figure and validation internals now live under `methods/postprocess/` and are orchestrated by `04_make_figures.py` (figures **1–13**); full-Pareto manifest figures (**14–23**) live under `methods/figures_stage3/`.
- Post-optimization simulation and manifest workflows live under `methods/ensemble/` (including `postprocess_sim` and MPI-based full-Pareto runs).
- Analysis utilities for aggregation, epsilon-nondominated filtering, and migration helpers live under `methods/analysis/`.
- Root-level legacy scripts are kept in `old/` for reference; active workflows use the current module paths in this README.
- Standard cluster entrypoints are `run_parallel_mmborg.sh`, `run_parallel_mmborg_multiseed.sh`, `build_mrf_masking_folder.sh`, and `run_postprocessing_and_figures.sh`.

## Repo Structure (key files)

| Path / File | Description |
|-------------|-------------|
| `01_retrieve_data.py` | Retrieve observational reservoir data |
| `02_process_data.py` | Process raw -> model-ready data |
| `03_parallel_borg_run.py` | Run MMBorgMOEA optimization (single policy/reservoir) |
| `04_make_figures.py` | **Figure pipeline** (step 04): stage **1** (**1–6**, optimization processing only), stage **2** (**7–11**, simulation + validation); `--figures`, `--plots-only`, `--skip-stage1` / `--skip-stage2`. |
| `methods/ensemble/postprocess_sim.py` | **Step 05** CLI entrypoint for post-opt batch sims (no matplotlib): independent reservoir CSV + Pywr scenario/selected HDF5s + JSON manifests (`python -m methods.ensemble.postprocess_sim simulate --help`). |
| `methods/postprocess/figures_primary.py` | Primary-stage implementation (Pareto, parallel axes, dynamics, …); was `methods/figure_pipeline/stage1.py`. |
| `methods/postprocess/figures_validation.py` | Validation figures (former `05_validate.py`); was `stage2.py`. CLI: `python -m methods.postprocess.figures_validation`. |
| `methods/figures_stage3/full_pareto_output_paths.py` | Paths for figures **14–23** (full-Pareto manifest) under each `figures/borg_*` bundle folder. |
| `methods/ensemble/` | Post-opt simulation helpers: solution IDs (`POLICY_rowIndex`), manifests, Parquet/CSV/HDF5 bundle loaders, `run_pareto_simulations` logic. |
| `methods/figure_pipeline/` | **Deprecated** shims only — import from `methods.postprocess` / `methods.ensemble` instead. |
| `old/` | Archived snapshots of earlier root scripts (reference only) |
| `run_preprocessing.sh` | SLURM script for data prep |
| `run_parallel_mmborg.sh` | SLURM script for multi-policy x multi-reservoir optimization sweep |
| `run_parallel_mmborg_multiseed.sh` | SLURM script: same **full / regression / perfect** phases as `run_parallel_mmborg.sh`, with loop seeds `CEE_MULTISEED_FROM`–`CEE_MULTISEED_TO` (default 1–10) |
| `build_mrf_masking_folder.sh` | MRF masking: pub from existing HDF5 or Pywr sim; perfect **HDF5 only** (no sim). Run before `run_parallel_*` when `USE_MRF=true` |
| `run_postprocessing_and_figures.sh` | SLURM script: baselines, summaries, figs 1-5 for full-series + regression-filtered + perfect-filtered runs |
| `methods/preprocessing/build_mrf_active_masks.py` | Pywr-DRB helper used by `build_mrf_masking_folder.sh` (contributions CSV + active-range JSON per bundle) |
| `methods/postprocess/summarize_optimization.py` | Counts Borg solutions before/after objective filtering (uses same env as figures) |
| `methods/postprocess/plot_baseline_dynamics.py` | Observed vs default Pywr series for baseline metric windows |
| `methods/config.py` | Local path/compatibility layer that re-exports canonical settings from `pywrdrb.release_policies.config` |
| `methods/borg_paths.py` | Resolves `outputs/MMBorg_*.csv` paths from `CEE_BORG_*` / `CEE_MRF_*` environment variables |
| `methods/reservoir/model.py` | Standalone Reservoir model implementation |
| `methods/load/` | Loaders for results and observations |
| `methods/plotting/` | Figure builders (Pareto, axes, dynamics, 9-panel, errors) |
| `obs_data/{raw,processed,pub_reconstruction}` | Observational data inputs |
| `outputs/` | Raw BORG CSVs (MRF-filtered runs use `_mrffiltered_regression` or `_mrffiltered_perfect`; legacy `*_mrfmasked_*` is still read by path resolution) |
| `preprocessing_outputs/masking/` | `pub_reconstruction/` and `perfect_information/` (MRF CSV + JSON for masking) |
| `figures/` | Generated figures; e.g. `borg_full_series/`, `borg_mrffiltered_regression/`, `borg_mrffiltered_perfect_foresight/` |
| `logs/` | SLURM logs |
| `borg.py`, `libborg*.so`, `MOEAFramework-5.0/`, `moeaframework/` | Borg Python wrapper + MOEAFramework Java dependency |
| `moeaframework/organize_borg_outputs.py` | Copy `outputs/*.runtime` into `Policy_*/runtime/<reservoir>/` for MOEA tools |
| `moeaframework/run_moea_workflow.sh` | Four-step MOEA CLI: .set, headers, refsets, metrics |
| `methods/analysis/collect_borg_csvs_for_analysis.py` | Stack all `MMBorg_*.csv` into one table (`mrf_filtered`, `mrf_filter_variant`, …) |
| `methods/analysis/migrate_mrfmasked_outputs_to_mrffiltered.py` | Rename legacy `*_mrfmasked_*` Borg outputs to `*_mrffiltered_*` |
| `methods/analysis/mmborg_eps_nondominated_set.py` | ε-nondominated subsets of `MMBorg_*.csv` via PyPI `pareto` — objectives/ε from `pywrdrb.release_policies.config`; per-reservoir pool (`--per-reservoir`) or merged (`--csv` / `--resolve`) |
| `methods/analysis/plot_eps_nondominated_figures.py` | Fig 1–2 style plots from `eps_nondominated_*.csv` (policy colors aligned with stage 1) |
| `methods/ensemble/run_full_pareto_pywr_mpi.py` | MPI: run all aligned filtered-Pareto rows through Pywr-DRB; outputs under `pywr_data/full_pareto_runs/`, merged **`_full_pareto_manifest.json`** |
| `methods/figures_stage3/plot_stage3_full_pareto_figures.py` | After full-Pareto MPI: figures **14–23** (`fig14_*` … `fig23_*`) under each `figures/borg_*` folder — HDF5-backed multipanels + diagnostics from `_full_pareto_manifest.json`. |
| `methods/figures_stage3/` | Loaders + `run_stage3_full_pareto_analysis` (multipanels + `advanced_plots`) — exports `aggregate_multipanel_daily_from_manifest`, `aggregate_stage3_multipanels_from_manifest` |
| `requirements.txt` | Python dependencies |

## Resources
- [BorgTraining (GitHub)](https://github.com/philip928lin/BorgTraining)
- [Everything You Need to Run Borg MOEA and Python Wrapper - Part 2 (WaterProgramming)](https://waterprogramming.wordpress.com/2025/02/19/everything-you-need-to-run-borg-moea-and-python-wrapper-part-2/)

---

## Workflow

### Quick start (fresh clone on Hopper)

Run these in order for an end-to-end reproduction:

```bash
cd /path/to/projects
git clone https://github.com/Pywr-DRB/PywrDRB-PolicyOptimizationExperiment.git
cd PywrDRB-PolicyOptimizationExperiment

module load python/3.11.5
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1) Observed-data preprocessing
sbatch run_preprocessing.sh

# 2) Build MRF masking bundles (pub + perfect JSON/CSV artifacts)
bash build_mrf_masking_folder.sh

# 3) Optimization (default phases: full, regression-filtered, perfect-filtered)
sbatch run_parallel_mmborg.sh

# 4) Postprocess + figures
sbatch run_postprocessing_and_figures.sh

# 5) Optional full-Pareto MPI + stage-3 figures
sbatch run_full_pareto_pywr_mpi.sh
```

Useful variants:

```bash
# Only selected optimization phases
CEE_BORG_MODES=regression,perfect sbatch run_parallel_mmborg.sh

# Multiseed sweeps
CEE_MULTISEED_FROM=1 CEE_MULTISEED_TO=10 sbatch run_parallel_mmborg_multiseed.sh

# Restrict postprocess to one bundle
CEE_POSTPROCESS_BUNDLE=full sbatch run_postprocessing_and_figures.sh
```

Quick checks:

```bash
squeue -u $USER
ls outputs | rg "MMBorg_"
ls figures
```

### Stage definitions

- **Stage 1 (optimization processing only):** uses optimization outputs only; no simulation runs.
- **Stage 2 (simulation + validation):** runs/caches simulations and produces validation figures.
- **Figures 14–23 (full-Pareto diagnostics):** generated **after** the MPI sweep via `methods/figures_stage3/plot_stage3_full_pareto_figures.py` (not `04_make_figures.py`).

### Environment prerequisites

**Python version:** use **3.10+** (3.11 recommended). The login-node default `python` on some clusters is older; if you see `future feature annotations is not defined` or similar syntax errors, load a modern module and use your venv before running `04_make_figures.py` or `python -m methods.ensemble.postprocess_sim`.

**Important:** this workflow expects `pywrdrb` from the **`release-policy`** branch, because core settings are sourced from `pywrdrb.release_policies.config`.

Quick check in your active environment:

```bash
python - <<'PY'
import pywrdrb
from pywrdrb.release_policies import config
print("pywrdrb:", pywrdrb.__file__)
print("release_policies config:", config.__file__)
PY
```

On Hopper:

```bash
module load python/3.11.5
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Borg/MOEA runtime prerequisites

Rather than compiling MMBorgMOEA from scratch, these instructions copy precompiled `libborg*.so` files from the `BorgTraining` (private) repo. If you encounter errors, you may need to clone `MMBorgMOEA` and recompile, following the WaterProgramming guide.

Copy required files into `CEE6400Project`:

```bash
cp ./BorgTraining/borg.py ./CEE6400Project/
cp -r ./BorgTraining/MOEAFramework-5.0/ ./CEE6400Project/
cp -r ./BorgTraining/moeaframework/ ./CEE6400Project/
cp ./BorgTraining/libborg.so ./CEE6400Project/
cp ./BorgTraining/libborgms.so ./CEE6400Project/
cp ./BorgTraining/libborgmm.so ./CEE6400Project/
```

## Running MMBorgMOEA

The following are designed for Hopper.

```bash
cd ./CEE6400Project
```

`03_parallel_borg_run.py` executes MMBorgMOEA for one `POLICY_TYPE` and one `RESERVOIR_NAME`.

Submit the multi-reservoir x multi-policy sweep with `run_parallel_mmborg.sh`.

### MRF masking folder and optimization filenames

**Recommended workflow:** run **`build_mrf_masking_folder.sh`** once (after `module load` / venv) so the bundle you need exists (`pub_reconstruction` and/or `perfect_information`). When **`USE_MRF=true`**, each `run_parallel_*` job checks that the JSON it will use (from **`MRF_RANGES_JSON`** or **`CEE_MRF_FILTER_SOURCE`**) is present and **exits immediately** if not. **`USE_MRF=false`** skips that check.

Bundles:

- `preprocessing_outputs/masking/pub_reconstruction/` — Pywr-DRB with **pub reconstruction** inflows; `lower_basin_mrf_contributions.csv`, `lower_basin_mrf_active_ranges.json`, daily mask CSV.
- `preprocessing_outputs/masking/perfect_information/` — masking derived from a **perfect-information Pywr HDF5** on disk only (`build_mrf_masking_folder.sh` never simulates this bundle).

Customize the build with env vars in `build_mrf_masking_folder.sh`. **Pub:** if an HDF5 already exists (e.g. `pywrdrb_output_<PUB_INFLOW>.hdf5` in the project root or `pywr_data/_pywr_default_cache/output_default_<start>_<end>_<PUB_INFLOW>.hdf5`), the script only extracts and writes masking files; otherwise it runs Pywr. Set **`FORCE_MRF_PYWR_SIM=1`** or **`MRF_FORCE_PUB_SIM=1`** to ignore existing pub HDF5 and simulate. **`SKIP_PUB_SIM=1`** requires a pub HDF5 or exits. **Perfect information:** the script **never** runs Pywr — you must supply `pywr_data/_pywr_perfect_information/<PERFECT_INFLOW>.hdf5` (or `PERFECT_HDF5=...`). Other knobs: `SKIP_PERFECT_MRF_PREBUILD=1`, `PUB_INFLOW`, `PERFECT_INFLOW`, `MRF_BUILD_START` / `MRF_BUILD_END` (pub Pywr sim only).

When **`USE_MRF=true`**, Borg writes CSVs/runtime files with an explicit **`_mrffiltered_*`** suffix so regression (pub-reconstruction JSON) and perfect filtered runs do not overwrite each other:

- `..._seed{N}_mrffiltered_regression.*` — objectives filtered using the **pub reconstruction** MRF JSON (`CEE_MRF_FILTER_SOURCE=regression_disagg` by default in the Slurm drivers).
- `..._seed{N}_mrffiltered_perfect.*` — objectives filtered using the **perfect-information** JSON (`CEE_MRF_FILTER_SOURCE=perfect`).

Full-series runs omit the `_mrffiltered_*` token and use `BORG_SEED_UNMASKED` (default 72). The Slurm scripts set **`CEE_MRF_FILTER_TAG`** (e.g. `regression_disagg` / `perfect`) for filtered jobs so `methods/borg_paths.py` resolves the correct file.

### Common submission commands (filtered/full-series, single/multiseed)

**Default (recommended):** both `run_parallel_mmborg.sh` and `run_parallel_mmborg_multiseed.sh` run **three phases** in order — **full** (full-series), **regression** (pub-reconstruction MRF), **perfect** — unless you override `CEE_BORG_MODES`.

```bash
# Single-seed defaults: seed 72 full-series, 71 for each MRF-filtered phase
sbatch run_parallel_mmborg.sh

# Multiseed defaults: seeds 1–10 per phase (override with CEE_MULTISEED_FROM / CEE_MULTISEED_TO)
sbatch run_parallel_mmborg_multiseed.sh

# Subset or reorder phases, e.g. only regression + perfect:
CEE_BORG_MODES=regression,perfect sbatch run_parallel_mmborg.sh
```

**Legacy single-mode sweep** (one `USE_MRF` / JSON choice for the whole job): set **`CEE_BORG_SINGLE_PHASE=1`** and use `USE_MRF`, `CEE_MRF_FILTER_SOURCE`, and optional `MRF_RANGES_JSON`.

```bash
# Example: regression-filtered bundle only, single-seed driver
CEE_BORG_SINGLE_PHASE=1 USE_MRF=true CEE_MRF_FILTER_SOURCE=regression_disagg \
  sbatch -o logs/masked_regr_%j.out -e logs/masked_regr_%j.err run_parallel_mmborg.sh

# Example: multiseed loop with one mode only
CEE_BORG_SINGLE_PHASE=1 USE_MRF=false sbatch run_parallel_mmborg_multiseed.sh
```

Check queue status:

```bash
squeue -u ms3654
```

The following are available through `methods/config.py` (re-exported from `pywrdrb.release_policies.config`, except local path helpers):
- Metrics
- Epsilon values
- Parameter bounds
- Seed number (default/fallback)

## Post-processing and Figures

After optimization completes, generate Pareto comparisons, parallel axes, dynamics, validation figures, baseline plots, and solution-count summaries:

```bash
sbatch run_postprocessing_and_figures.sh
```

Borg result CSVs are resolved by **`methods/borg_paths.py`** using:

- `CEE_BORG_SEED` (or `CEE_SEED`) — must match the sweep (`BORG_SEED_FILTERED` or `BORG_SEED_UNMASKED` in `run_parallel_mmborg.sh` and `run_postprocessing_and_figures.sh`, defaults 71 / 72).
- `CEE_BORG_MRF_FILTERED` — `1` for MRF-filtered objectives, `0` for full-series.
- `CEE_MRF_FILTER_TAG` or `CEE_MRF_FILTER_SOURCE` — user-facing names such as **`regression_disagg`** or **`perfect`** map to on-disk `*_mrffiltered_regression.csv` / `*_mrffiltered_perfect.csv` (see `methods/borg_paths.py`).
- `CEE_FIG_SUBDIR` — subfolder under `figures/` (e.g. `borg_mrffiltered_regression`) so runs do not overwrite each other.
- `CEE_PYWR_WORK_DIR` — Pywr JSON + parametric HDF5 for stage 1 (Figs 4–6) and stage 2 (7–11); shared cache naming (default `pywr_data/pywr_tmp_runs`). Legacy `CEE_PYWR_PICK_HDF5_DIR` is unused for parametric runs after the unified cache change.

**`run_postprocessing_and_figures.sh` bundle choice** — optional; default runs all three Borg result sets. Set **`CEE_POSTPROCESS_BUNDLE`** to limit work (overrides `DEBUG_FAST`’s RUN flags when set).

**Two separate knobs:**

1. **Which Borg CSV / MRF bundle** — `CEE_POSTPROCESS_BUNDLE` tokens:
   - `full` — full-series objectives (`CEE_BORG_MRF_FILTERED=0`, seed `BORG_SEED_UNMASKED`) → `figures/borg_full_series/`
   - **`regression_disagg`** — regression-disaggregation MRF-filtered Borg outputs → `figures/borg_mrffiltered_regression/` (aliases: `mrffiltered_regression`, `borg_mrffiltered_regression`). This is the bundle name; it matches the usual Pywr mode for that pass, not the on-disk Borg filename suffix.
   - `perfect` — perfect-foresight MRF-filtered Borg outputs → `figures/borg_mrffiltered_perfect_foresight/` (aliases: `mrffiltered_perfect`, `borg_mrffiltered_perfect_foresight`)
   - `all` — all three. Comma-separated runs multiple, e.g. `CEE_POSTPROCESS_BUNDLE=regression_disagg,perfect` or `full,regression_disagg`

2. **Pywr flow prediction mode** (`CEE_PYWR_FLOW_PREDICTION_MODE` in each pass) — in the shell script: **`FLOW_MODE_FULL`**, **`FLOW_MODE_REGRESSION_DISAGG`**, **`FLOW_MODE_PERFECT`** (defaults: `regression_disagg`, `regression_disagg`, `perfect_foresight`). These modes are independent of which MRF-filtered Borg CSV you select.

For **manual** `python 04_make_figures.py` calls, set e.g. `CEE_MRF_FILTER_TAG=regression_disagg` or `CEE_MRF_FILTER_TAG=perfect`; **`regression_disagg`** resolves to `*_mrffiltered_regression.csv`.

**Fig 4–6 (dynamics + policy surfaces) narrowing** — optional; leave unset for the full grid. Subsets the stage-1 loop so you can iterate on layout without regenerating every reservoir/pick/k:

- `CEE_FIG4_RESERVOIRS` (legacy: `CEE_FIG3_RESERVOIRS`) — comma-separated reservoir names (must match `reservoir_options`).
- `CEE_FIG4_POLICIES` (legacy: `CEE_FIG3_POLICIES`) — comma-separated subset of `CEE_FIGURE_POLICIES` (e.g. `PWL`).
- `CEE_FIG4_PICKS` (legacy: `CEE_FIG3_PICKS`) — comma-separated subset of `CEE_DESIRED_PICKS` labels.
- `CEE_FIG4_K` (legacy: `CEE_FIG3_K`) — single solution index `k` (1-based, per pick), not all aligned rows.

**Fig 1 (Pareto scatter)** — `CEE_FIG1_INCLUDE_BASELINE` (alias `CEE_FIG1_DEFAULT_POINT`): unset / `both` writes two PNGs (`*_with_pywr_default.png`, `*_no_pywr_default.png`); `with` / `without` writes only one (useful when the default-operation marker skews the axis scale).

**Fig 4–5 evaluation window** — `CEE_EVAL_START` / `CEE_EVAL_END` (YYYY-MM-DD, default `1980-01-01`–`2018-12-31`) slice the time series used for annual-aggregation Fig 4 and the 9-panel Fig 5; both the plot titles and output filenames include this range.

**`run_postprocessing_and_figures.sh`** runs, in order:

1. **`methods/postprocess/build_default_timeseries.py`** only if the default Pywr HDF5 cache is missing under `pywr_data/_pywr_default_cache/`. Otherwise it logs a skip. Set **`CEE_FORCE_DEFAULT_RERUN=1`** when submitting to always rebuild the default HDF5 baseline. Then **`methods/postprocess/compute_baseline_metrics.py`**, **`methods/postprocess/plot_baseline_dynamics.py`**
2. **`methods/postprocess/summarize_optimization.py`** (when each bundle is selected): MRF-filtered **regression-disaggregation**, MRF-filtered **perfect-foresight**, and full-series counts under `outputs/`
3. **`04_make_figures.py`** (runs stages 1+2) for **full-series** Borg outputs → `figures/borg_full_series/`
4. Same for **MRF-filtered regression-disaggregation** Borg CSVs → `figures/borg_mrffiltered_regression/`
5. Same for **MRF-filtered perfect-foresight** Borg CSVs → `figures/borg_mrffiltered_perfect_foresight/`

You need matching `outputs/MMBorg_*_mrffiltered_regression*.csv` and `..._mrffiltered_perfect*.csv` on disk (or run `methods/analysis/migrate_mrfmasked_outputs_to_mrffiltered.py` on older `*_mrfmasked_*` files).

Manual one-off examples from `CEE6400Project/`:

```bash
# All figures — MRF-filtered regression-disaggregation bundle, seed 71 (`*_mrffiltered_regression.csv`)
CEE_BORG_SEED=71 CEE_BORG_MRF_FILTERED=1 CEE_MRF_FILTER_TAG=regression_disagg CEE_FIG_SUBDIR=borg_mrffiltered_regression python 04_make_figures.py

# All figures — MRF-filtered perfect-foresight bundle
CEE_BORG_SEED=71 CEE_BORG_MRF_FILTERED=1 CEE_MRF_FILTER_TAG=perfect CEE_FIG_SUBDIR=borg_mrffiltered_perfect_foresight python 04_make_figures.py

# All figures — full series (seed 72)
CEE_BORG_SEED=72 CEE_BORG_MRF_FILTERED=0 CEE_FIG_SUBDIR=borg_full_series python 04_make_figures.py

# Pareto + parallel axes + parameter ranges — **figures 1–3** need no reservoir or Pywr simulation
CEE_FIG_SUBDIR=borg_full_series python 04_make_figures.py --figures 1 2 3 --skip-stage2

# Perfect-foresight Borg bundle + unified “best all objectives” pick + dynamics/validation (match Pywr flow mode)
CEE_BORG_SEED=71 CEE_BORG_MRF_FILTERED=1 CEE_MRF_FILTER_TAG=perfect CEE_FIG_SUBDIR=borg_mrffiltered_perfect_foresight \
  CEE_DESIRED_PICKS="Normalized equal-weight mean optimum" CEE_PYWR_FLOW_PREDICTION_MODE=perfect_foresight \
  python 04_make_figures.py --figures 1-6 7-11
```

That batch script also runs the steps above; figures include:

- **Figs 1–2:** Pareto front and parallel axes — **filtered Borg MOEA CSVs only** (objectives / parameters); no reservoir or Pywr simulation is required for these panels.
- **Fig 3:** Parameter / objective range bars from optimization (`fig3_parameter_ranges/`) — **no simulation** (runs immediately after loading Borg tables).
- **Fig 4:** Annual-aggregation comparison in two columns — independent reservoir model (left) vs Pywr-DRB parametric (right), same evaluation window (`CEE_EVAL_START` / `CEE_EVAL_END`) in the title and filename (`fig4_dynamics/fig4_*_eval_*_annual_agg_independent_vs_pywr.png`).
- **Fig 5:** 9-panel temporal aggregation evaluation (daily / monthly / annual overlays; Pywr parametric + independent + default + observed) in `fig5_temporal_aggregation_evaluation/`; filenames include the evaluation window (`CEE_EVAL_START` / `CEE_EVAL_END`).
- **Fig 6:** Policy surfaces in `fig6_policy_surfaces/`.
- **Figs 7–8, 10–11:** Validation stage — `fig7_validation_dynamics`, `fig8_error_vs_flow_percentile`, `fig10_lower_basin_context`, `fig11_trenton_flow` — see `methods/postprocess/figures_validation.py`.
- **Figs 14–23:** Full-Pareto HDF5 manifest suite — under each bundle folder as `fig14_full_pareto_*` … `fig23_full_pareto_*` (see `methods/figures_stage3/full_pareto_output_paths.py`); run `plot_stage3_full_pareto_figures.py` after `_full_pareto_manifest.json` exists.

### Policy batch manifest (structural runs + hydrologic scenarios)

Wide CSV format documented in `methods/ensemble/policy_manifest.py`. Each row builds one `release_policy_dict` (four MOEA reservoirs) and runs one full-basin Pywr job. Optional column `inflow_ensemble_indices` (legacy name; comma-separated ints) turns on native multi-scenario inflows; output stems include a slug of those indices so caches do not collide.

- Driver: `python -m methods.ensemble.run_policy_manifest --manifest <path> [--dry-run] [--limit N] [--pywr-inflow-type ...] [--pywr-start ... --pywr-end ...]`
- **Flow mode:** non-MRF Borg rows default to `perfect_foresight` in Pywr; MRF regression bundle → `regression_disagg`; MRF `perfect` → `perfect_foresight` (override with `flow_prediction_mode` when consistent).
- Optional multi-scenario inflows: set `CEE_INFLOW_ENSEMBLE_INDICES=0,1,...` when running ensemble-style Pywr setups (stem includes a scenario slug).

Load all Pywr `scenario_id` slices from one HDF5 via `parametric_results_all_scenarios_from_h5` in `methods/postprocess/pywr_parametric_run.py`.

### Post-optimization simulations (cached runs)

After optimization, batch the **independent reservoir model** and **Pywr-DRB** for selected picks (and optional scenario pickles for custom workflows), then plot from disk:

1. Run **`python -m methods.ensemble.postprocess_sim`** — writes long-format indie CSV + `picks_manifest.json`, optional Pywr HDF5 manifests, optional scenario-bundle pickle. Example:  
   `python -m methods.ensemble.postprocess_sim simulate --mode all`  
   or individually: `--mode indie`, `--mode selected`, `--mode ensemble --output outputs/pareto_ensemble_pywr.pkl` (`ensemble` is the legacy CLI token for scenario mode).

**Solution IDs:** each optimized row is keyed as `{POLICY_TYPE}_{row_index}` (Borg CSV row label after filtering), matching `picks_manifest.json` and the indie CSV columns.

| Figures | Primary data source |
|---------|---------------------|
| **1–2** | Filtered **Borg MOEA CSVs** (objectives / parameters); **no simulation** |
| **3** | Optimization tables only (`solution_vars`); **no simulation** |
| **4–6** | Cached **independent** and/or **Pywr** outputs (see manifests under `outputs/postprocess/`). |
| **7–11** | Validation stage (`figures_validation`); prefers cached Pywr HDF5 when **`CEE_PLOTS_ONLY`** / **`CEE_SKIP_SIMULATIONS`** |
| **14–23** | Full-Pareto manifest (`_full_pareto_manifest.json` + HDF5); `methods/figures_stage3/` — **not** produced by `04_make_figures.py` |

---

## ε-Nondominated subsets (Borg MOEA CSVs)

After optimization, you can reduce each Borg bundle to an **ε-nondominated** set in objective space using the **`pareto`** package (Woodruff & Herman). Objectives and **ε** values always follow **`pywrdrb.release_policies.config`** (`EPSILONS`, `MOEA_OBJECTIVE_CSV_KEYS`, minimized `obj*` on disk).

**Install:** `pareto` is listed in **`requirements.txt`** (`pareto>=1.1.1`). If needed alone:

```bash
pip install pareto
```

**Per reservoir, pooling STARFIT + PWL + RBF** — one ε-sort per reservoir across all policies (writes `moea_policy`):

```bash
for variant in full regression perfect; do
  python -m methods.analysis.mmborg_eps_nondominated_set \
    --per-reservoir \
    --out-dir "outputs/pareto_eps_nondominated_${variant}" \
    --borg-variant "$variant" \
    --print-counts
done
```

Outputs: `outputs/pareto_eps_nondominated_<variant>/eps_nondominated_<reservoir>.csv`. Seeded / MRF filenames are resolved like `04_make_figures` (see `methods/borg_paths.py`). Override seeds with e.g. `CEE_BORG_SEED`, `CEE_BORG_SEED_FULL`, `CEE_BORG_SEED_REGRESSION`, `CEE_BORG_SEED_PERFECT` as needed.

**Figures (Release vs Storage NSE + optional 4-axis parallel plot)** — uses the same `load_results` / `load_results_with_metadata` transforms as stage 1:

```bash
python -m methods.analysis.plot_eps_nondominated_figures --variants full regression perfect
```

Default output tree: `figures/eps_nondominated_<variant>/fig1_pareto_front_comparison/` and `.../fig2_parallel_axes/`. Useful flags: `--figures 1`, `--figures 2`, `--filter`, `--baseline`, `--in-root`, `--out-root`.

This path is **orthogonal** to `04_make_figures.py` (which reads raw `MMBorg_*.csv`, not the ε CSVs). Minimal Python snippets for custom plots live in the module docstring of `methods/analysis/mmborg_eps_nondominated_set.py`.

---

## Full-Pareto Pywr MPI sweep and Stage 3 (unified figures + diagnostics)

**`methods/ensemble/run_full_pareto_pywr_mpi.py`** runs **every** filtered Pareto alignment row through Pywr-DRB, **one MPI rank per combined-basin simulation**. HDF5 + per-rank manifests are written under **`CEE_FULL_PARETO_WORK_DIR`** (default: **`pywr_data/full_pareto_runs/`**). When ranks finish, rank 0 merges to **`_full_pareto_manifest.json`**. **`run_full_pareto_pywr_mpi.sh`** then runs **one** Python call — `methods/figures_stage3/plot_stage3_full_pareto_figures.py` — which writes figures **14–23** under each **`figures/borg_*`** bundle folder (multipanel daily/monthly plus diagnostics; see `full_pareto_output_paths.py`).

Environment matches policies, inflow, flow mode, and Borg variants — see the MPI script docstring (e.g. `CEE_BORG_RUN_VARIANTS`, `CEE_FIGURE_POLICIES`, `CEE_PYWR_FLOW_PREDICTION_MODE`).

**Example (after `module load` / venv):**

```bash
# Dry-run job counts
python -m methods.ensemble.run_full_pareto_pywr_mpi --dry-run

# Typical cluster run
mpirun -np 30 python -m methods.ensemble.run_full_pareto_pywr_mpi
```

This path reads **`_full_pareto_manifest.json`** and aggregates HDF5 simulations into **`figures/<borg_bundle>/fig14_*` … `fig23_*`** (split by Borg variant by default).

```bash
python -m methods.figures_stage3.plot_stage3_full_pareto_figures \
  --manifest pywr_data/full_pareto_runs/_full_pareto_manifest.json \
  --out-dir figures \
  --which all
```

**CLI:** `--out-dir` (figure tree root; default **project `figures/`**), `--which` (`daily` \| `monthly` \| `multipanels` \| `diagnostics` \| `all`; default **`all`**), `--mock` (synthetic multipanels; diagnostics need a real manifest), `--manifest`, `--borg-variant` (`full` \| `regression` \| `perfect` or `CEE_STAGE3_BORG_VARIANT`), `--max-runs` (cap rows per policy for tests).

**Optional env (same run):** `STAGE3_SKIP_MONTHLY=1` (skip monthly multipanel only), `STAGE3_SKIP_DIAGNOSTICS=1` (skip diagnostic PNGs), `STAGE3_EXTRA_ARGS='...'` passed through from `run_full_pareto_pywr_mpi.sh`.

**Notes:**

- **`aggregate_stage3_multipanels_from_manifest`** builds **daily + monthly** multipanels in one manifest walk; monthly inflow–release uses **observed** training inflow Q20/Q80 for regime shading.
- **`--which diagnostics`** only (with `--manifest`) runs diagnostic figures **without** re-aggregating multipanels (faster).
- **NWIS Prompton:** bulk aggregation calls `parametric_result_from_h5_path(..., fetch_prompton_nwis=False)` so USGS is **not** queried once per HDF5. Optional: `CEE_SKIP_PROMPTON_NWIS=1` for other loaders.
- Programmatic access: `from methods.figures_stage3.stage3_analysis import run_stage3_full_pareto_analysis` or `from methods.figures_stage3.data_loading import aggregate_stage3_multipanels_from_manifest`.

---

## Multiseed outputs: MOEA Framework folder + policy CSV for analysis

After `run_parallel_mmborg_multiseed.sh` finishes, Borg writes per-island `*.runtime` files and aggregated `MMBorg_*_seed*.csv` files under `outputs/`.

### A) One combined CSV of all policy solutions (simplest)

From `CEE6400Project/` (uses your existing Borg CSVs, not the MOEA Java tools):

```bash
python -m methods.analysis.collect_borg_csvs_for_analysis -o outputs/analysis_all_borg_runs.csv
```

This adds columns such as `policy`, `reservoir`, `seed`, `mrf_filtered`, `mrf_filter_variant` (`regression`, `perfect`, `none`, or legacy tokens), and concatenates every matching `MMBorg_*.csv` (including legacy `*_mrfmasked_*` names that still match the filename pattern).

### B) MOEA Framework pipeline (refsets, epsilon-grid metrics)

You do **not** move the whole repo; only **sync island runtimes** into `moeaframework/outputs/` in the layout the scripts expect (`Policy_<STARFIT|RBF|PWL>/runtime/<reservoir>/`).

```bash
cd /path/to/CEE6400Project/moeaframework
python organize_borg_outputs.py --src ../outputs --dst ./outputs
# Optional: omit MRF-filtered runs if you only want full-series seeds:
# python organize_borg_outputs.py --src ../outputs --dst ./outputs --skip-mrfmasked
```

Runtime names use `_mrffiltered_regression` / `_mrffiltered_perfect` before the island index (legacy `_mrfmasked_*` patterns are still recognized; see `moeaframework/organize_borg_outputs.py`).

Install / use the MOEA Framework 5 CLI (see `moeaframework/moeaframework_workflow.md` and `0-MOEAFramework5-install.sh` if needed). Then run the four steps:

```bash
export SEED_FROM=1 SEED_TO=10
# Must match the objective count expected by your MOEA workflow setup
export EPSILON=0.01,0.01,0.01,0.01
bash run_moea_workflow.sh
# or: bash run_moea_workflow.sh --cli "$(pwd)/MOEAFramework-5.0/cli"
```

Artifacts:

- `moeaframework/outputs/Policy_*/refsets/<reservoir>/` — merged `seed*.ref`, global `<reservoir>.ref`
- `moeaframework/outputs/Policy_*/metrics/<reservoir>/` — `*.metric` files (quality vs global reference)

Custom problem JARs under `MOEAFramework-5.0/native/*/Makefile` must match your MOEA header/problem naming. See `moeaframework_workflow.md` for `BuildProblem` / `make`.

## Reproducibility knobs

Key parameters and settings should be maintained in `pywrdrb.release_policies.config` (and are surfaced here via `methods/config.py`):

- `SEED` (default: 71)
- `NFE` (default: 30000)
- `ISLANDS` (default: 4)
- `EPSILONS`, `METRICS`, `OBJ_FILTER_BOUNDS`
- All policy parameter bounds (STARFIT/RBF/PWL)
- Per-reservoir capacities, inflow bounds, release min/max
- `NORMAL_OPERATING_RANGE_BY_RESERVOIR` and `get_normal_operating_range(...)` for explicit NOR fractions and storage bounds (MG) by reservoir.

### Workflow flags cheat sheet

For reproducible reruns, these are the primary switches to communicate to collaborators:

- `run_parallel_mmborg.sh` / `run_parallel_mmborg_multiseed.sh`:
  - `CEE_BORG_MODES=full,regression,perfect` (default three-phase sweep; subset/reorder as needed)
  - `CEE_BORG_SINGLE_PHASE=1` with `USE_MRF=true|false` and `CEE_MRF_FILTER_SOURCE=regression_disagg|perfect` (legacy one-mode sweep)
  - `BORG_SEED_UNMASKED` (full-series) and `BORG_SEED_MASKED` (MRF-filtered)
- `03_parallel_borg_run.py`:
  - args: `POLICY_TYPE RESERVOIR_NAME [seed] [mrf_json] [use_mrf]`
  - env override: `CEE_USE_MRF=0|1` (takes precedence over argv mask flag)
- `run_postprocessing_and_figures.sh`:
  - `CEE_POSTPROCESS_BUNDLE=full|regression_disagg|perfect|all`
  - `FLOW_MODE_FULL`, `FLOW_MODE_REGRESSION_DISAGG`, `FLOW_MODE_PERFECT` (Pywr flow mode per bundle)
  - `BORG_SEED_UNMASKED`, `BORG_SEED_FILTERED`
- `build_mrf_masking_folder.sh`:
  - Generates `preprocessing_outputs/masking/pub_reconstruction/` and/or `.../perfect_information/` JSON/CSV artifacts used by MRF-filtered optimization phases.

Additional reproducibility controls:

- SLURM job files:
  - `run_parallel_mmborg.sh` — Borg phase driver (`full`, `regression`, `perfect` by default); env: `CEE_BORG_MODES`, `CEE_BORG_SINGLE_PHASE`, `USE_MRF`, `CEE_MRF_FILTER_SOURCE`, seeds, `MRF_RANGES_JSON`
  - `run_parallel_mmborg_multiseed.sh` — same phase logic + multiseed loops
  - `build_mrf_masking_folder.sh` — can be run standalone to refresh `preprocessing_outputs/masking/` only
  - `run_postprocessing_and_figures.sh` — baselines, three optimization summaries, three figure trees

Upgrade note: there are two `config.py` files, but the canonical version is `pywrdrb.release_policies.config` in the release-policy branch. Keep seeds/objectives/bounds authoritative there; use `methods/config.py` in this repo only for local paths, wrappers, and backward-compatible aliases.

Backward compatibility aliases are still accepted by scripts, but prefer the names in this README (`CEE_MRF_FILTER_*`, `CEE_BORG_MRF_FILTERED`) for new runs.

## Policy sources

The parametric policy classes (STARFIT, RBF, PWL) used here originate from the [Pywr-DRB repository](https://github.com/Pywr-DRB/Pywr-DRB) (feature branch for parametric releases). This ensures that the policies optimized in this repo are consistent with those implemented in Pywr-DRB for validation and comparison.

