# MRF filtering for lower-basin policy optimization

How we separate **normal reservoir operations** from **lower-basin Trenton MRF (minimum release flow) support** when fitting parametric policies. Full simulation always runs; **only Borg objective calculation** is filtered.

---

## Problem

Blue Marsh, Beltzville (combined), and Nockamixon can release extra water for **Trenton flow targets** during drought-style management. Pywr-DRB records these as `lower_basin_mrf_contributions` (`mrf_trenton_<reservoir>`, **MGD** — often 0, sometimes up to ~300). That behavior is **not** baseline STARFIT operations. Without filtering, the optimizer fits drought/MRF signals.

**STARFIT NOR** (harmonic storage bounds) is unrelated to **MRF normal-ops filtering** here.

---

## Data source: Pywr-DRB HDF5

Filters are built from a **default** Pywr-DRB run (standard model policies), via `build_mrf_filtering_folder.sh` → `methods.preprocessing.build_mrf_active_filters`.

| Bundle | `flow_prediction_mode` | Typical use |
|--------|------------------------|-------------|
| `preprocessing_outputs/filtering/regression_disagg/` | `regression_disagg` | Default masked Borg (`CEE_MRF_FILTER_SOURCE=regression_disagg`) |
| `preprocessing_outputs/filtering/perfect_foresight/` | `perfect_foresight` | Optional; needs HDF5 or local Pywr flow data |

**Local first-time setup:**

```bash
cd CEE6400Project
module load python/3.11.5   # if needed
source /path/to/borg-env/bin/activate
bash build_mrf_filtering_folder.sh
```

- **regression_disagg:** uses existing HDF5 under `preprocessing_outputs/pywr/` if present; otherwise **runs Pywr** and writes a new HDF5 there.
- **perfect_foresight:** uses existing HDF5 if found; otherwise tries Pywr (may be skipped if Pywr flow directories are missing — see SETUP.md).

Force a fresh regression simulation:

```bash
FORCE_MRF_PYWR_SIM=1 bash build_mrf_filtering_folder.sh
```

---

## Outputs per bundle

| File | Purpose |
|------|---------|
| `lower_basin_mrf_contributions.csv` | Daily `mrf_trenton_*` (MGD) |
| `mrf_active_filter_daily.csv` | Raw active flags + `normal_ops_for_objectives` + `excluded_from_objectives` |
| `mrf_filtering_summary.txt` | **Percentages** of normal-ops vs excluded days (full period + Borg training overlap) |
| `lower_basin_mrf_contributions_active_periods.png` | Stacked contributions + raw active shading |
| `mrf_filtering_for_optimization.png` | Raw active vs **buffer** vs days **included in objectives** |
| `traceability.txt` | HDF5 path and flow mode used |

---

## Toggles and filter math

### 1. Raw MRF-active days

- **Threshold:** `MRF_ACTIVE_EPS_MGD` in `methods/config.py` (default **0.1 MGD**); CLI `--epsilon`.
- **Rule:** Day active if any `mrf_trenton_*` > ε.
- **Per-reservoir columns** in the daily CSV; `ANY_lower_basin` = union.

### 2. Buffer (excluded from objectives only)

- **Setting:** `MRF_NORMAL_OPS_BUFFER_DAYS` (default **5**); CLI `--buffer-days`.
- **Rule:** For each contiguous active range `[start, end]`, expand to `[start − buffer, end + buffer]` (clipped to the series).
- **Why:** Short release clusters (e.g. Tue + Thu) sit in a management window; buffer drops buildup/drawdown so objectives reflect truer normal ops. The optimizer has no buffer—only this preprocessing filter.

### 3. Normal operating days (for Borg)

- **Definition:** Not inside any buffered active window (`build_normal_ops_filter` in `mrf_filtering.py`).
- **`03_parallel_borg_run.py`:** loads `mrf_active_filter_daily.csv`, applies filter in `evaluate()` only; simulation runs all days.
- **Active period:** MRF contribution > 0 (from Pywr simulation — approximate timing, not perfect historical ops).
- During active days, imposed MRF releases stay in the sim; **policy parameters are not judged** on those timesteps.

### 4. Borg / Slurm

| Variable | Effect |
|----------|--------|
| `USE_MRF=true` | Enable filtering |
| `CEE_MRF_FILTER_SOURCE` | `regression_disagg` or `perfect_foresight` (daily CSV path) |

---

## Reservoir notes

- **`beltzvilleCombined`:** ResOps naming; volume ≈ Beltzville; tiny upstream pools ignored for management.
- **No legacy JSON required** for optimization — daily CSV is the source of truth (`03_parallel_borg_run.py`).

---

## Manual command

```bash
python -m methods.preprocessing.build_mrf_active_filters \
  --filter-bundle regression_disagg \
  --inflow-type pub_nhmv10_BC_withObsScaled \
  --flow-prediction-mode regression_disagg \
  --skip-simulation \
  --existing-output preprocessing_outputs/pywr/pywrdrb_output_pub_nhmv10_BC_withObsScaled_regression_disagg.hdf5
```

---

## Related

- `methods/preprocessing/mrf_filtering.py` — filter math and plots  
- `SETUP.md` — fresh clone, Blue Marsh inflow, perfect bundle skip  
- `methods/plotting/SELECTION.md` — Pareto solution picks after optimization  
