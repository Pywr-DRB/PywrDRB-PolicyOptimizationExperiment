# First-time setup (fresh git clone)

## Environment

```bash
cd CEE6400Project
module load python/3.11.5
source /path/to/borg-env/bin/activate
pip install -r requirements.txt
# Editable Pywr-DRB install per README
```

## `02_process_data.py` and Blue Marsh

There is **no USGS inflow gauge** for Blue Marsh, so `inflow.csv` has no `blueMarsh` column. That is intentional.

Use default (recommended):

```bash
python 02_process_data.py --inflow-source pub_only
```

Or:

```bash
python 02_process_data.py --inflow-source observed_with_bluemarsh_pub
```

Borg training uses **`inflow_pub.csv`**, which includes Blue Marsh.

## `build_mrf_filtering_folder.sh`

**Minimum for masked Borg:**

```bash
bash build_mrf_filtering_folder.sh
# → preprocessing_outputs/filtering/regression_disagg/mrf_active_filter_daily.csv
```

- **regression_disagg** runs Pywr locally if no HDF5 is cached under `preprocessing_outputs/pywr/`.
- **perfect_foresight** is optional; skipped with a warning if there is no HDF5 and Pywr flow data are missing (set `REQUIRE_PERFECT_MRF_PREBUILD=1` to fail instead).

Check normal-ops share:

```bash
cat preprocessing_outputs/filtering/regression_disagg/mrf_filtering_summary.txt
```

## Do not commit accidental deletions

If `git status` shows many `D` files (deleted tracked scripts), restore before committing:

```bash
git restore .
```
