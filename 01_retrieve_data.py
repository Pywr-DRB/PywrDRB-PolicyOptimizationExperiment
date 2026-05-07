"""
Retrieve and aggregate observed reservoir data products.

This script downloads raw USGS inflow/release/elevation series, converts elevation
to storage, and writes reservoir-level processed CSVs used by `02_process_data.py`.

Outputs:
- `obs_data/raw/{inflow_raw,release_raw,elevation_raw,storage_raw}.csv`
- `obs_data/processed/{inflow,release,storage}.csv`
"""

from methods.preprocessing.observed_data_retriever import ObservedDataRetriever
from methods.plotting.plot_obs_dynamics import plot_obs_reservoir_dynamics
import os
import pandas as pd

# Directories
from methods.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, FIG_DIR

from gauge_ids import inflow_gauges, release_gauges, storage_gauges, storage_curves

# Flatten gauges
def flatten_gauges(gauge_dict):
    return sorted({g for gauges in gauge_dict.values() for g in gauges})

if __name__ == "__main__":
    retriever = ObservedDataRetriever(out_dir=RAW_DATA_DIR)

    inflow_ids = flatten_gauges(inflow_gauges)
    release_ids = flatten_gauges(release_gauges)
    storage_ids = flatten_gauges(storage_gauges)

    inflows = retriever.get(inflow_ids, param_cd="00060")
    retriever.save_to_csv(inflows, "inflow_raw")

    releases = retriever.get(release_ids, param_cd="00060")
    retriever.save_to_csv(releases, "release_raw")

    elevations = retriever.get(storage_ids, param_cd="00062")
    retriever.save_to_csv(elevations, "elevation_raw")

    storages = retriever.convert_elevation_to_storage(elevations, storage_curves)
    retriever.save_to_csv(storages, "storage_raw")

    # Load raw files
    inflows_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, "inflow_raw.csv"), index_col="datetime", parse_dates=True)
    releases_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, "release_raw.csv"), index_col="datetime", parse_dates=True)
    storages_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, "storage_raw.csv"), index_col="datetime", parse_dates=True)

    # Postprocess: aggregate gauges into reservoir-level time series
    retriever.postprocess_and_save(inflows_raw, inflow_gauges, os.path.join(PROCESSED_DATA_DIR, "inflow.csv"))
    retriever.postprocess_and_save(releases_raw, release_gauges, os.path.join(PROCESSED_DATA_DIR, "release.csv"))
    retriever.postprocess_and_save(storages_raw, storage_gauges, os.path.join(PROCESSED_DATA_DIR, "storage.csv"))

    # Check for missing days
    for name, df in [("inflow", inflows), ("release", releases), ("storage", storages)]:
        missing = retriever.find_missing_dates(df)
        print(f"{name} missing dates: {len(missing)} days")

    # for res in sorted(set(inflows.columns).union(storages.columns).union(releases.columns)):
    #     print(f"\nPlotting reservoir: {res}")
    #     plot_obs_reservoir_dynamics(
    #         I=inflows, 
    #         S=storages, 
    #         R=releases,
    #         reservoir_name=res,
    #         title=f"{res} Reservoir Observed Data",
    #         timescale='daily',
    #         log=True
    #     )
