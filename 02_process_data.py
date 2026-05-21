"""
Build scaled inflow input for optimization.

Scaling context (lab review / reproducibility)
----------------------------------------------
1. **Release-volume scaling (this script)** — Per reservoir and month, scale observed
   inflows so monthly volume matches release (factor = sum(release)/sum(inflow), >= 1).
   Output: ``obs_data/processed/inflow_scaled.csv``. Exploratory / dynamics; not the
   main Borg training inflow.

2. **NHM-based pub reconstruction** — Pywr inflow ``pub_nhmv10_BC_withObsScaled`` and
   CSV ``inflow_pub.csv`` use NHM (gauge vs reservoir-base) style adjustments from the
   pub reconstruction workflow. **Borg optimization uses ``inflow_pub``.**

**Blue Marsh:** no USGS inflow gauge → not in ``inflow.csv``. Use ``--inflow-source
observed_with_bluemarsh_pub`` or default ``pub_only`` so Blue Marsh comes from ``inflow_pub``.

Inflow source flags:
- ``pub_only`` (default): ``inflow_pub.csv`` for every reservoir present there.
- ``observed_with_bluemarsh_pub``: ``inflow.csv`` when gauged; ``blueMarsh`` from ``inflow_pub``.

Output: ``obs_data/processed/inflow_scaled.csv``
"""

import argparse
import pandas as pd

from methods.load.observations import load_observations, get_overlapping_datetime_indices
from methods.load.observations import scale_inflow_observations

from methods.config import DATA_DIR, reservoir_options, PROCESSED_DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build inflow_scaled.csv from observed and/or public inflow sources.")
    parser.add_argument(
        "--inflow-source",
        choices=("pub_only", "observed_with_bluemarsh_pub"),
        default="pub_only",
        help="How to choose inflow source for each reservoir.",
    )
    args = parser.parse_args()
    
    ##############################################################
    ### Load data ################################################
    ##############################################################
    # raw observations for all reservoirs
    inflow_obs = load_observations(datatype='inflow',
                                   reservoir_name = None,
                                   data_dir = PROCESSED_DATA_DIR,
                                   as_numpy=False)
    inflow_pub = load_observations(datatype='inflow_pub',
                                   reservoir_name = None,
                                   data_dir = PROCESSED_DATA_DIR,
                                   as_numpy=False)
    
    release_obs = load_observations(datatype='release',
                                     reservoir_name = None,
                                     data_dir = PROCESSED_DATA_DIR,
                                     as_numpy=False)
    
    storage_obs = load_observations(datatype='storage',
                                        reservoir_name = None,
                                        data_dir = PROCESSED_DATA_DIR,
                                        as_numpy=False)
    
    ##############################################################
    ### Make scaled inflow dataset, with NA default values #######
    ##############################################################
    inflow_obs_scaled = inflow_obs.copy()
    inflow_obs_scaled.loc[:,:] = float('nan')

    # apply scaling 1 at a time
    for reservoir in reservoir_options:
        if args.inflow_source == "pub_only":
            if reservoir in inflow_pub.columns:
                inflow_source = inflow_pub.loc[:, [reservoir]]
            else:
                print(f"[02_process_data] '{reservoir}' missing in inflow_pub.csv; skipping.")
                continue
        elif args.inflow_source == "observed_with_bluemarsh_pub":
            if reservoir == "blueMarsh":
                if reservoir in inflow_pub.columns:
                    print("[02_process_data] forcing 'blueMarsh' from inflow_pub.csv.")
                    inflow_source = inflow_pub.loc[:, [reservoir]]
                else:
                    print("[02_process_data] 'blueMarsh' missing in inflow_pub.csv; skipping.")
                    continue
            elif reservoir in inflow_obs.columns:
                inflow_source = inflow_obs.loc[:, [reservoir]]
            elif reservoir in inflow_pub.columns:
                print(f"[02_process_data] '{reservoir}' missing in inflow.csv; using inflow_pub fallback.")
                inflow_source = inflow_pub.loc[:, [reservoir]]
            else:
                print(f"[02_process_data] '{reservoir}' missing in both inflow and inflow_pub; skipping.")
                continue
        else:
            raise ValueError(f"Unhandled --inflow-source mode: {args.inflow_source}")
        
        # Get overlapping datetime indices, for inflows and releases
        dt = get_overlapping_datetime_indices(inflow_source, 
                                            release_obs.loc[:, [reservoir]])
        
        # Scale inflow observations, to match the release volume
        res_inflow_scaled = scale_inflow_observations(inflow_source.loc[dt, [reservoir]], 
                                                      release_obs.loc[dt, [reservoir]])
        
        # Store scaled inflow observations
        inflow_obs_scaled[reservoir] = res_inflow_scaled[reservoir]
        inflow_obs_scaled.index.name = 'datetime'
        
    # Save scaled inflow observations
    inflow_obs_scaled.to_csv(PROCESSED_DATA_DIR + '/inflow_scaled.csv')