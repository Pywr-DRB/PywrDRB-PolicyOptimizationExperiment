import pandas as pd

from methods.load.observations import load_observations, get_overlapping_datetime_indices
from methods.load.observations import scale_inflow_observations

from methods.config import DATA_DIR, reservoir_options, PROCESSED_DATA_DIR

if __name__ == "__main__":
    
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
        # Some reservoirs may not have complete observed inflow columns in inflow.csv.
        # In that case, use inflow_pub for scaling against observed releases.
        if reservoir in inflow_obs.columns:
            inflow_source = inflow_obs.loc[:, [reservoir]]
        elif reservoir in inflow_pub.columns:
            print(f"[02_process_data] '{reservoir}' missing in inflow.csv; using inflow_pub fallback.")
            inflow_source = inflow_pub.loc[:, [reservoir]]
        else:
            print(f"[02_process_data] '{reservoir}' missing in both inflow and inflow_pub; skipping.")
            continue
        
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