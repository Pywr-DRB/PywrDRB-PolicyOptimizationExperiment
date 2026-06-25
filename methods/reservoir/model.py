import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pywrdrb.release_policies import RBF, PWL, STARFIT
from methods.utils.conversions import cfs_to_mgd
from pywrdrb.release_policies.config import get_policy_context


class Reservoir():
    def __init__(self, 
                 inflow, 
                 capacity,
                 policy_type,
                 policy_params,
                 dates = None,
                 release_min = None,
                 release_max = None,
                 inflow_max = None,
                 inflow_min = None,
                 initial_storage = None,
                 start_date = None,
                 name = None
                 ):
        """
        Reservoir model class, used to simulate the operation of a reservoir.
        
        Simulation data is stored as class attributes, to be accessed
        by the policy to inform release decisions, and to be retrieved
        after simulation for objective calculation.
        
        Parameters
        ----------
        inflow : np.array
            Array of inflow values for each timestep.
        capacity : float
            Maximum storage capacity of the reservoir.
        policy_type : str
            Type of policy to use for operation. Options are: "PWL", "RBF", "STARFIT"
        policy_params : np.array
            Array of policy parameters to be optimized.
        release_min : float, optional
            Minimum release constraint. Default is None.
        release_max : float, optional
            Maximum release constraint. Default is None.
        inflow_max : float, optional
            Maximum inflow used for scaling inflow inputs. If None, use max of observations.
        inflow_min : float, optional
            Minimum inflow used for scaling inflow inputs. If None, use 0.0.
        initial_storage : float, optional
            Initial storage level of the reservoir. Default is 0.8 * capacity.
        dates : pd.DatetimeIndex, optional
            Dates for the simulation timeseries. Default is None. 
        
        Methods
        -------
        run()
            Run the reservoir simulation for the given inflow and demand timeseries.
        
        get_release(t)
            Get the release for the given timestep, from the policy class.
            
        Returns
        -------
        None
        """
        try:
            inflow = np.array(inflow).flatten()
        except Exception as e:
            raise ValueError(f"Inflow must be a 1D array. Got {inflow.shape}") from e
        
        if dates is None:
            raise AssertionError("Dates must be provided to Reservoir during initialization.")
        if name is None:
            raise AssertionError("Reservoir 'name' is required to fetch policy context.")
        
        self.name = name 
        self.inflow_array = inflow
        self.T = len(inflow)
        self.start_date = pd.to_datetime(start_date) if start_date is not None else None
        self.dates = pd.to_datetime(dates)
        self.doy = self.dates.dayofyear.values
        assert len(self.doy) == self.T, "dates length must match inflow length"

        # Placeholders; will be set from context in initialize_policy()
        self.capacity = capacity
        self.release_min = None
        self.release_max = None
        self.inflow_min = None
        self.inflow_max = None
    
        self.initial_storage = 0.8 * capacity if initial_storage is None else initial_storage

        # reset simulation variables
        self.reset()
        
        # initialize policy class
        self.policy = None
        self.initalize_policy(policy_type, policy_params)
    
    def reset(self):
        """
        Reset the variable arrays prior to simulation.
        """
        self.storage_array = np.zeros(self.T)
        self.release_array = np.zeros(self.T)
        self.spill_array = np.zeros(self.T)
        return
    
    def initalize_policy(self, policy_type, policy_params):
        """
        Initialize the policy class based on the policy type.
        
        Each policy class will have a different set of parameters,
        which will be parsed inside the unique classes. 
        
        Parameters
        ----------
        policy_type : str
            Type of policy to use for operation. Options are: "PWL", "RBF", "STARFIT"
        policy_options : dict
            Dictionary of optional kwargs to pass to the policy class.
        policy_params : np.array
            Array of policy parameters to be optimized.        
        """
        # 1) build policy
        if policy_type == 'PWL':
            self.policy = PWL(policy_params=policy_params)
        elif policy_type == 'RBF':
            self.policy = RBF(policy_params=policy_params)
        elif policy_type == 'STARFIT':
            self.policy = STARFIT(policy_params=policy_params, reservoir_name=self.name)
            # Load only the fixed constants; keep the optimizer's 17 params (don't reload from CSV).
            self.policy.load_starfit_constants(reservoir_name=self.name)
            assert self.policy.I_bar is not None, "STARFIT I_bar failed to load"
        else:
            raise ValueError(f"Invalid policy type: {policy_type}")

        self.policy.reservoir_name = self.name

        # 2) pull context from CONFIG ONLY (no overrides)
        ctx = get_policy_context(self.name)

        # 3) mirror context onto the model so sim math matches policy scaling/enforcement
        self.capacity = float(ctx["storage_capacity"])
        self.release_min = float(ctx["release_min"])
        self.release_max = float(ctx["release_max"])
        self.inflow_min = float(ctx["x_min"][1])
        self.inflow_max = float(ctx["x_max"][1])

        # Set default initial storage AFTER we know capacity from context
        self.initial_storage = 0.8 * self.capacity if self.initial_storage is None else self.initial_storage

        # 4) hand the same context to the policy
        self.policy.set_context(**ctx)
        self.policy.validate_policy_params()
        return
    
    def get_release(self, timestep):
        """
        Use the policy class to get the release for the given timestep,
        the policy class can access reservoir attributes to make decisions.
        """
        # Build inputs at this timestep
        I_t = self.inflow_array[timestep]
        prev_S = self.initial_storage if timestep == 0 else self.storage_array[timestep-1]
        D_t = self.doy[timestep] 
        return self.policy.get_release(inflow=I_t, storage=prev_S, day_of_year=D_t)
    
    def run(self):
        # reset state arrays
        self.reset()
        # reset violation log for this run
        if hasattr(self.policy, "reset_violation_log"):
            self.policy.reset_violation_log()

        for t in range(self.T):
            # 1) policy target
            release_target = self.get_release(timestep=t)

            # 2) clamp by availability (correct t=0 handling)
            prev_S = self.initial_storage if t == 0 else self.storage_array[t-1]
            I_t = self.inflow_array[t]
            self.release_array[t] = min(release_target, prev_S + I_t)

            # 3) update storage
            if t == 0:
                self.storage_array[t] = self.initial_storage + I_t - self.release_array[t]
            else:
                self.storage_array[t] = self.storage_array[t-1] + I_t - self.release_array[t]

            # 4) spill and cap to capacity
            if self.storage_array[t] > self.capacity:
                self.spill_array[t] = self.storage_array[t] - self.capacity
                self.storage_array[t] = self.capacity
            else:
                self.spill_array[t] = 0.0

            # 5) hard guard
            assert self.release_array[t] <= prev_S + I_t + 1e-9, \
                "Release exceeds physically available water."

        return


    def get_results(self):
        # Make a pd.DataFrame of the results
        results = {
            "inflow": self.inflow_array,
            "release": self.release_array,
            "storage": self.storage_array,
            "spill": self.spill_array
        }
        
        results = pd.DataFrame(results)
        
        return results

    def plot(self,
             fname=None,
             save=False,
             storage_obs=None,
             release_obs=None,
             release_log_scale=True,
             release_smooth_window=None,
             title=None):
        """
        Plot storage and release data.
        
        Args:
            storage_obs (pd.Series): Observed storage data for the simulation period.
            release_obs (pd.Series): Observed release data for the simulation period.
            save (bool): Whether to save the plot as a file.
            fname (str): Filename for saving the plot.
        
        """
        
        storage_sim = self.storage_array
        release_sim = self.release_array
        
        
        ### plot simulated and obs storage and release
        fig, ax = plt.subplots(2, 1, figsize=(10, 8),
                                sharex=True)
        
        ax[0].plot(storage_obs, label='Observed', color='blue', alpha=0.5)
        ax[0].plot(storage_sim, label='Simulated', color='orange', alpha=0.5)
        ax[0].set_ylabel('Storage (MGD)')
        ax[0].set_xlim(0, self.T)
        ax[0].set_title(f"Reservoir Dynamics – {self.name.upper()}")

        
        # smooth releases
        if release_smooth_window is not None:
            w = release_smooth_window
            release_obs = pd.Series(release_obs.flatten()).rolling(window=w).mean().values
            release_sim = pd.Series(release_sim.flatten()).rolling(window=w).mean().values
        
        ax[1].plot(release_obs, label='Observed', color='blue', alpha=0.5)
        ax[1].plot(release_sim, label='Simulated', color='orange', alpha=0.5)
        if release_log_scale:
            ax[1].set_yscale('log')

        ax[1].set_ylabel('Release (MGD)')
        
        ax[1].set_xlabel('Time')
        ax[0].legend()
        
        if title is not None:
            fig.suptitle(title)
        
        if save:
            if fname is None:
                raise ValueError("Filename must be provided when save=True.")
            plt.savefig(fname, dpi=300)
        
        plt.tight_layout()
        plt.show()