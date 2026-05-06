import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.plotting.theme import SERIES_COLORS


def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.linspace(0, 1, len(sorted_data))
    return sorted_data, cdf


def plot_annual_storage_distribution(df, label, 
                                     color='black', ax=None, 
                                     quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """
    Plot shaded annual quantiles of storage by day-of-year.
    
    Parameters:
        df (DataFrame): Must contain columns 'datetime', 'year', 'doy', and a storage column
        label (str): Legend label
        color (str): Base color for shading
        ax (matplotlib.axes.Axes): Axes to plot on
        quantiles (list): List of quantiles to compute
    """
    if ax is None:
        ax = plt.gca()
    
    # Infer which storage column is present
    storage_col = [col for col in df.columns if 'storage' in col.lower()][0]
    
    # Group by DOY, compute quantiles
    grouped = df.groupby('doy')[storage_col].quantile(quantiles).unstack()
    
    # Plot quantile fills
    for q_lo, q_hi in zip(quantiles[:2], quantiles[-1:-3:-1]):
        ax.fill_between(grouped.index,
                        grouped[q_lo],
                        grouped[q_hi],
                        color=color,
                        alpha=q_lo,
                        label = label + f" {int(q_lo*100)}% - {int(q_hi*100)}%",
                        edgecolor='none')
    
    # Plot median
    ax.plot(grouped.index, grouped[0.5], color=color, lw=1.5, label=label + " Median")
    
    return ax


def plot_annual_storage_timeseries(df, label=None, color='black', ax=None):
    if ax is None:
        ax = plt.gca()
        
    storage_col = [col for col in df.columns if 'storage' in col.lower()][0]
    
    for y in sorted(df['year'].unique()):
        subset = df[df['year'] == y]
        ax.plot(subset['doy'], subset[storage_col], 
                color=color, alpha=0.3, 
                label=label if y == df['year'].min() else "")
    
    return ax

def plot_weekly_series_scatter(df,
                               cmap='viridis',
                               ax=None,
                               log_scale=False):
    """
    Makes a scatterplot of obs vs sim releases at a weekly timestep.
    The points are colored based on the week of the year. 
    
    Parameters:
        df (DataFrame): Must contain 'obs_release', 'sim_release', and datetime index
        cmap (str): Colormap for coloring points
        ax (matplotlib.axes.Axes): Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()
        
    # Convert datetime to pandas Series
    dt = pd.to_datetime(df.index)
    
    # Get weekly sums of inflow
    df_weekly = df.resample('W').sum()
    df_weekly['week'] = dt.isocalendar().week
    
    ax.scatter(df_weekly['obs_release'],
               df_weekly['sim_release'],
               c=df_weekly['week'],
               cmap=cmap,
               alpha=0.5)
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel("Observed Release (MGD)")
    ax.set_ylabel("Simulated Release (MGD)")
    
    return ax

def plot_annual_inflow_release_distribution(df,
                                            label=None,
                                            color='black',
                                            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                                            ax=None,
                                            show_legend: bool = True,
                                            ax_title: str = "Annual Inflow–Release Distribution",
                                            suppress_title: bool = False):
    """
    Plot quantile bands of inflow vs release curves, aggregated across years.
    
    Parameters:
        df (DataFrame): Must contain 'obs_inflow', 'release', 'year'
        label (str): Legend label
        color (str): Base color for quantile fills and median line
        quantiles (list): Quantile levels to compute
        ax (matplotlib.axes.Axes): Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()

    # Infer release column
    inflow_col = 'obs_inflow'
    release_col = [col for col in df.columns if 'release' in col.lower()][0]

    # Collect inflow-release curves for each year
    inflow_vals = []
    release_vals = []
    for year, group in df.groupby('year'):
        inflow = np.array(group[inflow_col])
        release = np.array(group[release_col])
        sorted_idx = np.argsort(inflow)
        inflow_sorted = inflow[sorted_idx]
        release_sorted = release[sorted_idx]
        inflow_vals.append(inflow_sorted)
        release_vals.append(release_sorted)

    # Stack all years into a 2D array (interpolation ensures aligned inflows)
    inflow_common = np.linspace(np.percentile(np.concatenate(inflow_vals), 1),
                                np.percentile(np.concatenate(inflow_vals), 99),
                                200)

    release_interp = np.array([
        np.interp(inflow_common, inflow_vals[i], release_vals[i])
        for i in range(len(inflow_vals))
    ])

    # Compute quantiles at each inflow level
    release_q = np.quantile(release_interp, q=quantiles, axis=0)

    # Plot quantile bands
    for q_lo, q_hi in zip(quantiles[:2], quantiles[-1:-3:-1]):
        lo = release_q[quantiles.index(q_lo)]
        hi = release_q[quantiles.index(q_hi)]
        ax.fill_between(inflow_common, lo, hi, 
                        color=color, 
                        alpha=q_lo, 
                        edgecolor='none',
                        label=f"{label} {int(q_lo*100)}–{int(q_hi*100)}%")

    # Plot median
    ax.plot(inflow_common, release_q[quantiles.index(0.5)], 
            color=color, lw=1.5, label=f"{label} Median")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Observed Inflow")
    ax.set_ylabel("Release")
    if not suppress_title:
        ax.set_title(ax_title)
    if show_legend:
        ax.legend()

    return ax



def plot_fdc(data, 
             label=None, 
             color='black', 
             ax=None, 
             logscale=True):
    """
    Plot the empirical CDF of the given data.
    
    Parameters:
        data (array-like): Data to compute the CDF for
        label (str): Label for the plot
        color (str): Color of the plot line
        ax (matplotlib.axes.Axes): Axes to plot on
    """
    sorted_data, cdf = compute_cdf(data)
    if ax is None:
        ax = plt.gca()
    ax.plot(cdf*100, sorted_data, label=label, color=color)
    
    if logscale:
        ax.set_yscale('log')
    return ax

def plot_storage_release_distributions(obs_storage, obs_release, 
                                       sim_storage, sim_release,
                                       obs_inflow,
                                       datetime,
                                       storage_distribution=True,
                                       inflow_vs_release=True,
                                       inflow_scatter=False,
                                       fname=None):

    """
    Plot observed vs simulated reservoir storage and release dynamics.
    
    Parameters:
        obs_storage (array-like): Observed storage volume
        obs_release (array-like): Observed release volume
        sim_storage (array-like): Simulated storage volume
        sim_release (array-like): Simulated release volume
        datetime (array-like): Corresponding datetime values (must be same length as others)
    """
    # Convert datetime to pandas Series
    dt = pd.to_datetime(datetime)
    df = pd.DataFrame({
        'datetime': dt,
        'year': dt.year,
        'doy': dt.dayofyear,
        'obs_storage': obs_storage,
        'sim_storage': sim_storage,
        'obs_release': obs_release,
        'sim_release': sim_release,
        'obs_inflow': obs_inflow
    })

    df.index = df['datetime']

    # Create figure with 2 vertical subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, 
                                   figsize=(5, 8), 
                                   sharex=False, constrained_layout=True)

    # Top Plot: Storage dynamics by year
    
    c_obs = SERIES_COLORS["observed"]
    c_sim = SERIES_COLORS["parametric"]
    if storage_distribution:
        ax1 = plot_annual_storage_distribution(df[['datetime', 'year', 'doy', 'obs_storage']].copy(),
                                               label='Observed', color=c_obs, ax=ax1)
        ax1 = plot_annual_storage_distribution(df[['datetime', 'year', 'doy', 'sim_storage']].copy(),
                                               label='Simulated (Pywr)', color=c_sim, ax=ax1)
    else:
        ax1 = plot_annual_storage_timeseries(df[['datetime', 'year', 'doy', 'obs_storage']].copy(),
                                             label='Observed', color=c_obs, ax=ax1)
        ax1 = plot_annual_storage_timeseries(df[['datetime', 'year', 'doy', 'sim_storage']].copy(),
                                             label='Simulated (Pywr)', color=c_sim, ax=ax1)

    
    ax1.set_ylabel("Storage (MG)")
    ax1.set_xlabel("Day of Year")
    # ax1.legend()

    # Bottom Plot: CDF of releases
    if inflow_vs_release:
        ax2 = plot_annual_inflow_release_distribution(df[['datetime', 'year', 'doy', 'obs_inflow', 'obs_release']].copy(),
                                                 label='Observed', color=c_obs, ax=ax2)
        ax2 = plot_annual_inflow_release_distribution(df[['datetime', 'year', 'doy', 'obs_inflow', 'sim_release']].copy(),
                                                    label='Simulated (Pywr)', color=c_sim, ax=ax2)
        
        ax2.set_ylabel("Release (MGD)")
        ax2.set_xlabel("Inflow (MGD)")
        
    elif inflow_scatter:
        ax2 = plot_weekly_series_scatter(df[['obs_release', 'sim_release']].copy(),
                                          cmap='viridis', ax=ax2, log_scale=True)

    else:
        ax2 = plot_fdc(obs_release, label='Observed',
                    color=c_obs, ax=ax2)
        ax2 = plot_fdc(sim_release, label='Simulated (Pywr)',
                    color=c_sim, ax=ax2)
        
        ax2.set_ylabel("Release (MGD)")
        ax2.set_xlabel("Non-exceedance Probability (%)")
        
    # Get legend handles and labels
    handles, labels = ax2.get_legend_handles_labels()
    # Set legend below the plot
    ax2.legend(handles, labels,
               ncol=2, 
               loc='upper center', 
               bbox_to_anchor=(0.5, -0.2), 
               borderaxespad=0., 
               fontsize='small')
    
    if fname is not None:
        plt.savefig(fname, dpi=250)
        plt.close(fig)

    return fig, (ax1, ax2)


def plot_storage_release_distributions_independent_vs_pywr_split(
    obs_storage,
    obs_release,
    indie_storage,
    indie_release,
    pywr_storage,
    pywr_release,
    obs_inflow,
    datetime,
    fname,
    eval_period_label: str,
    *,
    capacity_mg: float | None = None,
    nor_lo_pct_by_doy: np.ndarray | None = None,
    nor_hi_pct_by_doy: np.ndarray | None = None,
    suptitle_extra: str = "",
    pick_label: str | None = None,
    pywr_mode_label: str | None = None,
):
    """
    Two-column layout matching Figure 4 dynamics: left = observed vs independent reservoir model;
    right = observed vs Pywr-DRB parametric. Uses the *same* daily index for all series (caller
    should pass aligned arrays). ``eval_period_label`` is shown in the suptitle (e.g. ``1980-01-01 to 2018-12-31``).

    Optional (STARFIT): pass ``capacity_mg`` to plot storage as percent of capacity. Pass
    ``nor_lo_pct_by_doy`` / ``nor_hi_pct_by_doy`` (length 366, index 0 = DOY 1) to shade the seasonal
    NOR band on the storage-by-DOY panels (same units as percent storage).
    """
    dt = pd.to_datetime(datetime)
    df = pd.DataFrame(
        {
            "datetime": dt,
            "year": dt.year,
            "doy": dt.dayofyear,
            "obs_storage": obs_storage,
            "obs_release": obs_release,
            "obs_inflow": obs_inflow,
            "indie_storage": indie_storage,
            "indie_release": indie_release,
            "pywr_storage": pywr_storage,
            "pywr_release": pywr_release,
        }
    )
    df.index = df["datetime"]

    storage_ylabel = "Storage (MG)"
    if capacity_mg is not None and float(capacity_mg) > 0:
        cap = float(capacity_mg)
        for col in ("obs_storage", "indie_storage", "pywr_storage"):
            df[col] = 100.0 * pd.to_numeric(df[col], errors="coerce") / cap
        storage_ylabel = "Storage (% of capacity)"

    c_obs = SERIES_COLORS["observed"]
    c_ind = SERIES_COLORS["independent"]
    c_pyw = SERIES_COLORS["parametric"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    nor_ok = (
        nor_lo_pct_by_doy is not None
        and nor_hi_pct_by_doy is not None
        and len(nor_lo_pct_by_doy) >= 366
        and len(nor_hi_pct_by_doy) >= 366
    )
    if nor_ok:
        xs = np.arange(1, 367)
        lo = np.asarray(nor_lo_pct_by_doy[:366], dtype=float)
        hi = np.asarray(nor_hi_pct_by_doy[:366], dtype=float)
        for ax in (axs[0, 0], axs[0, 1]):
            ax.fill_between(
                xs,
                lo,
                hi,
                color="0.7",
                alpha=0.35,
                zorder=0,
                linewidth=0,
                label="STARFIT NOR",
            )

    # Row 0: storage quantiles by DOY — left: obs + indie; right: obs + pywr
    plot_annual_storage_distribution(
        df[["datetime", "year", "doy", "obs_storage"]].copy(),
        label="Observed",
        color=c_obs,
        ax=axs[0, 0],
    )
    plot_annual_storage_distribution(
        df[["datetime", "year", "doy", "indie_storage"]].rename(columns={"indie_storage": "sim_storage"}),
        label="Independent model",
        color=c_ind,
        ax=axs[0, 0],
    )
    axs[0, 0].set_ylabel(storage_ylabel)
    axs[0, 0].set_xlabel("Day of year")
    axs[0, 0].set_title("Storage (annual quantiles) — independent simulation")

    plot_annual_storage_distribution(
        df[["datetime", "year", "doy", "obs_storage"]].copy(),
        label="Observed",
        color=c_obs,
        ax=axs[0, 1],
    )
    plot_annual_storage_distribution(
        df[["datetime", "year", "doy", "pywr_storage"]].rename(columns={"pywr_storage": "sim_storage"}),
        label="Pywr-DRB parametric",
        color=c_pyw,
        ax=axs[0, 1],
    )
    axs[0, 1].set_ylabel(storage_ylabel)
    axs[0, 1].set_xlabel("Day of year")
    axs[0, 1].set_title("Storage (annual quantiles) — Pywr-DRB")

    # Row 1: inflow–release quantile curves — same observed inflow axis
    plot_annual_inflow_release_distribution(
        df[["datetime", "year", "doy", "obs_inflow", "obs_release"]].copy(),
        label="Observed",
        color=c_obs,
        ax=axs[1, 0],
        show_legend=False,
        suppress_title=True,
    )
    plot_annual_inflow_release_distribution(
        df[["datetime", "year", "doy", "obs_inflow", "indie_release"]].copy(),
        label="Independent model",
        color=c_ind,
        ax=axs[1, 0],
        show_legend=False,
        suppress_title=True,
    )
    axs[1, 0].set_title("Inflow–release (annual) — observed vs independent")

    plot_annual_inflow_release_distribution(
        df[["datetime", "year", "doy", "obs_inflow", "obs_release"]].copy(),
        label="Observed",
        color=c_obs,
        ax=axs[1, 1],
        show_legend=False,
        suppress_title=True,
    )
    plot_annual_inflow_release_distribution(
        df[["datetime", "year", "doy", "obs_inflow", "pywr_release"]].copy(),
        label="Pywr-DRB parametric",
        color=c_pyw,
        ax=axs[1, 1],
        show_legend=False,
        suppress_title=True,
    )
    axs[1, 1].set_title("Inflow–release (annual) — observed vs Pywr-DRB")

    actual_start = str(df.index.min().date()) if len(df.index) else None
    actual_end = str(df.index.max().date()) if len(df.index) else None
    date_label = f"{actual_start} to {actual_end}" if actual_start and actual_end else eval_period_label
    st = f"Annual aggregation — plotted period\n{date_label}"
    if pick_label:
        st += f"\nSelected solution: {pick_label}"
    if pywr_mode_label:
        st += f"\nPywr-DRB mode: {pywr_mode_label}"
    if suptitle_extra:
        st += f"\n{suptitle_extra}"
    fig.suptitle(st, fontsize=13, fontweight="bold")

    handles, labels = [], []
    for ax in axs.ravel():
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab):
            if ll and ll not in labels:
                handles.append(hh)
                labels.append(ll)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.005),
            ncol=3,
            fontsize="small",
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0.10, 1, 0.94])

    if fname is not None:
        fig.savefig(fname, dpi=250, bbox_inches="tight")
        plt.close(fig)
    return fig, axs
