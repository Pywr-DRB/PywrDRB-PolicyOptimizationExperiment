# %% [markdown]
# # Pywr-DRB Parametric Release — Example Usage Notebook
#
# This notebook shows how to **override reservoir release rules inline and via CSV id**
# using the new `ParametricReservoirRelease` parameter via the
# `ModelBuilder(options=...)` pattern. It focuses on:
#
# 1. Building a `release_policy_dict` mapping reservoir names to parametric specs.
# 2. Running Pywr-DRB with **inline parametric overrides**.
# 3. (Optionally) Running the **default model** for comparison.
# 4. Loading storage / release series and making quick checks vs observations.
#
# **Assumptions**
# - Units: flows in **MGD**, storage in **MG**.
# - Reservoir names follow `methods.config.reservoir_options`
#   (e.g., `fewalter`, `beltzvilleCombined`, `blueMarsh`, `prompton`).
# - Parameters follow policy definitions: `"STARFIT"`, `"RBF"`, `"PWL"`.
#
# **Goal**
# Use *selected best parameter sets* (you will paste in below) for:
# `fewalter`, `beltzvilleCombined`, `blueMarsh`, and `prompton`,
# run the parametric Pywr-DRB model, and compare to observations
# and/or the standalone Reservoir model.

# %% [markdown]
# ## 0. Imports, paths, and configuration

# %%
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Helper to find project roots (same pattern as other scripts)
# ------------------------------------------------------------------
def _find_up(name: str, start: Path | None = None) -> Path:
    p = Path.cwd() if start is None else Path(start).resolve()
    for q in [p, *p.parents]:
        cand = q / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find '{name}' above {p}")

PO_REPO       = _find_up("CEE6400Project")
PYWR_DRB_REPO = _find_up("Release_Policy_DRB")
PYWR_SRC      = PYWR_DRB_REPO / "src"

# Ensure import paths
for p in [str(PYWR_SRC), str(PO_REPO)]:
    if p in sys.path:
        sys.path.remove(p)
for p in [str(PYWR_SRC), str(PO_REPO)]:
    sys.path.insert(0, p)

# Project imports
import pywrdrb  # editable install
from methods.config import (
    reservoir_options as RESERVOIR_NAMES,
    reservoir_capacity,
    FIG_DIR,
    PROCESSED_DATA_DIR,
)
from methods.load.observations import get_observational_training_data
from methods.reservoir.model import Reservoir

# Convenience
FIG_DIR = Path(FIG_DIR)
PROCESSED_DATA_DIR = Path(PROCESSED_DATA_DIR)

# %% [markdown]
# ## 1. Policy parameter ordering (STARFIT / RBF / PWL)
#
# This dictionary encodes the **parameter order** expected by each policy type.
# When you paste best-fit parameter vectors from optimization, they must follow
# these orders exactly.

# %%
variable_names = {
    "STARFIT": [
        "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
        "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
        "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
        "Release_c", "Release_p1", "Release_p2",
    ],
    "RBF": [
        "rbf1_center_storage", "rbf1_center_inflow", "rbf1_center_doy",
        "rbf1_scale_storage",  "rbf1_scale_inflow",  "rbf1_scale_doy", "rbf1_weight",
        "rbf2_center_storage", "rbf2_center_inflow", "rbf2_center_doy",
        "rbf2_scale_storage",  "rbf2_scale_inflow",  "rbf2_scale_doy", "rbf2_weight",
    ],
    "PWL": [
        "storage_x1", "storage_x2", "storage_theta1", "storage_theta2", "storage_theta3",
        "inflow_x1",  "inflow_x2",  "inflow_theta1",  "inflow_theta2",  "inflow_theta3",
        "season_x1",  "season_x2",  "season_theta1", "season_theta2", "season_theta3",
    ],
}

for pol, names in variable_names.items():
    print(f"{pol} parameter order ({len(names)} params):")
    print("  " + ", ".join(names))
    print()

# %% [markdown]
# ## 2. Paste in best parameter sets
#
# Here you will define a nested dictionary:
#
# ```python
# BEST_PARAMS = {
#     "fewalter": {
#         "STARFIT": np.array([...]),
#         "PWL":     np.array([...]),
#         "RBF":     np.array([...]),
#     },
#     "beltzvilleCombined": { ... },
#     "blueMarsh": { ... },
#     "prompton": { ... },
# }
# ```
#
# For this tutorial we will focus on **STARFIT**, but the structure works the
# same for `"RBF"` and `"PWL"`.

# %%
BEST_PARAMS: Dict[str, Dict[str, np.ndarray]] = {
    # Fill these in with your selected "best" parameter vectors
    "fewalter": {
        "STARFIT": np.array([
            # NORhi_mu, NORhi_min, ..., Release_p2
            # TODO: paste your 17 STARFIT parameters here
        ], dtype=float),
    },
    "beltzvilleCombined": {
        "STARFIT": np.array([
            # TODO
        ], dtype=float),
    },
    "blueMarsh": {
        "STARFIT": np.array([
            # TODO
        ], dtype=float),
    },
    "prompton": {
        "STARFIT": np.array([
            # TODO
        ], dtype=float),
    },
}

# Quick sanity check on counts
for res, d in BEST_PARAMS.items():
    for pol, theta in d.items():
        expected = len(variable_names[pol])
        if theta.size != expected:
            print(f"[WARN] {res}/{pol}: expected {expected} params, got {theta.size}")
        else:
            print(f"[OK] {res}/{pol}: {theta.size} parameters")

# %% [markdown]
# ## 3. Helper: build the inline `release_policy_dict`
#
# To override a reservoir's release rule, pass an `options` dict to `ModelBuilder`:
#
# ```python
# options = {
#     "release_policy_dict": {
#         "fewalter": {
#             "class_type": "ParametricReservoirRelease",
#             "policy_type": "STARFIT",   # or "RBF"/"PWL"
#             "policy_id":   "inline",
#             "params":      "...comma-separated floats...",
#         },
#         "blueMarsh": { ... },
#     }
# }
# ```
#
# Below we write a helper that converts numpy vectors → comma-separated strings
# and builds this dictionary for any subset of reservoirs.

# %%
def build_release_policy_dict(
    reservoirs: List[str],
    policy_type: str,
    param_source: Dict[str, Dict[str, np.ndarray]],
    policy_id: str = "inline",
) -> Dict[str, Dict[str, str]]:
    """
    Build the `release_policy_dict` for ModelBuilder from BEST_PARAMS.

    Parameters
    ----------
    reservoirs : list of reservoir names
    policy_type : "STARFIT", "RBF", or "PWL"
    param_source : nested dict like BEST_PARAMS[res][policy_type] = np.ndarray
    policy_id : tag written into the model (used for CSV id etc.)

    Returns
    -------
    release_policy_dict : dict
        {res: {"class_type": "...", "policy_type": "...", "policy_id": "...",
               "params": "p1,p2,..."}}
    """
    rpd: Dict[str, Dict[str, str]] = {}
    for res in reservoirs:
        if res not in param_source or policy_type not in param_source[res]:
            raise KeyError(f"No parameters for {res}/{policy_type} in param_source.")
        theta = np.asarray(param_source[res][policy_type], float).ravel()
        params_str = ",".join(f"{float(x):.8g}" for x in theta)
        rpd[res] = {
            "class_type": "ParametricReservoirRelease",
            "policy_type": policy_type,
            "policy_id":   policy_id,
            "params":      params_str,
        }
    return rpd

# Example: build for all 4 reservoirs, STARFIT only
STARFIT_RES = ["fewalter", "beltzvilleCombined", "blueMarsh", "prompton"]
release_policy_dict_starfit = build_release_policy_dict(
    STARFIT_RES,
    policy_type="STARFIT",
    param_source=BEST_PARAMS,
)
release_policy_dict_starfit

# %% [markdown]
# ## 4. Run the **parametric** Pywr-DRB model
#
# We now create a `ModelBuilder` with an `options` dict that includes our
# `release_policy_dict`. This overrides the built-in release rules for the
# selected reservoirs, replacing them with `ParametricReservoirRelease`.

# %%
def run_pywr_parametric_once(
    reservoirs: List[str],
    policy_type: str,
    param_source: Dict[str, Dict[str, np.ndarray]],
    pywr_start: str = "2019-01-01",
    pywr_end:   str = "2023-12-31",
    inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    work_dir: Path | str = (FIG_DIR / "_pywr_parametric_example"),
):
    work_dir   = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_models = PYWR_DRB_REPO / "_tmp_models"
    tmp_models.mkdir(parents=True, exist_ok=True)

    # Build release_policy_dict
    rpd = build_release_policy_dict(reservoirs, policy_type, param_source)

    tag        = f"Parametric_{policy_type}_{'_'.join(reservoirs)}"
    model_json = tmp_models / f"model_{tag}.json"
    h5         = work_dir / f"output_{tag}.hdf5"

    options = {
        "release_policy_dict": rpd
    }

    mb = pywrdrb.ModelBuilder(
        start_date=pywr_start,
        end_date=pywr_end,
        inflow_type=inflow_type,
        options=options,
    )
    mb.make_model()
    mb.write_model(str(model_json))

    model = pywrdrb.Model.load(str(model_json))
    rec   = pywrdrb.OutputRecorder(model, str(h5))  # noqa: F841
    _     = model.run()

    print(f"[✓] Parametric run finished. Results in: {h5}")
    return h5

# Actually run the parametric model for the four LB reservoirs
parametric_h5 = run_pywr_parametric_once(
    STARFIT_RES,
    policy_type="STARFIT",
    param_source=BEST_PARAMS,
)

# %% [markdown]
# ## 5. (Optional) Run the default model for comparison
#
# If you want a baseline, you can build the same model **without** passing
# `release_policy_dict`. Everything else stays the same.

# %%
def run_pywr_default(
    pywr_start: str = "2019-01-01",
    pywr_end:   str = "2023-12-31",
    inflow_type: str = "pub_nhmv10_BC_withObsScaled",
    work_dir: Path | str = (FIG_DIR / "_pywr_parametric_example"),
):
    work_dir   = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_models = PYWR_DRB_REPO / "_tmp_models"
    tmp_models.mkdir(parents=True, exist_ok=True)

    tag        = "DefaultRelease_all"
    model_json = tmp_models / f"model_{tag}.json"
    h5         = work_dir / f"output_{tag}.hdf5"

    mb = pywrdrb.ModelBuilder(
        start_date=pywr_start,
        end_date=pywr_end,
        inflow_type=inflow_type,
        options=None,     # no overrides
    )
    mb.make_model()
    mb.write_model(str(model_json))

    model = pywrdrb.Model.load(str(model_json))
    rec   = pywrdrb.OutputRecorder(model, str(h5))  # noqa: F841
    _     = model.run()

    print(f"[✓] Default run finished. Results in: {h5}")
    return h5

default_h5 = run_pywr_default()

# %% [markdown]
# ## 6. Load Pywr outputs and compare to observations
#
# We now use `pywrdrb.Data` to pull out:
#
# - `res_storage` for a given reservoir
# - `reservoir_downstream_gage` (releases at the gage)
#
# and compare to observed series from `get_observational_training_data`.

# %%
def load_pywr_res_timeseries(
    h5_path: Path | str,
    reservoir: str,
) -> Tuple[pd.Series, pd.Series]:
    h5_path = Path(h5_path)
    dataP = pywrdrb.Data(
        print_status=False,
        results_sets=["res_storage", "reservoir_downstream_gage"],
        output_filenames=[str(h5_path)],
    )
    dataP.load_output()
    key = h5_path.stem

    R = dataP.reservoir_downstream_gage[key][0][reservoir].astype(float)
    S = dataP.res_storage[key][0][reservoir].astype(float)
    R.name = "release_model"
    S.name = "storage_model"
    return S, R

def get_obs_timeseries(
    reservoir: str,
    start: str,
    end: str,
    inflow_type: str = "inflow_pub",
) -> Tuple[pd.Series, Optional[pd.Series]]:
    inflow_df, release_df, storage_df = get_observational_training_data(
        reservoir_name=reservoir,
        data_dir=PROCESSED_DATA_DIR,
        as_numpy=False,
        inflow_type=inflow_type,
    )
    slicer = slice(start, end)
    S_obs = (storage_df.loc[slicer][reservoir]
             if reservoir in storage_df.columns
             else storage_df.loc[slicer].iloc[:, 0])
    R_obs = (release_df.loc[slicer][reservoir]
             if (release_df is not None and reservoir in release_df.columns)
             else None)
    return S_obs, R_obs

# %% [markdown]
# ## 7. Quick comparison plot for one reservoir
#
# Here we pick `fewalter` and overlay:
#
# - Observed storage & release
# - Parametric Pywr-DRB storage & release
# - (Optionally) Default-release Pywr-DRB storage & release
#
# You can adapt this cell to add NOR bands, zoom windows, etc.

# %%
RES = "fewalter"
plot_start = "2020-01-01"
plot_end   = "2021-12-31"
slicer = slice(plot_start, plot_end)

# Observations
S_obs, R_obs = get_obs_timeseries(RES, plot_start, plot_end)

# Parametric Pywr
S_param, R_param = load_pywr_res_timeseries(parametric_h5, RES)
S_param = S_param.loc[slicer]
R_param = R_param.loc[slicer]

# Default Pywr (optional)
S_def, R_def = load_pywr_res_timeseries(default_h5, RES)
S_def = S_def.loc[slicer]
R_def = R_def.loc[slicer]

dates = S_obs.index

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)

# Releases
ax = axes[0]
if R_obs is not None:
    ax.plot(dates, R_obs.loc[slicer].values, label="Observed release", color="k", lw=1.4)
ax.plot(dates, R_param.values, label="Parametric STARFIT release", color="tab:orange", lw=1.4)
ax.plot(dates, R_def.values, label="Default release", color="tab:blue", lw=1.0, alpha=0.7)
ax.set_ylabel("Release (MGD)")
ax.set_title(f"{RES} — Releases (Pywr Parametric vs Default vs Obs)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Storage (in % of capacity, to match your NOR diagnostics)
cap = float(reservoir_capacity[RES])
S_obs_pct   = 100.0 * S_obs / cap
S_param_pct = 100.0 * S_param / cap
S_def_pct   = 100.0 * S_def / cap

ax = axes[1]
ax.plot(dates, S_obs_pct.values,   label="Observed storage", color="k", lw=1.4)
ax.plot(dates, S_param_pct.values, label="Parametric STARFIT storage", color="tab:orange", lw=1.4)
ax.plot(dates, S_def_pct.values,   label="Default storage", color="tab:blue", lw=1.0, alpha=0.7)
ax.set_ylabel("Storage (% capacity)")
ax.set_xlabel("Date")
ax.set_title(f"{RES} — Storage (% cap)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.show()
