"""
Single implementation for multi-reservoir Pywr-DRB parametric ``ModelBuilder`` runs.

Figures 4–6 (stage 1) and 7–11 (validation) both call :func:`run_pywr_parametric_multi` so cached
HDF5 paths, ``flow_prediction_mode``, and initial-storage options stay aligned.

HDF5 names include policy, pick slug, flow-mode tag, inflow type, and a **bundle tag** derived from
``CEE_FIG_SUBDIR``, Borg seed, and MRF-filter flags so different Borg bundles never share a file.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Mapping

import pandas as pd

import pywrdrb

from methods.config import PROCESSED_DATA_DIR, reservoir_capacity
from methods.postprocess.pywr_output_metadata import write_pywr_run_artifacts
from methods.load.observations import get_observational_training_data
from methods.plotting.pick_labels import pick_filename_slug
from methods.utils.policy_parameter_naming import safe_name

_PYWR_DEFAULT_PER_RES_FRAC: dict[str, float] = {
    "beltzvilleCombined": 0.8,
    "blueMarsh": 0.15,
    "fewalter": 0.1,
    "prompton": 0.1,
}

# Process-local cache: NWIS Prompton DV for a date range (avoids USGS on every HDF5 in bulk loops).
_PROMPTON_NWIS_DV_MGD: dict[tuple[str, str], pd.DataFrame] = {}


def flow_prediction_mode_short_tag(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m == "perfect_foresight":
        return "pfi"
    if m == "regression_disagg":
        return "reg"
    return safe_name(m)[:16]


def resolve_parametric_run_bundle_tag() -> str:
    fig_sub = os.environ.get("CEE_FIG_SUBDIR", "").strip() or "no_fig_subdir"
    seed = (
        os.environ.get("CEE_BORG_SEED", "").strip()
        or os.environ.get("CEE_SEED", "").strip()
        or "no_seed"
    )
    mrf = os.environ.get("CEE_BORG_MRF_FILTERED", "").strip()
    tag = os.environ.get("CEE_MRF_FILTER_TAG", "").strip()
    parts = [safe_name(fig_sub), f"seed{safe_name(seed)}"]
    if mrf:
        parts.append(f"mrf{safe_name(mrf)}")
    if tag:
        parts.append(safe_name(tag))
    return "_".join(parts)


def parametric_hdf5_stem(
    policy: str,
    pick_label: str,
    flow_prediction_mode: str,
    inflow_type: str,
    *,
    bundle_tag: str | None = None,
) -> str:
    bundle = bundle_tag if bundle_tag is not None else resolve_parametric_run_bundle_tag()
    return "_".join(
        [
            "output_Parametric",
            safe_name(policy),
            pick_filename_slug(pick_label),
            flow_prediction_mode_short_tag(flow_prediction_mode),
            safe_name(inflow_type),
            bundle,
        ]
    )


def _env_initial_volume_from_obs() -> bool:
    v = os.environ.get("CEE_PYWR_INITIAL_VOLUME_FROM_OBS", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return True


def initial_volume_frac_dict_from_obs(
    reservoir_names: list[str],
    *,
    data_dir: str | None = None,
) -> dict[str, float]:
    root = data_dir or PROCESSED_DATA_DIR
    out: dict[str, float] = {}
    for res in reservoir_names:
        try:
            _, _, storage_df = get_observational_training_data(
                reservoir_name=res,
                data_dir=root,
                as_numpy=False,
                inflow_type="inflow_pub",
            )
            if storage_df is None or storage_df.empty:
                continue
            col = res if res in storage_df.columns else storage_df.columns[0]
            s0 = float(storage_df[col].iloc[0])
            cap = float(reservoir_capacity[res])
            if cap <= 0 or not (s0 == s0):
                continue
            out[res] = max(0.0, min(1.0, s0 / cap))
        except Exception:
            continue
    return out


def _merge_initial_volume_options(release_policy_dict: Mapping[str, Any]) -> dict[str, float] | None:
    if not _env_initial_volume_from_obs():
        return None
    names = set(release_policy_dict.keys())
    merged = {k: float(v) for k, v in _PYWR_DEFAULT_PER_RES_FRAC.items() if k in names}
    return merged or None


def maybe_fetch_prompton_obs_if_missing(
    data_obj,
    base_key,
    reservoir_name,
    scenario_id: int = 0,
    *,
    skip_fetch: bool = False,
):
    """Fill Prompton observed release from NWIS when missing. Use ``skip_fetch`` or ``CEE_SKIP_PROMPTON_NWIS=1`` for bulk HDF5 loops (Stage 3 manifest aggregation)."""
    if skip_fetch:
        return
    if os.environ.get("CEE_SKIP_PROMPTON_NWIS", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    if str(reservoir_name).lower() != "prompton":
        return
    try:
        df_sim_gage = data_obj.reservoir_downstream_gage[base_key][scenario_id]
        df_obs_gage = data_obj.reservoir_downstream_gage["obs"][0]
        if ("prompton" in df_obs_gage.columns) and (not df_obs_gage["prompton"].isna().all()):
            return
        from dataretrieval import nwis

        def _tz_naive(idx):
            idx = pd.to_datetime(idx)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            return idx

        idx_sim = _tz_naive(df_sim_gage.index)
        s_date, e_date = idx_sim.min().strftime("%Y-%m-%d"), idx_sim.max().strftime("%Y-%m-%d")
        site, param = "01429000", "00060"
        cache_key = (s_date, e_date)
        if cache_key in _PROMPTON_NWIS_DV_MGD:
            df_promp = _PROMPTON_NWIS_DV_MGD[cache_key].copy()
            target_index = df_obs_gage.index if len(df_obs_gage.index) else idx_sim
            df_promp = df_promp.reindex(target_index)
            df_obs_gage = df_obs_gage.copy()
            if len(df_obs_gage.index) == 0:
                df_obs_gage = pd.DataFrame(index=target_index)
            df_obs_gage.loc[:, "prompton"] = df_promp["prompton"].values
            data_obj.reservoir_downstream_gage["obs"][0] = df_obs_gage
            print("Prompton observations (MGD) from cache.")
            return

        print(f"Retrieving Prompton DV {param} {s_date}..{e_date} from NWIS...")
        df_raw = nwis.get_record(sites=site, start=s_date, end=e_date, parameterCd=param, service="dv")
        if df_raw is None or len(df_raw) == 0:
            print("NWIS returned no DV; skipping.")
            return
        df_raw.index = _tz_naive(pd.to_datetime(df_raw.index))
        df_raw.sort_index(inplace=True)
        col = next((c for c in df_raw.columns if "00060" in c and ("Mean" in c or c.endswith("_Mean"))), None)
        if col is None:
            non_qual = [c for c in df_raw.columns if "qual" not in c.lower()]
            col = non_qual[0] if non_qual else df_raw.columns[0]
        CFS_TO_MGD = 0.646317
        df_promp = df_raw[[col]].rename(columns={col: "prompton"}).astype(float)
        df_promp["prompton"] *= CFS_TO_MGD
        _PROMPTON_NWIS_DV_MGD[cache_key] = df_promp.copy()
        target_index = df_obs_gage.index if len(df_obs_gage.index) else idx_sim
        df_promp = df_promp.reindex(target_index)
        df_obs_gage = df_obs_gage.copy()
        if len(df_obs_gage.index) == 0:
            df_obs_gage = pd.DataFrame(index=target_index)
        df_obs_gage.loc[:, "prompton"] = df_promp["prompton"].values
        data_obj.reservoir_downstream_gage["obs"][0] = df_obs_gage
        print("Prompton appended to observations (MGD).")
    except Exception as e:
        print(f"Prompton NWIS append skipped: {e}")


def _parametric_result_from_loaded_data(
    dataP,
    kP: str,
    h5: str,
    release_policy_dict: dict,
    scenario_id: int = 0,
) -> dict:
    mf_map = dataP.major_flow[kP]
    mf = mf_map.get(scenario_id, mf_map.get(0))
    dfP_T = mf["delTrenton"].astype(float).rename("pywr_trenton") if "delTrenton" in mf.columns else None
    dfP_MRF = None
    if kP in dataP.lower_basin_mrf_contributions:
        lb_map = dataP.lower_basin_mrf_contributions[kP]
        dfP_MRF = lb_map.get(scenario_id, lb_map.get(0))
        if dfP_MRF is not None:
            dfP_MRF = dfP_MRF.copy()
    dfObs_T = (
        dataP.major_flow["obs"][0]["delTrenton"].astype(float).rename("obs_trenton")
        if ("obs" in dataP.major_flow and "delTrenton" in dataP.major_flow["obs"][0].columns)
        else None
    )
    dfP_target_T = None
    try:
        mrf_map = getattr(dataP, "mrf_targets", {}) or {}
        if kP in mrf_map:
            scen_mrf = mrf_map[kP]
            mtf = scen_mrf.get(scenario_id, scen_mrf.get(0))
            if mtf is not None and "mrf_target_delTrenton" in mtf.columns:
                dfP_target_T = mtf["mrf_target_delTrenton"].astype(float).rename("mrf_target_delTrenton")
    except Exception:
        dfP_target_T = None

    by_res = {}
    rel_map = dataP.res_release[kP]
    sto_map = dataP.res_storage[kP]
    rel = rel_map.get(scenario_id, rel_map.get(0))
    sto = sto_map.get(scenario_id, sto_map.get(0))
    for res in release_policy_dict.keys():
        if res not in rel.columns:
            raise KeyError(
                f"Reservoir {res!r} missing from res_release in {h5} "
                f"(scenario_id={scenario_id})"
            )
        dfP_R = rel[res].astype(float).rename("pywr_release")
        dfP_S = sto[res].astype(float).rename("pywr_storage")
        by_res[res] = (dfP_R, dfP_S)

    return {
        "by_res": by_res,
        "trenton": dfP_T,
        "mrf": dfP_MRF,
        "obs_trenton": dfObs_T,
        "mrf_target_trenton": dfP_target_T,
        "scenario_id": scenario_id,
    }


def parametric_result_from_h5_path(
    h5: str,
    release_policy_dict: dict,
    *,
    scenario_id: int = 0,
    fetch_prompton_nwis: bool = True,
) -> dict:
    """Load one HDF5 and return per-reservoir / Trenton series dict.

    ``fetch_prompton_nwis``: if False, skip USGS DV fetch for Prompton observed release (use for
    bulk manifest loops). Also respect ``CEE_SKIP_PROMPTON_NWIS=1``. When True, repeated same
    (start,end) date range uses a process-local cache after the first download.
    """
    results_sets = ["major_flow", "res_storage", "res_release", "reservoir_downstream_gage", "lower_basin_mrf_contributions"]
    dataP = pywrdrb.Data(print_status=False, results_sets=results_sets, output_filenames=[h5])
    dataP.load_output()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Resampling with a PeriodIndex is deprecated.*",
            category=FutureWarning,
        )
        dataP.load_observations()
    kP = os.path.splitext(os.path.basename(h5))[0]
    for res in release_policy_dict.keys():
        maybe_fetch_prompton_obs_if_missing(
            dataP,
            kP,
            res,
            scenario_id=scenario_id,
            skip_fetch=not fetch_prompton_nwis,
        )

    return _parametric_result_from_loaded_data(dataP, kP, h5, release_policy_dict, scenario_id)


def parametric_results_all_scenarios_from_h5(
    h5: str,
    release_policy_dict: dict,
    *,
    fetch_prompton_nwis: bool = True,
):
    from pywrdrb.utils.hdf5 import get_n_scenarios_from_pywrdrb_output_file

    n = int(get_n_scenarios_from_pywrdrb_output_file(h5))
    results_sets = ["major_flow", "res_storage", "res_release", "reservoir_downstream_gage", "lower_basin_mrf_contributions"]
    dataP = pywrdrb.Data(print_status=False, results_sets=results_sets, output_filenames=[h5])
    dataP.load_output()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Resampling with a PeriodIndex is deprecated.*",
            category=FutureWarning,
        )
        dataP.load_observations()
    kP = os.path.splitext(os.path.basename(h5))[0]
    for res in release_policy_dict.keys():
        maybe_fetch_prompton_obs_if_missing(
            dataP, kP, res, scenario_id=0, skip_fetch=not fetch_prompton_nwis
        )

    out = {}
    for sid in range(n):
        out[sid] = _parametric_result_from_loaded_data(dataP, kP, h5, release_policy_dict, sid)
    return out


def _artifact_extra_for_write(
    pywr_run_metadata: dict | None,
    *,
    initial_volume_frac_dict: dict[str, float] | None,
) -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    if pywr_run_metadata and pywr_run_metadata.get("extra"):
        ex = pywr_run_metadata.get("extra")
        if isinstance(ex, dict):
            extra.update(ex)
    if initial_volume_frac_dict:
        extra["initial_volume_frac_dict_applied"] = {
            k: round(float(v), 6) for k, v in initial_volume_frac_dict.items()
        }
    return extra if extra else None


def _write_run_artifacts(
    h5: str,
    *,
    release_policy_dict: dict,
    flow_prediction_mode: str,
    pywr_inflow_type: str | None,
    pywr_run_metadata: dict | None,
    initial_volume_frac_dict: dict[str, float] | None,
) -> None:
    extra = _artifact_extra_for_write(pywr_run_metadata, initial_volume_frac_dict=initial_volume_frac_dict)
    if pywr_run_metadata:
        pm = {k: v for k, v in pywr_run_metadata.items() if k != "extra"}
        write_pywr_run_artifacts(
            h5,
            release_policy_dict=release_policy_dict,
            flow_prediction_mode=flow_prediction_mode,
            pywr_inflow_type=pywr_inflow_type,
            extra=extra,
            **pm,
        )
    else:
        write_pywr_run_artifacts(
            h5,
            release_policy_dict=release_policy_dict,
            flow_prediction_mode=flow_prediction_mode,
            pywr_inflow_type=pywr_inflow_type,
            extra=extra,
        )


def run_pywr_parametric_multi(
    release_policy_dict: dict,
    start: str,
    end: str,
    inflow_type: str,
    work_dir: str,
    stem_base: str,
    flow_prediction_mode: str,
    *,
    pywr_run_metadata: dict | None = None,
    extra_model_options: dict | None = None,
    force_rerun: bool = False,
    keep_model_json: bool = False,
) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    h5 = os.path.join(work_dir, f"{stem_base}.hdf5")
    ens = (extra_model_options or {}).get("inflow_ensemble_indices")
    ens_note = f" | inflow_ensemble_indices={ens!r}" if ens is not None else ""
    iv_merged = _merge_initial_volume_options(release_policy_dict)
    iv_note = ""
    if iv_merged:
        iv_note = f" | initial_volume_frac_dict keys={sorted(iv_merged.keys())}"
    print(
        f"[pywr] Parametric reservoirs in this run: {sorted(release_policy_dict.keys())} "
        f"(n={len(release_policy_dict)}) | flow_prediction_mode={flow_prediction_mode}{ens_note}{iv_note}",
        flush=True,
    )

    if os.path.isfile(h5) and not force_rerun:
        try:
            import h5py

            # Close the probe handle before _write_run_artifacts: h5py cannot open the same path
            # for append while this process still holds a read-only HDF5 handle.
            reuse_ok = False
            with h5py.File(h5, "r") as f:
                reuse_ok = "time" in f.keys()
            if reuse_ok:
                print(f"[pywr] Reusing cached parametric HDF5: {h5}")
                _write_run_artifacts(
                    h5,
                    release_policy_dict=release_policy_dict,
                    flow_prediction_mode=flow_prediction_mode,
                    pywr_inflow_type=inflow_type,
                    pywr_run_metadata=pywr_run_metadata,
                    initial_volume_frac_dict=iv_merged,
                )
                return parametric_result_from_h5_path(h5, release_policy_dict, scenario_id=0)
        except Exception:
            pass

    options: dict[str, Any] = {
        "release_policy_dict": release_policy_dict,
        "flow_prediction_mode": flow_prediction_mode,
    }
    if iv_merged is not None:
        options["initial_volume_frac_dict"] = iv_merged
    if extra_model_options:
        options = {**options, **extra_model_options}
    mb = pywrdrb.ModelBuilder(start_date=start, end_date=end, inflow_type=inflow_type, options=options)
    mb.make_model()
    model_json = os.path.join(work_dir, f"model_{stem_base}.json")
    mb.write_model(model_json)
    model = pywrdrb.Model.load(model_json)
    if os.path.isfile(h5):
        try:
            os.remove(h5)
        except Exception:
            pass
    _ = pywrdrb.OutputRecorder(model, h5)
    _ = model.run()
    if not keep_model_json:
        try:
            os.remove(model_json)
        except Exception:
            pass
    print(
        f"[pywr] Wrote multi-reservoir parametric HDF5 ({len(release_policy_dict)} nodes): {h5}",
        flush=True,
    )
    _write_run_artifacts(
        h5,
        release_policy_dict=release_policy_dict,
        flow_prediction_mode=flow_prediction_mode,
        pywr_inflow_type=inflow_type,
        pywr_run_metadata=pywr_run_metadata,
        initial_volume_frac_dict=iv_merged,
    )
    return parametric_result_from_h5_path(h5, release_policy_dict, scenario_id=0)


def run_pywr_parametric_single_res(
    res_name: str,
    policy: str,
    params_vec,
    start: str,
    end: str,
    inflow_type: str,
    work_dir: str,
    flow_prediction_mode: str,
    stem_suffix: str,
) -> dict:
    import numpy as np

    rel = {
        res_name: {
            "class_type": "ParametricReservoirRelease",
            "policy_type": policy,
            "policy_id": "inline",
            "params": ",".join(str(x) for x in np.asarray(params_vec, float).tolist()),
        }
    }
    stem_base = f"output_Parametric_{policy}_{stem_suffix}"
    return run_pywr_parametric_multi(rel, start, end, inflow_type, work_dir, stem_base, flow_prediction_mode)
