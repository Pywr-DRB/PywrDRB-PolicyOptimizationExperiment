"""
Sidecar JSON + HDF5 root attributes for Pywr-DRB runs (Borg row ids, reservoirs, flow mode).

Pywr's own HDF5 layout is unchanged; we add ``cee_*`` attrs and a ``*_cee_meta.json`` next to each ``.hdf5``.

Canonical copy for this project lives next to :mod:`methods.postprocess.pywr_parametric_run`.
Baseline’s ``methods/ensemble/pywr_output_metadata.py`` is the upstream twin — keep behavior aligned.
"""

import json
import os

import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Union

CEE_META_VERSION = 1


def normalize_borg_row_label(idx: Any) -> Union[int, str]:
    """Stable JSON-friendly Borg row index label (aligned with ensemble ``solution_key`` conventions)."""
    if isinstance(idx, (int, np.integer)):
        return int(idx)
    if isinstance(idx, float) and float(idx).is_integer():
        return int(idx)
    return str(idx)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def build_run_metadata_payload(
    *,
    h5_path: str,
    release_policy_dict: Mapping[str, Any],
    flow_prediction_mode: str,
    pywr_inflow_type: Optional[str] = None,
    pick_label: Optional[str] = None,
    policy_type: Optional[str] = None,
    row_indices_by_reservoir: Optional[Dict[str, Any]] = None,
    row_index_labels_by_reservoir: Optional[Dict[str, Any]] = None,
    alignment_index: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical dict written to JSON and embedded in HDF5 attrs."""
    payload: Dict[str, Any] = {
        "cee_meta_version": CEE_META_VERSION,
        "hdf5_path": os.path.abspath(h5_path),
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "flow_prediction_mode": flow_prediction_mode,
        "parametric_reservoirs": sorted(release_policy_dict.keys()),
        "n_parametric_nodes": len(release_policy_dict),
    }
    if pywr_inflow_type is not None:
        payload["pywr_inflow_type"] = pywr_inflow_type
    if policy_type is not None:
        payload["policy_type"] = str(policy_type).upper()
    if pick_label is not None:
        payload["pick_label"] = pick_label
    if alignment_index is not None:
        payload["alignment_index"] = int(alignment_index)
    if row_indices_by_reservoir:
        payload["borg_row_indices_by_reservoir"] = _json_safe(dict(row_indices_by_reservoir))
    if row_index_labels_by_reservoir:
        payload["borg_row_index_labels_by_reservoir"] = _json_safe(dict(row_index_labels_by_reservoir))
    if extra:
        payload["extra"] = _json_safe(extra)
    return payload


def cee_meta_json_path(h5_path: str) -> str:
    """Sidecar JSON path written by :func:`write_pywr_run_artifacts` (``<stem>_cee_meta.json``)."""
    p = os.path.abspath(h5_path)
    stem, _ = os.path.splitext(p)
    return "{}_cee_meta.json".format(stem)


def write_pywr_run_artifacts(
    h5_path: str,
    *,
    release_policy_dict: Mapping[str, Any],
    flow_prediction_mode: str,
    pywr_inflow_type: Optional[str] = None,
    pick_label: Optional[str] = None,
    policy_type: Optional[str] = None,
    row_indices_by_reservoir: Optional[Dict[str, Any]] = None,
    row_index_labels_by_reservoir: Optional[Dict[str, Any]] = None,
    alignment_index: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Write ``<stem>_cee_meta.json`` beside ``hdf5`` and set HDF5 root attributes ``cee_meta`` (JSON string)
    and ``cee_meta_version``.
    Returns path to JSON or None if HDF5 missing.
    """
    h5_path = os.path.abspath(h5_path)
    if not os.path.isfile(h5_path):
        return None

    payload = build_run_metadata_payload(
        h5_path=h5_path,
        release_policy_dict=release_policy_dict,
        flow_prediction_mode=flow_prediction_mode,
        pywr_inflow_type=pywr_inflow_type,
        pick_label=pick_label,
        policy_type=policy_type,
        row_indices_by_reservoir=row_indices_by_reservoir,
        row_index_labels_by_reservoir=row_index_labels_by_reservoir,
        alignment_index=alignment_index,
        extra=extra,
    )
    stem, _ = os.path.splitext(h5_path)
    json_path = "{}_cee_meta.json".format(stem)
    parent = os.path.dirname(json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    try:
        import h5py

        with h5py.File(h5_path, "a") as f:
            meta_str = json.dumps(payload, separators=(",", ":"))
            f.attrs["cee_meta_version"] = CEE_META_VERSION
            f.attrs["cee_meta"] = meta_str
    except Exception as e:
        print("[pywr] Warning: could not attach cee_meta attrs to {}: {}".format(h5_path, e), flush=True)

    return json_path


def print_policy_row_counts(solution_objs: dict, policy_types: List[str], label: str = "Borg") -> None:
    """Print rows-per-policy for each reservoir (after filter) for quick sanity checks."""
    lines = []
    for res in sorted(solution_objs.keys()):
        pols = solution_objs.get(res) or {}
        parts = []
        for pol in policy_types:
            df = pols.get(pol)
            n = len(df) if df is not None else 0
            parts.append("{}={}".format(pol, n))
        lines.append("  {}: ".format(res) + ", ".join(parts))
    print("[{}] rows per reservoir × policy (filtered solutions):".format(label), flush=True)
    for ln in lines:
        print(ln, flush=True)
