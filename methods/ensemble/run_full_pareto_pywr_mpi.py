#!/usr/bin/env python3
"""
Run **all** filtered Pareto alignment rows through Pywr-DRB with one MPI rank per
combined-basin simulation (one ``release_policy_dict`` per filtered row × policy).

This mirrors :func:`methods.plotting.plot_pareto_ensemble_uncertainty.collect_pywr_ensemble_for_policy`
but shards work across MPI processes instead of looping serially, and names each HDF5 / model JSON
uniquely with alignment index + parameter fingerprint (so parallel runs never clobber files).

**Setup**

- Run from the CEE6400 project root (or put it on ``PYTHONPATH``). Pywr metadata helpers live in
  ``methods/postprocess/pywr_output_metadata.py`` (same module used by ``pywr_parametric_run``).
- Optional: set ``BASELINE_POLICY_OPTIMIZATION_ROOT`` if you need other Baseline-only modules on ``sys.path``.

**Environment** (same as Fig 12 / ensemble)

- ``CEE_FIGURE_POLICIES`` — comma-separated subset; default all (``STARFIT``, ``PWL``, ``RBF``).
- ``CEE_PYWR_INFLOW_TYPE`` — Pywr inflow type (default ``pub_nhmv10_BC_withObsScaled``).
- ``CEE_PYWR_FLOW_PREDICTION_MODE`` — ``regression_disagg`` / ``perfect_foresight`` / …
- ``CEE_FULL_PARETO_WORK_DIR`` — output directory for this script’s ``*.hdf5``, ``model_*.json``,
  manifests (default: ``pywr_data/full_pareto_runs``). **Not** ``CEE_PYWR_WORK_DIR``, which
  figures / ``04_make_figures`` use (default ``pywr_data/pywr_tmp_runs``).
- ``CEE_INFLOW_ENSEMBLE_INDICES`` — optional, comma-separated (passed to ``ModelBuilder``).
- ``CEE_FULL_PARETO_MAX_RUNS`` — cap sweep index per policy (default: sweep to end).
- ``CEE_FULL_PARETO_STRICT_ALIGNMENT`` — if ``1``, use the old Fig-12 rule: require the same row
  index in **all** reservoirs that have data, and set the sweep length to **min** row counts.
  Default **off**: sweep to **max** row count and include only reservoirs that have a row at
  each index (``release_policy_dict`` can have 1–4 parametric nodes).
- ``CEE_PYWR_KEEP_MODEL_JSON`` — ``1``/``true`` to keep ``model_<stem>.json`` after each run
  (same as ``--keep-model-json``).
- ``CEE_FULL_PARETO_MERGE_WAIT_SEC`` — rank 0 waits up to this many seconds for all per-rank
  manifest JSON files before merging (default ``604800``, i.e. 7 days; aligns with long Slurm runs).
- ``CEE_FULL_PARETO_MERGE_POLL_SEC`` — poll interval while waiting (default ``2``).

**Borg CSV paths** (same as ``04_make_figures`` / ``resolve_borg_moea_csv_path``)

- ``CEE_BORG_OUTPUT_DIR`` — directory containing ``MMBorg_*``.csv (default: project ``outputs/``).
- ``CEE_BORG_RUN_VARIANTS`` — comma-separated: ``full`` (no MRF suffix), ``regression``,
  ``perfect`` (default: all three). Each variant uses the matching CSVs, e.g. ``*_seed72.csv`` vs
  ``*_seed71_mrffiltered_regression.csv`` vs ``*_seed71_mrffiltered_perfect.csv``.
- Seeds for Borg CSV names are defined in ``pywrdrb.release_policies.config`` (``SEED``,
  ``BORG_SEED_FULL``); :mod:`methods.config` exposes ``BORG_SEED_MRF`` / ``BORG_SEED_FULL``.
  Optional env overrides (see ``borg_paths.borg_variant_resolve_kwargs``):

  - ``CEE_BORG_SEED_FULL`` — overrides ``BORG_SEED_FULL`` for **full** only.
  - ``CEE_BORG_SEED_FULL_TRY`` — extra seeds to probe after ``BORG_SEED_FULL`` / ``BORG_SEED_MRF``.
  - ``CEE_BORG_SEED`` / ``CEE_SEED`` — when set, used before ``BORG_SEED_MRF`` in path resolution.
  - ``CEE_BORG_SEED_REGRESSION`` / ``CEE_BORG_SEED_PERFECT`` / ``CEE_BORG_SEED_MRF``.

**Examples**

```bash
export BASELINE_POLICY_OPTIMIZATION_ROOT=/path/to/Baseline_Policy_Optimization
export CEE_FIG_SUBDIR=borg_full_series
export CEE_BORG_SEED=123
cd /path/to/CEE6400Project

# Dry-run: count jobs only
python -m methods.ensemble.run_full_pareto_pywr_mpi --dry-run

# 30 ranks, full filtered set, keep model JSON
export CEE_PYWR_KEEP_MODEL_JSON=1
mpirun -np 30 python -m methods.ensemble.run_full_pareto_pywr_mpi
```

Single-process (no MPI) runs all jobs on one rank::

  python -m methods.ensemble.run_full_pareto_pywr_mpi --no-mpi
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_baseline = os.environ.get("BASELINE_POLICY_OPTIMIZATION_ROOT", "").strip()
if _baseline:
    bp = os.path.abspath(_baseline)
    if bp not in sys.path:
        sys.path.insert(0, bp)

try:
    from mpi4py import MPI

    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()
except Exception:
    _COMM = None
    _RANK = 0
    _SIZE = 1


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


def _full_pareto_rank_manifest_paths(work_dir: str, size: int) -> List[str]:
    return [
        os.path.join(
            work_dir,
            f"_full_pareto_manifest_rank{r:04d}_of_{size:04d}.json",
        )
        for r in range(size)
    ]


def _wait_for_rank_manifest_files(
    paths: List[str],
    *,
    poll_sec: float,
    max_wait_sec: float,
    log_progress: bool,
) -> int:
    """Return the number of manifest paths that exist after waiting (up to ``max_wait_sec``)."""
    n_expected = len(paths)
    deadline = time.monotonic() + max_wait_sec
    last_log = 0.0
    while time.monotonic() < deadline:
        n_found = sum(1 for p in paths if os.path.isfile(p))
        if n_found >= n_expected:
            return n_found
        now = time.monotonic()
        if log_progress and now - last_log >= 60.0:
            print(
                f"[full_pareto_mpi] merge: waiting for rank manifests ({n_found}/{n_expected} on disk) ...",
                flush=True,
            )
            last_log = now
        time.sleep(poll_sec)
    return sum(1 for p in paths if os.path.isfile(p))


def _param_fingerprint(release_policy_dict: dict) -> str:
    parts = []
    for res in sorted(release_policy_dict.keys()):
        parts.append(str(release_policy_dict[res].get("params", "")))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def _ensemble_stem_slug(indices: Optional[List[int]]) -> str:
    """Match ``methods.ensemble.policy_manifest.ensemble_stem_slug`` (no external import)."""
    if not indices:
        return "scen0"
    idx = list(indices)
    if len(idx) <= 4:
        return "ens_" + "_".join(str(i) for i in idx)
    payload = ",".join(str(i) for i in idx)
    h = hashlib.md5(payload.encode()).hexdigest()[:12]
    return f"ens_n{len(idx)}_{h}"


def _parse_inflow_ensemble_indices_env() -> Optional[List[int]]:
    raw = os.environ.get("CEE_INFLOW_ENSEMBLE_INDICES", "").strip()
    if not raw:
        return None
    out = []  # type: List[int]
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.append(int(p))
    return out if out else None


def _pywr_flow_mode() -> str:
    v = os.environ.get("CEE_PYWR_FLOW_PREDICTION_MODE", "").strip()
    return v if v else "regression_disagg"


def _parse_policies(arg: Optional[str]) -> List[str]:
    raw = (arg or "").strip() or os.environ.get("CEE_FIGURE_POLICIES", "").strip()
    if raw:
        allowed = {"STARFIT", "PWL", "RBF"}
        out = []
        for tok in raw.split(","):
            t = tok.strip().upper()
            if t in allowed and t not in out:
                out.append(t)
        return out if out else ["STARFIT", "PWL", "RBF"]
    return ["STARFIT", "PWL", "RBF"]


def _print_zero_jobs_borg_hint() -> None:
    """Explain missing Borg CSVs when no aligned rows were loaded."""
    from methods.borg_paths import borg_variant_resolve_kwargs, resolve_borg_moea_csv_path

    print(
        "[full_pareto_mpi] No filtered Pareto rows — Borg MOEA CSVs were not found (or every load failed).",
        flush=True,
    )
    print(
        "  Set CEE_BORG_OUTPUT_DIR to the directory that contains your MMBorg_*.csv files.",
        flush=True,
    )
    for v in ("full", "regression", "perfect"):
        kw = borg_variant_resolve_kwargs(v)
        ex = resolve_borg_moea_csv_path(
            "STARFIT",
            "blueMarsh",
            seed=kw["borg_seed"],
            mrf_filtered=kw["borg_mrf_filtered"],
            mrf_filter_tag=kw["borg_mrf_filter_tag"],
        )
        print("  [{}] {}\n       exists: {}".format(v, ex, os.path.isfile(ex)), flush=True)
    print(
        "  Seeds: pywrdrb.config SEED & BORG_SEED_FULL; optional CEE_BORG_SEED_FULL, "
        "CEE_BORG_SEED_REGRESSION, CEE_BORG_SEED_PERFECT, CEE_BORG_SEED, CEE_SEED. "
        "See methods/borg_paths.borg_variant_resolve_kwargs.",
        flush=True,
    )


def _parse_run_variants(cli: Optional[str]) -> List[str]:
    """Borg objective variants: full | regression | perfect (see borg_paths.normalize_borg_variant)."""
    raw = (cli or "").strip() or os.environ.get("CEE_BORG_RUN_VARIANTS", "full,regression,perfect").strip()
    from methods.borg_paths import normalize_borg_variant

    out: List[str] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        can = normalize_borg_variant(t)
        if can not in out:
            out.append(can)
    return out if out else ["full", "regression", "perfect"]


def _strict_alignment() -> bool:
    """If True, match Fig 12 ensemble: min row counts, same index in every reservoir."""
    v = os.environ.get("CEE_FULL_PARETO_STRICT_ALIGNMENT", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _max_runs() -> Optional[int]:
    raw = os.environ.get("CEE_FULL_PARETO_MAX_RUNS", "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _build_jobs(
    solution_vars: dict,
    policies: List[str],
    *,
    max_runs: Optional[int],
    strict_alignment: bool,
) -> List[Tuple[str, int]]:
    from methods.plotting.plot_pareto_ensemble_uncertainty import PARAMETRIC_MOEA_RESERVOIR_KEYS

    jobs = []  # type: List[Tuple[str, int]]
    for policy in policies:
        moea_reservoirs = [
            k
            for k in PARAMETRIC_MOEA_RESERVOIR_KEYS
            if k in solution_vars and policy in solution_vars.get(k, {})
        ]
        lengths = []
        for res in moea_reservoirs:
            df = solution_vars.get(res, {}).get(policy)
            if df is None or df.empty:
                continue
            lengths.append(len(df))
        if not lengths:
            continue
        if strict_alignment:
            n_align = min(lengths)
        else:
            n_align = max(lengths)
        n_runs = n_align if max_runs is None else min(n_align, max_runs)
        for i in range(n_runs):
            jobs.append((policy, i))
    return jobs


def _run_one(
    policy: str,
    alignment_index: int,
    solution_vars: dict,
    *,
    borg_variant: str,
    strict_alignment: bool,
    pywr_inflow_type: str,
    work_dir: str,
    mode: str,
    extra_pywr_opts: Optional[dict],
    force_rerun: bool,
    keep_model_json: bool,
) -> Dict[str, Any]:
    from pywrdrb.utils.dates import model_date_ranges

    from methods.postprocess.pywr_parametric_run import run_pywr_parametric_multi
    from methods.postprocess.pywr_output_metadata import normalize_borg_row_label
    from methods.plotting.plot_pareto_ensemble_uncertainty import PARAMETRIC_MOEA_RESERVOIR_KEYS

    moea_reservoirs = [
        k
        for k in PARAMETRIC_MOEA_RESERVOIR_KEYS
        if k in solution_vars and policy in solution_vars.get(k, {})
    ]
    release_policy_dict: dict = {}
    row_labels: dict = {}
    row_ilocs: dict = {}
    for res in moea_reservoirs:
        var_df = solution_vars[res][policy]
        if alignment_index >= len(var_df):
            if strict_alignment:
                raise IndexError(
                    "strict alignment: missing row {} for {} / {} (len={})".format(
                        alignment_index, res, policy, len(var_df)
                    )
                )
            continue
        row = var_df.iloc[alignment_index]
        params = row.values.astype(float)
        release_policy_dict[res] = {
            "class_type": "ParametricReservoirRelease",
            "policy_type": policy,
            "policy_id": "inline",
            "params": ",".join(str(x) for x in params.tolist()),
        }
        row_labels[res] = normalize_borg_row_label(var_df.index[alignment_index])
        row_ilocs[res] = int(alignment_index)

    if not release_policy_dict:
        raise ValueError(
            "empty release_policy_dict at alignment_index={} policy={} — no reservoir had a row "
            "(check filtered CSVs)".format(alignment_index, policy)
        )

    pfp = _param_fingerprint(release_policy_dict)
    ens_idx = _parse_inflow_ensemble_indices_env()
    ens_slug = _ensemble_stem_slug(ens_idx)
    stem = (
        f"output_Parametric_{policy}_align{alignment_index:05d}_p{pfp}_{_safe(mode)}_{ens_slug}_borg{_safe(borg_variant)}"
    )

    start, end = model_date_ranges[pywr_inflow_type]
    start, end = str(start), str(end)

    pywr_run_metadata = {
        "policy_type": policy,
        "alignment_index": alignment_index,
        "row_index_labels_by_reservoir": row_labels,
        "row_indices_by_reservoir": row_ilocs,
        "extra": {
            "stem_base": stem,
            "pywr_work_dir": work_dir,
            "param_fingerprint": pfp,
            "borg_variant": borg_variant,
            "strict_alignment": strict_alignment,
            "n_parametric_nodes": len(release_policy_dict),
        },
    }

    t0 = time.perf_counter()
    _ = run_pywr_parametric_multi(
        release_policy_dict,
        start,
        end,
        pywr_inflow_type,
        work_dir,
        stem,
        mode,
        pywr_run_metadata=pywr_run_metadata,
        extra_model_options=extra_pywr_opts,
        force_rerun=force_rerun,
        keep_model_json=keep_model_json,
    )
    elapsed = time.perf_counter() - t0
    h5 = os.path.join(work_dir, f"{stem}.hdf5")
    mj = os.path.join(work_dir, f"model_{stem}.json")
    return {
        "ok": True,
        "borg_variant": borg_variant,
        "policy": policy,
        "alignment_index": alignment_index,
        "stem": stem,
        "hdf5": os.path.abspath(h5),
        "model_json": os.path.abspath(mj) if keep_model_json and os.path.isfile(mj) else None,
        "seconds": elapsed,
        "error": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MPI full-pareto Pywr-DRB runs (one rank per job).")
    ap.add_argument(
        "--policies",
        "--policy",
        dest="policies",
        default=None,
        help="Comma-separated STARFIT,PWL,RBF (default: CEE_FIGURE_POLICIES or all).",
    )
    ap.add_argument("--force-rerun", action="store_true", help="Ignore cached HDF5.")
    ap.add_argument(
        "--keep-model-json",
        action="store_true",
        help="Keep model_<stem>.json after each run (or set CEE_PYWR_KEEP_MODEL_JSON=1).",
    )
    ap.add_argument(
        "--no-mpi",
        action="store_true",
        help="Run all jobs on a single process (ignore MPI world size).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print job counts and exit.")
    ap.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Override CEE_FULL_PARETO_MAX_RUNS (cap rows per policy).",
    )
    ap.add_argument(
        "--variants",
        default=None,
        help="Comma-separated Borg variants: full, regression, perfect (default: CEE_BORG_RUN_VARIANTS or all three).",
    )
    args = ap.parse_args()

    rank = _RANK if not args.no_mpi else 0
    size = _SIZE if not args.no_mpi else 1

    policies = _parse_policies(args.policies)
    max_runs = args.max_runs if args.max_runs is not None else _max_runs()

    keep_json = args.keep_model_json or os.environ.get(
        "CEE_PYWR_KEEP_MODEL_JSON", ""
    ).strip().lower() in ("1", "true", "yes", "on")

    default_inflow = "pub_nhmv10_BC_withObsScaled"
    pywr_inflow = os.environ.get("CEE_PYWR_INFLOW_TYPE", default_inflow).strip() or default_inflow

    from methods.borg_paths import borg_variant_resolve_kwargs
    from methods.config import get_pywr_full_pareto_work_dir
    from methods.postprocess.figures_primary import (
        RESERVOIR_NAMES,
        load_filtered_borg_solution_tables,
    )

    run_variants = _parse_run_variants(args.variants)
    strict_alignment = _strict_alignment()

    if rank == 0:
        print(
            f"[full_pareto_mpi] rank={rank}/{size} policies={policies} "
            f"borg_variants={run_variants} "
            f"strict_alignment={strict_alignment} "
            f"max_runs={max_runs} "
            f"inflow={pywr_inflow!r} keep_model_json={keep_json}",
            flush=True,
        )

    solution_vars_by_variant = {}
    for variant in run_variants:
        vk = borg_variant_resolve_kwargs(variant)
        if rank == 0:
            print(
                f"[full_pareto_mpi] loading filtered Borg tables for variant={variant!r} (seed={vk['borg_seed']}, "
                f"mrf={vk['borg_mrf_filtered']!r}, mrf_tag={vk['borg_mrf_filter_tag']!r})",
                flush=True,
            )
        _solution_objs, sv, _adv_maps, _adv_cands = load_filtered_borg_solution_tables(
            RESERVOIR_NAMES, policies, verbose=(rank == 0), **vk
        )
        solution_vars_by_variant[variant] = sv
        if variant == "full" and rank == 0:
            n_align_full = len(_build_jobs(sv, policies, max_runs=None, strict_alignment=strict_alignment))
            if n_align_full == 0:
                print(
                    "[full_pareto_mpi] variant 'full' produced no aligned rows — unfiltered CSVs were "
                    "not found or empty. Typical fix: they use a different seed than MRF (e.g. "
                    "*_seed72.csv vs *_seed71_mrffiltered_*.csv). Set BORG_SEED_FULL in "
                    "pywrdrb.release_policies.config (or CEE_BORG_SEED_FULL), or omit 'full' via "
                    "CEE_BORG_RUN_VARIANTS=regression,perfect.",
                    flush=True,
                )

    jobs = []  # type: List[Tuple[str, str, int]]
    for variant in run_variants:
        sv = solution_vars_by_variant[variant]
        for policy, idx in _build_jobs(sv, policies, max_runs=max_runs, strict_alignment=strict_alignment):
            jobs.append((variant, policy, idx))

    if rank == 0:
        print(f"[full_pareto_mpi] total jobs: {len(jobs)}", flush=True)

    if len(jobs) == 0:
        if rank == 0:
            _print_zero_jobs_borg_hint()
        raise SystemExit(1)

    if args.dry_run:
        if rank == 0:
            for v, p, i in jobs[:20]:
                print(f"  [{v}] {p} align {i}")
            if len(jobs) > 20:
                print(f"  ... and {len(jobs) - 20} more")
        return

    my_jobs = jobs[rank::size]
    if rank == 0:
        print(f"[full_pareto_mpi] jobs per rank (approx): {len(jobs) // size + (1 if len(jobs) % size else 0)}", flush=True)

    work_dir = get_pywr_full_pareto_work_dir()
    os.makedirs(work_dir, exist_ok=True)
    if rank == 0:
        print(f"[full_pareto_mpi] work_dir={work_dir}", flush=True)
    mode = _pywr_flow_mode()
    ens_idx = _parse_inflow_ensemble_indices_env()
    extra_pywr_opts = {"inflow_ensemble_indices": ens_idx} if ens_idx is not None else None

    results = []  # type: List[Dict[str, Any]]
    for borg_variant, policy, alignment_index in my_jobs:
        rec = {
            "ok": False,
            "borg_variant": borg_variant,
            "policy": policy,
            "alignment_index": alignment_index,
            "error": None,
        }
        try:
            rec = _run_one(
                policy,
                alignment_index,
                solution_vars_by_variant[borg_variant],
                borg_variant=borg_variant,
                strict_alignment=strict_alignment,
                pywr_inflow_type=pywr_inflow,
                work_dir=work_dir,
                mode=mode,
                extra_pywr_opts=extra_pywr_opts,
                force_rerun=args.force_rerun,
                keep_model_json=keep_json,
            )
        except Exception as e:
            rec["error"] = f"{e}\n{traceback.format_exc()}"
            print(
                f"[full_pareto_mpi] FAIL rank={rank} [{borg_variant}] {policy}@{alignment_index}: {e}",
                flush=True,
            )
        results.append(rec)

    part_path = os.path.join(
        work_dir,
        f"_full_pareto_manifest_rank{rank:04d}_of_{size:04d}.json",
    )
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rank": rank,
                "size": size,
                "work_dir": os.path.abspath(work_dir),
                "n_jobs": len(my_jobs),
                "results": results,
            },
            f,
            indent=2,
        )
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    print(f"[full_pareto_mpi] rank={rank} wrote {part_path}", flush=True)

    # No MPI Barrier: collectives break on job cancel / OOM / rank death. Instead, **every** rank
    # waits until all per-rank manifest JSON files exist (or timeout), so fast ranks do not call
    # ``MPI_Finalize`` while slow ranks are still computing. Rank 0 then merges from disk.
    manifest_paths = _full_pareto_rank_manifest_paths(work_dir, size)
    max_wait = float(os.environ.get("CEE_FULL_PARETO_MERGE_WAIT_SEC", str(7 * 86400)))
    poll_sec = float(os.environ.get("CEE_FULL_PARETO_MERGE_POLL_SEC", "2"))
    n_found = _wait_for_rank_manifest_files(
        manifest_paths,
        poll_sec=poll_sec,
        max_wait_sec=max_wait,
        log_progress=(rank == 0),
    )

    if rank == 0:
        merged = []  # type: List[Dict[str, Any]]
        for pth in manifest_paths:
            if os.path.isfile(pth):
                with open(pth, encoding="utf-8") as f:
                    merged.extend(json.load(f).get("results", []))
        if n_found < size:
            print(
                f"[full_pareto_mpi] WARNING: found {n_found}/{size} rank manifest files "
                f"under {work_dir!r} — merged results are partial (missing ranks did not finish or "
                f"could not write).",
                flush=True,
            )
        out_manifest = os.path.join(work_dir, "_full_pareto_manifest.json")
        with open(out_manifest, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_total_jobs": len(jobs),
                    "borg_run_variants": run_variants,
                    "strict_alignment": strict_alignment,
                    "n_ranks": size,
                    "work_dir": os.path.abspath(work_dir),
                    "results": merged,
                },
                f,
                indent=2,
            )
        ok = sum(1 for x in merged if x.get("ok"))
        print(
            f"[full_pareto_mpi] merged manifest: {out_manifest} "
            f"({ok}/{len(merged)} ok)",
            flush=True,
        )


if __name__ == "__main__":
    main()
