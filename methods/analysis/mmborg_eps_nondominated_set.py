#!/usr/bin/env python3
"""
Epsilon-nondominated subsets of MMBorg (Borg MOEA) CSVs using ``pareto`` (Woodruff & Herman;
``pip install pareto``).

Objectives and ε values **always** come from ``pywrdrb.release_policies.config`` —
:data:`MOEA_OBJECTIVE_CSV_KEYS` (or ``obj1``…``objN`` from :data:`OBJ_LABELS`) and
:data:`EPSILONS`. Raw ``MMBorg_*.csv`` ``obj*`` values are the **minimization** form passed
to the MOEA (e.g. ``neg_nse``, |PBIAS| — see :data:`METRICS` in that config). ``pareto.eps_sort``
is therefore called with **default minimization on every objective** (no ``maximize=``).
Do not use :data:`SENSES_ALL` to infer pareto sense: those flags describe physical preference
for plotting, not the sign on disk.

**Reservoir × all policies (recommended):** use ``--per-reservoir --out-dir …``. For **each**
reservoir, the script loads that reservoir’s Borg CSV for **every** policy (default: STARFIT,
PWL, RBF; override with ``--policies``), **concatenates all solution rows**, then runs **one**
ε-nondominated sort on that **combined** pool in objective space. So the nondominated set is
defined **per reservoir, across all policies** — not separate Pareto sets per policy file.

**Merged mode** (``--csv`` / ``--resolve`` + ``-O``) is different: it ε-sorts only the rows in
the file(s) **you** pass (e.g. one policy’s CSV, or several paths you list). It does **not**
automatically loop reservoirs or policies unless you use ``--per-reservoir``.

**Output**

* **Merged mode** (``--csv`` / ``--resolve`` + ``-O``): Same columns as the input Borg CSVs
  (``obj*``, ``var*``, …), **fewer rows** — the ε-nondominated subset. No new columns.
* **Per-reservoir mode** (``--per-reservoir --out-dir``): One file
  ``eps_nondominated_<reservoir>.csv`` with a **moea_policy** column. Policy types differ in
  the number of ``var*`` (and optional ``constr*``) columns; rows are aligned to the **union**
  of headers with empty cells where a policy has no such parameter.

**Plotting (same codebase as Fig 1–2)**

Raw ``obj*`` are in MOEA (minimized) form; :func:`methods.load.results.load_results` and
:func:`methods.load.results.load_results_with_metadata` flip NSE-style columns for **figure**
space. For a **single-policy** ε-CSV, use ``load_results(path, obj_labels=OBJ_LABELS, filter=True)``;
pass ``obj_df`` to functions such as :func:`methods.plotting.plot_pareto_front_comparison.plot_pareto_front_comparison`.
For **per-reservoir** files with ``moea_policy``, use ``load_results_with_metadata`` to get
``obj_df`` plus ``meta_df``, group rows by ``meta_df['moea_policy']``, build one objective
dataframe per policy, and pass those dataframes and labels to
``plot_pareto_front_comparison`` (same pattern as ``figures_primary`` Fig 1).
:func:`~methods.plotting.plot_parallel_axis.custom_parallel_coordinates` and the rest of the
Fig 2 pipeline take the same ``obj_df`` / ``var_df`` shapes as stage 1—point them at these
frames instead of ``solution_objs[reservoir][policy]`` from full ``MMBorg_*.csv`` loads.

**Minimal snippets (no extra scripts):**

Single-policy ε file (merged mode)::

    from methods.config import OBJ_LABELS
    from methods.load.results import load_results
    from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison

    obj_df, var_df = load_results(
        "outputs/pareto_eps/starfit_blueMarsh_eps.csv",
        obj_labels=OBJ_LABELS,
        filter=True,  # optional: same OBJ_FILTER_BOUNDS as stage 1
    )
    plot_pareto_front_comparison(
        [obj_df],
        ["STARFIT"],
        obj_cols=["Release NSE", "Storage NSE"],
        title="ε-nondominated subset",
        fname="figures/debug/eps_nd_example.png",
    )

Per-reservoir file with ``moea_policy`` (``load_results`` drops extra columns; keep metadata)::

    from methods.config import OBJ_LABELS
    from methods.load.results import load_results_with_metadata
    from methods.plotting.plot_pareto_front_comparison import plot_pareto_front_comparison

    obj_df, var_df, meta = load_results_with_metadata(
        "outputs/pareto_eps_nondominated/eps_nondominated_blueMarsh.csv",
        obj_labels=OBJ_LABELS,
        filter=True,
    )
    policies = ["STARFIT", "PWL", "RBF"]
    obj_dfs, labels = [], []
    for p in policies:
        mask = meta["moea_policy"].astype(str).str.upper() == p
        if mask.any():
            obj_dfs.append(obj_df.loc[mask].copy())
            labels.append(p)
    plot_pareto_front_comparison(
        obj_dfs,
        labels,
        obj_cols=["Release NSE", "Storage NSE"],
        title="ε-nondominated (all policies)",
        fname="figures/debug/eps_nd_blueMarsh.png",
    )

Examples::

  cd /path/to/CEE6400Project
  pip install pareto

  # Explicit CSV path(s), merged into one nondominated set
  python -m methods.analysis.mmborg_eps_nondominated_set \\
      --csv outputs/MMBorg_4M_STARFIT_blueMarsh_nfe30000_seed71_mrffiltered_regression.csv \\
      -O outputs/pareto_eps/starfit_blueMarsh_eps.csv

  # Resolved path from policy / reservoir / variant
  python -m methods.analysis.mmborg_eps_nondominated_set --resolve --policy STARFIT --reservoir blueMarsh \\
      -O outputs/pareto_eps/one.csv

  # All reservoirs × all policies (combined per reservoir)
  python -m methods.analysis.mmborg_eps_nondominated_set --per-reservoir --out-dir outputs/pareto_eps \\
      --borg-variant regression --print-counts
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_csv_table(path: Path) -> Tuple[List[str], List[List[Any]]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _cell_float_or_str(c: Any) -> Any:
    if c is None or (isinstance(c, str) and not c.strip()):
        return ""
    if isinstance(c, str):
        try:
            return float(c)
        except ValueError:
            return c
    return c


def _numeric_suffix(prefix: str, col: str) -> int:
    if not col.startswith(prefix):
        return -1
    tail = col[len(prefix) :]
    return int(tail) if tail.isdigit() else -1


def _canonical_header_union(headers: List[List[str]]) -> List[str]:
    """
    Stable column order for merging Borg CSVs across policies (STARFIT vs PWL vs RBF have
    different ``var*`` counts). Order: ``obj*``, ``var*``, ``constr*``, other, ``moea_policy`` last.
    """
    all_c = set()
    for h in headers:
        all_c.update(h)
    objs = sorted([c for c in all_c if c.startswith("obj")], key=lambda c: _numeric_suffix("obj", c))
    vars_ = sorted([c for c in all_c if c.startswith("var")], key=lambda c: _numeric_suffix("var", c))
    constrs = sorted([c for c in all_c if c.startswith("constr")], key=lambda c: _numeric_suffix("constr", c))
    known = set(objs) | set(vars_) | set(constrs) | {"moea_policy"}
    other = sorted(c for c in all_c if c not in known)
    out = objs + vars_ + constrs + other
    if "moea_policy" in all_c:
        out.append("moea_policy")
    return out


def _project_row_to_header(header: List[str], row: List[Any], canonical: List[str]) -> List[Any]:
    if len(header) != len(row):
        raise SystemExit(
            "Row length {} does not match header length {} (columns {})".format(len(row), len(header), header)
        )
    d = {header[i]: row[i] for i in range(len(header))}
    return [d.get(c, "") for c in canonical]


def _pareto_spec_from_release_policy_config(header: Sequence[str]) -> Tuple[List[int], List[float]]:
    """
    Column indices for ``obj*`` objectives and ε list for ``pareto.eps_sort``.

    Borg writes minimization objectives to disk (``METRICS`` in release_policies config,
    e.g. ``neg_nse``); we pass **no** ``maximize=`` so pareto minimizes every objective —
    matching the MOEA. This path does **not** apply ``load_results`` sign flips used for figures.
    """
    from pywrdrb.release_policies import config as rp_cfg

    EPSILONS = rp_cfg.EPSILONS
    OBJ_LABELS = rp_cfg.OBJ_LABELS
    try:
        keys = list(rp_cfg.MOEA_OBJECTIVE_CSV_KEYS)
    except AttributeError:
        keys = [f"obj{i}" for i in range(1, len(OBJ_LABELS) + 1)]

    if len(EPSILONS) != len(keys):
        raise SystemExit(
            "Config mismatch: len(EPSILONS) != len(MOEA_OBJECTIVE_CSV_KEYS); check release_policies/config.py"
        )
    obj_cols: List[int] = []
    for k in keys:
        if k not in header:
            raise SystemExit(
                "CSV header missing objective column {!r} (expected from release_policies config). "
                "Got columns: {}".format(k, list(header)[:40])
            )
        obj_cols.append(header.index(k))

    return obj_cols, list(EPSILONS)


def _tag_rows_with_policy(
    header: List[str],
    data: List[List[Any]],
    policy: str,
    tag_col: str = "moea_policy",
) -> Tuple[List[str], List[List[Any]]]:
    out_header = list(header) + [tag_col]
    out_rows = [list(map(_cell_float_or_str, row)) + [policy] for row in data]
    return out_header, out_rows


def _run_eps_sort_on_rows(
    rows: List[List[Any]],
    obj_cols: List[int],
    eps: List[float],
) -> List[List[Any]]:
    import pareto

    if len(eps) != len(obj_cols):
        raise SystemExit("epsilons length must match number of objective columns")
    table = [[_cell_float_or_str(c) for c in row] for row in rows]
    # All objectives minimized (Borg CSV form); matches pareto default.
    return pareto.eps_sort([table], obj_cols, eps)


def _per_reservoir_mode(args: argparse.Namespace) -> None:
    from pywrdrb.release_policies.config import policy_type_options, reservoir_options

    from methods.borg_paths import borg_variant_resolve_kwargs, normalize_borg_variant, resolve_borg_moea_csv_path

    out_root = Path(os.path.abspath(args.out_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    reservoirs = (
        [x.strip() for x in args.reservoirs.split(",") if x.strip()]
        if args.reservoirs
        else list(reservoir_options)
    )
    policies = (
        [x.strip().upper() for x in args.policies.split(",") if x.strip()]
        if args.policies
        else list(policy_type_options)
    )

    v = normalize_borg_variant(args.borg_variant)
    bkw = borg_variant_resolve_kwargs(v)

    for res in reservoirs:
        batches: List[Tuple[List[str], List[List[Any]]]] = []
        n_files = 0
        for pol in policies:
            pth = resolve_borg_moea_csv_path(
                pol,
                res,
                seed=bkw["borg_seed"],
                mrf_filtered=bkw["borg_mrf_filtered"],
                mrf_filter_tag=bkw["borg_mrf_filter_tag"],
            )
            p = Path(pth)
            if not p.is_file():
                print(f"[mmborg_eps] skip missing {pol} / {res}: {p}", file=sys.stderr, flush=True)
                continue
            h, data = _load_csv_table(p)
            if not h or not data:
                print(f"[mmborg_eps] skip empty {p}", file=sys.stderr, flush=True)
                continue
            hh, tagged = _tag_rows_with_policy(h, data, pol)
            batches.append((hh, tagged))
            n_files += 1

        if not batches:
            print(f"[mmborg_eps] no inputs for reservoir={res!r} — skipping", file=sys.stderr, flush=True)
            continue

        combined_header = _canonical_header_union([b[0] for b in batches])
        combined_rows: List[List[Any]] = []
        for hh, tagged in batches:
            for row in tagged:
                combined_rows.append(_project_row_to_header(hh, row, combined_header))

        obj_cols, eps = _pareto_spec_from_release_policy_config(combined_header)
        nd = _run_eps_sort_on_rows(combined_rows, obj_cols, eps)
        out_path = out_root / "eps_nondominated_{}.csv".format(res)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(combined_header)
            for row in nd:
                w.writerow(row)
        if args.print_counts:
            print(
                "[mmborg_eps] reservoir={} files={} rows_in={} nondominated={} eps={} obj_cols={} -> {}".format(
                    res,
                    n_files,
                    len(combined_rows),
                    len(nd),
                    eps,
                    obj_cols,
                    out_path,
                ),
                file=sys.stderr,
                flush=True,
            )


def _merge_csvs_mode(args: argparse.Namespace) -> None:
    import pareto

    from methods.borg_paths import borg_variant_resolve_kwargs, normalize_borg_variant, resolve_borg_moea_csv_path

    if args.resolve:
        if not args.policy or not args.reservoir:
            raise SystemExit("--resolve requires --policy and --reservoir")
        v = normalize_borg_variant(args.borg_variant)
        kwargs = borg_variant_resolve_kwargs(v)
        paths = [
            Path(
                resolve_borg_moea_csv_path(
                    args.policy.strip().upper(),
                    args.reservoir.strip(),
                    seed=kwargs["borg_seed"],
                    mrf_filtered=kwargs["borg_mrf_filtered"],
                    mrf_filter_tag=kwargs["borg_mrf_filter_tag"],
                )
            )
        ]
    else:
        paths = [Path(os.path.abspath(p)) for p in (args.csv or [])]

    for p in paths:
        if not p.is_file():
            raise SystemExit("Input not found: {}".format(p))

    tables: List[List[List[Any]]] = []
    header: Optional[List[str]] = None
    total_in = 0
    for p in paths:
        h, data = _load_csv_table(p)
        if header is None:
            header = h
        elif h != header:
            raise SystemExit("Header mismatch: {} vs {}".format(paths[0], p))
        tables.append(data)
        total_in += len(data)

    if not header or total_in == 0:
        raise SystemExit("No data rows to sort.")

    obj_cols, eps = _pareto_spec_from_release_policy_config(header)

    tables_ll = [[[_cell_float_or_str(c) for c in row] for row in tab] for tab in tables]

    nondominated = pareto.eps_sort(tables_ll, obj_cols, eps)

    out_path = Path(os.path.abspath(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in nondominated:
            w.writerow(row)

    if args.print_counts:
        print(
            "[mmborg_eps] inputs={} rows_in={} nondominated={} objectives={} epsilons={} -> {}".format(
                len(paths), total_in, len(nondominated), obj_cols, eps, out_path
            ),
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Epsilon-nondominated sort for MMBorg CSVs (pareto.eps_sort). "
            "Objectives and epsilons come from pywrdrb.release_policies.config; "
            "raw obj* columns are minimized (MOEA / neg_nse form) — pareto default, no maximize=."
        ),
    )
    ap.add_argument(
        "--per-reservoir",
        action="store_true",
        help="For each reservoir, pool all policies' CSVs and write eps_nondominated_<reservoir>.csv under --out-dir.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="With --per-reservoir: output directory for one CSV per reservoir (required).",
    )
    ap.add_argument(
        "--reservoirs",
        default=None,
        help="Comma-separated reservoir keys (default: all from pywrdrb release_policies.config).",
    )
    ap.add_argument(
        "--policies",
        default=None,
        help="Comma-separated STARFIT,PWL,RBF (default: all from config).",
    )
    ap.add_argument(
        "--borg-variant",
        default="regression",
        help="full | regression | perfect (default: regression) for resolving CSV paths.",
    )

    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--csv",
        nargs="+",
        metavar="PATH",
        help="One or more Borg CSVs merged into one nondominated set.",
    )
    src.add_argument(
        "--resolve",
        action="store_true",
        help="Resolve a single CSV via --policy / --reservoir.",
    )

    ap.add_argument("--policy", default=None, help="With --resolve: policy type")
    ap.add_argument("--reservoir", default=None, help="With --resolve: reservoir key")

    ap.add_argument(
        "-O",
        "--output",
        default=None,
        help="Output CSV path (required for --csv / --resolve; not used with --per-reservoir).",
    )
    ap.add_argument("--print-counts", action="store_true", help="Print row counts to stderr.")

    args = ap.parse_args()

    if args.per_reservoir:
        if not args.out_dir:
            raise SystemExit("--per-reservoir requires --out-dir")
        if args.csv or args.resolve:
            raise SystemExit("Do not combine --per-reservoir with --csv or --resolve")
        if args.output:
            raise SystemExit("Use --out-dir with --per-reservoir, not -O/--output")
        _per_reservoir_mode(args)
        return

    if not args.csv and not args.resolve:
        raise SystemExit(
            "Specify --csv path(s), or --resolve with --policy and --reservoir, or use --per-reservoir"
        )
    if not args.output:
        raise SystemExit("Give -O/--output for merged output (or use --per-reservoir --out-dir)")

    _merge_csvs_mode(args)


if __name__ == "__main__":
    main()
