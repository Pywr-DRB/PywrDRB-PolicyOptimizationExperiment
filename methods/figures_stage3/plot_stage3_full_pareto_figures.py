#!/usr/bin/env python3
"""
Full-Pareto manifest figures (figures **12–21**): multipanel daily/monthly + diagnostics.

Writes under each Borg bundle folder (``figures/borg_full_series``, ``figures/borg_mrffiltered_*``)
when splitting by variant — see ``methods.figures_stage3.full_pareto_output_paths``.

CLI wrapper around ``methods.figures_stage3.stage3_analysis.run_stage3_full_pareto_analysis``.

``CEE_TRENTON_TARGET_MGD`` is read in ``stage3_analysis`` / ``data_loading``; if unset,
``DEFAULT_TRENTON_TARGET_MGD`` from ``methods.figures_stage3.constants`` applies.

Examples::

  cd /path/to/CEE6400Project
  python -m methods.figures_stage3.plot_stage3_full_pareto_figures --mock --out-dir figures

  python -m methods.figures_stage3.plot_stage3_full_pareto_figures \\
      --manifest pywr_data/full_pareto_runs/_full_pareto_manifest.json \\
      --out-dir figures --which all

Env (optional)::

  STAGE3_SKIP_MONTHLY=1      # skip monthly multipanel only
  STAGE3_SKIP_DIAGNOSTICS=1  # skip diagnostic PNGs (bias, attribution, …)
  CEE_STAGE3_BORG_VARIANT    # regression | perfect | full — narrows manifest rows
  CEE_STAGE3_SPLIT_VARIANTS=0  # write one combined set in --out-dir (default: split by variant)

After MPI, ``run_full_pareto_pywr_mpi.sh`` calls this script once with manifest (default ``--which all``).
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from methods.config import FIG_DIR


def main():
    ap = argparse.ArgumentParser(
        description="Full-Pareto manifest figures 12–21 (multipanels + diagnostics)."
    )
    ap.add_argument(
        "--out-dir",
        default=FIG_DIR,
        help="Figure tree root (default: project figures/). Variant subfolders are created below this.",
    )
    ap.add_argument(
        "--which",
        choices=("daily", "monthly", "multipanels", "both", "diagnostics", "all"),
        default="all",
        help="daily | monthly | multipanels (=both) | diagnostics | all (default: all).",
    )
    ap.add_argument(
        "--mock",
        action="store_true",
        help="Synthetic multipanels only; diagnostics require --manifest.",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="Path to _full_pareto_manifest.json (HDF5-backed aggregation).",
    )
    ap.add_argument(
        "--borg-variant",
        default=None,
        help="Optional filter: full | regression | perfect. Or set CEE_STAGE3_BORG_VARIANT.",
    )
    ap.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Cap manifest HDF5 rows loaded per policy (quicker tests).",
    )
    ap.add_argument(
        "--no-split-borg-variants",
        action="store_true",
        help="Do not create borg_full_series / borg_mrffiltered_* subdirs; aggregate all manifest rows into --out-dir.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.no_split_borg_variants:
        os.environ["CEE_STAGE3_SPLIT_VARIANTS"] = "0"

    skip_monthly = os.environ.get("STAGE3_SKIP_MONTHLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    skip_diag = os.environ.get("STAGE3_SKIP_DIAGNOSTICS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    from methods.figures_stage3.stage3_analysis import run_stage3_full_pareto_analysis

    which = args.which
    if which == "both":
        which = "multipanels"

    paths = run_stage3_full_pareto_analysis(
        manifest=args.manifest,
        out_dir=args.out_dir,
        mock=args.mock,
        borg_variant=args.borg_variant,
        max_runs=args.max_runs,
        which=which,
        skip_monthly=skip_monthly,
        skip_diagnostics=skip_diag,
    )
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
