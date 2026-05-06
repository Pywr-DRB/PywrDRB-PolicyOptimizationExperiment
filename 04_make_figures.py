#!/usr/bin/env python3
"""
04 — Main figure entrypoint for this repository.

What this script does:

- runs stage 1 (`methods.postprocess.figures_primary`) for figures **1–6**
- runs stage 2 (`methods.postprocess.figures_validation`) for figures **7–11**

Figures **14–23** (full-Pareto HDF5 manifest ensemble: multipanels + diagnostics) are **not** run here.
Generate them after the MPI sweep with::

    python -m methods.figures_stage3.plot_stage3_full_pareto_figures \\
      --manifest pywr_data/full_pareto_runs/_full_pareto_manifest.json \\
      --out-dir figures --which all

Reads Borg optimization CSVs and cached simulation artifacts; does not run Pywr ``model.run()`` unless
not in plots-only mode (see ``CEE_PLOTS_ONLY`` / ``CEE_SKIP_SIMULATIONS``).

``CEE_TRENTON_TARGET_MGD`` is still used by validation plotting and by ``methods/figures_stage3`` when
you run full-Pareto figures separately.

Examples::

  python 04_make_figures.py --figures 1-6 7-11
  python 04_make_figures.py --figures 7-11 --plots-only
  CEE_FIG_SUBDIR=borg_full_series python 04_make_figures.py

Selected picks (one Pywr HDF5 per policy × focal solution) for dynamics / validation figures::

  python -m methods.ensemble.postprocess_sim simulate --mode selected

Validation figures only::

  python -m methods.postprocess.figures_validation --help

Environment (see ``methods/borg_paths.py``, ``methods/pipeline_env.py``):

- ``CEE_FIGURE_NUMBERS`` — set implicitly from ``--figures`` (comma/ranges), or pass manually.
- ``CEE_PLOTS_ONLY`` / ``--plots-only`` — prefer cached Pywr HDF5 (sets ``CEE_SKIP_PYWR`` when plots-only).
- ``CEE_FIG_SUBDIR`` — e.g. ``borg_full_series``, ``borg_mrffiltered_regression``, ``borg_mrffiltered_perfect_foresight``.
"""

import sys

# Must run before importing project code (needs 3.10+ for PEP 604 unions / list[] builtins in deps).
if sys.version_info < (3, 10):
    raise SystemExit(
        "CEE6400Project requires Python 3.10+ (3.11 recommended). "
        "On Hopper: `module load python/3.11.5` and activate this repo's venv "
        "(see README). "
        f"Current: {sys.executable} — {sys.version.split()[0]}"
    )

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.postprocess.figures_primary import main_stage1
from methods.postprocess.figures_validation import main as main_stage2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--figures",
        nargs="*",
        metavar="N",
        help="Figure numbers to build (e.g. 1 3 5-7). Default: all (unset CEE_FIGURE_NUMBERS).",
    )
    ap.add_argument(
        "--plots-only",
        action="store_true",
        help="Reuse cached simulations (sets CEE_PLOTS_ONLY=1 and CEE_SKIP_PYWR=1 unless overridden).",
    )
    ap.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip stage 1 (figures 1–6).",
    )
    ap.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip stage 2 (figures 7–11).",
    )
    ap.add_argument(
        "--skip-stage3",
        action="store_true",
        help="Deprecated no-op (former ensemble Fig 12 removed). Ignored.",
    )
    ap.add_argument(
        "--all-subsets",
        action="store_true",
        help=(
            "Run stage1+stage2 sequentially for all standard subsets: "
            "borg_full_series, borg_mrffiltered_regression, "
            "borg_mrffiltered_perfect_foresight."
        ),
    )
    args = ap.parse_args()

    if args.skip_stage3:
        print(
            "[04_make_figures] --skip-stage3 is deprecated (ensemble Fig 12 removed); ignoring.",
            flush=True,
        )

    if args.plots_only:
        os.environ["CEE_PLOTS_ONLY"] = "1"
        os.environ.setdefault("CEE_SKIP_PYWR", "1")

    if args.figures:
        os.environ["CEE_FIGURE_NUMBERS"] = ",".join(args.figures)

    def _run_once() -> None:
        if not args.skip_stage1:
            main_stage1()
        if not args.skip_stage2:
            main_stage2([])

    if args.all_subsets:
        subset_envs = (
            {
                "CEE_FIG_SUBDIR": "borg_full_series",
                "CEE_BORG_MRF_FILTERED": "0",
                "CEE_MRF_FILTER_TAG": "",
                "CEE_PYWR_FLOW_PREDICTION_MODE": "perfect_foresight",
            },
            {
                "CEE_FIG_SUBDIR": "borg_mrffiltered_regression",
                "CEE_BORG_MRF_FILTERED": "1",
                "CEE_MRF_FILTER_TAG": "regression_disagg",
                "CEE_PYWR_FLOW_PREDICTION_MODE": "regression_disagg",
            },
            {
                "CEE_FIG_SUBDIR": "borg_mrffiltered_perfect_foresight",
                "CEE_BORG_MRF_FILTERED": "1",
                "CEE_MRF_FILTER_TAG": "perfect",
                "CEE_PYWR_FLOW_PREDICTION_MODE": "perfect_foresight",
            },
        )
        for env_map in subset_envs:
            for key, val in env_map.items():
                os.environ[key] = val
            print(
                "[04_make_figures] subset:",
                os.environ["CEE_FIG_SUBDIR"],
                "CEE_BORG_MRF_FILTERED=" + os.environ["CEE_BORG_MRF_FILTERED"],
                "CEE_MRF_FILTER_TAG=" + os.environ.get("CEE_MRF_FILTER_TAG", ""),
                flush=True,
            )
            _run_once()
    else:
        _run_once()


if __name__ == "__main__":
    main()
