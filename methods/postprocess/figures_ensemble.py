#!/usr/bin/env python3
"""Former ensemble Fig 12 (``plot_pareto_ensemble_uncertainty``) was removed from ``04_make_figures.py``.

Full-Pareto HDF5 manifest figures (**12–21**) live in ``methods/figures_stage3`` — run::

    python -m methods.figures_stage3.plot_stage3_full_pareto_figures \\
      --manifest pywr_data/full_pareto_runs/_full_pareto_manifest.json \\
      --out-dir figures --which all

The plotting helpers in ``methods.plotting.plot_pareto_ensemble_uncertainty`` remain for reuse
(multipanel aggregation, envelopes); they are not wired to a standalone “Fig 12” pipeline anymore.
"""


def main_stage3() -> None:
    raise RuntimeError(
        "Ensemble Figure 12 was removed from the main figure pipeline. "
        "Run full-Pareto figures 12–21 with:\n"
        "  python -m methods.figures_stage3.plot_stage3_full_pareto_figures "
        "--manifest pywr_data/full_pareto_runs/_full_pareto_manifest.json "
        "--out-dir figures --which all"
    )


if __name__ == "__main__":
    main_stage3()
