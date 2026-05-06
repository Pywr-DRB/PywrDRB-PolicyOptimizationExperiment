"""
Figures **12–21** — full-Pareto HDF5 manifest suite (policy uncertainty, Trenton, diagnostics).

Lives at ``methods/figures_stage3`` (not under ``methods.plotting``) so importing it does not
execute the eager imports in ``methods.plotting`` (matplotlib, etc.). Output paths are listed in
:mod:`methods.figures_stage3.full_pareto_output_paths`.

**Entry points**

- :func:`stage3_analysis.run_stage3_full_pareto_analysis` — CLI backend (multipanels + diagnostics)
- :func:`multipanel_daily.plot_multipanel_daily_uncertainty`
- :func:`multipanel_monthly.plot_multipanel_monthly_uncertainty`
- :func:`data_loading.load_full_pareto_manifest`
- :func:`data_loading.aggregate_stage3_multipanels_from_manifest`

**CLI**

.. code-block:: bash

   python -m methods.figures_stage3.plot_stage3_full_pareto_figures --manifest pywr_data/full_pareto_runs/_full_pareto_manifest.json --out-dir figures --which all
"""

# Keep package import side-effects minimal. Import concrete utilities from
# submodules directly (e.g., ``from methods.figures_stage3.stage3_analysis import ...``)
# to avoid circular imports during stage-1/stage-2 module loading.
__all__ = []
