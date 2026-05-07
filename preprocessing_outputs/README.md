# Preprocessing Outputs Layout

This folder is organized by **optimization filtering mode** so downstream scripts map directly to run flags.

## Folder contract

- `filtering/regression_disagg/`
  - `lower_basin_mrf_contributions.csv` — raw daily lower-basin MRF contributions extracted from Pywr output.
  - `mrf_active_filter_daily.csv` — binary daily filter (0/1) derived from contributions with `epsilon`.
  - `mrf_contributions_regression_disagg.png` — diagnostic stacked-area plot of contributions.
  - `traceability.txt` — inflow type, flow mode, and source HDF5 path used for extraction/build.

- `filtering/perfect_foresight/`
  - same file roles as above, for perfect-foresight flow prediction mode.

- `pywr/`
  - local-only Pywr model JSON/HDF5 artifacts produced when MRF bundles are regenerated locally.
  - filenames include both inflow type and flow mode for traceability, e.g.
    `pywrdrb_output_pub_nhmv10_BC_withObsScaled_regression_disagg.hdf5`,
    `pywrdrb_output_pub_nhmv10_BC_withObsScaled_perfect_foresight.hdf5`.

## Key distinction: contributions vs active filter

- `lower_basin_mrf_contributions.csv` stores **continuous MGD contribution values** by reservoir and day.
- `mrf_active_filter_daily.csv` stores **binary active/inactive flags** by day after thresholding contributions.
- optimization consumes `mrf_active_filter_daily.csv` directly; no additional JSON artifact is required.

## Data loader and extraction path

MRF extraction uses `pywrdrb.Data(..., results_sets=["lower_basin_mrf_contributions"])` in
`methods/preprocessing/build_mrf_active_filters.py`.

## Rebuild command

From project root:

```bash
bash scripts/prepare_preprocessing_outputs.sh
FORCE_MRF_FILTER_BUILD=1 bash scripts/prepare_preprocessing_outputs.sh
```
