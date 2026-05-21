# Pareto solution selection (active modules)

| Module | Role |
|--------|------|
| **`selection_unified.py`** | Single source of truth — `build_unified_picks()` |
| **`selection_utils.py`** | Figure pipeline wrapper → calls `selection_unified` |
| **`pick_labels.py`** | Canonical labels and filename slugs |

**Post-processing:** `figures_primary.py`, `figures_validation.py` import `selection_utils` + `pick_labels`.

There is **no** `selection_utilities.py`.

## Social-planner / single solution

Equal-weight average on sense-aware normalized objectives (minimum regret across objectives):

- **Label:** `Normalized equal-weight mean optimum (social planner)`
- **Slug:** `norm_eq_mean_opt` (unchanged for HDF5/PNG paths)
- **Not** legacy `Best Average All` (`best_avg_all`)

Env: `CEE_DESIRED_PICKS` (comma-separated; see `DEFAULT_STAGE1_PICKS` in `pick_labels.py`).

See `notes/pareto_selection_methods.md` for citations.
