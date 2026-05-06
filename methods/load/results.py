import pandas as pd
from pywrdrb.release_policies.config import OBJ_FILTER_BOUNDS


def filter_solutions(df: pd.DataFrame,
                     obj_bounds = OBJ_FILTER_BOUNDS) -> pd.DataFrame:
    
    """
    Filter solutions based on objective bounds.
    Args:
        df (pd.DataFrame): DataFrame containing the solutions.
        obj_bounds (dict): Dictionary with objective bounds.
            Keys are objective names and values are tuples of (min, max).
    Returns:
        pd.DataFrame: Filtered DataFrame.    
    """
    # check that keys of obj_bounds are in the DataFrame
    for obj in obj_bounds.keys():
        if obj not in df.columns:
            raise ValueError(f"Objective {obj} in obj_bounds not found in DataFrame columns.")


    # Filter the DataFrame based on the objective bounds
    for obj, bounds in obj_bounds.items():
        min_bound, max_bound = bounds
        df = df[(df[obj] >= min_bound) & (df[obj] <= max_bound)]
    
    # reset index
    df.reset_index(drop=True, inplace=True)
    
    return df



def _transform_borg_like_dataframe(
    results: pd.DataFrame,
    *,
    obj_labels=None,
):
    """
    Rename ``obj*`` using ``obj_labels``, negate NSE/KGE/inertia for figure convention.
    Returns ``(results, obj_cols_renamed, var_cols)``.
    """
    obj_cols = [col for col in results.columns if col.startswith("obj")]
    var_cols = [col for col in results.columns if col.startswith("var")]

    if obj_labels is not None:
        for col in obj_cols:
            new_col = obj_labels.get(col, col)
            results.rename(columns={col: new_col}, inplace=True)

        obj_cols = [obj_labels.get(col, col) for col in obj_cols]

    for col in obj_cols:
        if "nse" in col.lower():
            results[col] = -results[col]
        elif "kge" in col.lower():
            results[col] = -results[col]
        elif "inertia" in col.lower():
            results[col] = -results[col]

    return results, obj_cols, var_cols


def load_results(file_path: str,
                 obj_labels=None,
                 filter=False,
                 obj_bounds=OBJ_FILTER_BOUNDS) -> pd.DataFrame:
    """
    Load results from a CSV file and return a DataFrame.

    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    results = pd.read_csv(file_path)
    results, obj_cols, var_cols = _transform_borg_like_dataframe(
        results, obj_labels=obj_labels
    )

    if filter and obj_bounds is not None:
        before = len(results)
        results = filter_solutions(results, obj_bounds=obj_bounds)
        after = len(results)
        print(f"[FILTER] {file_path}: {before} → {after} rows")

    results_obj = results.loc[:, obj_cols]
    results_var = results.loc[:, var_cols]

    return results_obj, results_var


def load_results_with_metadata(
    file_path: str,
    obj_labels=None,
    filter=False,
    obj_bounds=OBJ_FILTER_BOUNDS,
):
    """
    Same transformation as :func:`load_results` (rename ``obj*``, NSE sign for figures, optional filter),
    but keeps extra CSV columns (e.g. ``moea_policy`` from ``methods/analysis/mmborg_eps_nondominated_set.py``
    ``--per-reservoir`` output) aligned by row.

    Returns ``(obj_df, var_df, meta_df)`` where ``meta_df`` has only non-``obj*`` / non-``var*``
    columns (empty frame if none).
    """
    results = pd.read_csv(file_path)
    obj_raw = [c for c in results.columns if str(c).startswith("obj")]
    var_raw = [c for c in results.columns if str(c).startswith("var")]
    meta_cols = [c for c in results.columns if c not in obj_raw and c not in var_raw]

    results, obj_cols, var_cols = _transform_borg_like_dataframe(results, obj_labels=obj_labels)

    if filter and obj_bounds is not None:
        before = len(results)
        results = filter_solutions(results, obj_bounds=obj_bounds)
        after = len(results)
        print(f"[FILTER] {file_path}: {before} → {after} rows")

    meta_df = results.loc[:, meta_cols].copy() if meta_cols else pd.DataFrame(index=results.index)
    results_obj = results.loc[:, obj_cols]
    results_var = results.loc[:, var_cols]

    return results_obj, results_var, meta_df