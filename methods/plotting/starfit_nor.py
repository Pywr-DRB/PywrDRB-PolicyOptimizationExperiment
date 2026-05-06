# methods/plotting/starfit_nor.py
"""STARFIT seasonal NOR (normal operating range) in storage-as-fraction-of-capacity space."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def try_compute_starfit_nor_pct_by_doy(
    policy_params,
    reservoir_name: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    NOR low / high bounds as **percent of capacity** (0–100), one value per calendar day-of-year 1..366.

    Uses the same harmonic NOR definition as ``pywrdrb.release_policies.STARFIT`` (``evaluate`` /
    ``test_nor_constraint``). Returns ``(None, None)`` if pywrdrb is unavailable or setup fails.
    """
    try:
        from pywrdrb.release_policies import STARFIT
        from pywrdrb.release_policies.config import get_policy_context
    except ImportError:
        return None, None

    try:
        pv = np.asarray(policy_params, dtype=float).ravel()
        pol = STARFIT(policy_params=pv, reservoir_name=reservoir_name)
        pol.load_starfit_params(reservoir_name=reservoir_name)
        ctx = get_policy_context(reservoir_name)
        pol.set_context(**ctx)
    except Exception:
        return None, None

    lo = np.zeros(366, dtype=float)
    hi = np.zeros(366, dtype=float)
    for i, doy in enumerate(range(1, 367)):
        s2, c2, _, _ = pol._seasonal_terms(float(doy))
        nh = float(
            np.clip(
                pol.NORhi_mu + pol.NORhi_alpha * s2 + pol.NORhi_beta * c2,
                pol.NORhi_min,
                pol.NORhi_max,
            )
        )
        nl = float(
            np.clip(
                pol.NORlo_mu + pol.NORlo_alpha * s2 + pol.NORlo_beta * c2,
                pol.NORlo_min,
                pol.NORlo_max,
            )
        )
        hi[i] = nh * 100.0
        lo[i] = nl * 100.0
    return lo, hi


def nor_pct_series_on_index(
    dt_index: pd.DatetimeIndex,
    lo366: np.ndarray,
    hi366: np.ndarray,
) -> Tuple[pd.Series, pd.Series]:
    """Map day-of-year NOR bands (366 values, index 0 = DOY 1) onto ``dt_index``."""
    doy = dt_index.dayofyear.to_numpy(dtype=int)
    lo = lo366[doy - 1]
    hi = hi366[doy - 1]
    return (
        pd.Series(lo, index=dt_index),
        pd.Series(hi, index=dt_index),
    )
