"""
Feature construction for discount-curve models.

For the pure curve-fitting task the main 'feature' is maturity.
This module provides helpers for lag features when we move to
dynamic / multi-day settings and for any exogenous covariates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def maturity_features(maturities: np.ndarray) -> np.ndarray:
    """
    Simple polynomial + log features of maturity (used by some linear baselines).
    """
    tau = np.asarray(maturities, dtype=float).reshape(-1, 1)
    return np.hstack([
        tau,
        tau ** 2,
        np.log1p(tau),
        1.0 / (1.0 + tau),
    ])


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add basic calendar features if a date column is present."""
    out = df.copy()
    if date_col not in out.columns:
        return out
    dt = pd.to_datetime(out[date_col])
    out["dow"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["doy"] = dt.dt.dayofyear
    return out
