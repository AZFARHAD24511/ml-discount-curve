"""
Data loading and preparation for Treasury discount / yield curve estimation.

Supports:
- Synthetic panel (for full reproducibility without external files)
- Real US Treasury constant-maturity yields in wide format
  (columns: date, US1M, US3M, ..., US30Y)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd


# Mapping from column names to maturity in years
TENOR_MAP: Dict[str, float] = {
    "US1M": 1 / 12,
    "US3M": 0.25,
    "US6M": 0.5,
    "US1Y": 1.0,
    "US2Y": 2.0,
    "US3Y": 3.0,
    "US5Y": 5.0,
    "US7Y": 7.0,
    "US10Y": 10.0,
    "US20Y": 20.0,
    "US30Y": 30.0,
}


def generate_synthetic_treasury_panel(
    n_days: int = 120,
    n_bonds: int = 45,
    max_maturity: float = 30.0,
    noise_level: float = 0.0012,
    outlier_fraction: float = 0.06,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a multi-day panel of synthetic Treasury-like observations
    around a slowly varying Nelson-Siegel curve, with realistic noise and outliers.
    """
    rng = np.random.default_rng(seed)

    base_beta0 = 0.035
    base_beta1 = -0.018
    base_beta2 = 0.012
    lam = 0.55

    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    records = []

    for i, dt in enumerate(dates):
        beta0 = base_beta0 + 0.004 * np.sin(2 * np.pi * i / 60)
        beta1 = base_beta1 + 0.006 * np.cos(2 * np.pi * i / 90)
        beta2 = base_beta2 + 0.003 * np.sin(2 * np.pi * i / 40)

        maturities = np.sort(rng.uniform(0.25, max_maturity, size=n_bonds))

        def ns_yield(tau):
            x = lam * tau
            factor = np.where(x > 1e-8, (1 - np.exp(-x)) / x, 1.0)
            return beta0 + beta1 * factor + beta2 * (factor - np.exp(-x))

        true_y = ns_yield(maturities)
        true_d = np.exp(-true_y * maturities)

        noise = rng.normal(0, noise_level, size=n_bonds)
        obs_d = true_d + noise

        n_out = int(outlier_fraction * n_bonds)
        out_idx = rng.choice(n_bonds, size=n_out, replace=False)
        obs_d[out_idx] += rng.normal(0, 0.018, size=n_out)

        obs_y = -np.log(np.clip(obs_d, 1e-6, None)) / maturities

        for j in range(n_bonds):
            records.append(
                {
                    "date": dt,
                    "maturity": maturities[j],
                    "true_discount": true_d[j],
                    "observed_discount": obs_d[j],
                    "true_yield": true_y[j],
                    "observed_yield": obs_y[j],
                    "is_outlier": j in out_idx,
                }
            )

    return pd.DataFrame.from_records(records)


def wide_yields_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format Treasury yields (one column per tenor) to long format
    expected by the rest of the pipeline.

    Input columns: date, US1M, US3M, ..., US30Y  (yields in percent)
    Output columns: date, maturity, observed_yield, observed_discount
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    tenor_cols = [c for c in df.columns if c in TENOR_MAP]
    if not tenor_cols:
        raise ValueError(
            f"No recognised tenor columns found. Expected one of: {list(TENOR_MAP.keys())}"
        )

    long = df.melt(
        id_vars=["date"],
        value_vars=tenor_cols,
        var_name="tenor",
        value_name="yield_pct",
    )
    long = long.dropna(subset=["yield_pct"])
    long["maturity"] = long["tenor"].map(TENOR_MAP)
    # Convert percent → decimal
    long["observed_yield"] = long["yield_pct"] / 100.0
    long["observed_discount"] = np.exp(-long["observed_yield"] * long["maturity"])
    long = long.drop(columns=["tenor", "yield_pct"])
    long = long.sort_values(["date", "maturity"]).reset_index(drop=True)
    return long


def load_quotes(
    path: Optional[Union[str, Path]] = None,
    source: str = "synthetic",
    **kwargs,
) -> pd.DataFrame:
    """
    Load Treasury quote / yield data.

    Parameters
    ----------
    path : path to a CSV
        - For source='csv' or 'treasury_wide': wide-format file with columns
          date, US1M, US3M, ..., US30Y
        - For source='long': already in long format (date, maturity, observed_yield)
    source : 'synthetic' | 'treasury_wide' | 'csv' | 'long'
    """
    if source == "synthetic" or path is None:
        return generate_synthetic_treasury_panel(**kwargs)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    raw = pd.read_csv(path)

    if source in ("treasury_wide", "csv"):
        if any(c in raw.columns for c in TENOR_MAP):
            return wide_yields_to_long(raw)
        source = "long"

    if source == "long":
        if "date" in raw.columns:
            raw["date"] = pd.to_datetime(raw["date"])
        if "observed_yield" not in raw.columns and "yield" in raw.columns:
            raw = raw.rename(columns={"yield": "observed_yield"})
        if "observed_discount" not in raw.columns and "observed_yield" in raw.columns:
            raw["observed_discount"] = np.exp(
                -raw["observed_yield"] * raw["maturity"]
            )
        return raw

    raise ValueError(f"Unknown source: {source}")


def prepare_cross_section(
    df: pd.DataFrame,
    date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Extract a single cross-section (one date) and apply light cleaning.
    If date is None, use the latest available date that has enough points.
    """
    df = df.copy()
    if "date" not in df.columns:
        return df.sort_values("maturity").reset_index(drop=True)

    if date is None:
        counts = df.groupby("date").size()
        good = counts[counts >= 6]
        date = good.index.max() if len(good) else df["date"].max()

    cs = df[df["date"] == date].copy()
    cs = cs.dropna(subset=["maturity", "observed_yield"])
    cs = cs[(cs["maturity"] >= 0.08) & (cs["maturity"] <= 30.5)]
    cs = cs[(cs["observed_discount"] > 0.01) & (cs["observed_discount"] < 1.5)]
    return cs.sort_values("maturity").reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
