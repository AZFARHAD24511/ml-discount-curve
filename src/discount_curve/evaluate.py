"""
Evaluation metrics and expanding-window style backtest helpers.
"""

from __future__ import annotations

from typing import Dict, List, Any

import numpy as np
import pandas as pd

from .train import NelsonSiegel, KernelDiscountCurve, fit_models


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def smoothness(maturities: np.ndarray, discounts: np.ndarray) -> float:
    """Discrete approximation of integrated squared second derivative."""
    order = np.argsort(maturities)
    t = np.asarray(maturities)[order]
    d = np.asarray(discounts)[order]
    if len(t) < 3:
        return 0.0
    dt = np.diff(t)
    d1 = np.diff(d) / dt
    d2 = np.diff(d1) / dt[:-1]
    return float(np.sum(d2 ** 2) * np.mean(dt))


def evaluate_cross_section(
    maturities: np.ndarray,
    observed_discount: np.ndarray,
    models: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Compute pricing metrics for a single cross-section."""
    results = {}
    for name, model in models.items():
        pred = model.predict_discount(maturities)
        results[name] = {
            "rmse_discount": rmse(observed_discount, pred),
            "mae_discount": mae(observed_discount, pred),
            "smoothness": smoothness(maturities, pred),
        }
    return results


def expanding_window_backtest(
    panel: pd.DataFrame,
    min_train_days: int = 20,
    step: int = 5,
) -> pd.DataFrame:
    """
    Simple expanding-window evaluation on a multi-day panel.
    For each test day we fit on all previous days' cross-sections
    (pooled) and evaluate pricing error on that day.
    """
    dates = sorted(panel["date"].unique())
    records = []

    for i in range(min_train_days, len(dates), step):
        train_dates = dates[:i]
        test_date = dates[i]

        train = panel[panel["date"].isin(train_dates)]
        test = panel[panel["date"] == test_date]

        # Pool training observations
        mat_tr = train["maturity"].values
        yld_tr = train["observed_yield"].values
        disc_tr = train["observed_discount"].values

        models = fit_models(mat_tr, yld_tr, disc_tr)

        mat_te = test["maturity"].values
        disc_te = test["observed_discount"].values

        for name, model in models.items():
            pred = model.predict_discount(mat_te)
            records.append(
                {
                    "test_date": test_date,
                    "model": name,
                    "rmse": rmse(disc_te, pred),
                    "mae": mae(disc_te, pred),
                    "n_train_days": i,
                    "n_test_bonds": len(test),
                }
            )

    return pd.DataFrame.from_records(records)
