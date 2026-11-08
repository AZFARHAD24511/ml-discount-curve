"""
Figure generation for reports and README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_discount_curve(
    maturities: np.ndarray,
    observed: np.ndarray,
    models: Dict[str, Any],
    is_outlier: Optional[np.ndarray] = None,
    true_discount: Optional[np.ndarray] = None,
    title: str = "Discount Curve Estimation",
    save_path: Optional[Path] = None,
) -> None:
    grid = np.linspace(0.25, float(maturities.max()), 300)

    fig, ax = plt.subplots(figsize=(10, 5))

    if is_outlier is not None:
        ax.scatter(
            maturities,
            observed,
            c=is_outlier,
            cmap="coolwarm",
            s=45,
            alpha=0.85,
            label="Observed (red = outlier)",
            zorder=3,
        )
    else:
        ax.scatter(maturities, observed, s=45, alpha=0.75, label="Observed", zorder=3)

    if true_discount is not None:
        # sort for line
        order = np.argsort(maturities)
        ax.plot(
            maturities[order],
            true_discount[order],
            color="black",
            lw=1.5,
            alpha=0.5,
            label="True curve",
            zorder=2,
        )

    styles = {"nelson_siegel": ("-", 2.0), "kernel_rf": ("--", 2.0)}
    for name, model in models.items():
        ls, lw = styles.get(name, ("-", 1.8))
        ax.plot(grid, model.predict_discount(grid), ls=ls, lw=lw, label=name.replace("_", " ").title())

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Discount factor")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_backtest_mae(
    backtest_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for model, grp in backtest_df.groupby("model"):
        ax.plot(grp["test_date"], grp["mae"], marker="o", label=model.replace("_", " ").title())
    ax.set_xlabel("Test date")
    ax.set_ylabel("MAE (discount)")
    ax.set_title("Expanding-window Backtest — MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_yield_curve(
    maturities: np.ndarray,
    observed_yield: np.ndarray,
    models: Dict[str, Any],
    is_outlier: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> None:
    grid = np.linspace(0.25, float(maturities.max()), 300)
    fig, ax = plt.subplots(figsize=(10, 5))

    if is_outlier is not None:
        ax.scatter(maturities, observed_yield * 100, c=is_outlier, cmap="coolwarm", s=45, alpha=0.85, zorder=3)
    else:
        ax.scatter(maturities, observed_yield * 100, s=45, alpha=0.75, zorder=3)

    for name, model in models.items():
        ax.plot(grid, model.predict_yield(grid) * 100, lw=2, label=name.replace("_", " ").title())

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_title("Yield Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
