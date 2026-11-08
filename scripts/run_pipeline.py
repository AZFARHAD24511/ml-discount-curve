#!/usr/bin/env python
"""
End-to-end pipeline for robust discount-curve estimation.

Usage
-----
# Real US Treasury yields (default if file exists)
python scripts/run_pipeline.py

# Explicit real data
python scripts/run_pipeline.py --source treasury_wide --data data/raw/us_treasury_yields_daily.csv

# Synthetic (for quick tests / full reproducibility)
python scripts/run_pipeline.py --source synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discount_curve.data import load_quotes, prepare_cross_section, save_processed
from discount_curve.train import fit_models
from discount_curve.evaluate import evaluate_cross_section, expanding_window_backtest
from discount_curve.visualize import plot_discount_curve, plot_yield_curve, plot_backtest_mae


def main():
    parser = argparse.ArgumentParser(description="Discount curve ML pipeline")
    parser.add_argument(
        "--source",
        choices=["synthetic", "treasury_wide", "csv", "long"],
        default=None,
        help="Data source (default: auto-detect real file if present)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "raw" / "us_treasury_yields_daily.csv"),
        help="Path to CSV",
    )
    parser.add_argument("--out", type=str, default="reports", help="Output directory")
    parser.add_argument("--n-days", type=int, default=80, help="Synthetic panel length")
    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="Start date for real-data panel (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    fig_dir = out / "figures"
    metrics_dir = out / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Robust ML Discount Curve Estimation — Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data ...")

    data_path = Path(args.data)
    if args.source is None:
        if data_path.exists():
            args.source = "treasury_wide"
            print(f"  Auto-detected real Treasury file: {data_path}")
        else:
            args.source = "synthetic"
            print("  No real file found — using synthetic data")

    if args.source == "synthetic":
        panel = load_quotes(source="synthetic", n_days=args.n_days, n_bonds=48, seed=42)
    else:
        panel = load_quotes(path=args.data, source=args.source)
        # Restrict to a recent window for speed and relevance
        start = pd.Timestamp(args.start)
        panel = panel[panel["date"] >= start].copy()
        print(f"  Filtered to dates >= {start.date()}")

    save_processed(panel, ROOT / "data" / "processed" / "panel.csv")
    print(f"  Panel shape: {panel.shape}  |  unique dates: {panel['date'].nunique()}")

    cs = prepare_cross_section(panel)
    print(f"  Cross-section date: {cs['date'].iloc[0].date()}  |  points: {len(cs)}")

    maturities = cs["maturity"].values
    obs_d = cs["observed_discount"].values
    obs_y = cs["observed_yield"].values
    is_out = cs["is_outlier"].values if "is_outlier" in cs.columns else None
    true_d = cs["true_discount"].values if "true_discount" in cs.columns else None

    # ------------------------------------------------------------------
    # 2. Fit models
    # ------------------------------------------------------------------
    print("\n[2/5] Fitting models ...")
    models = fit_models(maturities, obs_y, obs_d)
    for name, m in models.items():
        if hasattr(m, "get_params"):
            print(f"  {name}: {m.get_params()}")

    # ------------------------------------------------------------------
    # 3. Cross-section metrics
    # ------------------------------------------------------------------
    print("\n[3/5] Cross-section evaluation ...")
    metrics = evaluate_cross_section(maturities, obs_d, models)
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(metrics_dir / "cross_section_metrics.csv")
    print(metrics_df.round(6).to_string())

    # ------------------------------------------------------------------
    # 4. Expanding-window backtest (on a subsample of dates for speed)
    # ------------------------------------------------------------------
    print("\n[4/5] Expanding-window backtest ...")
    # Use monthly steps on real data to keep runtime reasonable
    dates = sorted(panel["date"].unique())
    if len(dates) > 150:
        # Keep roughly every 5th business day for the backtest
        step = max(5, len(dates) // 80)
    else:
        step = 5

    bt = expanding_window_backtest(panel, min_train_days=40, step=step)
    bt.to_csv(metrics_dir / "backtest_results.csv", index=False)

    summary = bt.groupby("model")[["rmse", "mae"]].mean()
    summary.to_csv(metrics_dir / "backtest_summary.csv")
    print("\nAverage backtest metrics:")
    print(summary.round(6).to_string())

    with open(metrics_dir / "summary.json", "w") as f:
        json.dump(
            {
                "cross_section": metrics,
                "backtest_mean": summary.to_dict(),
                "n_dates": int(panel["date"].nunique()),
                "cross_section_date": str(cs["date"].iloc[0].date()),
            },
            f,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------
    # 5. Figures
    # ------------------------------------------------------------------
    print("\n[5/5] Generating figures ...")
    plot_discount_curve(
        maturities,
        obs_d,
        models,
        is_outlier=is_out,
        true_discount=true_d,
        title=f"Discount Curve — {cs['date'].iloc[0].date()}",
        save_path=fig_dir / "discount_curve.png",
    )
    plot_yield_curve(
        maturities,
        obs_y,
        models,
        is_outlier=is_out,
        save_path=fig_dir / "yield_curve.png",
    )
    if len(bt) > 0:
        plot_backtest_mae(bt, save_path=fig_dir / "backtest_mae.png")

    print(f"\nFigures written to {fig_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
