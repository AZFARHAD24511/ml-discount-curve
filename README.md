# Robust Machine Learning for Discount Curve Estimation

Leakage-aware and robustness-focused estimation of the risk-free discount (zero-coupon) curve from Treasury securities, using classical parametric models and modern kernel / random-feature methods.

**Author:** Azizallah Farhadi · `farhadmy1974@gmail.com`  
**Status:** Research prototype (Nov 2026)  
**Target audience:** Quantitative finance research groups emphasising *applied machine learning* (e.g. Swissquote Chair in Quantitative Finance, EPFL)

---

## 1. Motivation

Accurate recovery of the discount function \(d(\tau)\) (or the corresponding yield curve) is a core problem in:

- Fixed-income pricing and relative-value analysis
- Risk management (duration, convexity, key-rate exposures)
- Monetary-policy and macro-finance research
- Downstream tasks such as stochastic control and optimal hedging

Classical parametric families (Nelson-Siegel, Svensson) are parsimonious and stable but can be rigid. Purely nonparametric smoothers risk overfitting, especially in the presence of quote errors and outliers. Recent work (including research associated with the Swissquote Chair) shows that carefully designed machine-learning estimators operating in economically motivated function spaces, combined with robust losses, can improve both statistical fit and economic performance.

This repository implements a transparent, reproducible comparison of:

- Nelson-Siegel (parametric baseline)
- Kernel ridge regression via random Fourier features (flexible ML estimator)

with explicit attention to **robustness** (outliers) and clean evaluation protocols.

---

## 2. Data

| Source                         | Content                                                                 | Coverage (this repo)      |
| ------------------------------ | ----------------------------------------------------------------------- | ------------------------- |
| Synthetic Treasury panel       | Multi-day cross-sections around a slowly varying Nelson-Siegel curve, with realistic noise and controlled outliers | 80 business days × ~48 bonds |
| User-supplied CSV (optional)   | Any panel with columns `date, maturity, observed_discount` (or `observed_yield`) | User-defined              |
Download the raw file from Kaggle / U.S. Treasury and place it at `data/raw/us_treasury_yields_daily.csv`.
> **Note on real market data**  
> The default pipeline uses a high-quality synthetic panel so that the repository is fully self-contained and reproducible without proprietary data licences. The same code accepts real cleaned Treasury quote files (CRSP, TreasuryDirect, or vendor data) via `--source csv --data path/to/file.csv`.

Raw files belong under `data/raw/` and are **excluded from Git**. Only processed summaries and small samples should be committed.

---

## 3. Modelling Protocol

Because the core task is static curve stripping (one cross-section at a time) or simple expanding-window evaluation, the main design choices are:

```
| Component                    | Choice                                              |
| ---------------------------- | --------------------------------------------------- |
| Target                       | Discount factor \(d(\tau)\) (primary) / yield (secondary) |
| Classical baseline           | Nelson-Siegel                                       |
| ML estimator                 | Random Fourier Features + Ridge (kernel approximation) |
| Regularisation               | Explicit ridge penalty; median-heuristic length-scale |
| Outliers                     | Controlled injection in synthetic data; residual diagnostics |
| Evaluation                   | In-sample pricing RMSE/MAE + expanding-window backtest |
```

All **primary results** reported in this README are produced by `scripts/run_pipeline.py` with the default synthetic configuration.

---

## 4. Method

1. **Data preparation**
   - Generate (or load) a panel of (date, maturity, observed discount/yield)
   - Extract single cross-sections or pool history for expanding-window tests

2. **Models**
   - `NelsonSiegel` — standard parametric form fitted by non-linear least squares
   - `KernelDiscountCurve` — random Fourier feature map + Ridge regression on the discount function

3. **Evaluation**
   - Cross-section pricing error (RMSE / MAE on discount factors)
   - Curve smoothness (discrete integrated squared second derivative)
   - Expanding-window backtest: fit on all previous days, evaluate on the next day

---

## 5. Results (default synthetic run)

### Cross-section (2024-06-21, real US Treasury yields)

```
| Model            | RMSE (discount) | MAE (discount) | Smoothness |
| ---------------- | --------------- | -------------- | ---------- |
| Nelson-Siegel    | 0.00310         | 0.00191        | 0.00251    |
| Kernel RF        | **0.00007**     | **0.00005**    | 0.00205    |
```

On a clean 11-point constant-maturity cross-section the flexible random-feature estimator essentially interpolates the observed discounts; Nelson-Siegel retains the classic parametric residual.

### Day-by-day backtest (2018–2024, step = 10 days)

```
| Model            | Mean RMSE       | Mean MAE       |
| ---------------- | --------------- | -------------- |
| Nelson-Siegel    | 0.00536         | 0.00325        |
| Kernel RF        | **0.00008**     | **0.00005**    |
```

Full tables: [`reports/metrics/backtest_summary.csv`](reports/metrics/backtest_summary.csv) · [`reports/metrics/cross_section_metrics.csv`](reports/metrics/cross_section_metrics.csv)

### Figures

After running the pipeline the following plots appear under `reports/figures/`:

**Discount curve (latest cross-section)**

![Discount curve](reports/figures/discount_curve.png)

**Yield curve**

![Yield curve](reports/figures/yield_curve.png)

**Backtest MAE over time**

![Backtest MAE](reports/figures/backtest_mae.png)

*(Run `python scripts/run_pipeline.py` to regenerate all figures and metrics.)*

---

## 6. Repository Structure

```
ml-discount-curve/
├── data/
│   ├── raw/                  # large CSVs – not tracked by Git
│   └── processed/
├── models/                   # fitted model artefacts (optional)
├── reports/
│   ├── figures/              # PNG charts for README & paper
│   ├── metrics/              # CSV + JSON summaries
│   └── methodology/          # notes on robustness & evaluation protocol
├── scripts/
│   └── run_pipeline.py       # end-to-end run
├── src/
│   └── discount_curve/
│       ├── data.py           # load / generate / clean
│       ├── features.py       # maturity & calendar helpers
│       ├── train.py          # Nelson-Siegel + Kernel RF
│       ├── evaluate.py       # metrics & expanding-window backtest
│       └── visualize.py      # report figures
├── tests/
├── notebooks/                # exploratory work (optional)
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 7. Quick Start

```bash
# clone & install
git clone <your-repo-url>
cd ml-discount-curve
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# run the default (synthetic) pipeline
python scripts/run_pipeline.py --out reports

# or supply your own cleaned Treasury panel
python scripts/run_pipeline.py \
  --source csv \
  --data data/raw/treasury_quotes.csv \
  --out reports
```

Tests (once populated):

```bash
pytest tests/ -q
```

---

## 8. Analysis Highlights

- The random-feature kernel estimator achieves modestly lower in-sample pricing error than Nelson-Siegel on data containing outliers, at the cost of a less smooth curve.
- Expanding-window results confirm that the flexible ML model remains competitive when parameters are re-estimated over time.
- Because the synthetic design injects known outliers, residual analysis can be used to illustrate the value of robust losses (planned extension).
- The same pipeline structure is ready for real CRSP-style or TreasuryDirect quote files; only the data loader needs to be pointed at the new CSV.

---

## 9. Roadmap

- [x] Nelson-Siegel baseline
- [x] Random Fourier Feature kernel estimator
- [x] Expanding-window backtest & automated reporting
- [x] Self-contained synthetic panel for full reproducibility
- [ ] Robust loss (Huber / IRLS) and explicit outlier down-weighting
- [ ] Neural-network discount function with monotonicity constraints
- [ ] Multi-curve / transfer-learning experiments
- [ ] Real U.S. Treasury panel (public or licensed) and economic evaluation (hedging error, duration stability)

---

## 10. Citation & Licence

If you use this code or results, please cite the repository and note the evaluation protocol (cross-section + expanding window).

This work is intended as a research prototype aligned with current academic agendas in quantitative finance and applied machine learning (including research themes of the Swissquote Chair at EPFL).

Licence: MIT (code). Any real market data remain subject to their original licences.
