# Robustness & Evaluation Protocol

## Design principles

1. **Reproducibility first**  
   Default runs use a fully synthetic panel with fixed seed so that every clone of the repository produces identical metrics and figures.

2. **Controlled outliers**  
   A known fraction of observations is contaminated. This allows residual diagnostics and future comparison of robust versus non-robust losses under a ground-truth outlier mask.

3. **Transparent baselines**  
   Nelson-Siegel is always reported alongside the ML estimator. No claim of superiority is made without both statistical and economic metrics.

4. **Expanding-window discipline**  
   When evaluating over time we never use future information. Parameters are re-estimated on all data up to (but not including) the test day.

## Planned robust extensions

- Huber or Tukey biweight loss inside the random-feature estimator
- Iteratively reweighted least squares (IRLS)
- Explicit leverage diagnostics and down-weighting

These will be added under the same interface so that the pipeline and reporting remain unchanged.
