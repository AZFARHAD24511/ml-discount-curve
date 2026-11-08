"""
Model fitting: classical parametric + kernel / random-feature ML estimators.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Classical
# ---------------------------------------------------------------------------

class NelsonSiegel:
    """Nelson-Siegel yield curve."""

    def __init__(self, lambda_fixed: Optional[float] = None):
        self.lambda_fixed = lambda_fixed
        self.params_: Optional[np.ndarray] = None

    def _y(self, tau: np.ndarray, p: np.ndarray) -> np.ndarray:
        b0, b1, b2, lam = p
        x = lam * tau
        f = np.where(x > 1e-8, (1 - np.exp(-x)) / x, 1.0)
        return b0 + b1 * f + b2 * (f - np.exp(-x))

    def fit(self, maturities: np.ndarray, yields: np.ndarray) -> "NelsonSiegel":
        maturities = np.asarray(maturities, float)
        yields = np.asarray(yields, float)

        def obj(p):
            if self.lambda_fixed is not None:
                p = np.r_[p, self.lambda_fixed]
            return np.mean((self._y(maturities, p) - yields) ** 2)

        x0 = np.array([0.03, -0.02, 0.01, 0.5])
        bounds = [(-0.1, 0.25), (-0.4, 0.4), (-0.4, 0.4), (0.05, 3.0)]
        if self.lambda_fixed is not None:
            x0, bounds = x0[:3], bounds[:3]

        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        self.params_ = np.r_[res.x, self.lambda_fixed] if self.lambda_fixed is not None else res.x
        return self

    def predict_yield(self, maturities: np.ndarray) -> np.ndarray:
        return self._y(np.asarray(maturities, float), self.params_)

    def predict_discount(self, maturities: np.ndarray) -> np.ndarray:
        y = self.predict_yield(maturities)
        return np.exp(-y * maturities)

    def get_params(self) -> Dict[str, float]:
        if self.params_ is None:
            return {}
        names = ["beta0", "beta1", "beta2", "lambda"]
        return dict(zip(names, self.params_))


# ---------------------------------------------------------------------------
# Kernel / Random Features
# ---------------------------------------------------------------------------

class KernelDiscountCurve:
    """
    Random Fourier Feature approximation to a kernel ridge estimator
    of the discount function d(τ).
    """

    def __init__(
        self,
        n_components: int = 100,
        alpha: float = 1e-3,
        gamma: Optional[float] = None,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state
        self.model_ = None
        self.W_ = None
        self.b_ = None

    def _rff(self, X: np.ndarray) -> np.ndarray:
        D = self.n_components
        return np.sqrt(2.0 / D) * np.cos(X @ self.W_ + self.b_)

    def fit(self, maturities: np.ndarray, discounts: np.ndarray) -> "KernelDiscountCurve":
        X = np.asarray(maturities, float).reshape(-1, 1)
        y = np.asarray(discounts, float)

        if self.gamma is None:
            dists = np.abs(X - X.T)
            med = np.median(dists[dists > 0])
            self.gamma = 1.0 / med if med > 0 else 1.0

        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(0, np.sqrt(2 * self.gamma), size=(1, self.n_components))
        self.b_ = rng.uniform(0, 2 * np.pi, size=self.n_components)

        Phi = self._rff(X)
        self.model_ = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=self.alpha))
        self.model_.fit(Phi, y)
        return self

    def predict_discount(self, maturities: np.ndarray) -> np.ndarray:
        X = np.asarray(maturities, float).reshape(-1, 1)
        return self.model_.predict(self._rff(X))

    def predict_yield(self, maturities: np.ndarray) -> np.ndarray:
        d = np.clip(self.predict_discount(maturities), 1e-8, None)
        return -np.log(d) / np.asarray(maturities)


def fit_models(
    maturities: np.ndarray,
    observed_yield: np.ndarray,
    observed_discount: np.ndarray,
) -> Dict[str, Any]:
    """Fit the standard suite of models and return them in a dict."""
    ns = NelsonSiegel().fit(maturities, observed_yield)
    kr = KernelDiscountCurve(n_components=120, alpha=8e-4).fit(maturities, observed_discount)
    return {"nelson_siegel": ns, "kernel_rf": kr}
