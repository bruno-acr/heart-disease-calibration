"""Probability, calibration and classification metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, matthews_corrcoef,
    roc_auc_score,
)

from .calibrators import clip_proba

# Short names used in result tables, with the direction that counts as better.
LOWER_IS_BETTER = {"brier": True, "log_loss": True, "auc": False, "ap": False}


def log_loss(y, p) -> float:
    """Mean negative log-likelihood. Probabilities are clipped to [1e-6, 1 - 1e-6], so a
    confidently wrong prediction of exactly 0 or 1 costs at most ln(1e6) ≈ 13.8."""
    y, p = np.asarray(y, dtype=float), clip_proba(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def calibration_intercept(y, p) -> float:
    """Calibration-in-the-large: `a` in logit P(Y=1) = a + logit(p), slope fixed at 1.
    Target 0; positive = predictions too low on average, negative = too high."""
    y, z = np.asarray(y, dtype=float), logit(clip_proba(p))
    return float(brentq(lambda a: np.sum(expit(a + z) - y), -50, 50))


def calibration_slope(y, p) -> float:
    """`b` in logit P(Y=1) = a + b * logit(p). Target 1; below 1 = predictions too extreme,
    above 1 = predictions too moderate. Returns NaN if the fit does not converge."""
    y, z = np.asarray(y, dtype=float), logit(clip_proba(p))
    design = np.column_stack([np.ones_like(z), z])

    def nll(theta):
        eta = design @ theta
        return np.sum(np.logaddexp(0, eta) - y * eta)

    def grad(theta):
        return design.T @ (expit(design @ theta) - y)

    fit = minimize(nll, [0.0, 1.0], jac=grad, method="BFGS")
    converged = fit.success or np.linalg.norm(grad(fit.x)) < 1e-5
    return float(fit.x[1]) if converged else np.nan


def probability_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    return {
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "auc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "mean_predicted": p.mean(),
        "observed_rate": y.mean(),
        "cal_intercept": calibration_intercept(y, p),
        "cal_slope": calibration_slope(y, p),
    }


def brier_skill_score(brier: float, brier_reference: float) -> float:
    return 1 - brier / brier_reference


def calibration_table(y, p, n_bins: int = 5, z: float = 1.96) -> pd.DataFrame:
    """Quantile-binned reliability table with Wilson 95% intervals for the observed rate.
    The intervals describe each bin separately, not the whole curve."""
    t = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    t["bin"] = pd.qcut(t["p"].rank(method="first"), q=n_bins, labels=False)
    out = t.groupby("bin").agg(n=("y", "size"), events=("y", "sum"),
                               predicted=("p", "mean"), observed=("y", "mean")).reset_index()
    n, f = out["n"].to_numpy(), out["observed"].to_numpy()
    centre = (f + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(f * (1 - f) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    out["ci_low"], out["ci_high"] = centre - half, centre + half
    return out


def _safe_div(a, b):
    return a / b if b else np.nan


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    """Metrics after converting probabilities into classes with `p >= threshold`.
    Undefined ratios (zero denominator) are NaN, not 0."""
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens, spec = _safe_div(tp, tp + fn), _safe_div(tn, tn + fp)
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": _safe_div(tp + tn, len(y)),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sens + spec) / 2,
        "mcc": matthews_corrcoef(y, pred),
    }


def paired_bootstrap(y, p_a, p_b, metric, n_boot: int = 2000, seed: int = 42) -> dict:
    """Percentile CI for metric(p_b) - metric(p_a), resampling test records with the same
    indices for both. Conditional on the fitted models: it ignores development uncertainty."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    diffs = []
    while len(diffs) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():  # AUC-type metrics need both classes
            continue
        diffs.append(metric(y[idx], p_b[idx]) - metric(y[idx], p_a[idx]))
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {"difference": metric(y, p_b) - metric(y, p_a), "ci_low": low, "ci_high": high}
