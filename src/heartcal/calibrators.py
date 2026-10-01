"""Post-hoc calibration maps for binary probabilities.

Every calibrator learns a function ``p -> p'`` from (predicted probability, outcome) pairs
and exposes the same interface: ``fit(p, y)`` and ``predict(p)``. They are ordered by
flexibility, which the selection rule uses to prefer simpler maps:

========================  ======  ===================================================
Method                    Params  Map
========================  ======  ===================================================
Uncalibrated              0       p' = p
Temperature scaling       1       logit p' = logit(p) / T
Platt (sigmoid)           2       logit p' = a * p + b   (scikit-learn ``"sigmoid"``)
Logistic recalibration    2       logit p' = a + b * logit(p)
Beta calibration          3       logit p' = c + a * ln(p) - b * ln(1 - p),  a, b >= 0
Isotonic regression       many    monotone step function (scikit-learn ``"isotonic"``)
========================  ======  ===================================================
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def clip_proba(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def _logistic_fit(features: np.ndarray, y: np.ndarray, sample_weight=None) -> LogisticRegression:
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000).fit(
        features, y, sample_weight=sample_weight)


class Calibrator:
    name: str = ""
    complexity: float = 0

    def fit(self, p, y):
        raise NotImplementedError

    def predict(self, p) -> np.ndarray:
        raise NotImplementedError


class Uncalibrated(Calibrator):
    name = "Uncalibrated"
    complexity = 0

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, dtype=float)


class TemperatureScaling(Calibrator):
    """Single-parameter rescaling of the logit. It can sharpen or soften predictions
    but cannot shift them, so it does not fix calibration-in-the-large."""

    name = "Temperature scaling"
    complexity = 1

    def fit(self, p, y):
        z, y = logit(clip_proba(p)), np.asarray(y, dtype=float)

        def nll(log_t):
            eta = z / np.exp(log_t)
            return np.sum(np.logaddexp(0, eta) - y * eta)

        self.temperature_ = float(np.exp(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x))
        return self

    def predict(self, p):
        return expit(logit(clip_proba(p)) / self.temperature_)


class PlattScaling(Calibrator):
    """Platt (1999) sigmoid on the raw probability, with Platt's smoothed targets, as in
    scikit-learn's ``CalibratedClassifierCV(method="sigmoid")`` for estimators without
    ``decision_function``."""

    name = "Platt (sigmoid)"
    complexity = 2

    def fit(self, p, y):
        p, y = np.asarray(p, dtype=float), np.asarray(y)
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        target = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        # Soft targets expressed as two weighted copies of each observation.
        features = np.concatenate([p, p])[:, None]
        labels = np.concatenate([np.ones_like(y), np.zeros_like(y)])
        weights = np.concatenate([target, 1 - target])
        self.model_ = _logistic_fit(features, labels, weights)
        return self

    def predict(self, p):
        return self.model_.predict_proba(np.asarray(p, dtype=float)[:, None])[:, 1]


class LogisticRecalibration(Calibrator):
    """Logistic regression of the outcome on logit(p) (Cox, 1958): the same model used to
    estimate the calibration intercept and slope, used here as a correction."""

    name = "Logistic recalibration"
    complexity = 2

    def fit(self, p, y):
        self.model_ = _logistic_fit(logit(clip_proba(p))[:, None], np.asarray(y))
        return self

    def predict(self, p):
        return self.model_.predict_proba(logit(clip_proba(p))[:, None])[:, 1]


class BetaCalibration(Calibrator):
    """Beta calibration (Kull, Silva Filho & Flach, 2017). Contains logistic recalibration as
    the special case a = b, but can also correct asymmetric distortions in the two tails.
    A negative coefficient would make the map non-monotone, so that feature is dropped and
    the model refitted, as in the reference implementation."""

    name = "Beta calibration"
    complexity = 3

    @staticmethod
    def _features(p):
        p = clip_proba(p)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, p, y):
        features, y = self._features(p), np.asarray(y)
        self.columns_ = [0, 1]
        self.model_ = _logistic_fit(features, y)
        coef = self.model_.coef_.ravel()
        if (coef < 0).any():
            self.columns_ = [int(np.argmax(coef))]
            self.model_ = _logistic_fit(features[:, self.columns_], y)
        return self

    @property
    def coefficients_(self) -> dict:
        coef = dict(zip(("a", "b"), (0.0, 0.0)))
        for col, value in zip(self.columns_, self.model_.coef_.ravel()):
            coef["ab"[col]] = float(value)
        coef["c"] = float(self.model_.intercept_[0])
        return coef

    def predict(self, p):
        return self.model_.predict_proba(self._features(p)[:, self.columns_])[:, 1]


class IsotonicCalibration(Calibrator):
    """Non-parametric monotone step function, as in scikit-learn's ``method="isotonic"``.
    The most flexible map: it can fix any monotone distortion but needs more data and
    can output exact 0 or 1."""

    name = "Isotonic regression"
    complexity = np.inf

    def fit(self, p, y):
        self.model_ = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(p, dtype=float), y)
        return self

    def predict(self, p):
        return self.model_.predict(np.asarray(p, dtype=float))


CALIBRATORS: dict[str, type[Calibrator]] = {
    cls.name: cls for cls in (
        Uncalibrated, TemperatureScaling, PlattScaling,
        LogisticRecalibration, BetaCalibration, IsotonicCalibration,
    )
}
