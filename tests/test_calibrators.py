import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from heartcal.calibrators import (
    CALIBRATORS, BetaCalibration, LogisticRecalibration, TemperatureScaling, Uncalibrated,
)
from heartcal.data import load_cleveland
from heartcal.experiment import crossfit
from heartcal.modeling import make_model

GRID = np.linspace(0.001, 0.999, 400)


def simulate(n=4000, a=0.0, b=1.0, seed=0):
    """Scores p whose true event probability is expit(a + b * logit(p))."""
    rng = np.random.default_rng(seed)
    p = expit(rng.normal(0, 1.5, n))
    y = rng.binomial(1, expit(a + b * logit(p)))
    return p, y


@pytest.mark.parametrize("name", list(CALIBRATORS))
def test_outputs_are_monotone_probabilities(name):
    p, y = simulate(500)
    out = CALIBRATORS[name]().fit(p, y).predict(GRID)
    assert np.all((out >= 0) & (out <= 1))
    assert np.all(np.diff(out) >= -1e-12)


def test_uncalibrated_is_identity():
    assert np.allclose(Uncalibrated().fit(GRID, GRID > 0.5).predict(GRID), GRID)


def test_logistic_recalibration_recovers_parameters():
    p, y = simulate(a=0.5, b=0.6)
    model = LogisticRecalibration().fit(p, y).model_
    assert model.intercept_[0] == pytest.approx(0.5, abs=0.15)
    assert model.coef_[0, 0] == pytest.approx(0.6, abs=0.1)


def test_temperature_scaling_recovers_temperature():
    p, y = simulate(b=0.5)  # true map: logit(p) / 2
    assert TemperatureScaling().fit(p, y).temperature_ == pytest.approx(2.0, rel=0.15)


def test_beta_calibration_contains_logistic_recalibration():
    p, y = simulate(a=-0.3, b=0.7)
    beta = BetaCalibration().fit(p, y).predict(GRID)
    logistic = LogisticRecalibration().fit(p, y).predict(GRID)
    assert np.max(np.abs(beta - logistic)) < 0.03


@pytest.fixture(scope="module")
def reference_fit():
    X, y = load_cleveland()
    X_tr, y_tr, X_ev = X.iloc[:240], y.iloc[:240], X.iloc[240:]
    fit = crossfit(X_tr, y_tr, X_ev, seed=0, n_estimators=50)
    return X_tr, y_tr, X_ev, fit


@pytest.mark.parametrize("method, ours, atol", [
    ("isotonic", "Isotonic regression", 1e-10),
    ("sigmoid", "Platt (sigmoid)", 1e-3),
])
def test_matches_scikit_learn(reference_fit, method, ours, atol):
    """Cross-fitting + our calibrators reproduce CalibratedClassifierCV(ensemble=False)."""
    X_tr, y_tr, X_ev, fit = reference_fit
    sk = CalibratedClassifierCV(make_model(0, 50), method=method, ensemble=False,
                                cv=StratifiedKFold(5, shuffle=True, random_state=0))
    expected = sk.fit(X_tr, y_tr).predict_proba(X_ev)[:, 1]
    np.testing.assert_allclose(fit.proba[ours], expected, atol=atol)
