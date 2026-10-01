import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss as sk_log_loss

from heartcal.data import load_cleveland
from heartcal.experiment import select_method
from heartcal.metrics import (
    calibration_intercept, calibration_slope, calibration_table, classification_metrics,
    log_loss, paired_bootstrap,
)


def test_data_loads_with_expected_outcome():
    X, y = load_cleveland()
    assert X.shape == (303, 13)
    assert "num" not in X.columns
    assert y.sum() == 139


def test_well_calibrated_scores_have_null_intercept_and_unit_slope():
    rng = np.random.default_rng(1)
    p = expit(rng.normal(0, 1.5, 20_000))
    y = rng.binomial(1, p)
    assert calibration_intercept(y, p) == pytest.approx(0, abs=0.05)
    assert calibration_slope(y, p) == pytest.approx(1, abs=0.05)


def test_overconfident_scores_have_slope_below_one():
    rng = np.random.default_rng(2)
    p_true = expit(rng.normal(0, 1, 20_000))
    y = rng.binomial(1, p_true)
    overconfident = expit(2 * logit(p_true))
    assert calibration_slope(y, overconfident) == pytest.approx(0.5, abs=0.05)


def test_log_loss_matches_scikit_learn_away_from_the_clip():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 200)
    y = rng.binomial(1, p)
    assert log_loss(y, p) == pytest.approx(sk_log_loss(y, p))


def test_classification_metrics_on_known_matrix():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.4, 0.8, 0.7])  # TN=3 FP=1 FN=1 TP=3
    m = classification_metrics(y, p)
    assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (3, 1, 1, 3)
    assert m["sensitivity"] == m["specificity"] == m["accuracy"] == 0.75


def test_calibration_table_bins_cover_all_records():
    rng = np.random.default_rng(4)
    p = rng.uniform(size=101)
    table = calibration_table(rng.binomial(1, p), p, n_bins=5)
    assert table["n"].sum() == 101
    assert (table["ci_low"] <= table["observed"]).all() and (table["observed"] <= table["ci_high"]).all()


def test_paired_bootstrap_of_identical_predictions_is_zero():
    rng = np.random.default_rng(5)
    p = rng.uniform(size=50)
    out = paired_bootstrap(rng.binomial(1, p), p, p, brier_score_loss, n_boot=200)
    assert out["difference"] == out["ci_low"] == out["ci_high"] == 0


def _fake_cv(means: dict, sd=0.01, k=20, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sd, k)  # shared across methods, as in paired CV splits
    return pd.DataFrame([{"method": m, "brier": mu + noise[i] + rng.normal(0, 0.001)}
                         for m, mu in means.items() for i in range(k)])


def test_one_se_rule_prefers_simpler_method_when_gain_is_negligible():
    cv = _fake_cv({"Uncalibrated": 0.150, "Logistic recalibration": 0.120,
                   "Isotonic regression": 0.119})
    selected, stats = select_method(cv)
    assert stats["mean"].idxmin() == "Isotonic regression"
    assert selected == "Logistic recalibration"


def test_one_se_rule_keeps_best_method_when_gain_is_clear():
    cv = _fake_cv({"Uncalibrated": 0.150, "Temperature scaling": 0.140,
                   "Isotonic regression": 0.100})
    assert select_method(cv)[0] == "Isotonic regression"
