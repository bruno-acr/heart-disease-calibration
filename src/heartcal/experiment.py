"""Cross-fitted calibration, repeated cross-validation and nested cross-validation.

Design
------
* **Cross-fitting** (scikit-learn's ``ensemble=False``): inside a training set, internal
  stratified K-fold produces out-of-fold (OOF) probabilities for every record; each
  calibrator is fitted on those OOF probabilities; the base model is then refitted on the
  whole training set. Calibrators never see probabilities the model produced for its own
  training data, and no record is set aside only for calibration.
* **Repeated CV** on the development set compares the calibrators on identical splits.
* **Nested CV** wraps the whole procedure (cross-fitting + method selection) in an outer
  loop, estimating how the *procedure* performs on unseen data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

from .calibrators import CALIBRATORS, Calibrator
from .metrics import probability_metrics
from .modeling import make_model


@dataclass
class CrossFitResult:
    """Everything produced by fitting the base model and all calibrators on one training set."""

    oof: np.ndarray                      # OOF probabilities on the training set
    y_train: np.ndarray
    calibrators: dict[str, Calibrator]
    proba: dict[str, np.ndarray] = field(default_factory=dict)   # per method, on the eval set
    model: object = None


def crossfit(X_train, y_train, X_eval, seed: int = 42, cv_seed: int | None = None,
             n_splits: int = 5, n_estimators: int = 300,
             methods: list[str] | None = None) -> CrossFitResult:
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed if cv_seed is None else cv_seed)
    oof = cross_val_predict(make_model(seed, n_estimators), X_train, y_train,
                            cv=cv, method="predict_proba")[:, 1]
    model = make_model(seed, n_estimators).fit(X_train, y_train)
    p_eval = model.predict_proba(X_eval)[:, 1]
    y_train = np.asarray(y_train)
    calibrators = {name: CALIBRATORS[name]().fit(oof, y_train) for name in (methods or CALIBRATORS)}
    proba = {name: cal.predict(p_eval) for name, cal in calibrators.items()}
    return CrossFitResult(oof, y_train, calibrators, proba, model)


def _metrics_frame(y_eval, proba: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame([{"method": m, **probability_metrics(y_eval, p)} for m, p in proba.items()])


def _evaluate_split(X, y, train_idx, eval_idx, seed, cv_seed, n_estimators):
    fit = crossfit(X.iloc[train_idx], y.iloc[train_idx], X.iloc[eval_idx],
                   seed=seed, cv_seed=cv_seed, n_estimators=n_estimators)
    return _metrics_frame(y.iloc[eval_idx], fit.proba)


def repeated_cv(X, y, n_splits: int = 5, n_repeats: int = 5, seed: int = 42,
                n_estimators: int = 300, n_jobs: int = -1) -> pd.DataFrame:
    """Long table: one row per (split, method) with probability and calibration metrics."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(splitter.split(X, y))
    frames = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_split)(X, y, tr, ev, seed, seed + i, n_estimators)
        for i, (tr, ev) in enumerate(splits))
    for i, frame in enumerate(frames):
        frame.insert(0, "repeat", i // n_splits)
        frame.insert(1, "fold", i % n_splits)
    return pd.concat(frames, ignore_index=True)


def summarize_cv(cv_results: pd.DataFrame, metrics=("brier", "log_loss", "auc",
                                                     "cal_intercept", "cal_slope")) -> pd.DataFrame:
    order = list(dict.fromkeys(cv_results["method"]))
    agg = cv_results.groupby("method")[list(metrics)].agg(["mean", "std"]).loc[order]
    return agg


def paired_differences(cv_results: pd.DataFrame, metric: str,
                       reference: str = "Uncalibrated") -> pd.DataFrame:
    """Per-split metric(method) - metric(reference); splits are identical across methods."""
    wide = cv_results.pivot_table(index=["repeat", "fold"], columns="method", values=metric)
    return wide.sub(wide[reference], axis=0).drop(columns=reference)


def select_method(cv_results: pd.DataFrame, metric: str = "brier") -> tuple[str, pd.DataFrame]:
    """Pre-specified one-standard-error rule.

    Take the method with the lowest mean CV metric; among all methods whose mean is within
    one standard error of it, choose the least flexible one (fewest parameters, ties broken
    by the lower mean). Folds of repeated CV are correlated, so the SE is approximate; the
    rule is a guard against choosing extra flexibility for a negligible gain, not a test.
    """
    k = cv_results.groupby("method").size()
    stats = cv_results.groupby("method")[metric].agg(["mean", "std"])
    stats["se"] = stats["std"] / np.sqrt(k)
    stats["complexity"] = [CALIBRATORS[m].complexity for m in stats.index]
    best = stats["mean"].idxmin()
    stats["within_1se"] = stats["mean"] <= stats.loc[best, "mean"] + stats.loc[best, "se"]
    selected = stats[stats["within_1se"]].sort_values(["complexity", "mean"]).index[0]
    stats["best_mean"] = stats.index == best
    stats["selected"] = stats.index == selected
    order = [m for m in CALIBRATORS if m in stats.index]
    return selected, stats.loc[order]


def _nested_split(X, y, train_idx, test_idx, split_id, seed, inner_splits, inner_repeats,
                  n_estimators, metric):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    inner = repeated_cv(X_tr, y_tr, n_splits=inner_splits, n_repeats=inner_repeats,
                        seed=seed + 1000 * (split_id + 1), n_estimators=n_estimators, n_jobs=1)
    selected, _ = select_method(inner, metric)
    fit = crossfit(X_tr, y_tr, X.iloc[test_idx], seed=seed, cv_seed=seed + split_id,
                   n_estimators=n_estimators)
    frame = _metrics_frame(y.iloc[test_idx], fit.proba)
    frame["selected"] = frame["method"] == selected
    return frame


def nested_cv(X, y, outer_splits: int = 5, outer_repeats: int = 3, inner_splits: int = 5,
              inner_repeats: int = 2, seed: int = 42, n_estimators: int = 300,
              metric: str = "brier", n_jobs: int = -1) -> pd.DataFrame:
    """Outer repeated CV over the full dataset; method selection happens inside each outer
    training set only. Returns every method's outer-fold metrics, flagging the selected one."""
    splitter = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=outer_repeats,
                                       random_state=seed)
    splits = list(splitter.split(X, y))
    frames = Parallel(n_jobs=n_jobs)(
        delayed(_nested_split)(X, y, tr, te, i, seed, inner_splits, inner_repeats,
                               n_estimators, metric)
        for i, (tr, te) in enumerate(splits))
    for i, frame in enumerate(frames):
        frame.insert(0, "repeat", i // outer_splits)
        frame.insert(1, "fold", i % outer_splits)
    return pd.concat(frames, ignore_index=True)
