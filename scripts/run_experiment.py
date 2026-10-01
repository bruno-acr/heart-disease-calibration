"""Run the full experiment and write tables to results/ and figures to figures/.

    python scripts/run_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from heartcal import plotting  # noqa: E402
from heartcal.data import load_cleveland  # noqa: E402
from heartcal.experiment import crossfit, nested_cv, repeated_cv, select_method, summarize_cv  # noqa: E402
from heartcal.metrics import (  # noqa: E402
    brier_skill_score, classification_metrics, log_loss, paired_bootstrap, probability_metrics,
)

SEED = 42
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"


def main():
    start = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    plotting.set_style()

    X, y = load_cleveland()
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 1. Compare calibrators with repeated CV on the development set; pre-specified selection.
    cv = repeated_cv(X_dev, y_dev, n_splits=5, n_repeats=5, seed=SEED)
    cv.to_csv(RESULTS / "cv_development.csv", index=False)
    summarize_cv(cv).round(4).to_csv(RESULTS / "cv_summary.csv")
    selected, selection = select_method(cv, "brier")
    selection.round(5).to_csv(RESULTS / "selection.csv")
    plotting.plot_cv_comparison(cv).savefig(FIGURES / "cv_comparison.png")

    # 2. Refit everything on the full development set; evaluate once on the held-out test set.
    fit = crossfit(X_dev, y_dev, X_test, seed=SEED)
    brier_ref = brier_score_loss(y_test, np.full(len(y_test), y_dev.mean()))
    test = pd.DataFrame([{"method": m, **probability_metrics(y_test, p)} for m, p in fit.proba.items()])
    test["brier_skill"] = brier_skill_score(test["brier"], brier_ref)
    test.round(4).to_csv(RESULTS / "test_metrics.csv", index=False)

    boot = pd.DataFrame([
        {"method": m, "metric": name, **paired_bootstrap(y_test, fit.proba["Uncalibrated"], p, fn, seed=SEED)}
        for m, p in fit.proba.items() if m != "Uncalibrated"
        for name, fn in [("brier", brier_score_loss), ("log_loss", log_loss)]
    ])
    boot.round(4).to_csv(RESULTS / "test_bootstrap.csv", index=False)

    # Classification comparison: the selected method, or the best-mean calibrator if no
    # calibration was selected, so that there is always something to compare against.
    comparator = selected if selected != "Uncalibrated" else (
        selection.drop(index="Uncalibrated")["mean"].idxmin())
    pair = {"Uncalibrated": fit.proba["Uncalibrated"], comparator: fit.proba[comparator]}
    pd.DataFrame({m: classification_metrics(y_test, p) for m, p in pair.items()}).T.round(4).to_csv(
        RESULTS / "test_classification.csv")
    plotting.plot_reliability(y_test, fit.proba).savefig(FIGURES / "reliability_test.png")
    plotting.plot_mappings(fit).savefig(FIGURES / "calibration_maps.png")
    plotting.plot_confusion(y_test, pair).savefig(FIGURES / "confusion_matrices.png")
    plotting.plot_roc_pr(y_test, pair).savefig(FIGURES / "roc_pr.png")

    # 3. Nested CV: performance of the whole procedure, including the selection step.
    nested = nested_cv(X, y, outer_splits=5, outer_repeats=3, inner_splits=5, inner_repeats=2, seed=SEED)
    nested.to_csv(RESULTS / "nested_cv.csv", index=False)
    plotting.plot_nested(nested).savefig(FIGURES / "nested_cv.png")

    sel = nested[nested["selected"]].set_index(["repeat", "fold"])
    unc = nested[nested["method"] == "Uncalibrated"].set_index(["repeat", "fold"])
    summary = {
        "selected_on_development": selected,
        "test": test.set_index("method")[["brier", "log_loss", "auc", "cal_intercept", "cal_slope"]].round(4).to_dict("index"),
        "nested": {
            "selection_counts": sel["method"].value_counts().to_dict(),
            "brier_uncalibrated_mean": round(unc["brier"].mean(), 4),
            "brier_procedure_mean": round(sel["brier"].mean(), 4),
            "brier_improved_in_folds": int((sel["brier"] < unc["brier"]).sum()),
            "n_outer_folds": len(sel),
            "brier_mean_by_fixed_method": nested.groupby("method")["brier"].mean().round(4).to_dict(),
            "log_loss_mean_by_fixed_method": nested.groupby("method")["log_loss"].mean().round(4).to_dict(),
        },
        "classification_comparator": comparator,
        "runtime_seconds": round(time.time() - start, 1),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
