"""Figures. Colours follow the method (fixed order), never its rank; text stays in ink tones."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from .experiment import CrossFitResult, paired_differences
from .metrics import calibration_table

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3de"
REFERENCE = "#8a8984"

METHOD_COLORS = {
    "Uncalibrated": REFERENCE,
    "Temperature scaling": "#2a78d6",
    "Platt (sigmoid)": "#eb6834",
    "Logistic recalibration": "#1baf7a",
    "Beta calibration": "#eda100",
    "Isotonic regression": "#e87ba4",
}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "text.color": INK, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.edgecolor": GRID, "axes.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9,
    })


def _strip_box(ax, data: dict[str, np.ndarray], reference_line: float, xlabel: str,
               colors: dict[str, str] | None = None, seed=0):
    """Horizontal box + jittered dots per row, rows in the given order; an empty row is
    labelled as the reference."""
    colors = colors or METHOD_COLORS
    rng = np.random.default_rng(seed)
    for i, (name, values) in enumerate(data.items()):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            ax.text(reference_line, i, "  reference", va="center", ha="left",
                    color=INK_SECONDARY, fontsize=9)
            continue
        ax.boxplot(values, positions=[i], orientation="horizontal", widths=0.5, showfliers=False,
                   medianprops={"color": INK, "linewidth": 1.5},
                   boxprops={"color": INK_SECONDARY}, whiskerprops={"color": INK_SECONDARY},
                   capprops={"color": INK_SECONDARY})
        ax.scatter(values, i + rng.uniform(-0.18, 0.18, len(values)), s=16,
                   color=colors[name], alpha=0.75, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.axvline(reference_line, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(data)), list(data))
    ax.set_ylim(len(data) - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y", visible=False)


def _method_order(columns) -> list[str]:
    return [m for m in METHOD_COLORS if m in set(columns)]


def plot_cv_comparison(cv_results):
    """Paired differences vs the uncalibrated model per CV split, plus the calibration slope."""
    d_brier = paired_differences(cv_results, "brier")
    d_ll = paired_differences(cv_results, "log_loss")
    slope = cv_results.pivot_table(index=["repeat", "fold"], columns="method", values="cal_slope")
    order = _method_order(slope.columns)

    def rows(frame):
        return {m: frame[m] if m in frame else [] for m in order}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    _strip_box(axes[0], rows(d_brier), 0, "Δ Brier vs uncalibrated  (← better)")
    _strip_box(axes[1], rows(d_ll), 0, "Δ log loss vs uncalibrated  (← better)")
    _strip_box(axes[2], rows(slope), 1, "Calibration slope  (target = 1)")
    for ax, title in zip(axes, ["Brier score", "Log loss", "Calibration slope"]):
        ax.set_title(title)
    fig.suptitle("Repeated cross-validation on the development set (one dot = one split)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_reliability(y, proba: dict[str, np.ndarray], n_bins: int = 5, ncols: int = 3):
    """Reliability diagrams as small multiples, one panel per method, same test records."""
    names = list(proba)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.3 * nrows), sharex=True,
                             sharey=True)
    for ax, name in zip(axes.flat, names):
        table = calibration_table(y, proba[name], n_bins)
        ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, linestyle="--")
        ax.errorbar(table["predicted"], table["observed"],
                    yerr=[table["observed"] - table["ci_low"], table["ci_high"] - table["observed"]],
                    fmt="-o", color=METHOD_COLORS[name], ecolor=METHOD_COLORS[name],
                    elinewidth=1, capsize=3, markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
        ax.plot(proba[name], np.full(len(proba[name]), -0.03), "|", color=INK_SECONDARY,
                alpha=0.35, markersize=8)
        ax.set_title(name)
        ax.set(xlim=(-0.05, 1.02), ylim=(-0.06, 1.02))
        ax.set_aspect("equal")
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed proportion")
    fig.suptitle(f"Reliability on the held-out test set ({n_bins} quantile bins, Wilson 95% CI; "
                 "ticks = individual predictions)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_mappings(fit: CrossFitResult):
    """The learned maps p -> p' (fitted on OOF predictions), with the OOF distribution below."""
    grid = np.linspace(0.001, 0.999, 500)
    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08, figure=fig)
    ax, ax_hist = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1.5, linestyle="--", label="Uncalibrated (identity)")
    for name, cal in fit.calibrators.items():
        if name == "Uncalibrated":
            continue
        ax.plot(grid, cal.predict(grid), color=METHOD_COLORS[name], label=name,
                drawstyle="steps-post" if name == "Isotonic regression" else "default")
    ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Calibrated probability")
    ax.set_title("Learned calibration maps (fitted on out-of-fold predictions)", loc="left")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left")
    bins = np.linspace(0, 1, 21)
    ax_hist.hist(fit.oof[fit.y_train == 0], bins=bins, color="#86b6ef", label="No disease")
    ax_hist.hist(fit.oof[fit.y_train == 1], bins=bins, color="#1c5cab", label="Disease",
                 histtype="step", linewidth=2)
    ax_hist.set(xlim=(0, 1), xlabel="Uncalibrated (out-of-fold) probability", ylabel="Records")
    ax_hist.legend(loc="upper center", ncols=2)
    return fig


METRIC_LABELS = {"brier": "Brier", "log_loss": "log loss"}
PROCEDURE = "Selected by inner CV"
PROCEDURE_COLOR = "#4a3aa7"


def plot_nested(nested, metric: str = "brier"):
    """Selection frequency across outer folds, and each option's outer-fold difference vs
    no calibration: the full procedure (selection included) and every fixed method."""
    sel = nested[nested["selected"]].set_index(["repeat", "fold"])[metric]
    wide = nested.pivot_table(index=["repeat", "fold"], columns="method", values=metric)
    diffs = wide.sub(wide["Uncalibrated"], axis=0)
    counts = nested.loc[nested["selected"], "method"].value_counts()
    order = _method_order(wide.columns)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    shown = [m for m in order if m in counts.index]
    ax1.barh(shown, counts[shown], height=0.5, color=[METHOD_COLORS[m] for m in shown])
    for i, m in enumerate(shown):
        ax1.text(counts[m] + 0.2, i, str(counts[m]), va="center", color=INK)
    ax1.set_ylim(len(shown) - 0.5, -0.5)
    ax1.set_xlabel(f"Outer folds where the method was selected (of {len(sel)})")
    ax1.set_title("Which method did the inner CV pick?", loc="left")
    ax1.grid(axis="y", visible=False)

    rows = {PROCEDURE: (sel - wide.loc[sel.index, "Uncalibrated"]).to_numpy()}
    rows.update({m: diffs[m].to_numpy() if m != "Uncalibrated" else [] for m in order})
    _strip_box(ax2, rows, 0, f"Δ {METRIC_LABELS.get(metric, metric)} vs uncalibrated on outer folds  (← better)",
               colors={**METHOD_COLORS, PROCEDURE: PROCEDURE_COLOR})
    ax2.set_title("Nested CV: whole procedure and each fixed method", loc="left")
    fig.tight_layout()
    return fig


def plot_confusion(y, proba: dict[str, np.ndarray], threshold: float = 0.5):
    names = list(proba)
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.2))
    mats = {n: confusion_matrix(y, (proba[n] >= threshold).astype(int), labels=[0, 1]) for n in names}
    vmax = max(m.max() for m in mats.values())
    for ax, name in zip(np.atleast_1d(axes), names):
        ConfusionMatrixDisplay(mats[name], display_labels=["No disease", "Disease"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d", im_kw={"vmin": 0, "vmax": vmax})
        ax.set_title(f"{name}  (threshold {threshold:.2f})")
        ax.set(xlabel="Predicted class", ylabel="Observed outcome")
        ax.grid(False)
    fig.tight_layout()
    return fig


def plot_roc_pr(y, proba: dict[str, np.ndarray]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, (name, p) in enumerate(proba.items()):
        style = "-" if i == 0 else "--"
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, style, color=METHOD_COLORS[name], label=f"{name}  AUC = {roc_auc_score(y, p):.3f}")
        ax2.step(rec, prec, where="post", linestyle=style, color=METHOD_COLORS[name],
                 label=f"{name}  AP = {average_precision_score(y, p):.3f}")
    ax1.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax2.axhline(np.mean(y), color=INK_SECONDARY, linewidth=1, linestyle=":", label="Event rate")
    ax1.set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC")
    ax2.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title="Precision-recall")
    for ax in (ax1, ax2):
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.legend(loc="lower right" if ax is ax1 else "lower left")
    fig.tight_layout()
    return fig
