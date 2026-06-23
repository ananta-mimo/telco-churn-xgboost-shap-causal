"""
src/evaluation/evaluate_model.py
----------------------------------
Reusable evaluation functions for the IBM Telco Churn project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score,
)


def full_report(name: str, y_true, y_prob, threshold: float = 0.5) -> dict:
    """Print classification report and return metrics dict."""
    y_pred  = (y_prob >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc  = average_precision_score(y_true, y_prob)
    f1      = f1_score(y_true, y_pred)

    print(f"{'='*50}")
    print(f"  {name}  (threshold={threshold})")
    print(f"{'='*50}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=["No Churn", "Churn"]))

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1}


def plot_roc_pr(models: list, y_true, save_path: str = None) -> None:
    """
    Plot ROC and PR curves for a list of models.

    Parameters
    ----------
    models : list of dicts, each with keys: name, y_prob
    y_true : true binary labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#4878CF", "#E87B4A", "#78C679", "#9E6DB5"]

    for i, m in enumerate(models):
        color = colors[i % len(colors)]
        # ROC
        fpr, tpr, _ = roc_curve(y_true, m["y_prob"])
        auc = roc_auc_score(y_true, m["y_prob"])
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f'{m["name"]}  (AUC={auc:.3f})')
        # PR
        prec, rec, _ = precision_recall_curve(y_true, m["y_prob"])
        pr_auc = average_precision_score(y_true, m["y_prob"])
        axes[1].plot(rec, prec, color=color, lw=2,
                     label=f'{m["name"]}  (PR-AUC={pr_auc:.3f})')

    axes[0].plot([0,1],[0,1], "k--", lw=1)
    axes[0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    axes[0].legend()

    baseline = np.mean(y_true)
    axes[1].axhline(baseline, color="k", lw=1, linestyle="--",
                    label=f"Baseline ({baseline:.2f})")
    axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()

    plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def find_best_threshold(y_true, y_prob,
                        metric: str = "f1",
                        plot: bool = True) -> float:
    """
    Sweep thresholds and return the value that maximises the chosen metric.

    Parameters
    ----------
    metric : "f1" or "recall" or "precision"
    """
    thresholds = np.arange(0.20, 0.71, 0.01)
    scores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            scores.append(f1_score(y_true, y_pred))
        elif metric == "recall":
            cm = confusion_matrix(y_true, y_pred)
            scores.append(cm[1,1] / (cm[1,:].sum() + 1e-9))
        elif metric == "precision":
            cm = confusion_matrix(y_true, y_pred)
            scores.append(cm[1,1] / (cm[:,1].sum() + 1e-9))

    best_idx = int(np.argmax(scores))
    best_t   = thresholds[best_idx]

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(thresholds, scores, color="#E87B4A", lw=2)
        plt.axvline(best_t, color="red", linestyle=":", lw=1.5,
                    label=f"Best={best_t:.2f}  ({metric}={scores[best_idx]:.3f})")
        plt.title(f"Threshold vs {metric.upper()}", fontweight="bold")
        plt.xlabel("Threshold")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"Best threshold: {best_t:.2f}  |  {metric} = {scores[best_idx]:.4f}")
    return best_t
