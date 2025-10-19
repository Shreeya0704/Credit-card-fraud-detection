import logging
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate the Area Under the Precision-Recall Curve."""
    return average_precision_score(y_true, y_score)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate the Area Under the Receiver Operating Characteristic Curve."""
    return roc_auc_score(y_true, y_score)


def recall_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.005
) -> float:
    """
    Calculate recall at a specific false positive rate.

    Args:
        y_true: True labels.
        y_score: Predicted scores.
        target_fpr: The desired false positive rate.

    Returns:
        The recall value at the given FPR.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Find the recall (tpr) corresponding to the first fpr that is >= target_fpr
    try:
        recall = tpr[np.min(np.where(fpr >= target_fpr))]
    except (ValueError, IndexError):
        recall = 0.0
        logger.warning(f"Could not find a threshold for target FPR {target_fpr}. Returning recall=0.0")
    return recall


def precision_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int
) -> float:
    """
    Calculate precision for the top K highest-scored predictions.

    Args:
        y_true: True labels.
        y_score: Predicted scores.
        k: The number of top predictions to consider.

    Returns:
        The precision value for the top K predictions.
    """
    # Sort scores and corresponding true labels
    idx = np.argsort(y_score)[::-1]
    top_k_true = y_true[idx][:k]

    return np.sum(top_k_true) / k


def expected_cost(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    threshold: float,
) -> float:
    """
    Calculate the total expected cost for a given threshold.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities for the positive class.
        cost_fp: The cost of a single false positive.
        cost_fn: The cost of a single false negative.
        threshold: The probability threshold to use for classification.

    Returns:
        The total expected cost.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    return total_cost


def plot_calibration_curve(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> plt.Figure:
    """
    Plots a calibration curve for a model's predictions.

    Args:
        y_true: True labels.
        y_score: Predicted scores.
        n_bins: Number of bins to use for calibration.

    Returns:
        A matplotlib Figure object containing the plot.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy='uniform')

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.tight_layout()

    return fig
