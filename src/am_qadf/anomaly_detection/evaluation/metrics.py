"""
Performance Metrics for Anomaly Detection

Provides comprehensive metrics for evaluating anomaly detection performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging

try:
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionMetrics:
    """Comprehensive metrics for anomaly detection evaluation."""

    # Classification metrics
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    sensitivity: float  # Same as recall

    # Ranking metrics
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None  # Precision-Recall AUC
    average_precision: Optional[float] = None

    # Statistical metrics
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    true_positive_rate: float = 0.0
    matthews_correlation: Optional[float] = None

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None

    # Additional info
    n_samples: int = 0
    n_anomalies: int = 0
    n_normal: int = 0


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: Optional[np.ndarray] = None
) -> AnomalyDetectionMetrics:
    """
    Calculate classification metrics for anomaly detection.

    Args:
        y_true: True labels (0=normal, 1=anomaly)
        y_pred: Predicted labels (0=normal, 1=anomaly)
        y_scores: Optional anomaly scores (for ranking metrics)

    Returns:
        AnomalyDetectionMetrics object
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for metrics. Install with: pip install scikit-learn")

    # Basic classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall  # Same as recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = sensitivity

    # Matthews Correlation Coefficient
    mcc = None
    if (tp + tn + fp + fn) > 0:
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator > 0:
            mcc = (tp * tn - fp * fn) / denominator

    # Ranking metrics (if scores provided)
    roc_auc = None
    pr_auc = None
    avg_precision = None

    if y_scores is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            pass

        try:
            pr_auc = average_precision_score(y_true, y_scores)
            avg_precision = pr_auc
        except ValueError:
            pass

    return AnomalyDetectionMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy,
        specificity=specificity,
        sensitivity=sensitivity,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        average_precision=avg_precision,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        true_positive_rate=tpr,
        matthews_correlation=mcc,
        confusion_matrix=cm,
        n_samples=len(y_true),
        n_anomalies=int(np.sum(y_true)),
        n_normal=int(len(y_true) - np.sum(y_true)),
    )


def calculate_ranking_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Calculate ranking metrics for anomaly detection.

    Args:
        y_true: True labels
        y_scores: Anomaly scores

    Returns:
        Dictionary of ranking metrics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for metrics")

    metrics = {}

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_scores)
        metrics["average_precision"] = metrics["pr_auc"]
    except ValueError:
        metrics["pr_auc"] = None
        metrics["average_precision"] = None

    return metrics


def calculate_statistical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical metrics for anomaly detection.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of statistical metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "true_negative_rate": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }

    # Matthews Correlation Coefficient
    if (tp + tn + fp + fn) > 0:
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator > 0:
            metrics["matthews_correlation"] = (tp * tn - fp * fn) / denominator
        else:
            metrics["matthews_correlation"] = 0.0
    else:
        metrics["matthews_correlation"] = 0.0

    return metrics
