"""
Method Comparison and Statistical Significance Testing

Provides tools for comparing different anomaly detection methods
and testing statistical significance of differences.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from scipy import stats

from ..core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyDetectionConfig,
)

# Try relative import first, then absolute import (for importlib compatibility)
try:
    from .metrics import AnomalyDetectionMetrics, calculate_classification_metrics
except ImportError:
    # Try absolute import (for when loaded via importlib)
    import sys
    from pathlib import Path
    import importlib.util

    metrics_path = None
    try:
        if "__file__" in globals():
            current_file = Path(__file__)
            metrics_path = current_file.parent / "metrics.py"
    except:
        pass

    if metrics_path is None or not metrics_path.exists():
        for base_path in [Path.cwd()] + [Path(p) for p in sys.path if p]:
            potential_path = (
                base_path / "src" / "data_pipeline" / "processing" / "anomaly_detection" / "evaluation" / "metrics.py"
            )
            if potential_path.exists():
                metrics_path = potential_path
                break

    if metrics_path and metrics_path.exists():
        spec = importlib.util.spec_from_file_location("metrics", metrics_path)
        metrics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics_module)
        AnomalyDetectionMetrics = metrics_module.AnomalyDetectionMetrics
        calculate_classification_metrics = metrics_module.calculate_classification_metrics
    else:
        raise ImportError("Could not find metrics module")

logger = logging.getLogger(__name__)


@dataclass
class DetectorComparisonResult:
    """Result of comparing multiple detectors."""

    detector_name: str
    metrics: AnomalyDetectionMetrics
    mean_score: float
    std_score: float


class AnomalyDetectionComparison:
    """
    Framework for comparing multiple anomaly detection methods.

    Provides comparison capabilities including statistical significance testing.
    """

    def __init__(self):
        """Initialize comparison framework."""
        self.comparison_results: Dict[str, DetectorComparisonResult] = {}

    def compare_detectors(
        self,
        detectors: Dict[str, BaseAnomalyDetector],
        data: Union[np.ndarray, list],
        labels: np.ndarray,
        metric: str = "f1_score",
    ) -> Dict[str, DetectorComparisonResult]:
        """
        Compare multiple detectors on the same dataset.

        Args:
            detectors: Dictionary mapping names to detector instances
            data: Test data
            labels: Ground truth labels
            metric: Metric to use for comparison ('f1_score', 'precision', 'recall', 'roc_auc', etc.)

        Returns:
            Dictionary mapping detector names to comparison results
        """
        results = {}

        # Convert to array if needed
        if isinstance(data, list):
            data = np.array(data)

        for name, detector in detectors.items():
            # Predict
            predictions = detector.predict(data)
            y_pred = np.array([r.is_anomaly for r in predictions])
            y_scores = np.array([r.anomaly_score for r in predictions])

            # Calculate metrics
            metrics = calculate_classification_metrics(labels, y_pred, y_scores)

            # Get the requested metric
            metric_value = getattr(metrics, metric, 0.0)

            result = DetectorComparisonResult(
                detector_name=name,
                metrics=metrics,
                mean_score=metric_value,
                std_score=0.0,  # Single evaluation, no std
            )
            results[name] = result

        self.comparison_results = results
        return results

    def rank_detectors(self, metric: str = "f1_score") -> List[Tuple[str, float]]:
        """
        Rank detectors by a specific metric.

        Args:
            metric: Metric to use for ranking

        Returns:
            List of (detector_name, score) tuples, sorted by score
        """
        rankings = []
        for name, result in self.comparison_results.items():
            score = getattr(result.metrics, metric, 0.0)
            rankings.append((name, score))

        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_best_detector(self, metric: str = "f1_score") -> Optional[str]:
        """
        Get the best detector for a specific metric.

        Args:
            metric: Metric to use for selection

        Returns:
            Name of best detector, or None if no results
        """
        rankings = self.rank_detectors(metric)
        return rankings[0][0] if rankings else None

    def compare_with_statistical_tests(
        self,
        detectors: Dict[str, BaseAnomalyDetector],
        data: Union[np.ndarray, list],
        labels: np.ndarray,
        metric: str = "f1_score",
        cv_folds: int = 5,
        test_type: str = "friedman",
    ) -> Dict[str, Any]:
        """
        Compare detectors with statistical significance testing using cross-validation.

        Args:
            detectors: Dictionary mapping names to detector instances
            data: Test data
            labels: Ground truth labels
            metric: Metric to use for comparison
            cv_folds: Number of cross-validation folds
            test_type: Type of statistical test ('friedman', 'paired_t', 'mcnemar')

        Returns:
            Dictionary with comparison results and statistical test results
        """
        from sklearn.model_selection import StratifiedKFold

        if isinstance(data, list):
            data = np.array(data)

        # Collect scores from cross-validation
        cv_scores = {name: [] for name in detectors.keys()}
        cv_predictions = {name: [] for name in detectors.keys()}

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in skf.split(data, labels):
            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            for name, detector in detectors.items():
                try:
                    # Fit on training fold
                    detector.fit(X_train)

                    # Predict on test fold
                    predictions = detector.predict(X_test)
                    y_pred = np.array([r.is_anomaly for r in predictions])
                    y_scores = np.array([r.anomaly_score for r in predictions])

                    # Calculate metrics
                    metrics = calculate_classification_metrics(y_test, y_pred, y_scores)
                    metric_value = getattr(metrics, metric, 0.0)

                    cv_scores[name].append(metric_value)
                    cv_predictions[name].append(y_pred)
                except Exception as e:
                    logger.warning(f"Error in CV fold for {name}: {e}")
                    cv_scores[name].append(0.0)
                    cv_predictions[name].append(np.zeros_like(y_test))

        # Convert to numpy arrays
        cv_scores_array = {name: np.array(scores) for name, scores in cv_scores.items()}

        # Perform statistical tests
        test_results = {}

        if test_type == "friedman" and len(detectors) >= 2:
            test_results["friedman"] = friedman_test(cv_scores_array)
            if test_results["friedman"].get("significant", False):
                test_results["nemenyi"] = post_hoc_nemenyi(cv_scores_array)

        # Pairwise tests
        pairwise_tests = {}
        method_names = list(detectors.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                name1, name2 = method_names[i], method_names[j]

                # Paired t-test
                pairwise_tests[f"{name1}_vs_{name2}"] = {
                    "paired_t": statistical_significance_test(
                        cv_scores_array[name1],
                        cv_scores_array[name2],
                        test_type="paired_t",
                    ),
                    "mcnemar": (
                        mcnemar_test(
                            labels,
                            cv_predictions[name1][0],  # Use first fold predictions
                            cv_predictions[name2][0],
                        )
                        if len(cv_predictions[name1]) > 0
                        else None
                    ),
                }

        test_results["pairwise"] = pairwise_tests

        # Calculate mean metrics
        mean_results = {}
        for name in detectors.keys():
            mean_score = np.mean(cv_scores_array[name])
            std_score = np.std(cv_scores_array[name])

            # Get overall metrics (on full dataset)
            try:
                detector = detectors[name]
                detector.fit(data)
                predictions = detector.predict(data)
                y_pred = np.array([r.is_anomaly for r in predictions])
                y_scores = np.array([r.anomaly_score for r in predictions])
                metrics = calculate_classification_metrics(labels, y_pred, y_scores)

                mean_results[name] = DetectorComparisonResult(
                    detector_name=name,
                    metrics=metrics,
                    mean_score=mean_score,
                    std_score=std_score,
                )
            except Exception as e:
                logger.warning(f"Error calculating overall metrics for {name}: {e}")

        return {
            "comparison_results": mean_results,
            "cv_scores": cv_scores_array,
            "statistical_tests": test_results,
        }


def statistical_significance_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    test_type: str = "paired_t",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform statistical significance test between two sets of scores.

    Args:
        scores1: Scores from method 1
        scores2: Scores from method 2
        test_type: Type of test ('paired_t', 'mcnemar', 'wilcoxon')
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with test results including p-value and significance
    """
    results = {
        "test_type": test_type,
        "p_value": None,
        "significant": False,
        "statistic": None,
        "alpha": alpha,
        "mean_diff": float(np.mean(scores1) - np.mean(scores2)),
    }

    if test_type == "paired_t":
        # Paired t-test
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        results["statistic"] = float(statistic)
        results["p_value"] = float(p_value)
        results["significant"] = p_value < alpha

    elif test_type == "wilcoxon":
        # Wilcoxon signed-rank test (non-parametric)
        try:
            statistic, p_value = stats.wilcoxon(scores1, scores2)
            results["statistic"] = float(statistic)
            results["p_value"] = float(p_value)
            results["significant"] = p_value < alpha
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            results["p_value"] = None

    elif test_type == "mcnemar":
        # McNemar's test (for binary classifications)
        # Requires confusion matrices - simplified version
        # This would need actual predictions, not just scores
        logger.warning("McNemar test requires binary predictions, not scores")
        results["p_value"] = None

    return results


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform McNemar's test for comparing two binary classifiers.

    Args:
        y_true: True labels
        y_pred1: Predictions from method 1
        y_pred2: Predictions from method 2
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with test results
    """
    from scipy.stats import chi2

    # Create contingency table
    # Both correct, both wrong, method1 correct method2 wrong, method1 wrong method2 correct
    both_correct = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    both_wrong = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    method1_correct = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    method2_correct = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))

    # McNemar's test statistic
    b = method1_correct
    c = method2_correct

    if b + c == 0:
        # No disagreements
        return {
            "test_type": "mcnemar",
            "p_value": 1.0,
            "significant": False,
            "statistic": 0.0,
            "alpha": alpha,
            "contingency_table": {
                "both_correct": int(both_correct),
                "both_wrong": int(both_wrong),
                "method1_only_correct": int(b),
                "method2_only_correct": int(c),
            },
        }

    # Chi-square statistic (with continuity correction)
    chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return {
        "test_type": "mcnemar",
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "statistic": float(chi2_stat),
        "alpha": alpha,
        "contingency_table": {
            "both_correct": int(both_correct),
            "both_wrong": int(both_wrong),
            "method1_only_correct": int(b),
            "method2_only_correct": int(c),
        },
    }


def friedman_test(scores: Dict[str, np.ndarray], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform Friedman test for comparing multiple methods.

    Args:
        scores: Dictionary mapping method names to score arrays
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with test results including p-value and rankings
    """
    if len(scores) < 2:
        return {
            "test_type": "friedman",
            "p_value": None,
            "significant": False,
            "statistic": None,
            "alpha": alpha,
            "message": "Need at least 2 methods for Friedman test",
        }

    # Convert to array format
    method_names = list(scores.keys())
    score_matrix = np.array([scores[name] for name in method_names]).T

    n_samples, n_methods = score_matrix.shape

    if n_samples < 2:
        return {
            "test_type": "friedman",
            "p_value": None,
            "significant": False,
            "statistic": None,
            "alpha": alpha,
            "message": "Need at least 2 samples for Friedman test",
        }

    # Rank scores within each sample (row)
    ranks = np.zeros_like(score_matrix)
    for i in range(n_samples):
        ranks[i, :] = stats.rankdata(score_matrix[i, :])

    # Calculate mean ranks
    mean_ranks = np.mean(ranks, axis=0)

    # Friedman statistic
    chi2_f = (12 * n_samples / (n_methods * (n_methods + 1))) * (
        np.sum(mean_ranks**2) - (n_methods * (n_methods + 1) ** 2) / 4
    )

    # p-value from chi-square distribution
    p_value = 1 - stats.chi2.cdf(chi2_f, df=n_methods - 1)

    # Create ranking
    ranking = sorted(zip(method_names, mean_ranks), key=lambda x: x[1])

    return {
        "test_type": "friedman",
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "statistic": float(chi2_f),
        "alpha": alpha,
        "mean_ranks": {name: float(rank) for name, rank in zip(method_names, mean_ranks)},
        "ranking": [(name, float(rank)) for name, rank in ranking],
        "n_samples": int(n_samples),
        "n_methods": int(n_methods),
    }


def post_hoc_nemenyi(scores: Dict[str, np.ndarray], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform Nemenyi post-hoc test after Friedman test.

    Args:
        scores: Dictionary mapping method names to score arrays
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with pairwise comparisons
    """
    from scipy.stats import norm

    method_names = list(scores.keys())
    n_methods = len(method_names)
    n_samples = len(scores[method_names[0]])

    if n_methods < 2:
        return {"message": "Need at least 2 methods for post-hoc test"}

    # Calculate mean ranks (same as Friedman)
    score_matrix = np.array([scores[name] for name in method_names]).T
    ranks = np.zeros_like(score_matrix)
    for i in range(n_samples):
        ranks[i, :] = stats.rankdata(score_matrix[i, :])
    mean_ranks = np.mean(ranks, axis=0)

    # Critical difference (CD) for Nemenyi test
    q_alpha = stats.norm.ppf(1 - alpha / (n_methods * (n_methods - 1) / 2))
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_samples))

    # Pairwise comparisons
    pairwise_results = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            rank_diff = abs(mean_ranks[i] - mean_ranks[j])
            significant = rank_diff > cd

            pairwise_results.append(
                {
                    "method1": method_names[i],
                    "method2": method_names[j],
                    "rank_diff": float(rank_diff),
                    "critical_difference": float(cd),
                    "significant": significant,
                }
            )

    return {
        "test_type": "nemenyi",
        "alpha": alpha,
        "critical_difference": float(cd),
        "pairwise_comparisons": pairwise_results,
        "mean_ranks": {name: float(rank) for name, rank in zip(method_names, mean_ranks)},
    }


def compare_detectors(
    detectors: Dict[str, BaseAnomalyDetector],
    data: Union[np.ndarray, list],
    labels: np.ndarray,
    metric: str = "f1_score",
) -> Dict[str, DetectorComparisonResult]:
    """
    Convenience function to compare multiple detectors.

    Args:
        detectors: Dictionary mapping names to detector instances
        data: Test data
        labels: Ground truth labels
        metric: Metric to use for comparison

    Returns:
        Dictionary mapping detector names to comparison results
    """
    comparison = AnomalyDetectionComparison()
    return comparison.compare_detectors(detectors, data, labels, metric)
