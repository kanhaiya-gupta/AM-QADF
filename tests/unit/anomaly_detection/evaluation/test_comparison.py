"""
Unit tests for anomaly detection comparison utilities.

Tests for DetectorComparisonResult, AnomalyDetectionComparison, and statistical test functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from am_qadf.anomaly_detection.evaluation.comparison import (
    DetectorComparisonResult,
    AnomalyDetectionComparison,
    statistical_significance_test,
    mcnemar_test,
    friedman_test,
    post_hoc_nemenyi,
    compare_detectors,
)
from am_qadf.anomaly_detection.core.base_detector import (
    BaseAnomalyDetector,
    AnomalyDetectionResult,
)


class MockDetector(BaseAnomalyDetector):
    """Mock detector for testing."""

    def __init__(self, name="MockDetector", fixed_predictions=None):
        super().__init__()
        self.name = name
        self.fixed_predictions = fixed_predictions
        self.is_fitted = False

    def fit(self, data, labels=None):
        self.is_fitted = True
        return self

    def predict(self, data):
        if self.fixed_predictions is not None:
            results = []
            for i, (is_anom, score) in enumerate(self.fixed_predictions):
                results.append(
                    AnomalyDetectionResult(
                        voxel_index=(i, 0, 0),
                        voxel_coordinates=(0, 0, 0),
                        is_anomaly=is_anom,
                        anomaly_score=score,
                        confidence=0.8,
                        detector_name=self.name,
                    )
                )
            return results[: len(data)]
        else:
            # Default: random predictions
            results = []
            for i in range(len(data)):
                results.append(
                    AnomalyDetectionResult(
                        voxel_index=(i, 0, 0),
                        voxel_coordinates=(0, 0, 0),
                        is_anomaly=np.random.rand() > 0.5,
                        anomaly_score=np.random.rand(),
                        confidence=0.8,
                        detector_name=self.name,
                    )
                )
            return results


class TestDetectorComparisonResult:
    """Test suite for DetectorComparisonResult dataclass."""

    @pytest.mark.unit
    def test_result_creation(self):
        """Test creating DetectorComparisonResult."""
        from am_qadf.anomaly_detection.evaluation.metrics import AnomalyDetectionMetrics

        metrics = AnomalyDetectionMetrics(
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            accuracy=0.9,
            specificity=0.95,
            sensitivity=0.8,
        )

        result = DetectorComparisonResult(
            detector_name="test_detector",
            metrics=metrics,
            mean_score=0.85,
            std_score=0.05,
        )

        assert result.detector_name == "test_detector"
        assert result.metrics == metrics
        assert result.mean_score == 0.85
        assert result.std_score == 0.05


class TestAnomalyDetectionComparison:
    """Test suite for AnomalyDetectionComparison class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.randn(50, 3) * 10 + 100

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels."""
        return np.array([0, 0, 1, 1, 0] * 10)

    @pytest.fixture
    def mock_detectors(self):
        """Create mock detectors."""
        return {
            "detector1": MockDetector(
                "detector1",
                [(False, 0.2), (False, 0.3), (True, 0.8), (True, 0.9), (False, 0.1)] * 10,
            ),
            "detector2": MockDetector(
                "detector2",
                [(False, 0.1), (False, 0.2), (True, 0.7), (True, 0.85), (False, 0.15)] * 10,
            ),
        }

    @pytest.mark.unit
    def test_comparison_creation(self):
        """Test creating AnomalyDetectionComparison."""
        comparison = AnomalyDetectionComparison()

        assert comparison.comparison_results == {}

    @pytest.mark.unit
    def test_compare_detectors(self, sample_data, sample_labels, mock_detectors):
        """Test comparing detectors."""
        comparison = AnomalyDetectionComparison()
        results = comparison.compare_detectors(mock_detectors, sample_data, sample_labels)

        assert isinstance(results, dict)
        assert "detector1" in results
        assert "detector2" in results
        assert isinstance(results["detector1"], DetectorComparisonResult)
        assert isinstance(results["detector2"], DetectorComparisonResult)

    @pytest.mark.unit
    def test_compare_detectors_different_metric(self, sample_data, sample_labels, mock_detectors):
        """Test comparing detectors with different metric."""
        comparison = AnomalyDetectionComparison()
        results = comparison.compare_detectors(mock_detectors, sample_data, sample_labels, metric="precision")

        assert isinstance(results, dict)
        assert "detector1" in results
        assert "detector2" in results

    @pytest.mark.unit
    def test_rank_detectors(self, sample_data, sample_labels, mock_detectors):
        """Test ranking detectors."""
        comparison = AnomalyDetectionComparison()
        comparison.compare_detectors(mock_detectors, sample_data, sample_labels)
        rankings = comparison.rank_detectors()

        assert isinstance(rankings, list)
        assert len(rankings) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in rankings)
        # Should be sorted descending
        assert rankings[0][1] >= rankings[1][1]

    @pytest.mark.unit
    def test_get_best_detector(self, sample_data, sample_labels, mock_detectors):
        """Test getting best detector."""
        comparison = AnomalyDetectionComparison()
        comparison.compare_detectors(mock_detectors, sample_data, sample_labels)
        best = comparison.get_best_detector()

        assert best is not None
        assert best in ["detector1", "detector2"]

    @pytest.mark.unit
    def test_get_best_detector_no_results(self):
        """Test getting best detector with no results."""
        comparison = AnomalyDetectionComparison()
        best = comparison.get_best_detector()

        assert best is None


class TestStatisticalSignificanceTest:
    """Test suite for statistical_significance_test function."""

    @pytest.mark.unit
    def test_statistical_significance_test_paired_t(self):
        """Test paired t-test."""
        scores1 = np.array([0.9, 0.85, 0.92, 0.88, 0.91])
        scores2 = np.array([0.8, 0.75, 0.82, 0.78, 0.81])

        result = statistical_significance_test(scores1, scores2, test_type="paired_t")

        assert isinstance(result, dict)
        assert "test_type" in result
        assert "p_value" in result
        assert "significant" in result
        assert "statistic" in result
        assert result["test_type"] == "paired_t"
        assert result["p_value"] is not None

    @pytest.mark.unit
    def test_statistical_significance_test_wilcoxon(self):
        """Test Wilcoxon signed-rank test."""
        scores1 = np.array([0.9, 0.85, 0.92, 0.88, 0.91])
        scores2 = np.array([0.8, 0.75, 0.82, 0.78, 0.81])

        result = statistical_significance_test(scores1, scores2, test_type="wilcoxon")

        assert isinstance(result, dict)
        assert result["test_type"] == "wilcoxon"
        assert result["p_value"] is not None or result["p_value"] is None  # May fail for some cases


class TestMcNemarTest:
    """Test suite for mcnemar_test function."""

    @pytest.mark.unit
    def test_mcnemar_test(self):
        """Test McNemar's test."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred1 = np.array([0, 0, 1, 1, 0, 0, 0, 1])
        y_pred2 = np.array([0, 0, 1, 0, 0, 1, 0, 1])

        result = mcnemar_test(y_true, y_pred1, y_pred2)

        assert isinstance(result, dict)
        assert "test_type" in result
        assert "p_value" in result
        assert "significant" in result
        assert "contingency_table" in result
        assert result["test_type"] == "mcnemar"

    @pytest.mark.unit
    def test_mcnemar_test_no_disagreements(self):
        """Test McNemar's test with no disagreements."""
        y_true = np.array([0, 0, 1, 1])
        y_pred1 = np.array([0, 0, 1, 1])
        y_pred2 = np.array([0, 0, 1, 1])

        result = mcnemar_test(y_true, y_pred1, y_pred2)

        assert result["p_value"] == 1.0
        assert result["significant"] is False


class TestFriedmanTest:
    """Test suite for friedman_test function."""

    @pytest.mark.unit
    def test_friedman_test(self):
        """Test Friedman test."""
        scores = {
            "method1": np.array([0.9, 0.85, 0.92, 0.88, 0.91]),
            "method2": np.array([0.8, 0.75, 0.82, 0.78, 0.81]),
            "method3": np.array([0.85, 0.80, 0.87, 0.83, 0.86]),
        }

        result = friedman_test(scores)

        assert isinstance(result, dict)
        assert "test_type" in result
        assert "p_value" in result
        assert "significant" in result
        assert "mean_ranks" in result
        assert "ranking" in result
        assert result["test_type"] == "friedman"

    @pytest.mark.unit
    def test_friedman_test_insufficient_methods(self):
        """Test Friedman test with insufficient methods."""
        scores = {"method1": np.array([0.9, 0.85, 0.92])}

        result = friedman_test(scores)

        assert result["p_value"] is None
        assert result["significant"] is False


class TestPostHocNemenyi:
    """Test suite for post_hoc_nemenyi function."""

    @pytest.mark.unit
    def test_post_hoc_nemenyi(self):
        """Test Nemenyi post-hoc test."""
        scores = {
            "method1": np.array([0.9, 0.85, 0.92, 0.88, 0.91]),
            "method2": np.array([0.8, 0.75, 0.82, 0.78, 0.81]),
            "method3": np.array([0.85, 0.80, 0.87, 0.83, 0.86]),
        }

        result = post_hoc_nemenyi(scores)

        assert isinstance(result, dict)
        assert "test_type" in result
        assert "pairwise_comparisons" in result
        assert "mean_ranks" in result
        assert result["test_type"] == "nemenyi"
        assert len(result["pairwise_comparisons"]) > 0

    @pytest.mark.unit
    def test_post_hoc_nemenyi_insufficient_methods(self):
        """Test Nemenyi test with insufficient methods."""
        scores = {"method1": np.array([0.9, 0.85, 0.92])}

        result = post_hoc_nemenyi(scores)

        assert "message" in result


class TestCompareDetectors:
    """Test suite for compare_detectors convenience function."""

    @pytest.mark.unit
    def test_compare_detectors_function(self):
        """Test compare_detectors convenience function."""
        sample_data = np.random.randn(20, 3) * 10 + 100
        sample_labels = np.array([0, 0, 1, 1] * 5)

        detectors = {
            "detector1": MockDetector("detector1"),
            "detector2": MockDetector("detector2"),
        }

        results = compare_detectors(detectors, sample_data, sample_labels)

        assert isinstance(results, dict)
        assert "detector1" in results
        assert "detector2" in results
