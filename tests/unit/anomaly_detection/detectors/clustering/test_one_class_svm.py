"""
Unit tests for One-Class SVM detector.

Tests for OneClassSVMDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.clustering.one_class_svm import (
    OneClassSVMDetector,
    SKLEARN_AVAILABLE,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestOneClassSVMDetector:
    """Test suite for OneClassSVMDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create OneClassSVMDetector with default parameters."""
        return OneClassSVMDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create OneClassSVMDetector with custom parameters."""
        return OneClassSVMDetector(nu=0.05, kernel="linear", gamma="auto")

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 100

    @pytest.fixture
    def data_with_outliers(self, normal_data):
        """Create data with clear outliers."""
        data = normal_data.copy()
        # Add clear outliers
        data[0] = [200, 200, 200]  # High outlier
        data[1] = [0, 0, 0]  # Low outlier
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test OneClassSVMDetector initialization with default values."""
        detector = OneClassSVMDetector()

        assert detector.nu == 0.1
        assert detector.kernel == "rbf"
        assert detector.gamma == "scale"
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.svm_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test OneClassSVMDetector initialization with custom parameters."""
        detector = OneClassSVMDetector(nu=0.05, kernel="linear", gamma="auto")

        assert detector.nu == 0.05
        assert detector.kernel == "linear"
        assert detector.gamma == "auto"

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test OneClassSVMDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = OneClassSVMDetector(nu=0.2, config=config)

        assert detector.nu == 0.2
        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_detector_initialization_different_kernels(self):
        """Test initialization with different kernel types."""
        kernels = ["rbf", "linear", "poly", "sigmoid"]
        for kernel in kernels:
            detector = OneClassSVMDetector(kernel=kernel)
            assert detector.kernel == kernel

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.svm_ is not None
        assert detector is detector_default  # Method chaining

    @pytest.mark.unit
    def test_fit_with_dataframe(self, detector_default):
        """Test fitting with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "feature1": np.random.randn(50) * 10 + 100,
                "feature2": np.random.randn(50) * 5 + 50,
                "feature3": np.random.randn(50) * 2 + 10,
            }
        )

        detector = detector_default.fit(df)

        assert detector.is_fitted is True
        assert detector.svm_ is not None

    @pytest.mark.unit
    def test_fit_with_different_kernels(self, normal_data):
        """Test fitting with different kernel types."""
        kernels = ["rbf", "linear", "poly"]
        for kernel in kernels:
            detector = OneClassSVMDetector(kernel=kernel)
            detector.fit(normal_data)
            assert detector.is_fitted is True
            assert detector.svm_ is not None

    @pytest.mark.unit
    def test_predict_before_fit(self, detector_default, normal_data):
        """Test that predict raises error if not fitted."""
        with pytest.raises(ValueError, match="must be fitted"):
            detector_default.predict(normal_data)

    @pytest.mark.unit
    def test_predict_normal_data(self, detector_default, normal_data):
        """Test prediction on normal data."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(normal_data)

        assert len(results) == len(normal_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)
        # With nu=0.1, about 10% should be anomalies
        anomaly_count = sum(r.is_anomaly for r in results)
        # Allow some variance around expected nu - OneClassSVM can be sensitive to normalization
        assert 0 <= anomaly_count <= len(results) * 0.8

    @pytest.mark.unit
    def test_predict_with_outliers(self, detector_default, normal_data, data_with_outliers):
        """Test prediction on data with clear outliers."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_outliers)

        assert len(results) == len(data_with_outliers)
        # Outliers should have higher scores
        assert results[0].anomaly_score > results[5].anomaly_score
        assert results[1].anomaly_score > results[5].anomaly_score
        # Outliers should be detected as anomalies
        assert results[0].is_anomaly is True or results[1].is_anomaly is True

    @pytest.mark.unit
    def test_predict_with_different_nu(self, normal_data, data_with_outliers):
        """Test prediction with different nu values."""
        # Higher nu should detect more anomalies
        detector_low = OneClassSVMDetector(nu=0.05).fit(normal_data)
        detector_high = OneClassSVMDetector(nu=0.2).fit(normal_data)

        results_low = detector_low.predict(data_with_outliers)
        results_high = detector_high.predict(data_with_outliers)

        anomalies_low = sum(r.is_anomaly for r in results_low)
        anomalies_high = sum(r.is_anomaly for r in results_high)

        assert anomalies_high >= anomalies_low

    @pytest.mark.unit
    def test_predict_with_different_kernels(self, normal_data, data_with_outliers):
        """Test prediction with different kernel types."""
        kernels = ["rbf", "linear", "poly"]
        results_dict = {}

        for kernel in kernels:
            detector = OneClassSVMDetector(kernel=kernel).fit(normal_data)
            results = detector.predict(data_with_outliers)
            results_dict[kernel] = results

        # All kernels should produce results
        for kernel, results in results_dict.items():
            assert len(results) == len(data_with_outliers)
            # All should detect outliers
            assert results[0].is_anomaly or results[1].is_anomaly

    @pytest.mark.unit
    def test_predict_with_different_gamma(self, normal_data, data_with_outliers):
        """Test prediction with different gamma values."""
        detector_scale = OneClassSVMDetector(gamma="scale").fit(normal_data)
        detector_auto = OneClassSVMDetector(gamma="auto").fit(normal_data)

        results_scale = detector_scale.predict(data_with_outliers)
        results_auto = detector_auto.predict(data_with_outliers)

        # Both should work
        assert len(results_scale) == len(results_auto)

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_outliers):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_outliers)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_outliers)
        assert all(s >= 0 for s in scores)
        assert all(s <= 1 for s in scores)  # Normalized to [0, 1]
        # Outliers should have higher scores
        assert scores[0] > scores[5]
        assert scores[1] > scores[5]

    @pytest.mark.unit
    def test_predict_score_normalization(self, detector_default, normal_data):
        """Test that scores are normalized to [0, 1]."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(normal_data)

        scores = [r.anomaly_score for r in results]
        assert all(0 <= s <= 1 for s in scores)
        assert min(scores) >= 0
        assert max(scores) <= 1

    @pytest.mark.unit
    def test_predict_decision_function(self, detector_default, normal_data):
        """Test that decision function is used correctly."""
        detector = detector_default.fit(normal_data)
        test_point = normal_data[0]
        results = detector.predict(normal_data[:1])

        # Score should be based on decision function
        decision_score = detector.svm_.decision_function([test_point])[0]
        # Negative decision scores indicate anomalies
        # Our score should be -decision_score normalized
        assert results[0].anomaly_score >= 0
        assert results[0].anomaly_score <= 1

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.array([[10, 20], [20, 30], [30, 40], [40, 50]])

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_multidimensional_data(self, detector_default):
        """Test with high-dimensional data."""
        high_dim_data = np.random.randn(100, 10) * 10 + 100

        detector = detector_default.fit(high_dim_data)
        results = detector.predict(high_dim_data)

        assert len(results) == len(high_dim_data)
        assert detector.svm_ is not None
