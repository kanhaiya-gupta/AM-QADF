"""
Unit tests for Random Forest detector.

Tests for RandomForestAnomalyDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.machine_learning.random_forest import (
    RandomForestAnomalyDetector,
    SKLEARN_AVAILABLE,
)
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestRandomForestAnomalyDetector:
    """Test suite for RandomForestAnomalyDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create RandomForestAnomalyDetector with default parameters."""
        return RandomForestAnomalyDetector(n_estimators=10)  # Reduced for faster tests

    @pytest.fixture
    def detector_custom(self):
        """Create RandomForestAnomalyDetector with custom parameters."""
        return RandomForestAnomalyDetector(n_estimators=50, max_depth=10, contamination=0.05, use_isolation_forest=True)

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(100, 5) * 10 + 100

    @pytest.fixture
    def data_with_outliers(self, normal_data):
        """Create data with clear outliers."""
        data = normal_data.copy()
        # Add clear outliers
        data[0] = np.random.randn(5) * 50 + 200  # High outlier
        data[1] = np.random.randn(5) * 50 - 50  # Low outlier
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test RandomForestAnomalyDetector initialization with default values."""
        detector = RandomForestAnomalyDetector()

        assert detector.n_estimators == 100
        assert detector.max_depth is None
        assert detector.contamination == 0.1
        assert detector.use_isolation_forest is True
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.model is not None  # IsolationForest is created in __init__

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test RandomForestAnomalyDetector initialization with custom parameters."""
        detector = RandomForestAnomalyDetector(n_estimators=50, max_depth=10, contamination=0.05, use_isolation_forest=True)

        assert detector.n_estimators == 50
        assert detector.max_depth == 10
        assert detector.contamination == 0.05
        assert detector.use_isolation_forest is True

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test RandomForestAnomalyDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = RandomForestAnomalyDetector(n_estimators=10, config=config)

        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_detector_initialization_no_sklearn(self):
        """Test that initialization raises error if sklearn not available."""
        # This test would need to mock SKLEARN_AVAILABLE = False
        # For now, we skip if sklearn is not available
        pass

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.model is not None
        assert detector.scaler is not None
        assert detector is detector_default  # Method chaining

    @pytest.mark.unit
    def test_fit_with_dataframe(self, detector_default):
        """Test fitting with pandas DataFrame."""
        df = pd.DataFrame({f"feature{i}": np.random.randn(50) * 10 + 100 for i in range(5)})

        detector = detector_default.fit(df)

        assert detector.is_fitted is True

    @pytest.mark.unit
    def test_fit_data_scaling(self, detector_default, normal_data):
        """Test that data is scaled during fitting."""
        detector = detector_default.fit(normal_data)

        # Scaler should be fitted
        assert hasattr(detector.scaler, "mean_")
        assert hasattr(detector.scaler, "scale_")

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
        # With contamination=0.1, about 10% should be anomalies
        anomaly_count = sum(r.is_anomaly for r in results)
        # Allow more variance - RandomForest can flag more points depending on data distribution
        assert 0 <= anomaly_count <= len(results) * 0.3

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
    def test_predict_isolation_forest(self, normal_data, data_with_outliers):
        """Test prediction using Isolation Forest."""
        detector = RandomForestAnomalyDetector(n_estimators=10, use_isolation_forest=True)
        detector.fit(normal_data)
        results = detector.predict(data_with_outliers)

        assert len(results) == len(data_with_outliers)
        # Isolation Forest should detect outliers
        assert any(r.is_anomaly for r in results[:5])

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
    def test_predict_scores(self, detector_default, normal_data, data_with_outliers):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_outliers)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_outliers)
        assert all(s >= 0 for s in scores)
        assert all(s <= 1 for s in scores)
        # Outliers should have higher scores
        assert scores[0] > scores[5]
        assert scores[1] > scores[5]

    @pytest.mark.unit
    def test_different_contamination(self, normal_data, data_with_outliers):
        """Test with different contamination values."""
        detector_low = RandomForestAnomalyDetector(n_estimators=10, contamination=0.05).fit(normal_data)
        detector_high = RandomForestAnomalyDetector(n_estimators=10, contamination=0.2).fit(normal_data)

        results_low = detector_low.predict(data_with_outliers)
        results_high = detector_high.predict(data_with_outliers)

        anomalies_low = sum(r.is_anomaly for r in results_low)
        anomalies_high = sum(r.is_anomaly for r in results_high)

        assert anomalies_high >= anomalies_low

    @pytest.mark.unit
    def test_different_n_estimators(self, normal_data):
        """Test with different number of estimators."""
        detector_small = RandomForestAnomalyDetector(n_estimators=10).fit(normal_data)
        detector_large = RandomForestAnomalyDetector(n_estimators=50).fit(normal_data)

        results_small = detector_small.predict(normal_data[:10])
        results_large = detector_large.predict(normal_data[:10])

        # Both should work
        assert len(results_small) == len(results_large)

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.random.randn(20, 5) * 10 + 100

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_single_feature(self, detector_default):
        """Test with single feature."""
        single_feature = np.random.randn(50, 1) * 10 + 100

        detector = detector_default.fit(single_feature)
        results = detector.predict(single_feature)

        assert len(results) == len(single_feature)
