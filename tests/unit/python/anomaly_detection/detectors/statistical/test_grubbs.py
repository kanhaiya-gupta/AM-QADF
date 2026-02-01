"""
Unit tests for Grubbs detector.

Tests for GrubbsDetector class.
"""

import pytest
import numpy as np
import pandas as pd
from am_qadf.anomaly_detection.detectors.statistical.grubbs import GrubbsDetector
from am_qadf.anomaly_detection.core.base_detector import (
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
)


class TestGrubbsDetector:
    """Test suite for GrubbsDetector class."""

    @pytest.fixture
    def detector_default(self):
        """Create GrubbsDetector with default parameters."""
        return GrubbsDetector()

    @pytest.fixture
    def detector_custom(self):
        """Create GrubbsDetector with custom alpha."""
        return GrubbsDetector(alpha=0.01)

    @pytest.fixture
    def normal_data(self):
        """Create normal data for training."""
        np.random.seed(42)
        return np.random.randn(100, 3) * 10 + 100

    @pytest.fixture
    def data_with_outlier(self, normal_data):
        """Create data with a single clear outlier."""
        data = normal_data.copy()
        # Add one extreme outlier
        data[0] = [300, 300, 300]  # Extreme outlier
        return data

    @pytest.mark.unit
    def test_detector_initialization_default(self):
        """Test GrubbsDetector initialization with default values."""
        detector = GrubbsDetector()

        assert detector.alpha == 0.05
        assert detector.config is not None
        assert detector.is_fitted is False
        assert detector.mean_ is None
        assert detector.std_ is None

    @pytest.mark.unit
    def test_detector_initialization_custom(self):
        """Test GrubbsDetector initialization with custom alpha."""
        detector = GrubbsDetector(alpha=0.01)

        assert detector.alpha == 0.01

    @pytest.mark.unit
    def test_detector_initialization_with_config(self):
        """Test GrubbsDetector initialization with custom config."""
        config = AnomalyDetectionConfig(threshold=0.5)
        detector = GrubbsDetector(alpha=0.02, config=config)

        assert detector.alpha == 0.02
        assert detector.config.threshold == 0.5

    @pytest.mark.unit
    def test_fit(self, detector_default, normal_data):
        """Test fitting the detector."""
        detector = detector_default.fit(normal_data)

        assert detector.is_fitted is True
        assert detector.mean_ is not None
        assert detector.std_ is not None
        assert detector.mean_.shape == (3,)
        assert detector.std_.shape == (3,)

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
        assert detector.mean_.shape == (3,)
        assert detector.std_.shape == (3,)

    @pytest.mark.unit
    def test_grubbs_statistic_calculation(self, detector_default):
        """Test Grubbs' statistic calculation."""
        data = np.array([10, 20, 30, 40, 50, 100])  # 100 is outlier

        G, max_idx = detector_default._grubbs_statistic(data)

        assert G > 0
        assert max_idx == 5  # Index of outlier
        # G should be large for the outlier
        assert G > 2.0

    @pytest.mark.unit
    def test_grubbs_statistic_small_data(self, detector_default):
        """Test Grubbs' statistic with small dataset."""
        data = np.array([10, 20])

        G, max_idx = detector_default._grubbs_statistic(data)

        # Should return 0.0 and -1 for small datasets
        assert G == 0.0
        assert max_idx == -1

    @pytest.mark.unit
    def test_grubbs_statistic_zero_std(self, detector_default):
        """Test Grubbs' statistic with zero standard deviation."""
        data = np.array([10, 10, 10, 10])

        G, max_idx = detector_default._grubbs_statistic(data)

        # Should return 0.0 for zero std
        assert G == 0.0

    @pytest.mark.unit
    def test_grubbs_critical_value(self, detector_default):
        """Test critical value calculation."""
        n = 100
        alpha = 0.05

        G_critical = detector_default._grubbs_critical_value(n, alpha)

        assert G_critical > 0
        # Critical value should increase with smaller alpha
        G_critical_strict = detector_default._grubbs_critical_value(n, 0.01)
        assert G_critical_strict > G_critical

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
        # Most points should not be anomalies
        anomaly_count = sum(r.is_anomaly for r in results)
        # Grubbs' test is conservative, should detect few anomalies
        assert anomaly_count <= len(results) * 0.1

    @pytest.mark.unit
    def test_predict_with_outlier(self, detector_default, normal_data, data_with_outlier):
        """Test prediction on data with a clear outlier."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_outlier)

        assert len(results) == len(data_with_outlier)
        # The outlier should be detected
        assert results[0].is_anomaly is True
        assert results[0].anomaly_score > results[5].anomaly_score

    @pytest.mark.unit
    def test_predict_with_different_alpha(self, normal_data, data_with_outlier):
        """Test prediction with different alpha values."""
        # Lower alpha (stricter) should detect fewer anomalies
        detector_strict = GrubbsDetector(alpha=0.01).fit(normal_data)
        detector_loose = GrubbsDetector(alpha=0.1).fit(normal_data)

        results_strict = detector_strict.predict(data_with_outlier)
        results_loose = detector_loose.predict(data_with_outlier)

        # Both should detect the clear outlier
        assert results_strict[0].is_anomaly or results_loose[0].is_anomaly

    @pytest.mark.unit
    def test_predict_multivariate(self, detector_default, normal_data, data_with_outlier):
        """Test prediction on multivariate data."""
        detector = detector_default.fit(normal_data)
        results = detector.predict(data_with_outlier)

        # Grubbs' test is applied per feature and combined
        assert len(results) == len(data_with_outlier)
        # At least one feature should detect the outlier
        assert any(r.is_anomaly for r in results[:5])

    @pytest.mark.unit
    def test_predict_scores(self, detector_default, normal_data, data_with_outlier):
        """Test predict_scores method."""
        detector = detector_default.fit(normal_data)
        scores = detector.predict_scores(data_with_outlier)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(data_with_outlier)
        assert all(s >= 0 for s in scores)
        # Outlier should have higher score
        assert scores[0] > scores[5]

    @pytest.mark.unit
    def test_edge_case_small_dataset(self, detector_default):
        """Test with small dataset."""
        small_data = np.array([[10, 20], [20, 30], [30, 40]])

        detector = detector_default.fit(small_data)
        results = detector.predict(small_data)

        assert len(results) == len(small_data)
        assert all(isinstance(r, AnomalyDetectionResult) for r in results)

    @pytest.mark.unit
    def test_edge_case_all_same_values(self, detector_default):
        """Test with all same values."""
        same_data = np.ones((50, 3)) * 100

        detector = detector_default.fit(same_data)
        results = detector.predict(same_data)

        # All same values should have zero std, so no outliers detected
        assert all(not r.is_anomaly for r in results)
        assert all(r.anomaly_score == 0.0 for r in results)

    @pytest.mark.unit
    def test_multidimensional_data(self, detector_default):
        """Test with high-dimensional data."""
        high_dim_data = np.random.randn(100, 10) * 10 + 100

        detector = detector_default.fit(high_dim_data)
        results = detector.predict(high_dim_data)

        assert len(results) == len(high_dim_data)
        assert detector.mean_.shape == (10,)
        assert detector.std_.shape == (10,)

    @pytest.mark.unit
    def test_grubbs_detects_single_outlier(self, detector_default):
        """Test that Grubbs' test is designed for single outlier detection."""
        # Create data with one clear outlier
        data = np.random.randn(50, 2) * 10 + 100
        data[0] = [500, 500]  # Single extreme outlier

        detector = detector_default.fit(data)
        results = detector.predict(data)

        # Should detect the outlier
        assert results[0].is_anomaly is True
        # Most other points should not be anomalies
        other_anomalies = sum(r.is_anomaly for r in results[1:])
        assert other_anomalies < len(results) * 0.2
